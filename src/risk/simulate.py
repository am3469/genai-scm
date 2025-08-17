"""
Policy-driven inventory simulator with scenario knobs.

Scenarios:
  --lead_mult 1.0           # multiply supplier lead times (e.g., 1.5 = +50%)
  --demand_spike 0.0        # +% demand across the board (0.2 = +20%)
  --supplier_outage ""      # comma-separated supplier_ids that won't deliver
  --horizon 28              # days to simulate
Inputs:
  sales_daily.csv           # for history & last on-hand
  inventory_daily.csv
  suppliers.csv
  policy_recs.csv           # from Day 4 (ROP, Target-S, EOQ, etc.)
  forecasts.csv/parquet     # OPTIONAL; falls back to flat forecast if missing
Outputs:
  data/processed/sim_results.csv
  data/processed/sim_overall.csv
"""
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

def read_df(path):
    return pd.read_csv(path)

def read_forecasts():
    try:
        fc = pd.read_csv("models/forecasts.csv", parse_dates=["date"])
        return fc
    except Exception:
        try:
            fc = pd.read_parquet("models/forecasts.parquet")
            if "date" in fc.columns:
                fc["date"] = pd.to_datetime(fc["date"])
            return fc
        except Exception:
            return None

def latest_on_hand(inv: pd.DataFrame) -> pd.DataFrame:
    inv = inv.sort_values("date")
    last = inv.groupby(["store_id","sku"], as_index=False).last()
    last = last.rename(columns={"on_hand":"on_hand_latest"})
    return last

def build_future_forecast(sales: pd.DataFrame, fc_opt: pd.DataFrame|None, horizon: int) -> pd.DataFrame:
    """Return future horizon forecast per (store, sku). If forecasts exist and extend
    beyond max sales date, use those; otherwise build flat forecast from last 28d mean."""
    sales["date"] = pd.to_datetime(sales["date"])
    max_sales = sales["date"].max()
    start = max_sales + pd.Timedelta(days=1)
    end   = start + pd.Timedelta(days=horizon-1)

    rows = []
    if fc_opt is not None and (fc_opt["date"] > max_sales).any():
        # Use provided forecasts for future days only
        f = fc_opt[fc_opt["date"] > max_sales].copy()
        # If multiple models, assume best already chosen in Day 4 script; otherwise just take what's there
        for (store, sku), g in f.groupby(["store_id","sku"]):
            g = g.sort_values("date")
            g = g[(g["date"] >= start) & (g["date"] <= end)]
            if g.empty:
                # fall back if not long enough
                mean7 = (sales[(sales["store_id"]==store)&(sales["sku"]==sku)]
                         .sort_values("date").tail(28)["qty"].mean())
                for d in pd.date_range(start, end, freq="D"):
                    rows.append({"date": d, "store_id": store, "sku": sku, "yhat": float(mean7 if not np.isnan(mean7) else 0.0)})
            else:
                for d, yhat in zip(g["date"], g["yhat"]):
                    rows.append({"date": d, "store_id": store, "sku": sku, "yhat": float(yhat)})
    else:
        # Flat forecast = last 28d average
        for (store, sku), g in sales.groupby(["store_id","sku"]):
            mean7 = g.sort_values("date").tail(28)["qty"].mean()
            for d in pd.date_range(start, end, freq="D"):
                rows.append({"date": d, "store_id": store, "sku": sku, "yhat": float(mean7 if not np.isnan(mean7) else 0.0)})

    return pd.DataFrame(rows)

def simulate(sales, inv, supp, policy, fc_future, horizon, lead_mult, demand_spike, outage_ids):
    # Prepare lookup tables
    inv_last = latest_on_hand(inv)
    base = (policy.merge(inv_last, on=["store_id","sku"], how="left")
                  .merge(supp[["sku","supplier_id","lt_std"]].drop_duplicates(), on=["sku","supplier_id"], how="left"))
    base["on_hand"] = base["on_hand"].fillna(base.get("on_hand_latest", 0)).fillna(0).astype(int)
    base["lt_std"] = base["lt_std"].fillna(0.0)

    results = []
    overall = {"demand":0.0,"fulfilled":0.0,"orders":0,"units_ordered":0,"stockout_days":0,"days":horizon,"sum_on_hand":0.0}

    for (store, sku), row in base.set_index(["store_id","sku"]).iterrows():
        on_hand = int(row["on_hand"])
        rop = float(row["rop"])
        target_S = float(row["target_S"])
        moq = int(row["moq"])
        L = int(max(1, round(row["lead_time_days"] * lead_mult)))
        supplier_id = str(row.get("supplier_id","S000"))

        # pre-filter horizon demand
        fut = fc_future[(fc_future["store_id"]==store) & (fc_future["sku"]==sku)].sort_values("date").head(horizon)
        if fut.empty:
            # no forecast -> zero horizon
            fut = pd.DataFrame({"date": pd.date_range(fc_future["date"].min(), periods=horizon, freq="D"),
                                "yhat": np.zeros(horizon)})
        demands = np.maximum(0.0, fut["yhat"].values * (1.0 + demand_spike))
        dates = fut["date"].values

        # pipeline of outstanding orders: list of (arrival_day_index, qty)
        pipeline = []
        demand_total = fulfilled_total = 0.0
        stockout_days = 0
        units_ordered = 0
        orders_placed = 0
        sum_on_hand = 0.0

        for t in range(horizon):
            # receive due orders
            arrivals = [q for (arrive_t, q) in pipeline if arrive_t == t]
            if arrivals:
                on_hand += int(sum(arrivals))
            pipeline = [(arrive_t, q) for (arrive_t, q) in pipeline if arrive_t != t]

            # demand & ship
            d = float(demands[t])
            ship = min(on_hand, math.ceil(d))
            on_hand -= ship
            unfilled = max(0.0, d - ship)
            if ship < d:
                stockout_days += 1
            demand_total += d
            fulfilled_total += ship
            sum_on_hand += on_hand

            # place order if below ROP (end-of-day check)
            if on_hand < rop:
                order_to_S = max(0.0, target_S - on_hand)
                # ensure MOQ
                rec_qty = int(math.ceil(order_to_S / max(1, moq)) * max(1, moq))
                if rec_qty > 0:
                    orders_placed += 1
                    units_ordered += rec_qty
                    # if supplier is in outage, order won't arrive within horizon (skip enqueuing)
                    if supplier_id not in outage_ids:
                        arrive_t = t + L
                        if arrive_t < horizon:
                            pipeline.append((arrive_t, rec_qty))
                        # if arrival after horizon, ignore for metrics (still counts as ordered)
                    # else: record order but no arrival

        fill_rate = (fulfilled_total / demand_total) if demand_total > 0 else 1.0
        results.append({
            "store_id": store, "sku": sku, "supplier_id": supplier_id,
            "horizon": horizon,
            "demand": round(demand_total,2), "fulfilled": round(fulfilled_total,2),
            "fill_rate": round(fill_rate,4),
            "stockout_days": int(stockout_days),
            "orders_placed": int(orders_placed),
            "units_ordered": int(units_ordered),
            "avg_on_hand": round(sum_on_hand / horizon,2),
            "ending_on_hand": int(on_hand),
            "lead_time_days_eff": L
        })

        # accumulate overall
        overall["demand"] += demand_total
        overall["fulfilled"] += fulfilled_total
        overall["orders"] += orders_placed
        overall["units_ordered"] += units_ordered
        overall["stockout_days"] += stockout_days
        overall["sum_on_hand"] += sum_on_hand

    overall_fill = (overall["fulfilled"]/overall["demand"]) if overall["demand"]>0 else 1.0
    overall_out = {
        "horizon": horizon,
        "demand": round(overall["demand"],2),
        "fulfilled": round(overall["fulfilled"],2),
        "fill_rate": round(overall_fill,4),
        "orders": overall["orders"],
        "units_ordered": overall["units_ordered"],
        "stockout_days": overall["stockout_days"],
        "avg_on_hand": round(overall["sum_on_hand"]/max(overall["days"],1),2)
    }

    return pd.DataFrame(results), pd.DataFrame([overall_out])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sales", default="data/processed/sales_daily.csv")
    ap.add_argument("--inventory", default="data/processed/inventory_daily.csv")
    ap.add_argument("--suppliers", default="data/processed/suppliers.csv")
    ap.add_argument("--policy", default="data/processed/policy_recs.csv")
    ap.add_argument("--horizon", type=int, default=28)
    ap.add_argument("--lead_mult", type=float, default=1.0)
    ap.add_argument("--demand_spike", type=float, default=0.0)
    ap.add_argument("--supplier_outage", default="", help="comma-separated supplier_ids")
    ap.add_argument("--outdir", default="data/processed")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    sales = read_df(args.sales); sales["date"] = pd.to_datetime(sales["date"])
    inv   = read_df(args.inventory); inv["date"] = pd.to_datetime(inv["date"])
    supp  = read_df(args.suppliers)
    policy= read_df(args.policy)

    fc_opt = read_forecasts()
    fc_future = build_future_forecast(sales, fc_opt, args.horizon)

    outage_ids = set([s for s in args.supplier_outage.split(",") if s.strip()])
    res, overall = simulate(sales, inv, supp, policy, fc_future, args.horizon,
                            args.lead_mult, args.demand_spike, outage_ids)

    out_path = Path(args.outdir)/"sim_results.csv"
    res.to_csv(out_path, index=False)
    (Path(args.outdir)/"sim_overall.csv").write_text(overall.to_csv(index=False))

    print(f"[simulate] wrote {len(res)} rows → {out_path}")
    print("[simulate] overall:")
    print(overall.to_string(index=False))

if __name__ == "__main__":
    main()
