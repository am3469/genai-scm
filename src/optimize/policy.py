"""
Compute single-echelon inventory policies from forecasts and history.

Inputs:
  --sales data/processed/sales_daily.csv
  --inventory data/processed/inventory_daily.csv
  --suppliers data/processed/suppliers.csv
  --forecasts models/forecasts.csv   (fallbacks to .parquet or naive if missing)
Params:
  --service 0.95      # cycle service level -> z
  --cycle_days 7      # order-up-to review period proxy
  --horizon 28        # forecast horizon if fallback needed
Outputs:
  data/processed/policy_recs.csv / .parquet
"""
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

PREF_MODELS = ["sarimax7117", "snaive7", "naive"]

def z_from_service(p: float) -> float:
    # Simple mapping with linear interpolation for common service levels
    table = {0.90: 1.2816, 0.95: 1.6449, 0.975: 1.96, 0.99: 2.3263}
    if p in table: return table[p]
    keys = sorted(table.keys())
    if p <= keys[0]: return table[keys[0]]
    if p >= keys[-1]: return table[keys[-1]]
    # linear interp between nearest keys
    for i in range(len(keys)-1):
        if keys[i] <= p <= keys[i+1]:
            x0, y0 = keys[i], table[keys[i]]
            x1, y1 = keys[i+1], table[keys[i+1]]
            t = (p - x0) / (x1 - x0)
            return y0 + t*(y1 - y0)
    return 1.65

def read_forecasts(path_csv: str, path_parquet: str|None=None) -> pd.DataFrame|None:
    try:
        fc = pd.read_csv(path_csv)
        fc["date"] = pd.to_datetime(fc["date"])
        return fc
    except Exception:
        if path_parquet:
            try:
                fc = pd.read_parquet(path_parquet)
                if "date" in fc.columns:
                    fc["date"] = pd.to_datetime(fc["date"])
                return fc
            except Exception:
                return None
        return None

def fallback_forecast(sales: pd.DataFrame, horizon: int) -> pd.DataFrame:
    # Per (store, sku), use last 7d average as flat forecast
    rows = []
    for (store, sku), g in sales.groupby(["store_id","sku"]):
        g = g.sort_values("date")
        tail = g.tail(7)
        avg = tail["qty"].mean() if len(tail) else 0.0
        dates = pd.date_range(sales["date"].max() + pd.Timedelta(days=1),
                              periods=horizon, freq="D")
        for d in dates:
            rows.append({"date": d, "store_id": store, "sku": sku,
                         "model": "naive_mean7", "yhat": float(avg)})
    return pd.DataFrame(rows)

def extend_series_to_length(vals: np.ndarray, L: int) -> np.ndarray:
    if len(vals) >= L: return vals[:L]
    if len(vals) == 0: return np.zeros(L)
    return np.concatenate([vals, np.repeat(vals[-1], L - len(vals))])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sales", default="data/processed/sales_daily.csv")
    ap.add_argument("--inventory", default="data/processed/inventory_daily.csv")
    ap.add_argument("--suppliers", default="data/processed/suppliers.csv")
    ap.add_argument("--forecasts", default="models/forecasts.csv")
    ap.add_argument("--service", type=float, default=0.95)
    ap.add_argument("--cycle_days", type=int, default=7)
    ap.add_argument("--hist_days", type=int, default=84)
    ap.add_argument("--horizon", type=int, default=28)
    ap.add_argument("--outdir", default="data/processed")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    sales = pd.read_csv(args.sales, parse_dates=["date"])
    inv   = pd.read_csv(args.inventory, parse_dates=["date"])
    supp  = pd.read_csv(args.suppliers)

    fc = read_forecasts(args.forecasts, "models/forecasts.parquet")
    if fc is None:
        print("[policy] forecasts not found; using naive fallback.")
        fc = fallback_forecast(sales, args.horizon)
    else:
        # choose preferred model if multiple
        order = {m:i for i,m in enumerate(PREF_MODELS)}
        # If model column missing, treat as already single-model
        if "model" in fc.columns:
            fc["model_rank"] = fc["model"].map(order).fillna(999).astype(int)
            # keep best-ranked per (date, store, sku)
            fc = (fc.sort_values(["store_id","sku","date","model_rank"])
                    .drop_duplicates(subset=["store_id","sku","date"], keep="first")
                    .drop(columns=["model_rank"]))
    # ensure types
    fc["store_id"] = fc["store_id"].astype(str)
    fc["sku"] = fc["sku"].astype(str)

    # Latest on-hand per (store, sku)
    inv_latest = inv.sort_values("date").groupby(["store_id","sku"], as_index=False).last()
    inv_latest = inv_latest.rename(columns={"on_hand":"on_hand_latest"})

    # Historical daily std (last hist_days)
    max_date = sales["date"].max()
    hist_start = max_date - pd.Timedelta(days=args.hist_days)
    hist = sales[sales["date"] >= hist_start].copy()
    hist["store_id"] = hist["store_id"].astype(str)
    hist["sku"] = hist["sku"].astype(str)
    hist_stats = (hist.groupby(["store_id","sku"])["qty"]
                    .agg(mu_daily="mean", sigma_daily="std", D_annual=lambda x: x.mean()*365.0)
                    .reset_index())
    hist_stats["sigma_daily"] = hist_stats["sigma_daily"].fillna(0.0)

    # Join supplier defaults
    supp = supp.copy()
    supp["sku"] = supp["sku"].astype(str)
    if "lead_time_days" not in supp.columns: supp["lead_time_days"] = 7
    if "lt_std" not in supp.columns: supp["lt_std"] = 2
    if "moq" not in supp.columns: supp["moq"] = 1
    if "unit_cost" not in supp.columns: supp["unit_cost"] = 10.0
    supp_sku = supp[["sku","supplier_id","lead_time_days","lt_std","moq","unit_cost"]].drop_duplicates()

    # Merge bases
    base = (hist_stats.merge(inv_latest, on=["store_id","sku"], how="left")
                    .merge(supp_sku, on="sku", how="left"))
    base["on_hand_latest"] = base["on_hand_latest"].fillna(0).astype(int)
    base["lead_time_days"] = base["lead_time_days"].fillna(7).astype(int)
    base["lt_std"] = base["lt_std"].fillna(2).astype(float)
    base["moq"] = base["moq"].fillna(1).astype(int)
    base["unit_cost"] = base["unit_cost"].fillna(10.0).astype(float)

    # Compute demand during lead time (DL) using forecasts
    z = z_from_service(args.service)

    recs = []
    for (store, sku), row in base.set_index(["store_id","sku"]).iterrows():
        L = int(max(1, row["lead_time_days"]))
        # next-L forecasts
        f = fc[(fc["store_id"]==store) & (fc["sku"]==sku)].sort_values("date")["yhat"].values
        fL = extend_series_to_length(f, L)
        mu_DL = float(np.sum(fL))

        # variability: sigma_DL ≈ sqrt(L) * sigma_daily (hist)
        sigma_daily = float(row.get("sigma_daily", 0.0) or 0.0)
        sigma_DL = float(np.sqrt(L) * sigma_daily)

        safety = float(z * sigma_DL)
        rop = float(mu_DL + safety)

        # Target S: plan to cover lead time + cycle stock (cycle_days of demand)
        mu_cycle = float(np.mean(f[:args.cycle_days]) if len(f) >= args.cycle_days else (row.get("mu_daily", 0.0) or 0.0) * args.cycle_days)
        target_S = float(rop + mu_cycle)

        # EOQ: sqrt(2 D K / hC). Use K=50, holding rate h=0.20 by default
        D = float(row.get("D_annual", 0.0) or 0.0)
        K = 50.0
        h_rate = 0.20
        C = float(row.get("unit_cost", 10.0))
        H = h_rate * C if C > 0 else h_rate*10.0
        eoq = float(np.sqrt(max(1e-6, 2.0 * max(D,0.0) * K / max(H,1e-6))))

        # Round/adjust by MOQ
        def round_up(x, step=1):
            return int(np.ceil(max(0.0, x) / step) * step)
        eoq_adj = round_up(eoq, max(int(row["moq"]), 1))

        on_hand = int(row["on_hand_latest"])
        moq = int(row["moq"])

        order_to_S = max(0.0, target_S - on_hand)
        rec_qty = round_up(max(order_to_S, eoq_adj, moq if on_hand <= rop else 0), max(1, moq))

        reason = "Below ROP" if on_hand <= rop else "At/above ROP"
        recs.append({
            "store_id": store, "sku": sku,
            "service_level": args.service, "z": z,
            "lead_time_days": L,
            "mu_DL": round(mu_DL,3), "sigma_DL": round(sigma_DL,3),
            "safety_stock": int(round(safety)),
            "rop": int(round(rop)),
            "target_S": int(round(target_S)),
            "eoq": int(round(eoq)), "eoq_adj": int(eoq_adj),
            "moq": moq,
            "on_hand": on_hand,
            "recommend_order_qty": int(rec_qty),
            "supplier_id": row.get("supplier_id", "S000"),
            "unit_cost": C,
            "reason": reason
        })

    out = pd.DataFrame(recs).sort_values(["store_id","sku"])
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(args.outdir)/"policy_recs.csv", index=False)
    try:
        out.to_parquet(Path(args.outdir)/"policy_recs.parquet", index=False)
    except Exception:
        pass

    print(f"[policy] wrote {len(out)} rows to {Path(args.outdir)/'policy_recs.csv'}")
    # Show top-10 biggest recommended orders
    if len(out):
        show = out.sort_values("recommend_order_qty", ascending=False).head(10)
        print(show[["store_id","sku","on_hand","rop","target_S","recommend_order_qty","supplier_id"]].to_string(index=False))

if __name__ == "__main__":
    main()
