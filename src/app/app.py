import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
try:
    from src.risk.simulate import simulate, build_future_forecast, read_forecasts
except Exception:
    simulate = build_future_forecast = read_forecasts = None

DATA = {
    "sales": "data/processed/sales_daily.csv",
    "inventory": "data/processed/inventory_daily.csv",
    "suppliers": "data/processed/suppliers.csv",
    "policy": "data/processed/policy_recs.csv",
    "metrics": "reports/metrics_summary.csv",
    "fc_csv": "models/forecasts.csv",
    "fc_parquet": "models/forecasts.parquet",
    "sim_results": "data/processed/sim_results.csv",
    "sim_overall": "data/processed/sim_overall.csv",
}

st.set_page_config(page_title="GenAI Supply Chain", layout="wide")

@st.cache_data(show_spinner=False)
def load_csv(path, parse_dates=None):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=parse_dates or [])

@st.cache_data(show_spinner=False)
def load_forecasts():
    # Prefer CSV (lighter), fallback to parquet, else None
    fc = load_csv(DATA["fc_csv"], parse_dates=["date"])
    if fc is not None:
        return fc
    p = Path(DATA["fc_parquet"])
    if p.exists():
        try:
            df = pd.read_parquet(p)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            return None
    return None

def guard(df, name):
    if df is None or len(df) == 0:
        st.warning(f"Missing or empty **{name}**. Expected at `{DATA[name] if name in DATA else name}`.")
        return False
    return True

def kpi_metric(label, value, help_text=None, fmt=None):
    if fmt:
        value = fmt(value)
    st.metric(label, value, help=help_text)

# ---------- Sidebar ----------
st.sidebar.title("GenAI Supply Chain")
st.sidebar.caption("Akhila • Inventory Forecasting & Policy")
page = st.sidebar.radio("Pages", ["Overview", "SKU Detail", "Scenarios", "Action Plan"])

sales = load_csv(DATA["sales"], parse_dates=["date"])
inv   = load_csv(DATA["inventory"], parse_dates=["date"])
supp  = load_csv(DATA["suppliers"])
policy= load_csv(DATA["policy"])
metrics = load_csv(DATA["metrics"])
fc = load_forecasts()

# Common sets
stores = sorted(sales["store_id"].astype(str).unique()) if guard(sales, "sales") else []
skus   = sorted(sales["sku"].astype(str).unique()) if guard(sales, "sales") else []

# ---------- Overview ----------
if page == "Overview":
    st.title("📊 Overview")
    ok = guard(sales, "sales")
    if not ok:
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    total_rows = len(sales)
    start, end = sales["date"].min(), sales["date"].max()
    sku_count = sales["sku"].nunique()
    store_count = sales["store_id"].nunique()
    with col1: kpi_metric("Rows", total_rows, fmt=lambda x: f"{x:,}")
    with col2: kpi_metric("Date Range", f"{start.date()} → {end.date()}")
    with col3: kpi_metric("SKUs", sku_count, fmt=lambda x: f"{x:,}")
    with col4: kpi_metric("Stores", store_count, fmt=lambda x: f"{x:,}")

    # Daily total demand
    daily = (sales.groupby("date")["qty"].sum().reset_index())
    chart = alt.Chart(daily).mark_line().encode(
        x="date:T", y=alt.Y("qty:Q", title="Units (All Stores/SKUs)")
    ).properties(height=280)
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Forecast Validation (from Day 3)")
    if metrics is not None and len(metrics):
        overall = (metrics.groupby("model")[["WAPE","sMAPE"]].mean().reset_index().sort_values("WAPE"))
        st.dataframe(overall, use_container_width=True)
    else:
        st.info("No `reports/metrics_summary.csv` found. Run Day 3 backtest to populate.")

    st.subheader("Top Recommended Orders (from Day 4)")
    if guard(policy, "policy"):
        top_orders = policy.sort_values("recommend_order_qty", ascending=False).head(10)
        st.dataframe(top_orders[["store_id","sku","on_hand","rop","target_S","recommend_order_qty","supplier_id"]], use_container_width=True)
    else:
        st.info("No `policy_recs.csv` yet. Run Day 4 policy step.")

# ---------- SKU Detail ----------
elif page == "SKU Detail":
    st.title("🔍 SKU Detail")
    if not (guard(sales, "sales") and guard(policy, "policy")):
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        store_pick = st.selectbox("Store", stores, index=0 if stores else None)
    # limit skus to those present in selected store for better UX
    skus_in_store = sorted(sales[sales["store_id"].astype(str)==str(store_pick)]["sku"].astype(str).unique()) if stores else []
    with c2:
        sku_pick = st.selectbox("SKU", skus_in_store, index=0 if skus_in_store else None)

    if not store_pick or not sku_pick:
        st.info("Select a store and SKU.")
        st.stop()

    hist = sales[(sales["store_id"].astype(str)==str(store_pick)) & (sales["sku"].astype(str)==str(sku_pick))].sort_values("date")
    st.write(f"**History points:** {len(hist)}")
    line = alt.Chart(hist).mark_line().encode(x="date:T", y=alt.Y("qty:Q", title="Units"))
    st.altair_chart(line.properties(height=280), use_container_width=True)

    # Forecast slice (if available)
    if fc is not None and "yhat" in fc.columns:
        fcx = fc[(fc["store_id"].astype(str)==str(store_pick)) & (fc["sku"].astype(str)==str(sku_pick))].copy()
        if "date" in fcx.columns:
            fcx = fcx.sort_values("date")
            st.caption("Forecast (from Day 3)")
            st.altair_chart(
                alt.Chart(fcx).mark_line(color="#E4572E").encode(
                    x="date:T", y=alt.Y("yhat:Q", title="Forecast")
                ).properties(height=200),
                use_container_width=True
            )
        else:
            st.info("Forecasts present but without dates. Check models/forecasts.* format.")
    else:
        st.info("No forecasts found. Run Day 3 backtest to generate forecasts.")

    # Policy row
    pr = policy[(policy["store_id"].astype(str)==str(store_pick)) & (policy["sku"].astype(str)==str(sku_pick))]
    if len(pr):
        st.subheader("Policy")
        show = pr.iloc[0][["on_hand","rop","target_S","recommend_order_qty","lead_time_days","safety_stock","moq","supplier_id","service_level"]]
        st.dataframe(show.to_frame("value"))
    else:
        st.info("No policy row for this Store/SKU.")

# ---------- Scenarios ----------
elif page == "Scenarios":
    st.title("🧪 Scenario Simulator")
    if simulate is None or build_future_forecast is None:
        st.error("Simulator functions not imported. Ensure Day 5 code exists at `src/risk/simulate.py`.")
        st.stop()

    if not (guard(sales, "sales") and guard(inv, "inventory") and guard(supp, "suppliers") and guard(policy, "policy")):
        st.stop()

    st.markdown("Use the controls to stress-test your policy.")
    cols = st.columns(5)
    with cols[0]:
        horizon = st.number_input("Horizon (days)", 7, 56, 28, 1)
    with cols[1]:
        lead_mult = st.slider("Lead-time ×", 0.5, 3.0, 1.0, 0.1)
    with cols[2]:
        demand_spike = st.slider("Demand spike %", 0, 200, 0, 5) / 100.0
    with cols[3]:
        store_filter = st.selectbox("Store (optional)", ["All"] + stores, index=0)
    with cols[4]:
        topk = st.slider("Top K SKUs (by volume)", 5, 200, 30, 5)

    # Subset policy by store
    pol = policy.copy()
    if store_filter != "All":
        pol = pol[pol["store_id"].astype(str)==str(store_filter)]

    # Reduce to topK by sales volume for speed
    vol = (sales.groupby(["store_id","sku"])["qty"].sum().reset_index())
    pol = pol.merge(vol, on=["store_id","sku"], how="left").sort_values("qty_y", ascending=False)
    pol = pol.head(topk).drop(columns=["qty_y"])

    # Build future forecast frame (uses Day 5 helper)
    fc_future = build_future_forecast(sales, load_forecasts(), horizon)
    outage_ids = st.multiselect("Supplier outage (IDs)", sorted(supp["supplier_id"].astype(str).unique()))

    run = st.button("Run Simulation")
    if run:
        with st.spinner("Simulating..."):
            res, overall = simulate(
                sales=sales, inv=inv, supp=supp, policy=pol,
                fc_future=fc_future, horizon=int(horizon),
                lead_mult=float(lead_mult), demand_spike=float(demand_spike),
                outage_ids=set(outage_ids)
            )
        st.success("Simulation complete.")

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_metric("Fill rate", float(overall["fill_rate"].iloc[0]), fmt=lambda x: f"{x:.2%}")
        with c2: kpi_metric("Orders placed", int(overall["orders"].iloc[0]))
        with c3: kpi_metric("Units ordered", int(overall["units_ordered"].iloc[0]))
        with c4: kpi_metric("Avg on-hand", float(overall["avg_on_hand"].iloc[0]), fmt=lambda x: f"{x:,.1f}")

        st.subheader("At-risk SKUs (lowest fill rate)")
        worst = res.sort_values(["fill_rate","stockout_days"]).head(20)
        st.dataframe(worst, use_container_width=True)

        # Download buttons
        st.download_button("Download per-SKU results (CSV)", res.to_csv(index=False).encode("utf-8"), "sim_results.csv", "text/csv")
        st.download_button("Download overall KPIs (CSV)", overall.to_csv(index=False).encode("utf-8"), "sim_overall.csv", "text/csv")

# ---------- Action Plan ----------
elif page == "Action Plan":
    st.title("🧠 Action Plan (Auto-POs)")
    if not guard(policy, "policy"):
        st.stop()

    st.caption("Generate a simple, rule-based purchase plan using your policy and (optional) scenario results.")
    use_sim = st.toggle("Incorporate last simulation results if available (prioritize SKUs with stockouts)")
    sim_res = load_csv(DATA["sim_results"])

    # Base candidates: those below ROP or with recommendation > 0
    base = policy.copy()
    base["need"] = base["recommend_order_qty"].fillna(0).astype(int)
    plan = base[base["need"] > 0].copy()

    if use_sim and sim_res is not None and len(sim_res):
        risk = sim_res[sim_res["fill_rate"] < 0.98].copy()
        risk["priority"] = np.where(risk["stockout_days"] > 0, "high", "medium")
        plan = plan.merge(risk[["store_id","sku","fill_rate","stockout_days"]], on=["store_id","sku"], how="left")
        plan["priority"] = plan["priority"].fillna("normal")
    else:
        plan["priority"] = "normal"

    plan["po_qty"] = plan["need"].clip(lower=plan["moq"])
    plan["rationale"] = np.where(
        plan["priority"]=="high",
        "Below ROP + simulated stockout risk",
        "Below ROP based on forecasted lead-time demand"
    )

    cols = ["supplier_id","store_id","sku","po_qty","unit_cost","moq","on_hand","rop","target_S","rationale","priority"]
    plan = plan.sort_values(["priority","po_qty"], ascending=[True, False])

    st.subheader("Proposed Purchase Orders")
    st.dataframe(plan[cols].reset_index(drop=True), use_container_width=True)
    st.download_button("Download PO CSV", plan[cols].to_csv(index=False).encode("utf-8"), "purchase_orders.csv", "text/csv")

    st.info("This is a rule-based plan for safety. If you want, we can add an LLM step later to turn this into a vendor email or a management summary.")

