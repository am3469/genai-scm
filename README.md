# GenAI Supply Chain (Forecasting → Inventory Policy → Simulator → App)

**Akhila Madanapati** • Repo: `am3469/genai-scm`  
End-to-end project that turns raw retail data into **forecasts**, **reorder policies (ROP/Target-S/EOQ)**, a **scenario simulator** (lead-time ×, demand spikes, supplier outages), and a **Streamlit dashboard**.

> **What you get**
> - ETL → clean daily tables  
> - Baseline forecasts + backtest metrics (WAPE, sMAPE)  
> - Policy recommendations (per Store × SKU) with safety stock & MOQ  
> - Simulator KPIs (fill rate, stockout days, orders)  
> - Streamlit UI to explore + download action plans (PO CSV)

---

# 1) environment
conda create -n genai-scm python=3.11 -y
conda activate genai-scm
pip install -r requirements.txt

# 2) put your raw file
# data/raw/retail.csv  (CSV or Parquet)

# 3) ETL → processed tables (+ optional SQLite)
python -m src.etl.prepare --raw data/raw/retail.csv --out data/processed --make-db

# 4) forecasts + validation
python -m src.forecast.backtest --sales data/processed/sales_daily.csv --horizon 28 --top_k_skus 10

# 5) inventory policy (ROP / Target-S / EOQ)
python -m src.optimize.policy --sales data/processed/sales_daily.csv --inventory data/processed/inventory_daily.csv --suppliers data/processed/suppliers.csv --forecasts models/forecasts.csv --service 0.95 --cycle_days 7 --horizon 28 --outdir data/processed

# 6) simulator (baseline & scenarios)
python -m src.risk.simulate --sales data/processed/sales_daily.csv --inventory data/processed/inventory_daily.csv --suppliers data/processed/suppliers.csv --policy data/processed/policy_recs.csv --horizon 28
python -m src.risk.simulate --sales data/processed/sales_daily.csv --inventory data/processed/inventory_daily.csv --suppliers data/processed/suppliers.csv --policy data/processed/policy_recs.csv --horizon 28 --lead_mult 1.5 --demand_spike 0.2 --supplier_outage "S007"

# 7) app
streamlit run src/app/app.py
