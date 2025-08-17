# GenAI Supply Chain — Forecasting → Inventory Policy → Simulator → App

**Akhila Madanapati** • Repo: `am3469/genai-scm`  
Turn raw retail data into **forecasts**, **reorder policies (ROP / Target-S / EOQ)**, a **scenario simulator** (lead-time ×, demand spikes, supplier outages), and a **Streamlit dashboard** for decisions.

## Why this is interesting
- Business KPI focus: **fill rate, stockout days, orders, on-hand**
- Practical constraints: **lead times, MOQ, supplier mapping**
- Clear pipeline: **ETL → backtest → policy → simulate → app**

---

## 🔧 Quickstart

```bash
# 1) environment
conda create -n genai-scm python=3.11 -y
conda activate genai-scm
pip install -r requirements.txt

# 2) place your raw file
# data/raw/retail.csv   (CSV or Parquet)

# 3) ETL → standardized daily tables (+ optional SQLite)
python -m src.etl.prepare --raw data/raw/retail.csv --out data/processed --make-db

# 4) forecasts + validation metrics (WAPE, sMAPE)
python -m src.forecast.backtest --sales data/processed/sales_daily.csv --horizon 28 --top_k_skus 10

# 5) inventory policy (ROP / Target-S / EOQ) with safety stock & MOQ
python -m src.optimize.policy \
  --sales data/processed/sales_daily.csv \
  --inventory data/processed/inventory_daily.csv \
  --suppliers data/processed/suppliers.csv \
  --forecasts models/forecasts.csv \
  --service 0.95 --cycle_days 7 --horizon 28 --outdir data/processed

# 6) simulator (baseline & scenarios)
python -m src.risk.simulate \
  --sales data/processed/sales_daily.csv \
  --inventory data/processed/inventory_daily.csv \
  --suppliers data/processed/suppliers.csv \
  --policy data/processed/policy_recs.csv \
  --horizon 28

# example stress test
python -m src.risk.simulate \
  --sales data/processed/sales_daily.csv \
  --inventory data/processed/inventory_daily.csv \
  --suppliers data/processed/suppliers.csv \
  --policy data/processed/policy_recs.csv \
  --horizon 28 --lead_mult 1.5 --demand_spike 0.2 --supplier_outage "S007"

# 7) dashboard
streamlit run src/app/app.py

flowchart LR
  A[Raw data: data/raw] --> B[ETL: src/etl/prepare.py]
  B --> C[Processed tables: data/processed/*]
  C --> D[Backtest: src/forecast/backtest.py]
  D --> E[Forecasts: models/forecasts.*]
  C --> F[Policy: src/optimize/policy.py]
  E --> F
  F --> G[Simulator: src/risk/simulate.py]
  G --> H[Streamlit App: src/app/app.py]


.
├── src
│   ├── app/        # Streamlit (app.py)
│   ├── etl/        # prepare.py (schema + standardization)
│   ├── forecast/   # backtest.py (baselines + metrics)
│   ├── optimize/   # policy.py (ROP/Target-S/EOQ, safety stock, MOQ)
│   └── risk/       # simulate.py (scenario engine + KPIs)
├── data
│   ├── raw/        # put retail.csv here (gitignored)
│   └── processed/  # ETL outputs, policy, sim outputs (gitignored)
├── models/         # forecasts (gitignored)
├── reports/        # metrics (WAPE/sMAPE)
├── requirements.txt
├── README.md
└── Makefile



### Two tiny extras (optional but great)
- In the repo’s **About** (right side), set the description to:  
  `ETL → forecasting → inventory policy → scenario simulator → Streamlit dashboard.`
- Add topics: `supply-chain`, `forecasting`, `inventory-optimization`, `time-series`, `streamlit`, `python`.

If anything still renders as plain text, ensure each block starts with three backticks (```), like ```bash, ```mermaid, or ```text, and ends with three backticks.
::contentReference[oaicite:0]{index=0}
