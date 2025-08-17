.PHONY: etl backtest policy simulate app lint test

etl:
\tpython -m src.etl.prepare --raw data/raw/retail.csv --out data/processed --make-db

backtest:
\tpython -m src.forecast.backtest --sales data/processed/sales_daily.csv --horizon 28 --top_k_skus 10

policy:
\tpython -m src.optimize.policy --sales data/processed/sales_daily.csv --inventory data/processed/inventory_daily.csv --suppliers data/processed/suppliers.csv --forecasts models/forecasts.csv --service 0.95 --cycle_days 7 --horizon 28 --outdir data/processed

simulate:
\tpython -m src.risk.simulate --sales data/processed/sales_daily.csv --inventory data/processed/inventory_daily.csv --suppliers data/processed/suppliers.csv --policy data/processed/policy_recs.csv --horizon 28 --outdir data/processed

app:
\tstreamlit run src/app/app.py

lint:
\truff check . && black --check .

test:
\tpytest -q
