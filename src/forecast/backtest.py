import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

def ensure_daily(series: pd.Series) -> pd.Series:
    """Fill missing days with 0 demand to get a continuous daily series."""
    idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(idx, fill_value=0)

def snaive_forecast(history: pd.Series, horizon: int, season: int = 7) -> np.ndarray:
    if len(history) <= season:
        return np.repeat(history.iloc[-1] if len(history) else 0, horizon)
    base = history.shift(season).dropna()
    last_season = history.iloc[-season:] if len(history) >= season else pd.Series([history.iloc[-1]]*season)
    rep = np.resize(last_season.values, horizon)
    return rep

def naive_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    last = history.iloc[-1] if len(history) else 0
    return np.repeat(last, horizon)

def sarimax_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    # Conservative default orders; robust for retail with weekly seasonality
    order = (1,1,1)
    sorder = (1,1,1,7)
    try:
        model = SARIMAX(history, order=order, seasonal_order=sorder,
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.forecast(steps=horizon)
        return fc.values
    except Exception:
        # fallback to seasonal naive if SARIMAX fails
        return snaive_forecast(history, horizon, season=7)

def wape(y, yhat):
    denom = np.maximum(np.sum(np.abs(y)), 1e-9)
    return np.sum(np.abs(y - yhat)) / denom

def smape(y, yhat):
    num = np.abs(y - yhat)
    den = (np.abs(y) + np.abs(yhat)) / 2.0
    den = np.where(den == 0, 1e-9, den)
    return np.mean(num / den)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sales", default="data/processed/sales_daily.csv")
    ap.add_argument("--horizon", type=int, default=28)
    ap.add_argument("--top_k_skus", type=int, default=10, help="limit to top SKUs by volume for speed")
    ap.add_argument("--store_id", default=None, help="optional: evaluate only this store_id")
    args = ap.parse_args()

    df = pd.read_csv(args.sales, parse_dates=["date"])
    if args.store_id:
        df = df[df["store_id"] == str(args.store_id)]

    # pick top K SKUs for speed
    sku_rank = (df.groupby(["store_id","sku"])["qty"].sum()
                  .reset_index().sort_values("qty", ascending=False))
    keep = sku_rank.groupby("store_id").head(args.top_k_skus)[["store_id","sku"]]
    df = df.merge(keep, on=["store_id","sku"], how="inner")

    # holdout split
    max_date = df["date"].max()
    split_date = max_date - pd.Timedelta(days=args.horizon)
    train = df[df["date"] <= split_date]
    test  = df[df["date"] >  split_date]

    rows = []
    metrics = []

    for (store, sku), g in df.groupby(["store_id","sku"]):
        g = g.sort_values("date")
        s = g.set_index("date")["qty"].astype(float)
        s = ensure_daily(s)

        train_s = s[s.index <= split_date]
        test_s  = s[s.index >  split_date]

        if len(train_s) < 30 or len(test_s) == 0:
            continue

        horizon = len(test_s)

        fc_naive = naive_forecast(train_s, horizon)
        fc_snaiv = snaive_forecast(train_s, horizon, season=7)
        fc_sarim = sarimax_forecast(train_s, horizon)

        # collect forecasts
        dates = test_s.index
        def push(model, fc):
            for d, yhat in zip(dates, fc):
                rows.append({"date": d.date(), "store_id": store, "sku": sku, "model": model, "yhat": float(yhat)})
        push("naive", fc_naive)
        push("snaive7", fc_snaiv)
        push("sarimax7117", fc_sarim)

        # metrics
        y = test_s.values
        for mdl, fc in [("naive", fc_naive), ("snaive7", fc_snaiv), ("sarimax7117", fc_sarim)]:
            metrics.append({
                "store_id": store, "sku": sku, "model": mdl,
                "WAPE": wape(y, fc), "sMAPE": smape(y, fc)
            })

    Path("models").mkdir(exist_ok=True, parents=True)
    Path("reports").mkdir(exist_ok=True, parents=True)

    if rows:
        fc_df = pd.DataFrame(rows)
        fc_df.to_parquet("models/forecasts.parquet", index=False)
        fc_df.to_csv("models/forecasts.csv", index=False)

    if metrics:
        met = pd.DataFrame(metrics)
        met.to_csv("reports/metrics_summary.csv", index=False)
        overall = (met.groupby("model")[["WAPE","sMAPE"]].mean()
                     .sort_values("WAPE").reset_index())
        print("\n=== Overall validation (lower is better) ===")
        print(overall.to_string(index=False))
    else:
        print("No series had enough data for evaluation.")

if __name__ == "__main__":
    main()
