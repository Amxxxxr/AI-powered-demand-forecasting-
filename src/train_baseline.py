"""Phase 6. Seasonal-naive baseline: forecast for a day = sales of the same weekday in the last observed week.

Run:  python -m src.train_baseline
"""
import pandas as pd
from backend import config as C
from src.common import load_long, split_dates, holdout_rows, frame_from_rows, save_predictions

MODEL = "seasonal_naive"


def seasonal_naive(df, origin, rows):
    """Predict `rows` (all dates after `origin`) using only sales up to `origin`."""
    hist = df[(~df.is_future) & (df.date <= origin)][["id", "date", "sales"]]
    rows = rows.copy()
    h = (rows["date"] - origin).dt.days                       # 1..28 days after origin
    rows["src"] = origin - pd.Timedelta(days=6) + pd.to_timedelta((h - 1) % 7, unit="D")
    m = rows[["id", "src"]].merge(hist.rename(columns={"date": "src", "sales": "pred"}), on=["id", "src"], how="left")
    return m["pred"].fillna(0).values


def main():
    df = load_long()
    s = split_dates(df)
    te, fut = holdout_rows(df, s), df[df.is_future]
    p_te = seasonal_naive(df, s["val_end"], te)
    p_fu = seasonal_naive(df, s["test_end"], fut)
    save_predictions([frame_from_rows(te, p_te, MODEL, "test"), frame_from_rows(fut, p_fu, MODEL, "future")], MODEL)
    print(f"Saved baseline forecasts ({len(te)} test rows, {len(fut)} future rows)")


if __name__ == "__main__":
    main()
