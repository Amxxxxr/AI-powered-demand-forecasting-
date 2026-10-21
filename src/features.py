"""Phase 4. Leakage-safe features.

Rule: every sales-based feature uses a lag of AT LEAST 28 days (= forecast horizon).
So for any day in the next 28 days, the feature only uses sales that are already known at the
forecast origin. This is the "direct" strategy: one model, no recursive feeding of its own predictions.
"""
import numpy as np
import pandas as pd

CAT_IDS = ["store_id", "item_id", "cat_id", "dept_id"]
FEATURES = ["lag_28", "lag_35", "lag_42", "roll_mean_7", "roll_mean_28", "roll_std_28", "roll_zero_share_28",
            "sell_price", "price_ratio", "dow", "wday", "month", "day_of_month", "week_of_year", "is_weekend",
            "event_flag", "snap"] + [c + "_code" for c in CAT_IDS]

FEATURE_DOC = {
    "lag_28/35/42": "Sales 28/35/42 days earlier (same weekday). Known at forecast time for all 28 horizon days.",
    "roll_mean_7/28, roll_std_28, roll_zero_share_28": "Rolling stats of sales shifted by 28 days: recent level, volatility, intermittency.",
    "sell_price, price_ratio": "Planned price and price vs its recent average. Prices are set by the retailer ahead of time.",
    "dow, wday, month, day_of_month, week_of_year, is_weekend": "Calendar facts, always known in advance.",
    "event_flag, snap": "Holiday/event and SNAP food-stamp days from the calendar, published in advance.",
    "*_code": "Integer codes for store, item, category, department (fixed identifiers, not derived from sales).",
}


def build_features(df):
    df = df.sort_values(["id", "date"]).reset_index(drop=True).copy()
    g = df.groupby("id", sort=False)["sales"]
    df["lag_28"], df["lag_35"], df["lag_42"] = g.shift(28), g.shift(35), g.shift(42)
    df["_s"] = df["lag_28"]  # sales shifted by the horizon; rolling windows are computed on THIS
    r = df.groupby("id", sort=False)["_s"]
    df["roll_mean_7"] = r.transform(lambda x: x.rolling(7).mean())
    df["roll_mean_28"] = r.transform(lambda x: x.rolling(28).mean())
    df["roll_std_28"] = r.transform(lambda x: x.rolling(28).std())
    df["roll_zero_share_28"] = r.transform(lambda x: x.eq(0).astype(float).where(x.notna()).rolling(28).mean())
    pm = df.groupby("id", sort=False)["sell_price"].transform(lambda x: x.shift(28).rolling(28, min_periods=7).mean())
    df["price_ratio"] = df["sell_price"] / pm
    d = df["date"].dt
    df["dow"], df["day_of_month"] = d.dayofweek, d.day
    df["week_of_year"] = d.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    for c in CAT_IDS:
        df[c + "_code"] = df[c].astype("category").cat.codes
    return df.drop(columns="_s")
