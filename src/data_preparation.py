"""Phase 2. Raw M5 CSVs -> one tidy long table: data/processed/sales_long.parquet.

Run:  python -m src.data_preparation
"""
import json
import numpy as np
import pandas as pd
from backend import config as C

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def load_raw():
    names = ["sales_train_validation", "calendar", "sell_prices"]
    missing = [f"{n}.csv" for n in names if not (C.RAW / f"{n}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Missing in data/raw: {missing}. See README step 'Get the data'.")
    sales, cal, prices = (pd.read_csv(C.RAW / f"{n}.csv") for n in names)
    return sales, cal, prices[prices.store_id.isin(C.STORES)]


def select_subset(sales):
    """Keep chosen stores, the last HISTORY_DAYS days, and the top-N items per category.
    Items are ranked using days BEFORE validation/test so the choice never peeks at the test period."""
    sales = sales[sales.store_id.isin(C.STORES)].copy()
    if sales.empty:
        raise ValueError(f"No rows for stores {C.STORES}")
    day_cols = sorted([c for c in sales.columns if c.startswith("d_")], key=lambda c: int(c[2:]))
    keep = day_cols[-C.HISTORY_DAYS:]
    rank_days = keep[:-2 * C.HORIZON]
    sales["_t"] = sales[rank_days].sum(axis=1)
    top = (sales.groupby(["cat_id", "item_id"])["_t"].sum().reset_index()
           .sort_values("_t", ascending=False).groupby("cat_id").head(C.ITEMS_PER_CAT))
    return sales[sales.item_id.isin(top.item_id)][ID_COLS + keep], keep


def build_long(sales, keep, cal, prices):
    cal = cal.copy()
    cal["d_num"] = cal["d"].str[2:].astype(int)
    cal["date"] = pd.to_datetime(cal["date"])
    cal["event_flag"] = (cal.event_name_1.notna() | cal.event_name_2.notna()).astype(int)
    cols = ["d", "d_num", "date", "wm_yr_wk", "wday", "month", "year", "event_flag", "snap_CA", "snap_TX", "snap_WI"]

    long = sales.melt(id_vars=ID_COLS, value_vars=keep, var_name="d", value_name="sales")
    n_before = len(long)
    long = long.merge(cal[cols], on="d", how="left", validate="many_to_one")
    assert len(long) == n_before and long.date.notna().all(), "Calendar join changed rows or missed dates"

    last = int(keep[-1][2:])
    fut_cal = cal[cal.d_num > last].sort_values("d_num").head(C.HORIZON)
    if len(fut_cal) < C.HORIZON:
        raise ValueError("calendar.csv has fewer than 28 days after the last sales day")
    fut = sales[ID_COLS].drop_duplicates().merge(fut_cal[cols], how="cross")
    fut["sales"] = np.nan
    long["is_future"], fut["is_future"] = False, True
    long = pd.concat([long, fut], ignore_index=True)

    long["snap"] = np.select([long.state_id == "CA", long.state_id == "TX", long.state_id == "WI"],
                             [long.snap_CA, long.snap_TX, long.snap_WI], 0)
    n_before = len(long)
    long = long.drop(columns=["snap_CA", "snap_TX", "snap_WI", "d"]).merge(
        prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one")
    assert len(long) == n_before, "Price join changed the row count"
    return long.sort_values(["id", "date"]).reset_index(drop=True)


def validate(long):
    h = long[~long.is_future]
    rep = {
        "rows": int(len(long)), "series": int(long.id.nunique()),
        "history_start": str(h.date.min().date()), "history_end": str(h.date.max().date()),
        "future_start": str(long[long.is_future].date.min().date()), "future_end": str(long[long.is_future].date.max().date()),
        "duplicate_id_date_rows": int(long.duplicated(["id", "date"]).sum()),
        "missing_sales_in_history": int(h.sales.isna().sum()), "negative_sales": int((h.sales < 0).sum()),
        "missing_price_share": round(float(long.sell_price.isna().mean()), 4),
        "zero_sales_share": round(float((h.sales == 0).mean()), 4),
        "stores": C.STORES, "items_per_category": C.ITEMS_PER_CAT, "history_days": C.HISTORY_DAYS,
        "assumptions": [
            "Zero sales are kept as zero demand (stock-outs cannot be separated from true zero demand).",
            "Missing sell_price means the item was not yet priced/sold that week; left as NaN.",
            "Top-selling items per category are chosen using data before validation/test only.",
        ]}
    if rep["duplicate_id_date_rows"] or rep["missing_sales_in_history"] or rep["negative_sales"]:
        raise ValueError(f"Data validation failed: {rep}")
    return rep


def main():
    C.ensure_dirs()
    sales, cal, prices = load_raw()
    sub, keep = select_subset(sales)
    long = build_long(sub, keep, cal, prices)
    rep = validate(long)
    long.to_parquet(C.LONG_FILE, index=False)
    (C.REPORTS / "data_validation.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"\nSaved {C.LONG_FILE}")


if __name__ == "__main__":
    main()
