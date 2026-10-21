"""Phase 3. Exploratory analysis. Saves charts to outputs/reports and prints findings.

Run:  python -m src.eda
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from backend import config as C
from src.common import load_long


def save(name):
    plt.tight_layout(); plt.savefig(C.REPORTS / f"eda_{name}.png", dpi=110); plt.close()


def main():
    C.ensure_dirs()
    df = load_long()
    h = df[~df.is_future].copy()
    daily = h.groupby("date").sales.sum()
    daily.plot(figsize=(10, 3.5), title="Total daily units"); save("daily")
    h["dow"] = h.date.dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    h.groupby(["date", "dow"]).sales.sum().groupby("dow").mean().reindex(order).plot.bar(title="Average daily units by weekday"); save("weekday")
    h.groupby(h.date.dt.to_period("M")).sales.sum().plot(figsize=(10, 3.5), title="Monthly units"); save("monthly")
    h.groupby("cat_id").sales.sum().plot.bar(title="Units by category"); save("category")
    h.groupby("store_id").sales.sum().plot.bar(title="Units by store"); save("store")
    per = h.groupby("id").sales.agg(["mean", lambda x: (x == 0).mean()])
    per.columns = ["mean", "zero_share"]
    per.zero_share.plot.hist(bins=20, title="Share of zero-sales days per series"); save("zeros")
    ev = h.groupby("event_flag").sales.mean()
    pr = h.dropna(subset=["sell_price"]).groupby("id").apply(lambda g: g.sales.corr(g.sell_price), include_groups=False).dropna()
    plt.figure(); h.groupby("cat_id").apply(lambda g: g.groupby("date").sales.sum().rolling(7).mean(), include_groups=False).T.plot(title="7-day average units by category", figsize=(10, 3.5)); save("category_trend")
    summary = {"avg_units_event_days": float(ev.get(1, np.nan)), "avg_units_normal_days": float(ev.get(0, np.nan)),
               "median_zero_share_per_series": float(per.zero_share.median()),
               "series_with_over_50pct_zero_days": int((per.zero_share > 0.5).sum()),
               "median_within_series_price_sales_corr": float(pr.median()) if len(pr) else None,
               "note": "Correlations are descriptive, not causal."}
    (C.REPORTS / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
