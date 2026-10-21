"""Phase 9. Compare models on IDENTICAL test rows; write metrics + plots; build the SQL database.

Run:  python -m src.evaluate
"""
import json
from datetime import datetime, timezone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from backend import config as C
from src.common import metrics


def main():
    C.ensure_dirs()
    tests = {}
    for f in sorted(C.FORECASTS.glob("predictions_*.csv")):
        d = pd.read_csv(f, parse_dates=["date"])
        tests[d.model.iloc[0]] = d[d.split == "test"].set_index(["id", "date"]).sort_index()
    if not tests:
        raise FileNotFoundError("No prediction files. Train the models first.")
    idx = None
    for d in tests.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    tests = {m: d.loc[idx] for m, d in tests.items()}
    models = [m for m in C.MODEL_ORDER if m in tests]
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "test_start": str(idx.get_level_values("date").min().date()), "test_end": str(idx.get_level_values("date").max().date()),
           "n_series": int(idx.get_level_values("id").nunique()), "n_rows": int(len(idx)),
           "note": "All models scored on identical series/dates. Ranking applies to this test window only.",
           "models": {}, "by_category": {}, "by_store": {}}
    for m in models:
        d = tests[m]
        out["models"][m] = metrics(d.actual, d.prediction)
        for key, col in (("by_category", "cat_id"), ("by_store", "store_id")):
            out[key][m] = {k: metrics(g.actual, g.prediction) for k, g in d.groupby(col)}
    out["lowest_wape_model"] = min(out["models"], key=lambda m: out["models"][m]["WAPE"])
    (C.METRICS / "model_metrics.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(out["models"]).T.round(4).to_csv(C.METRICS / "model_comparison.csv", index_label="model")
    print(pd.DataFrame(out["models"]).T.round(3).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    pd.DataFrame(out["models"]).T[["MAE", "RMSE"]].plot.bar(ax=ax[0], rot=0, title="Error by model (lower is better)")
    for m in models:
        tests[m].groupby("date").prediction.sum().plot(ax=ax[1], label=m)
    tests[models[0]].groupby("date").actual.sum().plot(ax=ax[1], label="actual", color="black", lw=2)
    ax[1].legend(); ax[1].set_title("Total units, test window")
    plt.tight_layout(); plt.savefig(C.REPORTS / "model_comparison.png", dpi=120)

    from backend import analytics, data_loader
    data_loader.reload()
    analytics.build_db()
    print("Metrics saved; SQL database rebuilt.")


if __name__ == "__main__":
    main()
