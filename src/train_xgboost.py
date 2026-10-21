"""Phase 7. XGBoost direct 28-day forecaster.

Procedure: tune on validation -> evaluate best config on untouched test -> refit on all history
(same number of trees) -> forecast the 28 future days. Run:  python -m src.train_xgboost
"""
import json
import pandas as pd
from backend import config as C
from src.common import load_long, split_dates, holdout_rows, frame_from_rows, save_predictions, metrics
from src.features import build_features, FEATURES, FEATURE_DOC

MODEL = "xgboost"
GRID = [dict(max_depth=4, learning_rate=0.05, min_child_weight=5),
        dict(max_depth=6, learning_rate=0.05, min_child_weight=5),
        dict(max_depth=8, learning_rate=0.03, min_child_weight=10)]
COMMON = dict(subsample=0.8, colsample_bytree=0.8, tree_method="hist", objective="reg:squarederror",
              random_state=C.SEED, n_jobs=-1)


def main():
    from xgboost import XGBRegressor
    C.ensure_dirs()
    raw = load_long()
    s = split_dates(raw)
    df = build_features(raw)
    ok = (~df.is_future) & df.lag_28.notna()
    tr = df[ok & (df.date <= s["train_end"])]
    va = df[ok & (df.date >= s["val_start"]) & (df.date <= s["val_end"])]
    te = holdout_rows(df, s)

    tuning, best = [], None
    for p in GRID:
        m = XGBRegressor(n_estimators=1000, early_stopping_rounds=30, **p, **COMMON)
        m.fit(tr[FEATURES], tr.sales, eval_set=[(va[FEATURES], va.sales)], verbose=False)
        v = metrics(va.sales, m.predict(va[FEATURES]).clip(0))
        tuning.append({"params": p, "best_iteration": int(m.best_iteration), "val": v})
        if best is None or v["WAPE"] < best[0]:
            best = (v["WAPE"], p, int(m.best_iteration), m)
    _, params, n_best, model = best

    p_te = model.predict(te[FEATURES]).clip(0)       # model trained on TRAIN only -> honest test score
    n_final = int(n_best * 1.1) + 1
    full = df[ok & (df.date <= s["test_end"])]
    final = XGBRegressor(n_estimators=n_final, **params, **COMMON).fit(full[FEATURES], full.sales)
    fut = df[df.is_future]
    p_fu = final.predict(fut[FEATURES]).clip(0)

    save_predictions([frame_from_rows(te, p_te, MODEL, "test"), frame_from_rows(fut, p_fu, MODEL, "future")], MODEL)
    final.save_model(str(C.MODELS / "xgboost_model.json"))
    imp = sorted(zip(FEATURES, final.feature_importances_.tolist()), key=lambda x: -x[1])
    (C.METRICS / "xgboost_config.json").write_text(json.dumps({
        "chosen_params": params, "n_estimators_final": n_final, "tuning": tuning, "splits": {k: str(v.date()) for k, v in s.items()},
        "feature_importance": [{"feature": f, "importance": round(i, 5)} for f, i in imp]}, indent=2))
    (C.PREPROCESSING / "xgboost_feature_schema.json").write_text(json.dumps({"features": FEATURES, "doc": FEATURE_DOC}, indent=2))
    print("Best params:", params, "| val WAPE:", round(best[0], 2))
    print("Top features:", imp[:5])


if __name__ == "__main__":
    main()
