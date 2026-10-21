import json
import numpy as np
import pandas as pd
import pytest
from backend import config as C
from src.common import load_long, split_dates, holdout_rows, metrics
from src.features import build_features, FEATURES
from src.train_baseline import seasonal_naive


def test_data_validation_report(pipeline):
    rep = json.loads((C.REPORTS / "data_validation.json").read_text())
    assert rep["duplicate_id_date_rows"] == 0 and rep["negative_sales"] == 0
    assert rep["stores"] == ["CA_1", "TX_1"]            # subset respected (WI_1 excluded)


def test_chronological_split(pipeline):
    s = split_dates(load_long())
    assert s["train_end"] < s["val_start"] <= s["val_end"] < s["test_start"] <= s["test_end"]
    assert (s["test_end"] - s["test_start"]).days == C.HORIZON - 1


def test_future_rows_follow_history(pipeline):
    df = load_long()
    assert df[df.is_future].date.min() > df[~df.is_future].date.max()
    assert df[df.is_future].groupby("id").size().eq(C.HORIZON).all()


def test_features_do_not_use_future_sales(pipeline):
    """Scrambling test-period sales must not change ANY test-row feature (no leakage)."""
    df = load_long(); s = split_dates(df)
    a = build_features(df)
    d2 = df.copy()
    m = (~d2.is_future) & (d2.date >= s["test_start"])
    d2.loc[m, "sales"] = np.random.default_rng(1).integers(100, 200, m.sum())
    b = build_features(d2)
    t = (a.date >= s["test_start"]) & (a.date <= s["test_end"]) & ~a.is_future   # future rows may legitimately use test-period sales as lags
    pd.testing.assert_frame_equal(a.loc[t, FEATURES].reset_index(drop=True), b.loc[t, FEATURES].reset_index(drop=True))


def test_baseline_ignores_sales_after_origin(pipeline):
    df = load_long(); s = split_dates(df)
    te = holdout_rows(df, s)
    p1 = seasonal_naive(df, s["val_end"], te)
    d2 = df.copy(); d2.loc[(~d2.is_future) & (d2.date > s["val_end"]), "sales"] = 9999
    assert np.array_equal(p1, seasonal_naive(d2, s["val_end"], holdout_rows(d2, s)))


def test_prediction_schema_and_identical_rows(pipeline):
    cols = ["id", "item_id", "store_id", "cat_id", "date", "split", "actual", "prediction", "model", "generated_at"]
    keys = []
    for f in C.FORECASTS.glob("predictions_*.csv"):
        d = pd.read_csv(f)
        assert list(d.columns) == cols and (d.prediction >= 0).all()
        assert d[d.split == "future"].actual.isna().all() and d[d.split == "test"].actual.notna().all()
        keys.append(set(zip(d[d.split == "test"].id, d[d.split == "test"].date)))
    assert all(k == keys[0] for k in keys)


def test_metrics_formulas():
    m = metrics([0, 10, 10], [0, 8, 14])
    assert m["MAE"] == pytest.approx(2.0) and m["RMSE"] == pytest.approx(np.sqrt(20 / 3)) and m["WAPE"] == pytest.approx(30.0)


def test_metrics_file_matches_recomputation(pipeline):
    mm = json.loads((C.METRICS / "model_metrics.json").read_text())
    d = pd.read_csv(C.FORECASTS / "predictions_xgboost.csv")
    d = d[d.split == "test"]
    assert mm["models"]["xgboost"]["MAE"] == pytest.approx(metrics(d.actual, d.prediction)["MAE"])


def test_model_files_saved_and_loadable(pipeline):
    from xgboost import XGBRegressor
    m = XGBRegressor(); m.load_model(str(C.MODELS / "xgboost_model.json"))
    assert (C.PREPROCESSING / "xgboost_feature_schema.json").exists()
    if (C.MODELS / "lstm_model.keras").exists():
        from tensorflow import keras
        assert keras.models.load_model(C.MODELS / "lstm_model.keras").output_shape[-1] == C.HORIZON


def test_sql_matches_pandas(pipeline):
    from backend import analytics as A, data_loader as D
    h = D.hist_df()
    assert A.kpis({})["total_units"] == pytest.approx(h.sales.sum())
    top = h.groupby("item_id").sales.sum().sort_values(ascending=False)
    sql = A.performance("item_id", {}, 1).iloc[0]
    assert sql.key == top.index[0] and sql.units == pytest.approx(top.iloc[0])
    f = {"store_id": "CA_1", "cat_id": "FOODS"}
    assert A.kpis(f)["total_units"] == pytest.approx(h[(h.store_id == "CA_1") & (h.cat_id == "FOODS")].sales.sum())
