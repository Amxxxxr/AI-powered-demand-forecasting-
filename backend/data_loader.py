"""Cached, read-only access to processed data and model outputs."""
import json
import pandas as pd
from backend import config as C

_cache = {}


class NotReady(Exception):
    """Raised when pipeline outputs are missing."""


def reload():
    _cache.clear()


def long_df():
    if "long" not in _cache:
        if not C.LONG_FILE.exists():
            raise NotReady("Processed data not found. Run: python -m src.data_preparation")
        _cache["long"] = pd.read_parquet(C.LONG_FILE)
    return _cache["long"]


def hist_df():
    if "hist" not in _cache:
        d = long_df()
        _cache["hist"] = d[~d.is_future].reset_index(drop=True)
    return _cache["hist"]


def available_models():
    found = {p.stem.replace("predictions_", "") for p in C.FORECASTS.glob("predictions_*.csv")}
    return [m for m in C.MODEL_ORDER if m in found]


def predictions():
    if "pred" not in _cache:
        ms = available_models()
        if not ms:
            raise NotReady("No forecasts found. Run the training scripts (see README).")
        _cache["pred"] = pd.concat([pd.read_csv(C.FORECASTS / f"predictions_{m}.csv", parse_dates=["date"]) for m in ms])
    return _cache["pred"]


def metrics_json():
    f = C.METRICS / "model_metrics.json"
    if not f.exists():
        raise NotReady("Metrics not found. Run: python -m src.evaluate")
    return json.loads(f.read_text())


def filter_options():
    h, d = hist_df(), long_df()
    fut = d[d.is_future]
    return {
        "stores": sorted(h.store_id.unique()), "categories": sorted(h.cat_id.unique()),
        "items": h[["item_id", "cat_id"]].drop_duplicates().sort_values("item_id").to_dict("records"),
        "models": available_models(), "horizon": C.HORIZON,
        "history_start": str(h.date.min().date()), "history_end": str(h.date.max().date()),
        "forecast_start": str(fut.date.min().date()), "forecast_end": str(fut.date.max().date()),
    }
