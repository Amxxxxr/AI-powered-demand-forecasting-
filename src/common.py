"""Shared helpers: loading, chronological split dates, metrics, prediction file format."""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from backend import config as C

PRED_COLS = ["id", "item_id", "store_id", "cat_id", "date", "split", "actual", "prediction", "model", "generated_at"]


def load_long():
    if not C.LONG_FILE.exists():
        raise FileNotFoundError("Run `python -m src.data_preparation` first.")
    return pd.read_parquet(C.LONG_FILE).sort_values(["id", "date"]).reset_index(drop=True)


def split_dates(df):
    """Chronological split: train | validation (28d) | test (28d). Future rows are excluded."""
    d = np.sort(df.loc[~df.is_future, "date"].unique())
    H, ts = C.HORIZON, pd.Timestamp
    return {"train_end": ts(d[-2 * H - 1]), "val_start": ts(d[-2 * H]), "val_end": ts(d[-H - 1]),
            "test_start": ts(d[-H]), "test_end": ts(d[-1])}


def holdout_rows(df, s):
    return df[(~df.is_future) & (df.date >= s["test_start"]) & (df.date <= s["test_end"])]


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = y - p
    wape = float(np.abs(err).sum() / y.sum() * 100) if y.sum() > 0 else float("nan")
    return {"MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean())), "WAPE": wape}


def frame_from_rows(rows, pred, model, split):
    out = rows[["id", "item_id", "store_id", "cat_id", "date"]].copy()
    out["split"] = split
    out["actual"] = rows["sales"].values if split == "test" else np.nan
    out["prediction"] = np.clip(np.asarray(pred, float), 0, None).round(3)
    out["model"] = model
    out["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out[PRED_COLS]


def save_predictions(frames, model):
    C.ensure_dirs()
    pd.concat(frames).to_csv(C.FORECASTS / f"predictions_{model}.csv", index=False)
