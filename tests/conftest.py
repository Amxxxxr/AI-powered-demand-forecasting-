"""Builds a SYNTHETIC dataset in the exact M5 file format (test fixture only - never real results),
runs the full pipeline on it in a temp folder, and exposes a Flask test client."""
import importlib.util
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

TMP = Path(tempfile.mkdtemp(prefix="demandai_test_"))
os.environ.update({"DEMANDAI_ROOT": str(TMP), "DEMANDAI_STORES": "CA_1,TX_1", "DEMANDAI_ITEMS_PER_CAT": "4",
                   "DEMANDAI_HISTORY_DAYS": "300", "DEMANDAI_LSTM_EPOCHS": "2", "DEMANDAI_LSTM_STRIDE": "5"})
os.environ["DEMANDAI_DISABLE_DOTENV"] = "1"
os.environ.pop("ANTHROPIC_API_KEY", None)
HAS_TF = importlib.util.find_spec("tensorflow") is not None


def make_raw():
    rng = np.random.default_rng(0)
    raw = TMP / "data" / "raw"
    raw.mkdir(parents=True)
    n_days, n_cal = 400, 428
    dates = pd.date_range("2020-01-04", periods=n_cal)           # starts on a Saturday like M5
    cal = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "wm_yr_wk": 11101 + np.arange(n_cal) // 7,
                        "weekday": dates.day_name(), "wday": (np.arange(n_cal) % 7) + 1, "month": dates.month, "year": dates.year,
                        "d": [f"d_{i + 1}" for i in range(n_cal)], "event_name_1": [None] * n_cal, "event_type_1": [None] * n_cal,
                        "event_name_2": [None] * n_cal, "event_type_2": [None] * n_cal,
                        "snap_CA": rng.integers(0, 2, n_cal), "snap_TX": rng.integers(0, 2, n_cal), "snap_WI": rng.integers(0, 2, n_cal)})
    cal.loc[::37, "event_name_1"] = "Holiday"
    cal.to_csv(raw / "calendar.csv", index=False)
    rows, prices = [], []
    for cat in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
        for i in range(6):
            item = f"{cat}_1_{i:03d}"
            for store in ["CA_1", "TX_1", "WI_1"]:
                base = rng.uniform(1, 8)
                wk = 1 + 0.5 * np.sin(2 * np.pi * np.arange(n_days) / 7)
                s = rng.poisson(base * wk).astype(int)
                rows.append({"id": f"{item}_{store}_validation", "item_id": item, "dept_id": f"{cat}_1", "cat_id": cat, "store_id": store,
                             "state_id": store[:2], **{f"d_{d + 1}": s[d] for d in range(n_days)}})
                for w in np.unique(cal.wm_yr_wk):
                    prices.append((store, item, w, round(rng.uniform(1, 5), 2)))
    pd.DataFrame(rows).to_csv(raw / "sales_train_validation.csv", index=False)
    pd.DataFrame(prices, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"]).to_csv(raw / "sell_prices.csv", index=False)


make_raw()


@pytest.fixture(scope="session")
def pipeline():
    from src import data_preparation, train_baseline, train_xgboost, train_lstm, evaluate
    data_preparation.main(); train_baseline.main(); train_xgboost.main()
    if HAS_TF:
        train_lstm.main()
    evaluate.main()
    return TMP


@pytest.fixture(scope="session")
def client(pipeline):
    from backend.app import app
    app.testing = True
    return app.test_client()
