# DemandAI: AI-Powered Retail Demand Forecasting and Conversational Analytics

Forecasts the next **28 days** of retail demand on the public **Walmart M5** dataset, compares a **seasonal-naive baseline, XGBoost and an LSTM**, and shows everything in ONE page (`index.html`): an interactive dashboard with an AI chat assistant directly below it. A Python (Flask) backend serves the page and a JSON API from the same origin.

```
 Kaggle M5 CSVs ──► src/data_preparation ──► data/processed/sales_long.parquet
                                   │
        src/train_baseline · train_xgboost · train_lstm ──► outputs/forecasts/predictions_*.csv, models/
                                   │
                     src/evaluate ──► outputs/metrics/model_metrics.json + SQLite (demandai.db)
                                   │
 index.html (dashboard + chat) ◄── fetch() ──► backend/app.py (Flask) ──► analytics.py (SQL) · chatbot.py
```

## 1. Setup (Windows, PowerShell, VS Code)
```powershell
cd path\to\retail-demand-forecasting
.\.venv311\Scripts\Activate.ps1          # prompt must start with (.venv311)
python --version                          # Python 3.11.x
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import pandas, xgboost, tensorflow, flask; print('imports OK')"
```
If the install was interrupted, just run the `pip install -r` line again; pip skips what is already installed. If TensorFlow fails, run `python -m pip install tensorflow` on its own and read the error.
Script-activation error? Run once: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.

## 2. Get the data
Download from https://www.kaggle.com/competitions/m5-forecasting-accuracy/data (accept the rules once) and put these 3 files in `data\raw\`:
`sales_train_validation.csv`, `calendar.csv`, `sell_prices.csv`
(Or: `pip install kaggle`, then `kaggle competitions download -c m5-forecasting-accuracy -p data\raw` and unzip.)

| File | Purpose |
|---|---|
| sales_train_validation | One row per item-store series (30,490), columns `d_1..d_1913` = units sold per day (wide format) |
| calendar | Maps `d_N` to a date; weekday, month, events, SNAP (food-stamp) flags; also covers the 28 days after the data, used for the future forecast |
| sell_prices | Weekly price per store-item, joined on `store_id, item_id, wm_yr_wk` |

## 3. Run everything
```powershell
python run_pipeline.py            # data prep, EDA, baseline, XGBoost, LSTM, evaluation
python -m backend.app             # then open http://127.0.0.1:5000
```
Useful options: `python run_pipeline.py --skip-lstm` (faster), or run steps one by one:
`python -m src.data_preparation`, `python -m src.eda`, `python -m src.train_baseline`, `python -m src.train_xgboost`, `python -m src.train_lstm`, `python -m src.evaluate`.

**Subset (laptop-friendly default):** stores CA_1, TX_1, WI_1; last 730 days; top 30 items per category (270 series). Items are ranked on data *before* validation/test. Scale up via environment variables, e.g. `$env:DEMANDAI_ITEMS_PER_CAT=200; $env:DEMANDAI_HISTORY_DAYS=1500` (also `DEMANDAI_STORES`, `DEMANDAI_LSTM_EPOCHS`). Limitation: top sellers are less intermittent than the full catalogue, so scores on this subset will look better than on all 30,490 series.

## 4. Method (what to defend in an interview)
* **Split (chronological):** train | validation (28 d) | test (28 d). Forecast origin = last day before each window. Random splits would leak the future into training.
* **Strategy: direct.** All sales features use lags ≥ 28 days, so every one of the 28 target days only needs data known at the origin (no recursive error build-up). `tests/test_pipeline.py` proves scrambling test-period sales changes no test feature.
* **Baseline:** seasonal naive: forecast = sales on the same weekday in the last observed week.
* **XGBoost:** lags 28/35/42, rolling mean/std/zero-share (on the 28-day-shifted series), price, calendar, events, SNAP, id codes. Grid tuned on validation with early stopping; best config scored on test (trained on train only); then refit on all history (same tree count) for the future forecast.
* **LSTM:** input = 56 past days × [scaled sales, sin/cos weekday, SNAP, event]; output = 28 values at once; per-series scale = *train* mean; EarlyStopping on validation loss. Its future forecast uses the model trained through the validation period (not refit).
* **Metrics:** MAE, RMSE, WAPE = Σ|error| / Σactual. MAPE is avoided because of zero-sales days. All models are scored on identical series and dates (`src/evaluate.py` intersects rows).
* **Rolling-origin evaluation** (several forecast origins) is the natural next step; this version uses one test window.

## 5. API (all JSON: `{"status","data","meta"}`; errors `{"status":"error","error":{"code","message"}}`)
Filters accepted by most endpoints: `item_id, store_id, cat_id, start, end` (dates `YYYY-MM-DD`), plus `model`.

| Endpoint | Used by the page for |
|---|---|
| `GET /api/health` | startup check, shows setup message if outputs are missing |
| `GET /api/filters` | fills dropdowns from real data |
| `GET /api/summary` | KPI cards (units, avg/day, MAE/RMSE/WAPE for the selection) |
| `GET /api/historical-sales` | history chart (SQL 7-day window average), weekday chart |
| `GET /api/forecast?split=test\|future&limit=` | actual-vs-predicted, future forecast, table |
| `GET /api/model-metrics` | model comparison chart/table |
| `GET /api/product-performance?by=item_id\|store_id\|cat_id` | comparison chart |
| `GET /api/forecast.csv` | CSV download |
| `POST /api/chat` `{message, context}` | chatbot |
| `POST /api/inventory` | optional reorder-point / order-quantity calculator |

## 6. Chatbot
`chatbot.py` classifies the question, computes the answer with fixed, parameterised SQL/Pandas functions (no model-written SQL or code), and returns it as **"Verified from data"**. If `ANTHROPIC_API_KEY` is set on the **server**, an LLM adds a separately labelled **"AI explanation"** that may only rephrase those facts. No key, or the API is down: you still get the verified answer.
Easiest: copy `.env.example` to `.env` (same folder as `index.html`), paste your key after `ANTHROPIC_API_KEY=`, save, restart the server. `.env` is git-ignored.
Alternative for one terminal session: `$env:ANTHROPIC_API_KEY="sk-ant-..."` then `python -m backend.app`. Check it worked at `/api/health` (`"llm_configured": true`).
Never put the key in `index.html` or commit it.

## 7. Tests
```powershell
python -m pytest -q
```
35 tests build a small **synthetic** M5-format dataset in a temp folder (test fixture only, never shown in the app) and run the entire pipeline: validation, no-leakage, chronological split, forecast schema, identical test rows, metric formulas, model save/load, SQL vs Pandas, every API endpoint, invalid input, missing model files, chatbot numbers, CSV, inventory maths.

## 8. Deployment (Render example)
1. Run the pipeline locally. Commit `data/processed/sales_long.parquet`, `outputs/`, `models/` (small). Do **not** commit raw CSVs or `.env`.
2. Push to GitHub. On render.com create a **Web Service** from the repo: Build `pip install -r requirements-serve.txt`, Start `gunicorn backend.app:app --workers 1 --timeout 120` (the `Procfile` has it). Serving needs no TensorFlow/XGBoost because forecasts are precomputed.
3. Add `ANTHROPIC_API_KEY` as a secret environment variable (optional). Health check path: `/api/health`.
4. To update models: retrain locally, commit the new `outputs/` files, push; the service redeploys.
Check the free tier's current RAM/sleep limits before choosing a plan.

## 9. Power BI (optional)
Get Data → Text/CSV: `outputs/forecasts/predictions_*.csv` (append them) and `outputs/metrics/model_comparison.csv`. Build slicers on store/category/model, a line chart of `actual` vs `prediction` by `date`, and measures `WAPE = DIVIDE(SUMX(T, ABS(T[actual]-T[prediction])), SUM(T[actual]))`.

## 10. Results
Run `python -m src.evaluate`; your real scores appear in `outputs/metrics/model_metrics.json` and in the dashboard. Paste them here with screenshots. Nothing in this repo claims results that were not measured.

## 11. Limitations / future work
Subset only; one test window; sales ≠ demand (stock-outs look like zeros); point forecasts only (no intervals); LSTM has no item embeddings; no promotions data; inventory feature needs user-supplied stock inputs. Next: rolling-origin CV, quantile forecasts, hierarchical reconciliation, full 30,490 series, LightGBM.
