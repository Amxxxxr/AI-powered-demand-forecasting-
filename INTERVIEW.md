# Interview prep (fill in YOUR measured numbers from outputs/metrics/model_metrics.json)

**Resume bullet (template, only after you've run it):** Built an end-to-end 28-day retail demand forecasting system on Walmart M5 (Python, XGBoost, TensorFlow LSTM, Flask, SQL, Chart.js); compared models against a seasonal-naive baseline on a chronological hold-out using WAPE/MAE/RMSE; shipped a single-page dashboard with a data-grounded AI chat assistant. *(Add: "XGBoost X% WAPE vs baseline Y%" with your real numbers.)*

**2-minute story:** Problem (stores need to plan inventory) → data (M5, 3 files, wide→long) → split (chronological train/val/test, 28-day horizon) → baseline first → XGBoost with lag≥28 features (direct strategy, no leakage) → LSTM (56→28 window) → evaluation on identical rows → Flask API + one-page dashboard + chatbot that computes facts in Python/SQL and only lets an LLM rephrase them → tests + limitations.

**Likely questions (short answers)**
* *Why not a random split?* Rows are ordered in time; random splits train on the future and give optimistic scores.
* *What is leakage here?* Using information unavailable at forecast time: future lags, rolling windows including future days, scaling/encoding fitted on test, tuning on test. We use lag≥28, train-only scaling, tuning on validation only.
* *Direct vs recursive?* Recursive feeds its own predictions back (errors compound, needs day-by-day features); direct uses features valid for the whole horizon (our lag≥28). Trade-off: direct can't use the most recent 27 days.
* *Why a baseline?* It sets the bar; a complex model must beat it to justify itself.
* *Why WAPE not MAPE?* MAPE divides by actuals; zeros break it. WAPE = Σ|e|/Σactual is stable but weights high-volume items more.
* *How does an LSTM work?* Hidden state carries a summary; cell state carries long-term memory; forget gate drops old info, input gate adds new, output gate decides what to expose. Shape: (samples, 56 steps, 5 features) → 28 outputs. Early stopping watches validation loss.
* *Did the LSTM beat XGBoost?* Answer with your measured result; never assume. Tabular lag features with trees are often very strong on retail data.
* *Intermittent demand?* ADI>1.32 and CV²>0.49 classes (Syntetos-Boylan); consider Croston/Tweedie loss as improvements.
* *Chatbot design/hallucination control?* Intent routing → whitelisted SQL/Pandas → verified text; LLM only rephrases, key stays server-side; fallback without LLM; unknown questions say "no verified data".
* *Security?* Parameterised queries, whitelisted columns, input validation, no tracebacks to users, secrets in env vars.
* *Production?* Precompute forecasts, retrain on schedule, monitor WAPE against new actuals, version models/schema, health check, logs.
* *Business impact?* Don't claim reduced stockouts/revenue without measurement; say the forecast *supports* inventory planning and what you'd measure.
