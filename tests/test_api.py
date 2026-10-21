import pandas as pd
import pytest
from backend import config as C, data_loader as D


def test_health_and_filters(client):
    h = client.get("/api/health").get_json()["data"]
    assert h["data_ready"] and "xgboost" in h["models"] and h["llm_configured"] is False
    f = client.get("/api/filters").get_json()["data"]
    assert f["stores"] == ["CA_1", "TX_1"] and f["horizon"] == 28 and len(f["items"]) > 0


def test_summary_and_filter_changes_numbers(client):
    a = client.get("/api/summary").get_json()["data"]["kpis"]["total_units"]
    b = client.get("/api/summary?store_id=CA_1").get_json()["data"]["kpis"]["total_units"]
    assert 0 < b < a
    assert client.get("/api/summary").get_json()["data"]["selection_metrics"]["WAPE"] >= 0


def test_forecast_schema_and_horizon(client):
    j = client.get("/api/forecast?split=future&model=seasonal_naive").get_json()
    assert j["status"] == "ok" and len(j["data"]["daily"]) == 28
    assert set(j["data"]["rows"][0]) >= {"date", "item_id", "store_id", "cat_id", "prediction", "model", "split", "generated_at"}


@pytest.mark.parametrize("url,code", [
    ("/api/summary?store_id=NOPE", 400), ("/api/forecast?split=bad", 400), ("/api/forecast?model=prophet", 404),
    ("/api/historical-sales?start=notadate", 400), ("/api/historical-sales?start=2020-12-01&end=2020-01-01", 400),
    ("/api/product-performance?by=id;DROP TABLE", 400), ("/api/forecast?limit=abc", 400), ("/api/nothing", 404)])
def test_invalid_inputs(client, url, code):
    r = client.get(url)
    assert r.status_code == code and r.get_json()["status"] == "error" and "Traceback" not in r.get_data(as_text=True)


def test_csv_export(client):
    r = client.get("/api/forecast.csv?model=xgboost&split=future&store_id=CA_1")
    assert r.mimetype == "text/csv"
    df = pd.read_csv(pd.io.common.StringIO(r.get_data(as_text=True)))
    assert len(df) == 12 * 28 and df.store_id.eq("CA_1").all()   # 3 cats x 4 items x 1 store x 28 days
    assert list(df.columns)[:2] == ["date", "item_id"]


def test_chat_numbers_match_pandas(client):
    h = D.hist_df()
    top_store = h.groupby("store_id").sales.sum().idxmax()
    r = client.post("/api/chat", json={"message": "Which store has the highest sales?"}).get_json()["data"]
    assert r["intent"] == "top_store" and top_store in r["verified_answer"] and r["llm_used"] is False
    units = h.groupby("item_id").sales.sum().max()
    r = client.post("/api/chat", json={"message": "Which product has the highest demand?"}).get_json()["data"]
    assert f"{units:,.0f}" in r["verified_answer"]


@pytest.mark.parametrize("q,intent", [("What is the predicted demand for the next 28 days?", "forecast_total"),
    ("Why is demand expected to increase?", "why_forecast"), ("How does LSTM compare with XGBoost?", "model_comparison"),
    ("What does the WAPE metric mean?", "metric_definition"), ("Which products show irregular demand?", "irregular"),
    ("What are the limitations of this forecast?", "limitations"), ("What inventory-planning considerations should the business examine?", "inventory"),
    ("what is the weather", "unknown")])
def test_chat_intents(client, q, intent):
    r = client.post("/api/chat", json={"message": q, "context": {"cat_id": "FOODS", "model": "xgboost"}})
    assert r.status_code == 200 and r.get_json()["data"]["intent"] == intent


def test_chat_validation(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 400
    assert client.post("/api/chat", json={"message": "x" * 2000}).status_code == 400
    assert client.post("/api/chat", data="not json").status_code == 400
    assert client.post("/api/chat", json={"message": "hi", "context": {"store_id": "ZZ"}}).status_code == 400


def test_inventory(client):
    ctx = {"store_id": "CA_1"}
    r = client.post("/api/inventory", json={"context": ctx})
    assert r.status_code == 400 and "Missing" in r.get_json()["error"]["message"]
    r = client.post("/api/inventory", json={"context": ctx, "lead_time_days": 7, "service_level": 0.95, "current_stock": 50, "moq": 10})
    d = r.get_json()["data"]
    assert r.status_code == 200 and d["recommended_order_qty"] % 10 == 0 and d["reorder_point"] >= d["forecast_demand_lead_time"]


def test_missing_model_files_gives_clear_error(client, monkeypatch, tmp_path):
    monkeypatch.setattr(C, "FORECASTS", tmp_path)
    D.reload()
    try:
        r = client.get("/api/forecast")
        assert r.status_code == 503 and "Run" in r.get_json()["error"]["message"]
    finally:
        monkeypatch.undo(); D.reload()


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and b"DemandAI" in r.data
