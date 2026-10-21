"""Phase 13. Flask app: serves index.html AND the JSON API from the same origin (no CORS needed).

Run locally:  python -m backend.app      Production: gunicorn backend.app:app
"""
import io
import logging
from flask import Flask, jsonify, request, send_from_directory, Response
from backend import analytics as A, chatbot, config as C, data_loader as D, forecasting as F
from backend.schemas import ApiError, parse_filters, parse_model, parse_split, parse_int, parse_chat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("demandai")
app = Flask(__name__, static_folder=None)


def ok(data, **meta):
    return jsonify({"status": "ok", "data": data, "meta": meta})


@app.errorhandler(ApiError)
def _api_err(e):
    return jsonify({"status": "error", "error": {"code": e.code, "message": e.message}}), e.status


@app.errorhandler(D.NotReady)
def _not_ready(e):
    return jsonify({"status": "error", "error": {"code": "not_ready", "message": str(e)}}), 503


@app.errorhandler(404)
def _nf(e):
    return jsonify({"status": "error", "error": {"code": "not_found", "message": "Unknown endpoint."}}), 404


@app.errorhandler(Exception)
def _boom(e):
    log.exception("Unhandled error")          # traceback stays in the server log only
    return jsonify({"status": "error", "error": {"code": "internal", "message": "Something went wrong on the server."}}), 500


@app.get("/")
def index():
    return send_from_directory(C.APP_DIR, "index.html")


@app.get("/static/<path:name>")
def static_files(name):
    return send_from_directory(C.APP_DIR / "static", name)


@app.get("/api/health")
def health():
    ready = C.LONG_FILE.exists() and bool(D.available_models())
    return ok({"status": "healthy", "data_ready": ready, "models": D.available_models(), "llm_configured": bool(C.LLM_API_KEY)})


@app.get("/api/filters")
def filters():
    return ok(D.filter_options())


@app.get("/api/summary")
def summary():
    f, m = parse_filters(request.args), parse_model(request.args)
    sel = A.selection_metrics(f, m)
    return ok({"kpis": A.kpis(f), "model": m, "horizon_days": C.HORIZON, "selection_metrics": sel, "global_metrics": D.metrics_json()["models"].get(m)},
              metric_notes={"MAE": "mean absolute error (units/row)", "RMSE": "root mean squared error (units/row)", "WAPE": "sum|error| / sum(actual), %"})


@app.get("/api/historical-sales")
def historical():
    f = parse_filters(request.args)
    return ok({"daily": A.daily(f).round(2).to_dict("records"), "weekday": A.weekday(f).round(2).to_dict("records"), "monthly": A.monthly(f).to_dict("records")})


@app.get("/api/forecast")
def forecast():
    f, m, sp = parse_filters(request.args), parse_model(request.args), parse_split(request.args)
    lim = parse_int(request.args, "limit", 500, 1, 5000)
    rows = A.forecast_rows(f, m, sp, lim)
    return ok({"daily": A.forecast_daily(f, m, sp).round(2).where(lambda d: d.notna(), None).to_dict("records"),
               "rows": rows.round(3).where(rows.notna(), None).to_dict("records")},
              model=m, split=sp, total_rows=A.forecast_count(f, m, sp),
              label="historical evaluation (test) predictions" if sp == "test" else "future point forecast, not a guarantee")


@app.get("/api/forecast.csv")
def forecast_csv():
    f, m, sp = parse_filters(request.args), parse_model(request.args), parse_split(request.args)
    buf = io.StringIO()
    A.forecast_rows(f, m, sp).to_csv(buf, index=False)
    return Response(buf.getvalue(), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename=forecast_{m}_{sp}.csv"})


@app.get("/api/model-metrics")
def model_metrics():
    f = parse_filters(request.args)
    data = D.metrics_json()
    if f:
        data["selection"] = {m: A.selection_metrics(f, m) for m in D.available_models()}
    return ok(data)


@app.get("/api/product-performance")
def performance():
    by = request.args.get("by", "item_id")
    if by not in ("item_id", "store_id", "cat_id", "dept_id"):
        raise ApiError("'by' must be item_id, store_id, cat_id or dept_id.", 400, "invalid_group")
    f = parse_filters(request.args)
    return ok(A.performance(by, f, parse_int(request.args, "limit", 10, 1, 100)).round(3).to_dict("records"), by=by)


@app.post("/api/chat")
def chat():
    msg, ctx = parse_chat(request.get_json(silent=True))
    f, m = parse_filters(ctx), parse_model(ctx)
    return ok(chatbot.answer(msg, f, m))


@app.post("/api/inventory")
def inventory():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ApiError("Request body must be JSON.", 400, "invalid_body")
    ctx = body.get("context") or {}
    return ok(F.inventory_plan(parse_filters(ctx), parse_model(ctx), body))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(__import__("os").getenv("PORT", 5000)), debug=False)
