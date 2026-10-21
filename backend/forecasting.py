"""Forecast retrieval + inventory planning (Phase 14)."""
import math
from statistics import NormalDist
from backend import analytics as A, data_loader as D
from backend.schemas import ApiError


def future_total(f, model):
    d = A.forecast_daily(f, model, "future")
    return float(d.prediction.sum()), d


def last_actual_total(f, days=28):
    d = A.daily(dict(f, start=None, end=None))
    return float(d.units.tail(days).sum())


def inventory_plan(f, model, inp):
    need = [k for k in ("lead_time_days", "service_level", "current_stock") if inp.get(k) in (None, "")]
    if need:
        raise ApiError("Inventory planning needs your own inputs (sales data cannot reveal stock). Missing: " + ", ".join(need), 400, "missing_inputs")
    try:
        L, sl = int(inp["lead_time_days"]), float(inp["service_level"])
        stock, onord = float(inp["current_stock"]), float(inp.get("on_order") or 0)
        moq, R = float(inp.get("moq") or 0), int(inp.get("review_days") or 7)
    except (TypeError, ValueError):
        raise ApiError("Inventory inputs must be numbers.", 400, "invalid_number")
    if not (1 <= L and L + R <= 28 and R >= 1):
        raise ApiError("lead_time_days + review_days must be between 2 and 28 (the forecast horizon).", 400, "invalid_number")
    if not 0.5 <= sl < 1 or min(stock, onord, moq) < 0:
        raise ApiError("service_level must be in [0.5, 1) and quantities cannot be negative.", 400, "invalid_number")
    fut = A.forecast_daily(f, model, "future").prediction.values
    test = A.forecast_daily(f, model, "test")
    if len(fut) < L + R or len(test) < 7:
        raise ApiError("Not enough forecast/test data for this selection.", 400, "insufficient_data")
    sigma = float((test.actual - test.prediction).std())            # daily forecast error on the test window
    z = NormalDist().inv_cdf(sl)
    d_lead, d_cycle = float(fut[:L].sum()), float(fut[:L + R].sum())
    safety = z * sigma * math.sqrt(L + R)
    rop = d_lead + z * sigma * math.sqrt(L)
    order_up_to = d_cycle + safety
    qty = max(0.0, order_up_to - stock - onord)
    if qty > 0 and moq > 0:
        qty = math.ceil(qty / moq) * moq
    return {"model": model, "reorder_point": round(rop, 1), "safety_stock": round(safety, 1), "order_up_to_level": round(order_up_to, 1),
            "recommended_order_qty": round(qty, 1), "forecast_demand_lead_time": round(d_lead, 1),
            "daily_error_std": round(sigma, 2), "z_score": round(z, 3),
            "assumptions": ["Order-up-to policy with periodic review; demand = point forecast of the selection.",
                            "Safety stock = z * daily forecast-error std (test window) * sqrt(lead time + review days).",
                            "Errors assumed independent day to day; stock, open orders, lead time, MOQ and service level are YOUR inputs."]}
