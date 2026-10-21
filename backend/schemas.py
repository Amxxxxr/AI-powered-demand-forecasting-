"""Request validation and the common error type."""
import pandas as pd
from backend import data_loader as D

FILTERS = {"item_id": "items", "store_id": "stores", "cat_id": "categories"}


class ApiError(Exception):
    def __init__(self, message, status=400, code="bad_request"):
        super().__init__(message)
        self.message, self.status, self.code = message, status, code


def parse_filters(src):
    opts = D.filter_options()
    valid = {"item_id": {i["item_id"] for i in opts["items"]}, "store_id": set(opts["stores"]), "cat_id": set(opts["categories"])}
    f = {}
    for k in FILTERS:
        v = (src.get(k) or "").strip()
        if v:
            if v not in valid[k]:
                raise ApiError(f"Unknown {k}: '{v[:40]}'.", 400, "invalid_filter")
            f[k] = v
    for k in ("start", "end"):
        v = (src.get(k) or "").strip()
        if v:
            try:
                f[k] = str(pd.to_datetime(v).date())
            except Exception:
                raise ApiError(f"'{k}' must be a date like 2016-03-01.", 400, "invalid_date")
    if f.get("start") and f.get("end") and f["start"] > f["end"]:
        raise ApiError("'start' must not be after 'end'.", 400, "invalid_date")
    return f


def parse_model(src, default=None):
    avail = D.available_models()
    if not avail:
        raise D.NotReady("No forecasts found. Run the training scripts (see README).")
    m = (src.get("model") or default or ("xgboost" if "xgboost" in avail else avail[0])).strip()
    if m not in avail:
        raise ApiError(f"Model '{m[:30]}' has no saved forecasts. Available: {', '.join(avail)}.", 404, "model_unavailable")
    return m


def parse_split(src, default="future"):
    s = src.get("split") or default
    if s not in ("test", "future"):
        raise ApiError("'split' must be 'test' or 'future'.", 400, "invalid_split")
    return s


def parse_int(src, name, default, lo, hi):
    try:
        v = int(src.get(name, default))
    except (TypeError, ValueError):
        raise ApiError(f"'{name}' must be an integer.", 400, "invalid_number")
    if not lo <= v <= hi:
        raise ApiError(f"'{name}' must be between {lo} and {hi}.", 400, "invalid_number")
    return v


def parse_chat(body):
    if not isinstance(body, dict):
        raise ApiError("Request body must be JSON.", 400, "invalid_body")
    msg = body.get("message")
    if not isinstance(msg, str) or not msg.strip():
        raise ApiError("'message' is required.", 400, "invalid_message")
    if len(msg) > 1000:
        raise ApiError("'message' must be at most 1000 characters.", 400, "invalid_message")
    ctx = body.get("context") or {}
    if not isinstance(ctx, dict):
        raise ApiError("'context' must be an object.", 400, "invalid_body")
    return msg.strip(), ctx
