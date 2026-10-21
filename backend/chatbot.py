"""Phase 12. Chatbot: classify the question -> compute verified facts with fixed Python/SQL functions ->
(optionally) let an LLM rephrase ONLY those facts. No model-generated SQL or code is ever executed."""
import logging
import pandas as pd
import requests
from backend import analytics as A, config as C, data_loader as D, forecasting as F

log = logging.getLogger("demandai.chat")
DEFS = ("MAE = average absolute error in units per row. RMSE = square root of the mean squared error; punishes big misses more. "
        "WAPE = total absolute error / total actual units (shown as %); easy to read and safe when some days have zero sales, "
        "but it is dominated by high-volume products. Lower is better for all three.")
LIMITS = ("1) Trained on a subset (selected stores, top items per category, recent history). 2) Point forecasts, no guarantee of demand. "
          "3) Sales are not demand: stock-outs show up as zeros. 4) One 28-day test window; rankings can change in other periods. "
          "5) Price/promotion changes and new items are not anticipated beyond calendar and price features.")
INVENTORY = ("Examine supplier lead time, target service level, current stock and open orders, minimum order quantities, and forecast error "
             "(safety stock). This dataset has no inventory data, so use the inventory panel and enter your own values.")


def _scope(f):
    parts = [f[k] for k in ("cat_id", "store_id", "item_id") if f.get(k)]
    return " / ".join(parts) if parts else "all selected stores and categories"


def classify(q):
    q = q.lower()
    has = lambda *w: any(x in q for x in w)
    if has("limitation", "limit", "caveat", "reliable", "trust"): return "limitations"
    if has("inventory", "reorder", "stock", "safety"): return "inventory"
    if has("wape", "mae", "rmse", "metric", "mape"): return "metric_definition"
    if has("compare", "lstm", "xgboost", "best model", "which model", "baseline"): return "model_comparison"
    if has("irregular", "intermittent", "zero", "lumpy", "erratic"): return "irregular"
    if q.startswith("why") or "why is" in q or "why are" in q: return "why_forecast"
    if "store" in q and not has("product", "item"): return "top_store"
    if has("product", "item", "highest demand", "best seller", "top sell"): return "top_product"
    if has("category", "department"): return "top_category"
    if has("forecast", "predict", "next 28", "next month", "expected demand"): return "forecast_total"
    return "unknown"


def _top(by, f, label):
    d = A.performance(by, f, 3)
    if d.empty: return "No sales found for this selection.", {}
    rows = [{"key": r.key, "units": float(r.units)} for r in d.itertuples()]
    txt = f"Top {label} in {_scope(f)}: " + "; ".join(f"{r['key']} ({r['units']:,.0f} units)" for r in rows) + "."
    return txt, {"top": rows}


def _irregular(f):
    h = D.hist_df()
    for k in ("item_id", "store_id", "cat_id"):
        if f.get(k): h = h[h[k] == f[k]]
    out = []
    for sid, g in h.groupby("id"):
        nz = g.sales[g.sales > 0]
        if len(nz) < 2: continue
        adi, cv2 = len(g) / len(nz), (nz.std() / nz.mean()) ** 2
        out.append({"id": sid, "zero_share": round(float((g.sales == 0).mean()), 3), "adi": round(adi, 2), "cv2": round(float(cv2), 2),
                    "pattern": ("lumpy" if adi > 1.32 and cv2 > 0.49 else "intermittent" if adi > 1.32 else "erratic" if cv2 > 0.49 else "smooth")})
    if not out: return "Not enough sales history to classify demand patterns here.", {}
    d = pd.DataFrame(out)
    counts = d.pattern.value_counts().to_dict()
    top = d.sort_values("zero_share", ascending=False).head(5).to_dict("records")
    txt = (f"In {_scope(f)}, demand patterns (Syntetos-Boylan classes; ADI>1.32 = intermittent, CV2>0.49 = erratic): {counts}. "
           "Most irregular series (highest share of zero-sales days): " + "; ".join(f"{r['id']} ({r['zero_share']:.0%} zero days, {r['pattern']})" for r in top) + ".")
    return txt, {"pattern_counts": counts, "top_irregular": top}


def _forecast_total(f, model):
    tot, d = F.future_total(f, model)
    if d.empty: return "No forecast available for this selection.", {}
    last = F.last_actual_total(f)
    return (f"{model} forecasts {tot:,.0f} units for {_scope(f)} over {d.date.min()} to {d.date.max()} "
            f"(about {tot / len(d):,.0f}/day). The previous 28 observed days totalled {last:,.0f} units. This is a point forecast, not a guarantee."), \
           {"model": model, "forecast_total": tot, "previous_28d_actual": last, "days": len(d)}


def _why(f, model):
    tot, d = F.future_total(f, model)
    if d.empty: return "No forecast available for this selection.", {}
    last = F.last_actual_total(f)
    chg = (tot / last - 1) * 100 if last else float("nan")
    fut = D.long_df(); fut = fut[fut.is_future]
    events, snap = int(fut.drop_duplicates("date").event_flag.sum()), int(fut.drop_duplicates(["date", "state_id"]).snap.sum())
    direction = "higher" if chg > 0 else "lower"
    txt = (f"For {_scope(f)}, the {model} forecast is {abs(chg):.1f}% {direction} than the previous 28 observed days "
           f"({tot:,.0f} vs {last:,.0f} units). Calendar facts in the forecast window: {events} event day(s), {snap} state-level SNAP day-flags. "
           "The model learns patterns from history (weekday cycle, recent level, price, events); this explains correlation, not proven cause.")
    if chg <= 0: txt = "The forecast does not indicate an increase. " + txt
    return txt, {"change_pct": chg, "forecast_total": tot, "previous_28d_actual": last, "event_days": events}


def _compare(f):
    m = D.metrics_json()
    ms = m["models"]
    txt = "Test-window metrics (" + f"{m['test_start']} to {m['test_end']}, {m['n_rows']:,} rows): " + "; ".join(
        f"{k}: MAE {v['MAE']:.3f}, RMSE {v['RMSE']:.3f}, WAPE {v['WAPE']:.1f}%" for k, v in ms.items()) + \
        f". Lowest WAPE on this window: {m['lowest_wape_model']}. This ranking applies to this window only."
    return txt, {"metrics": ms}


def verified(intent, f, model):
    if intent == "top_product": return _top("item_id", f, "items")
    if intent == "top_store": return _top("store_id", f, "stores")
    if intent == "top_category": return _top("cat_id", f, "categories")
    if intent == "forecast_total": return _forecast_total(f, model)
    if intent == "why_forecast": return _why(f, model)
    if intent == "model_comparison": return _compare(f)
    if intent == "irregular": return _irregular(f)
    if intent == "metric_definition": return DEFS, {}
    if intent == "limitations": return LIMITS, {}
    if intent == "inventory": return INVENTORY, {}
    return ("I can answer questions about top products/stores/categories, the 28-day forecast, why it changes, model comparison, "
            "metric meanings, irregular demand, limitations and inventory planning. I don't have verified data for that question."), {}


def llm_explain(question, text, facts):
    if not C.LLM_API_KEY: return None
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=20,
                          headers={"x-api-key": C.LLM_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          json={"model": C.LLM_MODEL, "max_tokens": 400,
                                "system": "You explain retail forecasting results in simple English. Use ONLY the verified facts given. "
                                          "Never add numbers, products or trends not in them. If they do not answer the question, say so. Max 120 words.",
                                "messages": [{"role": "user", "content": f"Question: {question}\nVerified facts: {text}\nData: {facts}"}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", [])).strip() or None
    except Exception as e:  # LLM is optional: fall back silently to the verified answer
        log.warning("LLM unavailable: %s", type(e).__name__)
        return None


def answer(question, f, model):
    intent = classify(question)
    text, facts = verified(intent, f, model)
    ai = llm_explain(question, text, facts) if intent != "unknown" else None
    return {"intent": intent, "verified_answer": text, "ai_explanation": ai, "llm_used": ai is not None, "facts": facts,
            "context": {**f, "model": model}}
