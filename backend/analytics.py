"""Phase 15. SQL analytics on SQLite. Only whitelisted columns are interpolated; all values are bound parameters."""
import sqlite3
import pandas as pd
from backend import config as C, data_loader as D
from src.common import metrics

FILTER_COLS = ("item_id", "store_id", "cat_id")
GROUP_COLS = ("item_id", "store_id", "cat_id", "dept_id")


def build_db():
    h = D.hist_df()[["id", "item_id", "dept_id", "cat_id", "store_id", "date", "sales", "sell_price", "event_flag"]].copy()
    h["date"] = h["date"].dt.strftime("%Y-%m-%d")
    p = D.predictions().copy()
    p["date"] = p["date"].dt.strftime("%Y-%m-%d")
    C.DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(C.DB_FILE) as con:
        h.to_sql("sales", con, if_exists="replace", index=False)
        p.to_sql("forecasts", con, if_exists="replace", index=False)
        con.executescript("""
            CREATE INDEX IF NOT EXISTS ix_sales ON sales(item_id, store_id, date);
            CREATE INDEX IF NOT EXISTS ix_fc ON forecasts(model, split, date);
            DROP VIEW IF EXISTS daily_sales;
            CREATE VIEW daily_sales AS SELECT date, SUM(sales) AS units FROM sales GROUP BY date;""")


def _con():
    stale = not C.DB_FILE.exists() or C.DB_FILE.stat().st_mtime < C.LONG_FILE.stat().st_mtime
    if stale:
        build_db()
    return sqlite3.connect(C.DB_FILE)


def _q(sql, params=()):
    with _con() as con:
        return pd.read_sql_query(sql, con, params=list(params))


def _where(f, dates=True, extra=()):
    c, p = [], []
    for k in FILTER_COLS:
        if f.get(k):
            c.append(f"{k} = ?"); p.append(f[k])
    if dates:
        if f.get("start"):
            c.append("date >= ?"); p.append(f["start"])
        if f.get("end"):
            c.append("date <= ?"); p.append(f["end"])
    for clause, v in extra:
        c.append(clause); p.append(v)
    return ("WHERE " + " AND ".join(c)) if c else "", p


def kpis(f):
    w, p = _where(f)
    r = _q(f"SELECT COALESCE(SUM(sales),0) units, COUNT(DISTINCT date) days, COUNT(DISTINCT id) series, "
           f"MIN(date) first_day, MAX(date) last_day FROM sales {w}", p).iloc[0]
    units, days = float(r.units), int(r.days)
    return {"total_units": units, "days": days, "series": int(r.series), "avg_daily_units": units / days if days else 0.0,
            "first_day": r.first_day, "last_day": r.last_day}


def daily(f):
    w, p = _where(f)
    return _q(f"""SELECT date, units, AVG(units) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma7
                  FROM (SELECT date, SUM(sales) units FROM sales {w} GROUP BY date) ORDER BY date""", p)


def weekday(f):
    w, p = _where(f)
    return _q(f"""SELECT CAST(dow AS INTEGER) dow, AVG(units) avg_units FROM
                  (SELECT date, strftime('%w', date) dow, SUM(sales) units FROM sales {w} GROUP BY date)
                  GROUP BY dow ORDER BY dow""", p)


def monthly(f):
    w, p = _where(f)
    return _q(f"SELECT strftime('%Y-%m', date) month, SUM(sales) units FROM sales {w} GROUP BY month ORDER BY month", p)


def performance(by, f, limit=10):
    if by not in GROUP_COLS:
        raise ValueError("invalid grouping")
    w, p = _where(f)
    return _q(f"""SELECT {by} AS key, SUM(sales) units, AVG(sales) avg_daily_per_series,
                  1.0*SUM(CASE WHEN sales=0 THEN 1 ELSE 0 END)/COUNT(*) zero_share
                  FROM sales {w} GROUP BY {by} ORDER BY units DESC LIMIT ?""", p + [int(limit)])


def forecast_daily(f, model, split):
    w, p = _where(f, dates=False, extra=[("model = ?", model), ("split = ?", split)])
    return _q(f"SELECT date, SUM(actual) actual, SUM(prediction) prediction FROM forecasts {w} GROUP BY date ORDER BY date", p)


def forecast_rows(f, model, split, limit=None):
    w, p = _where(f, dates=False, extra=[("model = ?", model), ("split = ?", split)])
    lim = f"LIMIT {int(limit)}" if limit else ""
    return _q(f"SELECT date, item_id, store_id, cat_id, prediction, actual, model, split, generated_at "
              f"FROM forecasts {w} ORDER BY date, item_id, store_id {lim}", p)


def forecast_count(f, model, split):
    w, p = _where(f, dates=False, extra=[("model = ?", model), ("split = ?", split)])
    return int(_q(f"SELECT COUNT(*) n FROM forecasts {w}", p).n[0])


def selection_metrics(f, model):
    """MAE/RMSE/WAPE of `model` on the TEST rows inside the current filter (row level)."""
    w, p = _where(f, dates=False, extra=[("model = ?", model), ("split = ?", "test")])
    r = _q(f"SELECT actual, prediction FROM forecasts {w}", p)
    return metrics(r.actual, r.prediction) if len(r) else None
