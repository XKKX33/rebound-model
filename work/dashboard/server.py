"""Read-only dashboard adapter around the existing repositories."""
from __future__ import annotations
import json
import math
import os
import sys
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path

from paths import PROJECT, AGORA_REPO, MODEL_REPO, DATABASE, PROFILE
sys.path[:0] = [
    str(Path(__file__).parent),
    str(PROJECT / "work/runtime"),
    str(AGORA_REPO),
    str(MODEL_REPO),
]
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from starlette.middleware.gzip import GZipMiddleware
import httpx
from market_utils import aggregate, Stock, rank
from rebound.store import Store
import duckdb
import incremental
import research_views
from start_debate_ui import EmbeddedDebate


class MarketStore(Store):
    """UI overlay of separate increments; research calculations retain the frozen database."""
    def __init__(self, db_path, *, visible_through=None):
        self.path = db_path
        self.read_only = True
        self.visible_through = visible_through
        self._con = duckdb.connect()
        incremental.attach_views(self._con)


Store = MarketStore

STATIC = Path(__file__).parent
SERVICE_STATE = STATIC / "service_state.json"


@asynccontextmanager
async def lifespan(application):
    debate = EmbeddedDebate()
    application.state.debate = debate
    application.state.debate_client = httpx.AsyncClient(
        base_url=debate.origin, timeout=None, trust_env=False,
    )
    state = {
        "PID": os.getpid(),
        "Port": int(os.environ.get("DASHBOARD_PORT", "8875")),
        "Script": str(Path(__file__).resolve()),
        "DebateDirectory": str(debate.directory),
        "StartedAt": datetime.now(timezone.utc).isoformat(),
    }
    pending_state = SERVICE_STATE.with_name(f".service-state-{os.getpid()}.tmp")
    pending_state.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    pending_state.replace(SERVICE_STATE)
    try:
        yield
    finally:
        try:
            await application.state.debate_client.aclose()
            debate.close()
        finally:
            try:
                if json.loads(SERVICE_STATE.read_text(encoding="utf-8")).get("PID") == os.getpid():
                    SERVICE_STATE.unlink()
            except (OSError, ValueError):
                pass


app = FastAPI(title="A 股量化选股与智能研判", docs_url=None, redoc_url=None, lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)
app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

_PROXY_DROP_HEADERS = {
    "host", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "content-length",
}


@app.get("/api/health")
def health():
    return {"status": "ready", "title": app.title, "debate_path": "/debate/"}


@app.api_route("/debate/{route:path}", methods=["GET", "POST", "DELETE"])
async def debate_proxy(route: str, request: Request):
    """Forward member APIs and streaming responses through the dashboard URL."""
    query = request.url.query
    target = "/" + route + ("?" + query if query else "")
    client = request.app.state.debate_client
    outbound = client.build_request(
        request.method, target,
        headers={key: value for key, value in request.headers.items() if key.lower() not in _PROXY_DROP_HEADERS},
        content=await request.body(),
    )
    try:
        upstream = await client.send(outbound, stream=True)
    except httpx.RequestError as exc:
        raise HTTPException(503, "AI 辩论台暂时无法连接，请稍后重试") from exc
    return StreamingResponse(
        upstream.aiter_raw(), status_code=upstream.status_code,
        headers={key: value for key, value in upstream.headers.items() if key.lower() not in _PROXY_DROP_HEADERS},
        background=BackgroundTask(upstream.aclose),
    )


def records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=10))


@lru_cache(maxsize=1)
def metadata():
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    with Store(DATABASE) as store:
        basic = store.connection.execute(
            "SELECT ts_code,symbol,name,industry,market,list_date FROM stock_basic ORDER BY ts_code"
        ).fetchall()
    catalog = [
        Stock(row[0], row[1] or row[0].split(".")[0], row[2] or "", row[3] or "",
              row[4] or "", str(row[5]) if row[5] else None)
        for row in basic
    ]
    return {
        "schema_version": "1",
        "dataset_id": "research-" + profile["provenance"]["sha256"][:12],
        "provider": "Tushare Pro",
        "source_url": "https://tushare.pro",
        "daily_docs": "https://tushare.pro/document/2?doc_id=27",
        "basic_docs": "https://tushare.pro/document/2?doc_id=32",
        "first_date": profile["tables"]["daily"]["date_min"],
        "last_date": profile["tables"]["daily"]["date_max"],
        "tables": [
            {"name": name, **{k: v for k, v in info.items() if k != "columns"}}
            for name, info in profile["tables"].items()
        ],
        "quality": profile["daily_quality"],
        "catalog": catalog,
    }


def available_date(as_of):
    latest = date.fromisoformat(get_metadata()["last_date"])
    target = as_of or latest
    if target > latest:
        raise HTTPException(422, f"行情已更新至 {latest}，请选择已收录日期")
    if target < date.fromisoformat(metadata()["first_date"]):
        raise HTTPException(422, "日期早于本地行情范围")
    return target


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/meta")
def get_metadata():
    result = {k: v for k, v in metadata().items() if k != "catalog"}
    result["incremental"] = incremental.get_status()
    result["last_date"] = max(result["last_date"], result["incremental"].get("published_last_date") or result["last_date"])
    result["model_data"] = research_views.model_metadata()
    return result


@app.middleware("http")
async def refresh_incremental_cache(request, call_next):
    global _incremental_revision
    revision = incremental.STATUS.stat().st_mtime_ns if incremental.STATUS.exists() else 0
    if revision != _incremental_revision:
        metadata.cache_clear()
        snapshot.cache_clear()
        raw_history.cache_clear()
        stock_history.cache_clear()
        _incremental_revision = revision
    return await call_next(request)


_incremental_revision = 0


@app.get("/api/stocks/search")
def stock_search(q: str = Query(max_length=60)):
    ranked = rank(metadata()["catalog"], q)
    return {"total": len(ranked), "results": [stock.as_dict() for stock, _ in ranked[:20]]}


@lru_cache(maxsize=8)
def snapshot(as_of):
    with Store(DATABASE, visible_through=as_of) as store:
        frame = store.connection.execute("""
            SELECT d.ts_code,d.trade_date,d.close,d.pct_chg,d.amount/100000 amount_yi,
                   b.symbol,b.name,b.industry,b.market,x.turnover_rate,x.total_mv/10000 total_mv_yi,
                   x.circ_mv/10000 circ_mv_yi
            FROM daily d LEFT JOIN stock_basic b USING(ts_code)
            LEFT JOIN daily_basic x USING(ts_code,trade_date)
            WHERE d.trade_date=?
        """, [as_of]).fetchdf()
        indexes = store.connection.execute(
            "SELECT ts_code,close,pct_chg FROM index_daily WHERE trade_date=? ORDER BY ts_code", [as_of]
        ).fetchdf()
    return frame, indexes


@app.get("/api/market")
def market(
    as_of: date | None = None,
    market: str = Query("all", pattern="^(all|main|gem|star|bj)$"),
    q: str = Query("", max_length=60),
    sort: str = Query("amount", pattern="^(amount|change|price|turnover|code)$"),
    desc: bool = True,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=5, le=50),
    codes: str = Query("", max_length=2000),
):
    target = available_date(as_of)
    full, indexes = snapshot(target)
    frame = full.copy()
    if market != "all":
        boards = {"main": "主板", "gem": "创业板", "star": "科创板", "bj": "北交所"}
        frame = frame[frame.market == boards[market]]
    if codes:
        frame = frame[frame.ts_code.isin(codes.split(","))]
    if q.strip():
        matches = {stock.ts_code for stock, _ in rank(metadata()["catalog"], q)}
        frame = frame[frame.ts_code.isin(matches)]
    col = {"amount": "amount_yi", "change": "pct_chg", "price": "close", "turnover": "turnover_rate", "code": "ts_code"}[sort]
    frame = frame.sort_values([col, "ts_code"] if col != "ts_code" else [col],
                              ascending=[not desc, True] if col != "ts_code" else [not desc],
                              na_position="last", kind="stable")
    start = (page - 1) * page_size
    index_names = {"000001.SH": "上证指数", "000300.SH": "沪深300", "000905.SH": "中证500", "000852.SH": "中证1000"}
    index_records = records(indexes)
    for row in index_records:
        row["name"] = index_names.get(row["ts_code"], row["ts_code"])
    return {
        "as_of": target.isoformat(),
        "total": len(frame), "page": page, "page_size": page_size,
        "results": records(frame.iloc[start:start + page_size]),
        "indexes": index_records,
        "breadth": {
            "count": len(full), "rising": int((full.pct_chg > 0).sum()),
            "falling": int((full.pct_chg < 0).sum()), "unchanged": int((full.pct_chg == 0).sum()),
            "amount_yi": float(full.amount_yi.sum()),
        },
    }


@lru_cache(maxsize=16)
def raw_history(code, target):
    with Store(DATABASE, visible_through=target) as store:
        frame = store.connection.execute("""
            SELECT d.trade_date,d.open,d.high,d.low,d.close,d.pre_close,d.pct_chg,d.vol,d.amount,
                   a.adj_factor,m.net_mf_amount,x.turnover_rate,x.total_mv,x.circ_mv,
                   l.up_limit,l.down_limit
            FROM daily d LEFT JOIN adj_factor a USING(ts_code,trade_date)
            LEFT JOIN moneyflow m USING(ts_code,trade_date)
            LEFT JOIN daily_basic x USING(ts_code,trade_date)
            LEFT JOIN daily_limit l USING(ts_code,trade_date)
            WHERE d.ts_code=? AND d.trade_date<=?
            ORDER BY d.trade_date
        """, [code, target]).fetchdf()
    return frame


@lru_cache(maxsize=72)
def stock_history(code, target, period, adjustment, count):
    frame = raw_history(code, target).copy()
    if frame.empty:
        raise HTTPException(404, "所选日期之前没有该股票行情")
    frame.trade_date = pd.to_datetime(frame.trade_date)
    raw = frame.iloc[-1].to_dict()
    basic = next((s for s in metadata()["catalog"] if s.ts_code == code), None)
    adjusted = frame.copy()
    if adjustment == "qfq":
        if adjusted.adj_factor.isna().any() or (adjusted.adj_factor <= 0).any():
            raise HTTPException(422, "复权因子存在缺口，请切换不复权")
        factor = adjusted.adj_factor / adjusted.adj_factor.iloc[-1]
        for col in ["open", "high", "low", "close", "pre_close"]:
            adjusted[col] *= factor
    columns = ["trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount", "net_mf_amount"]
    grouped = aggregate(adjusted[columns].copy(), period=period, sums=("net_mf_amount",)).copy()
    grouped["ma5"] = grouped.close.rolling(5).mean()
    grouped["ma10"] = grouped.close.rolling(10).mean()
    grouped["ma20"] = grouped.close.rolling(20).mean()
    grouped["amount_yi"] = grouped.amount / 100000
    grouped["net_mf_yi"] = grouped.net_mf_amount / 10000
    grouped["volume_wan"] = grouped.vol / 100
    grouped.trade_date = pd.to_datetime(grouped.trade_date).dt.strftime("%Y-%m-%d")
    quote = {}
    for k, v in raw.items():
        if k == "trade_date":
            quote[k] = str(pd.Timestamp(v).date())
        elif pd.isna(v) or (isinstance(v, float) and not math.isfinite(v)):
            quote[k] = None
        else:
            quote[k] = v
    return {
        "ts_code": code, "name": basic.name if basic else code,
        "industry": basic.industry if basic else "",
        "market": basic.market if basic else "",
        "list_date": basic.list_date if basic else None,
        "as_of": target.isoformat(), "quote_date": quote["trade_date"],
        "period": period, "adjustment": adjustment,
        "quote": quote, "bars": records(grouped.tail(count)),
        "available_bars": len(grouped),
    }


@lru_cache(maxsize=256)
def _sequence_window(code: str, target: date):
    """Build the delivered 60 x 11 LSTM window with its original rank transform."""
    import numpy as np
    import pandas as pd
    from ml import seq_data
    from ml.preprocess import cross_section_rank

    first_raw = raw_history(code, target)
    if first_raw.empty:
        raise HTTPException(404, "所选日期之前没有该股票行情")
    target_ts = pd.Timestamp(target)
    with Store(DATABASE, visible_through=target) as store:
        panel_dates = store.connection.execute(
            "SELECT DISTINCT trade_date FROM daily WHERE trade_date <= ? ORDER BY trade_date",
            [target],
        ).df()["trade_date"]
        panel_dates = pd.DatetimeIndex(pd.to_datetime(panel_dates))
        target_index = panel_dates.searchsorted(target_ts, side="right") - 1
        if target_index < 59:
            raise HTTPException(422, "该日期前的市场交易历史不足60个交易日")
        dates = panel_dates[target_index - 59:target_index + 1]
        start = dates[0].date()
        raw = seq_data.raw_range(store.connection, start, target)
        days = raw.trade_date.to_numpy()
        ranked = cross_section_rank(raw[list(seq_data.RANKED)], days).fillna(0.0)
        ranked["limit"] = raw["limit"].to_numpy(np.float32)
        ranked["traded"] = np.float32(1.0)
        ranked = ranked[list(seq_data.CHANNELS)]
        per_date = {}
        selected = raw.ts_code.eq(code).to_numpy()
        for day, positions in raw.groupby("trade_date", sort=False).indices.items():
            positions = np.asarray(positions, dtype=np.int64)
            hit = positions[selected[positions]]
            if len(hit):
                per_date[pd.Timestamp(day)] = ranked.iloc[hit[-1]].to_numpy(dtype=np.float32)
    if not per_date:
        raise HTTPException(422, "所选股票不在 LSTM 使用的沪深主板序列样本中")
    rows = []
    for day in dates:
        key = pd.Timestamp(day)
        values = per_date.get(key)
        if values is None:
            values = np.zeros(len(seq_data.CHANNELS), dtype=np.float32)
            values[-1] = 0.0
        rows.append({"date": key.strftime("%Y-%m-%d"), **{name: float(value) for name, value in zip(seq_data.CHANNELS, values)}})
    return {"as_of": rows[-1]["date"], "start_date": rows[0]["date"], "code": code,
            "window": len(rows), "channels": list(seq_data.CHANNELS), "rows": rows}


@app.get("/api/stocks/{code}/sequence")
def sequence_window(code: str, as_of: date | None = None):
    if not any(stock.ts_code == code for stock in metadata()["catalog"]):
        raise HTTPException(404, "股票代码不在本地目录中")
    target = available_date(as_of)
    if target > date.fromisoformat(metadata()["last_date"]):
        raise HTTPException(422, "序列研究使用冻结历史库，请选择历史库已收录日期")
    return _sequence_window(code, target)


@app.get("/api/stocks/{code}/bars")
def bars(
    code: str, as_of: date | None = None,
    period: str = Query("day", pattern="^(day|week|month)$"),
    adjustment: str = Query("none", pattern="^(none|qfq)$"),
    count: int = Query(100, ge=30, le=500),
):
    if not any(s.ts_code == code for s in metadata()["catalog"]):
        raise HTTPException(404, "股票代码不在本地目录中")
    return stock_history(code, available_date(as_of), period, adjustment, count)


@app.get("/api/modules/{name}")
def module_status(
    name: str,
    as_of: date | None = None,
    score_as_of: date | None = None,
    code: str = "600519.SH",
    model: str = "xgb_F69",
    rule: str = "DB-K30-b8",
    period: str = "test",
):
    if name not in {"features", "models", "portfolio", "agents"}:
        raise HTTPException(404, "模块不存在")
    if model not in research_views.MODEL_IDS:
        raise HTTPException(422, "请选择已交付的模型")
    target = available_date(as_of)
    try:
        if name == "features":
            frozen_end = date.fromisoformat(metadata()["last_date"])
            if target > frozen_end:
                raise HTTPException(422, "特征研究使用冻结历史库，请选择历史库已收录日期")
            return research_views.feature_data(
                target,
                code,
                raw_history(code, target),
                model,
                score_as_of=score_as_of,
            )
        frame, _ = snapshot(target)
        if name == "models":
            return research_views.model_comparison(target, model, frame, rule=rule, period=period)
        selected_code = code if name == "agents" else None
        return research_views.module_data(target, frame, model, selected_code=selected_code)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/incremental/status")
def incremental_status():
    return incremental.get_status()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, host=os.environ.get("DASHBOARD_HOST", "127.0.0.1"),
        port=int(os.environ.get("DASHBOARD_PORT", "8875")), log_level="warning",
    )
