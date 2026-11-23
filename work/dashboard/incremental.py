"""Separate, validated Tushare increments. Historical data is always attached read-only."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import time
import sys
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from paths import PROJECT, DATABASE as BASE, MODEL_REPO, PROFILE, INCREMENTAL_DIRECTORY as DIRECTORY
sys.path.insert(0, str(PROJECT / "work/runtime"))
import numpy as np
import pandas as pd
import duckdb

DATABASE = DIRECTORY / "market.duckdb"
READER = DIRECTORY / "market.read.duckdb"
STATUS = DIRECTORY / "status.json"
TABLES = ("daily", "adj_factor", "daily_basic", "moneyflow", "daily_limit", "index_daily",
          "trade_cal", "stock_basic")
KEYS = {t: ("ts_code", "trade_date") for t in TABLES}
KEYS.update(trade_cal=("exchange", "cal_date"), stock_basic=("ts_code",))
INDEXES = ("000001.SH", "000300.SH", "000905.SH", "000852.SH")


def token():
    value = os.environ.get("TUSHARE_TOKEN", "").strip()
    if value:
        return value
    for path in (DIRECTORY / ".env", PROJECT / ".env", MODEL_REPO / ".env"):
        if path.is_file():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                key, sep, val = line.partition("=")
                if sep and key.strip() == "TUSHARE_TOKEN":
                    value = val.strip().strip("'\"")
                    if value and value != "YOUR_TOKEN":
                        return value
    return None


def profile():
    return json.loads(PROFILE.read_text(encoding="utf-8"))


def write_status(payload):
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = datetime.now().astimezone().isoformat()
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(STATUS)


def get_status():
    result = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {
        "status": "not_initialized", "last_date": None, "tables": {},
    }
    return {**result, "credential_present": bool(token()), "provider": "Tushare Pro",
            "historical_last_date": profile()["tables"]["daily"]["date_max"]}


def publish():
    """Copy only after the writer connection closes; readers never attach the active writer."""
    temp = READER.with_suffix(".tmp.duckdb")
    shutil.copy2(DATABASE, temp)
    for attempt in range(5):
        try:
            temp.replace(READER)
            break
        except PermissionError:
            if attempt == 4:
                temp.unlink(missing_ok=True)
                raise RuntimeError("行情查询暂占用读取快照，请稍后重试发布") from None
            time.sleep(.3)
    state = get_status()
    state["published_at"] = datetime.now().astimezone().isoformat()
    state["published_last_date"] = state.get("last_date")
    write_status(state)


def initialize(path=DATABASE, base=BASE):
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(path)) as con:
        con.execute(f"ATTACH '{base.as_posix()}' AS history (READ_ONLY)")
        for table in TABLES:
            columns = con.execute(f"DESCRIBE history.{table}").fetchall()
            fields = ",".join(f'"{r[0]}" {r[1]}' for r in columns)
            primary = ",".join(KEYS[table])
            con.execute(f"CREATE TABLE IF NOT EXISTS {table} ({fields}, PRIMARY KEY ({primary}))")
        con.execute("CREATE TABLE IF NOT EXISTS raw_adj_factor (ts_code VARCHAR, trade_date DATE, "
                    "adj_factor DOUBLE, PRIMARY KEY(ts_code,trade_date))")
        con.execute("CREATE TABLE IF NOT EXISTS factor_scale (ts_code VARCHAR PRIMARY KEY, scale DOUBLE, "
                    "overlap_days BIGINT, relative_spread DOUBLE)")
        con.execute("CREATE TABLE IF NOT EXISTS sync_log (trade_date DATE, endpoint VARCHAR, status VARCHAR, "
                    "rows BIGINT, detail VARCHAR, updated_at TIMESTAMP, PRIMARY KEY(trade_date,endpoint))")
        con.execute("CREATE TABLE IF NOT EXISTS committed_days (trade_date DATE PRIMARY KEY)")
    return path


def validated(frame, table, columns, target=None):
    """Missing fields remain null; keys, date, bounds and finite numbers are verified."""
    if frame.empty:
        raise ValueError(f"{table} 返回空数据")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{table} 缺少字段: {','.join(sorted(missing))}")
    frame = frame[list(columns)].copy()
    for key in KEYS[table]:
        if frame[key].isna().any():
            raise ValueError(f"{table} 主键为空")
    for col in ("trade_date", "cal_date", "list_date"):
        if col in frame:
            frame[col] = pd.to_datetime(frame[col], errors="raise").dt.date
    if frame.duplicated(list(KEYS[table])).any():
        raise ValueError(f"{table} 主键重复")
    if target and "trade_date" in frame and (frame.trade_date != target).any():
        raise ValueError(f"{table} 日期与请求不一致")
    numeric = frame.select_dtypes(include="number")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError(f"{table} 含非有限数值")
    if table == "daily":
        if frame[["open", "high", "low", "close", "pre_close", "amount", "vol"]].isna().any().any():
            raise ValueError("daily 关键字段为空")
        if (frame[["open", "high", "low", "close", "pre_close"]] <= 0).any().any():
            raise ValueError("daily 价格非正")
        if (frame.high < frame[["open", "close", "low"]].max(axis=1)).any() or (
            frame.low > frame[["open", "close", "high"]].min(axis=1)).any():
            raise ValueError("daily 高低价关系错误")
        if (frame[["amount", "vol"]] < 0).any().any():
            raise ValueError("daily 成交量额为负")
    if table == "adj_factor" and (frame.adj_factor.isna().any() or (frame.adj_factor <= 0).any()):
        raise ValueError("adj_factor 无效")
    return frame


def scales_from_overlap(history, online, known_codes):
    merged = history.merge(online, on=["ts_code", "trade_date"], suffixes=("_history", "_online"))
    mapping = []
    for code, group in merged.groupby("ts_code", observed=True):
        ratios = group.adj_factor_history / group.adj_factor_online
        scale = float(ratios.median())
        spread = float((ratios / scale - 1).abs().max())
        if len(group) < 2 or not np.isfinite(scale) or scale <= 0 or spread > 5e-4:
            raise ValueError(f"{code} 复权接缝证据不足或尺度不稳定")
        mapping.append({"ts_code": code, "scale": scale, "overlap_days": len(group),
                        "relative_spread": spread})
    seen = {r["ts_code"] for r in mapping}
    unknown = set(online.ts_code) - seen - set(known_codes)
    mapping.extend({"ts_code": c, "scale": 1.0, "overlap_days": 0, "relative_spread": 0.0}
                   for c in sorted(unknown))
    return pd.DataFrame(mapping, columns=["ts_code", "scale", "overlap_days", "relative_spread"])


def upsert(con, table, frame):
    con.register("_incoming", frame)
    columns = ",".join(f'"{c}"' for c in frame.columns)
    con.execute(f"INSERT OR REPLACE INTO {table} ({columns}) SELECT {columns} FROM _incoming")
    con.unregister("_incoming")


class Source:
    """HTTPS endpoint matches the pinned transport in rebound.data.tushare_client."""
    def __init__(self, credential):
        self.credential = credential

    def call(self, endpoint, fields, **params):
        payload = json.dumps({"api_name": endpoint, "token": self.credential,
                              "params": params, "fields": ",".join(fields)}).encode("utf-8")
        req = urllib.request.Request(f"https://api.tushare.pro/dataapi/{endpoint}", data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=35) as response:
                body = json.load(response)
        except Exception:
            raise RuntimeError(f"{endpoint} 网络请求失败，请检查连接后重试") from None
        if body.get("code") != 0:
            # Never serialize request bodies, credentials or vendor exception strings.
            raise RuntimeError(f"{endpoint} 请求被拒绝（代码 {body.get('code')}），请检查接口权限或频率")
        data = body.get("data") or {}
        return pd.DataFrame(data.get("items", []), columns=data.get("fields", []))


def sync(end, max_days=1):
    initialize()
    credential = token()
    if not credential:
        write_status({**get_status(), "status": "credential_required",
                      "detail": "本机未配置 TUSHARE_TOKEN，尚未请求在线数据"})
        return get_status()
    source = Source(credential)
    schemas = {t: [c["name"] for c in profile()["tables"][t]["columns"]] for t in TABLES}
    historical_end = date.fromisoformat(profile()["tables"]["daily"]["date_max"])
    if end > date.today() or end <= historical_end:
        raise ValueError("增量结束日期应晚于历史截止且不晚于今天")
    write_status({**get_status(), "status": "running", "detail": "正在读取交易日历"})
    with duckdb.connect(str(DATABASE)) as con:
        con.execute(f"ATTACH '{BASE.as_posix()}' AS history (READ_ONLY)")
        try:
            cal = validated(source.call("trade_cal", schemas["trade_cal"], exchange="SSE",
                            start_date=(historical_end - timedelta(days=30)).strftime("%Y%m%d"),
                            end_date=end.strftime("%Y%m%d")), "trade_cal", schemas["trade_cal"])
            upsert(con, "trade_cal", cal)
            overlap_dates = sorted(cal.loc[(cal.is_open == 1) & (cal.cal_date <= historical_end), "cal_date"])[-3:]
            if len(overlap_dates) < 2:
                raise ValueError("交易日历不足以建立复权重叠校验")
            pieces = [validated(source.call("adj_factor", schemas["adj_factor"], trade_date=d.strftime("%Y%m%d")),
                                "adj_factor", schemas["adj_factor"], d) for d in overlap_dates]
            online = pd.concat(pieces, ignore_index=True)
            history = con.execute("SELECT * FROM history.adj_factor WHERE trade_date>=? AND trade_date<=?",
                                  [min(overlap_dates), historical_end]).fetchdf()
            history.trade_date = pd.to_datetime(history.trade_date).dt.date
            known = con.execute("SELECT DISTINCT ts_code FROM history.adj_factor").fetchnumpy()["ts_code"]
            mapping = scales_from_overlap(history, online, known)
            upsert(con, "factor_scale", mapping)
            upsert(con, "raw_adj_factor", online)
            for listing in ("L", "D", "P"):
                basic = source.call("stock_basic", schemas["stock_basic"], exchange="", list_status=listing)
                if not basic.empty:
                    upsert(con, "stock_basic", validated(basic, "stock_basic", schemas["stock_basic"]))
            done = {r[0] for r in con.execute("SELECT trade_date FROM committed_days").fetchall()}
            repair = {r[0] for r in con.execute("SELECT DISTINCT trade_date FROM sync_log WHERE status='unavailable'").fetchall()}
            days = sorted(d for d in cal.loc[cal.is_open == 1, "cal_date"]
                          if historical_end < d <= end and (d not in done or d in repair))[:max_days]
            for day in days:
                write_status({**get_status(), "status": "running", "detail": f"正在请求 {day}"})
                daily = validated(source.call("daily", schemas["daily"], trade_date=day.strftime("%Y%m%d")),
                                  "daily", schemas["daily"], day)
                previous = con.execute("SELECT count(*) FROM history.daily WHERE trade_date=?", [historical_end]).fetchone()[0]
                if len(daily) < previous * .8:
                    raise ValueError("日行情覆盖显著不足，拒绝发布；需核查接口截断或尚未定稿")
                raw = validated(source.call("adj_factor", schemas["adj_factor"], trade_date=day.strftime("%Y%m%d")),
                                "adj_factor", schemas["adj_factor"], day)
                joined = raw.merge(mapping, on="ts_code", how="left")
                new_codes = joined.ts_code[~joined.ts_code.isin(known) & joined.scale.isna()]
                joined.loc[joined.ts_code.isin(new_codes), "scale"] = 1.0
                if joined.loc[joined.ts_code.isin(daily.ts_code), "scale"].isna().any():
                    raise ValueError("部分历史股票缺少重叠复权证据，请扩大接缝范围后再发布")
                if not set(daily.ts_code) <= set(raw.ts_code):
                    raise ValueError("日行情缺少对应复权因子")
                normalized = joined[["ts_code", "trade_date", "adj_factor"]].copy()
                normalized["adj_factor"] *= joined.scale
                normalized = normalized.dropna(subset=["adj_factor"])
                con.execute("BEGIN")
                try:
                    upsert(con, "daily", daily)
                    upsert(con, "raw_adj_factor", raw)
                    upsert(con, "adj_factor", normalized)
                    con.execute("INSERT OR REPLACE INTO committed_days VALUES (?)", [day])
                    for t, f in (("daily", daily), ("adj_factor", normalized)):
                        con.execute("INSERT OR REPLACE INTO sync_log VALUES (?,?,'completed',?,'',current_timestamp)",
                                    [day, t, len(f)])
                    con.execute("COMMIT")
                    write_status({**get_status(), "status": "running",
                                  "last_date": str(con.execute("SELECT max(trade_date) FROM committed_days").fetchone()[0]),
                                  "detail": f"{day}日行情与复权因子已提交，补充其他接口"})
                except Exception:
                    con.execute("ROLLBACK")
                    raise
                for table in ("daily_basic", "moneyflow", "daily_limit", "index_daily"):
                    try:
                        if table == "index_daily":
                            parts = [source.call(table, schemas[table], ts_code=c, trade_date=day.strftime("%Y%m%d")) for c in INDEXES]
                            result = pd.concat(parts, ignore_index=True)
                        else:
                            endpoint = "stk_limit" if table == "daily_limit" else table
                            result = source.call(endpoint, schemas[table], trade_date=day.strftime("%Y%m%d"))
                        result = validated(result, table, schemas[table], day)
                        if table == "index_daily" and set(result.ts_code) != set(INDEXES):
                            raise ValueError("四个指数未完整返回")
                        if table in ("daily_basic", "daily_limit") and len(result) < len(daily) * .8:
                            raise ValueError(f"{table} 覆盖显著不足，未作为完整日快照发布")
                        upsert(con, table, result)
                        con.execute("INSERT OR REPLACE INTO sync_log VALUES (?,?,'completed',?,'',current_timestamp)",
                                    [day, table, len(result)])
                    except Exception as exc:
                        con.execute("INSERT OR REPLACE INTO sync_log VALUES (?,?,'unavailable',0,?,current_timestamp)",
                                    [day, table, str(exc)[:160]])
            last = con.execute("SELECT max(trade_date) FROM committed_days").fetchone()[0]
            table_status = {}
            for table in TABLES:
                dt = "cal_date" if table == "trade_cal" else "list_date" if table == "stock_basic" else "trade_date"
                count, latest = con.execute(f"SELECT count(*),max({dt}) FROM {table}").fetchone()
                table_status[table] = {"rows": count, "last_date": str(latest) if latest else None}
            unavailable = con.execute("SELECT count(*) FROM sync_log WHERE status='unavailable'").fetchone()[0]
            write_status({"status": "partial" if unavailable else "completed", "last_date": str(last) if last else None,
                          "tables": table_status, "detail": "逐表缺口见同步日志；扩展数据仅用于行情，研究数据版本保持冻结"})
        except Exception as exc:
            write_status({**get_status(), "status": "failed", "detail": str(exc)[:180]})
    publish()
    return get_status()


def attach_views(con, base=BASE, incremental=READER):
    con.execute(f"ATTACH '{base.as_posix()}' AS history (READ_ONLY)")
    if incremental.exists():
        con.execute(f"ATTACH '{incremental.as_posix()}' AS extra (READ_ONLY)")
    base_end = profile()["tables"]["daily"]["date_max"]
    names = [r[0] for r in con.execute("SHOW TABLES FROM history").fetchall()]
    for table in names:
        if table in TABLES and incremental.exists():
            if table == "stock_basic":
                sql = f"SELECT * FROM extra.{table} UNION ALL SELECT h.* FROM history.{table} h WHERE NOT EXISTS (SELECT 1 FROM extra.{table} e WHERE e.ts_code=h.ts_code)"
            elif table == "trade_cal":
                sql = f"SELECT * FROM history.{table} UNION ALL SELECT e.* FROM extra.{table} e WHERE NOT EXISTS (SELECT 1 FROM history.{table} h WHERE h.exchange=e.exchange AND h.cal_date=e.cal_date)"
            else:
                sql = f"SELECT * FROM history.{table} UNION ALL SELECT e.* FROM extra.{table} e WHERE e.trade_date>DATE '{base_end}' AND e.trade_date IN (SELECT trade_date FROM extra.committed_days) AND NOT EXISTS (SELECT 1 FROM history.{table} h WHERE h.ts_code=e.ts_code AND h.trade_date=e.trade_date)"
            con.execute(f"CREATE TEMP VIEW {table} AS {sql}")
        else:
            con.execute(f"CREATE TEMP VIEW {table} AS SELECT * FROM history.{table}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--end", type=date.fromisoformat, default=date.today() - timedelta(days=1))
    parser.add_argument("--max-days", type=int, default=1)
    args = parser.parse_args()
    if args.init:
        initialize()
        write_status({"status": "ready" if token() else "credential_required", "last_date": None, "tables": {},
                      "detail": "独立增量库已初始化，尚未请求在线数据"})
        result = get_status()
    else:
        if not 1 <= args.max_days <= 30:
            parser.error("--max-days 必须为 1–30")
        result = sync(args.end, args.max_days)
    print(json.dumps(result, ensure_ascii=False, indent=2))
