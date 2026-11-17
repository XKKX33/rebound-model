"""不限价版的十年面板：准入池去掉"未复权收盘 < 70 元"这一条，其余不变（ST 按当时的名称、成交额 ≥ 0.8 亿、
至少 150 根日线、主板；< 2 元照旧在训练和模拟里剔除）。

从现成的 panel10.parquet 重算 eligible / eligible_now（它有未复权收盘、成交额、日线数），不重跑特征。
先用同一条公式加回 70 元上限算一遍，必须和面板里存的 eligible 一模一样，再写不限价版。
输出到 DECADE_DATA（不限价版的目录）：panel10.parquet、panel10.meta.json、axes.json。
用法（run_nocap.py 会设好环境变量）：DECADE_DATA=nocap python -X utf8 nocap_panel.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb

from common import DATA, DB
import panel10

HERE = Path(__file__).resolve().parent
SRC = HERE / "panel10.parquet"  # 限 70 元的原版面板
SETTINGS = panel10.SETTINGS + "; SET memory_limit='1GB'; SET threads=4"


def rule(name: str, cap: bool) -> str:
    """准入条件（和 panel10.write_final 同一套，缺值都算不准入）；cap=False 去掉 70 元上限。"""
    price = "r.raw_c < 70" if cap else "isfinite(r.raw_c)"
    return (f"(coalesce(NOT regexp_matches({name}, '{panel10.ST}'), false) AND coalesce({price}, false)"
            f" AND coalesce(r.amount >= 80000, false) AND coalesce(r.bar_no >= 149, false))")


def flagged() -> str:
    """面板每行接上当时的名称（namechange 里生效日不晚于当天的最后一个名字），以及现在的名称。"""
    return f"""
        WITH names AS (
            SELECT ts_code, name, CAST(start_date AS TIMESTAMP) AS start_ts FROM (
                SELECT *, row_number() OVER (PARTITION BY ts_code, start_date ORDER BY ann_date DESC NULLS LAST, name) AS k
                FROM db.namechange) WHERE k = 1
        ), r AS (
            SELECT r.*, n.name AS name_then FROM read_parquet('{SRC.as_posix()}') r
            ASOF LEFT JOIN names n ON r.ts_code = n.ts_code AND r.trade_date >= n.start_ts
        )
        SELECT r.*, coalesce(r.name_then, b.name) AS name_at, b.name AS name_now FROM r
        LEFT JOIN db.stock_basic b ON r.ts_code = b.ts_code"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("nocap", "stnow"), default="nocap",
                    help="nocap：去掉 70 元上限；stnow：上限照旧，ST 改按现在的名称判断（p0p1 的口径，有前视）")
    mode = ap.parse_args().mode
    if DATA.resolve() == HERE:
        raise SystemExit("DECADE_DATA 要指到另一个目录（例如 nocap、stnow），不能覆盖原版面板")
    DATA.mkdir(parents=True, exist_ok=True)
    start = time.time()
    con = duckdb.connect()
    try:
        con.execute(SETTINGS)
        con.execute(f"ATTACH '{DB.as_posix()}' AS db (READ_ONLY)")
        check = con.execute(f"""SELECT
            sum(CAST({rule('r.name_at', True)} <> coalesce(r.eligible, false) AS INT)),
            sum(CAST({rule('r.name_now', True)} <> coalesce(r.eligible_now, false) AS INT)),
            sum(CAST({rule('r.name_at', True)} AND NOT {rule('r.name_at', False)} AS INT)),
            sum(CAST({rule('r.name_at', True)} AS INT)), sum(CAST({rule('r.name_at', False)} AS INT))
            FROM ({flagged()}) r""").fetchone()
        cross = {"capped_vs_stored_diff": int(check[0]), "capped_now_vs_stored_diff": int(check[1]),
                 "capped_not_in_nocap": int(check[2]), "capped_rows": int(check[3]), "nocap_rows": int(check[4])}
        print(f"核对 {cross} ({time.time() - start:.0f}s)", flush=True)
        if cross["capped_vs_stored_diff"] or cross["capped_now_vs_stored_diff"] or cross["capped_not_in_nocap"]:
            raise SystemExit("加回 70 元上限后和面板里存的准入池对不上，停")
        if mode == "nocap":
            cols = f"{rule('r.name_at', False)} AS eligible, {rule('r.name_now', False)} AS eligible_now"
        else:
            cols = f"{rule('r.name_now', True)} AS eligible, {rule('r.name_now', True)} AS eligible_now"
        con.execute(f"""COPY (
            SELECT * EXCLUDE (name_then, name_at, name_now, eligible, eligible_now), {cols}
            FROM ({flagged()}) r ORDER BY r.ts_code, r.trade_date
        ) TO '{panel10.OUT.as_posix()}' (FORMAT PARQUET)""")
    finally:
        con.close()
    meta = panel10.write_meta(time.time() - start)
    meta.update({"mode": mode, "price_max": None if mode == "nocap" else 70, "st_by": "name_at" if mode == "nocap" else "name_now",
                 "derived_from": str(SRC), "cross_check": cross})
    (DATA / "panel10.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print({k: meta[k] for k in ("rows", "eligible_rows", "pool_per_day_by_year", "calendar_matches")}, flush=True)
    if not meta["calendar_matches"]:
        raise SystemExit("面板的日期和 SSE 开市日不一致")


if __name__ == "__main__":
    main()
