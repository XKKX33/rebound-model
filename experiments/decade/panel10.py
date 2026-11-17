"""十年面板（2013-01 ~ 2026-10）：主板日线，复权与未复权开收盘、涨跌停受阻、准入池、28 个量价特征。

只读十年库（common.DB，默认 data/research-2013-2026.duckdb，可用环境变量 DECADE_DB 指到别的库）。
特征的 SQL 和后处理与 p0p1 的 panel.py 同一套（直接调用它）。和 p0p1 的区别只有一处：准入池的 ST 按当时的名称判断。
输出 panel10.parquet、panel10.meta.json、axes.json。用法：python -X utf8 panel10.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))
sys.path.insert(0, str(HERE.parent / "shared"))

import panel as p0p1_panel  # noqa: E402
from stage_data import FEATURES, SQL  # noqa: E402

from common import DATA, DB  # noqa: E402

RAW, EXTRA, OUT = DATA / "panel10.raw.parquet", DATA / "panel10.extra.parquet", DATA / "panel10.parquet"
SETTINGS = p0p1_panel.SETTINGS + f"; SET temp_directory='{(DATA / 'duckdb-tmp').as_posix()}'"
ST = r"ST|\*"


def write_raw() -> None:
    con = duckdb.connect(str(DB), read_only=True)
    try:
        con.execute(SETTINGS)
        con.execute(f"COPY (SELECT * REPLACE (CAST(trade_date AS TIMESTAMP) AS trade_date) FROM ({SQL}) "
                    f"ORDER BY ts_code, trade_date) TO '{RAW.as_posix()}' (FORMAT PARQUET)")
    finally:
        con.close()


def write_extra() -> None:
    """MACD 比值、截面排名、涨跌停受阻：直接用 p0p1 的 extra_columns（它的准入池按现在的名称，这里不用）。"""
    p0p1_panel.RAW, p0p1_panel.DB = RAW, DB
    p0p1_panel.extra_columns().drop(columns=["eligible"]).to_parquet(EXTRA, index=False)


def write_final() -> None:
    """合并成面板；准入池的 ST 按当时的名称（namechange 里生效日不晚于当天的最后一个名字），没有记录才用现在的名称。"""
    feats = ", ".join(f"CAST(CASE WHEN isfinite(r.{f}) THEN r.{f} END AS FLOAT) AS {f}"
                      if f not in ("dif_ratio", "hist_ratio", "rank_r20", "rank_r60")
                      else f"CAST(CASE WHEN isfinite(x.{f}) THEN x.{f} END AS FLOAT) AS {f}" for f in FEATURES)
    con = duckdb.connect()
    try:
        con.execute(SETTINGS)
        con.execute(f"ATTACH '{DB.as_posix()}' AS db (READ_ONLY)")
        con.execute(f"""COPY (
            WITH names AS (
                SELECT ts_code, name, CAST(start_date AS TIMESTAMP) AS start_ts FROM (
                    SELECT *, row_number() OVER (PARTITION BY ts_code, start_date ORDER BY ann_date DESC NULLS LAST, name) AS k
                    FROM db.namechange) WHERE k = 1
            ), r AS (
                SELECT r.*, n.name AS name_then FROM read_parquet('{RAW.as_posix()}') r
                ASOF LEFT JOIN names n ON r.ts_code = n.ts_code AND r.trade_date >= n.start_ts
            )
            SELECT r.ts_code, CAST(r.trade_date AS TIMESTAMP) AS trade_date, CAST(r.bar_no AS INTEGER) AS bar_no,
                   r.o, r.c, r.open AS raw_o, r.close AS raw_c, r.amount,
                   x.up_open, x.up_close, x.dn_open, x.dn_close,
                   coalesce(r.close < 2.0, false) AS below2,
                   coalesce(NOT regexp_matches(coalesce(r.name_then, b.name), '{ST}'), false)
                     AND r.close < 70 AND r.amount >= 80000 AND r.bar_no >= 149 AS eligible,
                   coalesce(NOT regexp_matches(b.name, '{ST}'), false)
                     AND r.close < 70 AND r.amount >= 80000 AND r.bar_no >= 149 AS eligible_now,
                   {feats}
            FROM r JOIN read_parquet('{EXTRA.as_posix()}') x
              ON r.ts_code = x.ts_code AND r.trade_date = CAST(x.trade_date AS TIMESTAMP)
            LEFT JOIN db.stock_basic b ON r.ts_code = b.ts_code
            ORDER BY r.ts_code, r.trade_date
        ) TO '{OUT.as_posix()}' (FORMAT PARQUET)""")
    finally:
        con.close()


def write_meta(seconds: float) -> dict:
    con = duckdb.connect()
    try:
        con.execute(SETTINGS)
        con.execute(f"ATTACH '{DB.as_posix()}' AS db (READ_ONLY)")
        src = f"read_parquet('{OUT.as_posix()}')"
        stats = con.execute(f"""SELECT count(*), count(DISTINCT ts_code), count(DISTINCT trade_date), min(trade_date),
            max(trade_date), sum(CAST(eligible AS INT)), sum(CAST(eligible_now AS INT)),
            sum(CAST(eligible AND NOT eligible_now AS INT)), sum(CAST(eligible_now AND NOT eligible AS INT))
            FROM {src}""").fetchone()
        per_year = dict(con.execute(f"""SELECT year(trade_date), round(avg(n)) FROM (
            SELECT trade_date, sum(CAST(eligible AND NOT below2 AS INT)) AS n FROM {src} GROUP BY 1) GROUP BY 1 ORDER BY 1
            """).fetchall())
        codes = [c for (c,) in con.execute(f"SELECT DISTINCT ts_code FROM {src} ORDER BY 1").fetchall()]
        days = [str(d.date()) for (d,) in con.execute(f"SELECT DISTINCT trade_date FROM {src} ORDER BY 1").fetchall()]
        cal = [str(d) for (d,) in con.execute(f"""SELECT cal_date FROM db.trade_cal WHERE exchange='SSE' AND is_open=1
            AND cal_date BETWEEN DATE '{days[0]}' AND DATE '{days[-1]}' ORDER BY 1""").fetchall()]
        seam = json.loads(con.execute("SELECT value_json FROM db.merge_meta WHERE key = 'seam_checks'").fetchone()[0])
    finally:
        con.close()
    (DATA / "axes.json").write_text(json.dumps({"codes": codes, "dates": days}), encoding="utf-8")
    meta = {"rows": stats[0], "codes": stats[1], "dates": stats[2], "first": str(stats[3]), "last": str(stats[4]),
            "eligible_rows": int(stats[5]), "eligible_now_rows": int(stats[6]),
            "eligible_only_by_name_then": int(stats[7]), "eligible_only_by_name_now": int(stats[8]),
            "pool_per_day_by_year": {str(k): v for k, v in per_year.items()}, "calendar_matches": cal == days,
            "source": str(DB), "seam": seam, "seconds": round(seconds)}
    (DATA / "panel10.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return meta


def main() -> None:
    if not DB.is_file():
        raise SystemExit(f"缺十年库 {DB}：先用 rebound sync 建库，再设 DECADE_DB 指向它")
    start = time.time()
    for step in (write_raw, write_extra, write_final):
        step()
        print(f"{step.__name__} done ({time.time() - start:.0f}s)", flush=True)
    meta = write_meta(time.time() - start)
    RAW.unlink()
    EXTRA.unlink()
    print(meta, flush=True)
    if not meta["calendar_matches"]:
        raise SystemExit("面板的日期和 SSE 开市日不一致")


if __name__ == "__main__":
    main()
