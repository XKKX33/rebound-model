"""行业特征（补充登记七的 V1）：给准入池（剔除 < 2 元，和训练标签同一批行、同一顺序）每行补 9 列。

- ind_r5 / ind_r20 / ind_r60：所在行业当天的平均近 5 / 20 / 60 日涨幅；
- rel_r5 / rel_r20 / rel_r60：个股减行业平均；
- ind_breadth：行业里站上 20 日线的比例；
- ind_heat：行业平均近 20 日涨停次数；
- ind_rank20：行业近 20 日涨幅在当天所有行业里的百分位（0 最弱，1 最强）。

行业用 stock_basic 现在的行业（和正式策略的行业上限同一口径，有轻微前视）。行业平均的范围：当天全部主板、按当时的
名称不是 ST、至少 60 根日线、收盘 ≥ 2 元，不限成交额和 70 元（池子外的龙头也算进行业热度）。当天不足 5 只的行业记为缺失。

先把原版的面板、标签硬链接到变体目录（DECADE_DATA，不能是本目录），再写 ind_features.parquet（di, ji + 9 列）。
用法：DECADE_DATA=ind python -X utf8 ind_features.py [--force]
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import panel10
from common import DATA, DB, axes, row_index
from train_cadence import panel_key

HERE = Path(__file__).resolve().parent
SRC = HERE / "panel10.parquet"
OUT = DATA / "ind_features.parquet"
COLS = ("ind_r5", "ind_r20", "ind_r60", "rel_r5", "rel_r20", "rel_r60", "ind_breadth", "ind_heat", "ind_rank20")
MIN_NAMES = 5
SETTINGS = panel10.SETTINGS + "; SET memory_limit='1500MB'; SET threads=4"


def link(name: str) -> None:
    """变体目录里放原版文件的硬链接（同一个盘不占地方）；链不上就复制。已经有了就不动。"""
    src, dst = HERE / name, DATA / name
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def query(src: Path = SRC) -> str:
    """面板每行接上当时的名称和现在的行业 → 行业 × 日的平均（排除 ST、日线不足 60、< 2 元，不足 5 只记缺失）→ 接回准入池的行。
    db 是挂上的十年合成库（要 namechange、stock_basic）。"""
    return f"""
        WITH names AS (
            SELECT ts_code, name, CAST(start_date AS TIMESTAMP) AS start_ts FROM (
                SELECT *, row_number() OVER (PARTITION BY ts_code, start_date ORDER BY ann_date DESC NULLS LAST, name) AS k
                FROM db.namechange) WHERE k = 1
        ), p AS (
            SELECT ts_code, trade_date, bar_no, below2, eligible, r5, r20, r60, dev_ma20, limit_up20
            FROM read_parquet('{src.as_posix()}')
        ), r AS (
            SELECT p.*, n.name AS name_then FROM p ASOF LEFT JOIN names n ON p.ts_code = n.ts_code AND p.trade_date >= n.start_ts
        ), base AS (
            SELECT r.*, b.industry AS g, coalesce(r.name_then, b.name, '') AS name_at
            FROM r LEFT JOIN db.stock_basic b ON r.ts_code = b.ts_code
        ), agg AS (
            SELECT trade_date, g, avg(r5) AS ind_r5, avg(r20) AS ind_r20, avg(r60) AS ind_r60,
                   avg(CASE WHEN dev_ma20 > 0 THEN 1.0 ELSE 0.0 END) AS ind_breadth, avg(limit_up20) AS ind_heat
            FROM base
            WHERE g IS NOT NULL AND g <> '' AND bar_no >= 60 AND NOT below2 AND r60 IS NOT NULL AND r20 IS NOT NULL
              AND r5 IS NOT NULL AND NOT regexp_matches(name_at, '{panel10.ST}')
            GROUP BY trade_date, g HAVING count(*) >= {MIN_NAMES}
        ), ranked AS (
            SELECT *, percent_rank() OVER (PARTITION BY trade_date ORDER BY ind_r20) AS ind_rank20 FROM agg
        )
        SELECT base.ts_code, base.trade_date,
               CAST(a.ind_r5 AS FLOAT) AS ind_r5, CAST(a.ind_r20 AS FLOAT) AS ind_r20, CAST(a.ind_r60 AS FLOAT) AS ind_r60,
               CAST(base.r5 - a.ind_r5 AS FLOAT) AS rel_r5, CAST(base.r20 - a.ind_r20 AS FLOAT) AS rel_r20,
               CAST(base.r60 - a.ind_r60 AS FLOAT) AS rel_r60, CAST(a.ind_breadth AS FLOAT) AS ind_breadth,
               CAST(a.ind_heat AS FLOAT) AS ind_heat, CAST(a.ind_rank20 AS FLOAT) AS ind_rank20
        FROM base LEFT JOIN ranked a ON base.trade_date = a.trade_date AND base.g = a.g
        WHERE base.eligible AND NOT base.below2
        ORDER BY base.ts_code, base.trade_date"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="已经有 ind_features.parquet 也重算")
    args = ap.parse_args()
    if DATA.resolve() == HERE:
        raise SystemExit("DECADE_DATA 要指到变体目录（例如 ind），不能写进原版目录")
    DATA.mkdir(parents=True, exist_ok=True)
    for name in ("panel10.parquet", "panel10.meta.json", "axes.json"):
        link(name)
    labels = f"labels10_o1o5_{panel_key()}.parquet"
    link(labels)
    if OUT.is_file() and not args.force:
        print(f"{OUT.name} 已经有了，跳过（--force 重算）", flush=True)
        return
    start = time.time()
    con = duckdb.connect()
    try:
        con.execute(SETTINGS)
        con.execute(f"ATTACH '{DB.as_posix()}' AS db (READ_ONLY)")
        t = con.execute(query()).fetch_arrow_table()
    finally:
        con.close()
    codes, dates = axes()
    di, ji = row_index(t.column("ts_code").to_pandas().astype("category"), t.column("trade_date").to_pandas(), codes, dates)
    lab = pq.read_table(DATA / labels, columns=["di", "ji"])
    if lab.num_rows != len(di) or (lab.column("di").to_numpy() != di).any() or (lab.column("ji").to_numpy() != ji).any():
        raise SystemExit("行业特征和训练标签的行对不上")
    out = pa.table({"di": di.astype(np.int32), "ji": ji.astype(np.int32), **{c: t.column(c) for c in COLS}})
    tmp = OUT.with_name(OUT.stem + ".tmp.parquet")
    pq.write_table(out, tmp)
    os.replace(tmp, OUT)
    missing = {c: round(t.column(c).null_count / t.num_rows * 100, 2) for c in COLS}
    print(f"{OUT.name}: {t.num_rows} 行，缺失 % {missing} ({time.time() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
