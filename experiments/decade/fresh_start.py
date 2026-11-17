"""换起点重新起步：同一条规则、同样的本金，分别从 2015、2017、…、2025 年第一个开市日起建仓，一直模拟到最后。
看小本金的结果有多依赖开头那一两年（账户先涨大了，5 元最低佣金的占比就小了）。另算免五（没有最低佣金）作对照。
2015 年起步就是原来的全期模拟（核对用）。输出 fresh_start.json。
用法：python -X utf8 fresh_start.py
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from common import DATA, load_market10, signal_range
from engine import candidates, pool_daily_return
from rule_grid import parse, specs
from rule_search import cost_model
from sim import YEAR, simulate
from train_cadence import no_power_throttling

CASES = [(20000, "M", "DB-K10-b12-o1o5"), (20000, "M", "DB-K10-b4-o1o5"), (20000, "M", "DB-K8-b6-o1o5"),
         (200000, "M", "DB-K30-b8-o1o5")]
STARTS = (2015, 2017, 2019, 2021, 2023, 2025)


def window(nav: np.ndarray, pool: np.ndarray, d0: int, d_end: int, dates: pd.DatetimeIndex) -> dict:
    """d0 收盘起步到 d_end：头一年收益、年化、最大回撤、翻了几倍、准入池同期年化。"""
    seg = nav[d0:d_end + 1] / nav[d0]
    year_end = min(int(dates.searchsorted(pd.Timestamp(dates[d0].year + 1, 1, 1))) - 1, d_end)
    years = (d_end - d0) / YEAR
    pool_nav = np.cumprod(1 + np.nan_to_num(pool[d0 + 1:d_end + 1]))
    return {"first_year": round(float(nav[year_end] / nav[d0] - 1) * 100, 1),
            "cagr": round((float(seg[-1]) ** (1 / years) - 1) * 100, 1),
            "max_dd": round(float((seg / np.maximum.accumulate(seg) - 1).min()) * 100, 1),
            "final_x": round(float(seg[-1]), 2), "pool_cagr": round((float(pool_nav[-1]) ** (1 / years) - 1) * 100, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", nargs="*", default=[], help="本金:频率:规则名，不给就用 CASES")
    ap.add_argument("--fees", nargs="+", default=["min5", "free5"])
    ap.add_argument("--starts", nargs="+", type=int, default=list(STARTS))
    ap.add_argument("--out", default="fresh_start.json")
    ap.add_argument("--all", type=int, default=0, help="给本金时：每月重训的全部 150 条规则都算")
    args = ap.parse_args()
    cases = [(int(c.split(":")[0]), *c.split(":")[1:]) for c in args.case] or CASES
    if args.all:
        cases = [(args.all, "M", s.name) for s in specs()]
    start = time.time()
    print(f"power throttling off: {no_power_throttling()}", flush=True)
    m, low, _ = load_market10(raw=True)
    dates = m.dates
    d_first, d_last = signal_range(dates)
    mf = replace(m, eligible=m.eligible & ~low)
    pool = pool_daily_return(mf)
    cache: dict[str, tuple] = {}

    def picks(cadence: str) -> tuple:
        """这个重训频率的候选和排名（第一次用到时才读打分）。"""
        if cadence not in cache:
            scores = pd.read_parquet(DATA / f"scores10_{cadence}.parquet")
            cache[cadence] = candidates(m, scores[~low[scores.di.to_numpy(), scores.ji.to_numpy()]])
        return cache[cadence]

    out = {}
    for capital, cadence, name in cases:
        cands, rank = picks(cadence)
        for fee in args.fees:
            cost = cost_model(capital, "real")
            cost = replace(cost, min_commission=0.0) if fee == "free5" else cost
            for y in args.starts:
                d0 = max(int(dates.searchsorted(pd.Timestamp(y, 1, 1))), d_first)
                res = simulate(mf, cands, rank, parse(name).rule, cost, d0, d_last)
                out[f"{capital}|{cadence}|{name}|{fee}|{y}"] = r = window(res.nav, pool, d0, d_last + 1, dates)
                print(capital, cadence, name, fee, y, r, f"({time.time() - start:.0f}s)", flush=True)
        if len(out) % 100 < len(args.starts) * len(args.fees):  # 全量跑时隔一阵存一次
            (DATA / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (DATA / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
