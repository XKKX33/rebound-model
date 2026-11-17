"""这条规则每年实际拿了什么：按持仓天数算的行业占比和前几大个股（只读，不改结果）。

持仓从成交记录还原：某天在手的 = entry_day <= d < exit_day 的那些笔（2021 的仓位后来都平掉了，所以是完整的）。
用法：python -X utf8 holdings_check.py [--years 2021 2026]
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import duckdb
import numpy as np
import pandas as pd

from common import DATA, DB, load_market10, signal_range
from decade_sim import scoring_days
from engine import candidates
from rule_grid import parse
from rule_search import cost_model
from sim import simulate


def names_and_industry() -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    try:
        return con.execute("SELECT ts_code, name, industry FROM stock_basic").df()
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=200_000.0)
    ap.add_argument("--cadence", default="M")
    ap.add_argument("--name", default="DB-K30-b8-o1o5")
    ap.add_argument("--years", nargs="+", type=int, default=[2021, 2026])
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    m, low, _ = load_market10(raw=True)
    dates = m.dates
    d_first, d_last = signal_range(dates)
    mf = replace(m, eligible=m.eligible & ~low)
    scores = pd.read_parquet(DATA / f"scores10_{args.cadence}.parquet")
    cands, rank = candidates(m, scores[~low[scores.di.to_numpy(), scores.ji.to_numpy()]])
    del scores
    res = simulate(mf, cands, rank, parse(args.name).rule, cost_model(args.capital, "real"), d_first, d_last)
    t = res.trades.copy()
    t["exit"] = t.exit_day.where(t.exit_day >= 0, len(dates) - 1)
    info = names_and_industry().set_index("ts_code")
    codes = np.asarray(mf.codes)
    out = []
    for year in args.years:
        days = np.flatnonzero(dates.year == year)
        lo, hi = int(days[0]), int(days[-1])
        span = t[(t.entry_day <= hi) & (t.exit > lo)]
        rows = []
        for d in range(lo, hi + 1):
            live = span[(span.entry_day <= d) & (span.exit > d)]
            for j in live.j:
                rows.append((codes[int(j)], d))
        frame = pd.DataFrame(rows, columns=["ts_code", "day"])
        frame["industry"] = frame.ts_code.map(info.industry).fillna("未知")
        frame["name"] = frame.ts_code.map(info.name).fillna("")
        tot = len(frame)
        print(f"\n===== {year}：{tot / max(len(days), 1):.1f} 个仓位/天，共 {len(days)} 个交易日")
        ind = frame.industry.value_counts().head(args.top)
        print("行业（按仓位天数占比）：")
        for k, v in ind.items():
            print(f"  {k:8s} {v / tot * 100:5.1f}%")
        top = frame.groupby(["ts_code", "name", "industry"]).size().sort_values(ascending=False).head(args.top)
        print("个股（按持仓天数）：")
        for (code, name, industry), v in top.items():
            print(f"  {code} {name:6s} {industry:8s} {v:4d} 天")
        out.append(frame)
    return out


if __name__ == "__main__":
    main()
