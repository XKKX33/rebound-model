"""五年研究（p0p1）和十年研究在同一段留出期上为什么不一样：同一套模拟，换一个条件看一次。

  A  十年原版：ST 按当时的名称，训练从 2013 年起
  B  ST 按现在的名称（p0p1 的口径：今天是 ST 的票，过去每一天都不算）
  C  训练只用 2021-12 以后的数据（p0p1 能用的那么多）
  D  B + C，最接近 p0p1
  A_M  十年原版、每月重训（推荐规则用的频率），作参照
都是每年重训（A_M 除外）、20 万真实账户（现行印花税、最低 5 元），2024-01-02 起建仓，按段报账户总收益，
以及各自准入池（A/C 按当时的名称、B/D 按现在的名称）同段的收益。
输出 holdout_gap.json。用法：python -X utf8 holdout_gap.py
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import DATA, PANEL, axes, load_market10, row_index, signal_range
from engine import candidates, pool_daily_return
from rule_grid import parse
from rule_search import cost_model
from sim import YEAR, simulate
from train_cadence import no_power_throttling

START = pd.Timestamp("2024-01-02")
SLICES = {"2024H2": ("2024-07-01", "2024-12-31"), "2025": ("2025-01-01", "2025-12-31"),
          "2026": ("2026-01-01", "2026-12-31"), "留出期": ("2024-07-01", "2026-12-31")}
RULES = ("FH-K6-next-o1o5", "FH-K2-next-o1o5", "DB-K3-b2-o1o5", "DB-K10-b4-o1o5", "DB-K30-b8-o1o5")
VARIANTS = {"A": ("main", "scores10_Y.parquet"), "B": ("stnow", "stnow/scores10_Y.parquet"),
            "C": ("main", "scores10_Y_from20211203.parquet"), "D": ("stnow", "stnow/scores10_Y_from20211203.parquet"),
            "A_M": ("main", "scores10_M.parquet")}


def slice_stats(ret: np.ndarray, dates: pd.DatetimeIndex, d_end: int) -> dict:
    """逐日收益（ret[t] = 第 t 天相对前一天）→ 各段总收益和年化（%）。"""
    out = {}
    for name, (a, b) in SLICES.items():
        s0, s1 = int(dates.searchsorted(pd.Timestamp(a))), min(int(dates.searchsorted(pd.Timestamp(b), "right")) - 1, d_end)
        r = ret[s0:s1 + 1]
        total = float(np.prod(1 + np.nan_to_num(r))) - 1
        out[name] = {"total": round(total * 100, 1), "cagr": round(((1 + total) ** (YEAR / len(r)) - 1) * 100, 1),
                     "days": int(len(r))}
    return out


def eligible_now(dates: pd.DatetimeIndex, codes: np.ndarray, low: np.ndarray) -> np.ndarray:
    """面板里的 eligible_now（ST 按现在的名称），去掉 < 2 元。"""
    t = pq.read_table(PANEL, columns=["ts_code", "trade_date", "eligible_now"], read_dictionary=["ts_code"]).to_pandas()
    di, ji = row_index(t.ts_code, t.trade_date, codes, dates)
    out = np.zeros((len(dates), len(codes)), dtype=bool)
    out[di, ji] = t.eligible_now.fillna(False).to_numpy(dtype=bool)
    return out & ~low


def main() -> None:
    print(f"power throttling off: {no_power_throttling()}", flush=True)
    m, low, _ = load_market10(raw=True)
    dates = m.dates
    _, d_last = signal_range(dates)
    d_first, d_end = int(dates.searchsorted(START)), d_last + 1
    mf = replace(m, eligible=m.eligible & ~low)
    pools = {"main": pool_daily_return(mf), "stnow": pool_daily_return(replace(m, eligible=eligible_now(dates, m.codes, low)))}
    out = {"pool": {k: slice_stats(v, dates, d_end) for k, v in pools.items()}, "runs": {}}
    cost = cost_model(200_000, "real")
    for variant, (pool_key, path) in VARIANTS.items():
        scores = pd.read_parquet(DATA / path)
        cands, rank = candidates(m, scores[~low[scores.di.to_numpy(), scores.ji.to_numpy()]])
        first_scored = str(dates[int(scores.di.min())].date())
        del scores
        for name in RULES:
            res = simulate(mf, cands, rank, parse(name).rule, cost, d_first, d_last)
            ret = np.full(len(dates), np.nan)
            ret[1:] = res.nav[1:] / res.nav[:-1] - 1
            out["runs"][f"{variant}|{name}"] = r = {"pool": pool_key, "first_scored": first_scored,
                                                    **slice_stats(ret, dates, d_end)}
            print(variant, name, {k: v["total"] for k, v in r.items() if isinstance(v, dict)}, flush=True)
        del cands, rank
    (DATA / "holdout_gap.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("pool", {k: {s: v[s]["total"] for s in SLICES} for k, v in out["pool"].items()}, flush=True)


if __name__ == "__main__":
    main()
