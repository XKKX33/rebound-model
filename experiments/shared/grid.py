"""P1 持有规则对比：3 个格子 × 9 组持有规则 × 3 档成本，按 PREREG.md 的 G1 规则挑选（补充登记二的口径）。

另跑：FH 的 skip 敏感性；cycle_wave 终选按 FH、K=6 跑同样三格作对照；FH-K6-skip 零成本时逐笔对逐笔引擎；
逐笔和组合两种口径的对账；时点匹配基准；停牌按最后收盘价计值的基准核对。
用法：python -X utf8 grid.py → p1_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import (CELLS, COSTS, DEV_END, candidates, dedup_top, load_market, matched_pool, pool_daily_return,
                    pool_daily_return_carry, pool_legs, pool_mean, register, trades)
from sim import WARMUP, YEAR, Rule, SimResult, _annual, _days, paired, simulate, summarize

HERE = Path(__file__).resolve().parent
KS, BUFFERS = (6, 10, 20), (2, 4)


def cycle_candidates(m) -> tuple[dict[int, np.ndarray], int, int]:
    """cycle_wave 每天过门槛的票按池内顺序；同时核对按这个顺序去重取前 6 是否就是它的终选。"""
    cands, mismatch, days = {}, 0, 0
    for path in sorted((HERE / "cycle-cache").glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if d["status"] != "ok":
            continue
        di = int(m.date_index([pd.Timestamp(d["as_of"])])[0])
        order = m.code_index([p["ts_code"] for p in d["passed"]])
        cands[di] = order[order >= 0]
        days += 1
        # 行业去重用的是研究库 stock_basic 的行业，和正式 dedup_and_top 同源
        rebuilt = [str(m.codes[j]) for j in dedup_top(cands[di], m.industry, k=6, cap=2)]
        mismatch += rebuilt != d["picks"]
    return cands, days, mismatch


def rules_for(cell) -> list[Rule]:
    return [*(Rule("FH", k, cell) for k in KS), *(Rule("DB", k, cell, buffer=b) for k in KS for b in BUFFERS)]


def pool_stats(m, pool: np.ndarray, d_first: int, d_last: int) -> dict:
    """准入池等权（不扣成本、不可交易）自己的绝对收益和回撤，作背景。"""
    nav = np.full(len(m.dates), np.nan)
    nav[d_first] = 1.0
    nav[d_first + 1:] = np.cumprod(1 + np.nan_to_num(pool[d_first + 1:]))
    zeros = np.zeros(len(m.dates))
    fake = SimResult(nav, zeros.astype(np.int32), zeros, zeros, pd.DataFrame(), 0, 0)
    return {p: {"cagr": v["cagr"], "max_dd": v["max_dd"]}
            for p, v in summarize(m, fake, pool, d_first, d_last).items() if p != "yearly_excess"}

def excess_by_period(m, res: SimResult, bench: np.ndarray, d_first: int, d_last: int) -> dict:
    """组合扣成本后的日收益减一个基准，各期年化及按月重抽区间（与 summarize 同一套计分日）。"""
    ret = np.full(len(m.dates), np.nan)
    ret[1:] = res.nav[1:] / res.nav[:-1] - 1
    return {p: _annual(pd.Series(ret[idx] - bench[idx], index=m.dates[idx]))
            for p, idx in _days(m, d_first, d_last).items() if len(idx)}


def consistency(m, cands, rank, cell, d_first, d_last) -> dict:
    """FH-K6-skip 零成本：模拟器的每一笔和逐笔引擎（carry）比。引擎最多顺延 20 天，模拟器一直拿到能卖，
    所以引擎算不出的那几笔单独报（不屏蔽），连同模拟器给它们记的收益。"""
    res = simulate(m, cands, rank, Rule("FH", 6, cell, fill="skip"), (0, 0), d_first, d_last)
    eng = trades(m, res.trades.signal_day.to_numpy(), res.trades.j.to_numpy(), cell, "carry")
    both = np.isfinite(eng["ret"])
    gross = res.trades.gross.to_numpy()
    gap = np.abs(gross[both] - eng["ret"][both])
    return {"trades": int(len(res.trades)), "engine_resolved": int(both.sum()),
            "max_abs_diff": float(gap.max()) if both.any() else None,
            "exit_day_mismatch": int((res.trades.exit_day.to_numpy()[both] != eng["exit_day"][both]).sum()),
            "engine_unresolved": int((~both).sum()),
            "sim_gross_on_unresolved_pct": [round(float(g) * 100, 1) for g in gross[~both]]}


def reconcile(m, cands, rank, cell, legs, d_first, d_last) -> dict:
    """FH-K6 零成本：组合口径（对每日调仓的准入池）和逐笔口径（对买入持有的准入池）为什么不一样。

    逐笔年化 = 每笔超额 × 242 / P。两者之差的大头应是"再平衡溢价"：同一段持有期里，
    每日调仓的准入池比买入持有的准入池多赚多少（A 股短期反转明显时为正，组合的基准因此更难跑赢）。
    """
    res = simulate(m, cands, rank, Rule("FH", 6, cell), (0, 0), d_first, d_last)
    t = res.trades[res.trades.signal_day >= d_first + WARMUP]
    days = np.arange(d_first, d_last + 1)
    bh = pool_mean(m, days, cell, "carry")
    per_trade = float(np.nanmean(t.gross.to_numpy() - bh[t.signal_day.to_numpy()]))
    reb = np.full(len(m.dates), np.nan)
    for d in days[(days >= d_first + WARMUP)]:
        e, x = d + 1, d + cell.x
        r = 1 + legs["intra"][e] if cell.entry == "open" else 1.0
        r *= np.prod(1 + legs["cc"][e + 1:x])
        r *= (1 + legs["cc"][x]) if cell.exit == "close" else (1 + legs["night"][x])
        reb[d] = r - 1 - bh[d]
    portfolio = summarize(m, res, pool_daily_return(m), d_first, d_last)["all"]["excess"]["ann"]
    return {"portfolio_zero_cost_ann": portfolio,
            "per_trade_ann": round(per_trade * YEAR / cell.period * 100, 2),
            "rebalance_premium_ann": round(float(np.nanmean(reb)) * YEAR / cell.period * 100, 2),
            "per_trade_mean_pct": round(per_trade * 100, 3), "trades": int(len(t))}


def passes_g1(s: dict, key: str = "excess") -> bool:
    """G1：开发期 mid 档年化超额的区间下限 > 0，且 high 档点估计 > 0。"""
    return s["mid"]["dev"][key]["ci"][0] > 0 and s["high"]["dev"][key]["ann"] > 0

def add_matched(m, res: SimResult, summary: dict, bench: np.ndarray, d_first: int, d_last: int) -> dict:
    """把时点匹配基准下的年化超额放进 summary 的各期（键名 matched）。"""
    for period, value in excess_by_period(m, res, bench, d_first, d_last).items():
        summary[period]["matched"] = value
    return summary


def main() -> None:
    m = load_market()
    pool, legs = pool_daily_return(m), pool_legs(m)
    d_first, d_last = m.signal_range
    cyc, cyc_days, cyc_mismatch = cycle_candidates(m)
    missing = [d for d in range(d_first, d_last + 1) if d not in cyc]
    if missing:
        raise SystemExit(f"cycle-cache 缺 {len(missing)} 天，先跑完 cycle_daily.py")
    idx = _days(m, d_first, d_last)
    carry = pool_daily_return_carry(m)
    cells = {c.name: c for c in CELLS}
    chosen = json.loads((HERE / "p1_cells.json").read_text(encoding="utf-8"))
    out = {"window": [str(m.dates[d_first].date()), str(m.dates[d_last].date())], "warmup_days": WARMUP,
           "cycle_rebuild": {"days": cyc_days, "mismatch_days": cyc_mismatch},
           "pool": pool_stats(m, pool, d_first, d_last),
           "pool_carry_minus_pool_ann": {p: round(float(np.nanmean(carry[i] - pool[i])) * YEAR * 100, 2)
                                         for p, i in idx.items()},
           "cells": {}}
    primary = {}
    for name in chosen:
        cell = cells[name]
        cands, rank = candidates(m, pd.read_parquet(HERE / f"scores_{name}.parquet"))
        cell_out = {"consistency": consistency(m, cands, rank, cell, d_first, d_last),
                    "reconcile": reconcile(m, cands, rank, cell, legs, d_first, d_last),
                    "rules": {}, "skip": {}, "cycle": {}}
        cycle_runs = {}
        bench_fh = matched_pool(legs, "FH", cell)
        for tier, cost in COSTS.items():
            cycle_runs[tier] = simulate(m, cyc, None, Rule("FH", 6, cell), cost, d_first, d_last)
            cell_out["cycle"][tier] = add_matched(m, cycle_runs[tier],
                                                  summarize(m, cycle_runs[tier], pool, d_first, d_last),
                                                  bench_fh, d_first, d_last)
        for rule in rules_for(cell):
            bench = matched_pool(legs, rule.kind, cell)
            row = {}
            for tier, cost in COSTS.items():
                res = simulate(m, cands, rank, rule, cost, d_first, d_last)
                row[tier] = add_matched(m, res, summarize(m, res, pool, d_first, d_last), bench, d_first, d_last)
                row[tier]["vs_cycle"] = paired(m, res, cycle_runs[tier], d_first, d_last)
                row[tier]["blocked_buys"], row[tier]["stuck_days"] = res.blocked_buys, res.stuck_days
                if rule.kind == "FH" and tier == "mid":
                    skip = simulate(m, cands, rank, Rule("FH", rule.k, cell, fill="skip"), cost, d_first, d_last)
                    cell_out["skip"][rule.name] = summarize(m, skip, pool, d_first, d_last)
            row["g1_pass"], row["g1_pass_matched"] = passes_g1(row), passes_g1(row, "matched")
            cell_out["rules"][rule.name] = primary[rule.name] = row
            register("P1-grid-v2", {"rule": rule.name, "costs": COSTS, "warmup": WARMUP},
                     {t: row[t]["dev"]["excess"] for t in COSTS} | {"g1_pass": row["g1_pass"],
                                                                     "g1_pass_matched": row["g1_pass_matched"]})
            print(f"{rule.name:22s} dev mid {row['mid']['dev']['excess']['ann']:+6.1f} {row['mid']['dev']['excess']['ci']} "
                  f"matched {row['mid']['dev']['matched']['ann']:+6.1f} | hold {row['mid']['hold']['excess']['ann']:+6.1f} "
                  f"| pass {row['g1_pass']}/{row['g1_pass_matched']}", flush=True)
        out["cells"][name] = cell_out
    g1 = {}
    for key, flag in (("registered", "g1_pass"), ("matched", "g1_pass_matched")):
        metric = "excess" if key == "registered" else "matched"
        passed = {k: v for k, v in primary.items() if v[flag]}
        best = max(passed, key=lambda k: passed[k]["mid"]["dev"][metric]["ann"]) if passed else None
        g1[key] = {"passed": sorted(passed), "selected": best,
                   "selected_hold": None if best is None else {t: primary[best][t]["hold"][metric] for t in COSTS},
                   "selected_vs_cycle": None if best is None else primary[best]["mid"]["vs_cycle"]}
    out["g1"] = g1
    (HERE / "p1_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    register("P1-G1-v2", {"n_configs": len(primary), "rule": "dev mid CI low > 0 and dev high > 0", "warmup": WARMUP}, g1)
    print(json.dumps({"g1": {k: v["passed"] for k, v in g1.items()}, "cycle_rebuild": out["cycle_rebuild"],
                      "pool_carry_minus_pool_ann": out["pool_carry_minus_pool_ann"],
                      "consistency_reconcile": {n: {"c": c["consistency"], "r": c["reconcile"]}
                                                for n, c in out["cells"].items()}}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
