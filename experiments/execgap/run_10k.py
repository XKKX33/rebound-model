"""1 万本金（用户问"2 万半仓是不是就是 1 万永远全仓"）：正式库、每月重训、真实账户口径，和 run_sched 同一套。

- 1 万全仓：现状（回测按开盘价，实盘做不到）、现状 + 0.1% / 0.2%、尾盘卖（前一晚定）+ 第二天开盘买；
  规则 DB-K1-b16（正式的 1 万推荐，也是 2 万半仓按小本金选法挑出的）、DB-K2-b8（2 万正式推荐）。
- 2 万 = 1 万全仓 + 1 万现金一直不动：合起来的年化、回撤、四个起点；现状下这 1 万现金每年能付得起
  多少比例要靠卖出的钱的开盘买单（付不起的部分要等 9:25 卖出的钱，9:30 才成交）。
- 逐年年末：1 万全仓的净值，和 2 万半仓的股票市值、现金。
- 1 万网格（45 条 DB 规则，正式小本金选法）：现状、尾盘卖（前一晚定）、现状 + 0.1% / 0.2%。
用法：python -X utf8 run_10k.py   输出 results_10k.json、grid_10000*.json
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from run_sched import (HERE, SCHEDULES, CapitalRequest, cached_row, check, grid, load, parse, reserve_slots, run,
                       save, say, simulate_sched, stats)

CAPITAL = 10_000
RULES = ("DB-K1-b16", "DB-K2-b8")
SLIPS = (0.001, 0.002)
OUT = HERE / "results_10k.json"
BY = {s.name: s for s in SCHEDULES}


def plus_cash(c, rule, cost, sched) -> dict:
    """2 万 = 1 万按 sched 全仓 + 1 万现金一直不动。现状下另报这 1 万现金付得起的开盘买单比例（逐年、全期）。"""
    res, ledger = simulate_sched(c.m, c.cands, c.rank, rule, cost, c.d_first, c.d_last, sched)
    extra = float(CAPITAL)
    total = res.nav + extra
    p = stats.period(total, c.pool, c.m.dates, c.idx)
    fresh = {}
    for y, d0 in c.starts.items():
        again, _ = simulate_sched(c.m, c.cands, c.rank, rule, cost, d0, c.d_last, sched)
        fresh[str(y)] = stats.window(again.nav + extra, c.pool, d0, c.today, c.m.dates)["cagr"]
    out = {"cagr": p["cagr"], "max_dd": p["max_dd"], "final": round(float(total[c.idx[-1]])), "fresh": fresh,
           "worst": min(fresh.values())}
    if sched.name == "now":
        seen = ledger[ledger.day.isin(c.idx) & ledger.funded]
        day_value = seen.groupby("day").value.sum()
        covered = np.minimum(day_value.to_numpy(), extra)
        g = pd.DataFrame({"y": c.m.dates[day_value.index.to_numpy()].year, "value": day_value.to_numpy(),
                          "covered": covered}).groupby("y").sum()
        out["covered_share"] = {str(y): round(float(cv / v) * 100, 1)
                                for y, v, cv in zip(g.index, g["value"], g["covered"])}
        out["covered_total"] = round(float(covered.sum() / day_value.sum()) * 100, 1)
    return out


def side_by_side(c, rule) -> dict:
    """逐年年末：1 万全仓（现状，回测价）的净值；2 万半仓的股票市值和现金。"""
    full, _ = simulate_sched(c.m, c.cands, c.rank, rule, CapitalRequest(capital=float(CAPITAL)).cost, c.d_first,
                             c.d_last, BY["now"])
    half, _ = simulate_sched(c.m, c.cands, c.rank, rule, CapitalRequest(capital=2.0 * CAPITAL).cost, c.d_first,
                             c.d_last, BY["half"], reserve=reserve_slots(rule.k, "half"))
    years = c.m.dates[c.idx].year
    out = {}
    for y in sorted(set(years)):
        d = int(c.idx[years == y][-1])
        out[str(y)] = {"full_10k": round(float(full.nav[d])), "half_stock": round(float(half.nav[d] - half.cash[d])),
                       "half_cash": round(float(half.cash[d]))}
    return out


def main() -> None:
    t0 = time.time()
    c = load()
    say(f"[行情] 载入完（{time.time() - t0:.0f}s）")
    cost = CapitalRequest(capital=float(CAPITAL)).cost
    out = {"rules": {}}
    for name in RULES:
        rule = parse(name).rule
        check(c, rule, cost)
        rows = {"now": run(c, rule, cost, BY["now"])[0],
                "close_sell_prev_rank": run(c, rule, cost, BY["close_sell_prev_rank"])[0]}
        for s in SLIPS:
            rows[f"now+{s * 100:g}%"] = run(c, rule, cost, BY["now"], s)[0]
        ref, same = cached_row(CAPITAL, parse(name).short), None
        if ref is not None and ref.get("fresh"):
            prod = {"cagr": ref["cagr"], "fresh": {y: v["cagr"] for y, v in ref["fresh"]["starts"].items()}}
            same = prod == {"cagr": rows["now"]["cagr"], "fresh": rows["now"]["fresh"]}
        plus = {k: plus_cash(c, rule, cost, BY[k]) for k in ("now", "close_sell_prev_rank")}
        out["rules"][name] = {"full_10k": rows, "plus_10k_cash": plus, "same_as_cache": same,
                              "side_by_side": side_by_side(c, rule)}
        save(OUT, out)
        say(f"[1 万] {name} 和正式缓存一致：{same}")
        for k, r in rows.items():
            say(f"  1 万全仓 {k:<22} 年化 {r['cagr']:>6}  回撤 {r['max_dd']:>6}  最差起点 {r['worst']:>5} "
                f"{list(r['fresh'].values())}  期末 {r['final']}")
        for k, r in plus.items():
            say(f"  1 万 + 1 万现金 {k:<16} 年化 {r['cagr']:>6}  回撤 {r['max_dd']:>6}  最差起点 {r['worst']:>5} "
                f"{list(r['fresh'].values())}  期末 {r['final']}  付得起 {r.get('covered_total')}% {r.get('covered_share')}")
        say(f"  逐年：{out['rules'][name]['side_by_side']}")
    configs = [("now", BY["now"], 0.0), ("close_sell_prev_rank", BY["close_sell_prev_rank"], 0.0)] + \
              [(f"now+{s * 100:g}%", BY["now"], s) for s in SLIPS]
    grid(c, CAPITAL, configs, HERE / "grid_10000.json", HERE / "grid_10000_picks.json")
    say(f"[完] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
