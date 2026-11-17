"""2 万本金、股票封顶（用户 2026-11-01 的方案）：手里 2 万，买的股票总额不超过 1 万，另 1 万（以及以后赚的钱）留成现金，
开盘卖的同时用现金在集合竞价买，所以按开盘价成交。和"永远半仓"（股票跟着净值的一半长）比；1 万全仓回测价见 run_10k。

- 规则：DB-K1-b16（1 万正式推荐，也是 2 万半仓按小本金选法挑出的）、DB-K2-b8（2 万正式推荐）。
- 每种：全期年化、回撤、四个起点；逐年年末净值、股票市值、现金；全期赚的钱折成"1 万本金每年赚多少"（单利）。
- 网格：封顶时点在 45 条 DB 规则里按正式小本金选法重新挑。
用法：python -X utf8 run_cap.py   输出 results_cap.json、grid_20000_cap*.json
"""
from __future__ import annotations

import time

from run_sched import (HERE, SCHEDULES, YEAR, CapitalRequest, grid, load, parse, reserve_slots, run, save, say,
                       simulate_sched)

CAPITAL = 20_000
RULES = ("DB-K1-b16", "DB-K2-b8")
OUT = HERE / "results_cap.json"
BY = {s.name: s for s in SCHEDULES}


def yearly(c, rule, cost, sched) -> dict:
    """逐年年末：净值、股票市值、现金（2015 年起的那个账户）。"""
    res, _ = simulate_sched(c.m, c.cands, c.rank, rule, cost, c.d_first, c.d_last, sched,
                            reserve=reserve_slots(rule.k, sched.keep_cash))
    years = c.m.dates[c.idx].year
    out = {}
    for y in sorted(set(years)):
        d = int(c.idx[years == y][-1])
        out[str(y)] = {"nav": round(float(res.nav[d])), "stock": round(float(res.nav[d] - res.cash[d])),
                       "cash": round(float(res.cash[d]))}
    return out


def main() -> None:
    t0 = time.time()
    c = load()
    say(f"[行情] 载入完（{time.time() - t0:.0f}s）")
    cost = CapitalRequest(capital=float(CAPITAL)).cost
    span = (c.idx[-1] - c.d_first) / YEAR
    out = {"years": round(span, 2), "rules": {}}
    for name in RULES:
        rule = parse(name).rule
        rows, detail = {}, {}
        for key in ("cap", "half"):
            r, _ = run(c, rule, cost, BY[key])
            r["simple_per_10k"] = round((r["final"] - CAPITAL) / (CAPITAL / 2) / span * 100, 1)
            rows[key], detail[key] = r, yearly(c, rule, cost, BY[key])
        out["rules"][name] = {"rows": rows, "yearly": detail}
        save(OUT, out)
        for key, r in rows.items():
            say(f"[{name}] {key:<5} 年化 {r['cagr']:>6}  回撤 {r['max_dd']:>6}  最差起点 {r['worst']:>5} "
                f"{list(r['fresh'].values())}  期末 {r['final']}  折 1 万本金每年 {r['simple_per_10k']}%")
            say(f"    逐年 {detail[key]}")
    grid(c, CAPITAL, [("cap", BY["cap"], 0.0)], HERE / "grid_20000_cap.json", HERE / "grid_20000_cap_picks.json")
    say(f"[完] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
