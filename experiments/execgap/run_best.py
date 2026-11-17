"""2 万本金：实盘能执行、收益最高的买卖时点和持股规则（用户 2026-11-01）。

"能执行"：每笔都在集合竞价成交（开盘 9:25、收盘 15:00），下单时买单的钱已经在账上，用的排名是下单前收盘后就算好的。
要盘中估当天排名的（标"偏乐观"）只作上限。口径和 run_sched 相同（正式库、每月重训、真实账户、四个起点）。

已算过的（grid_20000*.json）：现状（做不到，参照）、尾盘卖（前一晚定）、空仓一天、留一只现金、永远半仓、股票封顶、
开盘卖尾盘买、尾盘卖（当天排名，偏乐观）。这里补：
- 混合：留 R 只票的现金（R = 1、2、3），开盘集合竞价能换几只就换几只，换不过来的同一天尾盘卖、第二天开盘买。
- 开盘卖、同一天尾盘买（当天排名，偏乐观）。
- 固定持有（正式模拟器的 FH 规则，每天买前 K 名、分批）：开盘买、第 x 天尾盘卖；尾盘买、第 x 天开盘卖
  （前一天的排名，能执行；当天排名，偏乐观）。K ∈ {1,2,3,4}，x ∈ {2,3,4,5,6,8,10}。
每类按正式小本金选法挑（四个起点都赚、最差起点年化最高），另报邻近规则的最差起点平均（看是不是孤峰）、全期年化最高的那条。
用法：python -X utf8 run_best.py   输出 grid_20000_best.json、grid_20000_fh.json、best_summary.json
"""
from __future__ import annotations

import json
import time

import numpy as np

from run_sched import (GRID_FIELDS, GRID_K, HERE, SCHEDULES, CapitalRequest, grid, load, parse, run, save, say,
                       simulate_sched, stats)
from rebound.labels import Cell
from rebound.backtest.rules import DB_B
from rebound.backtest.sim import Rule, simulate
from sched_sim import hybrid

CAPITAL = 20_000
FH_K, FH_X = (1, 2, 3, 4), (2, 3, 4, 5, 6, 8, 10)
DB_OUT, FH_OUT, SUMMARY_OUT = HERE / "grid_20000_best.json", HERE / "grid_20000_fh.json", HERE / "best_summary.json"
BY = {s.name: s for s in SCHEDULES}
FAMILIES = {  # 类 → （说明，能不能执行）
    "close_sell_prev_rank": ("尾盘卖（前一晚定）、第二天开盘买", True),
    "hybrid_r1": ("留 1 只的现金开盘换，换不过来的尾盘卖", True),
    "hybrid_r2": ("留 2 只的现金开盘换，换不过来的尾盘卖", True),
    "hybrid_r3": ("留 3 只的现金开盘换，换不过来的尾盘卖", True),
    "half": ("永远半仓，开盘卖开盘买", True),
    "reserve": ("留约 1 只的现金开盘换，换不过来的等第二天", True),
    "cap": ("股票封顶 1 万，开盘卖开盘买", True),
    "wait_a_day": ("开盘卖、空仓一天、第二天开盘买", True),
    "open_sell_close_buy": ("开盘卖、尾盘买（前一天排名）", True),
    "fh_o2c": ("固定持有：开盘买、第 x 天尾盘卖", True),
    "fh_c2o": ("固定持有：尾盘买（前一天排名）、第 x 天开盘卖", True),
    "close_sell_same_day": ("尾盘卖（当天排名，偏乐观）、第二天开盘买", False),
    "open_sell_close_buy_same_day": ("开盘卖、尾盘买（当天排名，偏乐观）", False),
    "fh_c2o_same": ("固定持有：尾盘买（当天排名，偏乐观）、第 x 天开盘卖", False),
    "now": ("现状：开盘先卖后买，回测价（做不到）", False),
}


def check_hybrid0(c, cost) -> None:
    """混合、R = 0 必须和"尾盘卖（前一晚定）"逐位相同。"""
    rule = parse("DB-K2-b8").rule
    a, _ = simulate_sched(c.m, c.cands, c.rank, rule, cost, c.d_first, c.d_last, BY["close_sell_prev_rank"])
    b, _ = simulate_sched(c.m, c.cands, c.rank, rule, cost, c.d_first, c.d_last, hybrid(0), reserve=0)
    if not (np.array_equal(a.nav, b.nav, equal_nan=True) and a.trades.equals(b.trades)):
        raise SystemExit("混合 R = 0 和尾盘卖（前一晚定）对不上")


def fh_one(c, rule, cost, cands) -> dict:
    res = simulate(c.m, cands, c.rank, rule, cost, c.d_first, c.d_last)
    s = stats.summary(res, c.pool, c.m.dates, c.idx)
    fresh = {str(y): stats.window(simulate(c.m, cands, c.rank, rule, cost, d0, c.d_last).nav, c.pool, d0, c.today,
                                  c.m.dates)["cagr"] for y, d0 in c.starts.items()}
    return {"cagr": s["cagr"], "excess": s["excess"], "max_dd": s["max_dd"], "worst": min(fresh.values()),
            "fresh": fresh, "turnover": s["turnover"], "hold_days": s["hold_days"], "final": s["final"]}


def fh_grid(c, cost) -> None:
    """固定持有。fh_c2o_same：把信号日往前挪一天，T 日尾盘就用 T 日的排名买（偏乐观）。"""
    done = json.loads(FH_OUT.read_text(encoding="utf-8")) if FH_OUT.is_file() else {}
    shifted = {e - 1: v for e, v in c.cands.items()}
    t0 = time.time()
    for key, entry, exit_, cands in (("fh_o2c", "open", "close", c.cands), ("fh_c2o", "close", "open", c.cands),
                                     ("fh_c2o_same", "close", "open", shifted)):
        for k in FH_K:
            for x in FH_X:
                tag = f"{key}|K{k}-x{x}"
                if tag not in done:
                    done[tag] = fh_one(c, Rule("FH", k, Cell(entry, x, exit_)), cost, cands)
            save(FH_OUT, done)
        say(f"[固定持有] {key} 算完（{time.time() - t0:.0f}s）")


def position(name: str) -> tuple[int, int]:
    if name.startswith("DB-"):
        _, k, b = name.split("-")
        return GRID_K.index(int(k[1:])), DB_B.index(float(b[1:]))
    k, x = name.split("-")
    return FH_K.index(int(k[1:])), FH_X.index(int(x[1:]))
def neighbor_worst(mine: dict, name: str) -> float:
    """同一类里网格上相邻（两个参数各差一格以内，含自己）的规则的最差起点年化，取平均：看挑出的是不是孤峰。"""
    i, j = position(name)
    near = [v["worst"] for n, v in mine.items() if abs(position(n)[0] - i) <= 1 and abs(position(n)[1] - j) <= 1]
    return round(float(np.mean(near)), 1)


def summarize(years: float) -> dict:
    rows = {}
    for f in (HERE / "grid_20000.json", HERE / "grid_20000_cap.json", DB_OUT, FH_OUT):
        rows.update(json.loads(f.read_text(encoding="utf-8")))
    out = {}
    for key, (label, executable) in FAMILIES.items():
        mine = {t.split("|", 1)[1]: v for t, v in rows.items() if t.split("|", 1)[0] == key}
        robust = {n: v for n, v in mine.items() if v["worst"] > 0}
        pick = max(robust, key=lambda n: robust[n]["worst"]) if robust else None
        top = max(mine, key=lambda n: mine[n]["cagr"])
        grow = lambda r: round(CAPITAL * (1 + r["cagr"] / 100) ** years)  # noqa: E731（按计分期年化折算）
        row = {"label": label, "executable": executable, "robust_count": f"{len(robust)}/{len(mine)}",
               "pick": pick, "top_cagr": top, "top_row": {**mine[top], "to": grow(mine[top])}}
        if pick:
            row["row"] = {**mine[pick], "to": grow(mine[pick]), "neighbor_worst": neighbor_worst(mine, pick)}
        out[key] = row
        r = row.get("row") or {}
        say(f"  {'能' if executable else '否'} {label:<30} 选 {pick}  年化 {r.get('cagr')}  回撤 {r.get('max_dd')}  "
            f"最差起点 {r.get('worst')}（邻近 {r.get('neighbor_worst')}）  2 万 → {r.get('to')}  都赚的 "
            f"{row['robust_count']}  | 全期年化最高 {top} {mine[top]['cagr']}（最差起点 {mine[top]['worst']}）")
    save(SUMMARY_OUT, out)
    return out


def main() -> None:
    t0 = time.time()
    c = load()
    say(f"[行情] 载入完（{time.time() - t0:.0f}s）")
    cost = CapitalRequest(capital=float(CAPITAL)).cost
    check_hybrid0(c, cost)
    say("[核对] 混合 R = 0 和尾盘卖（前一晚定）逐位相同")
    configs = [(f"hybrid_r{r}", hybrid(r), 0.0) for r in (1, 2, 3)] + \
              [("open_sell_close_buy_same_day", BY["open_sell_close_buy_same_day"], 0.0)]
    grid(c, CAPITAL, configs, DB_OUT, HERE / "grid_20000_best_picks.json")
    fh_grid(c, cost)
    summarize(len(c.idx) / 242)
    say(f"[完] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
