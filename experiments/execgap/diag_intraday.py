"""诊断（只解释机制，结论以完整模拟为准）：正式推荐规则按现状时点买进的票，买入当天 开盘 → 收盘、收盘 → 第二天开盘
各涨多少；卖掉的票卖出当天 开盘 → 收盘（"尾盘才卖"多拿的那段）。都和同一天的准入池（前一天收盘在池里的票等权）比。

用法：python -X utf8 diag_intraday.py   输出 diag_intraday.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from run_sched import HERE, RULES, CapitalRequest, load, parse, save, say, simulate, stats

OUT = HERE / "diag_intraday.json"


def pool_move(m, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐日：前一天收盘在准入池里的票，b / a − 1 的等权均值（两头都有价的）。"""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = b / a - 1
    ok = np.zeros_like(m.eligible)
    ok[1:] = m.eligible[:-1]
    ok &= np.isfinite(r)
    return np.where(ok.any(axis=1), np.where(ok, r, 0).sum(axis=1) / np.maximum(ok.sum(axis=1), 1), np.nan)


def describe(stock: np.ndarray, pool: np.ndarray, when: pd.DatetimeIndex) -> dict:
    keep = np.isfinite(stock) & np.isfinite(pool)
    gap = pd.Series((stock - pool)[keep], index=when[keep])
    lo, hi = stats.month_ci(gap)
    return {"n": int(keep.sum()), "stock": round(float(stock[keep].mean()) * 100, 3),
            "pool": round(float(pool[keep].mean()) * 100, 3), "gap": round(float(gap.mean()) * 100, 3),
            "ci": [round(lo * 100, 3), round(hi * 100, 3)], "up_share": round(float((stock[keep] > 0).mean()) * 100, 1)}


def main() -> None:
    c = load()
    o, cl = c.m.price["open"], c.m.price["close"]
    nxt = np.full_like(o, np.nan)
    nxt[:-1] = o[1:]
    oc, co = pool_move(c.m, o, cl), pool_move(c.m, cl, nxt)
    out = {}
    for capital, names in RULES.items():
        cost = CapitalRequest(capital=float(capital)).cost
        for name in names:
            res = simulate(c.m, c.cands, c.rank, parse(name).legacy_rule, cost, c.d_first, c.d_last)  # 现状 = 开盘卖
            t = res.trades[res.trades.entry_day.isin(c.idx) & ~res.trades.carried]
            bd, bj, sd, sj = (t.entry_day.to_numpy(), t.j.to_numpy(), t.exit_day.to_numpy(), t.j.to_numpy())
            with np.errstate(invalid="ignore", divide="ignore"):
                buy_oc, buy_co = cl[bd, bj] / o[bd, bj] - 1, nxt[bd, bj] / cl[bd, bj] - 1
                sell_oc = cl[sd, sj] / o[sd, sj] - 1
            out[f"{capital}|{name}"] = {
                "buy_day_open_to_close": describe(buy_oc, oc[bd], c.m.dates[bd]),
                "buy_day_close_to_next_open": describe(buy_co, co[bd], c.m.dates[bd]),
                "sell_day_open_to_close": describe(sell_oc, oc[sd], c.m.dates[sd]),
                "buy_closes_limit_up": round(float(c.m.buy_blocked["close"][bd, bj].mean()) * 100, 1)}
            say(f"{capital} {name}: {json.dumps(out[f'{capital}|{name}'], ensure_ascii=False)}")
    save(OUT, out)


if __name__ == "__main__":
    main()
