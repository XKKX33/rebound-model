"""DB-K2-b6（每月重训、不限价、2 万、佣金万五最低 5 元）2026 年逐月盈亏。

主账户：2025-12-31 收盘用 2 万开新账户，2026 年第一个开市日（01-05）开盘起按规则买卖。
对照：从 2015 年一直按它做的账户（本金测算页里的那个）2026 年每月的收益；DB-K3-b16 同样 2026 年初开 2 万新账户。
读正式库实跑前的那份副本（行情到 2026-10-30、正式模型的打分都一样），不碰正式库本身。
用法：python -X utf8 -B year2026_k2b6.py   输出 year2026_k2b6.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from october_k2b6 import DB, HERE, positions_of, say  # 同一目录：复用股数还原和日志

from rebound.store import Store
from rebound.backtest.costs import CostModel
from rebound.backtest.market import candidates, load_markets, model_scores, pool_daily_return
from rebound.backtest.rules import parse
from rebound.backtest.sim import simulate

OUT = HERE / "year2026_k2b6.json"
CAPITAL = 20_000.0


def monthly(nav, pool, dates, d0, today, res=None, trades=()) -> list[dict]:
    """每个月：月初、月底的账户，盈亏，准入池同期，月内最大回撤；给了 res 再加买卖次数、费用、亏得最多和赚得最多的几笔。"""
    out, prev, period, at = [], d0, dates.to_period("M"), np.arange(len(dates))
    for mo in pd.period_range("2026-01", "2026-12", freq="M"):
        idx = np.flatnonzero((period == mo) & (at > d0) & (at <= today))
        if not len(idx):
            continue
        last = int(idx[-1])
        seg = np.concatenate([[nav[prev]], nav[idx]])
        row = {"month": str(mo), "start": float(nav[prev]), "end": float(nav[last]), "ret": float(nav[last] / nav[prev] - 1),
               "pnl": float(nav[last] - nav[prev]), "pool": float(np.prod(1 + pool[idx]) - 1),
               "max_dd": float((seg / np.maximum.accumulate(seg) - 1).min()), "days": int(len(idx))}
        row["excess"] = row["ret"] - row["pool"]
        if res is not None:
            closed = [o for o in trades if o.get("exit") is not None and idx[0] <= o["exit"] <= last]
            ranked = sorted((round(o["pnl"]), o["name"], str(dates[o["entry"]].date())) for o in closed)
            row.update(fees=float(res.costs[idx].sum()), buys=int(sum(idx[0] <= o["entry"] <= last for o in trades)),
                       sells=len(closed), wins=int(sum(o["pnl"] > 0 for o in closed)),
                       worst=ranked[:3], best=ranked[::-1][:3])
        out.append(row)
        prev = last
    return out


def main() -> None:
    with Store(DB) as store:
        basic = store.connection.execute("SELECT ts_code, name FROM stock_basic").df()
    names = dict(zip(basic.ts_code, basic.name))
    m = load_markets(DB, log=say)["nocap"]
    dates, today = m.dates, len(m.dates) - 1
    d0 = int(dates.get_loc(pd.Timestamp("2025-12-31")))
    cands, rank = candidates(m, model_scores(DB, m, "nocap", "M"))
    pool = pool_daily_return(m)
    cost = CostModel(capital=CAPITAL)
    k2 = parse("DB-K2-b6")
    say("模拟")
    fresh = simulate(m, cands, rank, k2.rule, cost, d0, today - 1)
    trades = positions_of(fresh, m, rank, cost, names)
    nav = fresh.nav[d0:today + 1]
    trough = int(np.argmin(nav / np.maximum.accumulate(nav)))
    peak = int(np.argmax(nav[:trough + 1]))
    closed = [o for o in trades if o.get("exit") is not None]
    summary = {"end": float(fresh.nav[today]), "ret": float(fresh.nav[today] / CAPITAL - 1),
               "pool": float(np.prod(1 + pool[d0 + 1:today + 1]) - 1),
               "max_dd": float(nav[trough] / nav[peak] - 1), "peak": [str(dates[d0 + peak].date()), float(nav[peak])],
               "trough": [str(dates[d0 + trough].date()), float(nav[trough])],
               "fees": float(fresh.costs[d0 + 1:today + 1].sum()), "buys": len(trades), "closed": len(closed),
               "wins": int(sum(o["pnl"] > 0 for o in closed)),
               "hold_days": float(np.mean([o["exit"] - o["entry"] for o in closed])),
               "open": [(o["name"], round(o["pnl"])) for o in trades if o.get("exit") is None]}
    full = simulate(m, cands, rank, k2.rule, cost, int(dates.searchsorted(pd.Timestamp("2015-01-01"))), today - 1)
    k3 = simulate(m, cands, rank, parse("DB-K3-b16").rule, cost, d0, today - 1)
    result = {"months": monthly(fresh.nav, pool, dates, d0, today, fresh, trades), "summary": summary,
              "since_2015": {"months": monthly(full.nav, pool, dates, d0, today),
                             "ytd": float(full.nav[today] / full.nav[d0] - 1), "nav_1231": float(full.nav[d0])},
              "k3_b16_fresh": {"months": monthly(k3.nav, pool, dates, d0, today),
                               "ytd": float(k3.nav[today] / CAPITAL - 1)}}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    say(f"2026 年初新开 2 万：{summary['ret']:+.2%}（准入池 {summary['pool']:+.2%}）；老账户今年 {result['since_2015']['ytd']:+.2%}"
        f"（本金测算页记的是 -19.28%）；K3-b16 新开 {result['k3_b16_fresh']['ytd']:+.2%}")


if __name__ == "__main__":
    main()
