"""DB-K2-b6（每月重训、不限价、2 万、佣金万五最低 5 元）2026 年 10 月逐日买卖。

主账户：9-30 收盘用 2 万开新账户，10-08 开盘起按规则买卖（9-30 的信号在国庆后第一个开市日成交）。
对照：从 2015 年一直按它做的账户（本金测算页里的那个）10 月的收益和买的票；DB-K3-b16 同样开新账户的 10 月收益。
读正式库实跑前的那份副本（行情到 2026-10-30、正式模型的打分都一样），不碰正式库本身。
用法：python -X utf8 -B october_k2b6.py   输出 october_k2b6.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from rebound.store import Store  # noqa: E402
from rebound.backtest.costs import CostModel  # noqa: E402
from rebound.backtest.market import candidates, load_markets, model_scores, pool_daily_return  # noqa: E402
from rebound.backtest.rules import parse  # noqa: E402
from rebound.backtest.sim import simulate  # noqa: E402

DB = ROOT / "data/market.duckdb.bak-before-return-model-run-202610311928"
OUT = HERE / "october_k2b6.json"
VARIANT, CADENCE, CAPITAL = "nocap", "M", 20_000.0
T0 = time.time()


def say(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [{time.time() - T0:5.0f}s] {msg}", flush=True)


def positions_of(res, m, rank, cost, names, since: int = -1) -> list[dict]:
    """平掉的和还拿着的每一笔（`since` 那天之前就平掉的不要）：实际股数、不复权的买卖价、费用、盈亏、排名。"""
    raw_open = m.raw["open"]
    out = []
    items = [dict(j=int(t.j), signal=int(t.signal_day), entry=int(t.entry_day), exit=int(t.exit_day),
                  cost=float(t.cost_basis), net=float(t.net), stuck=bool(t.carried)) for t in res.trades.itertuples()
             if int(t.exit_day) > since]
    items += [dict(j=int(p.j), signal=int(p.signal_day), entry=int(p.entry_day), exit=None, cost=float(p.cost_basis),
                   value_now=float(p.shares * p.last_px)) for p in res.positions]
    for o in items:
        j, r = o["j"], float(raw_open[o["entry"], o["j"]])
        lots = int(o["cost"] // (r * cost.lot))  # 买入费比一手贵时会多算，往下退到"股数 + 费用 = 成本"那一档
        while lots > 0 and lots * cost.lot * r + cost.buy_fee(lots * cost.lot * r) > o["cost"] + 1e-6:
            lots -= 1
        value = lots * cost.lot * r
        if abs(value + cost.buy_fee(value) - o["cost"]) > 1e-6 * max(1.0, o["cost"]):
            raise SystemExit(f"股数还原不出来：{o}")
        o.update(code=str(m.codes[j]), name=names.get(str(m.codes[j]), ""), shares=lots * cost.lot, buy_px=r,
                 buy_value=value, buy_fee=o["cost"] - value, buy_rank=float(rank[o["signal"], j]))
        if o["exit"] is not None:
            proceeds = o["cost"] * (1 + o["net"])
            o.update(sell_px=float(raw_open[o["exit"], j]), proceeds=proceeds, pnl=proceeds - o["cost"],
                     sell_rank=float(rank[o["exit"] - 1, j]))
        else:
            o["pnl"] = o["value_now"] - o["cost"]
        out.append(o)
    return out


def skipped_before(m, cands, rank, cost, res, opened, d, k) -> list[dict]:
    """d 这天开盘补仓时，排在买进的票前面、被跳过的票和原因（和模拟器的判断顺序相同）。"""
    buys = [o for o in opened if o["entry"] == d]
    if not buys:
        return []
    before = [o for o in opened if o["entry"] < d and (o["exit"] is None or o["exit"] > d)]
    held = {o["j"] for o in before}
    slots = k - len(before)
    cash = float(res.cash[d]) + sum(o["cost"] for o in buys)
    per_slot = cash / slots
    raw, adj, blocked = m.raw["open"][d], m.price["open"][d], m.buy_blocked["open"][d]
    want, chosen, notes = [o["j"] for o in buys], [], []
    for j in cands[d - 1]:
        if len(chosen) == len(want):
            break
        j = int(j)
        if j in held:
            notes.append((j, "已经拿着"))
        elif blocked[j] or not np.isfinite(adj[j]) or not np.isfinite(raw[j]):
            notes.append((j, "开盘涨停或停牌，买不进"))
        elif cost.lot_need(np.array([raw[j]]))[0] > per_slot + 1e-9:
            notes.append((j, f"一手要 {cost.lot_need(np.array([raw[j]]))[0]:,.0f} 元，每个空位只分到 {per_slot:,.0f} 元"))
        else:
            chosen.append(j)
    if sorted(chosen) != sorted(want):  # 成交记录按卖出的顺序排，比的是买了哪几只
        raise SystemExit(f"{m.dates[d].date()} 复算的买入 {chosen} 和模拟器的 {want} 不一样")
    return [{"code": str(m.codes[j]), "rank": float(rank[d - 1, j]), "why": why} for j, why in notes]


def main() -> None:
    with Store(DB) as store:
        basic = store.connection.execute("SELECT ts_code, name FROM stock_basic").df()
    names = dict(zip(basic.ts_code, basic.name))
    m = load_markets(DB, log=say)["nocap"]
    dates, today = m.dates, len(m.dates) - 1
    assert dates[today] == pd.Timestamp("2026-10-30"), dates[today]
    d0 = int(dates.get_loc(pd.Timestamp("2026-09-30")))
    cands, rank = candidates(m, model_scores(DB, m, VARIANT, CADENCE))
    pool = pool_daily_return(m)
    cost = CostModel(capital=CAPITAL)
    k2 = parse("DB-K2-b6")
    say("模拟")
    fresh = simulate(m, cands, rank, k2.rule, cost, d0, today - 1)
    opened = positions_of(fresh, m, rank, cost, names)
    days = []
    for d in range(d0 + 1, today + 1):
        held = [o for o in opened if o["entry"] <= d and (o["exit"] is None or o["exit"] > d)]
        days.append({"day": str(dates[d].date()), "nav": float(fresh.nav[d]), "ret": float(fresh.nav[d] / fresh.nav[d - 1] - 1),
                     "pool": float(pool[d]), "fee": float(fresh.costs[d]),
                     "sells": [o["code"] for o in opened if o["exit"] == d],
                     "buys": [o["code"] for o in opened if o["entry"] == d],
                     "held": [o["code"] for o in held],
                     "skipped": skipped_before(m, cands, rank, cost, fresh, opened, d, k2.k)})
    nav = fresh.nav[d0:today + 1]
    summary = {"start": CAPITAL, "end": float(fresh.nav[today]), "ret": float(fresh.nav[today] / CAPITAL - 1),
               "pool": float(np.prod(1 + pool[d0 + 1:today + 1]) - 1),
               "max_dd": float((nav / np.maximum.accumulate(nav) - 1).min()), "fees": float(fresh.costs[d0 + 1:].sum())}
    summary["excess"] = summary["ret"] - summary["pool"]

    full = simulate(m, cands, rank, k2.rule, cost, int(dates.searchsorted(pd.Timestamp("2015-01-01"))), today - 1)
    full_opened = positions_of(full, m, rank, cost, names, since=d0)
    say(f"老账户 10-30 收盘 {full.nav[today]:,.0f} 元（本金测算页记的是 2,028,849）")
    old = {"nav_0930": float(full.nav[d0]), "nav_1030": float(full.nav[today]),
           "ret": float(full.nav[today] / full.nav[d0] - 1),
           "held_0930": [o["code"] for o in full_opened if o["entry"] <= d0 and (o["exit"] is None or o["exit"] > d0)],
           "bought_oct": [(str(dates[o["entry"]].date()), o["code"], o["shares"]) for o in full_opened if o["entry"] > d0],
           "held_1030": [o["code"] for o in full_opened if o["exit"] is None]}
    k3 = simulate(m, cands, rank, parse("DB-K3-b16").rule, cost, d0, today - 1)
    k3_opened = positions_of(k3, m, rank, cost, names)
    other = {"ret": float(k3.nav[today] / CAPITAL - 1),
             "bought": [(str(dates[o["entry"]].date()), o["code"]) for o in k3_opened]}
    result = {"trades": opened, "days": days, "summary": summary, "since_2015": old, "k3_b16_fresh": other,
              "names": {o["code"]: o["name"] for o in opened + full_opened + k3_opened}}
    for o in opened:
        for key in ("entry", "exit", "signal"):
            if o.get(key) is not None:
                o[f"{key}_day"] = str(dates[o[key]].date())
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    say(f"10 月新开 2 万：{summary['ret']:+.2%}，准入池 {summary['pool']:+.2%}；老账户 10 月 {old['ret']:+.2%}；"
        f"K3-b16 新开 {other['ret']:+.2%}")


if __name__ == "__main__":
    main()
