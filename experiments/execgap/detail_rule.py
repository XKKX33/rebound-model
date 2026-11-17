"""一条规则、一种时点的逐月、逐年明细（2 万本金，2015 年起的那个账户），另加四个起点的年末资金。

默认：尾盘卖（前一晚定）、第二天开盘买，DB-K2-b6。先核对期末金额、年化、最差起点和 best_finals.json 一样。
另报：尾盘卖掉、第二天开盘又买回同一只的次数；排在买进的票前面、手里没有的票开盘涨停（只能顺延）的买入次数；
最后一个开市日的成交、收盘后的持仓，以及下一个开市日开盘要买、尾盘要卖的票。
用法：python -X utf8 detail_rule.py [--rule DB-K2-b6] [--sched close_sell_prev_rank]
输出 detail_<规则>_<时点>.json、monthly_<规则>_<时点>.csv、yearly_<规则>_<时点>.csv
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from run_sched import (DB_PATH, HERE, SCHEDULES, CapitalRequest, load, open_store, parse, reserve_slots, run, save,
                       say, simulate_sched, stats)
from sched_sim import hybrid

CAPITAL = 20_000


def sched_of(key: str):
    if key.startswith("hybrid_r"):
        return hybrid(int(key[len("hybrid_r"):]))
    return {s.name: s for s in SCHEDULES}[key]


def stock_names() -> dict[str, str]:
    with open_store(DB_PATH) as store:
        rows = store.connection.execute("SELECT ts_code, name FROM stock_basic").fetchall()
    return {str(c): "" if n is None else str(n) for c, n in rows}


def periods(nav, pool, costs, dates, d0: int, d1: int, monthly: bool) -> list[dict]:
    """[d0, d1] 按月（或按年）切段：收益、段末资金、准入池同期、段内最大回撤、费用。第一段从 d0 收盘起算。"""
    days = np.arange(d0, d1 + 1)
    when = dates[days]
    last: dict[tuple, int] = {}
    for key, d in zip(zip(when.year, when.month) if monthly else zip(when.year), days):
        last[key] = int(d)
    out, prev = [], d0
    for key, d in sorted(last.items()):
        seg = nav[prev:d + 1]
        out.append({"key": [int(k) for k in key], "ret": float(nav[d] / nav[prev] - 1), "nav": float(nav[d]),
                    "pool": float(np.prod(1 + np.nan_to_num(pool[prev + 1:d + 1])) - 1),
                    "dd": float((seg / np.maximum.accumulate(seg) - 1).min()),
                    "fees": float(costs[prev + 1:d + 1].sum())})
        prev = d
    return out


def worst_drawdown(nav, dates, d0: int, d1: int) -> dict:
    seg = nav[d0:d1 + 1]
    dd = seg / np.maximum.accumulate(seg) - 1
    t = int(np.argmin(dd))
    p = int(np.argmax(seg[:t + 1]))
    after = np.flatnonzero(seg[t:] >= seg[p])
    return {"dd": float(dd[t]), "peak": str(dates[d0 + p].date()), "peak_nav": float(seg[p]),
            "trough": str(dates[d0 + t].date()), "trough_nav": float(seg[t]),
            "recovered": str(dates[d0 + t + int(after[0])].date()) if len(after) else None}


def held_mask(res, ledger, sold_at_open: bool) -> list[set[int]]:
    """每笔买单下单那个开盘手里拿着的票（之前买进、还没卖的，加上同一个开盘先买进的）。
    sold_at_open：卖在开盘（同一个开盘先卖后买，卖出日不算拿着）；否则卖在尾盘（卖出日开盘还拿着）。"""
    t = res.trades
    j = np.concatenate([t.j.to_numpy(), np.array([p.j for p in res.positions], dtype=np.int64)])
    a = np.concatenate([t.entry_day.to_numpy(), np.array([p.entry_day for p in res.positions], dtype=np.int64)])
    b = np.concatenate([t.exit_day.to_numpy(), np.full(len(res.positions), np.iinfo(np.int64).max)])
    out, seen_day, same = [], None, set()
    for d, jj in zip(ledger.day.tolist(), ledger.j.tolist()):
        if d != seen_day:
            seen_day, same = d, set()
        alive = (a < d) & ((d < b) if sold_at_open else (d <= b))
        out.append(set(j[alive].tolist()) | same)
        same = same | {jj}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="DB-K2-b6")
    ap.add_argument("--sched", default="close_sell_prev_rank")
    args = ap.parse_args()
    c = load()
    m, cost, rule, sched = c.m, CapitalRequest(capital=float(CAPITAL)).cost, parse(args.rule).rule, sched_of(args.sched)
    reserve = reserve_slots(rule.k, sched.keep_cash)
    res, ledger = simulate_sched(m, c.cands, c.rank, rule, cost, c.d_first, c.d_last, sched, reserve=reserve)
    row, _ = run(c, rule, cost, sched)
    ref = json.loads((HERE / "best_finals.json").read_text(encoding="utf-8")).get(f"{args.sched}|{args.rule}")
    if ref is not None and (ref["final"], ref["cagr"], ref["worst"]) != (row["final"], row["cagr"], row["worst"]):
        raise SystemExit(f"和 best_finals.json 对不上：{ref} 对 {row}")
    if round(float(res.nav[c.today])) != row["final"]:
        raise SystemExit("期末金额和 run() 的不一样")
    nav, d0, d1, dates, names = res.nav, c.d_first, c.today, m.dates, stock_names()
    monthly = periods(nav, c.pool, res.costs, dates, d0, d1, monthly=True)
    yearly = periods(nav, c.pool, res.costs, dates, d0, d1, monthly=False)
    per_year = ledger.assign(y=dates[ledger.day.to_numpy()].year).groupby("y").size()
    for r in yearly:
        r["buys"] = int(per_year.get(r["key"][0], 0))
    fresh = {}
    for y, s0 in c.starts.items():
        again, _ = simulate_sched(m, c.cands, c.rank, rule, cost, s0, c.d_last, sched, reserve=reserve)
        fresh[str(y)] = {"cagr": row["fresh"][str(y)], "year_end": {
            str(r["key"][0]): round(r["nav"]) for r in periods(again.nav, c.pool, again.costs, dates, s0, d1, False)}}
    sold = set(zip(res.trades.exit_day.tolist(), res.trades.j.tolist()))
    back = sum((d - 1, j) in sold for d, j in zip(ledger.day.tolist(), ledger.j.tolist()))
    up = m.buy_blocked[sched.buy_leg]
    fallback = 0
    for (d, j, r), held in zip(ledger[["day", "j", "rank"]].itertuples(index=False, name=None),
                               held_mask(res, ledger, sched.sell_leg == "open" and not sched.hybrid)):
        ahead = [int(x) for x in c.cands[d - sched.buy_lag][:max(int(r) - 1, 0)] if int(x) not in held]
        fallback += any(up[d, x] for x in ahead)
    label = lambda j: f"{m.codes[j]} {names.get(str(m.codes[j]), '')}".strip()  # noqa: E731
    t_last = d1
    now_held = {p.j for p in res.positions}
    last = {
        "day": str(dates[t_last].date()),
        "bought_open": [{"stock": label(j), "rank": int(r), "value": round(v)} for j, r, v in
                        ledger[ledger.day == t_last][["j", "rank", "value"]].itertuples(index=False, name=None)],
        "sold_close": [{"stock": label(j), "since": str(dates[a].date()), "net_pct": round(x * 100, 1)} for j, a, x in
                       res.trades[res.trades.exit_day == t_last][["j", "entry_day", "net"]].itertuples(
                           index=False, name=None)],
        "holding": [{"stock": label(p.j), "since": str(dates[p.entry_day].date()),
                     "rank_tonight": int(c.rank[t_last, p.j]) if np.isfinite(c.rank[t_last, p.j]) else None,
                     "value": round(p.shares * p.last_px), "ret_pct": round((p.shares * p.last_px / p.cost_basis - 1) * 100, 1)}
                    for p in res.positions],
        "cash": round(float(res.cash[t_last])),
        "next_open_buy": [label(int(x)) for x in c.cands[t_last] if int(x) not in now_held][:max(rule.k - len(now_held), 0) + 3]
        if rule.k > len(now_held) else [],
        "next_close_sell": [label(p.j) for p in res.positions if c.rank[t_last, p.j] > rule.buffer * rule.k]}
    out = {"rule": args.rule, "sched": args.sched, "row": row, "drawdown": worst_drawdown(nav, dates, d0, d1),
           "monthly": monthly, "yearly": yearly, "fresh": fresh, "buys": int(len(ledger)), "buyback": int(back),
           "fallback_after_limit_up": int(fallback), "last_day": last}
    tag = f"{args.rule}_{args.sched}"
    save(HERE / f"detail_{tag}.json", out)
    mdf = pd.DataFrame([{"年": r["key"][0], "月": r["key"][1], "收益%": round(r["ret"] * 100, 2), "月末资金": round(r["nav"]),
                         "准入池%": round(r["pool"] * 100, 2)} for r in monthly])
    mdf.to_csv(HERE / f"monthly_{tag}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"年": r["key"][0], "收益%": round(r["ret"] * 100, 2), "年末资金": round(r["nav"]),
                   "准入池%": round(r["pool"] * 100, 2), "年内最大回撤%": round(r["dd"] * 100, 1),
                   "费用": round(r["fees"]), "买入次数": r["buys"]} for r in yearly]).to_csv(
        HERE / f"yearly_{tag}.csv", index=False, encoding="utf-8-sig")
    say(f"[{tag}] 期末 {row['final']}  年化 {row['cagr']}  最差起点 {row['worst']}  回撤 {out['drawdown']}")
    for r in yearly:
        say(f"  {r['key'][0]}  收益 {r['ret'] * 100:7.1f}%  年末 {r['nav']:>10.0f}  池 {r['pool'] * 100:6.1f}%  "
            f"年内回撤 {r['dd'] * 100:6.1f}%  费用 {r['fees']:8.0f}  买 {r['buys']}")
    say(mdf.pivot(index="年", columns="月", values="收益%").to_string())
    pos = (mdf["收益%"] > 0).mean() * 100
    say(f"  赚钱的月 {pos:.0f}%  最好 {mdf['收益%'].max()}  最差 {mdf['收益%'].min()}")
    say(f"  换起点：{json.dumps(fresh, ensure_ascii=False)}")
    say(f"  买入 {len(ledger)} 笔，尾盘卖掉第二天开盘又买回 {back} 笔，前面的票开盘涨停只能顺延 {fallback} 笔")
    say(f"  最后一天：{json.dumps(last, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
