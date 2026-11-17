"""现在拿着什么、下一个开市日买卖什么：照模拟器同一套规则往后多走一天。

明天的价明天才知道，股数和金额先按今天的收盘价估。规则和模拟器一样：开盘涨停买不进的顺延到下一名（所以多列几只备选）；
卖的那一腿跌停或停牌卖不出的就拿着，之后每天同一腿再试。
卖腿是开盘（o1o5）还是尾盘（o1c5）跟着规则走，见 `next_orders`。只列清单，不下单。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rebound.backtest.costs import CostModel
from rebound.backtest.market import Market
from rebound.backtest.sim import Position, Rule, _pick

#: 买单之外多列几只备选（前面的买不进时按顺序顶上）。
ALTERNATES = 3


def _stock(m: Market, names: Mapping[str, tuple[str, str]], j: int) -> dict[str, Any]:
    code = str(m.codes[j])
    name, industry = names.get(code, ("", ""))
    return {"ts_code": code, "name": name, "industry": industry}


def _lots(alloc: float, price: float, cost: CostModel) -> int:
    """这么多钱按这个价最多买几手（含费）：和模拟器同一个算法。"""
    if not np.isfinite(price) or price <= 0:
        return 0
    lots = int(alloc // (price * cost.lot))
    while lots > 0 and lots * cost.lot * price + cost.buy_fee(lots * cost.lot * price) > alloc + 1e-9:
        lots -= 1
    return lots


def holdings(m: Market, names: Mapping[str, tuple[str, str]], positions: Sequence[Position],
             today: int) -> list[dict[str, Any]]:
    """收盘后的持仓：股数换回真实股数（复权股数 × 复权价 / 不复权价），按今天收盘计市值。"""
    out = []
    for p in positions:
        adj, raw = m.price["close"][today, p.j], m.raw["close"][today, p.j]
        value = p.shares * p.last_px
        shares = int(round(p.shares * adj / raw)) if np.isfinite(adj) and np.isfinite(raw) and raw > 0 else None
        out.append({**_stock(m, names, p.j), "shares": shares, "cost": round(p.cost_basis, 2),
                    "value": round(value, 2), "return_pct": round((value / p.cost_basis - 1) * 100, 2),
                    "bought": m.dates[p.entry_day].date().isoformat(), "halted": not np.isfinite(adj),
                    "due": None if p.exit_due < 0 else int(p.exit_due - today)})
    return sorted(out, key=lambda row: -row["value"])


def next_orders(m: Market, names: Mapping[str, tuple[str, str]], cands: Mapping[int, np.ndarray],
                rank: np.ndarray, rule: Rule, cost: CostModel, positions: Sequence[Position],
                tranche_cash: Sequence[float], today: int) -> dict[str, Any]:
    """today 的下一个开市日的卖单、买单、备选。

    开盘买、开盘卖（o1o5）：同一个开盘先卖后买，卖出的钱当场补位。
    开盘买、尾盘卖（o1c5 等）：开盘只补现在空着的位置、只用现在账上的钱；该卖的当天尾盘（收盘竞价）卖，
    卖的钱下一个开市日开盘才用。返回里另带 buy_leg / sell_leg，和 buy_cash（明天开盘买的那一批手上的钱；
    `cash` 是各批合计，FH 分批时开盘只动其中一批）。
    """
    close_exit = (rule.cell.entry, rule.cell.exit) == ("open", "close")
    if not close_exit and (rule.cell.entry, rule.cell.exit) != ("open", "open"):
        raise ValueError(f"{rule.name}：清单只支持开盘买的规则（开盘卖或尾盘卖）")
    if close_exit and rule.kind == "FH" and rule.cell.x < 2:
        raise ValueError(f"{rule.name}：开盘买、当天尾盘卖不合 T+1，尾盘卖至少要 x = 2")
    sell_word = "尾盘卖" if close_exit else "开盘卖"
    tranches = rule.cell.period if rule.kind == "FH" else 1
    cash = list(tranche_cash) if tranche_cash else [cost.capital / tranches] * tranches
    adj, raw = m.price["close"][today], m.raw["close"][today]
    when = m.dates[today]
    live = today in cands
    sells, keep = [], []
    for p in positions:
        if rule.kind == "FH":
            due = today + 1 >= p.exit_due
            reason = "到期"
        else:
            due = live and today + 1 > p.entry_day and rank[today, p.j] > rule.buffer * rule.k
            reason = f"排名掉到第 {int(rank[today, p.j]) if np.isfinite(rank[today, p.j]) else '—'} 名"
        if not due:
            keep.append(p)
            continue
        value = p.shares * p.last_px
        halted = not (np.isfinite(adj[p.j]) and np.isfinite(raw[p.j]) and raw[p.j] > 0)
        shares = None if halted else int(round(p.shares * adj[p.j] / raw[p.j]))
        if halted:
            # 今天停牌：明天多半还卖不出（模拟器也是拿着、每天同一腿再试）。位置和钱都不能先算成空出来的，
            # 不然清单会拿这只票的钱去买别的，而账户里并没有这笔钱。
            keep.append(p)
            sells.append({**_stock(m, names, p.j), "shares": shares, "amount": round(value, 2),
                          "reason": f"{reason}；停牌中，复牌后{sell_word}（这笔钱复牌卖掉之后才能用）", "halted": True})
            continue
        if close_exit:
            keep.append(p)  # 尾盘才卖：明天开盘还拿着，位置不空出来，卖的钱后天开盘才用
        else:
            cash[p.tranche] += value - cost.sell_fee(value, when)
        sells.append({**_stock(m, names, p.j), "shares": shares, "amount": round(value, 2), "reason": reason,
                      "halted": False})
    buys, alternates = [], []
    t = today % tranches
    mine = [p for p in keep if p.tranche == t]
    slots = rule.k - len(mine)
    if live and slots > 0:
        counts: dict[int, int] = {}
        for p in mine:
            counts[m.industry[p.j]] = counts.get(m.industry[p.j], 0) + 1
        budget = cash[t]
        with np.errstate(invalid="ignore"):
            poor = cost.lot_need(raw) > budget / slots + 1e-9
        no_buy = ~np.isfinite(raw) | ~np.isfinite(adj)
        chosen, _, _ = _pick(cands[today], slots + ALTERNATES, {p.j for p in mine}, counts, m.industry, rule.cap,
                             adj, no_buy, "next", poor)
        left = budget
        for n, j in enumerate(chosen):  # 前 slots 只是买单（钱不够一手的那个位置空着，和模拟器一样），后面是备选
            # 备选是顶替某个买不进的位置：按一个位置分到的钱估（DB 的买单按剩下的钱依次分，和模拟器一样）
            alloc = budget / slots if rule.kind == "FH" or n >= slots else left / max(slots - len(buys), 1)
            lots = _lots(alloc, float(raw[j]), cost)
            row = {**_stock(m, names, j), "rank": int(rank[today, j]), "price": round(float(raw[j]), 2),
                   "shares": lots * cost.lot, "amount": round(lots * cost.lot * float(raw[j]), 2)}
            if n >= slots:
                alternates.append(row)
            elif lots:
                buys.append(row)
                left -= row["amount"] + cost.buy_fee(row["amount"])
    out = {"signal_day": when.date().isoformat(), "tranche": t, "tranches": tranches,
           "cash": round(sum(cash), 2), "sells": sells, "buys": buys, "alternates": alternates[:ALTERNATES]}
    if not close_exit:
        return out
    return {**out, "buy_leg": "open", "sell_leg": "close", "buy_cash": round(cash[t], 2)}
