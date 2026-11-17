"""换手当天的买卖时点：模拟器（`rebound/backtest/sim.py` 的 simulate 原样，只把卖和买拆成可以放在不同时点）。

问题：A 股开盘集合竞价 9:25 一次撮合。满仓的账户，买单的钱要等卖单 9:25 成交回到账上，最早 9:30 连续竞价才成交；
正式回测（`now`）却让同一个开盘先卖后买、都按开盘价。这里每种时点（Schedule）定下：卖在哪一腿、用哪天的排名，
买在哪一腿、用哪天的排名，买单能不能用同一腿卖出的钱。

- 买单能用同一腿卖出的钱（`own_only=False`）：开盘前账上的闲钱付得起的按这一腿的价成交；付不起、要靠刚卖出的钱的，
  按 价 × (1 + slip) 成交（slip 是 9:25 到 9:30 之后的价差，用来做敏感性）。`now` 在 slip = 0 时和正式模拟器逐位相同。
- 只用开盘前就在账上的钱（`own_only=True`）：当天卖出的钱第二天才用；reserve 个"空位"的钱一直留着现金，
  所以每天的买单 = 前一天卖出空出来的钱 + 留着的现金里够得上的部分。

排名：信号日 e 收盘后才有 rank[e]（模型用当天收盘的数据）。"同一天尾盘按当天排名卖"要盘中估排名，回测里用的是收盘后的
排名，偏乐观（标出来，只作上限）。T+1：当天买的不能当天卖（day > entry_day）。只支持真实账户口径、DB 规则（买不进顺延）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rebound.backtest.costs import CostModel
from rebound.backtest.market import Market
from rebound.backtest.sim import MIN_ALLOC, TRADE_COLS, Position, Rule, SimResult, _pick_next

BUY_COLS = ["day", "j", "rank", "funded", "value", "nav_prev"]
#: own_only：开盘前的现金不到这个空位应分到的钱的一半，就不买零头，空位等第二天用今天卖出的钱补
#: （不然一手的零头会占住空位好几天，卖出的钱一直闲着）。
SLIVER = 0.5


@dataclass(frozen=True)
class Schedule:
    name: str
    label: str
    sell_leg: str        # "open" / "close"
    sell_lag: int        # 卖：看哪天的排名（0 = 当天收盘后的排名，偏乐观；1 = 前一天的）
    buy_leg: str
    own_only: bool = False
    keep_cash: str = ""  # own_only 时留多少现金（见 reserve_slots）
    hybrid: bool = False  # own_only：开盘只换现金换得过来的几只（排名最差的先卖），其余同一天尾盘卖
    buy_lag: int = 1      # 买：看哪天的排名（0 = 当天收盘后的排名，只能尾盘买，偏乐观）


SCHEDULES = (
    Schedule("now", "现状：开盘先卖后买（回测按同一个开盘价）", "open", 1, "open"),
    Schedule("open_sell_close_buy", "开盘卖、同一天尾盘（收盘竞价）买", "open", 1, "close"),
    Schedule("wait_a_day", "开盘卖、空仓一天、第二天开盘买", "open", 1, "open", own_only=True),
    Schedule("close_sell_prev_rank", "尾盘卖（用前一天的排名）、第二天开盘买", "close", 1, "open"),
    Schedule("close_sell_same_day", "尾盘卖（当天排名，偏乐观）、第二天开盘买", "close", 0, "open"),
    Schedule("reserve", "留一部分现金：开盘卖的同时用现金在集合竞价买", "open", 1, "open", own_only=True,
             keep_cash="part"),
    Schedule("half", "永远半仓：开盘卖的同时用另一半现金在集合竞价买", "open", 1, "open", own_only=True,
             keep_cash="half"),
    Schedule("cap", "股票封顶：买入不超过本金的一半，赚的钱留成现金，开盘集合竞价买", "open", 1, "open", own_only=True,
             keep_cash="cap"),
    Schedule("open_sell_close_buy_same_day", "开盘卖、同一天尾盘买（当天排名，偏乐观）", "open", 1, "close", buy_lag=0),
)


def hybrid(n: int) -> Schedule:
    """留 n 只票的现金：开盘集合竞价能换几只就换几只（排名最差的先卖），换不过来的同一天尾盘卖、第二天开盘买。
    n = 0 就是"尾盘卖（前一晚定）、第二天开盘买"。"""
    return Schedule(f"hybrid_r{n}", f"留 {n} 只的现金开盘换，换不过来的尾盘卖", "open", 1, "open", own_only=True,
                    keep_cash=f"r{n}", hybrid=True)


def reserve_slots(k: int, mode: str) -> int:
    """留几个"空位"的钱当现金。part：K = 2 留 1 个（三分之一现金），K = 30 留 3 个（约十分之一）；
    half：留 K 个（一半现金，K 只票同一天全换也付得起）；cap：同 half，另外每只票买入不超过 本金 / 2 / K
    （本金 2 万、K = 1 就是 1 万），赚的钱留成现金；rN：留 N 个（混合时点用）；其余不留。"""
    if mode in ("half", "cap"):
        return k
    if mode.startswith("r"):
        return int(mode[1:])
    return max(1, round(k / 10)) if mode == "part" else 0


def _lots(alloc: float, r: float, costs: CostModel) -> int:
    lots = int(alloc // (r * costs.lot))
    while lots > 0 and lots * costs.lot * r + costs.buy_fee(lots * costs.lot * r) > alloc + 1e-9:
        lots -= 1
    return lots


def simulate_sched(m: Market, cands: dict[int, np.ndarray], rank: np.ndarray, rule: Rule, costs: CostModel,
                   d_first: int, d_last: int, sched: Schedule, slip: float = 0.0,
                   reserve: int = 0) -> tuple[SimResult, pd.DataFrame]:
    """和 `simulate` 相同的账户、费用、选股，只是卖和买的时点按 sched。另返回每笔买单
    （哪天、哪只、排第几、是不是靠同一腿卖出的钱、成交额、前一天收盘净值）。"""
    if rule.kind != "DB" or rule.fill != "next" or not isinstance(costs, CostModel) or m.raw is None:
        raise ValueError("只支持 DB 规则（买不进顺延）、真实账户口径")
    if sched.sell_leg == "open" and sched.sell_lag == 0:
        raise ValueError("开盘卖只能用前一天的排名")
    if sched.buy_leg == "open" and sched.buy_lag == 0:
        raise ValueError("开盘买只能用前一天的排名")
    n, start = len(m.dates), costs.capital
    reserve = reserve if sched.own_only else 0
    cap_each = start / 2 / rule.k if sched.keep_cash == "cap" else np.inf  # 每只票买入的上限
    cash = np.full(1, start / 1)
    positions: list[Position] = []
    nav, holdings, cash_track = np.full(n, np.nan), np.zeros(n, dtype=np.int32), np.full(n, np.nan)
    traded, spent = np.zeros(n), np.zeros(n)
    log, buys, blocked, poor, stuck = [], [], 0, 0, 0
    buy_events = poor_events = 0
    nav[d_first] = cash_track[d_first] = start
    for day in range(d_first + 1, n):
        when = m.dates[day]
        later: set[int] = set()  # 混合：开盘该卖、现金不够当场换的，这一天尾盘卖
        for leg in ("open", "close"):
            price, no_sell = m.price[leg], m.sell_blocked[leg]
            own = float(cash[0])  # 这一腿之前就在账上的钱
            es = day - sched.sell_lag
            live_sell = d_first <= es <= d_last and es in cands
            if sched.hybrid:
                if leg == "open":
                    flagged = sorted((p for p in positions if live_sell and day > p.entry_day
                                      and rank[es, p.j] > rule.buffer * rule.k), key=lambda p: -rank[es, p.j])
                    due, later = {id(p) for p in flagged[:reserve]}, {id(p) for p in flagged[reserve:]}
                else:
                    due = later
            elif leg == sched.sell_leg and live_sell:
                due = {id(p) for p in positions if day > p.entry_day and rank[es, p.j] > rule.buffer * rule.k}
            else:
                due = set()
            if due:
                keep = []
                for p in positions:
                    if id(p) not in due:
                        keep.append(p)
                        continue
                    px = price[day, p.j]
                    if not np.isfinite(px) or no_sell[day, p.j]:
                        p.stuck, stuck = True, stuck + 1
                        keep.append(p)
                        continue
                    value = p.shares * px
                    fee = costs.sell_fee(value, when)
                    proceeds = value - fee
                    cash[0] += proceeds
                    traded[day] += value
                    spent[day] += fee
                    log.append((p.j, p.signal_day, p.entry_day, day, p.entry_px, px, px / p.entry_px - 1,
                                proceeds / p.cost_basis - 1 if p.cost_basis > 0 else np.nan, p.stuck, p.tranche,
                                p.cost_basis))
                positions[:] = keep
            e = day - sched.buy_lag
            if leg != sched.buy_leg or not (d_first <= e <= d_last and e in cands):
                continue
            slots = rule.k - len(positions)
            if slots <= 0:
                continue
            counts: dict[int, int] = {}
            for p in positions:
                counts[m.industry[p.j]] = counts.get(m.industry[p.j], 0) + 1
            raw = m.raw[leg][day]
            no_buy = m.buy_blocked[leg][day] | ~np.isfinite(raw)
            budget = min(own, cash[0] / (slots + reserve), cap_each) if sched.own_only else cash[0] / slots
            with np.errstate(invalid="ignore"):
                poor_row = costs.lot_need(raw) > budget + 1e-9
            chosen, hit, short = _pick_next(cands[e], slots, {p.j for p in positions}, counts, m.industry, rule.cap,
                                            price[day], no_buy, poor_row)
            blocked, poor = blocked + hit, poor + short
            buy_events += 1
            poor_events += short > 0
            filled = 0
            for j in chosen:
                if sched.own_only:
                    target = min(cash[0] / (slots - filled + reserve), cap_each)
                    alloc = min(own, target)
                    if alloc < SLIVER * target:
                        break
                else:
                    alloc = cash[0] / (slots - filled)
                if alloc <= MIN_ALLOC * start:
                    break
                px, r = price[day, j], raw[j]
                lots = _lots(alloc, r, costs)
                funded = (not sched.own_only and lots > 0
                          and lots * costs.lot * r + costs.buy_fee(lots * costs.lot * r) > own + 1e-9)
                paid = r * (1 + slip) if funded and slip else r
                if paid != r:
                    lots = _lots(alloc, paid, costs)
                if lots == 0:
                    poor += 1
                    continue
                value = lots * costs.lot * paid
                fee = costs.buy_fee(value)
                shares = value / px if paid == r else lots * costs.lot * r / px  # 滑点只让同样的钱买到更少的股
                cost = value + fee
                own = 0.0 if funded else own - cost
                cash[0] -= cost
                traded[day] += value
                spent[day] += fee
                buys.append((day, j, float(rank[e, j]), funded, value, nav[day - 1]))
                entry = px if paid == r else px * (1 + slip)
                positions.append(Position(j, shares, day, entry, cost, e, 0, -1, px))
                filled += 1
        close = m.price["close"][day]
        for p in positions:
            if np.isfinite(close[p.j]):
                p.last_px = close[p.j]
        nav[day] = cash.sum() + sum(p.shares * p.last_px for p in positions)
        holdings[day] = len(positions)
        cash_track[day] = cash.sum()
    res = SimResult(nav, holdings, traded, spent, pd.DataFrame(log, columns=TRADE_COLS), blocked, stuck,
                    cash=cash_track, poor_buys=poor, poor_events=poor_events, buy_events=buy_events,
                    positions=tuple(positions), tranche_cash=tuple(float(c) for c in cash))
    return res, pd.DataFrame(buys, columns=BUY_COLS)
