"""组合模拟器：每天的打分 → 持仓、成交、逐日净值。买卖腿相同的规则（o1o5）和十年实验（`p0p1-20261029/sim.py`）逐行相同；
DB 买卖腿不同（o1c5、c1o5）时研究版在买腿上卖，这里在卖腿上卖，两边不再相同。

FH（固定持有、错开分批）：资金分成 P 个批次（P = 买卖时点的资金占用天数），信号日 t 归第 t mod P 批；
这一批在 T+1 的 entry 腿买当天前 K 名（行业去重），到 T+x 的 exit 腿卖。上一批卖不出的票占着名额。
DB（每日调仓带缓冲）：一个资金池，最多持 K 只；在 exit 腿上，昨天排名掉出 buffer·K 的卖掉，在 entry 腿上空位按昨天的排名补。
entry = exit（o1o5，十年实验的规则）：同一个开盘先卖后买。exit 是收盘（o1c5）：前一晚定好、尾盘卖，卖的钱第二天开盘再补位
（满仓账户同一个开盘先卖后买实盘做不到：买单的钱要等 9:25 卖出成交才到账，见 `data/research/execgap-20261101`）。
共同：T 日信号最早 T+1 成交；当天买的不能当天卖；同一腿先卖后买；买不进时 next 顺延、skip 空着；
卖不出（跌停锁死或停牌）就拿着，之后每天同一腿重试；停牌按最后一个收盘价计值。净值按收盘计，起步那天是本金。

另外和研究版不同：`SimResult` 另外带出期末的持仓和各批次现金（给"现在拿着什么、明天买卖什么"用），这一处不影响任何数字。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rebound.labels import Cell
from rebound.backtest.costs import CostModel
from rebound.backtest.market import Market

#: 一年按 242 个开市日算。
YEAR = 242
#: 计分前跳过的净值日数：FH 最多要 5 天才建满仓，建仓期半仓和满仓的基准不可比。所有配置统一跳过。
WARMUP = 5
#: 单笔开仓的最小金额（相对本金）。
MIN_ALLOC = 1e-9
TRADE_COLS = ["j", "signal_day", "entry_day", "exit_day", "entry_px", "exit_px", "gross", "net", "carried", "tranche",
              "cost_basis"]


@dataclass(frozen=True)
class Rule:
    kind: str           # "FH" 或 "DB"
    k: int
    cell: Cell
    fill: str = "next"  # 买不进时：next 顺延到下一名，skip 空着这个位置
    buffer: float = 2.0
    cap: int = 2        # 每个行业最多几只

    @property
    def name(self) -> str:
        tail = f"b{self.buffer:g}" if self.kind == "DB" else self.fill
        return f"{self.kind}-K{self.k}-{tail}-{self.cell.name}"


@dataclass
class Position:
    j: int
    shares: float       # 复权股数：复权股数 × 复权价 = 现在值多少元（除权前后连续）
    entry_day: int
    entry_px: float
    cost_basis: float
    signal_day: int
    tranche: int
    exit_due: int
    last_px: float
    stuck: bool = False


@dataclass
class SimResult:
    nav: np.ndarray
    holdings: np.ndarray
    traded: np.ndarray
    costs: np.ndarray
    trades: pd.DataFrame
    blocked_buys: int
    stuck_days: int
    cash: np.ndarray | None = None
    poor_buys: int = 0     # 依次往下找时，因为一手的钱超过这个位置分到的钱而跳过的票数
    poor_events: int = 0   # 有过这种跳过的建仓次数
    buy_events: int = 0    # 有空位、尝试建仓的次数
    positions: tuple[Position, ...] = ()   # 最后一天收盘后的持仓
    tranche_cash: tuple[float, ...] = ()   # 最后一天收盘后各批次的现金


def _pick(order, slots, held, counts, industry, cap, price_row, blocked_row, fill, poor_row=None):
    """按顺序挑最多 slots 个名额；返回能买的票、买不进的次数、买不起一手的次数。skip 模式下这两种照样占名额。"""
    out, blocked, poor, taken, counts = [], 0, 0, 0, dict(counts)
    for j in order:
        if taken >= slots:
            break
        g = industry[j]
        if j in held or counts.get(g, 0) >= cap:
            continue
        no_buy = not np.isfinite(price_row[j]) or blocked_row[j]
        too_poor = not no_buy and poor_row is not None and poor_row[j]
        if no_buy or too_poor:
            blocked += no_buy
            poor += too_poor
            if fill == "skip":
                counts[g] = counts.get(g, 0) + 1
                taken += 1
            continue
        out.append(int(j))
        counts[g] = counts.get(g, 0) + 1
        taken += 1
    return out, blocked, poor


def _pick_next(order, slots, held, counts, industry, cap, price_row, blocked_row, poor_row=None):
    """`_pick(fill="next")` 的快版，三个返回值逐项相同（有随机对拍测试）。

    小本金时每个名额分到的钱少，大多数票一手都买不起，原版每天要把整张候选单一只只看完。这里先成批筛掉
    买不进、买不起的，只在能买的里逐个过"已持有、行业上限"；再按原版的口径补数跳过的票：只数停下之前、
    没持有、当时行业还没满的那些（行业满不满取决于它前面已经买进了哪些）。
    """
    if slots <= 0:
        return [], 0, 0
    order = np.asarray(order)
    no_buy = ~np.isfinite(price_row[order]) | blocked_row[order]
    too_poor = ~no_buy & poor_row[order] if poor_row is not None else np.zeros(len(order), dtype=bool)
    start, counts = dict(counts), dict(counts)
    out, positions = [], []
    stop = len(order)  # 原版看过的位置是 [0, stop)
    for at in np.flatnonzero(~no_buy & ~too_poor):
        j = int(order[at])
        g = industry[j]
        if j in held or counts.get(g, 0) >= cap:
            continue
        out.append(j)
        positions.append(int(at))
        counts[g] = counts.get(g, 0) + 1
        if len(out) >= slots:
            stop = int(at) + 1
            break
    bad = np.flatnonzero((no_buy | too_poor)[:stop])
    if not len(bad):
        return out, 0, 0
    js = order[bad]
    g_bad = industry[js]
    before = np.zeros(int(industry.max()) + 1, dtype=np.int64)
    for g, n in start.items():  # 开头的行业计数，再加上每只跳过的票之前已买进的同行业只数
        before[int(g)] += n
    seen = before[g_bad]
    if out:
        pick_g = industry[np.array(out)]
        seen = seen + ((np.array(positions)[None, :] < bad[:, None]) & (pick_g[None, :] == g_bad[:, None])).sum(axis=1)
    keep = (seen < cap) & ~np.isin(js, np.fromiter(held, dtype=np.int64, count=len(held)))
    return out, int(no_buy[bad][keep].sum()), int(too_poor[bad][keep].sum())


def simulate(m: Market, cands: dict[int, np.ndarray], rank: np.ndarray | None, rule: Rule,
             costs: tuple[float, float] | CostModel, d_first: int, d_last: int) -> SimResult:
    """costs 是 (买, 卖) 比例时：净值从 1 起、按比例收费、不取整（资金不受限）。是 CostModel 时：真实账户口径。"""
    n, cell = len(m.dates), rule.cell
    if rule.kind == "FH" and (cell.entry, cell.exit) == ("open", "close") and cell.x < 2:
        raise ValueError(f"{rule.name}：开盘买、当天尾盘卖不合 T+1，尾盘卖至少要 x = 2")
    real = isinstance(costs, CostModel)
    if real and m.raw is None:
        raise ValueError("真实账户口径需要未复权价")
    buy_fee, sell_fee = (0.0, 0.0) if real else costs
    start = costs.capital if real else 1.0
    tranches = cell.period if rule.kind == "FH" else 1
    cash = np.full(tranches, start / tranches)
    positions: list[Position] = []
    nav, holdings, cash_track = np.full(n, np.nan), np.zeros(n, dtype=np.int32), np.full(n, np.nan)
    traded, spent = np.zeros(n), np.zeros(n)
    log, blocked, poor, stuck = [], 0, 0, 0
    buy_events = poor_events = 0
    nav[d_first] = cash_track[d_first] = start
    for day in range(d_first + 1, n):
        e = day - 1
        live = d_first <= e <= d_last and e in cands
        when = m.dates[day]
        for leg in ("open", "close"):
            price, no_sell = m.price[leg], m.sell_blocked[leg]
            keep = []
            for p in positions:
                if rule.kind == "FH":
                    due = leg == cell.exit and day >= p.exit_due
                else:
                    due = leg == cell.exit and live and day > p.entry_day and rank[e, p.j] > rule.buffer * rule.k
                if not due:
                    keep.append(p)
                    continue
                px = price[day, p.j]
                if not np.isfinite(px) or no_sell[day, p.j]:
                    p.stuck, stuck = True, stuck + 1
                    keep.append(p)
                    continue
                value = p.shares * px
                fee = costs.sell_fee(value, when) if real else value * sell_fee
                proceeds = value - fee
                cash[p.tranche] += proceeds
                traded[day] += value
                spent[day] += fee
                log.append((p.j, p.signal_day, p.entry_day, day, p.entry_px, px, px / p.entry_px - 1,
                            proceeds / p.cost_basis - 1 if p.cost_basis > 0 else np.nan, p.stuck, p.tranche,
                            p.cost_basis))
            positions[:] = keep
            if leg != cell.entry or not live:
                continue
            t = e % tranches
            mine = [p for p in positions if p.tranche == t]
            counts: dict[int, int] = {}
            for p in mine:
                counts[m.industry[p.j]] = counts.get(m.industry[p.j], 0) + 1
            slots = rule.k - len(mine)
            if slots <= 0:
                continue
            budget = cash[t]
            no_buy = m.buy_blocked[leg][day]
            poor_row = None
            if real:
                raw = m.raw[leg][day]
                no_buy = no_buy | ~np.isfinite(raw)
                with np.errstate(invalid="ignore"):
                    poor_row = costs.lot_need(raw) > budget / slots + 1e-9
            held = {p.j for p in mine}
            if rule.fill == "next":
                chosen, hit, short = _pick_next(cands[e], slots, held, counts, m.industry, rule.cap,
                                                price[day], no_buy, poor_row)
            else:
                chosen, hit, short = _pick(cands[e], slots, held, counts, m.industry, rule.cap,
                                           price[day], no_buy, rule.fill, poor_row)
            blocked, poor = blocked + hit, poor + short
            buy_events += 1
            poor_events += short > 0
            filled = 0
            for j in chosen:
                alloc = budget / slots if rule.kind == "FH" else cash[t] / (slots - filled)
                if alloc <= MIN_ALLOC * start:
                    break  # 没有钱就不开仓，免得 0 股的仓位占住名额
                px = price[day, j]
                if real:
                    r = m.raw[leg][day, j]
                    lots = int(alloc // (r * costs.lot))
                    while lots > 0 and lots * costs.lot * r + costs.buy_fee(lots * costs.lot * r) > alloc + 1e-9:
                        lots -= 1
                    if lots == 0:
                        poor += 1
                        continue
                    value = lots * costs.lot * r
                    fee = costs.buy_fee(value)
                    shares, cost = value / px, value + fee  # 记成复权股数：之后按复权价计值、卖出
                else:
                    shares = alloc / (px * (1 + buy_fee))
                    value, fee, cost = shares * px, shares * px * buy_fee, alloc
                cash[t] -= cost
                traded[day] += value
                spent[day] += fee
                due_day = e + cell.x if rule.kind == "FH" else -1
                positions.append(Position(j, shares, day, px, cost, e, t, due_day, px))
                filled += 1
        close = m.price["close"][day]
        for p in positions:
            if np.isfinite(close[p.j]):
                p.last_px = close[p.j]
        nav[day] = cash.sum() + sum(p.shares * p.last_px for p in positions)
        holdings[day] = len(positions)
        cash_track[day] = cash.sum()
    return SimResult(nav, holdings, traded, spent, pd.DataFrame(log, columns=TRADE_COLS), blocked, stuck,
                     cash=cash_track, poor_buys=poor, poor_events=poor_events, buy_events=buy_events,
                     positions=tuple(positions), tranche_cash=tuple(float(c) for c in cash))
