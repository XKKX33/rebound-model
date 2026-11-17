"""最少持有天数：组合模拟器（p0p1 `sim.simulate`）只改 DB 的卖出条件，外加报告用的几个小工具。

DB 原来：前一天排名掉出 buffer·K、且不是当天买的，就在 entry 腿卖。这里：排名掉出 buffer·K、且持有天数
（卖出日 − 买入日，开市日下标之差，停牌的日子照算）≥ min_hold 才卖。N 天之内掉出缓冲的票照样拿着、占着名额。
min_hold = 1 就是原规则（test_minhold.py 和 run_minhold.py --check 逐项核对）。FH 本来就是到期才卖，不接受 min_hold。

其余逐行照抄 p0p1 的 simulate。挑票用正式系统 `rebound/backtest/sim.py` 的 `_pick_next`（`_pick(fill="next")` 的快版，
三个返回值逐项相同；抄在这里，免得为它导入整个 rebound 包）：2 万本金、K 大时，原版每天要把整张候选单一只只看完。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from engine import Market, month_ci  # noqa: E402
from sim import MIN_ALLOC, TRADE_COLS, YEAR, CostModel, Position, Rule, SimResult, _pick  # noqa: E402

#: 标签 o1o5 的持有天数：T+1 开盘买、T+5 开盘卖
LABEL_HOLD = 4
#: 要算的 N（PREREG.md：主假设 N = 4，其余只画曲线）。先算 1 和 4，配对比较最早能做
NS = (1, 4, 2, 3, 5, 6, 8, 10)
#: 每种本金：决定用的规则、参照规则（PREREG.md"主比较"）
PRIMARY = {20000: ("DB-K2-b6-o1o5", "DB-K3-b16-o1o5"), 200000: ("DB-K30-b4-o1o5", "DB-K30-b8-o1o5")}
FRESH_YEARS = (2017, 2019, 2021, 2023)
#: 准入池变体 → 十年实验里对应的目录（面板、坐标轴、每月重训打分从那里硬链接到本目录的同名子目录）
VARIANTS = {"nocap": HERE.parent / "decade" / "nocap", "cap70": HERE.parent / "decade"}


def pick_next(order, slots, held, counts, industry, cap, price_row, blocked_row, poor_row=None):
    """`_pick(fill="next")` 的快版（正式 `rebound/backtest/sim.py` 的 `_pick_next` 原样），三个返回值逐项相同。

    先成批筛掉买不进、买不起的，只在能买的里逐个过"已持有、行业上限"；再按原版的口径补数跳过的票：
    只数停下之前、没持有、当时行业还没满的那些（行业满不满取决于它前面已经买进了哪些）。
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


def simulate_minhold(m: Market, cands: dict[int, np.ndarray], rank: np.ndarray | None, rule: Rule,
                     costs: "tuple[float, float] | CostModel", d_first: int, d_last: int,
                     min_hold: int = 1) -> SimResult:
    """p0p1 `sim.simulate` + DB 的最少持有天数（见模块说明）。min_hold = 1 时和原版逐项相同。"""
    if min_hold < 1:
        raise ValueError(f"min_hold={min_hold}：至少是 1（当天买的不能当天卖）")
    if rule.kind != "DB" and min_hold != 1:
        raise ValueError(f"min_hold={min_hold}：只对 DB 规则有意义，FH 本来就是到期才卖")
    n, cell = len(m.dates), rule.cell
    real = isinstance(costs, CostModel)
    if real and m.raw is None:
        raise ValueError("真实账户口径需要未复权价：先 engine.attach_raw(market)")
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
                else:  # 唯一的改动：原来是 day > p.entry_day
                    due = (leg == cell.entry and live and day - p.entry_day >= min_hold
                           and rank[e, p.j] > rule.buffer * rule.k)
                if not due:
                    keep.append(p)
                    continue
                px = price[day, p.j]
                if not np.isfinite(px) or no_sell[day, p.j]:
                    p.stuck, stuck = True, stuck + 1
                    keep.append(p)
                    continue
                value = p.shares * px  # 复权股数 × 复权价 = 这笔持仓现在值多少元（除权前后连续）
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
                chosen, hit, short = pick_next(cands[e], slots, held, counts, m.industry, rule.cap,
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
                     cash=cash_track, poor_buys=poor, poor_events=poor_events, buy_events=buy_events)


def hold_mix(trades: pd.DataFrame, top: int = 6) -> dict[str, dict]:
    """成交按持有天数分组（top 天及以上并成一组）：占成交笔数的百分比、平均每笔净收益（%）。"""
    if trades.empty:
        return {}
    held = (trades.exit_day - trades.entry_day).clip(upper=top)
    out = {}
    for h, g in trades.groupby(held):
        key = f"{h}+" if h >= top else str(h)
        out[key] = {"share": round(len(g) / len(trades) * 100, 1), "net": round(float(g.net.mean()) * 100, 2)}
    return out


def rest_of_window(m: Market, trades: pd.DataFrame, hold: int = LABEL_HOLD) -> pd.DataFrame:
    """持有不到 hold 天就卖掉的成交，如果一直拿到买入后第 hold 天开盘：从实际卖出那天开盘起，这只票自己赚多少（stock）、
    同期准入池等权赚多少（pool：卖出前一天收盘在池里、两头都有开盘价的票）、两者之差（gap）。

    只看这只票自己，不管空出来的位置买了什么，所以只是解释机制。拿满 hold 天的、目标日超出数据的不算；
    这只票两头有一头没有开盘价（停牌）的也不算。
    """
    o = m.price["open"]
    t = trades[(trades.exit_day - trades.entry_day) < hold]
    t = t.assign(target=t.entry_day + hold)
    t = t[t.target < len(m.dates)]
    j, x, y = t.j.to_numpy(), t.exit_day.to_numpy(), t.target.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        stock = o[y, j] / o[x, j] - 1
        pools = {}
        for a, b in set(zip(x.tolist(), y.tolist())):
            ok = m.eligible[a - 1] & np.isfinite(o[a]) & np.isfinite(o[b])
            pools[a, b] = float(np.mean(o[b, ok] / o[a, ok] - 1)) if ok.any() else np.nan
    pool = np.array([pools[a, b] for a, b in zip(x.tolist(), y.tolist())], dtype=float)
    t = t.assign(stock=stock, pool=pool, gap=stock - pool)
    return t[np.isfinite(t.stock) & np.isfinite(t.pool)].reset_index(drop=True)


def paired(a: np.ndarray, b: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    """两条逐日收益之差（准入池基准两边相同，相减抵消）：年化（%）、按月整块重抽 95% 区间、逐年年化。"""
    diff = pd.Series(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), index=dates)
    lo, hi = month_ci(diff)
    years = {str(y): round(float(s.mean()) * YEAR * 100, 2) for y, s in diff.groupby(diff.index.year)}
    return {"ann": round(float(diff.mean()) * YEAR * 100, 2),
            "ci": [round(lo * YEAR * 100, 2), round(hi * YEAR * 100, 2)], "years": years}


def verdict(ci, share: float, worst_ok: bool = True) -> str:
    """预注册的判断（PREREG.md"主比较"）。ci：决定用的规则 N 对 N=1 配对差的年化区间；share：90 条 DB 规则里
    N 的全期超额高于 N=1 的比例（0~1）；worst_ok：2 万的最差起点年化不比 N=1 低（20 万不看，传 True）。"""
    lo, hi = ci
    if lo > 0 and share >= 0.6 and worst_ok:
        return "更好"
    if hi < 0 and share <= 0.4:
        return "更差"
    return "分不出"
