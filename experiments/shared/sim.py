"""组合模拟器：每天的打分 → 持仓、成交、逐日净值。规则见 PREREG.md。

FH（固定持有、错开分批）：资金分成 P 个批次（P = 格子的资金占用天数），信号日 t 归第 t mod P 批；
这一批在 T+1 的 entry 腿买当天前 K 名（行业去重），到 T+x 的 exit 腿卖。上一批卖不出的票占着名额。
DB（每日调仓带缓冲）：一个资金池，最多持 K 只；在 entry 腿上，昨天排名掉出 buffer·K 的卖掉，空位按昨天的排名补。
共同：T 日信号最早 T+1 成交；当天买的不能当天卖；同一腿先卖后买；买不进时 next 顺延、skip 空着；
卖不出（跌停锁死或停牌）就拿着，之后每天同一腿重试；停牌按最后一个收盘价计值。净值按收盘计，d_first 当天为 1。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from engine import DEV_END, Cell, Market, month_ci

YEAR = 242
#: 计分前跳过的净值日数：FH 最多要 5 天才建满仓，建仓期半仓和满仓的基准不可比。所有配置统一跳过。
WARMUP = 5
#: 单笔开仓的最小金额（净值从 1 起算）。
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
    shares: float
    entry_day: int
    entry_px: float
    cost_basis: float
    signal_day: int
    tranche: int
    exit_due: int
    last_px: float
    stuck: bool = False


@dataclass(frozen=True)
class CostModel:
    """真实账户：本金按元计，买入按一手取整，佣金有最低额，卖出收印花税。

    默认是用户 2026-10-30 给的口径：本金 2 万，佣金万五（每笔最低 5 元），印花税万五（卖出），过户费十万分之一（双边）。
    stamp_before：2023-08-28 印花税减半之前的税率（None 表示全期按现行税率）。
    """

    capital: float = 20_000.0
    commission: float = 0.0005
    min_commission: float = 5.0
    stamp: float = 0.0005
    transfer: float = 0.00001
    lot: int = 100
    stamp_before: float | None = None

    def buy_fee(self, value: float) -> float:
        return max(self.min_commission, value * self.commission) + value * self.transfer if value > 0 else 0.0

    def sell_fee(self, value: float, when: pd.Timestamp) -> float:
        """不超过成交额本身：券商从卖出所得里扣费，扣不出负数（否则本金很小时现金会被扣成负的）。"""
        if value <= 0:
            return 0.0
        stamp = self.stamp_before if self.stamp_before is not None and when < STAMP_CUT else self.stamp
        return min(value, max(self.min_commission, value * self.commission) + value * (stamp + self.transfer))

    def lot_need(self, raw_price: np.ndarray) -> np.ndarray:
        """买一手要多少钱（含费），逐只股票。"""
        value = raw_price * self.lot
        return value + np.maximum(self.min_commission, value * self.commission) + value * self.transfer

    def scaled(self, factor: float) -> "CostModel":
        """所有费率和最低佣金乘以 factor（G1 的 high 档）。"""
        return replace(self, commission=self.commission * factor, min_commission=self.min_commission * factor,
                       stamp=self.stamp * factor, transfer=self.transfer * factor,
                       stamp_before=None if self.stamp_before is None else self.stamp_before * factor)


STAMP_CUT = pd.Timestamp("2023-08-28")


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


def simulate(m: Market, cands: dict[int, np.ndarray], rank: np.ndarray | None, rule: Rule,
             costs: "tuple[float, float] | CostModel", d_first: int, d_last: int) -> SimResult:
    """costs 是 (买, 卖) 比例时：净值从 1 起、按比例收费、不取整（资金不受限）。是 CostModel 时：真实账户口径。"""
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
                else:
                    due = leg == cell.entry and live and day > p.entry_day and rank[e, p.j] > rule.buffer * rule.k
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
            chosen, hit, short = _pick(cands[e], slots, {p.j for p in mine}, counts, m.industry, rule.cap,
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

def _days(m: Market, d_first: int, d_last: int) -> dict[str, np.ndarray]:
    """组合的计分日：d_first+1+WARMUP .. d_last+1（建仓期和最后一个信号日之后的收尾不计）。按净值日分开发期和留出期。"""
    idx = np.arange(d_first + 1 + WARMUP, d_last + 2)
    dates = m.dates[idx]
    return {"dev": idx[dates <= DEV_END], "hold": idx[dates > DEV_END], "all": idx}


def _annual(daily: pd.Series) -> dict:
    lo, hi = month_ci(daily)
    return {"ann": round(float(daily.mean()) * YEAR * 100, 2), "ci": [round(lo * YEAR * 100, 2), round(hi * YEAR * 100, 2)]}


def summarize(m: Market, res: SimResult, pool: np.ndarray, d_first: int, d_last: int) -> dict:
    """各期：年化超额（对准入池等权）及按月重抽区间、绝对年化收益、最大回撤、年化换手、成本拖累、平均持仓。"""
    ret = np.full(len(m.dates), np.nan)
    ret[1:] = res.nav[1:] / res.nav[:-1] - 1
    out = {}
    for period, idx in _days(m, d_first, d_last).items():
        if len(idx) == 0:
            continue
        excess = pd.Series(ret[idx] - pool[idx], index=m.dates[idx])
        nav = res.nav[idx[0] - 1:idx[-1] + 1] / res.nav[idx[0] - 1]
        avg_nav = float(np.mean(res.nav[idx]))
        years = len(idx) / YEAR
        out[period] = {
            "days": int(len(idx)), "excess": _annual(excess),
            "cagr": round((float(nav[-1]) ** (1 / years) - 1) * 100, 2),
            "max_dd": round(float((nav / np.maximum.accumulate(nav) - 1).min()) * 100, 2),
            "turnover": round(float(res.traded[idx].sum()) / 2 / avg_nav / years, 1),
            "cost_drag": round(float(res.costs[idx].sum()) / avg_nav / years * 100, 2),
            "holdings": round(float(res.holdings[idx].mean()), 1),
        }
    all_idx = _days(m, d_first, d_last)["all"]
    excess = pd.Series(ret[all_idx] - pool[all_idx], index=m.dates[all_idx])
    out["yearly_excess"] = {int(y): round(float(s.mean()) * YEAR * 100, 2) for y, s in excess.groupby(excess.index.year)}
    return out


def paired(m: Market, a: SimResult, b: SimResult, d_first: int, d_last: int) -> dict:
    """两个组合逐日收益之差（准入池基准相减抵消），年化及按月重抽区间。"""
    ra, rb = a.nav[1:] / a.nav[:-1] - 1, b.nav[1:] / b.nav[:-1] - 1
    diff = np.concatenate([[np.nan], ra - rb])
    out = {}
    for period, idx in _days(m, d_first, d_last).items():
        if len(idx):
            out[period] = _annual(pd.Series(diff[idx], index=m.dates[idx]))
    return out
