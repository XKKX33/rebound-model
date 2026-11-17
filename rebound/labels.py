"""反弹模型的标签：T+1 买、T+x 卖的逐笔收益，换算成当天准入池里的百分位。

逐笔收益的规则和研究版（`data/research/p0p1-20261029/engine.py` 的 `trades`，carry 口径）相同：

- T+1 买的那一腿没有价格（停牌）记 `ENTRY_MISSING`；那一腿是涨停（开盘买看一字涨停，
  收盘买看收盘涨停）记 `ENTRY_BLOCKED`。两种都不算一笔交易，没有收益。
- T+x 卖的那一腿停牌或跌停，就顺延到下一个开市日的同一腿，最多顺延 `CARRY_MAX` 天；
  还卖不出记 `EXIT_UNRESOLVED`，没有收益。数据到头了还没卖出也是这样。
- 价格用复权价（`model_features` 的 `o` / `c` 列），"往后第 k 天"是往后第 k 个开市日。

百分位在当天的准入池里排（有收益的行才参与，并列取平均名次）。

研究版（十年实验 `train_cadence.build_labels`）先在全部行上排百分位，再丢掉卖出日不早于重训日的行，
于是丢掉的行仍然影响了留下来的行的标签（轻微泄漏）。这里 `percentile(known=...)` 先挑出
"重训日已经知道收益"的行再排，修掉这一点。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from rebound.data.errors import DataError

#: 一笔交易的状态码。
OK, CARRIED, ENTRY_MISSING, ENTRY_BLOCKED, EXIT_UNRESOLVED = range(5)
STATUS: tuple[str, ...] = ("ok", "carried", "entry_missing", "entry_blocked", "exit_unresolved")
#: 卖不出时最多往后顺延几个开市日。
CARRY_MAX = 20
LEGS: tuple[str, ...] = ("open", "close")


@dataclass(frozen=True, slots=True)
class Cell:
    """一种买卖时点：T+1 的 `entry` 腿买，T+`x` 的 `exit` 腿卖。"""

    entry: str
    x: int
    exit: str

    @property
    def name(self) -> str:
        return f"{self.entry[0]}1{self.exit[0]}{self.x}"

    @property
    def period(self) -> int:
        """资金占用的开市日数，也是固定持有规则的批次数：开盘买收盘卖是 x，其余是 x − 1。"""
        return self.x if (self.entry, self.exit) == ("open", "close") else self.x - 1


#: 模型的买卖时点：T+1 开盘买、T+5 开盘卖（o1o5）。十年实验的 150 组持股规则都用这一种打分
#: （2026-10-31 定：o1c5、c1o5 不再训）。
MODEL_CELL = Cell("open", 5, "open")
CELLS: tuple[Cell, ...] = (MODEL_CELL,)


@dataclass(frozen=True)
class PriceMatrix:
    """日 × 股的复权价和涨跌停受阻标记。日期轴是开市日，没有行情的格子价格为 NaN。"""

    dates: pd.DatetimeIndex
    codes: np.ndarray
    price: dict[str, np.ndarray]         # 腿 -> 复权价
    buy_blocked: dict[str, np.ndarray]   # 腿 -> 这一腿涨停买不进
    sell_blocked: dict[str, np.ndarray]  # 腿 -> 这一腿跌停卖不出（停牌另由 NaN 价表示）


def price_matrix(panel: pd.DataFrame, sessions: list[date]) -> PriceMatrix:
    """把面板（`model_features.feature_panel` 的输出，至少要 ts_code、trade_date、o、c 和
    四个受阻标记）排成日 × 股的矩阵。日期轴是面板首尾之间的全部开市日。

    面板首尾之间有开市日没有任何行情就报错：日期轴有洞，"往后第 k 个开市日"会整体错位，
    算出来的是别的日子的收益，而且没有任何症状。
    """
    trade_date = pd.DatetimeIndex(panel["trade_date"])
    calendar = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in sessions))
    dates = calendar[(calendar >= trade_date.min()) & (calendar <= trade_date.max())]
    codes = np.array(sorted(str(c) for c in panel["ts_code"].astype("category").cat.remove_unused_categories()
                            .cat.categories))
    matrix = empty_matrix(dates, codes)
    fill_matrix(matrix, panel)
    check_matrix(matrix)
    return matrix


def empty_matrix(dates: pd.DatetimeIndex, codes: np.ndarray) -> PriceMatrix:
    """先把日 × 股的格子开好（价格 NaN、受阻 False），再按年 `fill_matrix`：
    全历史的面板不必整份拼起来，省一份几百 MB 的拷贝。"""
    shape = (len(dates), len(codes))
    flags = lambda: {leg: np.zeros(shape, dtype=bool) for leg in LEGS}  # noqa: E731
    return PriceMatrix(dates=pd.DatetimeIndex(dates), codes=np.asarray(codes),
                       price={leg: np.full(shape, np.nan) for leg in LEGS}, buy_blocked=flags(), sell_blocked=flags())


def fill_matrix(matrix: PriceMatrix, panel: pd.DataFrame) -> None:
    """把一段面板写进格子（就地写数组：矩阵几百 MB，复制一份不划算）。"""
    if panel.empty:
        return
    di = matrix.dates.get_indexer(pd.DatetimeIndex(panel["trade_date"]))
    codes = panel["ts_code"].astype("category")
    ji = pd.Index(matrix.codes).get_indexer(codes.cat.categories.astype(str))[codes.cat.codes.to_numpy()]
    if (di < 0).any() or (ji < 0).any():
        raise DataError("面板里有日期或代码不在价格矩阵的轴上：轴是按库建的，这说明两次读库之间库变了")
    for leg, price, up, down in (("open", "o", "up_open", "dn_open"), ("close", "c", "up_close", "dn_close")):
        matrix.price[leg][di, ji] = panel[price].to_numpy(dtype=np.float64)
        matrix.buy_blocked[leg][di, ji] = panel[up].to_numpy(dtype=bool)
        matrix.sell_blocked[leg][di, ji] = panel[down].to_numpy(dtype=bool)


def check_matrix(matrix: PriceMatrix) -> None:
    """日期轴上每个开市日都要有行情，否则"往后第 k 个开市日"整体错位。"""
    empty = ~np.isfinite(matrix.price["open"]).any(axis=1) & ~np.isfinite(matrix.price["close"]).any(axis=1)
    if empty.any():
        missing = matrix.dates[empty]
        shown = "、".join(d.date().isoformat() for d in missing[:5])
        raise DataError(f"有 {len(missing)} 个开市日没有任何行情（{shown}…）：日期轴有洞，不能算往后第 k 天")


def carry_trades(m: PriceMatrix, d: np.ndarray, j: np.ndarray, cell: Cell) -> dict[str, np.ndarray]:
    """信号日下标 `d`、股票下标 `j` 的每一笔：收益 `ret`、状态 `status`、实际卖出日下标 `exit_day`。"""
    d, j = np.asarray(d, dtype=np.int64), np.asarray(j, dtype=np.int64)
    if len(d) and (d.min() < 0 or j.min() < 0):
        # `get_indexer` 找不到时给 -1：拿 -1 当下标会读到矩阵最后一行，算出一个假的收益
        raise DataError("信号的日期或代码不在价格矩阵的轴上（下标为 -1）")
    n_days = len(m.dates)
    ret = np.full(len(d), np.nan)
    status = np.full(len(d), EXIT_UNRESOLVED, dtype=np.int8)
    exit_day = np.full(len(d), -1, dtype=np.int64)
    e = d + 1
    inside = e < n_days
    entry_price = np.full(len(d), np.nan)
    entry_price[inside] = m.price[cell.entry][e[inside], j[inside]]
    blocked = np.zeros(len(d), dtype=bool)
    blocked[inside] = m.buy_blocked[cell.entry][e[inside], j[inside]]
    missing = ~np.isfinite(entry_price)
    status[missing] = ENTRY_MISSING
    status[~missing & blocked] = ENTRY_BLOCKED
    pending = ~missing & ~blocked
    for k in range(CARRY_MAX + 1):
        x = d + cell.x + k
        idx = np.flatnonzero(pending & (x < n_days))
        px = m.price[cell.exit][x[idx], j[idx]]
        ok = np.isfinite(px) & ~m.sell_blocked[cell.exit][x[idx], j[idx]]
        done = idx[ok]
        ret[done] = px[ok] / entry_price[done] - 1
        exit_day[done] = x[done]
        status[done] = OK if k == 0 else CARRIED
        pending[done] = False
    return {"ret": ret, "status": status, "exit_day": exit_day}


def percentile(values: np.ndarray, day: np.ndarray, *, known: np.ndarray | None = None) -> np.ndarray:
    """每行在同一天里的百分位（`rank(pct=True)`：并列取平均，空值不参与）。

    `known` 给定时只在这些行里排，其余行为 NaN：训练段只能用段起点之前已经卖出的行，
    没卖出的行既没有标签，也不该影响别人的百分位。
    """
    values = np.asarray(values, dtype=np.float64)
    keep = ~np.isnan(values) if known is None else (~np.isnan(values) & np.asarray(known, dtype=bool))
    out = np.full(len(values), np.nan)
    idx = np.flatnonzero(keep)
    out[idx] = pd.Series(values[idx]).groupby(np.asarray(day)[idx]).rank(pct=True).to_numpy()
    return out