"""本金测算用的日 × 股矩阵：复权价、不复权价（算手数和费用）、涨跌停受阻、准入池、行业。

价格、受阻标记、准入池都取自 `model_features.feature_panel` / `eligible`：和反弹模型训练、打分同一张表，
所以模拟器的准入池基准和模型的准入池逐行相同（和十年实验的 `common.load_market10` 逐格核对过）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from rebound import params as model_params
from rebound.features import eligible, feature_panel
from rebound.labels import LEGS
from rebound.universe import NON_MAIN_PREFIXES
from rebound.model.artifacts import ModelError
from rebound.model.train import data_sessions, open_store, year_spans

_NON_MAIN = "^(" + "|".join(NON_MAIN_PREFIXES) + ")"


@dataclass(frozen=True)
class Market:
    dates: pd.DatetimeIndex
    codes: np.ndarray
    price: dict[str, np.ndarray]         # 腿 → 复权价（没有行情为 NaN）
    buy_blocked: dict[str, np.ndarray]   # 腿 → 这一腿买不进（涨停）
    sell_blocked: dict[str, np.ndarray]  # 腿 → 这一腿卖不出（跌停；停牌另由 NaN 价表示）
    eligible: np.ndarray                 # 当天收盘在准入池里
    industry: np.ndarray                 # 每只股票的行业编号（同一个行业名同一个号，缺行业的算一组）
    raw: dict[str, np.ndarray] | None = None  # 腿 → 不复权价

    def code_index(self, ts_code) -> np.ndarray:
        return pd.Index(self.codes).get_indexer(np.asarray(ts_code))


def load_markets(db_path: Path, *, start: date | None = None, end: date | None = None,
                 log: Callable[[str], None] | None = None, stop: Any = None) -> dict[str, Market]:
    """两个准入池版本的矩阵（价格、受阻、行业共用一份，只有准入池不同）。按年算特征，省内存。

    `start`：只读这一天所在的那年起的行情（跟踪一条规则不用读全部历史）。从那年 1 月 1 日读起，切法和全量读的
    那几年一样；复权价、受阻、准入池都是逐行算的（`bar_no` 按全历史），所以矩阵里每个格子和全量读逐位相同。
    `stop`：后台任务的句柄（有 `checkpoint()`），每读完一年看一次要不要停：读全部历史要一两分钟。
    """
    say = log or (lambda _line: None)
    lo = None if start is None else date(start.year, 1, 1)
    with open_store(db_path) as store:
        sessions = [d for d in data_sessions(store) if (end is None or d <= end) and (lo is None or d >= lo)]
        if not sessions:
            raise ModelError(f"{lo} ~ {end} 之间库里没有开市日的行情")
        codes = np.array([c for (c,) in store.connection.execute(
            f"SELECT DISTINCT ts_code FROM daily WHERE NOT regexp_matches(ts_code, '{_NON_MAIN}') ORDER BY 1").fetchall()])
        basic = store.connection.execute("SELECT ts_code, industry FROM stock_basic").df()
        dates = pd.DatetimeIndex([pd.Timestamp(d) for d in sessions])
        shape = (len(dates), len(codes))
        flags = lambda: {leg: np.zeros(shape, dtype=bool) for leg in LEGS}  # noqa: E731
        price, raw = ({leg: np.full(shape, np.nan) for leg in LEGS} for _ in range(2))
        up, down = flags(), flags()
        pools = {v: np.zeros(shape, dtype=bool) for v in model_params.VARIANTS}
        index = pd.Index(codes)
        for lo, hi in year_spans(sessions[0], sessions[-1]):
            panel = feature_panel(store, lo, hi)
            di = dates.get_indexer(pd.DatetimeIndex(panel["trade_date"]))
            cat = panel["ts_code"].astype("category")
            ji = index.get_indexer(cat.cat.categories.astype(str))[cat.cat.codes.to_numpy()]
            if (di < 0).any() or (ji < 0).any():
                raise ValueError("面板里有日期或代码不在矩阵的轴上：两次读库之间库变了")
            for leg, adj, plain, u, d in (("open", "o", "open", "up_open", "dn_open"),
                                          ("close", "c", "close", "up_close", "dn_close")):
                price[leg][di, ji] = panel[adj].to_numpy(dtype=np.float64)
                raw[leg][di, ji] = panel[plain].to_numpy(dtype=np.float64)
                up[leg][di, ji] = panel[u].to_numpy(dtype=bool)
                down[leg][di, ji] = panel[d].to_numpy(dtype=bool)
            for variant, universe in model_params.VARIANTS.items():
                pools[variant][di, ji] = eligible(panel, universe, min_bars=model_params.MIN_BARS)
            say(f"[行情] {lo.year} 年：{len(panel)} 行")
            del panel
            if stop is not None:
                stop.checkpoint()
    base = Market(dates=dates, codes=codes, price=price, buy_blocked=up, sell_blocked=down,
                  eligible=pools[model_params.DEFAULT_VARIANT], industry=industries(codes, basic), raw=raw)
    return {variant: replace(base, eligible=pool) for variant, pool in pools.items()}


def stock_names(db_path: Path) -> dict[str, tuple[str, str]]:
    """代码 → （名称, 行业），清单和持仓上显示用。"""
    with open_store(db_path) as store:
        basic = store.connection.execute("SELECT ts_code, name, industry FROM stock_basic").df()
    return {c: ("" if pd.isna(n) else str(n), "" if pd.isna(i) else str(i))
            for c, n, i in zip(basic.ts_code, basic.name, basic.industry)}


def industries(codes: np.ndarray, basic: pd.DataFrame) -> np.ndarray:
    """行业名 → 编号（按名称排序编号；缺行业的是空串，算一组：和正式选股的行业去重同一口径）。"""
    ind_of = dict(zip(basic.ts_code, basic.industry.fillna("")))
    names = {name: i for i, name in enumerate(sorted({ind_of.get(c, "") for c in codes}))}
    return np.array([names[ind_of.get(c, "")] for c in codes], dtype=np.int32)


def model_scores(db_path: Path, m: Market, variant: str, cadence: str) -> pd.DataFrame:
    """`model_score` 里这个版本、这种重训频率的分数，排成 (di, ji, score)：日期、代码都在 `m` 的轴上。"""
    with open_store(db_path) as store:
        frame = store.connection.execute(
            "SELECT trade_date, ts_code, score FROM model_score WHERE variant = ? AND cadence = ? "
            "AND trade_date BETWEEN ? AND ?", [variant, cadence, m.dates[0].date(), m.dates[-1].date()]).df()
    di = m.dates.get_indexer(pd.DatetimeIndex(frame["trade_date"]))
    ji = m.code_index(frame["ts_code"])
    if (di < 0).any() or (ji < 0).any():
        raise ValueError("model_score 里有日期或代码不在行情矩阵的轴上")
    out = pd.DataFrame({"di": di.astype(np.int64), "ji": ji.astype(np.int64), "score": frame["score"].to_numpy()})
    repeated = int(out.duplicated(["di", "ji"]).sum())
    if repeated:  # 不拦的话，同一只票会被当成两只候选，一天买两次
        raise ModelError(f"model_score 里{variant} {cadence} 有 {repeated} 行是同一天同一只票的重复分"
                         "（两个写库任务同时补过分）：删掉那几天这条序列的分，再跑 rebound model score")
    return out


def candidates(m: Market, scores: pd.DataFrame) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """scores（列 di, ji, score）→ 每个信号日按分数降序的股票下标，以及 日 × 股 的排名矩阵（1 起，没分为 inf）。"""
    s = scores.dropna(subset=["score"]).sort_values(["di", "score", "ji"], ascending=[True, False, True],
                                                    kind="mergesort")
    rank = np.full((len(m.dates), len(m.codes)), np.inf, dtype=np.float32)
    rank[s.di.to_numpy(), s.ji.to_numpy()] = s.groupby("di").cumcount().to_numpy() + 1
    cands = {int(d): g.ji.to_numpy() for d, g in s.groupby("di", sort=True)}
    return cands, rank


def pool_daily_return(m: Market) -> np.ndarray:
    """组合基准：昨天收盘在准入池里、今天也有收盘价的票，等权收盘对收盘收益。不可交易、不扣成本。"""
    c = m.price["close"]
    out = np.full(len(m.dates), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = c[1:] / c[:-1] - 1
    ok = m.eligible[:-1] & np.isfinite(r)
    out[1:] = np.where(ok.any(axis=1), np.where(ok, r, 0).sum(axis=1) / np.maximum(ok.sum(axis=1), 1), np.nan)
    return out
