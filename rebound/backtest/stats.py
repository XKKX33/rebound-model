"""本金测算的统计：一段净值的收益、超额（按月重抽区间）、β、回撤；逐年；换起点；四种选法和滚动样本外。

全部原样取自十年实验（`decade_sim.period` / `by_year` / `scoring_days`、`rule_search.summary`、
`fresh_start.window`、`rule_select.scores` / `walk_forward` / `series_stats` / `best_of`、`engine.month_ci`），
数字逐位相同。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rebound.backtest.rules import Spec, neighbors
from rebound.backtest.sim import WARMUP, YEAR, SimResult

#: 滚动样本外：每年年初只用之前的数据选规则（2017 ~ 2026）。
TEST_YEARS = tuple(range(2017, 2027))
#: 四种选法：平滑后超额、不平滑的超额、信息比率、平滑后复利增长。
PICKERS = ("smooth", "raw", "ir", "growth")
#: 滚动样本外超额差不到这么多（百分点）算打平，打平时比区间下限。
TIE = 0.5


def month_ci(daily: pd.Series, n_boot: int = 2000, seed: int = 20261029) -> list[float]:
    """日均值按月整块重抽的 95% 区间（返回小数）。"""
    daily = daily.dropna()
    if daily.empty:
        return [float("nan"), float("nan")]
    month = daily.index.to_period("M")
    sums, counts = daily.groupby(month).sum().to_numpy(), daily.groupby(month).count().to_numpy()
    pick = np.random.default_rng(seed).integers(0, len(sums), size=(n_boot, len(sums)))
    boots = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return [float(q) for q in np.percentile(boots, [2.5, 97.5])]


def scoring_days(d_first: int, d_last: int) -> np.ndarray:
    """计分的净值日：跳过建仓期。"""
    return np.arange(d_first + 1 + WARMUP, d_last + 2)


def period(nav: np.ndarray, pool: np.ndarray, dates: pd.DatetimeIndex, idx: np.ndarray) -> dict:
    """一段净值日 idx：总收益、年化、超额（年化 [按月重抽区间]）、β、α（年化）、最大回撤。"""
    ret = nav[idx] / nav[idx - 1] - 1
    bench = pool[idx]
    ok = np.isfinite(ret) & np.isfinite(bench)
    ret, bench, when = ret[ok], bench[ok], dates[idx][ok]
    lo, hi = month_ci(pd.Series(ret - bench, index=when))
    beta = float(np.cov(ret, bench)[0, 1] / np.var(bench, ddof=1)) if len(ret) > 2 else float("nan")
    seg = nav[idx[0] - 1:idx[-1] + 1] / nav[idx[0] - 1]
    years = len(idx) / YEAR
    return {"days": int(len(idx)), "total": round((float(seg[-1]) - 1) * 100, 2),
            "cagr": round((float(seg[-1]) ** (1 / years) - 1) * 100, 2),
            "excess": round(float((ret - bench).mean()) * YEAR * 100, 2),
            "excess_ci": [round(lo * YEAR * 100, 2), round(hi * YEAR * 100, 2)],
            "beta": round(beta, 3), "alpha": round(float((ret - beta * bench).mean()) * YEAR * 100, 2),
            "max_dd": round(float((seg / np.maximum.accumulate(seg) - 1).min()) * 100, 2)}


def by_year(nav: np.ndarray, pool: np.ndarray, dates: pd.DatetimeIndex, idx: np.ndarray) -> dict:
    years = dates[idx].year
    return {str(y): period(nav, pool, dates, idx[years == y]) for y in np.unique(years)}


def summary(res: SimResult, pool: np.ndarray, dates: pd.DatetimeIndex, idx: np.ndarray) -> dict:
    """一组规则的全期数字、逐年超额和收益，以及交易量、费用、空仓。"""
    nav = res.nav
    out = period(nav, pool, dates, idx)
    years = by_year(nav, pool, dates, idx)
    out["years"] = {y: {"total": v["total"], "excess": v["excess"]} for y, v in years.items()}
    out["final"] = round(float(nav[idx[-1]]), 0)
    trades = res.trades
    out["buys_per_day"] = round(len(trades) / len(idx), 2)
    out["hold_days"] = round(float((trades.exit_day - trades.entry_day).mean()), 1) if len(trades) else None
    span = nav[idx].mean()
    out["turnover"] = round(float(res.traded[idx].sum() / span / (len(idx) / YEAR)) / 2, 1)
    out["cost_pct"] = round(float(res.costs[idx].sum() / span / (len(idx) / YEAR)) * 100, 2)
    out["idle_cash_pct"] = round(float(np.nanmean(res.cash[idx] / nav[idx])) * 100, 1)
    out["avg_holdings"] = round(float(res.holdings[idx].mean()), 1)
    out["poor_buys"] = int(res.poor_buys)
    return out


def window(nav: np.ndarray, pool: np.ndarray, d0: int, d_end: int, dates: pd.DatetimeIndex) -> dict:
    """d0 收盘起步到 d_end：头一年收益、年化、最大回撤、翻了几倍、准入池同期年化（换起点体检）。"""
    seg = nav[d0:d_end + 1] / nav[d0]
    year_end = min(int(dates.searchsorted(pd.Timestamp(dates[d0].year + 1, 1, 1))) - 1, d_end)
    years = (d_end - d0) / YEAR
    pool_nav = np.cumprod(1 + np.nan_to_num(pool[d0 + 1:d_end + 1]))
    return {"first_year": round(float(nav[year_end] / nav[d0] - 1) * 100, 1),
            "cagr": round((float(seg[-1]) ** (1 / years) - 1) * 100, 1),
            "max_dd": round(float((seg / np.maximum.accumulate(seg) - 1).min()) * 100, 1),
            "final_x": round(float(seg[-1]), 2), "pool_cagr": round((float(pool_nav[-1]) ** (1 / years) - 1) * 100, 1)}


# --------------------------------------------------------------------------- #
# 选法（十年实验 rule_select.py）
# --------------------------------------------------------------------------- #


def annual(x: np.ndarray, axis: int = -1):
    return x.mean(axis=axis) * YEAR * 100


def neighbor_index(configs: list[tuple[str, Spec]]) -> list[np.ndarray]:
    """每组的网格邻居（同一频率、同一类，含自己）在 configs 里的下标。"""
    where = {c: i for i, c in enumerate(configs)}
    by_cadence: dict[str, set[Spec]] = {}
    for cadence, spec in configs:
        by_cadence.setdefault(cadence, set()).add(spec)
    return [np.array([where[(cadence, s)] for s in neighbors(spec, by_cadence[cadence])]) for cadence, spec in configs]


def scores(book: np.ndarray, excess: np.ndarray, near: list[np.ndarray]) -> dict[str, np.ndarray]:
    """四种选法各自的打分（越大越好）。book、excess 是 [组, 日] 的逐日收益和超额。"""
    mean = excess.mean(axis=1)
    sd = excess.std(axis=1, ddof=1)
    smooth = lambda score: np.array([score[ix].mean() for ix in near])  # noqa: E731
    return {"smooth": smooth(mean), "raw": mean, "ir": mean / np.where(sd > 0, sd, np.inf),
            "growth": smooth(np.log1p(book).mean(axis=1))}


def walk_forward(book, excess, dates: pd.DatetimeIndex, near, picker: str, years=TEST_YEARS) -> dict:
    """每年年初只用之前的数据选一组，记它这一年的逐日收益，拼成一条样本外序列。"""
    picks, b_oos, e_oos, when = {}, [], [], []
    for y in years:
        past, now = dates.year < y, dates.year == y
        if not now.any() or not past.any():
            continue
        i = int(np.argmax(scores(book[:, past], excess[:, past], near)[picker]))
        picks[str(y)] = i
        b_oos.append(book[i, now])
        e_oos.append(excess[i, now])
        when.append(dates[now])
    return {"picks": picks, "book": np.concatenate(b_oos), "excess": np.concatenate(e_oos),
            "dates": when[0].append(when[1:]) if len(when) > 1 else when[0]}


def series_stats(book: np.ndarray, excess: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    """一条逐日序列：年化超额 [按月重抽区间]、年化收益、最大回撤、逐年超额。"""
    lo, hi = month_ci(pd.Series(excess, index=dates))
    nav = np.cumprod(1 + book)
    peak = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
    return {"excess": round(float(annual(excess)), 2), "ci": [round(lo * YEAR * 100, 2), round(hi * YEAR * 100, 2)],
            "cagr": round((float(nav[-1]) ** (YEAR / len(nav)) - 1) * 100, 2),
            "max_dd": round(float((nav / peak - 1).min()) * 100, 2),
            "years": {str(y): round(float(annual(excess[dates.year == y])), 2) for y in np.unique(dates.year)}}


def best_of(walked: dict) -> str:
    """滚动样本外超额最高的选法；差不到 TIE 的算打平，打平时取区间下限高的。"""
    top = max(w["excess"] for w in walked.values())
    tied = [p for p in walked if walked[p]["excess"] >= top - TIE]
    return max(tied, key=lambda p: walked[p]["ci"][0])
