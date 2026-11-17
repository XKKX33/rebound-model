"""逐笔收益引擎：买卖时点格子、两种卖出受阻口径（drop / carry）、准入池基准、行业去重。

价格用复权开收盘（研究库复权因子），受阻规则见 PREREG.md。日期 × 股票的矩阵按研究库的开市日排好，
偏移 k 就是往后第 k 个开市日。只读 panel.parquet 和研究库的 stock_basic。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PANEL = HERE / "panel.parquet"
TEST_FROM, DEV_END = pd.Timestamp("2022-09-01"), pd.Timestamp("2024-06-28")
CARRY_MAX = 20
LEGS = ("open", "close")
#: 单边成本（买, 卖），按成交额。
COSTS = {"low": (0.0005, 0.0015), "mid": (0.0015, 0.0025), "high": (0.0030, 0.0030)}

OK, CARRIED, ENTRY_MISSING, ENTRY_BLOCKED, EXIT_DROPPED, EXIT_UNRESOLVED = range(6)
STATUS = ("ok", "carried", "entry_missing", "entry_blocked", "exit_dropped", "exit_unresolved")


@dataclass(frozen=True)
class Cell:
    """T+1 的 entry 腿买，T+x 的 exit 腿卖。"""

    entry: str
    x: int
    exit: str

    @property
    def name(self) -> str:
        return f"{self.entry[0]}1{self.exit[0]}{self.x}"

    @property
    def period(self) -> int:
        """资金占用的开市日数，也是 FH 的批次数：开盘买收盘卖是 x，其余是 x−1。"""
        return self.x if (self.entry, self.exit) == ("open", "close") else self.x - 1


CELLS = tuple(Cell(e, x, leg) for e in LEGS for x in (2, 3, 5) for leg in LEGS)
BASELINE = Cell("open", 5, "open")


@dataclass
class Market:
    dates: pd.DatetimeIndex
    codes: np.ndarray
    price: dict[str, np.ndarray]         # 腿 → 复权价（没有行情为 NaN）
    buy_blocked: dict[str, np.ndarray]   # 腿 → 这一腿买不进
    sell_blocked: dict[str, np.ndarray]  # 腿 → 这一腿卖不出（停牌另由 NaN 价表示）
    eligible: np.ndarray                 # 当天收盘在准入池里
    industry: np.ndarray                 # 每只股票的行业编号
    raw: dict[str, np.ndarray] | None = None  # 腿 → 未复权价（真实账户算手数和费用用；见 attach_raw）

    def code_index(self, ts_code) -> np.ndarray:
        return pd.Index(self.codes).get_indexer(np.asarray(ts_code))

    def date_index(self, trade_date) -> np.ndarray:
        return self.dates.get_indexer(pd.DatetimeIndex(trade_date))

    @property
    def signal_range(self) -> tuple[int, int]:
        """信号日下标 [first, last]：2022-09-01 起，到 T+5 还在数据里的最后一天。"""
        return int(self.dates.searchsorted(TEST_FROM)), len(self.dates) - 6


def load_market(path: Path = PANEL) -> Market:
    cols = ["ts_code", "trade_date", "o", "c", "up_open", "up_close", "dn_open", "dn_close", "eligible"]
    p = pd.read_parquet(path, columns=cols, read_dictionary=["ts_code"])
    codes = np.array(sorted(p.ts_code.cat.categories))
    ci = pd.Categorical(p.ts_code, categories=codes).codes
    dates = pd.DatetimeIndex(np.sort(p.trade_date.unique())).astype("datetime64[ns]")
    di = dates.get_indexer(p.trade_date.astype("datetime64[ns]"))
    shape = (len(dates), len(codes))

    def mat(col: str, fill, dtype) -> np.ndarray:
        out = np.full(shape, fill, dtype=dtype)
        out[di, ci] = p[col].to_numpy()
        return out

    con = duckdb.connect(str(ROOT / "data/research.duckdb"), read_only=True)
    try:
        basic = con.execute("SELECT ts_code, industry FROM stock_basic").df()
    finally:
        con.close()
    ind_of = dict(zip(basic.ts_code, basic.industry.fillna("")))
    names = {name: i for i, name in enumerate(sorted({ind_of.get(c, "") for c in codes}))}
    return Market(dates=dates, codes=codes,
                  price={"open": mat("o", np.nan, np.float64), "close": mat("c", np.nan, np.float64)},
                  buy_blocked={"open": mat("up_open", False, bool), "close": mat("up_close", False, bool)},
                  sell_blocked={"open": mat("dn_open", False, bool), "close": mat("dn_close", False, bool)},
                  eligible=mat("eligible", False, bool),
                  industry=np.array([names[ind_of.get(c, "")] for c in codes], dtype=np.int32))

def attach_raw(m: Market) -> Market:
    """从研究库读未复权开收盘，排成和 m 一样的 日 × 股 矩阵，返回带 raw 的新 Market。"""
    from dataclasses import replace
    con = duckdb.connect(str(ROOT / "data/research.duckdb"), read_only=True)
    try:
        con.register("c_idx", pd.DataFrame({"ts_code": m.codes, "ji": np.arange(len(m.codes))}))
        con.register("d_idx", pd.DataFrame({"trade_date": m.dates.date, "di": np.arange(len(m.dates))}))
        got = con.execute("SELECT d_idx.di, c_idx.ji, daily.open, daily.close FROM daily "
                          "JOIN c_idx USING (ts_code) JOIN d_idx ON CAST(daily.trade_date AS DATE) = d_idx.trade_date"
                          ).fetchnumpy()
    finally:
        con.close()
    raw = {leg: np.full((len(m.dates), len(m.codes)), np.nan) for leg in LEGS}
    raw["open"][got["di"], got["ji"]] = got["open"]
    raw["close"][got["di"], got["ji"]] = got["close"]
    return replace(m, raw=raw)


def trades(m: Market, d: np.ndarray, j: np.ndarray, cell: Cell, convention: str = "carry") -> dict[str, np.ndarray]:
    """信号日 d、股票 j 的一笔交易：T+1 的 entry 腿买，T+x 的 exit 腿卖。

    drop：卖出那一腿受阻或停牌，这笔记 exit_dropped，不进均值（现有实验和正式回填的做法）。
    carry：之后每天同一腿重试，最多 CARRY_MAX 天，按实际卖出价算。
    """
    d, j = np.asarray(d, dtype=np.int64), np.asarray(j, dtype=np.int64)
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
    for k in range(CARRY_MAX + 1 if convention == "carry" else 1):
        x = d + cell.x + k
        live = pending & (x < n_days)
        idx = np.flatnonzero(live)
        px = m.price[cell.exit][x[idx], j[idx]]
        ok = np.isfinite(px) & ~m.sell_blocked[cell.exit][x[idx], j[idx]]
        done = idx[ok]
        ret[done] = px[ok] / entry_price[done] - 1
        exit_day[done] = x[done]
        status[done] = OK if k == 0 else CARRIED
        pending[done] = False
        if convention == "drop":
            status[idx[~ok]] = EXIT_DROPPED
            pending[idx[~ok]] = False
    return {"ret": ret, "status": status, "exit_day": exit_day}


def pool_mean(m: Market, days: np.ndarray, cell: Cell, convention: str = "carry") -> np.ndarray:
    """每个信号日准入池等权的逐笔收益（同一口径、买不进和没结果的不算）。返回长度为日数的数组。"""
    dd, jj = np.nonzero(m.eligible[days])
    dd = np.asarray(days)[dd]
    r = trades(m, dd, jj, cell, convention)["ret"]
    ok = np.isfinite(r)
    sums = np.bincount(dd[ok], weights=r[ok], minlength=len(m.dates))
    counts = np.bincount(dd[ok], minlength=len(m.dates))
    with np.errstate(invalid="ignore"):
        out = np.where(counts > 0, sums / counts, np.nan)
    return out


def pool_daily_return(m: Market) -> np.ndarray:
    """组合基准：昨天收盘在准入池里、今天也有收盘价的票，等权收盘对收盘收益。不可交易、不扣成本。"""
    c = m.price["close"]
    out = np.full(len(m.dates), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = c[1:] / c[:-1] - 1
    ok = m.eligible[:-1] & np.isfinite(r)
    out[1:] = np.where(ok.any(axis=1), np.where(ok, r, 0).sum(axis=1) / np.maximum(ok.sum(axis=1), 1), np.nan)
    return out


def pool_legs(m: Market) -> dict[str, np.ndarray]:
    """准入池（昨天收盘在池里、今天开收盘都有）等权的三段日收益：收盘对收盘、隔夜（昨收 → 今开）、日内（今开 → 今收）。"""
    o, c = m.price["open"], m.price["close"]
    out = {k: np.full(len(m.dates), np.nan) for k in ("cc", "night", "intra")}
    with np.errstate(invalid="ignore", divide="ignore"):
        legs = {"cc": c[1:] / c[:-1] - 1, "night": o[1:] / c[:-1] - 1, "intra": c[1:] / o[1:] - 1}
    ok = m.eligible[:-1] & np.isfinite(legs["cc"]) & np.isfinite(legs["night"]) & np.isfinite(legs["intra"])
    n = ok.sum(axis=1)
    for k, r in legs.items():
        out[k][1:] = np.where(n > 0, np.where(ok, r, 0).sum(axis=1) / np.maximum(n, 1), np.nan)
    return out


def matched_pool(legs: dict[str, np.ndarray], kind: str, cell: Cell) -> np.ndarray:
    """时点匹配基准：准入池按组合同样的作息持有。

    FH 开盘买、收盘卖：每天 P 个批次里有 1 个隔夜空仓 → (P−1)/P 收盘对收盘 + 1/P 日内。
    FH 收盘买、开盘卖：每天有 1 个批次日内空仓 → (P−1)/P 收盘对收盘 + 1/P 隔夜。
    其余（同一腿买卖的 FH、全部 DB）一直满仓 → 收盘对收盘。
    """
    p = cell.period
    if kind == "FH" and (cell.entry, cell.exit) == ("open", "close"):
        return (p - 1) / p * legs["cc"] + legs["intra"] / p
    if kind == "FH" and (cell.entry, cell.exit) == ("close", "open"):
        return (p - 1) / p * legs["cc"] + legs["night"] / p
    return legs["cc"]


def pool_daily_return_carry(m: Market) -> np.ndarray:
    """准入池基准的另一种算法（只作核对）：停牌的票按最后收盘价计值留在池里，复牌那天把缺口算进去。
    "在池里"按停牌前最后一个交易日的准入状态延续；最后一个交易日之后不再计入。"""
    c = m.price["close"]
    have = np.isfinite(c)
    idx = np.where(have, np.arange(len(m.dates))[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    cols = np.arange(c.shape[1])
    filled = np.where(np.maximum.accumulate(have, axis=0), c[idx, cols], np.nan)
    elig = m.eligible[idx, cols] & np.maximum.accumulate(have, axis=0)
    last = len(m.dates) - 1 - np.argmax(have[::-1], axis=0)
    alive = np.arange(len(m.dates))[:, None] <= last[None, :]
    out = np.full(len(m.dates), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = filled[1:] / filled[:-1] - 1
    ok = elig[:-1] & alive[1:] & np.isfinite(r)
    out[1:] = np.where(ok.any(axis=1), np.where(ok, r, 0).sum(axis=1) / np.maximum(ok.sum(axis=1), 1), np.nan)
    return out


def dedup_top(order: np.ndarray, industry: np.ndarray, k: int = 6, cap: int = 2) -> list[int]:
    """按给定顺序，每个行业最多 cap 只，取满 k 只（与正式 dedup_and_top 同一逻辑）。"""
    picks, count = [], {}
    for j in order:
        g = industry[j]
        if count.get(g, 0) >= cap:
            continue
        picks.append(int(j))
        count[g] = count.get(g, 0) + 1
        if len(picks) >= k:
            break
    return picks


def month_ci(daily: pd.Series, n_boot: int = 2000, seed: int = 20261029) -> list[float]:
    """日均值按月整块重抽的 95% 区间（与 stage_strategy.month_ci 同法，返回小数）。"""
    daily = daily.dropna()
    if daily.empty:
        return [float("nan"), float("nan")]
    month = daily.index.to_period("M")
    sums, counts = daily.groupby(month).sum().to_numpy(), daily.groupby(month).count().to_numpy()
    pick = np.random.default_rng(seed).integers(0, len(sums), size=(n_boot, len(sums)))
    boots = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return [float(q) for q in np.percentile(boots, [2.5, 97.5])]


def register(stage: str, config: dict, metrics: dict) -> None:
    """试验登记：每次运行追加一行（时间、阶段、配置指纹、配置、主要指标），供以后按试验次数打折扣。"""
    import hashlib
    import json
    import time
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    rec = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage,
           "fingerprint": hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12], "config": config, "metrics": metrics}
    with (HERE / "registry.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def candidates(m: Market, scores: pd.DataFrame) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """scores（列 di, ji, score）→ 每个信号日按分数降序的股票下标，以及 日 × 股 的排名矩阵（1 起，没分为 inf）。"""
    s = scores.dropna(subset=["score"]).sort_values(["di", "score", "ji"], ascending=[True, False, True],
                                                    kind="mergesort")
    rank = np.full((len(m.dates), len(m.codes)), np.inf, dtype=np.float32)
    rank[s.di.to_numpy(), s.ji.to_numpy()] = s.groupby("di").cumcount().to_numpy() + 1
    cands = {int(d): g.ji.to_numpy() for d, g in s.groupby("di", sort=True)}
    return cands, rank
