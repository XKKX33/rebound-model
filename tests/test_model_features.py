"""反弹模型的 28 个特征（`rebound/features.py`）。

参照实现 `_reference` 是逐只票用 pandas 写的另一套算法：和正式的 DuckDB 窗口写法对得上，
说明窗口边界、"前 N 根不算"的门槛、复权、停牌和缺复权因子的处理没写错。
和研究版逐格核对另在 `data/research/model-parity-20261031/`（要用真实的研究库）。
"""

from __future__ import annotations

import shutil
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from rebound.config import Universe
from rebound.data.db import WriteStore
from rebound.data.errors import DataError
from rebound.features import FEATURES, LOOKBACK_BARS, eligible, feature_panel
from rebound.store import LookaheadBlocked, Store

SESSIONS = [d for d in (date(2023, 1, 2) + timedelta(days=i) for i in range(620)) if d.weekday() < 5][:420]
#: 代码 -> (名称, 从第几个开市日起有行情, 停牌的开市日下标)
STOCKS = {
    "600001.SH": ("甲股份", 0, ()),
    "600002.SH": ("乙股份", 0, tuple(range(200, 215))),
    "000003.SZ": ("丙股份", 180, ()),
    "600004.SH": ("*ST丁", 0, ()),
    "300005.SZ": ("创业戊", 0, ()),  # 非主板：不进面板
}
NO_FACTOR = ("600001.SH", 300)  # 这一天缺复权因子：和研究版一样整行不进面板
#: 当时的名称：600002 前 100 个开市日叫 ST乙，600004 从第 300 个开市日起才戴帽；600001、000003 没有记录（用现在的名称）。
NAMES = [
    ("600002.SH", "ST乙", SESSIONS[0], SESSIONS[99], SESSIONS[0], "ST"),
    ("600002.SH", "乙股份", SESSIONS[100], None, SESSIONS[98], "撤销ST"),
    ("600004.SH", "丁股份", SESSIONS[0], SESSIONS[299], SESSIONS[0], "其他"),
    ("600004.SH", "*ST丁", SESSIONS[300], None, SESSIONS[298], "*ST"),
]


def _bars(code: str, seed: int) -> list[tuple]:
    _, first, halted = STOCKS[code]
    rng = np.random.default_rng(seed)
    rows, close, factor = [], 8.0 + 4 * seed, 1.0
    for i, day in enumerate(SESSIONS):
        if i < first or i in halted:
            continue
        pre, move = close, float(np.clip(rng.normal(0.001, 0.02), -0.09, 0.09))
        up, down = round(pre * 1.1, 2), round(pre * 0.9, 2)
        close = up if i % 37 == 5 else round(pre * (1 + move), 2)  # 涨停收盘
        open_ = round(pre * (1 + move / 2), 2)
        if i % 53 == 7:
            open_ = close = up  # 一字涨停
        if i == 250:
            factor *= 1.25  # 除权：复权因子跳一档
        amount = float(rng.uniform(5e4, 3e5))
        rows.append((code, day, open_, max(open_, close), min(open_, close), close, pre,
                     (close / pre - 1) * 100, amount / 10, amount, up, down, factor, i))
    return rows


@pytest.fixture(scope="module")
def db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("model") / "market.duckdb"
    rows = [row for seed, code in enumerate(STOCKS) for row in _bars(code, seed)]
    with WriteStore(path) as store:
        con = store.connection
        con.executemany("INSERT INTO trade_cal VALUES ('SSE', ?, 1)", [(d,) for d in SESSIONS])
        con.executemany("INSERT INTO stock_basic VALUES (?, ?, ?, '', ?, '主板', DATE '2000-01-04')",
                        [(c, c[:6], name, f"行业{c[-4]}") for c, (name, _, _) in STOCKS.items()])
        con.executemany("INSERT INTO daily VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [r[:10] for r in rows])
        con.executemany("INSERT INTO daily_limit VALUES (?, ?, ?, ?)", [(r[0], r[1], r[10], r[11]) for r in rows])
        con.executemany("INSERT INTO adj_factor VALUES (?, ?, ?)",
                        [(r[0], r[1], r[12]) for r in rows if (r[0], r[13]) != NO_FACTOR])
        con.executemany("INSERT INTO namechange VALUES (?, ?, ?, ?, ?, ?)", NAMES)
    return path


def _reference(db: Path) -> pd.DataFrame:
    """逐只票用 pandas 算一遍（和正式的 SQL 窗口是两套写法）。"""
    con = duckdb.connect(str(db), read_only=True)
    d = con.execute("""SELECT d.*, a.adj_factor, l.up_limit FROM daily d JOIN adj_factor a USING (ts_code, trade_date)
                       LEFT JOIN daily_limit l USING (ts_code, trade_date)
                       WHERE d.ts_code NOT LIKE '300%' ORDER BY ts_code, trade_date""").df()
    con.close()
    parts = []
    for code, g in d.groupby("ts_code", sort=True):
        g = g.reset_index(drop=True)
        c, bar = g.close * g.adj_factor, pd.Series(np.arange(len(g)))
        lu = ((g.close - g.up_limit).abs() <= 1e-6).astype(float)
        ma60 = c.rolling(60, min_periods=1).mean()
        dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        dea = dif.ewm(span=9, adjust=False).mean()
        launch = ((g.pct_chg >= 5) | (lu == 1)).astype(float).shift(1).rolling(10, min_periods=1).sum()
        parts.append(pd.DataFrame({
            "ts_code": code, "trade_date": pd.to_datetime(g.trade_date), "bar_no": bar,
            "r1": c / c.shift(1) - 1, "r20": c / c.shift(20) - 1,
            "dev_ma5": (c / c.rolling(5, min_periods=1).mean() - 1).where(bar >= 4),
            "from_low60": (c / c.rolling(60, min_periods=1).min() - 1).where(bar >= 59),
            "ma60_slope10": (ma60 / ma60.shift(10) - 1).where(bar >= 69),
            "vol5": (g.amount / g.amount.shift(1).rolling(5, min_periods=1).mean()).where(bar >= 5),
            "launch10": launch.where(bar >= 10),
            "limit_up20": lu.rolling(20, min_periods=1).sum().where(bar >= 19),
            "amp20": ((g.high - g.low) / g.close).rolling(20, min_periods=1).mean().where(bar >= 19),
            "dif_ratio": (dif / c).where(bar >= 33), "hist_ratio": (2 * (dif - dea) / c).where(bar >= 33),
        }))
    ref = pd.concat(parts, ignore_index=True)
    ref["rank_r20"] = ref.groupby("trade_date")["r20"].rank(pct=True)
    return ref


def _panel(db: Path, start: date = SESSIONS[0], end: date = SESSIONS[-1]) -> pd.DataFrame:
    with Store(db) as store:
        got = feature_panel(store, start, end)
    return got.assign(ts_code=got.ts_code.astype(str))


def test_features_match_an_independent_pandas_reference(db: Path) -> None:
    ref, got = _reference(db), _panel(db)
    both = got.merge(ref, on=["ts_code", "trade_date"], suffixes=("", "_ref"), validate="1:1")
    assert len(both) == len(got) == len(ref)
    assert (both.bar_no == both.bar_no_ref).all()
    for col in [c for c in ref.columns if c not in ("ts_code", "trade_date", "bar_no")]:
        want = both[f"{col}_ref"].astype(np.float32).to_numpy()
        assert np.allclose(both[col].to_numpy(), want, rtol=1e-5, atol=1e-6, equal_nan=True), col


def test_every_feature_is_float32_and_the_lookback_is_long_enough() -> None:
    assert len(FEATURES) == 28 and len(set(FEATURES)) == 28
    assert LOOKBACK_BARS >= 250  # 计划：至少 250 根历史，MACD 的指数均线才收敛


def test_scoring_one_day_matches_the_full_computation(db: Path) -> None:
    """每天打分只算当天，训练按段算：两条路必须给出同一份特征。"""
    whole = _panel(db)
    days = [SESSIONS[i] for i in (40, 199, 260, 419)]
    one_day = pd.concat([_panel(db, d, d) for d in days], ignore_index=True)
    chunk = _panel(db, SESSIONS[250], SESSIONS[330])
    for part in (one_day, chunk):
        both = part.merge(whole, on=["ts_code", "trade_date"], suffixes=("", "_whole"), validate="1:1")
        assert len(both) == len(part) > 0
        assert (both.bar_no == both.bar_no_whole).all()
        for col in FEATURES:
            assert part[col].dtype == np.float32, col
            assert np.allclose(both[col], both[f"{col}_whole"], rtol=1e-6, atol=1e-7, equal_nan=True), col


def test_the_panel_keeps_only_mainboard_rows_that_have_a_factor(db: Path) -> None:
    got = _panel(db)
    assert set(got.ts_code) == {"600001.SH", "600002.SH", "000003.SZ", "600004.SH"}
    a = got[got.ts_code == "600001.SH"]
    assert pd.Timestamp(SESSIONS[300]) not in set(a.trade_date) and len(a) == 419
    b = got[got.ts_code == "600002.SH"].reset_index(drop=True)
    assert b.bar_no.tolist() == list(range(405))  # 停牌的 15 天不占根数
    c = got[got.ts_code == "000003.SZ"]
    assert c.bar_no.min() == 0 and c.trade_date.min() == pd.Timestamp(SESSIONS[180])


def test_limit_flags_follow_the_touch_rule(db: Path) -> None:
    a = _panel(db)
    a = a[a.ts_code == "600001.SH"].set_index("trade_date")
    sealed, closed = pd.Timestamp(SESSIONS[7]), pd.Timestamp(SESSIONS[5])
    assert a.loc[sealed, ["up_open", "up_close"]].tolist() == [True, True]
    assert a.loc[closed, ["up_open", "up_close", "limit_up_today"]].tolist() == [False, True, 1.0]
    assert not a.dn_open.any()


def test_eligible_applies_the_strategy_universe_and_the_history_floor(db: Path) -> None:
    got = _panel(db)
    universe = Universe(board="mainboard", exclude_st=True, price_max=70.0, min_amount_yi=0.8, price_min=2.0)
    mask = eligible(got, universe, min_bars=150)
    day = got.trade_date
    st_then = (((got.ts_code == "600002.SH") & (day < pd.Timestamp(SESSIONS[100])))
               | ((got.ts_code == "600004.SH") & (day >= pd.Timestamp(SESSIONS[300]))))
    want = (~st_then & (got.close < 70) & (got.close >= 2) & (got.amount >= 80_000) & (got.bar_no >= 149)).to_numpy()
    assert mask.dtype == bool and (mask == want).all() and 0 < mask.sum() < len(got)
    assert mask[(got.ts_code == "600004.SH").to_numpy()].any()  # 戴帽之前在池里


def test_st_is_judged_by_the_name_at_the_time(db: Path) -> None:
    """600002 前 100 天叫 ST乙、600004 第 300 天起才戴帽；没有改名记录的用现在的名称。"""
    got = _panel(db).set_index(["ts_code", "trade_date"])["name_ok"]
    at = lambda code, i: bool(got.loc[(code, pd.Timestamp(SESSIONS[i]))])  # noqa: E731
    assert (at("600002.SH", 99), at("600002.SH", 100)) == (False, True)
    assert (at("600004.SH", 299), at("600004.SH", 300)) == (True, False)
    assert at("600001.SH", 10) and at("000003.SZ", 200)


def test_a_delisting_period_name_leaves_the_pool_from_its_start_date(db: Path, tmp_path: Path) -> None:
    """退市整理期改名"XX退"（或"退市XX"）：从生效那天起不进池。没有涨跌幅限制，第一天常常腰斩，2026-10-31 之前漏进过池子。"""
    copy = tmp_path / "market.duckdb"
    shutil.copy2(db, copy)
    with WriteStore(copy) as store:
        store.connection.execute("INSERT INTO namechange VALUES ('600001.SH', '甲退', ?, NULL, ?, '退市整理期')",
                                 [SESSIONS[400], SESSIONS[398]])
    got = _panel(copy).set_index(["ts_code", "trade_date"])["name_ok"]
    assert bool(got.loc[("600001.SH", pd.Timestamp(SESSIONS[399]))]) is True
    assert not got.loc["600001.SH"].loc[pd.Timestamp(SESSIONS[400]):].any()


def _without_names(db: Path, tmp_path: Path, where: str) -> Path:
    copy = tmp_path / "market.duckdb"
    shutil.copy2(db, copy)
    with WriteStore(copy) as store:
        store.connection.execute(f"DELETE FROM namechange {where}")
    return copy


def test_without_a_name_record_the_current_name_decides(db: Path, tmp_path: Path) -> None:
    got = _panel(_without_names(db, tmp_path, "WHERE ts_code = '600004.SH'"))
    assert not got[got.ts_code == "600004.SH"].name_ok.any()  # 现在叫 *ST丁
    assert got[got.ts_code == "600001.SH"].name_ok.all()


def test_a_database_without_name_history_is_refused(db: Path, tmp_path: Path) -> None:
    """按现在的名称判 ST 会把后来才戴帽的票提前剔掉（前视）：没有历史名称就不算。"""
    with Store(_without_names(db, tmp_path, "")) as store, pytest.raises(DataError, match="namechange"):
        feature_panel(store, SESSIONS[200], SESSIONS[200])


def test_reading_past_the_visible_window_is_refused(db: Path) -> None:
    with Store(db, visible_through=SESSIONS[300]) as store:
        assert not feature_panel(store, SESSIONS[300], SESSIONS[300]).empty
        with pytest.raises(LookaheadBlocked):
            feature_panel(store, SESSIONS[300], SESSIONS[301])


def test_an_empty_range_gives_an_empty_panel_with_every_column(db: Path) -> None:
    with Store(db) as store:
        got = feature_panel(store, SESSIONS[10], SESSIONS[9])
    assert got.empty and set(FEATURES) <= set(got.columns)
