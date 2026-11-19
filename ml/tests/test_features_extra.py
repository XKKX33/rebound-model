"""任务 1.4 新特征的单元测试：在一个手工构造的小库上，逐个核对特征的计算和单位换算，并检查没有前视。

运行：python -X utf8 -m pytest -p no:cacheprovider -o addopts="" -q ml/tests
"""

from __future__ import annotations

from datetime import date

import duckdb
import numpy as np
import pandas as pd
import pytest

from ml.features_extra import BASE_NAMES, add_ranks, compute_range

N = 270                                  # 交易日数（> 250，才能算 250 日特征）
CODES = ("600001.SH", "600002.SH")       # 两只主板股票
GEM = "300001.SZ"                        # 一只创业板，不应出现在结果里


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2020-01-01", periods=N)
    rows, adj, lim, basic, mf = [], [], [], [], []
    for k, code in enumerate(CODES + (GEM,)):
        close = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, N)))
        open_ = close * np.exp(rng.normal(0, 0.005, N))
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, N))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, N))
        factor = np.where(np.arange(N) >= 100, 1.1, 1.0)       # 第 100 天除权：不复权价下跳，复权因子上调
        close_raw, open_raw = close / factor, open_ / factor
        high_raw, low_raw = high / factor, low / factor
        pre = np.r_[close_raw[0], close_raw[:-1]]
        for i, d in enumerate(days):
            rows.append((code, d.date(), open_raw[i], high_raw[i], low_raw[i], close_raw[i], pre[i],
                         (close_raw[i] / pre[i] - 1) * 100, 1000.0, 2000.0 + 10 * i))     # amount 千元
            adj.append((code, d.date(), factor[i]))
            dn = close_raw[i] if (code == CODES[0] and i in (200, 205)) else close_raw[i] * 0.9
            lim.append((code, d.date(), pre[i] * 1.1, dn))
            basic.append((code, d.date(), 1.0, 1.0 + 0.01 * i, 1.5, 1e4, 8e3, 4e3, 1e6, 5e5 * (k + 1)))
            mf.append((code, d.date(), 100.0, 30.0, 10.0, 20.0, 5.0))   # 万元：净流入 100；大单买 30 卖 10，特大单买 20 卖 5
    con = duckdb.connect(str(tmp_path_factory.mktemp("db") / "w.duckdb"))
    con.execute("CREATE TABLE daily(ts_code VARCHAR, trade_date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, "
                "pre_close DOUBLE, pct_chg DOUBLE, vol DOUBLE, amount DOUBLE)")
    con.executemany("INSERT INTO daily VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    con.execute("CREATE TABLE adj_factor(ts_code VARCHAR, trade_date DATE, adj_factor DOUBLE)")
    con.executemany("INSERT INTO adj_factor VALUES (?,?,?)", adj)
    con.execute("CREATE TABLE daily_limit(ts_code VARCHAR, trade_date DATE, up_limit DOUBLE, down_limit DOUBLE)")
    con.executemany("INSERT INTO daily_limit VALUES (?,?,?,?)", lim)
    con.execute("CREATE TABLE daily_basic(ts_code VARCHAR, trade_date DATE, turnover_rate DOUBLE, turnover_rate_f DOUBLE, "
                "volume_ratio DOUBLE, total_share DOUBLE, float_share DOUBLE, free_share DOUBLE, total_mv DOUBLE, circ_mv DOUBLE)")
    con.executemany("INSERT INTO daily_basic VALUES (?,?,?,?,?,?,?,?,?,?)", basic)
    con.execute("CREATE TABLE moneyflow(ts_code VARCHAR, trade_date DATE, net_mf_amount DOUBLE, buy_lg_amount DOUBLE, "
                "sell_lg_amount DOUBLE, buy_elg_amount DOUBLE, sell_elg_amount DOUBLE)")
    con.executemany("INSERT INTO moneyflow VALUES (?,?,?,?,?,?,?)", mf)
    con.execute("CREATE TABLE stock_basic(ts_code VARCHAR, list_date DATE)")
    con.executemany("INSERT INTO stock_basic VALUES (?,?)", [(c, date(2019, 1, 1)) for c in CODES + (GEM,)])
    idx_close = 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, N + 30)))
    idx_days = pd.bdate_range(end=days[-1], periods=N + 30)
    con.execute("CREATE TABLE index_daily(ts_code VARCHAR, trade_date DATE, close DOUBLE, pct_chg DOUBLE)")
    con.executemany("INSERT INTO index_daily VALUES (?,?,?,?)",
                    [("000852.SH", d.date(), float(c), 0.0) for d, c in zip(idx_days, idx_close)])
    raw = pd.DataFrame(rows, columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "pct_chg",
                                      "vol", "amount"])
    raw["adj"] = [a[2] for a in adj]
    feats = compute_range(con, days[0].date(), days[-1].date())
    yield {"con": con, "days": days, "raw": raw, "feats": feats,
           "idx": pd.Series(idx_close, index=idx_days.date)}
    con.close()


def _one(world, code=CODES[0]):
    f = world["feats"]
    f = f[f["ts_code"] == code].reset_index(drop=True)
    r = world["raw"]
    r = r[r["ts_code"] == code].reset_index(drop=True)
    return f, r


def test_only_main_board_and_all_columns(world):
    f = world["feats"]
    assert set(f["ts_code"]) == set(CODES)
    assert set(BASE_NAMES) <= set(f.columns)
    assert len(f) == 2 * N


def test_units_moneyflow(world):
    f, r = _one(world)
    i = 150
    amt_wan = r.loc[i, "amount"] / 10            # 千元 → 万元
    assert f.loc[i, "mf_net1"] == pytest.approx(100 / amt_wan, rel=1e-5)
    assert f.loc[i, "big_net1"] == pytest.approx((30 - 10 + 20 - 5) / amt_wan, rel=1e-5)
    amt5 = r.loc[i - 4:i, "amount"].sum() / 10
    assert f.loc[i, "mf_net5"] == pytest.approx(500 / amt5, rel=1e-5)


def test_turnover_and_size(world):
    f, r = _one(world)
    i = 150
    tr = 1.0 + 0.01 * np.arange(N)
    assert f.loc[i, "turn5"] == pytest.approx(np.log(tr[i - 4:i + 1].mean() + 1e-6), rel=1e-5)
    assert f.loc[i, "turn_abn"] == pytest.approx(tr[i - 4:i + 1].mean() / tr[i - 59:i + 1].mean(), rel=1e-5)
    assert f.loc[i, "ln_circ_mv"] == pytest.approx(np.log(5e5), rel=1e-6)
    assert f.loc[i, "free_ratio"] == pytest.approx(4e3 / 8e3)
    age = (world["days"][i].date() - date(2019, 1, 1)).days
    assert f.loc[i, "ln_age"] == pytest.approx(np.log(1 + age), rel=1e-6)


def test_adjusted_returns_and_moments(world):
    f, r = _one(world)
    c = (r["close"] * r["adj"]).to_numpy()
    o = (r["open"] * r["adj"]).to_numpy()
    ret = c[1:] / c[:-1] - 1
    ret = np.r_[np.nan, ret]
    i = 110                                        # 窗口跨过第 100 天的除权：复权后不应有假跳变
    w = ret[i - 19:i + 1]
    assert f.loc[i, "rv20"] == pytest.approx(np.std(w, ddof=1), rel=1e-4)
    assert f.loc[i, "max_ret20"] == pytest.approx(w.max(), rel=1e-4)
    assert f.loc[i, "min_ret20"] == pytest.approx(w.min(), rel=1e-4)
    assert abs(f["max_ret20"].max()) < 0.2      # 除权日的不复权跌幅约 −9%，复权后没有
    dvr = np.sqrt(np.mean(np.minimum(w, 0) ** 2)) / np.sqrt(np.mean(w ** 2))
    assert f.loc[i, "down_vol_ratio"] == pytest.approx(dvr, rel=1e-4)
    ovn = np.log(o[1:] / c[:-1])
    assert f.loc[i, "ovn5"] == pytest.approx(ovn[i - 5:i].sum(), rel=1e-4)
    assert f.loc[i, "intra20"] == pytest.approx(np.log(c / o)[i - 19:i + 1].sum(), rel=1e-4)


def test_candles(world):
    f, r = _one(world)
    i = 150
    h, l, c, o = r.loc[i, ["high", "low", "close", "open"]]
    assert f.loc[i, "clv1"] == pytest.approx((2 * c - h - l) / (h - l), rel=1e-5)
    ls = ((np.minimum(r["open"], r["close"]) - r["low"]) / r["close"])[i - 4:i + 1].mean()
    assert f.loc[i, "low_shadow5"] == pytest.approx(ls, rel=1e-4)


def test_beta_and_ivol(world):
    f, r = _one(world)
    i = 200
    c = (r["close"] * r["adj"]).to_numpy()
    ret = c[1:] / c[:-1] - 1
    idx = world["idx"]
    ri = (idx / idx.shift(1) - 1).reindex([d.date() for d in world["days"]]).to_numpy()
    y, x = ret[i - 60:i], ri[i - 59:i + 1]
    slope, inter = np.polyfit(x, y, 1)
    resid = y - (slope * x + inter)
    assert f.loc[i, "beta60"] == pytest.approx(slope, rel=1e-4)
    assert f.loc[i, "ivol60"] == pytest.approx(np.sqrt((resid ** 2).sum() / (60 - 2)), rel=1e-4)


def test_limit_down_streak_and_250(world):
    f, r = _one(world)
    assert f.loc[200, "dist_dn"] == pytest.approx(0.0, abs=1e-6)       # 第 200、205 天收盘就是跌停价
    assert f.loc[205, "ld20"] == 2
    assert f.loc[226, "ld20"] == 0                                    # 两天都已滑出 20 日窗口
    c = (r["close"] * r["adj"]).to_numpy()
    neg = np.r_[False, c[1:] < c[:-1]]
    streak, k = [], 0
    for x in neg:
        k = k + 1 if x else 0
        streak.append(k)
    assert (f["down_streak"].to_numpy() == np.array(streak)).all()
    assert f["from_low250"].iloc[:249].isna().all() and f["from_low250"].iloc[249:].notna().all()
    i = 260
    assert f.loc[i, "dd250"] == pytest.approx(c[i] / c[i - 249:i + 1].max() - 1, rel=1e-5)


def test_market_state_same_for_all_stocks(world):
    f = world["feats"]
    g = f.groupby("trade_date")[["mkt_ret20", "breadth5", "limit_down_pct"]].nunique(dropna=False)
    assert (g <= 1).all().all()


def test_no_lookahead(world):
    """截断到某一天重算（SQL 里所有表只读到这一天），结果必须和全量计算的那一天完全相同。"""
    full = world["feats"]
    for i in (60, 150, 255):
        day = world["days"][i]
        cut = compute_range(world["con"], day.date(), day.date())
        a = cut.set_index("ts_code")[list(BASE_NAMES)].sort_index()
        b = full[full["trade_date"] == day].set_index("ts_code")[list(BASE_NAMES)].sort_index()
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-6)


def test_ranks_within_day():
    frame = pd.DataFrame({n: [3.0, 1.0, np.nan, 2.0] for n in
                          ("turn20", "amihud20", "ln_circ_mv", "mf_net5", "big_net5", "max_ret20", "ivol60", "ovn20")})
    out = add_ranks(frame, np.array([0, 0, 0, 1]))
    assert out["rk_turn20"].tolist()[:2] == [1.0, 0.5]
    assert np.isnan(out["rk_turn20"].iloc[2]) and out["rk_turn20"].iloc[3] == 1.0
