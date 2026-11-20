"""任务 1.6 序列数据的单元测试：在手工构造的小库上核对 11 个通道的定义、单位、复牌后的收益、排名、窗口切片和防前视。"""

from __future__ import annotations

from datetime import date

import duckdb
import numpy as np
import pandas as pd
import pytest

from ml.seq_data import CHANNELS, ranked, raw_range, window


@pytest.fixture()
def con():
    """3 只主板股票、1 只创业板（应被排除），5 个交易日；B 在第 3 天停牌；第 2 天 A 涨停、C 跌停。"""
    c = duckdb.connect()
    days = pd.bdate_range("2024-01-02", periods=5).date
    rows, adj, lim, basic, mf = [], [], [], [], []
    px = {"000001.SZ": [10, 11, 11.5, 11, 12], "600000.SH": [20, 19, None, 21, 20], "000002.SZ": [5, 4.5, 4.6, 4.7, 4.8],
          "300001.SZ": [30, 31, 32, 33, 34]}
    for code, closes in px.items():
        for k, (d, cl) in enumerate(zip(days, closes)):
            if cl is None:
                continue
            op = cl * 0.98
            rows.append((code, d, op, cl * 1.02, cl * 0.97, cl, 1000.0 * (k + 1)))      # amount 千元
            adj.append((code, d, 2.0 if (code == "600000.SH" and k >= 3) else 1.0))    # B 复牌那天复权因子变了
            prev = closes[k - 1] if k and closes[k - 1] is not None else cl
            lim.append((code, d, round(prev * 1.1, 2), round(prev * 0.9, 2)))
            basic.append((code, d, 1.5 + k))
            mf.append((code, d, 10.0 * (k + 1), 5.0, 2.0, 1.0, 1.0))                    # 万元
    c.execute("CREATE TABLE daily (ts_code VARCHAR, trade_date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, amount DOUBLE)")
    c.executemany("INSERT INTO daily VALUES (?,?,?,?,?,?,?)", rows)
    c.execute("CREATE TABLE adj_factor (ts_code VARCHAR, trade_date DATE, adj_factor DOUBLE)")
    c.executemany("INSERT INTO adj_factor VALUES (?,?,?)", adj)
    c.execute("CREATE TABLE daily_limit (ts_code VARCHAR, trade_date DATE, up_limit DOUBLE, down_limit DOUBLE)")
    c.executemany("INSERT INTO daily_limit VALUES (?,?,?,?)", lim)
    c.execute("CREATE TABLE daily_basic (ts_code VARCHAR, trade_date DATE, turnover_rate_f DOUBLE)")
    c.executemany("INSERT INTO daily_basic VALUES (?,?,?)", basic)
    c.execute("CREATE TABLE moneyflow (ts_code VARCHAR, trade_date DATE, net_mf_amount DOUBLE, buy_lg_amount DOUBLE, "
              "buy_elg_amount DOUBLE, sell_lg_amount DOUBLE, sell_elg_amount DOUBLE)")
    c.executemany("INSERT INTO moneyflow VALUES (?,?,?,?,?,?,?)", mf)
    # A 第 2 天收在涨停价、C 第 2 天收在跌停价
    c.execute("UPDATE daily SET close = 11.0 WHERE ts_code = '000001.SZ' AND trade_date = ?", [days[1]])
    c.execute("UPDATE daily_limit SET up_limit = 11.0 WHERE ts_code = '000001.SZ' AND trade_date = ?", [days[1]])
    c.execute("UPDATE daily_limit SET down_limit = 4.5 WHERE ts_code = '000002.SZ' AND trade_date = ?", [days[1]])
    yield c, days
    c.close()


def test_raw_channels(con):
    c, days = con
    raw = raw_range(c, days[0], days[-1])
    assert "300001.SZ" not in set(raw["ts_code"])                                    # 创业板排除
    a = raw[raw["ts_code"] == "000001.SZ"].set_index("trade_date")
    d1 = pd.Timestamp(days[1])
    assert a.loc[d1, "ret"] == pytest.approx(np.log(11.0 / 10))
    assert a.loc[d1, "ovn"] == pytest.approx(np.log(11 * 0.98 / 10))
    assert a.loc[d1, "intra"] == pytest.approx(np.log(11.0 / (11 * 0.98)))
    assert a.loc[d1, "hl"] == pytest.approx(np.log(1.02 / 0.97))
    assert a.loc[d1, "ln_amt"] == pytest.approx(np.log(2000.0))
    assert a.loc[d1, "ln_turn"] == pytest.approx(np.log(2.5))
    assert a.loc[d1, "mf"] == pytest.approx(20.0 / 200.0)                             # 万元 / (千元 / 10)
    assert a.loc[d1, "big"] == pytest.approx((5 + 2 - 1 - 1) / 200.0)
    assert a.loc[d1, "limit"] == 0.5
    cc = raw[raw["ts_code"] == "000002.SZ"].set_index("trade_date")
    assert cc.loc[d1, "limit"] == -0.5
    assert np.isnan(a.loc[pd.Timestamp(days[0]), "ret"])                              # 第一天没有前收


def test_return_across_suspension_uses_adjusted_prices(con):
    """B 第 3 天停牌、第 4 天复牌且复权因子从 1 变 2：收益 = ln(21×2 / (19×1))，跨过停牌期。"""
    c, days = con
    raw = raw_range(c, days[0], days[-1])
    b = raw[raw["ts_code"] == "600000.SH"].set_index("trade_date")
    assert pd.Timestamp(days[2]) not in b.index
    assert b.loc[pd.Timestamp(days[3]), "ret"] == pytest.approx(np.log(21 * 2.0 / 19))


def test_no_lookahead(con):
    c, days = con
    full = raw_range(c, days[0], days[-1])
    cut = raw_range(c, days[0], days[2])                                              # 只读到第 3 天
    m = full["trade_date"] <= pd.Timestamp(days[2])
    pd.testing.assert_frame_equal(full[m].reset_index(drop=True), cut.reset_index(drop=True))


def test_ranked_channels(con):
    c, days = con
    raw = raw_range(c, days[0], days[-1])
    r = ranked(raw)
    assert list(r.columns) == list(CHANNELS)
    assert (r["traded"] == 1).all()
    assert r.iloc[:, :9].abs().max().max() < 0.5
    assert np.isfinite(r.to_numpy()).all()                                           # 空值已填 0
    day = raw["trade_date"].to_numpy()
    assert r.groupby(day)["ret"].mean().abs().max() < 1e-9                           # 每天居中（第一天全空 → 全 0）
    d1 = raw["trade_date"] == pd.Timestamp(days[1])
    order = raw.loc[d1, "ret"].rank().to_numpy()
    assert (np.argsort(order) == np.argsort(r.loc[d1, "ret"].to_numpy())).all()       # 排名保序


def test_window_slices():
    X = np.arange(2 * 10 * 1, dtype=np.float32).reshape(2, 10, 1)
    w = window(X, np.array([1, 0]), np.array([9, 4]), length=3)
    np.testing.assert_array_equal(w[0, :, 0], X[1, 7:10, 0])
    np.testing.assert_array_equal(w[1, :, 0], X[0, 2:5, 0])
    with pytest.raises(ValueError):
        window(X, np.array([0]), np.array([1]), length=3)
