"""组合模拟器和逐笔引擎的规则测试，全部用合成行情。运行：python -m pytest -q test_sim.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine import CELLS, EXIT_DROPPED, Cell, Market, candidates, matched_pool, pool_daily_return, pool_legs, trades
from sim import WARMUP, CostModel, Rule, simulate, summarize

O1O2, O1O3 = Cell("open", 2, "open"), Cell("open", 3, "open")


def make_market(open_, close=None, industry=None, raw=None, **flags) -> Market:
    """合成行情；raw（未复权开收盘）缺省等于复权价，即期间没有除权。"""
    open_ = np.asarray(open_, dtype=float)
    close = open_.copy() if close is None else np.asarray(close, dtype=float)
    n, k = open_.shape

    def flag(name: str) -> np.ndarray:
        value = flags.get(name)
        return np.zeros((n, k), bool) if value is None else np.asarray(value, bool)

    return Market(dates=pd.bdate_range("2023-01-02", periods=n), codes=np.array([f"S{i}" for i in range(k)]),
                  price={"open": open_, "close": close},
                  buy_blocked={"open": flag("up_open"), "close": flag("up_close")},
                  sell_blocked={"open": flag("dn_open"), "close": flag("dn_close")},
                  eligible=np.ones((n, k), bool),
                  industry=np.asarray(np.arange(k) if industry is None else industry, dtype=np.int32),
                  raw=raw if raw is not None else {"open": open_.copy(), "close": close.copy()})


def random_market(rng: np.random.Generator, n: int = 40, k: int = 8, block: float = 0.08,
                  suspend: float = 0.03) -> Market:
    open_ = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, k)), axis=0))
    close = open_ * (1 + rng.normal(0, 0.01, (n, k)))
    off = rng.random((n, k)) < suspend
    open_[off], close[off] = np.nan, np.nan
    flags = {name: rng.random((n, k)) < block for name in ("up_open", "up_close", "dn_open", "dn_close")}
    return make_market(open_, close, industry=rng.integers(0, 4, k), **flags)


def random_signals(rng: np.random.Generator, m: Market):
    d, j = np.meshgrid(np.arange(len(m.dates)), np.arange(len(m.codes)), indexing="ij")
    scores = pd.DataFrame({"di": d.ravel(), "ji": j.ravel(), "score": rng.random(d.size)})
    return candidates(m, scores)


def test_capital_days_per_cell():
    assert Cell("open", 5, "open").period == 4
    assert Cell("open", 5, "close").period == 5
    assert Cell("close", 2, "close").period == 1
    assert Cell("close", 3, "open").period == 2


def test_flat_prices_zero_cost_keep_nav_at_one():
    m = make_market(np.full((12, 3), 10.0))
    res = simulate(m, {d: np.array([0, 1, 2]) for d in range(12)}, None, Rule("FH", 2, O1O3), (0, 0), 0, 8)
    assert np.allclose(res.nav, 1.0)
    assert len(res.trades) > 0


def test_round_trip_on_flat_prices_costs_exactly_the_fees():
    m = make_market(np.full((12, 3), 10.0))
    b, s = 0.0015, 0.0025
    res = simulate(m, {d: np.array([0, 1, 2]) for d in range(12)}, None, Rule("FH", 2, O1O3), (b, s), 0, 8)
    assert np.allclose(res.trades.gross, 0.0)
    assert np.allclose(res.trades.net, (1 - s) / (1 + b) - 1)


def test_nav_tracks_a_known_path():
    open_ = np.array([[10, 10], [10, 10], [11, 10], [11, 10], [11, 10]], float)
    close = np.array([[10, 10], [10.5, 10], [11, 10], [11, 10], [11, 10]], float)
    res = simulate(make_market(open_, close), {0: np.array([0, 1])}, None, Rule("FH", 1, O1O2), (0, 0), 0, 0)
    assert res.nav[1] == pytest.approx(1.05)
    assert res.nav[2] == pytest.approx(1.10)


def test_no_position_is_sold_on_its_buy_day():
    rng = np.random.default_rng(0)
    for cell in (Cell("open", 2, "open"), Cell("open", 2, "close"), Cell("close", 2, "open"), Cell("close", 2, "close")):
        m = random_market(rng)
        cands, rank = random_signals(rng, m)
        for rule in (Rule("FH", 2, cell), Rule("DB", 2, cell, buffer=1)):
            res = simulate(m, cands, rank, rule, (0.001, 0.002), 0, 30)
            assert len(res.trades) > 0
            assert (res.trades.exit_day > res.trades.entry_day).all()


def test_blocked_buy_moves_to_next_name_or_leaves_the_slot_empty():
    up_open = np.zeros((6, 3), bool)
    up_open[1, 0] = True
    m = make_market(np.full((6, 3), 10.0), up_open=up_open)
    nxt = simulate(m, {0: np.array([0, 1, 2])}, None, Rule("FH", 1, O1O2, fill="next"), (0, 0), 0, 0)
    skp = simulate(m, {0: np.array([0, 1, 2])}, None, Rule("FH", 1, O1O2, fill="skip"), (0, 0), 0, 0)
    assert list(nxt.trades.j) == [1]
    assert skp.trades.empty
    assert nxt.blocked_buys == 1 and skp.blocked_buys == 1


def test_limit_down_exit_is_carried_to_the_first_sellable_open():
    open_ = np.full((8, 2), 10.0)
    open_[5:, 0] = 8.0
    dn = np.zeros((8, 2), bool)
    dn[3, 0] = dn[4, 0] = True
    m = make_market(open_, dn_open=dn)
    res = simulate(m, {0: np.array([0, 1])}, None, Rule("FH", 1, O1O3), (0, 0), 0, 0)
    t = res.trades.iloc[0]
    assert (t.j, t.entry_day, t.exit_day, bool(t.carried)) == (0, 1, 5, True)
    assert t.gross == pytest.approx(-0.2)
    assert res.stuck_days == 2
    assert trades(m, np.array([0]), np.array([0]), O1O3, "carry")["ret"][0] == pytest.approx(-0.2)
    assert trades(m, np.array([0]), np.array([0]), O1O3, "drop")["status"][0] == EXIT_DROPPED


def test_suspended_name_is_held_at_last_close_and_sold_after_resuming():
    open_, close = np.full((7, 2), 10.0), np.full((7, 2), 10.0)
    close[2, 0] = 11.0
    open_[3, 0] = close[3, 0] = np.nan
    open_[4, 0] = 12.0
    res = simulate(make_market(open_, close), {0: np.array([0, 1])}, None, Rule("FH", 1, O1O3), (0, 0), 0, 0)
    t = res.trades.iloc[0]
    assert (t.exit_day, bool(t.carried)) == (4, True)
    assert t.gross == pytest.approx(0.2)
    # O1O3 有 2 个批次，每批 0.5：这一批全买了股票 0（0.05 股），另一批空着。
    assert res.nav[3] == pytest.approx(0.5 + 0.05 * 11.0)


def test_industry_cap_limits_two_names_per_industry():
    m = make_market(np.full((6, 4), 10.0), industry=[0, 0, 0, 1])
    res = simulate(m, {0: np.array([0, 1, 2, 3])}, None, Rule("FH", 3, O1O2), (0, 0), 0, 0)
    assert sorted(res.trades.j) == [0, 1, 3]


def test_db_sells_when_rank_leaves_the_buffer_and_refills_the_slot():
    m = make_market(np.full((6, 3), 10.0))
    scores = pd.DataFrame({"di": [0, 0, 0, 1, 1, 1], "ji": [0, 1, 2, 0, 1, 2], "score": [3, 2, 1, 1, 3, 2]})
    cands, rank = candidates(m, scores)
    res = simulate(m, cands, rank, Rule("DB", 1, O1O2, buffer=1), (0, 0), 0, 1)
    t = res.trades.iloc[0]
    assert (t.j, t.entry_day, t.exit_day) == (0, 1, 2)
    assert res.holdings[2] == 1


def test_matched_benchmark_removes_the_overnight_and_intraday_cash_gap():
    """收盘对收盘为 0、全部涨跌都在隔夜和日内：同一腿买卖的 FH 没有虚假超额；不同腿的有，时点匹配后消失。"""
    n, k, night = 60, 12, -0.005
    close = np.full((n, k), 10.0)
    m = make_market(close * (1 + night), close)
    legs, pool = pool_legs(m), pool_daily_return(m)
    cands = {d: np.arange(k) for d in range(n)}
    for cell in (Cell("open", 5, "open"), Cell("open", 5, "close"), Cell("close", 5, "open")):
        res = simulate(m, cands, None, Rule("FH", 3, cell), (0, 0), 0, 50)
        raw = summarize(m, res, pool, 0, 50)["all"]["excess"]["ann"]
        matched = summarize(m, res, matched_pool(legs, "FH", cell), 0, 50)["all"]["excess"]["ann"]
        assert abs(matched) < 0.2, (cell.name, matched)
        if cell.entry != cell.exit:
            assert abs(raw) > 10, (cell.name, raw)


def test_no_zero_share_position_after_selling_for_nothing():
    open_ = np.full((8, 2), 10.0)
    open_[2:, 0] = 0.0
    up = np.zeros((8, 2), bool)
    up[2:, 0] = True
    m = make_market(open_, open_.copy(), up_open=up, up_close=up)
    scores = pd.DataFrame({"di": [0, 0, 1, 1], "ji": [0, 1, 0, 1], "score": [2, 1, 1, 2]})
    cands, rank = candidates(m, scores)
    res = simulate(m, cands, rank, Rule("DB", 1, Cell("open", 2, "open"), buffer=1), (0, 0), 0, 1)
    assert list(res.trades.exit_px) == [0.0]
    assert res.holdings[2] == 0 and res.nav[2] == pytest.approx(0.0)


def test_metrics_skip_the_build_up_days():
    m = make_market(np.full((12, 3), 10.0))
    res = simulate(m, {d: np.array([0, 1, 2]) for d in range(12)}, None, Rule("FH", 2, O1O3), (0, 0), 0, 8)
    assert summarize(m, res, pool_daily_return(m), 0, 8)["all"]["days"] == 8 + 1 - WARMUP


def test_real_account_buys_whole_lots_and_pays_real_fees():
    """2 万本金、33.3 元：买 6 手（600 股），佣金万五、过户费十万分之一；卖出另收万五印花税。"""
    m = make_market(np.full((5, 2), 33.3))
    res = simulate(m, {0: np.array([0, 1])}, None, Rule("FH", 1, O1O2), CostModel(), 0, 0)
    buy_fee = 19980 * 0.0005 + 19980 * 0.00001
    sell_fee = 19980 * 0.0005 + 19980 * 0.0005 + 19980 * 0.00001
    assert res.nav[1] == pytest.approx(20_000 - buy_fee)
    assert res.nav[2] == pytest.approx(20_000 - buy_fee - sell_fee)
    assert res.trades.gross.iloc[0] == pytest.approx(0.0)


def test_small_order_pays_the_minimum_commission():
    m = make_market(np.full((5, 2), 10.0))
    res = simulate(m, {0: np.array([0, 1])}, None, Rule("FH", 1, O1O2), CostModel(capital=2_000), 0, 0)
    assert res.costs[1] == pytest.approx(5.0 + 1000 * 0.00001)  # 只买得起 1 手，佣金按最低 5 元
    assert res.cash[1] == pytest.approx(2_000 - 1000 - 5.01)


def test_name_whose_lot_is_too_expensive_moves_to_the_next():
    open_ = np.full((5, 2), 10.0)
    open_[:, 0] = 25.0  # 一手 2500 元，超过 2000 元本金
    m = make_market(open_)
    nxt = simulate(m, {0: np.array([0, 1])}, None, Rule("FH", 1, O1O2), CostModel(capital=2_000), 0, 0)
    skp = simulate(m, {0: np.array([0, 1])}, None, Rule("FH", 1, O1O2, fill="skip"), CostModel(capital=2_000), 0, 0)
    assert list(nxt.trades.j) == [1] and nxt.poor_buys == 1
    assert skp.trades.empty and skp.poor_buys == 1


def test_minimum_commission_never_takes_more_than_the_sale():
    """本金很小、股价跌到几分钱时，最低佣金比卖出所得还多：只能扣光所得，现金不能变成负的。"""
    open_ = np.full((5, 2), 2.0)
    open_[2:, 0] = 0.01
    m = make_market(open_)
    res = simulate(m, {0: np.array([0, 1])}, None, Rule("FH", 1, O1O2), CostModel(capital=300, min_commission=50), 0, 0)
    assert res.costs[2] == pytest.approx(1.0)  # 100 股 × 0.01 元 = 1 元，费用最多 1 元
    assert np.nanmin(res.cash) >= 0 and np.nanmin(res.nav) >= 0


def test_counts_build_ups_where_the_top_name_was_too_expensive():
    open_ = np.full((5, 2), 10.0)
    open_[:, 0] = 25.0
    res = simulate(make_market(open_), {0: np.array([0, 1])}, None, Rule("FH", 1, O1O2), CostModel(capital=2_000), 0, 0)
    assert (res.buy_events, res.poor_events, res.poor_buys) == (1, 1, 1)


def test_split_keeps_value_continuous_in_a_real_account():
    """2 送 1：未复权价从 10 跌到 5，复权价不变；持仓市值和卖出所得都不该跳。"""
    adj = np.full((8, 2), 10.0)
    raw = adj.copy()
    raw[3:, 0] = 5.0
    m = make_market(adj, raw={"open": raw.copy(), "close": raw.copy()})
    free = CostModel(capital=40_000, commission=0, min_commission=0, stamp=0, transfer=0)
    res = simulate(m, {0: np.array([0, 1])}, None, Rule("FH", 1, Cell("open", 5, "open")), free, 0, 0)
    assert np.allclose(res.nav[:7], 40_000)
    assert res.trades.gross.iloc[0] == pytest.approx(0.0)


def test_stamp_duty_before_the_2023_cut_is_optional():
    m = make_market(np.full((5, 2), 33.3))  # 合成日期从 2023-01-02 起，都在减半之前
    old = simulate(m, {0: np.array([0, 1])}, None, Rule("FH", 1, O1O2), CostModel(stamp_before=0.001), 0, 0)
    new = simulate(m, {0: np.array([0, 1])}, None, Rule("FH", 1, O1O2), CostModel(), 0, 0)
    assert old.nav[2] == pytest.approx(new.nav[2] - 19980 * 0.0005)


def test_fh_trade_returns_match_the_per_trade_engine():
    rng = np.random.default_rng(7)
    m = random_market(rng)
    cands, rank = random_signals(rng, m)
    for cell in CELLS:
        res = simulate(m, cands, rank, Rule("FH", 3, cell, fill="skip"), (0, 0), 0, 40 - 6 - cell.x)
        eng = trades(m, res.trades.signal_day.to_numpy(), res.trades.j.to_numpy(), cell, "carry")
        assert len(res.trades) > 0
        assert np.allclose(res.trades.gross.to_numpy(), eng["ret"])
        assert (res.trades.exit_day.to_numpy() == eng["exit_day"]).all()
