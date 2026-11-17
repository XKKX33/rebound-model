"""最少持有天数模拟器的测试，全部用合成行情。运行：python -m pytest -q test_minhold.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from engine import Cell, candidates  # noqa: E402
from sim import CostModel, Rule, _pick, simulate  # noqa: E402
from test_sim import make_market, random_market, random_signals  # noqa: E402

from minhold import hold_mix, paired, pick_next, rest_of_window, simulate_minhold, verdict  # noqa: E402
from rotation import rotation  # noqa: E402

O1O5 = Cell("open", 5, "open")


def same(a, b) -> None:
    assert np.array_equal(a.nav, b.nav, equal_nan=True)
    assert np.array_equal(a.holdings, b.holdings)
    assert np.array_equal(a.traded, b.traded)
    assert np.array_equal(a.costs, b.costs)
    assert np.array_equal(a.cash, b.cash, equal_nan=True)
    pd.testing.assert_frame_equal(a.trades, b.trades)
    assert (a.blocked_buys, a.stuck_days, a.poor_buys, a.poor_events, a.buy_events) == \
        (b.blocked_buys, b.stuck_days, b.poor_buys, b.poor_events, b.buy_events)


@pytest.mark.parametrize("seed", range(6))
def test_min_hold_one_is_the_original_simulator(seed):
    rng = np.random.default_rng(seed)
    m = random_market(rng, n=60, k=10)
    cands, rank = random_signals(rng, m)
    rules = [Rule("DB", 2, O1O5, buffer=1.5), Rule("DB", 3, O1O5, buffer=3), Rule("FH", 2, O1O5),
             Rule("FH", 2, O1O5, fill="skip")]
    for rule in rules:
        for cost in ((0.001, 0.002), CostModel(capital=2_500.0), CostModel(capital=50_000.0)):
            same(simulate_minhold(m, cands, rank, rule, cost, 0, 50, min_hold=1),
                 simulate(m, cands, rank, rule, cost, 0, 50))


@pytest.mark.parametrize("seed", range(4))
def test_pick_next_matches_the_original_pick(seed):
    # 和正式 tests/test_portfolio.py 同样的抽法：行业计数的键取自已有的行业（模拟器里它来自持仓的行业）
    rng = np.random.default_rng(100 + seed)
    for _ in range(1000):
        k = int(rng.integers(3, 40))
        industry = rng.integers(0, int(rng.integers(1, 8)), k).astype(np.int32)
        price = np.where(rng.random(k) < 0.1, np.nan, rng.uniform(1, 50, k))
        blocked = rng.random(k) < rng.uniform(0, 0.4)
        poor = rng.random(k) < rng.uniform(0, 0.9) if rng.random() < 0.8 else None
        order = rng.permutation(k)[: int(rng.integers(1, k + 1))]
        held = set(int(j) for j in rng.choice(k, int(rng.integers(0, 4)), replace=False))
        counts = {int(g): int(rng.integers(0, 3)) for g in rng.choice(industry, int(rng.integers(0, 3)))}
        slots, cap = int(rng.integers(0, 8)), int(rng.integers(1, 3))
        want = _pick(order, slots, held, counts, industry, cap, price, blocked, "next", poor)
        assert pick_next(order, slots, held, counts, industry, cap, price, blocked, poor) == want


@pytest.mark.parametrize("n_hold", [2, 3, 4, 6])
def test_no_db_position_is_sold_before_min_hold(n_hold):
    rng = np.random.default_rng(n_hold)
    m = random_market(rng, n=80, k=10)
    cands, rank = random_signals(rng, m)
    res = simulate_minhold(m, cands, rank, Rule("DB", 2, O1O5, buffer=1), (0.0, 0.0), 0, 70, min_hold=n_hold)
    held = res.trades.exit_day - res.trades.entry_day
    assert len(res.trades) > 5
    assert (held >= n_hold).all()
    assert (held == n_hold).any()  # 约束真的起了作用：随机打分下大部分票一过 N 天就掉出缓冲
    base = simulate_minhold(m, cands, rank, Rule("DB", 2, O1O5, buffer=1), (0.0, 0.0), 0, 70, min_hold=1)
    assert ((base.trades.exit_day - base.trades.entry_day) < n_hold).any()


def ranked(top_by_day: list[int], n: int, k: int = 3):
    """每个信号日把指定的票排第一，其余按编号；返回 (cands, rank)。"""
    rows = []
    for d, top in enumerate(top_by_day):
        order = [top] + [j for j in range(k) if j != top]
        rows += [(d, j, float(k - i)) for i, j in enumerate(order)]
    scores = pd.DataFrame(rows, columns=["di", "ji", "score"])
    m = make_market(np.full((n, k), 10.0))
    return m, *candidates(m, scores)


def trade_list(res) -> list[tuple[int, int, int]]:
    return [(int(t.j), int(t.entry_day), int(t.exit_day)) for t in res.trades.itertuples()]


def test_dropped_out_name_is_held_until_min_hold_then_sold():
    # 信号日 0~8 的第一名：0 1 1 1 0 0 0 0 0；K=1、b=1：掉出第 1 名就卖
    m, cands, rank = ranked([0, 1, 1, 1, 0, 0, 0, 0, 0], n=10)
    rule = Rule("DB", 1, O1O5, buffer=1)
    one = simulate_minhold(m, cands, rank, rule, (0.0, 0.0), 0, 8, min_hold=1)
    assert trade_list(one) == [(0, 1, 2), (1, 2, 5)]
    three = simulate_minhold(m, cands, rank, rule, (0.0, 0.0), 0, 8, min_hold=3)
    assert trade_list(three) == [(0, 1, 4), (1, 4, 7)]  # 0 号第 2 天就掉出，拿满 3 天才卖；1 号同理


def test_name_back_in_band_after_min_hold_is_kept():
    # 0 号第 1 天买，第 1、2 个信号日掉出、第 3 个信号日回到第一：拿满 3 天时还在缓冲里，留着；第 4 个信号日再掉出才卖
    m, cands, rank = ranked([0, 1, 1, 0, 2, 2, 2, 2, 2], n=10)
    rule = Rule("DB", 1, O1O5, buffer=1)
    assert trade_list(simulate_minhold(m, cands, rank, rule, (0.0, 0.0), 0, 8, min_hold=1))[0] == (0, 1, 2)
    assert trade_list(simulate_minhold(m, cands, rank, rule, (0.0, 0.0), 0, 8, min_hold=3)) == [(0, 1, 5)]


def test_min_hold_is_only_for_db():
    m, cands, rank = ranked([0] * 9, n=10)
    with pytest.raises(ValueError):
        simulate_minhold(m, cands, rank, Rule("FH", 1, O1O5), (0.0, 0.0), 0, 8, min_hold=3)
    with pytest.raises(ValueError):
        simulate_minhold(m, cands, rank, Rule("DB", 1, O1O5), (0.0, 0.0), 0, 8, min_hold=0)


def test_hold_mix_buckets_and_means():
    trades = pd.DataFrame({"entry_day": [1, 1, 2, 3, 3], "exit_day": [2, 3, 6, 4, 13],
                           "net": [0.01, -0.02, 0.03, 0.05, -0.04]})
    mix = hold_mix(trades, top=6)
    assert mix["1"] == {"share": 40.0, "net": 3.0}
    assert mix["2"] == {"share": 20.0, "net": -2.0}
    assert mix["4"] == {"share": 20.0, "net": 3.0}
    assert mix["6+"] == {"share": 20.0, "net": -4.0}
    assert hold_mix(trades.iloc[:0], top=6) == {}


def test_rest_of_window_vs_pool():
    open_ = np.array([[10, 10, 10], [10, 10, 10], [10, 20, 10], [11, 20, 10], [11, 20, 10], [12, 22, 9]], float)
    elig = np.ones(open_.shape, bool)
    m = make_market(open_)
    trades = pd.DataFrame({"j": [0, 1, 0], "entry_day": [1, 1, 1], "exit_day": [2, 3, 5]})
    got = rest_of_window(m, trades, hold=4)
    assert list(got.j) == [0, 1]  # 拿满 4 天的不算
    assert got.stock.iloc[0] == pytest.approx(0.2)            # 0 号第 2 天开盘 10 → 第 5 天开盘 12
    assert got.pool.iloc[0] == pytest.approx((0.2 + 0.1 - 0.1) / 3)
    assert got.stock.iloc[1] == pytest.approx(0.1)            # 1 号第 3 天开盘 20 → 第 5 天 22
    elig[2, 2] = False                                        # 1 号那笔从第 3 天卖：池子看第 2 天收盘
    m2 = make_market(open_)
    m2.eligible = elig
    got2 = rest_of_window(m2, trades, hold=4)
    assert got2.pool.iloc[1] == pytest.approx(((12 / 11 - 1) + 0.1) / 2)
    assert got2.pool.iloc[0] == pytest.approx(got.pool.iloc[0])  # 第一笔看第 1 天收盘，不受影响


def test_rotation_compares_the_new_buy_with_the_sold_name():
    open_ = np.array([[10, 10, 10], [10, 10, 10], [10, 20, 10], [11, 20, 10], [11, 20, 10], [12, 22, 9]], float)
    m = make_market(open_)
    trades = pd.DataFrame({"j": [0, 1, 2], "entry_day": [1, 1, 2], "exit_day": [2, 3, 5]})
    got = rotation(m, trades, rest_of_window(m, trades, hold=4))
    assert list(got.j) == [0, 1]  # 2 号拿到第 6 天超出数据，不算
    # 0 号第 2 天卖，同一个开盘买进 2 号：2 号第 2 → 5 天 −10%，0 号同一段 +20%
    assert got.rotation.iloc[0] == pytest.approx(-0.1 - 0.2)
    assert got.repl.iloc[0] == pytest.approx(-0.1 - got.pool.iloc[0])
    assert got.n_new.iloc[0] == 1
    assert np.isnan(got.repl.iloc[1]) and got.n_new.iloc[1] == 0  # 1 号第 3 天卖，那天没有新买进


def test_paired_constant_gap():
    dates = pd.bdate_range("2020-01-01", periods=300)
    out = paired(np.full(300, 0.001), np.zeros(300), dates)
    assert out["ann"] == pytest.approx(0.001 * 242 * 100, abs=1e-6)
    assert out["ci"][0] == pytest.approx(out["ann"], abs=1e-6) and out["ci"][1] == pytest.approx(out["ann"], abs=1e-6)
    assert set(out["years"]) == {"2020", "2021"}


@pytest.mark.parametrize("ci, share, worst_ok, want", [
    ((0.5, 3.0), 0.7, True, "更好"), ((0.5, 3.0), 0.55, True, "分不出"), ((0.5, 3.0), 0.7, False, "分不出"),
    ((-3.0, -0.2), 0.3, True, "更差"), ((-3.0, -0.2), 0.45, True, "分不出"), ((-1.0, 2.0), 0.9, True, "分不出"),
])
def test_verdict(ci, share, worst_ok, want):
    assert verdict(ci, share, worst_ok) == want
