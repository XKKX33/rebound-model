"""sched_sim 的时点规则：合成行情（行情构造借正式测试的 make_market / random_market）。

用法：python -X utf8 -m pytest -q test_sched.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(HERE)]

from rebound.backtest.costs import CostModel  # noqa: E402
from rebound.backtest.market import candidates  # noqa: E402
from rebound.backtest.rules import parse  # noqa: E402
from rebound.backtest.sim import simulate  # noqa: E402
from test_backtest import make_market, random_market, random_signals  # noqa: E402

from sched_sim import SCHEDULES, hybrid, simulate_sched  # noqa: E402

BY = {s.name: s for s in SCHEDULES}
COST = CostModel(capital=20_000.0)


def rotation(n: int = 9):
    """3 只票、价格不动（开盘 10、收盘 11），第 d 天排第一的是 S(d mod 3)：K = 1、缓冲 1 时每天都换。"""
    m = make_market(np.full((n, 3), 10.0), np.full((n, 3), 11.0))
    d, j = np.meshgrid(np.arange(n), np.arange(3), indexing="ij")
    cands, rank = candidates(m, pd.DataFrame({"di": d.ravel(), "ji": j.ravel(),
                                              "score": (3 - (j - d) % 3).ravel().astype(float)}))
    return m, cands, rank, parse("DB-K1-b1").rule


def held(res) -> list[int]:
    return res.holdings.tolist()


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("name", ["DB-K2-b2", "DB-K3-b1.5", "DB-K1-b4"])
def test_now_without_slip_matches_the_official_simulator(seed, name):
    rng = np.random.default_rng(seed)
    m = random_market(rng)
    cands, rank = random_signals(rng, m)
    rule, cost = parse(name).legacy_rule, CostModel(capital=50_000.0)  # "现状"比的是开盘卖的正式规则
    base = simulate(m, cands, rank, rule, cost, 0, len(m.dates) - 2)
    mine, _ = simulate_sched(m, cands, rank, rule, cost, 0, len(m.dates) - 2, BY["now"])
    assert np.array_equal(base.nav, mine.nav, equal_nan=True) and base.trades.equals(mine.trades)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("sched", SCHEDULES + (hybrid(1), hybrid(2)), ids=lambda s: s.name)
def test_every_schedule_keeps_t_plus_one_and_cash_non_negative(seed, sched):
    rng = np.random.default_rng(100 + seed)
    m = random_market(rng)
    cands, rank = random_signals(rng, m)
    res, _ = simulate_sched(m, cands, rank, parse("DB-K2-b2").rule, CostModel(capital=50_000.0), 0,
                            len(m.dates) - 2, sched, slip=0.01, reserve=1)
    assert (res.trades.exit_day > res.trades.entry_day).all()
    assert np.nanmin(res.cash) > -1e-6 and np.isfinite(res.nav[1:]).all()


def test_now_buys_with_the_sale_money_at_the_same_open():
    m, cands, rank, rule = rotation()
    res, ledger = simulate_sched(m, cands, rank, rule, COST, 0, 7, BY["now"])
    assert held(res)[1:9] == [1] * 8
    assert ledger.funded.tolist() == [False] + [True] * 7  # 第一笔用本金，之后每笔都靠同一个开盘卖出的钱


def test_slip_only_hits_buys_that_need_the_sale_money():
    m, cands, rank, rule = rotation()
    base, _ = simulate_sched(m, cands, rank, rule, COST, 0, 7, BY["now"])
    slipped, _ = simulate_sched(m, cands, rank, rule, COST, 0, 7, BY["now"], slip=0.01)
    assert slipped.trades.entry_px.iloc[0] == base.trades.entry_px.iloc[0] == 10.0
    assert np.allclose(slipped.trades.entry_px.iloc[1:], 10.1)
    assert slipped.nav[-1] < base.nav[-1]


def test_wait_a_day_leaves_the_slot_empty_on_the_sell_day():
    m, cands, rank, rule = rotation()
    res, _ = simulate_sched(m, cands, rank, rule, COST, 0, 7, BY["wait_a_day"])
    assert held(res)[1:8] == [1, 0, 1, 0, 1, 0, 1]
    assert res.trades.j.tolist()[:2] == [0, 2]  # 第 3 天用第 2 天的排名买（排第一的是 S2）


def test_open_sell_close_buy_buys_at_the_close():
    m, cands, rank, rule = rotation()
    res, ledger = simulate_sched(m, cands, rank, rule, COST, 0, 7, BY["open_sell_close_buy"])
    assert (res.trades.entry_px == 11.0).all() and (res.trades.exit_px == 10.0).all()
    assert not ledger.funded.any() and held(res)[1:8] == [1] * 7


def test_close_sells_wait_for_t_plus_one():
    m, cands, rank, rule = rotation()
    prev, _ = simulate_sched(m, cands, rank, rule, COST, 0, 7, BY["close_sell_prev_rank"])
    same, _ = simulate_sched(m, cands, rank, rule, COST, 0, 7, BY["close_sell_same_day"])
    for res in (prev, same):
        assert held(res)[1:8] == [1, 0, 1, 0, 1, 0, 1] and (res.trades.exit_px == 11.0).all()
    assert prev.trades.exit_day.tolist()[:2] == same.trades.exit_day.tolist()[:2] == [2, 4]


def test_reserve_keeps_about_one_slot_of_cash():
    m = make_market(np.full((12, 4), 10.0))
    d, j = np.meshgrid(np.arange(12), np.arange(4), indexing="ij")
    cands, rank = candidates(m, pd.DataFrame({"di": d.ravel(), "ji": j.ravel(),
                                              "score": ((j - d) % 4).ravel().astype(float)}))
    res, ledger = simulate_sched(m, cands, rank, parse("DB-K2-b1").rule, COST, 0, 10, BY["reserve"], reserve=1)
    share = res.cash[2:11] / res.nav[2:11]
    assert ((share > 0.30) & (share < 0.40)).all() and not ledger.funded.any()


def test_half_buys_a_full_two_name_rotation_at_the_same_open():
    """前两名每天整组换（S0 S1 ↔ S2 S3）。半仓：同一个开盘卖两只、用另一半现金买两只，每笔都是足额；
    只留一个空位的钱时，第二只当天买不起足额，不买零头，空着等第二天。"""
    n = 10
    m = make_market(np.full((n, 4), 10.0))
    d, j = np.meshgrid(np.arange(n), np.arange(4), indexing="ij")
    top = np.where(d % 2 == 0, j < 2, j >= 2)
    cands, rank = candidates(m, pd.DataFrame({"di": d.ravel(), "ji": j.ravel(),
                                              "score": (top * 10 + (3 - j)).ravel().astype(float)}))
    rule, cost = parse("DB-K2-b1").rule, CostModel(capital=1_000_000.0)
    half, ledger = simulate_sched(m, cands, rank, rule, cost, 0, n - 2, BY["half"], reserve=2)
    assert held(half)[1:n] == [2] * (n - 1) and not ledger.funded.any()
    assert ledger.value.min() > 0.99 * ledger.value.max()
    share = half.cash[1:n] / half.nav[1:n]
    assert ((share > 0.49) & (share < 0.52)).all()
    part_res, part = simulate_sched(m, cands, rank, rule, cost, 0, n - 2, BY["half"], reserve=1)
    assert held(part_res)[1:6] == [2, 1, 2, 1, 2] and len(part[part.day == 2]) == 1


def test_cap_keeps_each_buy_at_half_the_capital_while_half_grows():
    """价格每天涨 5%、每天换票（K = 1）。封顶：每笔买入不超过 1 万（本金 2 万的一半），赚的钱留成现金；
    半仓：买入跟着净值的一半长。两种都只用开盘前的现金。"""
    n = 14
    m = make_market(10.0 * 1.05 ** np.arange(n)[:, None] * np.ones((1, 3)))
    d, j = np.meshgrid(np.arange(n), np.arange(3), indexing="ij")
    cands, rank = candidates(m, pd.DataFrame({"di": d.ravel(), "ji": j.ravel(),
                                              "score": (3 - (j - d) % 3).ravel().astype(float)}))
    rule = parse("DB-K1-b1").rule
    cap, cap_buys = simulate_sched(m, cands, rank, rule, COST, 0, n - 2, BY["cap"], reserve=1)
    _, half_buys = simulate_sched(m, cands, rank, rule, COST, 0, n - 2, BY["half"], reserve=1)
    assert cap_buys.value.max() <= 10_000 and half_buys.value.iloc[-1] > 10_000
    assert not cap_buys.funded.any() and held(cap)[1:n] == [1] * (n - 1)
    assert cap.cash[n - 1] > 0.55 * cap.nav[n - 1]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("name", ["DB-K2-b2", "DB-K3-b1.5", "DB-K1-b4"])
def test_hybrid_without_cash_is_close_sell_decided_the_night_before(seed, name):
    rng = np.random.default_rng(200 + seed)
    m = random_market(rng)
    cands, rank = random_signals(rng, m)
    rule, cost = parse(name).rule, CostModel(capital=50_000.0)
    a, _ = simulate_sched(m, cands, rank, rule, cost, 0, len(m.dates) - 2, BY["close_sell_prev_rank"])
    b, _ = simulate_sched(m, cands, rank, rule, cost, 0, len(m.dates) - 2, hybrid(0), reserve=0)
    assert np.array_equal(a.nav, b.nav, equal_nan=True) and a.trades.equals(b.trades)


def test_hybrid_swaps_what_the_cash_covers_at_the_open_and_sells_the_rest_at_the_close():
    """前两名每天整组换，开盘 10、收盘 11。留 1 只的现金：第 2 天两只都该卖，排名最差的 S1 开盘卖、
    用现金同时买一只新票，S0 拿到尾盘卖。"""
    n = 6
    m = make_market(np.full((n, 4), 10.0), np.full((n, 4), 11.0))
    d, j = np.meshgrid(np.arange(n), np.arange(4), indexing="ij")
    top = np.where(d % 2 == 0, j < 2, j >= 2)
    cands, rank = candidates(m, pd.DataFrame({"di": d.ravel(), "ji": j.ravel(),
                                              "score": (top * 10 + (3 - j)).ravel().astype(float)}))
    res, ledger = simulate_sched(m, cands, rank, parse("DB-K2-b1").rule, CostModel(capital=1_000_000.0), 0, n - 2,
                                 hybrid(1), reserve=1)
    day2 = res.trades[res.trades.exit_day == 2].set_index("j")
    assert day2.exit_px.to_dict() == {1: 10.0, 0: 11.0}
    assert ledger[ledger.day == 2].j.tolist() == [2] and not ledger.funded.any()


def test_close_buy_same_day_uses_that_days_rank():
    m, cands, rank, rule = rotation()
    same, ledger = simulate_sched(m, cands, rank, rule, COST, 0, 7, BY["open_sell_close_buy_same_day"])
    stale, _ = simulate_sched(m, cands, rank, rule, COST, 0, 7, BY["open_sell_close_buy"])
    assert ledger.iloc[0][["day", "j"]].tolist() == [1, 1]  # 第 1 天尾盘按第 1 天的排名买（排第一的是 S1）
    assert same.trades.iloc[0][["j", "entry_px", "exit_px"]].tolist() == [1, 11.0, 10.0]
    assert stale.trades.iloc[0].j == 0  # 前一天的排名：第 0 天排第一的 S0


def test_open_buys_cannot_use_that_days_rank():
    m, cands, rank, rule = rotation()
    with pytest.raises(ValueError):
        simulate_sched(m, cands, rank, rule, COST, 0, 7, replace(BY["now"], buy_lag=0))
