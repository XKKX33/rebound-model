"""本金测算的模拟器、规则网格、选法（`rebound/backtest/`）：全部用合成行情。

模拟器的规则测试从十年实验 `p0p1-20261029/test_sim.py` 搬来（断言原样）；和研究版逐项比真实打分的核对在
`data/research/model-parity-20261031/capital_check.py`。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rebound.labels import Cell, carry_trades
from rebound.backtest import stats
from rebound.backtest.costs import CostModel
from rebound.backtest.market import Market, candidates, pool_daily_return
from rebound.backtest.rules import Spec, neighbors, parse, specs
from rebound.backtest.sim import WARMUP, Rule, _pick, _pick_next, simulate

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
    return candidates(m, pd.DataFrame({"di": d.ravel(), "ji": j.ravel(), "score": rng.random(d.size)}))


def test_capital_days_per_cell():
    assert Cell("open", 5, "open").period == 4
    assert Cell("open", 5, "close").period == 5
    assert Cell("close", 2, "close").period == 1
    assert Cell("close", 3, "open").period == 2


def test_flat_prices_zero_cost_keep_nav_at_one():
    m = make_market(np.full((12, 3), 10.0))
    res = simulate(m, {d: np.array([0, 1, 2]) for d in range(12)}, None, Rule("FH", 2, O1O3), (0, 0), 0, 8)
    assert np.allclose(res.nav, 1.0) and len(res.trades) > 0


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
    assert res.nav[1] == pytest.approx(1.05) and res.nav[2] == pytest.approx(1.10)


def test_no_position_is_sold_on_its_buy_day():
    rng = np.random.default_rng(0)
    for cell in (Cell("open", 2, "open"), Cell("open", 2, "close"), Cell("close", 2, "open"), Cell("close", 2, "close")):
        m = random_market(rng)
        cands, rank = random_signals(rng, m)
        for rule in (Rule("FH", 2, cell), Rule("DB", 2, cell, buffer=1)):
            res = simulate(m, cands, rank, rule, (0.001, 0.002), 0, 30)
            assert len(res.trades) > 0 and (res.trades.exit_day > res.trades.entry_day).all()


def test_blocked_buy_moves_to_next_name_or_leaves_the_slot_empty():
    up_open = np.zeros((6, 3), bool)
    up_open[1, 0] = True
    m = make_market(np.full((6, 3), 10.0), up_open=up_open)
    nxt = simulate(m, {0: np.array([0, 1, 2])}, None, Rule("FH", 1, O1O2, fill="next"), (0, 0), 0, 0)
    skp = simulate(m, {0: np.array([0, 1, 2])}, None, Rule("FH", 1, O1O2, fill="skip"), (0, 0), 0, 0)
    assert list(nxt.trades.j) == [1] and skp.trades.empty
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
    assert t.gross == pytest.approx(-0.2) and res.stuck_days == 2


def test_suspended_name_is_held_at_last_close_and_sold_after_resuming():
    open_, close = np.full((7, 2), 10.0), np.full((7, 2), 10.0)
    close[2, 0] = 11.0
    open_[3, 0] = close[3, 0] = np.nan
    open_[4, 0] = 12.0
    res = simulate(make_market(open_, close), {0: np.array([0, 1])}, None, Rule("FH", 1, O1O3), (0, 0), 0, 0)
    t = res.trades.iloc[0]
    assert (t.exit_day, bool(t.carried)) == (4, True) and t.gross == pytest.approx(0.2)
    assert res.nav[3] == pytest.approx(0.5 + 0.05 * 11.0)  # 2 个批次各 0.5：一批买了 0.05 股，一批空着


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
    assert (t.j, t.entry_day, t.exit_day) == (0, 1, 2) and res.holdings[2] == 1
    assert [p.j for p in res.positions] == [1] and len(res.tranche_cash) == 1


def test_no_zero_share_position_after_selling_for_nothing():
    open_ = np.full((8, 2), 10.0)
    open_[2:, 0] = 0.0
    up = np.zeros((8, 2), bool)
    up[2:, 0] = True
    m = make_market(open_, open_.copy(), up_open=up, up_close=up)
    cands, rank = candidates(m, pd.DataFrame({"di": [0, 0, 1, 1], "ji": [0, 1, 0, 1], "score": [2, 1, 1, 2]}))
    res = simulate(m, cands, rank, Rule("DB", 1, Cell("open", 2, "open"), buffer=1), (0, 0), 0, 1)
    assert list(res.trades.exit_px) == [0.0]
    assert res.holdings[2] == 0 and res.nav[2] == pytest.approx(0.0)


def test_real_account_buys_whole_lots_and_pays_real_fees():
    """2 万本金、33.3 元：买 6 手（600 股），佣金万五、过户费十万分之一；卖出另收万五印花税。"""
    res = simulate(make_market(np.full((5, 2), 33.3)), {0: np.array([0, 1])}, None, Rule("FH", 1, O1O2),
                   CostModel(), 0, 0)
    buy_fee = 19980 * 0.0005 + 19980 * 0.00001
    sell_fee = 19980 * 0.0005 + 19980 * 0.0005 + 19980 * 0.00001
    assert res.nav[1] == pytest.approx(20_000 - buy_fee)
    assert res.nav[2] == pytest.approx(20_000 - buy_fee - sell_fee)
    assert res.trades.gross.iloc[0] == pytest.approx(0.0)


def test_small_order_pays_the_minimum_commission():
    res = simulate(make_market(np.full((5, 2), 10.0)), {0: np.array([0, 1])}, None, Rule("FH", 1, O1O2),
                   CostModel(capital=2_000), 0, 0)
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
    assert (nxt.buy_events, nxt.poor_events) == (1, 1)


def test_minimum_commission_never_takes_more_than_the_sale():
    open_ = np.full((5, 2), 2.0)
    open_[2:, 0] = 0.01
    res = simulate(make_market(open_), {0: np.array([0, 1])}, None, Rule("FH", 1, O1O2),
                   CostModel(capital=300, min_commission=50), 0, 0)
    assert res.costs[2] == pytest.approx(1.0)  # 100 股 × 0.01 元 = 1 元，费用最多 1 元
    assert np.nanmin(res.cash) >= 0 and np.nanmin(res.nav) >= 0


def test_split_keeps_value_continuous_in_a_real_account():
    """2 送 1：未复权价从 10 跌到 5，复权价不变；持仓市值和卖出所得都不该跳。"""
    adj = np.full((8, 2), 10.0)
    raw = adj.copy()
    raw[3:, 0] = 5.0
    m = make_market(adj, raw={"open": raw.copy(), "close": raw.copy()})
    free = CostModel(capital=40_000, commission=0, min_commission=0, stamp=0, transfer=0)
    res = simulate(m, {0: np.array([0, 1])}, None, Rule("FH", 1, Cell("open", 5, "open")), free, 0, 0)
    assert np.allclose(res.nav[:7], 40_000) and res.trades.gross.iloc[0] == pytest.approx(0.0)


def test_stamp_duty_before_the_2023_cut_is_optional():
    m = make_market(np.full((5, 2), 33.3))  # 合成日期从 2023-01-02 起，都在减半之前
    old = simulate(m, {0: np.array([0, 1])}, None, Rule("FH", 1, O1O2), CostModel(stamp_before=0.001), 0, 0)
    new = simulate(m, {0: np.array([0, 1])}, None, Rule("FH", 1, O1O2), CostModel(), 0, 0)
    assert old.nav[2] == pytest.approx(new.nav[2] - 19980 * 0.0005)


def test_fh_trade_returns_match_the_per_trade_engine():
    """模拟器里每笔 FH 交易的收益、卖出日和逐笔标签引擎（`model_labels.carry_trades`）相同。"""
    rng = np.random.default_rng(7)
    m = random_market(rng)
    cands, rank = random_signals(rng, m)
    for cell in (Cell(e, x, leg) for e in ("open", "close") for x in (2, 3, 5) for leg in ("open", "close")):
        res = simulate(m, cands, rank, Rule("FH", 3, cell, fill="skip"), (0, 0), 0, 40 - 6 - cell.x)
        eng = carry_trades(m, res.trades.signal_day.to_numpy(), res.trades.j.to_numpy(), cell)
        assert len(res.trades) > 0
        assert np.allclose(res.trades.gross.to_numpy(), eng["ret"])
        assert (res.trades.exit_day.to_numpy() == eng["exit_day"]).all()


def test_the_fast_pick_matches_the_original_on_random_days() -> None:
    """`_pick_next` 是 `_pick(fill="next")` 的快版：挑中的票、买不进和买不起的次数都要逐项相同。"""
    rng = np.random.default_rng(20261031)
    for _ in range(3000):
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
        assert _pick_next(order, slots, held, counts, industry, cap, price, blocked, poor) == want


def test_metrics_skip_the_build_up_days():
    m = make_market(np.full((12, 3), 10.0))
    res = simulate(m, {d: np.array([0, 1, 2]) for d in range(12)}, None, Rule("FH", 2, O1O3), (0, 0), 0, 8)
    idx = stats.scoring_days(0, 8)
    assert len(idx) == 8 + 1 - WARMUP
    assert stats.summary(res, pool_daily_return(m), m.dates, idx)["days"] == len(idx)


# --------------------------------------------------------------------------- #
# 规则网格和选法（十年实验 test_rules.py）
# --------------------------------------------------------------------------- #


def test_grid_has_150_unique_rules_and_names_round_trip() -> None:
    grid = specs()
    assert len(grid) == len(set(grid)) == 150
    assert sum(s.family == "FH" for s in grid) == 60 and sum(s.family == "DB" for s in grid) == 90
    for spec in grid:
        assert parse(spec.name) == spec == parse(spec.short) == parse(spec.legacy, legacy_ok=True)
        with pytest.raises(ValueError, match=spec.name):  # 老名字默认不认：拿它取 .rule 会悄悄变成尾盘卖
            parse(spec.legacy)
    assert parse("DB-K30-b8-o1c5").plain == "最多 30 只，排名掉出前 240 名才卖"
    assert Spec("FH", 6, 4).name == "FH-K6-next-o1c5" and Spec("FH", 6, 4).legacy == "FH-K6-next-o1o5"
    with pytest.raises(ValueError):
        parse("FH-K30-next-o1c5")  # 30 × 4 = 120 > 100，不在网格里


def test_grid_rules_buy_at_the_open_and_sell_at_the_close_old_names_keep_their_open_sell() -> None:
    """2026-11-01 起网格里的规则都是开盘买、尾盘卖（前一晚定）。FH 卖在原来那个卖出日的尾盘：持有 h 天、资金分 h + 1 批。
    老名字（-o1o5 这类，跟踪文件里存着的）照原来的开盘卖回放；时点后缀对不上的名字不认，免得悄悄按另一个时点算。"""
    from rebound.backtest.rules import rule_of

    fh, db = parse("FH-K6-h4"), parse("DB-K2-b6")
    assert (fh.rule.cell.entry, fh.rule.cell.x, fh.rule.cell.exit, fh.rule.cell.period) == ("open", 5, "close", 5)
    assert (db.rule.cell.entry, db.rule.cell.exit, db.rule.buffer) == ("open", "close", 6.0)
    assert rule_of("DB-K2-b6") == rule_of("DB-K2-b6-o1c5") == db.rule
    assert rule_of("DB-K2-b6-o1o5") == Rule("DB", 2, Cell("open", 5, "open"), buffer=6.0)
    assert rule_of("FH-K6-next-o1o5") == Rule("FH", 6, Cell("open", 5, "open"))
    assert parse("FH-K6-next-o1c4") == Spec("FH", 6, 3)  # 是 FH-K6-h3 的全名，不是错的
    for bad in ("DB-K2-b6-c1c5", "DB-K2-b6-o1c4", "FH-K6-next-c1o5", "FH-K6-next-c1c5", "FH-K6-skip-o1c5", "DB-K2"):
        with pytest.raises(ValueError):
            parse(bad)


def test_neighbors_stay_on_the_grid_and_respect_the_book_cap() -> None:
    grid = set(specs())
    near = neighbors(Spec("FH", 20, 4), grid)
    assert Spec("FH", 20, 4) in near and all(s.k * s.p <= 100 for s in near)
    assert len(neighbors(Spec("DB", 1, 1), grid)) == 4  # 角上：自己 + 3 个


def test_walk_forward_only_uses_the_past_and_splices_the_picked_years() -> None:
    dates = pd.bdate_range("2015-01-01", "2018-12-31")
    rng = np.random.default_rng(1)
    book = rng.normal(0, 0.01, (3, len(dates)))
    book[1, dates.year < 2017] += 0.01   # 第 1 组前两年最好、2017 年起最差
    book[1, dates.year >= 2017] -= 0.01
    configs = [("M", Spec("DB", 1, b)) for b in (1, 1.5, 2)]
    w = stats.walk_forward(book, book.copy(), dates, stats.neighbor_index(configs), "raw", years=(2017, 2018))
    assert w["picks"]["2017"] == 1 and len(w["dates"]) == int((dates.year >= 2017).sum())
    assert np.array_equal(w["book"][: int((dates.year == 2017).sum())], book[1, dates.year == 2017])


def test_best_picker_breaks_near_ties_by_the_interval_lower_bound() -> None:
    walked = {"smooth": {"excess": 10.0, "ci": [1.0, 19.0]}, "ir": {"excess": 9.8, "ci": [4.0, 15.0]},
              "raw": {"excess": 8.0, "ci": [6.0, 10.0]}}
    assert stats.best_of(walked) == "ir"


def test_series_stats_of_a_constant_daily_return() -> None:
    dates = pd.bdate_range("2020-01-01", periods=242)
    out = stats.series_stats(np.full(242, 0.001), np.full(242, 0.0005), dates)
    assert out["excess"] == pytest.approx(0.0005 * 242 * 100, abs=0.01)
    assert out["cagr"] == pytest.approx((1.001 ** 242 - 1) * 100, abs=0.01) and out["max_dd"] == 0.0


def test_tomorrows_list_does_not_spend_the_money_of_a_halted_holding() -> None:
    """今天停牌的持仓到了该卖的时候：清单写"复牌后卖"，但这笔钱和这个位置都不算空出来（模拟器也是拿着），不拿它去买别的。"""
    from rebound.backtest.plan import next_orders
    from rebound.backtest.sim import Position

    n = 6
    open_ = np.full((n, 2), 10.0)
    open_[-1, 0] = np.nan  # S0 今天停牌：没有价、没有分
    m = make_market(open_, open_.copy())
    cands = {d: np.array([1]) if d == n - 1 else np.array([0, 1]) for d in range(n)}
    rank = np.full((n, 2), np.inf, dtype=np.float32)
    rank[:, 1], rank[: n - 1, 0] = 1, 2
    rule = parse("DB-K1-b2").rule
    cost = CostModel(capital=20_000.0, commission=0.0005, min_commission=5.0, stamp=0.0005, transfer=0.00001)
    held = Position(j=0, shares=1900.0, entry_day=1, entry_px=10.0, cost_basis=19_000.0, signal_day=0, tranche=0,
                    exit_due=-1, last_px=10.0)
    plan = next_orders(m, {}, cands, rank, rule, cost, [held], [990.0], n - 1)
    assert [s["ts_code"] for s in plan["sells"]] == ["S0"] and plan["sells"][0]["halted"]
    assert "复牌后" in plan["sells"][0]["reason"] and plan["sells"][0]["shares"] is None
    assert plan["buys"] == [] and plan["cash"] == 990.0


O1C5 = Cell("open", 5, "close")


def test_db_with_a_close_exit_sells_at_the_close_and_refills_at_the_next_open():
    """DB 的卖腿是收盘：前一天的排名掉出缓冲，当天尾盘卖；空出来的位置第二天开盘按前一天的排名补（钱已经在账上）。
    开盘 10、收盘 11，第 d 天排第一的是 S(d mod 3)，K = 1、缓冲 1：每天都有一只掉出去。"""
    n = 8
    m = make_market(np.full((n, 3), 10.0), np.full((n, 3), 11.0))
    d, j = np.meshgrid(np.arange(n), np.arange(3), indexing="ij")
    cands, rank = candidates(m, pd.DataFrame({"di": d.ravel(), "ji": j.ravel(),
                                              "score": (3 - (j - d) % 3).ravel().astype(float)}))
    res = simulate(m, cands, rank, Rule("DB", 1, O1C5, buffer=1), (0, 0), 0, n - 2)
    trades = res.trades[["j", "entry_day", "exit_day", "entry_px", "exit_px"]].values.tolist()
    assert trades == [[0, 1, 2, 10.0, 11.0], [2, 3, 4, 10.0, 11.0], [1, 5, 6, 10.0, 11.0]]
    assert res.holdings[1:n].tolist() == [1, 0, 1, 0, 1, 0, 1]


def test_tomorrows_list_with_a_close_exit_buys_only_the_empty_slots_with_the_cash_on_hand():
    """尾盘卖的清单：明天开盘只补现在空着的位置、只用现在账上的钱；该卖的明天尾盘卖，卖的钱后天开盘才用。"""
    from rebound.backtest.plan import next_orders
    from rebound.backtest.sim import Position

    n = 6
    m = make_market(np.full((n, 3), 10.0))
    cands = {d: np.array([1, 2, 0]) for d in range(n)}
    rank = np.tile(np.array([3, 1, 2], dtype=np.float32), (n, 1))
    held = Position(j=0, shares=1000.0, entry_day=2, entry_px=10.0, cost_basis=10_005.0, signal_day=1, tranche=0,
                    exit_due=-1, last_px=10.0)
    plan = next_orders(m, {}, cands, rank, Rule("DB", 2, O1C5, buffer=1), CostModel(capital=20_000.0), [held],
                       [10_000.0], n - 1)
    assert [s["ts_code"] for s in plan["sells"]] == ["S0"] and not plan["sells"][0]["halted"]
    assert [b["ts_code"] for b in plan["buys"]] == ["S1"] and plan["buys"][0]["shares"] == 900
    assert plan["cash"] == plan["buy_cash"] == 10_000.0 and (plan["buy_leg"], plan["sell_leg"]) == ("open", "close")


def _upto(m: Market, last: int) -> Market:
    """行情只留到第 last 天（含）：模拟到"今天收盘"，清单按今天收盘后出。"""
    cut = slice(0, last + 1)
    pick = lambda legs: {leg: a[cut] for leg, a in legs.items()}  # noqa: E731
    return Market(dates=m.dates[cut], codes=m.codes, price=pick(m.price), buy_blocked=pick(m.buy_blocked),
                  sell_blocked=pick(m.sell_blocked), eligible=m.eligible[cut], industry=m.industry,
                  raw=None if m.raw is None else pick(m.raw))


CONSISTENCY_RULES = {
    "DB-close": Rule("DB", 2, O1C5, buffer=1.5), "FH-close": Rule("FH", 1, Cell("open", 3, "close")),
    "DB-open": Rule("DB", 2, Cell("open", 5, "open"), buffer=1.5), "FH-open": Rule("FH", 1, Cell("open", 3, "open")),
}


def _next_day(m, cands, rank, rule, cost, today):
    """今天收盘后的清单，和模拟器第二天真做的：（清单、第二天买进的（代码, 股数）、第二天卖出的代码、今天收盘的账户）。
    尾盘卖的清单另核对 buy_cash：明天开盘买的那一批手上的钱（FH 分批时不是各批合计）。"""
    from rebound.backtest.plan import next_orders

    before = simulate(_upto(m, today), cands, rank, rule, cost, 0, today - 1)
    plan = next_orders(_upto(m, today), {}, cands, rank, rule, cost, before.positions, before.tranche_cash, today)
    after = simulate(_upto(m, today + 1), cands, rank, rule, cost, 0, today)
    bought = sorted((str(m.codes[p.j]), round(p.shares)) for p in after.positions if p.entry_day == today + 1)
    sold = sorted(str(m.codes[j]) for j in after.trades[after.trades.exit_day == today + 1].j)
    if plan.get("sell_leg") == "close":
        assert plan["buy_cash"] == round(before.tranche_cash[plan["tranche"]], 2)
    return plan, bought, sold, before


@pytest.mark.parametrize("key", list(CONSISTENCY_RULES))
def test_tomorrows_list_is_what_the_simulator_trades_next_day(key):
    """价格不动、没有涨跌停时，今天收盘后的清单就是模拟器第二天买、卖的那几笔（名字、股数都一样）：开盘卖、尾盘卖两种卖腿。"""
    rule, rng = CONSISTENCY_RULES[key], np.random.default_rng(7)
    n = 24
    m = make_market(np.full((n, 6), 10.0))
    cands, rank = random_signals(rng, m)
    cost = CostModel(capital=50_000.0)
    seen_buys = seen_sells = 0
    for today in range(3, n - 2):
        plan, bought, sold, _ = _next_day(m, cands, rank, rule, cost, today)
        assert sorted((b["ts_code"], b["shares"]) for b in plan["buys"]) == bought
        assert sorted(s["ts_code"] for s in plan["sells"]) == sold
        seen_buys, seen_sells = seen_buys + len(bought), seen_sells + len(sold)
    assert seen_buys > 0 and seen_sells > 0


@pytest.mark.parametrize("key", ["DB-close", "FH-close"])
def test_tomorrows_close_exit_list_keeps_the_slots_of_sells_stuck_at_limit_down(key):
    """尾盘卖时有跌停卖不出的：卖不出的票接着列在清单上、位置不空出来。模拟器第二天尾盘卖掉的 = 清单列的减去第二天尾盘跌停的；
    开盘买的那几笔不受影响（名字、股数都一样）。"""
    rule, rng = CONSISTENCY_RULES[key], np.random.default_rng(11)
    n = 40
    dn = rng.random((n, 6)) < 0.15
    m = make_market(np.full((n, 6), 10.0), dn_close=dn)
    cands, rank = random_signals(rng, m)
    cost = CostModel(capital=50_000.0)
    stuck_and_listed = 0
    for today in range(3, n - 2):
        plan, bought, sold, before = _next_day(m, cands, rank, rule, cost, today)
        held = before.positions
        listed = [s["ts_code"] for s in plan["sells"]]
        blocked = {str(m.codes[j]) for j in np.flatnonzero(dn[today + 1])}
        assert sorted((b["ts_code"], b["shares"]) for b in plan["buys"]) == bought
        assert sorted(c for c in listed if c not in blocked) == sold
        stuck_and_listed += sum(p.stuck and p.tranche == plan["tranche"] and str(m.codes[p.j]) in listed
                                for p in held)
    assert stuck_and_listed > 0


def test_fh_cannot_buy_at_the_open_and_sell_at_the_same_close():
    from rebound.backtest.plan import next_orders

    m = make_market(np.full((6, 2), 10.0))
    rule = Rule("FH", 1, Cell("open", 1, "close"))
    with pytest.raises(ValueError):
        simulate(m, {0: np.array([0, 1])}, None, rule, (0, 0), 0, 3)
    with pytest.raises(ValueError):
        next_orders(m, {}, {}, None, rule, CostModel(), (), (), 5)


def test_tomorrows_list_rejects_rules_that_buy_at_the_close():
    from rebound.backtest.plan import next_orders

    m = make_market(np.full((4, 2), 10.0))
    with pytest.raises(ValueError):
        next_orders(m, {}, {}, None, Rule("FH", 1, Cell("close", 2, "open")), CostModel(), (), (), 3)
