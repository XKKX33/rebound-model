"""反弹模型的标签（`rebound/labels.py`）：逐笔收益的顺延规则、截面百分位、日期轴。"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from rebound.data.errors import DataError
from rebound.labels import (
    CARRIED, CARRY_MAX, CELLS, ENTRY_BLOCKED, ENTRY_MISSING, EXIT_UNRESOLVED, MODEL_CELL, OK, Cell, PriceMatrix,
    carry_trades, empty_matrix, fill_matrix, percentile, price_matrix,
)

N_DAYS, N_CODES = 40, 3


def _matrix() -> PriceMatrix:
    """开盘价 = 10 + 天数，收盘价 = 开盘 + 0.5：每笔收益都能手算。"""
    days = np.arange(N_DAYS, dtype=float)[:, None] + np.zeros((1, N_CODES))
    flags = lambda: {"open": np.zeros((N_DAYS, N_CODES), bool), "close": np.zeros((N_DAYS, N_CODES), bool)}  # noqa: E731
    return PriceMatrix(
        dates=pd.DatetimeIndex(pd.date_range("2024-01-01", periods=N_DAYS, freq="B")),
        codes=np.array([f"60000{i}.SH" for i in range(N_CODES)]),
        price={"open": 10 + days, "close": 10.5 + days},
        buy_blocked=flags(), sell_blocked=flags(),
    )


def _one(m: PriceMatrix, cell: Cell, d: int, j: int = 0) -> tuple[float, int, int]:
    got = carry_trades(m, np.array([d]), np.array([j]), cell)
    return float(got["ret"][0]), int(got["status"][0]), int(got["exit_day"][0])


def test_the_model_trades_t1_open_to_t5_open_only() -> None:
    """十年实验只用 o1o5（2026-10-31 定：o1c5、c1o5 不再训）。"""
    assert MODEL_CELL.name == "o1o5" and CELLS == (MODEL_CELL,)
    assert CARRY_MAX == 20


def test_a_clean_trade_buys_t1_and_sells_t5() -> None:
    m = _matrix()
    ret, status, exit_day = _one(m, Cell("open", 5, "open"), d=3)
    assert (status, exit_day) == (OK, 8) and ret == pytest.approx((10 + 8) / (10 + 4) - 1)
    ret, status, exit_day = _one(m, Cell("close", 5, "open"), d=3)
    assert (status, exit_day) == (OK, 8) and ret == pytest.approx((10 + 8) / (10.5 + 4) - 1)


def test_no_entry_price_or_a_sealed_limit_up_makes_no_trade() -> None:
    m = _matrix()
    m.price["open"][4, 0] = np.nan  # T+1 停牌
    m.buy_blocked["open"][4, 1] = True  # T+1 一字涨停
    cell = Cell("open", 5, "open")
    assert _one(m, cell, 3, 0)[1:] == (ENTRY_MISSING, -1) and np.isnan(_one(m, cell, 3, 0)[0])
    assert _one(m, cell, 3, 1)[1:] == (ENTRY_BLOCKED, -1)


def test_a_blocked_or_halted_exit_is_carried_to_the_next_session() -> None:
    m = _matrix()
    m.sell_blocked["open"][8, 0] = True  # T+5 跌停开盘
    m.price["open"][9, 0] = np.nan  # T+6 停牌
    ret, status, exit_day = _one(m, Cell("open", 5, "open"), d=3)
    assert (status, exit_day) == (CARRIED, 10) and ret == pytest.approx(20 / 14 - 1)


def test_an_exit_still_blocked_after_twenty_extra_sessions_stays_unresolved() -> None:
    m = _matrix()
    m.sell_blocked["open"][3 + 5: 3 + 5 + CARRY_MAX, 0] = True  # T+5 … T+24 都卖不出
    assert _one(m, Cell("open", 5, "open"), d=3)[1:] == (CARRIED, 3 + 5 + CARRY_MAX)
    m.sell_blocked["open"][3 + 5 + CARRY_MAX, 0] = True
    ret, status, exit_day = _one(m, Cell("open", 5, "open"), d=3)
    assert (status, exit_day) == (EXIT_UNRESOLVED, -1) and np.isnan(ret)


def test_an_off_axis_signal_is_refused_not_read_from_the_last_row() -> None:
    with pytest.raises(DataError, match="下标为 -1"):
        carry_trades(_matrix(), np.array([3, -1]), np.array([0, 0]), MODEL_CELL)
    with pytest.raises(DataError, match="下标为 -1"):
        carry_trades(_matrix(), np.array([3]), np.array([-1]), MODEL_CELL)


def test_a_trade_running_past_the_data_is_unresolved() -> None:
    assert _one(_matrix(), Cell("open", 5, "open"), d=N_DAYS - 3)[1:] == (EXIT_UNRESOLVED, -1)


def test_percentile_ranks_within_each_day_and_ignores_gaps_and_unknown_rows() -> None:
    day = np.array([0, 0, 0, 0, 1, 1])
    value = np.array([0.1, 0.3, 0.3, np.nan, -0.2, 0.5])
    got = percentile(value, day)
    assert np.allclose(got[[0, 1, 2, 4, 5]], [1 / 3, 2.5 / 3, 2.5 / 3, 0.5, 1.0]) and np.isnan(got[3])
    # 训练段起点之后才卖出的行不知道收益：既没有标签，也不参与别人的百分位（研究版的轻微泄漏）
    known = np.array([True, False, True, True, True, True])
    got = percentile(value, day, known=known)
    assert np.isnan(got[1]) and np.allclose(got[[0, 2]], [0.5, 1.0])


def _panel(days: list[date]) -> pd.DataFrame:
    rows = [{"ts_code": code, "trade_date": pd.Timestamp(d), "o": 1.0 + i, "c": 2.0 + i,
             "up_open": False, "up_close": i == 1, "dn_open": False, "dn_close": False}
            for i, d in enumerate(days) for code in ("600001.SH", "000002.SZ")]
    return pd.DataFrame(rows).astype({"ts_code": "category"})


def test_price_matrix_places_every_row_on_the_trading_calendar() -> None:
    sessions = [date(2024, 1, 2) + timedelta(days=i) for i in range(3)]
    m = price_matrix(_panel(sessions), sessions)
    assert list(m.codes) == ["000002.SZ", "600001.SH"] and len(m.dates) == 3
    assert m.price["open"][2, 1] == 3.0 and m.price["close"][0, 0] == 2.0
    assert m.buy_blocked["close"][1].all() and not m.buy_blocked["open"].any()


def test_filling_rows_off_the_matrix_axes_is_refused() -> None:
    """按年填矩阵时，轴是一开始按库建的：面板里出现轴外的日子或代码，说明两次读库之间库变了。"""
    sessions = [date(2024, 1, 2) + timedelta(days=i) for i in range(3)]
    matrix = empty_matrix(pd.DatetimeIndex([pd.Timestamp(d) for d in sessions[:2]]), np.array(["000002.SZ", "600001.SH"]))
    with pytest.raises(DataError, match="不在价格矩阵的轴上"):
        fill_matrix(matrix, _panel(sessions))
    fill_matrix(matrix, _panel(sessions[:2]))
    assert np.isfinite(matrix.price["open"]).all()


def test_price_matrix_refuses_a_session_without_quotes() -> None:
    """日期轴有洞，"往后第 k 个开市日"就整体错位，收益算成别的日子的：必须报错。"""
    sessions = [date(2024, 1, 2) + timedelta(days=i) for i in range(3)]
    with pytest.raises(DataError, match="2024-01-03"):
        price_matrix(_panel([sessions[0], sessions[2]]), sessions)
