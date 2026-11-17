"""新鲜度闸门：选股的时候要确保是最新的数据。

这组测试的重点不是"函数返回了什么"，而是**它拒绝的东西**：
库落后两个交易日时照常选股并当今天展示，在这里必须变成硬失败。
"""

from __future__ import annotations

from datetime import date, time
from pathlib import Path

import pytest

from rebound.data import freshness as fresh
from rebound.data.db import TableCoverage, WriteStore

# 2026-10-14 周三、10-15 周四、10-16 周五，都是开市日。
D14 = date(2026, 10, 14)
D15 = date(2026, 10, 15)
D16 = date(2026, 10, 16)
D17 = date(2026, 10, 17)  # 周六，不开市
SESSIONS = (D14, D15, D16)


class FakeStore:
    """只实现闸门真正用到的那四个方法，让测试不依赖 DuckDB。"""

    def __init__(
        self,
        sessions: tuple[date, ...],
        *,
        rows: dict[str, dict[date, int]] | None = None,
        tables: tuple[str, ...] = ("daily", "adj_factor"),
        coverage: tuple[TableCoverage, ...] | None = None,
    ) -> None:
        self._sessions = tuple(sessions)
        self._rows = rows if rows is not None else {}
        self._tables = set(tables)
        self._coverage = coverage

    def open_dates(self, *, exchange: str = "SSE") -> list[date]:
        return list(self._sessions)

    def has_table(self, table: str) -> bool:
        return table in self._tables

    def dates_with_rows(self, table: str) -> dict[date, int]:
        return dict(self._rows.get(table, {}))

    def coverage(self) -> tuple[TableCoverage, ...]:
        if self._coverage is not None:
            return self._coverage
        out: list[TableCoverage] = []
        for table in sorted(self._tables):
            dates = sorted(self._rows.get(table, {}))
            if not dates:
                continue
            out.append(
                TableCoverage(table, sum(self._rows[table].values()), dates[0], dates[-1])
            )
        return tuple(out)


def _full(rows_per_day: int = 5000) -> dict[str, dict[date, int]]:
    return {
        "daily": {item: rows_per_day for item in SESSIONS},
        "adj_factor": {item: rows_per_day for item in SESSIONS},
    }


# --------------------------------------------------------------------------- #
# expected_session：纯函数，边界就是逻辑本身
# --------------------------------------------------------------------------- #


def test_before_close_the_answer_is_the_previous_session() -> None:
    assert (
        fresh.expected_session(SESSIONS, today=D16, now=time(14, 0))
        == D15
    )


def test_at_close_time_today_counts_as_closed() -> None:
    """`now < close_time` 才是"没收盘"；正好 15:30 算已收盘。"""
    assert (
        fresh.expected_session(SESSIONS, today=D16, now=time(15, 30))
        == D16
    )


def test_after_close_the_answer_is_today() -> None:
    assert fresh.expected_session(SESSIONS, today=D16, now=time(18, 0)) == D16


def test_non_trading_day_falls_back_to_the_last_open_day() -> None:
    """周六问"最新已收盘交易日"，答案是周五 —— 与时刻无关。"""
    assert fresh.expected_session(SESSIONS, today=D17, now=time(9, 0)) == D16


def test_calendar_with_only_today_before_close_has_no_answer() -> None:
    assert fresh.expected_session((D16,), today=D16, now=time(9, 0)) is None


def test_empty_calendar_has_no_answer() -> None:
    assert fresh.expected_session((), today=D16, now=time(18, 0)) is None


# --------------------------------------------------------------------------- #
# check_freshness：四类拒绝，顺序即逻辑
# --------------------------------------------------------------------------- #


def test_empty_calendar_is_calendar_missing() -> None:
    result = fresh.check_freshness(FakeStore(()), today=D16, now=time(18, 0))
    assert result.reason == fresh.CALENDAR_MISSING
    assert result.ok is False
    assert result.expected is None


def test_calendar_behind_today_is_calendar_stale() -> None:
    """日历最新只到 10-14，而今天是 10-16：算不出今天，绝不用 10-14 顶替。"""
    result = fresh.check_freshness(FakeStore((D14,)), today=D16, now=time(18, 0))
    assert result.reason == fresh.CALENDAR_STALE
    assert result.calendar_max == D14
    assert result.expected is None


def test_today_still_open_is_not_closed() -> None:
    result = fresh.check_freshness(
        FakeStore((D16,), rows=_full()), today=D16, now=time(10, 0)
    )
    assert result.reason == fresh.NOT_CLOSED


def test_fresh_data_passes_and_reports_the_authoritative_session() -> None:
    store = FakeStore(SESSIONS, rows=_full())
    result = fresh.check_freshness(
        store, windows={"daily": 3, "adj_factor": 3}, today=D16, now=time(18, 0)
    )
    assert result.ok is True
    assert result.reason == fresh.OK
    assert result.expected == D16
    assert fresh.assert_fresh(result) == D16


def test_a_session_with_zero_rows_is_data_missing() -> None:
    rows = _full()
    rows["daily"][D15] = 0
    result = fresh.check_freshness(
        FakeStore(SESSIONS, rows=rows), windows={"daily": 3}, today=D16, now=time(18, 0)
    )
    assert result.reason == fresh.DATA_MISSING
    assert [(item.table, item.trade_date, item.rows) for item in result.missing] == [
        ("daily", D15, 0)
    ]


def test_a_session_with_too_few_rows_is_data_thin() -> None:
    """有数据但不够 —— 与"完全没有"是两件事，报的文案也不同。"""
    rows = _full()
    rows["daily"][D14] = 10
    result = fresh.check_freshness(
        FakeStore(SESSIONS, rows=rows), windows={"daily": 3}, today=D16, now=time(18, 0)
    )
    assert result.reason == fresh.DATA_THIN
    assert result.missing[0].rows == 10
    assert result.missing[0].required == 1000


def test_a_missing_table_counts_as_a_hole() -> None:
    """表还没建出来（首次搭库）不能算通过。"""
    result = fresh.check_freshness(
        FakeStore(SESSIONS, rows={"daily": {item: 5000 for item in SESSIONS}},
                  tables=("daily",)),
        windows={"daily": 1, "adj_factor": 1},
        today=D16,
        now=time(18, 0),
    )
    assert result.reason == fresh.DATA_MISSING
    assert any(item.table == "adj_factor" for item in result.missing)


def test_only_the_window_is_checked() -> None:
    """窗口外的老缺口不拦选股 —— 否则一天补不齐就永远选不了股。"""
    rows = _full()
    rows["daily"][D14] = 0
    result = fresh.check_freshness(
        FakeStore(SESSIONS, rows=rows), windows={"daily": 2}, today=D16, now=time(18, 0)
    )
    assert result.ok is True
    assert result.expected == D16


def test_calendar_and_local_max_are_reported_even_when_stale() -> None:
    """拒绝时要能把"本地最新到哪天"说清楚，否则用户只知道失败不知道差多少。"""
    store = FakeStore(
        SESSIONS,
        rows={"daily": {D14: 5000, D15: 0, D16: 0}},
        coverage=(TableCoverage("daily", 5000, D14, D14),),
    )
    result = fresh.check_freshness(
        store, windows={"daily": 3}, today=D16, now=time(18, 0)
    )
    assert result.calendar_max == D16
    assert result.local_max == D14
    text = fresh.describe(result)
    assert "本地 daily 最新到 2026-10-14" in text


# --------------------------------------------------------------------------- #
# recommended_sync_days：缺口在很老的位置时，只补几天是不够的
# --------------------------------------------------------------------------- #


def test_recommendation_falls_back_to_the_floor_when_everything_is_fresh() -> None:
    result = fresh.check_freshness(
        FakeStore(SESSIONS, rows=_full()), windows={"daily": 3}, today=D16, now=time(18, 0)
    )
    assert fresh.recommended_sync_days(result, floor=10) == 10


def test_recommendation_reaches_back_to_the_oldest_hole() -> None:
    """10-05 有洞、今天 10-16：只补 3 天补不到它。"""
    rows = _full()
    holiday = date(2026, 10, 5)
    sessions = (holiday, D14, D15, D16)
    rows["daily"] = {item: 5000 for item in sessions}
    rows["daily"][holiday] = 0
    result = fresh.check_freshness(
        FakeStore(sessions, rows=rows), windows={"daily": 4}, today=D16, now=time(18, 0)
    )
    assert result.reason == fresh.DATA_MISSING
    # 日历跨度 11 天 ≈ 7 个开市日，+5 余量 = 12；必须 ≥ 覆盖到 10-05 的天数。
    assert fresh.recommended_sync_days(result, floor=10) == 12


def test_recommendation_respects_the_cap() -> None:
    rows = _full()
    old = date(2020, 1, 2)
    sessions = (old, D16)
    rows["daily"] = {old: 0, D16: 5000}
    result = fresh.check_freshness(
        FakeStore(sessions, rows=rows), windows={"daily": 2}, today=D16, now=time(18, 0)
    )
    assert fresh.recommended_sync_days(result, floor=10, cap=300) == 300


# --------------------------------------------------------------------------- #
# StaleDataError：拒绝必须带上足够的信息
# --------------------------------------------------------------------------- #


def test_stale_data_error_carries_the_report_and_a_fix_command() -> None:
    store = FakeStore(
        SESSIONS, rows={"daily": {D14: 5000, D15: 0, D16: 0}, "adj_factor": {}}
    )
    result = fresh.check_freshness(
        store, windows={"daily": 3}, today=D16, now=time(18, 0)
    )
    with pytest.raises(fresh.StaleDataError) as info:
        fresh.assert_fresh(result)
    assert info.value.freshness is result
    text = str(info.value)
    assert "python -m rebound sync --days" in text
    assert "2026-10-15" in text


def test_assert_fresh_refuses_when_there_is_no_expected_session() -> None:
    result = fresh.check_freshness(FakeStore(()), today=D16, now=time(18, 0))
    with pytest.raises(fresh.StaleDataError):
        fresh.assert_fresh(result)
