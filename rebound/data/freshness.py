"""新鲜度闸门：选股只允许跑在**最新已收盘交易日**的数据上。

这条要求不能靠"记得先跑 sync"来满足。库停在两个交易日前、而选股默认取"本地最新交易日"时，
**它会安静地拿两天前的数据选股，并把结果当今天的结果展示**。所以每个选股入口都要过闸门。

四类拒绝，顺序即逻辑：

1. `calendar_missing` —— 本地日历里一个开市日都没有。不知道今天是哪天，
   就不许选股。
2. `calendar_stale` —— 日历最新日期早于今天。此时"最新已收盘交易日"算不出来；
   **绝不沿用上一个开市日**：否则"今天其实是交易日"会把今天的批次写到昨天的日期上。
3. `not_closed` —— 今天开市但还没到收盘时刻，今天的数据必然还不完整。
4. `data_missing` / `data_thin` —— 该交易日某张表没数据或行数不足。

为什么窗口是一段而不是一天：特征要看前 150 根 K 线（`data.history_bars`），只检查 `as_of` 当天
是不够的。窗口里缺任何一天，特征都会静默变形 —— `r20` 少一根 K 线，而结果看起来完全正常。
所以闸门按**窗口**检查，缺哪一天都要报出来。

"今天""现在"按北京时间（`beijing_now()`），不看这台电脑的时区。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import TYPE_CHECKING

from rebound.clock import beijing_now
from rebound.data.errors import DataError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from rebound.data.db import WriteStore

# 今日日线算定稿的时刻。收盘 15:00 + 结算时间。
DEFAULT_CLOSE_TIME = time(15, 30)

OK = "ok"
CALENDAR_MISSING = "calendar_missing"
CALENDAR_STALE = "calendar_stale"
NOT_CLOSED = "not_closed"
DATA_MISSING = "data_missing"
DATA_THIN = "data_thin"


@dataclass(frozen=True, slots=True)
class MissingSession:
    """某张表缺了某个交易日。`rows == 0` 是"完全没有"，否则是"行数不足"。"""

    table: str
    trade_date: date
    rows: int
    required: int

    def as_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "trade_date": self.trade_date.isoformat(),
            "rows": self.rows,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class Freshness:
    ok: bool
    reason: str
    today: date
    now: time
    expected: date | None = None
    local_max: date | None = None
    calendar_max: date | None = None
    missing: tuple[MissingSession, ...] = ()
    windows: tuple[tuple[str, int], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "today": self.today.isoformat(),
            "now": self.now.strftime("%H:%M:%S"),
            "expected_session": self.expected.isoformat() if self.expected else None,
            "local_max": self.local_max.isoformat() if self.local_max else None,
            "calendar_max": self.calendar_max.isoformat() if self.calendar_max else None,
            "missing": [item.as_dict() for item in self.missing],
            "windows": {name: size for name, size in self.windows},
        }


class StaleDataError(DataError):
    """数据不是最新的，拒绝选股。"""

    def __init__(self, freshness: Freshness) -> None:
        super().__init__(describe(freshness))
        self.freshness = freshness


def expected_session(
    open_dates: Iterable[date], *, today: date, now: time, close_time: time = DEFAULT_CLOSE_TIME
) -> date | None:
    """权威的"最新已收盘交易日"。纯函数。

    今天是开市日但还没收盘时，答案是**上一个**开市日，不是今天。
    没有上一个（日历里只有今天）时返回 `None`。
    """
    past = [item for item in open_dates if item <= today]
    if not past:
        return None
    last = past[-1]
    if last == today and now < close_time:
        return past[-2] if len(past) > 1 else None
    return last


def check_freshness(
    store: WriteStore,
    *,
    exchange: str = "SSE",
    min_rows: int = 1000,
    windows: Mapping[str, int] | None = None,
    today: date | None = None,
    now: time | None = None,
    close_time: time = DEFAULT_CLOSE_TIME,
) -> Freshness:
    """过一遍闸门。`windows` 是"每张表要往前看多少个开市日"。

    默认窗口是 `{"daily": 1}` —— 只保证截面那天有数据；
    真正调用的入口（CLI）会传 `{"daily": history_bars}`。
    """
    moment = beijing_now()  # 北京时间：今天收盘没有，不看这台电脑的时区
    today = today or moment.date()
    now = now or moment.time()
    requested: dict[str, int] = dict(windows or {"daily": 1})
    given = tuple((name, int(size)) for name, size in requested.items())

    open_dates = store.open_dates(exchange=exchange)
    if not open_dates:
        return Freshness(
            False, CALENDAR_MISSING, today, now, windows=given
        )
    calendar_max = open_dates[-1]
    local_max = store.coverage()
    daily_span = next((item for item in local_max if item.table == "daily"), None)
    local_last = daily_span.last if daily_span else None

    if calendar_max < today:
        return Freshness(
            False, CALENDAR_STALE, today, now,
            local_max=local_last, calendar_max=calendar_max, windows=given,
        )

    expected = expected_session(open_dates, today=today, now=now, close_time=close_time)
    if expected is None:
        return Freshness(
            False, NOT_CLOSED, today, now,
            local_max=local_last, calendar_max=calendar_max, windows=given,
        )

    up_to = [item for item in open_dates if item <= expected]
    missing: list[MissingSession] = []
    for table, size in requested.items():
        if size <= 0:
            continue
        if not store.has_table(table):
            missing.append(MissingSession(table, expected, 0, min_rows))
            continue
        counts = store.dates_with_rows(table)
        for trade_date in up_to[-int(size):]:
            rows = counts.get(trade_date, 0)
            if rows < min_rows:
                missing.append(MissingSession(table, trade_date, rows, min_rows))

    if not missing:
        return Freshness(
            True, OK, today, now,
            expected=expected, local_max=local_last, calendar_max=calendar_max, windows=given,
        )
    reason = DATA_MISSING if any(item.rows == 0 for item in missing) else DATA_THIN
    return Freshness(
        False, reason, today, now,
        expected=expected, local_max=local_last, calendar_max=calendar_max,
        missing=tuple(missing), windows=given,
    )


_REASON_TEXT = {
    CALENDAR_MISSING: "本地交易日历是空的",
    CALENDAR_STALE: "本地交易日历已经落后于今天，算不出今天的交易日",
    NOT_CLOSED: "今天还没到收盘时刻，今天的数据必然不完整",
    DATA_MISSING: "本地行情缺了需要的交易日",
    DATA_THIN: "本地行情某些交易日的行数不足",
}


def recommended_sync_days(
    freshness: Freshness, *, floor: int = 10, cap: int = 300
) -> int:
    """该回补多少个开市日才可能把缺口填上。

    不能只看"缺了几天的个数"：缺口可能散布在很老的日期上，而 `sync --days N`
    是从今天往回 N 个开市日 —— 要覆盖到**最老的那个缺口**才行。
    """
    if not freshness.missing or freshness.expected is None:
        return floor
    oldest = min(item.trade_date for item in freshness.missing)
    calendar_span = (freshness.expected - oldest).days
    approx_sessions = int(calendar_span * 5 / 7) + 5  # 一周 5 个开市日
    return int(min(max(approx_sessions, floor), cap))


def describe(freshness: Freshness) -> str:
    """给人和给日志都读得懂的一句话，外加怎么修。"""
    lines = [f"拒绝选股：{_REASON_TEXT.get(freshness.reason, freshness.reason)}。"]
    lines.append(
        f"  现在 {freshness.today} {freshness.now.strftime('%H:%M')}，"
        f"最新已收盘交易日应为 "
        f"{freshness.expected if freshness.expected else '（算不出来）'}。"
    )
    if freshness.calendar_max is not None:
        lines.append(f"  本地日历最新到 {freshness.calendar_max}。")
    if freshness.local_max is not None:
        lines.append(f"  本地 daily 最新到 {freshness.local_max}。")
    if freshness.missing:
        by_table: dict[str, list[date]] = {}
        for item in freshness.missing:
            by_table.setdefault(item.table, []).append(item.trade_date)
        for table, dates in by_table.items():
            shown = "、".join(str(item) for item in sorted(dates)[:6])
            more = "…" if len(dates) > 6 else ""
            lines.append(f"  {table} 缺 {shown}{more}（共 {len(dates)} 个交易日）")
    lines.append(
        f"  修复：python -m rebound sync --days {recommended_sync_days(freshness)}"
        "（幂等，已有的跳过）"
    )
    return "\n".join(lines)


def assert_fresh(freshness: Freshness) -> date:
    """闸门通过则返回权威交易日，否则抛。"""
    if not freshness.ok or freshness.expected is None:
        raise StaleDataError(freshness)
    return freshness.expected
