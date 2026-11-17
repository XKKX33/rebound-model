"""增量同步 + 缺口检测。

两条纪律：

1. **按需拉**。已经完整落库的交易日不再拉（`daily` / `daily_limit` / `adj_factor`
   三张表各自判断）。某张表只覆盖了一部分交易日、却没有任何地方能说出这件事，是最难查的
   数据问题 —— 这里的 `audit()` 就是说出这件事的地方。
2. **一天是一个事务**。校验通过才写，写就一起写。半截状态（日线写了、复权因子没写）
   在下游无法与"这天本来就缺"区分。
3. **一天失败不牵连别的天**。上游限流或那天还没发布，只跳过那一天并记下原因
   （`fetch_failed`）；整趟回补不会因为一次限流白跑。跳过的天由新鲜度闸门兜住 ——
   少了最近的数据就拒绝选股，而不是安静地拿旧数据当今天。

`calendar_lookahead_days` 不是凑数：交易日的**未来**几天必须先在本地日历里，
否则 T+1 开盘买、T+5 开盘卖这些标签和回测用到的日子算不出来。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from rebound.clock import beijing_now
from rebound.data.db import TableCoverage, WriteStore
from rebound.data.errors import DataError, NetworkError, ValidationError
from rebound.data.tushare_client import MarketDataSource

# 回补窗口的日历起点：`days` 个开市日最坏情况下跨越约 days*2 个自然日（周末），
# 再加 buffer 兜住长假。
_CALENDAR_BACKFILL_SLACK_DAYS = 14


@dataclass(frozen=True, slots=True)
class DateSync:
    """某个交易日实际写入了什么。"""

    trade_date: date
    daily_rows: int = 0
    limit_rows: int = 0
    adj_rows: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "trade_date": self.trade_date.isoformat(),
            "daily": self.daily_rows,
            "daily_limit": self.limit_rows,
            "adj_factor": self.adj_rows,
        }


@dataclass(frozen=True, slots=True)
class SkippedDate:
    """某个交易日没同步，以及为什么。理由必须显式，不许沉默。"""

    trade_date: date
    reason: str
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "trade_date": self.trade_date.isoformat(),
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class SyncReport:
    calendar_rows: int = 0
    stock_basic_rows: int = 0
    synced: tuple[DateSync, ...] = ()
    skipped: tuple[SkippedDate, ...] = ()
    coverage: tuple[TableCoverage, ...] = ()
    name_rows: int = 0

    @property
    def written_rows(self) -> int:
        return self.calendar_rows + self.stock_basic_rows + self.name_rows + sum(
            item.daily_rows + item.limit_rows + item.adj_rows
            for item in self.synced
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "calendar_rows": self.calendar_rows,
            "stock_basic_rows": self.stock_basic_rows,
            "name_rows": self.name_rows,
            "written_rows": self.written_rows,
            "synced": [item.as_dict() for item in self.synced],
            "skipped": [item.as_dict() for item in self.skipped],
            "coverage": [item.as_dict() for item in self.coverage],
        }


@dataclass(frozen=True, slots=True)
class AuditReport:
    """体检结论：哪张表在哪个区间有洞、落后多少、结论健不健康。"""

    exchange: str
    min_rows: int
    today: date
    calendar: tuple[date, date] | None
    coverage: tuple[TableCoverage, ...]
    holes: dict[str, tuple[date, ...]] = field(default_factory=dict)
    backlog: tuple[date, ...] = ()
    empty_tables: tuple[str, ...] = ()

    @property
    def healthy(self) -> bool:
        """四个条件全满足才算健康。

        「某张表一行都没有」必须单独判：洞是按"该表自己跨度内的开市日"算的，
        空表没有跨度，于是不会有洞 —— 一个完全空库会因此被报成健康，
        这正是 v1「库停在上一个交易日却照常选股」的镜像。
        """
        return (
            self.calendar is not None
            and not self.empty_tables
            and not self.backlog
            and not any(self.holes.values())
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "exchange": self.exchange,
            "min_rows": self.min_rows,
            "today": self.today.isoformat(),
            "calendar": [item.isoformat() for item in self.calendar] if self.calendar else None,
            "coverage": [item.as_dict() for item in self.coverage],
            "holes": {name: [item.isoformat() for item in dates] for name, dates in self.holes.items()},
            "backlog": [item.isoformat() for item in self.backlog],
            "empty_tables": list(self.empty_tables),
            "healthy": self.healthy,
        }


def sync_metadata(
    store: WriteStore,
    source: MarketDataSource,
    *,
    exchange: str = "SSE",
    end: date | None = None,
    days: int = 10,
    calendar_lookahead_days: int = 45,
) -> tuple[int, int]:
    """同步交易日历与股票基础信息。返回 (日历行数, 基础信息行数)。

    日历一次性覆盖 [起点, 今天+未来] —— 未来部分供标签和回测算 T+N 交易日。
    """
    today = end or beijing_now().date()
    start = today - timedelta(days=days * 2 + _CALENDAR_BACKFILL_SLACK_DAYS)
    calendar = source.trade_cal(
        start=start, end=today + timedelta(days=calendar_lookahead_days), exchange=exchange
    )
    with store.transaction():
        calendar_rows = store.upsert("trade_cal", calendar)
    basic = source.stock_basic()
    with store.transaction():
        basic_rows = store.upsert("stock_basic", basic)
    return calendar_rows, basic_rows


#: 库里的 (代码, 生效日) 有多少比例这次没取到就拒绝。上游改一条记录的生效日，库里会多一个旧键
#: （这张表不删行），那是几个键；翻页撞限流少一页是几千个键。
NAME_LOSS_MAX = 0.01
#: 最新交易日有行情的代码里，至少这个比例要有名称记录（实测正式库 5571/5571 都有）。
NAME_COVERAGE_MIN = 0.98


def sync_names(store: WriteStore, source: MarketDataSource) -> int:
    """历史名称（反弹模型准入池按当时的名称判 ST）：全量拉一遍，按 (代码, 生效日) 覆盖写。返回写入行数。

    上游删掉的旧记录不删：这张表只会往前长，少一行比多一行危险（会把当时是 ST 的票放进池子）。
    取得不全就拒绝、一行都不写，两道关：

    - 库里已有的键，这次没取到的超过 `NAME_LOSS_MAX`（比行数不行：上游改一条记录的生效日，库里就永远比上游多一行）；
    - 库里最新交易日有行情的代码，有名称记录的不到 `NAME_COVERAGE_MIN`（库里还没有名称、键没得比时靠这一道）。

    名称在行情之前同步（`run_sync`），所以这里看的是库里已有的最新交易日；全新的库那时还没有日线，
    这一道跳过：要是那次取得不全，下一次同步会把缺的键补上（这张表只增不删）。
    """
    names = source.namechange()
    stored = set(store.connection.execute("SELECT ts_code, start_date FROM namechange").fetchall())
    fetched = set(zip(names["ts_code"].astype(str), names["start_date"]))
    lost = len(stored - fetched)
    if stored and lost > len(stored) * NAME_LOSS_MAX:
        raise DataError(f"Tushare namechange 这次取到 {len(names)} 行，库里有 {lost} 条（共 {len(stored)} 条）这次没取到："
                        "多半是翻页撞了每分钟调用上限、少了一截，这次不写；过几分钟再同步")
    latest = store.connection.execute(
        "SELECT DISTINCT ts_code FROM daily WHERE trade_date = (SELECT max(trade_date) FROM daily)").fetchall()
    codes = {code for (code,) in latest}
    named = codes & set(names["ts_code"].astype(str))
    if codes and len(named) < len(codes) * NAME_COVERAGE_MIN:
        raise DataError(f"Tushare namechange 这次取到 {len(names)} 行，最新交易日 {len(codes)} 只有行情的票里"
                        f"只有 {len(named)} 只有名称记录：多半是翻页撞了每分钟调用上限、少了一截，这次不写；过几分钟再同步")
    with store.transaction():
        return store.upsert("namechange", names)


def sync_market(
    store: WriteStore,
    source: MarketDataSource,
    *,
    exchange: str = "SSE",
    end: date | None = None,
    days: int = 10,
    min_rows: int = 1000,
    with_limits: bool = True,
    with_adj_factor: bool = True,
    on_progress: Callable[[int, int, DateSync | SkippedDate], None] | None = None,
    stop: Any = None,
) -> SyncReport:
    """只补三张行情表（日历与基础信息由 `sync_metadata` 负责）。

    缺什么拉什么，已有的不动；同一次运行里刚写进去的日期不会再拉第二次。
    `on_progress(已处理数, 目标数, 记录)` 每处理完一个交易日回调一次 ——
    回补一年是几百次网络调用，没有进度输出就只能干等。
    """
    if days < 1:
        raise DataError(f"days 必须为正，收到 {days}")
    today = end or beijing_now().date()
    window_start = today - timedelta(days=days * 2 + _CALENDAR_BACKFILL_SLACK_DAYS)
    open_dates = store.open_dates(exchange=exchange, start=window_start, end=today)
    if not open_dates:
        raise DataError(
            f"本地日历里没有 {window_start}~{today} 的开市日：日历没同步成功，"
            "不能拿'没有开市日'当'今天不用跑'"
        )
    targets = open_dates[-days:]

    processed = 0

    def _emit(record: DateSync | SkippedDate) -> None:
        nonlocal processed
        processed += 1
        if on_progress is not None:
            on_progress(processed, len(targets), record)

    existing = {
        "daily": store.dates_with_rows("daily"),
        "daily_limit": store.dates_with_rows("daily_limit") if with_limits else {},
        "adj_factor": store.dates_with_rows("adj_factor") if with_adj_factor else {},
    }

    synced: list[DateSync] = []
    skipped: list[SkippedDate] = []
    for trade_date in targets:
        if stop is not None:
            stop.checkpoint()
        need_daily = existing["daily"].get(trade_date, 0) < min_rows
        need_limit = with_limits and existing["daily_limit"].get(trade_date, 0) < min_rows
        need_adj = with_adj_factor and existing["adj_factor"].get(trade_date, 0) < min_rows
        if not (need_daily or need_limit or need_adj):
            record = SkippedDate(trade_date, "already_present", f"要的表都已 ≥ {min_rows} 行")
            skipped.append(record)
            _emit(record)
            continue

        daily_rows = 0
        limit_rows = 0
        adj_rows = 0
        try:
            if need_daily:
                daily = source.daily(trade_date=trade_date)
                daily_rows = len(daily)
                if daily_rows < min_rows:
                    # 上游只有零星几行 = 这一天还没收盘确认。不写库，也不当缺口。
                    # 注意是严格小于：`min_rows` 是"一个完整交易日至少要有多少行"，
                    # 恰好等于它就是完整的。上面判"已在库里"用的也是 `<`，
                    # 两处必须是同一条规则，否则同一天会因"谁先看"而得到相反结论。
                    record = SkippedDate(
                        trade_date,
                        "not_closed",
                        f"daily 只有 {daily_rows} 行，低于 min_rows={min_rows}",
                    )
                    skipped.append(record)
                    _emit(record)
                    continue

            # 几张表都先取回来，再开事务写：事务里不做网络 IO，写锁窗口最短。
            if need_limit:
                limits = source.daily_limit(trade_date=trade_date)
            if need_adj:
                factors = source.adj_factor(trade_date=trade_date)

            with store.transaction():
                if need_daily:
                    daily_rows = store.upsert("daily", daily)
                if need_limit:
                    limit_rows = store.upsert("daily_limit", limits)
                if need_adj:
                    adj_rows = store.upsert("adj_factor", factors)
            # 本地统计随写入更新，避免同一次运行里重复拉同一天。
            # 只在事务提交之后更新：中途回滚了就不许 in-memory 统计说"这天已经有数据"。
            if need_daily:
                existing["daily"][trade_date] = daily_rows
            if need_limit:
                existing["daily_limit"][trade_date] = limit_rows
            if need_adj:
                existing["adj_factor"][trade_date] = adj_rows
        except (NetworkError, ValidationError) as error:
            # 一天的失败不该让整趟回补白跑：记下来、继续下一天。几张表在同一个
            # 事务里，所以这天要么全进要么全不进，不会留下半截状态。漏掉的那天
            # 会被新鲜度闸门在下次选股时拦下来，不会被静默忽略。
            record = SkippedDate(
                trade_date,
                "fetch_failed",
                f"{type(error).__name__}: {error}",
            )
            skipped.append(record)
            _emit(record)
            continue
        record = DateSync(trade_date, daily_rows, limit_rows, adj_rows)
        synced.append(record)
        _emit(record)

    return SyncReport(
        synced=tuple(synced),
        skipped=tuple(skipped),
        coverage=store.coverage(),
    )


def audit(
    store: WriteStore,
    *,
    exchange: str = "SSE",
    min_rows: int = 1000,
    today: date | None = None,
    tables: tuple[str, ...] = ("daily", "daily_limit", "adj_factor"),
) -> AuditReport:
    """离线体检：区间内的洞 + 落后多少。

    「洞」= 本地日历判定为开市、但该表行数少于 `min_rows` 的日期，只在
    **该表自己的日期跨度内**统计 —— 否则日历会往前延伸到未来，把每一天
    都报成缺口。
    """
    today = today or beijing_now().date()
    coverage = store.coverage()
    calendar = store.calendar_range(exchange=exchange)
    holes: dict[str, tuple[date, ...]] = {}
    empty: list[str] = []
    for table in tables:
        span = next((item for item in coverage if item.table == table), None)
        if not store.has_table(table) or span is None or span.first is None or span.last is None:
            # 表不存在或一行数据都没有：这是"空表"，不是"没有洞"。
            holes[table] = ()
            empty.append(table)
            continue
        open_dates = store.open_dates(exchange=exchange, start=span.first, end=span.last)
        counts = store.dates_with_rows(table)
        holes[table] = tuple(d for d in open_dates if counts.get(d, 0) < min_rows)

    daily = next((item for item in coverage if item.table == "daily"), None)
    backlog: tuple[date, ...] = ()
    if daily is not None and daily.last is not None:
        backlog = tuple(
            d for d in store.open_dates(exchange=exchange, start=daily.last, end=today)
            if d > daily.last
        )
    return AuditReport(
        exchange=exchange,
        min_rows=min_rows,
        today=today,
        calendar=calendar,
        coverage=coverage,
        holes=holes,
        backlog=backlog,
        empty_tables=tuple(empty),
    )
