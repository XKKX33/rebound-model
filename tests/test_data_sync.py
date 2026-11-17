"""数据层：幂等写入、一天一事务、按需拉、缺口审计。

`min_rows` 在这些测试里一律压到 2（几行假数据就够），因为这里要验的是
**纪律**而不是吞吐：该写的写、该跳的跳、该拒绝的拒绝。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from rebound.data import schema
from rebound.data.db import WriteStore
from rebound.data.errors import DataError, NetworkError, ValidationError
from rebound.data.sync import audit, sync_market, sync_metadata, sync_names

D14 = date(2026, 10, 14)
D15 = date(2026, 10, 15)
D16 = date(2026, 10, 16)
SESSIONS = (D14, D15, D16)
MIN_ROWS = 2


# --------------------------------------------------------------------------- #
# 假数据源：形状与真源一致（校验在客户端层做，这里直接给"已校验"的帧）
# --------------------------------------------------------------------------- #


def _daily(day: date, codes: tuple[str, ...] = ("000001.SZ", "600000.SH")) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": code,
                "trade_date": day,
                "open": 10.0,
                "high": 10.5,
                "low": 9.9,
                "close": 10.2,
                "pre_close": 10.0,
                "pct_chg": 2.0,
                "vol": 1000.0,
                "amount": 10000.0,
            }
            for code in codes
        ],
        columns=list(schema.TABLE_FIELDS["daily"]),
    )


def _limits(day: date, codes: tuple[str, ...] = ("000001.SZ", "600000.SH")) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ts_code": code, "trade_date": day, "up_limit": 11.0, "down_limit": 9.0}
            for code in codes
        ],
        columns=list(schema.TABLE_FIELDS["daily_limit"]),
    )


def _adj(day: date, codes: tuple[str, ...] = ("000001.SZ", "600000.SH")) -> pd.DataFrame:
    return pd.DataFrame(
        [{"ts_code": code, "trade_date": day, "adj_factor": 1.5} for code in codes],
        columns=list(schema.TABLE_FIELDS["adj_factor"]),
    )


def _basic() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "symbol": "000001",
                "name": "平安银行",
                "area": "深圳",
                "industry": "银行",
                "market": "主板",
                "list_date": date(1991, 4, 3),
            }
        ],
        columns=list(schema.TABLE_FIELDS["stock_basic"]),
    )


class FakeSource:
    """可注入的假数据源，同时记录被调用了哪些端点。"""

    def __init__(
        self,
        sessions: tuple[date, ...] = SESSIONS,
        *,
        daily_codes: tuple[str, ...] = ("000001.SZ", "600000.SH"),
        thin_days: tuple[date, ...] = (),
        fail: tuple[str, date] | None = None,
        trade_cal_order: str = "desc",
    ) -> None:
        self.sessions = sessions
        self.daily_codes = daily_codes
        self.thin_days = set(thin_days)
        self.fail = fail
        self.trade_cal_order = trade_cal_order
        self.names: pd.DataFrame | None = None  # 历史名称：缺省 `_names()`
        self.calls: list[tuple[str, date | None]] = []

    def _guard(self, endpoint: str, day: date) -> None:
        """模拟上游限流：`_call` 退避用尽后抛出的是 `NetworkError`。"""
        if self.fail == (endpoint, day):
            raise NetworkError(
                f"Tushare {endpoint} 连续 4 次返回空表"
                "（超出每分钟调用上限，或这天的数据还没发布）"
            )

    def trade_cal(self, *, start: date, end: date, exchange: str = "SSE") -> pd.DataFrame:
        self.calls.append(("trade_cal", None))
        inside = sorted(d for d in self.sessions if start <= d <= end)
        if self.trade_cal_order == "desc":
            # Tushare 实际是按 cal_date 降序返回的。
            inside = list(reversed(inside))
        return pd.DataFrame(
            [{"exchange": exchange, "cal_date": d, "is_open": 1} for d in inside],
            columns=list(schema.TABLE_FIELDS["trade_cal"]),
        )

    def stock_basic(self) -> pd.DataFrame:
        self.calls.append(("stock_basic", None))
        return _basic()

    def daily(self, *, trade_date: date) -> pd.DataFrame:
        self.calls.append(("daily", trade_date))
        self._guard("daily", trade_date)
        if trade_date in self.thin_days:
            return _daily(trade_date, self.daily_codes[:1])
        return _daily(trade_date, self.daily_codes)

    def daily_limit(self, *, trade_date: date) -> pd.DataFrame:
        self.calls.append(("daily_limit", trade_date))
        self._guard("daily_limit", trade_date)
        return _limits(trade_date, self.daily_codes)

    def adj_factor(self, *, trade_date: date) -> pd.DataFrame:
        self.calls.append(("adj_factor", trade_date))
        self._guard("adj_factor", trade_date)
        return _adj(trade_date, self.daily_codes)

    def namechange(self) -> pd.DataFrame:
        self.calls.append(("namechange", None))
        return _names() if self.names is None else self.names


def _names(rows: int = 2) -> pd.DataFrame:
    """已校验过的历史名称（日期已是 `date`）。"""
    names = [("600000.SH", "ST浦发", date(2001, 1, 2), date(2003, 5, 1), date(2000, 12, 28), "ST"),
             ("600000.SH", "浦发银行", date(2003, 5, 2), None, date(2003, 4, 30), "撤销ST"),
             ("000001.SZ", "平安银行", date(2012, 8, 2), None, date(2012, 7, 31), "其他")]
    return pd.DataFrame(names[:rows], columns=list(schema.TABLE_FIELDS["namechange"]))


@pytest.fixture()
def store(tmp_path: Path):
    with WriteStore(tmp_path / "market.duckdb") as handle:
        yield handle


# --------------------------------------------------------------------------- #
# 建表与幂等写入
# --------------------------------------------------------------------------- #


def test_write_store_creates_the_schema_on_open(store: WriteStore) -> None:
    """回归：`WriteStore` 打开即可写。

    原先 __init__ 不建表，第一次 upsert 会报"库内列与 schema 不一致: ()" ——
    表根本不存在，那句话把人带到错误的排查方向。
    """
    for table in schema.TABLE_FIELDS:
        assert store.has_table(table), table
    assert store.upsert("daily", _daily(D14)) == 2


def test_schema_has_no_daily_basic_yet() -> None:
    """市值/换手没有任何因子在读，所以这张表刻意不建。

    真要用（例如加市值因子）时先加 `schema.TABLE_FIELDS` 再加同步，
    不许"先建着以后可能会用"。
    """
    assert "daily_basic" not in schema.TABLE_FIELDS
    assert "daily_basic" not in schema.TABLE_ORDER


def test_adj_factor_is_a_market_table() -> None:
    """反弹模型的特征用复权价（研究版用 Tushare 复权因子算）：它是行情，覆盖率和缺口检测都要看它。"""
    assert "adj_factor" in schema.MARKET_TABLES
    assert schema.TABLE_PRIMARY_KEYS["adj_factor"] == ("ts_code", "trade_date")


def test_upsert_is_idempotent_on_the_business_key(store: WriteStore) -> None:
    store.upsert("daily", _daily(D14))
    changed = _daily(D14)
    changed.loc[0, "close"] = 99.0
    store.upsert("daily", changed)
    assert store.row_count("daily") == 2
    close = store.connection.execute(
        "SELECT close FROM daily WHERE ts_code = '000001.SZ'"
    ).fetchone()
    assert close[0] == 99.0


def test_upsert_refuses_columns_outside_the_table(store: WriteStore) -> None:
    frame = _daily(D14)
    frame["pe"] = 5.0
    with pytest.raises(DataError, match="表外列"):
        store.upsert("daily", frame)


def test_upsert_refuses_missing_columns(store: WriteStore) -> None:
    frame = _daily(D14).drop(columns=["amount"])
    with pytest.raises(DataError, match="缺少列"):
        store.upsert("daily", frame)


def test_upsert_refuses_unknown_tables(store: WriteStore) -> None:
    with pytest.raises(DataError, match="未知的表"):
        store.upsert("picks", _daily(D14))


def test_transaction_rolls_back_so_a_day_is_all_or_nothing(store: WriteStore) -> None:
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.upsert("daily", _daily(D14))
            raise RuntimeError("资金流拉失败了")
    assert store.row_count("daily") == 0


# --------------------------------------------------------------------------- #
# sync_metadata / sync_market
# --------------------------------------------------------------------------- #


def test_sync_metadata_writes_calendar_and_basics(store: WriteStore) -> None:
    source = FakeSource()
    calendar_rows, basic_rows = sync_metadata(
        store, source, end=D16, days=3, calendar_lookahead_days=45
    )
    assert calendar_rows == 3
    assert basic_rows == 1
    # Tushare 给的是降序，库里必须是升序：`open_dates[-1]` 要等于最新那天。
    assert store.open_dates(exchange="SSE") == (D14, D15, D16)


def test_sync_names_upserts_the_name_history(store: WriteStore) -> None:
    source = FakeSource()
    assert sync_names(store, source) == 2
    source.names = _names(3)
    assert sync_names(store, source) == 3  # 再拉一遍：按 (代码, 生效日) 覆盖，不会多出重复行
    assert store.row_count("namechange") == 3
    assert store.connection.execute(
        "SELECT name FROM namechange WHERE ts_code = '600000.SH' ORDER BY start_date").fetchall() == [("ST浦发",), ("浦发银行",)]


def _many_names(codes: int, *, start: date = date(2020, 1, 2)) -> pd.DataFrame:
    return pd.DataFrame([(f"{600000 + i}.SH", "某某", start, None, start, "其他") for i in range(codes)],
                        columns=list(schema.TABLE_FIELDS["namechange"]))


def test_sync_names_refuses_a_fetch_that_lost_part_of_the_table(store: WriteStore) -> None:
    """库里的键这次有一截没取到：多半是翻页撞了限流，写进去会把当时的 ST 放进池子。一行都不写。"""
    source = FakeSource()
    source.names = _many_names(200)
    sync_names(store, source)
    source.names = _many_names(150)
    with pytest.raises(DataError, match="库里有 50 条（共 200 条）这次没取到"):
        sync_names(store, source)
    assert store.row_count("namechange") == 200


def test_sync_names_accepts_an_upstream_correction_of_one_start_date(store: WriteStore) -> None:
    """上游改了一条记录的生效日：库里多出一个旧键（这张表不删行），行数从此比上游多，照样要能同步。"""
    source = FakeSource()
    source.names = _many_names(200)
    sync_names(store, source)
    corrected = _many_names(200)
    corrected.loc[0, "start_date"] = date(2020, 1, 3)
    source.names = corrected
    assert sync_names(store, source) == 200
    assert store.row_count("namechange") == 201
    source.names = corrected  # 再同步一次：库里 201 个键，上游 200 个，只差那一个旧键
    assert sync_names(store, source) == 200


def test_sync_names_refuses_when_most_listed_stocks_have_no_name(store: WriteStore) -> None:
    """第一次同步库里没有键可比：最新交易日有行情的票大多没有名称记录，就是没取全。"""
    source = FakeSource()
    sync_metadata(store, source, end=D16, days=3)
    sync_market(store, source, end=D16, days=3, min_rows=MIN_ROWS)
    source.names = _names(1)
    with pytest.raises(DataError, match="只有 .* 只有名称记录"):
        sync_names(store, source)
    assert store.row_count("namechange") == 0


def test_sync_market_writes_every_requested_day(store: WriteStore) -> None:
    source = FakeSource()
    sync_metadata(store, source, end=D16, days=3)
    report = sync_market(store, source, end=D16, days=3, min_rows=MIN_ROWS)
    assert [item.trade_date for item in report.synced] == list(SESSIONS)
    assert report.skipped == ()
    assert report.written_rows == 3 * 6  # 每天 2 行 × 三张表
    assert store.row_count("daily_limit") == 6
    assert store.row_count("adj_factor") == 6
    assert report.synced[0].as_dict()["adj_factor"] == 2


def test_sync_market_skips_days_already_present(store: WriteStore) -> None:
    source = FakeSource()
    sync_metadata(store, source, end=D16, days=3)
    sync_market(store, source, end=D16, days=3, min_rows=MIN_ROWS)
    source.calls.clear()
    second = sync_market(store, source, end=D16, days=3, min_rows=MIN_ROWS)
    assert second.synced == ()
    assert {item.reason for item in second.skipped} == {"already_present"}
    # 一天都没再拉过行情端点。
    assert [name for name, _ in source.calls if name != "trade_cal"] == []


def test_sync_market_checks_for_stop_between_days(store: WriteStore) -> None:
    """停止只发生在日与日之间；已经提交的那一天完整保留。"""

    class Stopped(Exception):
        pass

    class Stop:
        def __init__(self) -> None:
            self.calls = 0

        def checkpoint(self) -> None:
            self.calls += 1
            if self.calls == 2:
                raise Stopped()

    source = FakeSource()
    sync_metadata(store, source, end=D16, days=3)
    stop = Stop()

    with pytest.raises(Stopped):
        sync_market(store, source, end=D16, days=3, min_rows=MIN_ROWS, stop=stop)

    assert store.row_count("daily", trade_date=D14) == 2
    assert store.row_count("daily", trade_date=D15) == 0


def test_a_thin_day_is_not_closed_and_leaves_nothing_behind(store: WriteStore) -> None:
    """上游只回几行 = 这天还没收盘确认。

    关键是**整天都不写**：不许出现"日线写了、复权因子没写"的半截状态
    （下游无法与"这天本来就缺"区分）。
    """
    source = FakeSource(thin_days=(D15,))
    sync_metadata(store, source, end=D16, days=3)
    report = sync_market(store, source, end=D16, days=3, min_rows=MIN_ROWS)

    assert [item.trade_date for item in report.synced] == [D14, D16]
    skipped = {item.trade_date: item for item in report.skipped}
    assert skipped[D15].reason == "not_closed"
    assert store.row_count("daily", trade_date=D15) == 0
    assert store.row_count("daily_limit", trade_date=D15) == 0
    assert store.row_count("adj_factor", trade_date=D15) == 0


def test_a_failed_fetch_skips_only_that_day(store: WriteStore) -> None:
    """回归：一次限流不该废掉整趟回补。

    真机上 `daily_limit` 回补到第 47 天时返回空表（限流），而那句
    `ValidationError("daily_limit 返回空数据")` 把整趟 170 天的回补中止了。
    限流是 `NetworkError`（暂时失败，重试有意义），与 `ValidationError`
    （同样的输入重试也不会变好）是两件事；而且失败只该影响那一天。
    """
    source = FakeSource(fail=("daily_limit", D15))
    sync_metadata(store, source, end=D16, days=3)
    report = sync_market(store, source, end=D16, days=3, min_rows=MIN_ROWS)

    assert [item.trade_date for item in report.synced] == [D14, D16]
    skipped = {item.trade_date: item for item in report.skipped}
    assert skipped[D15].reason == "fetch_failed"
    assert "NetworkError" in skipped[D15].detail
    # 三张表在同一个事务里：那天的 daily 已经 upsert 过了，也必须跟着回滚。
    for table in ("daily", "daily_limit", "adj_factor"):
        assert store.row_count(table, trade_date=D15) == 0, table


def test_a_failed_adj_factor_fetch_rolls_back_the_whole_day(store: WriteStore) -> None:
    """复权因子缺了，反弹模型的特征就算不出来：它和行情同进同退，不许单独缺。"""
    source = FakeSource(fail=("adj_factor", D15))
    sync_metadata(store, source, end=D16, days=3)
    report = sync_market(store, source, end=D16, days=3, min_rows=MIN_ROWS)

    assert [item.trade_date for item in report.synced] == [D14, D16]
    assert {item.trade_date: item.reason for item in report.skipped} == {D15: "fetch_failed"}
    for table in ("daily", "daily_limit", "adj_factor"):
        assert store.row_count(table, trade_date=D15) == 0, table


def test_sync_market_fills_adj_factor_for_days_that_only_lack_it(store: WriteStore) -> None:
    """老库有行情、没有复权因子：再同步一次只补复权因子，别的表不重拉。"""
    source = FakeSource()
    sync_metadata(store, source, end=D16, days=3)
    sync_market(store, source, end=D16, days=3, min_rows=MIN_ROWS, with_adj_factor=False)
    assert store.row_count("adj_factor") == 0
    source.calls.clear()

    report = sync_market(store, source, end=D16, days=3, min_rows=MIN_ROWS)
    assert [item.trade_date for item in report.synced] == list(SESSIONS)
    assert store.row_count("adj_factor") == 6
    assert {name for name, _ in source.calls} == {"adj_factor"}
    assert report.written_rows == 3 * 2


def test_sync_market_refuses_to_run_without_a_calendar(store: WriteStore) -> None:
    """日历没同步成功时不许"安静地什么都不做"。"""
    with pytest.raises(DataError, match="本地日历里没有"):
        sync_market(store, FakeSource(), end=D16, days=3, min_rows=MIN_ROWS)


def test_sync_market_rejects_nonpositive_days(store: WriteStore) -> None:
    with pytest.raises(DataError, match="days 必须为正"):
        sync_market(store, FakeSource(), end=D16, days=0)


def test_sync_market_can_skip_the_optional_tables(store: WriteStore) -> None:
    source = FakeSource()
    sync_metadata(store, source, end=D16, days=3)
    report = sync_market(
        store, source, end=D16, days=3, min_rows=MIN_ROWS,
        with_limits=False, with_adj_factor=False,
    )
    assert report.written_rows == 3 * 2
    assert store.row_count("daily_limit") == 0
    assert store.row_count("adj_factor") == 0


def test_sync_market_only_looks_back_the_requested_number_of_sessions(
    store: WriteStore,
) -> None:
    source = FakeSource()
    sync_metadata(store, source, end=D16, days=3)
    report = sync_market(store, source, end=D16, days=1, min_rows=MIN_ROWS)
    assert [item.trade_date for item in report.synced] == [D16]


# --------------------------------------------------------------------------- #
# audit：缺口必须能被说出来
# --------------------------------------------------------------------------- #


def test_audit_reports_a_hole_inside_a_tables_own_span(store: WriteStore) -> None:
    """某张表只覆盖了一部分交易日：缺的那天要被说出来。"""
    source = FakeSource()
    sync_metadata(store, source, end=D16, days=3)
    sync_market(store, source, end=D16, days=3, min_rows=MIN_ROWS)
    store.connection.execute("DELETE FROM adj_factor WHERE trade_date = ?", [D15])

    report = audit(store, min_rows=MIN_ROWS, today=D16)
    assert report.holes["adj_factor"] == (D15,)
    assert report.holes["daily"] == ()
    assert report.healthy is False


def test_audit_does_not_count_days_outside_a_tables_span_as_holes(store: WriteStore) -> None:
    """`daily` 只同步到 10-14 时，"10-15/10-16 没有 daily"不算洞，算落后。

    否则日历延伸到未来会把每一天都报成缺口，报告就没有信息量了。
    """
    source = FakeSource()
    sync_metadata(store, source, end=D16, days=3)
    sync_market(store, source, end=D14, days=1, min_rows=MIN_ROWS)

    report = audit(store, min_rows=MIN_ROWS, today=D16)
    assert report.holes["daily"] == ()
    assert report.backlog == (D15, D16)
    assert report.healthy is False


def test_audit_is_healthy_when_everything_is_present(store: WriteStore) -> None:
    source = FakeSource()
    sync_metadata(store, source, end=D16, days=3)
    sync_market(store, source, end=D16, days=3, min_rows=MIN_ROWS)
    report = audit(store, min_rows=MIN_ROWS, today=D16)
    assert report.healthy is True
    assert report.backlog == ()
    assert report.calendar == (D14, D16)


def test_audit_survives_an_empty_database(store: WriteStore) -> None:
    report = audit(store, min_rows=MIN_ROWS, today=D16)
    assert report.healthy is False
    assert report.calendar is None
    assert {item.table for item in report.coverage} == set(schema.MARKET_TABLES)


def test_audit_reports_a_database_that_never_synced_adj_factor(store: WriteStore) -> None:
    """老库升级后复权因子是空表：体检必须说出来，而不是报健康。"""
    source = FakeSource()
    sync_metadata(store, source, end=D16, days=3)
    sync_market(store, source, end=D16, days=3, min_rows=MIN_ROWS, with_adj_factor=False)
    report = audit(store, min_rows=MIN_ROWS, today=D16)
    assert report.empty_tables == ("adj_factor",)
    assert report.healthy is False


# --------------------------------------------------------------------------- #
# TushareClient.adj_factor：和别的行情端点走同一条校验
# --------------------------------------------------------------------------- #


def _client_returning(monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame) -> tuple[object, list]:
    from rebound.data.tushare_client import TushareClient

    client = TushareClient("token-for-tests")
    calls: list[tuple[str, dict]] = []

    def fake_attempt(endpoint: str, **kwargs: object) -> pd.DataFrame:
        calls.append((endpoint, kwargs))
        return frame

    monkeypatch.setattr(client, "_attempt", fake_attempt)
    return client, calls


def test_tushare_adj_factor_asks_for_the_table_fields_and_parses_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = pd.DataFrame({"ts_code": ["000001.SZ", "600000.SH"], "trade_date": ["20261016", "20261016"],
                        "adj_factor": ["108.031", "11.2"]})
    client, calls = _client_returning(monkeypatch, raw)
    frame = client.adj_factor(trade_date=D16)
    assert calls == [("adj_factor", {"trade_date": "20261016", "fields": "ts_code,trade_date,adj_factor"})]
    assert list(frame.columns) == list(schema.TABLE_FIELDS["adj_factor"])
    assert set(frame.trade_date) == {D16} and frame.adj_factor.tolist() == [108.031, 11.2]


@pytest.mark.parametrize(
    ("trade_date", "factor", "message"),
    [("20261015", "1.0", "其他交易日"), ("20261016", "-1.0", "负数"), ("20261016", None, "缺失值"),
     # 0 过得了"非负"，但复权价 = 价 × 0 会把整段历史抹成 0
     ("20261016", "0", "因子为 0")],
)
def test_tushare_adj_factor_rejects_bad_rows(
    monkeypatch: pytest.MonkeyPatch, trade_date: str, factor: str | None, message: str
) -> None:
    from rebound.data.errors import ValidationError

    raw = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [trade_date], "adj_factor": [factor]})
    client, _ = _client_returning(monkeypatch, raw)
    with pytest.raises(ValidationError, match=message):
        client.adj_factor(trade_date=D16)


# --------------------------------------------------------------------------- #
# TushareClient.namechange：翻页取完全部历史名称
# --------------------------------------------------------------------------- #


def _name_page(start: int, rows: int) -> pd.DataFrame:
    return pd.DataFrame({"ts_code": [f"{600000 + i}.SH" for i in range(start, start + rows)], "name": "某某",
                         "start_date": "20200102", "end_date": None, "ann_date": "20191231", "change_reason": "其他"})


def _paged_client(monkeypatch: pytest.MonkeyPatch, pages: list[pd.DataFrame]) -> tuple[object, list[int]]:
    from rebound.data.tushare_client import TushareClient

    client = TushareClient("token-for-tests")
    offsets: list[int] = []

    def fake_attempt(endpoint: str, **kwargs: object) -> pd.DataFrame:
        assert endpoint == "namechange" and kwargs["limit"] == 3
        offsets.append(int(kwargs["offset"]))  # type: ignore[arg-type]
        return pages.pop(0) if pages else pd.DataFrame()

    monkeypatch.setattr(client, "_attempt", fake_attempt)
    monkeypatch.setattr(TushareClient, "_NAME_PAGE", 3)
    monkeypatch.setattr(TushareClient, "_EMPTY_BACKOFF_SECONDS", (0.0,))
    return client, offsets


def test_namechange_walks_the_pages_until_an_empty_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """不满一页之后再确认一次（只一次、不退避）：拿到空页才算取完。"""
    client, offsets = _paged_client(monkeypatch, [_name_page(0, 3), _name_page(3, 3), _name_page(6, 2)])
    frame = client.namechange()
    assert offsets == [0, 3, 6, 8] and len(frame) == 8
    assert list(frame.columns) == list(schema.TABLE_FIELDS["namechange"])
    assert set(frame.start_date) == {date(2020, 1, 2)} and set(frame.ann_date) == {date(2019, 12, 31)}


def test_namechange_ends_on_an_empty_page_after_a_full_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """总行数正好是整页的倍数：最后一页是空的。按限流退避重试，仍是空页才算取完（不再报错）。"""
    client, offsets = _paged_client(monkeypatch, [_name_page(0, 3), _name_page(3, 3)])
    frame = client.namechange()
    assert len(frame) == 6 and offsets[:2] == [0, 3] and set(offsets[2:]) == {6} and len(offsets) > 3


def test_namechange_keeps_paging_when_upstream_pages_are_smaller_than_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    """上游每页只给 2 行（要的是 3 行）：不能把第一页当成全部。"""
    client, offsets = _paged_client(monkeypatch, [_name_page(0, 2), _name_page(2, 2), _name_page(4, 1)])
    frame = client.namechange()
    assert len(frame) == 5 and offsets[:4] == [0, 2, 4, 5]


def test_namechange_with_nothing_on_the_first_page_is_an_upstream_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _paged_client(monkeypatch, [])
    with pytest.raises(NetworkError, match="namechange"):
        client.namechange()


def test_namechange_keeps_the_latest_announcement_for_one_start_date(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.DataFrame({"ts_code": ["600000.SH"] * 2, "name": ["旧名", "新名"], "start_date": ["20200102"] * 2,
                        "end_date": [None, None], "ann_date": ["20191201", "20191231"], "change_reason": ["其他"] * 2})
    client, _ = _paged_client(monkeypatch, [raw])
    frame = client.namechange()
    assert frame.name.tolist() == ["新名"]


def test_a_failed_name_sync_stops_before_any_new_market_day_is_written(tmp_path: Path, monkeypatch) -> None:
    """名称在行情之前同步：名称取不全就整趟停下，新的一天的行情一行不写。
    否则再跑一次流水线时行情已齐、不再同步，反弹模型会拿旧名称（新戴帽的 ST 还不是 ST）给这一天打分，而分写下就不再改。"""
    from dataclasses import replace

    from rebound import cli, config
    from rebound.data import tushare_client

    base = config.load_settings()
    settings = replace(base, data=replace(base.data, min_daily_rows=MIN_ROWS))

    class NoNames(FakeSource):
        def namechange(self) -> pd.DataFrame:
            raise NetworkError("Tushare namechange 连续 6 次返回空表（测试替身）")

    ok, failing = tmp_path / "ok.duckdb", tmp_path / "failing.duckdb"
    quiet = lambda _line: None  # noqa: E731
    from datetime import datetime

    from rebound.data import sync as sync_module

    monkeypatch.setattr(sync_module, "beijing_now", lambda: datetime.combine(D16, datetime.min.time()))
    monkeypatch.setattr(tushare_client, "tushare_source", lambda _settings: FakeSource())
    assert cli.run_sync(settings, ok, days=3, log=quiet)[0].synced  # 对照：名称取得到就照常写行情
    monkeypatch.setattr(tushare_client, "tushare_source", lambda _settings: NoNames())
    with pytest.raises(NetworkError, match="namechange"):
        cli.run_sync(settings, failing, days=3, log=quiet)
    with WriteStore(failing) as store:
        assert store.row_count("daily") == 0 and store.row_count("namechange") == 0


def test_namechange_stops_paging_when_upstream_never_gives_an_empty_page(monkeypatch: pytest.MonkeyPatch) -> None:
    from rebound.data.tushare_client import TushareClient

    client, offsets = _paged_client(monkeypatch, [])
    monkeypatch.setattr(TushareClient, "_NAME_MAX_PAGES", 4)
    monkeypatch.setattr(client, "_attempt", lambda endpoint, **kw: (offsets.append(kw["offset"]),
                                                                   _name_page(int(kw["offset"]), 3))[1])
    with pytest.raises(ValidationError, match="翻了 4 页还没到头"):
        client.namechange()
    assert len(offsets) == 4


def test_namechange_strips_control_characters_from_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """名称会原样打到终端上：上游给的转义序列不能在命令行里生效。"""
    page = _name_page(0, 2)
    page.loc[0, "name"] = "某\x1b[31m某\n"
    client, _ = _paged_client(monkeypatch, [page])
    frame = client.namechange()
    assert set(frame.name) == {"某[31m某", "某某"}
