"""一次性历史导入：研究库 → 正式库。

研究库只读；正式库写之前先备份；只补正式库完全没有的日子，已有的日子一行不动；
重叠的日子逐行核对价格、成交、涨跌停价和复权因子，对不上就拒绝写。
源库照研究库的真实表结构建（`stock_basic` 多出几列、`trade_cal.is_open` 是 DOUBLE）。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import duckdb
import pytest

from rebound.cli import main
from rebound.data.db import WriteStore
from rebound.data.errors import DataError
from rebound.data.history_import import import_history

D1, D2, D3, D4, D5 = (date(2022, 12, 30), date(2023, 12, 29), date(2024, 1, 2), date(2024, 1, 3),
                      date(2024, 1, 4))
HOLIDAY, FUTURE = date(2024, 1, 1), date(2024, 1, 5)
A, B, GONE, SRC_ONLY, TGT_ONLY = "000001.SZ", "600000.SH", "600001.SH", "600002.SH", "600999.SH"
#: 每天有哪些票：源库 D1、D2 有一只后来退市的票，D3 多一只正式库没有的票
SOURCE_DAYS = {D1: (A, B, GONE), D2: (A, B, GONE), D3: (A, B, SRC_ONLY), D4: (A, B)}
TARGET_DAYS = {D3: (A, B, TGT_ONLY), D4: (A, B, TGT_ONLY), D5: (A, B)}


def _price(code: str, day: date) -> float:
    return round(10 + sum(map(ord, code)) % 7 + day.toordinal() % 5 * 0.1, 2)


def _rows(days: dict) -> list[tuple]:
    return [(code, day) for day, codes in days.items() for code in codes]


def _source(path: Path) -> Path:
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE trade_cal (exchange VARCHAR, cal_date DATE, is_open DOUBLE, PRIMARY KEY (exchange, cal_date))")
    con.executemany("INSERT INTO trade_cal VALUES ('SSE', ?, ?)",
                    [(d, 1.0) for d in (D1, D2, D3, D4)] + [(HOLIDAY, 0.0)])
    con.execute("CREATE TABLE stock_basic (ts_code VARCHAR PRIMARY KEY, symbol VARCHAR, name VARCHAR, area VARCHAR, "
                "industry VARCHAR, market VARCHAR, exchange VARCHAR, list_status VARCHAR, list_date DATE, delist_date DATE)")
    con.executemany("INSERT INTO stock_basic VALUES (?, ?, ?, '深圳', ?, '主板', 'SZSE', ?, DATE '2000-01-04', ?)",
                    [(A, "000001", "旧名", "银行", "L", None), (B, "600000", "浦发银行", "银行", "L", None),
                     (GONE, "600001", "退市票", "钢铁", "D", date(2024, 1, 2))])
    con.execute("CREATE TABLE daily (ts_code VARCHAR, trade_date DATE, open DOUBLE, high DOUBLE, low DOUBLE, "
                "close DOUBLE, pre_close DOUBLE, pct_chg DOUBLE, vol DOUBLE, amount DOUBLE, PRIMARY KEY (ts_code, trade_date))")
    con.executemany("INSERT INTO daily VALUES (?, ?, ?, ?, ?, ?, ?, 1.0, 100.0, 1000.0)",
                    [(c, d, *([_price(c, d)] * 5)) for c, d in _rows(SOURCE_DAYS)])
    con.execute("CREATE TABLE daily_limit (ts_code VARCHAR, trade_date DATE, up_limit DOUBLE, down_limit DOUBLE, "
                "PRIMARY KEY (ts_code, trade_date))")
    con.executemany("INSERT INTO daily_limit VALUES (?, ?, ?, ?)",
                    [(c, d, _price(c, d) * 1.1, _price(c, d) * 0.9) for c, d in _rows(SOURCE_DAYS)])
    con.execute("CREATE TABLE adj_factor (ts_code VARCHAR, trade_date DATE, adj_factor DOUBLE, PRIMARY KEY (ts_code, trade_date))")
    con.executemany("INSERT INTO adj_factor VALUES (?, ?, 2.5)", _rows(SOURCE_DAYS))
    con.close()
    return path


def _target(path: Path) -> Path:
    with WriteStore(path) as store:
        con = store.connection
        con.executemany("INSERT INTO trade_cal VALUES ('SSE', ?, 1)", [(d,) for d in (D3, D4, D5, FUTURE)])
        con.executemany("INSERT INTO stock_basic VALUES (?, ?, ?, '深圳', ?, '主板', DATE '2000-01-04')",
                        [(A, "000001", "新名", "银行"), (B, "600000", "浦发银行", "银行"), (TGT_ONLY, "600999", "正式才有", "电力")])
        con.executemany("INSERT INTO daily VALUES (?, ?, ?, ?, ?, ?, ?, 1.0, 100.0, 1000.0)",
                        [(c, d, *([_price(c, d)] * 5)) for c, d in _rows(TARGET_DAYS)])
        con.executemany("INSERT INTO daily_limit VALUES (?, ?, ?, ?)",
                        [(c, d, _price(c, d) * 1.1, _price(c, d) * 0.9) for c, d in _rows(TARGET_DAYS)])
        con.execute("DROP TABLE adj_factor")  # 升级之前的老库没有这张表
    return path


@pytest.fixture()
def dbs(tmp_path: Path) -> tuple[Path, Path]:
    return _source(tmp_path / "research.duckdb"), _target(tmp_path / "market.duckdb")


def _backups(target: Path) -> list[Path]:
    return sorted(target.parent.glob(target.name + ".bak-before-history-import-*"))

def test_dry_run_plans_every_table_and_changes_nothing(dbs: tuple[Path, Path]) -> None:
    source, target = dbs
    before = target.read_bytes()
    report = import_history(source, target, dry_run=True)
    plans = report.tables
    assert (plans["trade_cal"].insert, plans["trade_cal"].overlap, plans["trade_cal"].mismatch) == (3, 2, 0)
    for table in ("daily", "daily_limit"):
        assert (plans[table].insert, plans[table].overlap, plans[table].mismatch) == (6, 4, 0), table
        assert (plans[table].target_only, plans[table].source_only) == (2, 1), table
    assert (plans["adj_factor"].insert, plans["adj_factor"].overlap) == (11, 0)  # 老库没有这张表
    assert plans["stock_basic"].insert == 1
    assert not report.refused and not report.written and report.backup is None
    assert target.read_bytes() == before and not _backups(target)
    json.dumps(report.as_dict(), ensure_ascii=False)


def test_import_fills_only_missing_days_after_a_backup(dbs: tuple[Path, Path]) -> None:
    source, target = dbs
    before = target.read_bytes()
    report = import_history(source, target)
    assert report.written and report.backup == _backups(target)[0]
    assert report.backup.read_bytes() == before
    con = duckdb.connect(str(target), read_only=True)
    try:
        count = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731
        assert count("SELECT count(*) FROM daily") == 8 + 6
        assert count("SELECT count(*) FROM daily_limit") == 8 + 6
        assert count("SELECT count(*) FROM adj_factor") == 11
        assert count("SELECT count(*) FROM trade_cal") == 4 + 3
        assert count("SELECT is_open FROM trade_cal WHERE cal_date = DATE '2024-01-01'") == 0
        # 退市票的历史和基础信息都进来了（历史股票池不缺票）；正式库改过的名字不被覆盖
        assert count(f"SELECT count(*) FROM daily WHERE ts_code = '{GONE}'") == 2
        assert count(f"SELECT name FROM stock_basic WHERE ts_code = '{GONE}'") == "退市票"
        assert count(f"SELECT name FROM stock_basic WHERE ts_code = '{A}'") == "新名"
        # 正式库已有的日子一行不动：源库那天多出来的票不补进去，存档过的扫描不受影响
        assert count(f"SELECT count(*) FROM daily WHERE ts_code = '{SRC_ONLY}'") == 0
        assert count(f"SELECT count(*) FROM daily WHERE ts_code = '{TGT_ONLY}'") == 2
    finally:
        con.close()


def test_rerunning_the_import_writes_nothing_and_makes_no_new_backup(dbs: tuple[Path, Path]) -> None:
    source, target = dbs
    import_history(source, target)
    again = import_history(source, target)
    assert not again.written and again.backup is None and len(_backups(target)) == 1
    assert all(plan.insert == 0 for plan in again.tables.values())
    assert all(plan.mismatch == 0 for plan in again.tables.values())


@pytest.mark.parametrize(
    ("table", "sql"),
    [("daily", f"UPDATE daily SET close = close + 0.01 WHERE ts_code = '{A}' AND trade_date = DATE '2024-01-02'"),
     ("daily_limit", f"UPDATE daily_limit SET up_limit = NULL WHERE ts_code = '{B}' AND trade_date = DATE '2024-01-03'"),
     ("trade_cal", "UPDATE trade_cal SET is_open = 0 WHERE cal_date = DATE '2024-01-03'")],
)
def test_a_mismatch_on_an_overlapping_day_refuses_the_whole_import(
    dbs: tuple[Path, Path], table: str, sql: str
) -> None:
    source, target = dbs
    con = duckdb.connect(str(target))
    con.execute(sql)
    con.close()
    before = target.read_bytes()
    plan = import_history(source, target, dry_run=True)
    assert plan.refused and plan.tables[table].mismatch == 1 and plan.tables[table].samples
    with pytest.raises(DataError, match="对不上"):
        import_history(source, target)
    assert target.read_bytes() == before and not _backups(target)


@pytest.mark.parametrize(
    ("table", "sql"),
    [("adj_factor", f"UPDATE adj_factor SET adj_factor = 0 WHERE ts_code = '{A}' AND trade_date = DATE '2022-12-30'"),
     ("adj_factor", f"UPDATE adj_factor SET adj_factor = CAST('inf' AS DOUBLE) WHERE ts_code = '{A}' AND trade_date = DATE '2022-12-30'"),
     ("trade_cal", "UPDATE trade_cal SET is_open = 0.5 WHERE cal_date = DATE '2022-12-30'")],
)
def test_bad_values_in_the_rows_to_copy_refuse_the_import(tmp_path: Path, table: str, sql: str) -> None:
    """复权因子 0 会把复权价抹成 0，开市标记 0.5 会被悄悄取整：同样整次拒绝。"""
    source = _source(tmp_path / "research.duckdb")
    con = duckdb.connect(str(source))
    con.execute(sql)
    con.close()
    target = _target(tmp_path / "market.duckdb")
    before = target.read_bytes()
    plan = import_history(source, target, dry_run=True)
    assert plan.refused and plan.tables[table].invalid == 1
    with pytest.raises(DataError, match="坏值"):
        import_history(source, target)
    assert target.read_bytes() == before and not _backups(target)


def test_beijing_exchange_rows_are_copied_and_compared_like_any_other(tmp_path: Path) -> None:
    """北交所照常搬、照常核对（用户 2026-10-30 定的）：不许有按交易所的例外。"""
    bj = "920001.BJ"
    source = _source(tmp_path / "research.duckdb")
    con = duckdb.connect(str(source))
    con.execute(f"INSERT INTO stock_basic VALUES ('{bj}', '920001', '北交票', '北京', '机械', '北交所', 'BSE', 'L', DATE '2021-01-04', NULL)")
    con.executemany("INSERT INTO daily VALUES (?, ?, 5, 5, 5, 5, 5, 1.0, 100.0, 1000.0)", [(bj, D1), (bj, D4)])
    con.executemany("INSERT INTO adj_factor VALUES (?, ?, 433.6)", [(bj, D1), (bj, D4)])
    con.close()
    target = _target(tmp_path / "market.duckdb")
    con = duckdb.connect(str(target))
    con.execute(f"INSERT INTO daily VALUES ('{bj}', DATE '2024-01-03', 5, 5, 5, 5, 5, 1.0, 999.0, 9999.0)")  # 与研究库不同
    con.close()

    plan = import_history(source, target, dry_run=True)
    assert plan.refused and plan.tables["daily"].mismatch == 1
    assert plan.tables["daily"].samples[0]["key"]["ts_code"] == bj
    assert (plan.tables["daily"].insert, plan.tables["adj_factor"].insert, plan.tables["stock_basic"].insert) == (7, 13, 2)


def _table_rows(path: Path, table: str) -> list[tuple]:
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute(f"SELECT * FROM {table} ORDER BY ALL").fetchall()
    finally:
        con.close()


def test_an_interrupted_import_keeps_whole_years_and_a_rerun_finishes_it(
    dbs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """断在中间：已提交的年份完整、没提交的那年整年回滚；从近往远写，留下的历史从今天往回
    是连续的；重跑补完之后，和一次跑完的结果逐行一样。"""
    from rebound.data import history_import

    source, target = dbs
    before = target.read_bytes()
    real = history_import._insert

    def flaky(con: object, table: str, span: tuple[date, date] | None = None) -> int:
        if span is not None and span[0].year == 2023 and table == "daily":
            raise duckdb.OutOfMemoryException("模拟内存不够")
        return real(con, table, span)

    monkeypatch.setattr(history_import, "_insert", flaky)
    lines: list[str] = []
    with pytest.raises(DataError, match="导入中途失败") as failed:
        import_history(source, target, log=lines.append)
    assert "stock_basic、2024 年" in str(failed.value)
    cal = {row[1] for row in _table_rows(target, "trade_cal")}  # (exchange, cal_date, is_open)
    assert HOLIDAY in cal and D2 not in cal  # 2024 年整年在；2023 年先写的日历也跟着回滚了
    assert len(_table_rows(target, "daily")) == 8

    monkeypatch.setattr(history_import, "_insert", real)
    assert import_history(source, target, log=lines.append).written
    years = [int(match.group(1)) for line in lines if (match := re.match(r"(\d{4}) 年：", line))]
    assert years == [2024, 2023, 2022]
    backups = [path for path in _backups(target) if path.suffix != ".wal"]
    assert len(backups) == 2 and backups[0].read_bytes() == before  # 最早那份就是导入之前

    clean = _target(tmp_path / "clean.duckdb")
    import_history(source, clean)
    for table in ("trade_cal", "stock_basic", "daily", "daily_limit", "adj_factor"):
        assert _table_rows(target, table) == _table_rows(clean, table), table


def test_the_backup_carries_an_unmerged_wal_and_restores_by_the_runbook(dbs: tuple[Path, Path]) -> None:
    """正式库带着没合并的 `.wal`（上次没正常关）：备份要连它一起；按说明还原
    （先删现在的 `.wal`，再把备份和它的 `.wal` 复制回来）回到的正是导入之前。"""
    source, target = dbs
    con = duckdb.connect(str(target))
    con.execute("PRAGMA disable_checkpoint_on_shutdown")
    con.execute("INSERT INTO model_score VALUES (DATE '2026-10-30', '600000.SH', 'nocap', 'M', 0.5, 1, "
                "DATE '2026-10-01', 'abc', '2026-10-30T00:00:00')")
    con.close()
    wal = target.with_name(target.name + ".wal")
    assert wal.exists()

    report = import_history(source, target)
    backup_wal = report.backup.with_name(report.backup.name + ".wal")
    assert backup_wal.exists()
    target.unlink()
    wal.unlink(missing_ok=True)
    shutil.copy2(report.backup, target)
    shutil.copy2(backup_wal, wal)
    assert len(_table_rows(target, "model_score")) == 1  # 只在 WAL 里的那一行还在
    assert len(_table_rows(target, "daily")) == 8  # 导入的行不在


def test_cli_reports_a_failure_during_the_write_as_a_data_error(
    dbs: tuple[Path, Path], capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """写到一半出 DuckDB 的错（内存不够、磁盘满）：退出码 4，说清楚备份在哪，不是一串 traceback。"""
    from rebound.data import history_import

    def broken(*_args: object, **_kwargs: object) -> int:
        raise duckdb.OutOfMemoryException("模拟内存不够")

    monkeypatch.setattr(history_import, "_insert", broken)
    source, target = dbs
    assert main(["data", "import-history", "--source", str(source), "--db", str(target)]) == 4
    err = capsys.readouterr().err
    assert "导入中途失败" in err and ".bak-before-history-import-" in err


def test_missing_databases_are_refused(tmp_path: Path, dbs: tuple[Path, Path]) -> None:
    source, target = dbs
    with pytest.raises(DataError, match="研究库不存在"):
        import_history(tmp_path / "nope.duckdb", target, dry_run=True)
    with pytest.raises(DataError, match="正式库不存在"):
        import_history(source, tmp_path / "nope.duckdb", dry_run=True)
    with pytest.raises(DataError, match="同一个文件"):
        import_history(target, target, dry_run=True)


def test_a_target_held_open_by_another_process_is_refused(tmp_path: Path, dbs: tuple[Path, Path]) -> None:
    """`serve` 开着时 DuckDB 不许第二个进程打开：要说清楚先停 serve，而不是抛一串底层报错。"""
    source, target = dbs
    ready = tmp_path / "ready"
    holder = subprocess.Popen([sys.executable, "-c", (
        "import duckdb, pathlib, time\n"
        f"con = duckdb.connect({str(target)!r})\n"
        f"pathlib.Path({str(ready)!r}).write_text('ok')\n"
        "time.sleep(60)\n")])
    try:
        deadline = time.time() + 30
        while not ready.exists() and time.time() < deadline:
            time.sleep(0.1)
        assert ready.exists(), "持锁进程没起来"
        with pytest.raises(DataError, match="被占用"):
            import_history(source, target, dry_run=True)
    finally:
        holder.kill()
        holder.wait()


def test_cli_dry_run_prints_the_plan_as_json(dbs: tuple[Path, Path], capsys: pytest.CaptureFixture) -> None:
    source, target = dbs
    before = target.read_bytes()
    assert main(["data", "import-history", "--source", str(source), "--db", str(target), "--dry-run", "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["dry_run"] is True and body["tables"]["daily"]["insert"] == 6
    assert target.read_bytes() == before


def test_cli_dry_run_with_a_mismatch_exits_nonzero(dbs: tuple[Path, Path], capsys: pytest.CaptureFixture) -> None:
    source, target = dbs
    con = duckdb.connect(str(target))
    con.execute(f"UPDATE daily SET close = close + 1 WHERE ts_code = '{A}' AND trade_date = DATE '2024-01-02'")
    con.close()
    assert main(["data", "import-history", "--source", str(source), "--db", str(target), "--dry-run", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["refused"] is True


