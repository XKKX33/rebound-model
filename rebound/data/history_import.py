"""一次性历史导入：另一个库（源库） → 本库。

`rebound sync --days N` 从 Tushare 按天回补，是建库的常规做法；手上已经有一份按同一套 Tushare
接口存好的库（比如以前做研究时拉的）时，用这里把本库缺的那几年直接搬过来，省掉几千次网络调用。规则是：

1. **源库只读**。以 `READ_ONLY` 挂上，这里的任何 bug 都碰不到它。
2. **只补本库完全没有的日子**（每张表各自判断），已有的日子一行不动 ——
   已经打过分的日子是用那天的数据算的，补进新行等于改了它的输入。
   `stock_basic` 只补本库没有的代码（主要是已退市的票，历史股票池里少了它们
   就有幸存者偏差），本库已有的名称、行业不覆盖。
3. **重叠的日子逐行核对**价格、成交、涨跌停价、复权因子和开市标记。对不上一行
   就整次拒绝，在备份和写入之前 —— 对不上说明两边不是同一份历史，拼起来的
   序列不可信。要搬的行里出现坏值（复权因子 ≤ 0、开市标记不是 0/1）同样拒绝。
4. **写之前先备份**：库文件（和没合并的 `.wal`）复制成
   `<库名>.bak-before-history-import-<时间>`。没东西可补就不备份、不写。
   还原时先删掉本库现在的 `.wal`，否则 DuckDB 会把它重放到还原回来的文件上。
5. **按年提交，从近往远**，同一年的几张表在一个事务里，每提交一年就 `CHECKPOINT`。
   中途断了，已提交的年份完整、没提交的那年整年回滚，留下的历史从今天往回连续；
   重跑只补剩下的。

北交所（`.BJ`）照常搬，但不要直接信源库的北交所复权因子：曾经抽查过，同样的日子和现在的
Tushare 比，有十来只每天都对不上。股票池都是 `board: mainboard`，没有代码用北交所；
要算北交所的复权价，先从 Tushare 重拉这几只的因子。

DuckDB 同一时间只许一个进程打开写：别的进程开着本库时，这里连只读都打不开，
报错会说是谁占着。
"""

from __future__ import annotations

import math
import re
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb

from rebound.data import schema
from rebound.data.db import WriteStore
from rebound.data.errors import DataError

#: 研究库挂上来的名字。不叫 `src`：正式库文件要是正好叫 `src.duckdb` 就撞名了。
SOURCE = "research_src"

#: 按"日子"补的表 -> 哪几列定义"一天"。`trade_cal` 按交易所分，其余按交易日。
DAY_SCOPE: dict[str, tuple[str, ...]] = {
    "trade_cal": ("exchange", "cal_date"),
    "daily": ("trade_date",),
    "daily_limit": ("trade_date",),
    "adj_factor": ("trade_date",),
}

#: 报告和写入的顺序：基础信息先进来，后面的行情才有名字。
TABLES: tuple[str, ...] = ("stock_basic",) + tuple(DAY_SCOPE)

#: 要搬的行里算坏值的条件。研究库是 `DOUBLE`，正式库 `is_open` 是 `INTEGER`：
#: 0.5 写进去会被悄悄取整，所以先挡住。`isfinite` 对 NaN 和 ±inf 都是假。
INVALID: dict[str, str] = {
    "trade_cal": "is_open IS NULL OR isnan(is_open) OR is_open NOT IN (0, 1)",
    "adj_factor": "adj_factor IS NULL OR NOT isfinite(adj_factor) OR adj_factor <= 0",
}

#: 导入时 DuckDB 的内存和线程上限。默认是物理内存的 80%、全部核心（这台机器上
#: 12.5 GiB / 24 线程），一次性的后台导入不该和别的进程抢。实测 2026-10-31：
#: 2150 万行的全量复制品在这个上限下 42 秒写完，峰值 714 MB；不设上限峰值 1,023 MB。
MEMORY_LIMIT = "600MB"
THREADS = 2

#: 一张表最多带几行对不上的样例，够人看清是哪种差异就行。
SAMPLE_LIMIT = 5

_HOLDER = re.compile(r"File is already open in\s*(.+)", re.DOTALL)


@dataclass(frozen=True, slots=True)
class TablePlan:
    """一张表要补什么、重叠部分核对得怎么样。

    `overlap` 是两边都有的行：按日子补的表逐行核对，`mismatch` 是其中对不上的；
    `stock_basic` 只数不比（改名、换行业是正常的，正式库为准）。
    `target_only` / `source_only` 只在重叠的日子里数：那几天某一边多出来的票。
    """

    table: str
    insert: int = 0
    insert_days: int = 0
    first: date | None = None
    last: date | None = None
    overlap: int = 0
    mismatch: int = 0
    target_only: int = 0
    source_only: int = 0
    invalid: int = 0
    samples: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "insert": self.insert,
            "insert_days": self.insert_days,
            "first": self.first.isoformat() if self.first else None,
            "last": self.last.isoformat() if self.last else None,
            "overlap": self.overlap,
            "mismatch": self.mismatch,
            "target_only": self.target_only,
            "source_only": self.source_only,
            "invalid": self.invalid,
            "samples": list(self.samples),
        }

@dataclass(frozen=True, slots=True)
class ImportReport:
    source: Path
    target: Path
    dry_run: bool
    tables: dict[str, TablePlan] = field(default_factory=dict)
    written: bool = False
    backup: Path | None = None

    @property
    def refused(self) -> bool:
        """重叠日有一行对不上、或要搬的行里有坏值，就整次不写。"""
        return any(plan.mismatch or plan.invalid for plan in self.tables.values())

    @property
    def insert_rows(self) -> int:
        return sum(plan.insert for plan in self.tables.values())

    def as_dict(self) -> dict[str, object]:
        return {
            "source": str(self.source),
            "target": str(self.target),
            "dry_run": self.dry_run,
            "refused": self.refused,
            "written": self.written,
            "backup": str(self.backup) if self.backup else None,
            "insert_rows": self.insert_rows,
            "tables": {name: plan.as_dict() for name, plan in self.tables.items()},
        }


def import_history(
    source: Path,
    target: Path,
    *,
    dry_run: bool = False,
    log: Callable[[str], None] | None = None,
) -> ImportReport:
    """先核对、再备份、再按年写。

    演练（`dry_run`）只读两边，对不上也照样返回报告（`refused` 为真）；真写时
    对不上直接抛 `DataError`，正式库和备份目录都不动。
    """
    source, target = Path(source), Path(target)
    if not source.exists():
        raise DataError(f"研究库不存在：{source}")
    if not target.exists():
        # 不替人新建：路径写错时安静地建一个空库再往里倒几年历史，比报错难查得多
        raise DataError(f"正式库不存在：{target}（先跑一次 rebound sync 建库）")
    if source.resolve() == target.resolve():
        raise DataError(f"研究库和正式库是同一个文件：{source}")
    say = log or (lambda _line: None)

    with _opened(source, target, read_only=True) as store:
        plans = _plan(store.connection)
    report = ImportReport(source=source, target=target, dry_run=dry_run, tables=plans)
    if report.refused:
        if dry_run:
            return report
        raise DataError(_refusal(report))
    if dry_run or report.insert_rows == 0:
        return report

    backup = _backup(target)
    say(f"已备份正式库：{backup}")
    committed: list[str] = []
    try:
        with _opened(source, target, read_only=False) as store:
            written = _write(store, plans, say, committed)
    except (duckdb.Error, OSError) as exc:
        # 内存不够、磁盘满这类不是 DataError 的失败：也要说清楚哪些写进去了、备份在哪
        raise _interrupted(target, backup, committed, exc) from exc
    return replace(report, tables=written, written=True, backup=backup)

@contextmanager
def _opened(source: Path, target: Path, *, read_only: bool) -> Iterator[WriteStore]:
    """正式库按需只读或可写（可写时建齐缺的表），研究库永远只读挂上。"""
    try:
        store = WriteStore(target, read_only=read_only)
    except duckdb.IOException as exc:
        raise _locked("正式库", target, exc) from exc
    try:
        store.connection.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
        store.connection.execute(f"SET threads = {THREADS}")
        try:
            store.connection.execute(f"ATTACH {_literal(source)} AS {SOURCE} (READ_ONLY)")
        except duckdb.IOException as exc:
            raise _locked("研究库", source, exc) from exc
        yield store
    finally:
        store.close()


def _locked(label: str, path: Path, exc: Exception) -> DataError:
    """DuckDB 的锁冲突报错夹着本地化乱码，只留下"谁占着"这一句。"""
    holder = _HOLDER.search(str(exc))
    if holder is None:
        return DataError(f"{label}打不开：{path}：{exc}")
    return DataError(
        f"{label}被占用：{path} 正被 {holder.group(1).strip()} 打开。"
        "DuckDB 同一时间只许一个进程打开写，先停掉别的开着它的进程再导入"
    )


def _literal(path: Path) -> str:
    """路径写进 SQL 字符串字面量（`ATTACH` 不接受参数绑定）。"""
    return "'" + str(path).replace("'", "''") + "'"


def _plan(con: duckdb.DuckDBPyConnection) -> dict[str, TablePlan]:
    for table in TABLES:
        _require_source(con, table)
    return {
        table: _plan_basic(con) if table == "stock_basic" else _plan_days(con, table)
        for table in TABLES
    }


def _require_source(con: duckdb.DuckDBPyConnection, table: str) -> None:
    """研究库的表必须有正式库要的每一列，日期列必须是 DATE（不替人猜日期格式）。"""
    found = dict(con.execute(
        "SELECT column_name, data_type FROM duckdb_columns() "
        "WHERE database_name = ? AND schema_name = 'main' AND table_name = ?",
        [SOURCE, table],
    ).fetchall())
    if not found:
        raise DataError(f"研究库里没有 {table} 表")
    missing = [column for column in schema.TABLE_FIELDS[table] if column not in found]
    if missing:
        raise DataError(f"研究库 {table} 缺列：{missing}")
    day = schema.TABLE_DATE_COLUMNS[table]
    if found[day] != "DATE":
        raise DataError(f"研究库 {table}.{day} 是 {found[day]}，不是 DATE")


def _target_table(con: duckdb.DuckDBPyConnection, table: str) -> str | None:
    """正式库这张表的全名（带库名）；表不存在返回 None。

    显式带库名：挂上研究库之后，不带库名的表名虽然仍指正式库，
    但这里要的是"读错库、写错库"在结构上就不可能。
    """
    target = con.execute("SELECT current_database()").fetchone()[0]
    exists = con.execute(
        "SELECT count(*) FROM duckdb_tables() "
        "WHERE database_name = ? AND schema_name = 'main' AND table_name = ?",
        [target, table],
    ).fetchone()[0]
    if not exists:
        return None
    return '"' + target.replace('"', '""') + f'".main.{table}'


def _target_rows(con: duckdb.DuckDBPyConnection, table: str) -> str:
    """正式库的这张表；老库还没有它（比如升级前的 `adj_factor`）时按空表算。"""
    name = _target_table(con, table)
    if name is not None:
        return name
    columns = ", ".join(schema.TABLE_FIELDS[table])
    return f"(SELECT {columns} FROM {SOURCE}.main.{table} WHERE false)"

def _plan_days(con: duckdb.DuckDBPyConnection, table: str) -> TablePlan:
    scope = DAY_SCOPE[table]
    keys = schema.TABLE_PRIMARY_KEYS[table]
    values = [column for column in schema.TABLE_FIELDS[table] if column not in keys]
    day = schema.TABLE_DATE_COLUMNS[table]
    source = f"{SOURCE}.main.{table}"
    target = _target_rows(con, table)
    scope_list = ", ".join(scope)
    invalid = INVALID.get(table, "false")

    # 正式库完全没有的日子：这些行整天搬过去。
    insert, insert_days, first, last, bad = con.execute(f"""
        SELECT count(*), count(DISTINCT {day}), min({day}), max({day}),
               count(*) FILTER (WHERE {invalid})
        FROM {source} s ANTI JOIN (SELECT DISTINCT {scope_list} FROM {target}) t USING ({scope_list})
    """).fetchone()

    # 两边都有的日子：逐行全外连接，数出对不上的行和只有一边有的票。
    on = " AND ".join(f"s.{key} = t.{key}" for key in keys)
    differs = " OR ".join(f"s.{column} IS DISTINCT FROM t.{column}" for column in values)
    both = f"""
        WITH days AS (SELECT DISTINCT {scope_list} FROM {source}
                      INTERSECT SELECT DISTINCT {scope_list} FROM {target}),
             s AS (SELECT * FROM {source} SEMI JOIN days USING ({scope_list})),
             t AS (SELECT * FROM {target} SEMI JOIN days USING ({scope_list}))
    """
    overlap, mismatch, target_only, source_only = con.execute(f"""
        {both}
        SELECT count(*) FILTER (WHERE s.{keys[0]} IS NOT NULL AND t.{keys[0]} IS NOT NULL),
               count(*) FILTER (WHERE s.{keys[0]} IS NOT NULL AND t.{keys[0]} IS NOT NULL
                                AND ({differs})),
               count(*) FILTER (WHERE s.{keys[0]} IS NULL),
               count(*) FILTER (WHERE t.{keys[0]} IS NULL)
        FROM s FULL OUTER JOIN t ON {on}
    """).fetchone()

    samples: tuple[dict[str, Any], ...] = ()
    if mismatch:
        picked = ", ".join(
            [f"s.{key}" for key in keys]
            + [f"s.{c} IS DISTINCT FROM t.{c}, s.{c}, t.{c}" for c in values]
        )
        rows = con.execute(f"""
            {both}
            SELECT {picked} FROM s JOIN t ON {on} WHERE {differs}
            ORDER BY {", ".join(f"s.{key}" for key in keys)} LIMIT {SAMPLE_LIMIT}
        """).fetchall()
        samples = tuple(_sample(row, keys, values) for row in rows)
    return TablePlan(
        table=table, insert=insert, insert_days=insert_days, first=first, last=last,
        overlap=overlap, mismatch=mismatch, target_only=target_only,
        source_only=source_only, invalid=bad, samples=samples,
    )

def _sample(row: tuple, keys: tuple[str, ...], values: list[str]) -> dict[str, Any]:
    """一行对不上的样例：主键 + 只列出真的不一样的那几列（研究库值 / 正式库值）。"""
    head, rest = row[: len(keys)], row[len(keys):]
    diff = {
        column: {"source": _plain(rest[3 * i + 1]), "target": _plain(rest[3 * i + 2])}
        for i, column in enumerate(values)
        if rest[3 * i]
    }
    return {"key": {key: _plain(value) for key, value in zip(keys, head)}, "diff": diff}


def _plain(value: Any) -> Any:
    """给 JSON 的值：日期转字符串，NaN 转 null。"""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _plan_basic(con: duckdb.DuckDBPyConnection) -> TablePlan:
    """只补正式库没有的代码。重叠的代码不比：改名、换行业是正常的，正式库为准。"""
    target = _target_rows(con, "stock_basic")
    insert, overlap = con.execute(f"""
        SELECT count(*) FILTER (WHERE t.ts_code IS NULL), count(t.ts_code)
        FROM {SOURCE}.main.stock_basic s LEFT JOIN {target} t ON s.ts_code = t.ts_code
    """).fetchone()
    return TablePlan(table="stock_basic", insert=insert, overlap=overlap)


def _write(
    store: WriteStore,
    plans: dict[str, TablePlan],
    say: Callable[[str], None],
    committed: list[str],
) -> dict[str, TablePlan]:
    """先补基础信息，再从近往远逐年补带日期的几张表；返回的计划里 `insert` 换成实际写入数。

    从近往远：断在中间时，留下的是从今天往回连续的一段。从远往近断在中间会留下一个洞，
    而 `Store.sessions()` 按开市日数数，会直接跨过它，把隔了几年的日线拼在一起。
    每提交一批就 `CHECKPOINT`，把 WAL 并进库文件：之后被杀掉，库文件本身就是完整的。
    `committed` 是给调用方的进度：写到一半失败时，报错要说清楚哪几批已经在库里。
    """
    con = store.connection
    written = dict.fromkeys(plans, 0)
    if plans["stock_basic"].insert:
        with store.transaction():
            written["stock_basic"] = _insert(con, "stock_basic")
        committed.append("stock_basic")
        con.execute("CHECKPOINT")
        say(f"stock_basic 补了 {written['stock_basic']} 只代码")
    for year in reversed(_years(plans)):
        span = (date(year, 1, 1), date(year, 12, 31))
        with store.transaction():
            counts = {table: _insert(con, table, span) for table in DAY_SCOPE if plans[table].insert}
        committed.append(f"{year} 年")
        con.execute("CHECKPOINT")
        written = {table: written[table] + counts.get(table, 0) for table in written}
        say(f"{year} 年：" + "，".join(f"{table} {rows} 行" for table, rows in counts.items()))
    return {table: replace(plan, insert=written[table]) for table, plan in plans.items()}


def _interrupted(target: Path, backup: Path, committed: list[str], exc: Exception) -> DataError:
    done = "、".join(committed) or "无"
    return DataError(
        f"导入中途失败（{type(exc).__name__}: {exc}）。\n"
        f"    已提交：{done}；没提交的那一批整批回滚了，库里没有半截的年份。\n"
        f"    导入之前的备份：{backup}\n"
        "    原样重跑同一条命令只补剩下的。要退回导入之前：按 docs/RUNBOOK.md 的“错误与恢复”，"
        "确认目标库和备份路径后再恢复。"
    )

def _years(plans: dict[str, TablePlan]) -> range:
    spans = [
        (plan.first.year, plan.last.year)
        for table, plan in plans.items()
        if table in DAY_SCOPE and plan.insert and plan.first and plan.last
    ]
    if not spans:
        return range(0)
    return range(min(first for first, _ in spans), max(last for _, last in spans) + 1)


def _insert(con: duckdb.DuckDBPyConnection, table: str, span: tuple[date, date] | None = None) -> int:
    """把研究库里正式库缺的行写进去，返回写入行数。

    缺什么在写的这一刻重新算（反连接正式库），所以重跑、或核对之后有人同步过，
    都不会写重。
    """
    target = _target_table(con, table)
    if target is None:  # 可写打开时 ensure_schema 已经建齐，走到这里就是 schema 漏了表
        raise DataError(f"正式库没有 {table} 表，写不进去")
    columns = schema.TABLE_FIELDS[table]
    picked = ", ".join(
        f"CAST(s.{column} AS INTEGER)" if (table, column) == ("trade_cal", "is_open") else f"s.{column}"
        for column in columns
    )
    source = f"{SOURCE}.main.{table}"
    if table == "stock_basic":
        sql = f"""
            INSERT INTO {target} ({", ".join(columns)})
            SELECT {picked} FROM {source} s ANTI JOIN {target} t ON s.ts_code = t.ts_code
        """
        return int(con.execute(sql).fetchone()[0])
    scope = ", ".join(DAY_SCOPE[table])
    day = schema.TABLE_DATE_COLUMNS[table]
    sql = f"""
        INSERT INTO {target} ({", ".join(columns)})
        SELECT {picked} FROM {source} s ANTI JOIN (SELECT DISTINCT {scope} FROM {target}) t USING ({scope})
        WHERE s.{day} BETWEEN ? AND ?
    """
    return int(con.execute(sql, list(span or (date.min, date.max))).fetchone()[0])


def _backup(target: Path) -> Path:
    """库文件和没合并进去的 `.wal` 一起复制；只复制主文件会丢掉 WAL 里的写入。

    同一秒里重跑（比如失败后马上重跑）时名字加 `-2`、`-3`，不覆盖已有的备份。
    复制失败（磁盘满）就删掉半截的备份再报错：半截的备份比没有备份更危险。
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{target.name}.bak-before-history-import-{stamp}"
    backup = target.with_name(base)
    serial = 2
    while backup.exists():
        backup = target.with_name(f"{base}-{serial}")
        serial += 1
    wal = target.with_name(target.name + ".wal")
    copies = [(target, backup)] + ([(wal, backup.with_name(backup.name + ".wal"))] if wal.exists() else [])
    try:
        for origin, copy in copies:
            shutil.copy2(origin, copy)
    except OSError as exc:
        for _, copy in copies:
            copy.unlink(missing_ok=True)
        raise DataError(f"备份正式库失败，什么都没写：{exc}") from exc
    return backup


def _refusal(report: ImportReport) -> str:
    lines = ["研究库和正式库对不上，或要搬的行里有坏值：整次拒绝导入，正式库没动、也没备份。"]
    for plan in report.tables.values():
        if plan.mismatch:
            lines.append(f"  {plan.table}：重叠日 {plan.overlap} 行里 {plan.mismatch} 行对不上，例如 {plan.samples[0]}")
        if plan.invalid:
            lines.append(f"  {plan.table}：要搬的行里 {plan.invalid} 行是坏值（{INVALID[plan.table]}）")
    lines.append("用 rebound data import-history --dry-run 看全部样例")
    return "\n".join(lines)
