"""写入层：幂等落库 + 覆盖率体检。

与只读层（`rebound.store.Store`）分开是**有意**的：内核永远拿不到写句柄，
所以"读路径不写库"是类型与依赖上的事实，而不是一句约定。

幂等靠 `INSERT OR REPLACE`（DuckDB 1.5.5 已实测）：同一个业务键重复写就是覆盖，
所以"今天重跑一次"不会产生第二行。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import TracebackType

import duckdb
import pandas as pd

from rebound.data import schema
from rebound.data.errors import DataError


@dataclass(frozen=True, slots=True)
class TableCoverage:
    """一张表的体量：多少行、日期跨度多大。"""

    table: str
    rows: int
    first: date | None
    last: date | None

    def as_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "rows": self.rows,
            "first": self.first.isoformat() if self.first else None,
            "last": self.last.isoformat() if self.last else None,
        }


class WriteStore:
    """可写的 DuckDB 句柄。`with` 管理生命周期，`close()` 幂等。"""

    def __init__(self, db_path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(db_path)
        self.read_only = bool(read_only)
        if not self.read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con: duckdb.DuckDBPyConnection | None = duckdb.connect(
            str(self.path), read_only=self.read_only
        )
        if not self.read_only:
            # 建表全是 IF NOT EXISTS，幂等。放在这里、而不是指望每个调用方记得先调
            # ensure_schema()：忘了的话第一个 upsert 会报"库内列与 schema 不一致: ()"
            # ——表根本还不存在，那句话会把人带到错误的排查方向上去。
            self.ensure_schema()

    # ------------------------------------------------------------- 生命周期
    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            raise RuntimeError("WriteStore 已关闭")
        return self._con

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> WriteStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """一天的数据要么全进、要么全不进。

        半截状态（日线写了、复权因子没写）在下游无法区分"这天本来就缺"和"那天写崩了"。
        事务把这两种情况分开。
        """
        connection = self.connection
        connection.execute("BEGIN TRANSACTION")
        try:
            yield
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")

    # ----------------------------------------------------------------- 结构
    def ensure_schema(self) -> None:
        schema.ensure_schema(self.connection)

    def table_columns(self, table: str) -> tuple[str, ...]:
        rows = self.connection.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name = ? ORDER BY ordinal_position
            """,
            [table],
        ).fetchall()
        return tuple(row[0] for row in rows)

    def has_table(self, table: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [table]
        ).fetchone()
        return row is not None

    # ----------------------------------------------------------------- 写入
    def upsert(self, table: str, frame: pd.DataFrame) -> int:
        """按业务主键幂等写入。返回写入行数。

        列必须与建表列**完全一致**：多一列或漏一列都抛。静默穿透是 v1 里
        "表里有个字段从来没人填"的来源。
        """
        if table not in schema.TABLE_FIELDS:
            raise DataError(f"未知的表: {table}（schema.TABLE_FIELDS 之外的表不许写）")
        expected = schema.TABLE_FIELDS[table]
        actual = self.table_columns(table)
        if not actual:
            raise DataError(
                f"表 {table} 不存在（用 WriteStore(read_only=False) 打开时会自动建表）"
            )
        if actual != expected:
            raise DataError(
                f"{table} 的库内列与 schema.TABLE_FIELDS 不一致: {actual} != {expected}"
                "（先跑 ensure_schema，或同步修改 schema.py）"
            )
        if frame.empty:
            return 0
        missing = [column for column in expected if column not in frame.columns]
        if missing:
            raise DataError(f"{table} 写入缺少列: {missing}")
        extra = [column for column in frame.columns if column not in expected]
        if extra:
            raise DataError(f"{table} 写入含表外列: {extra}（请显式丢弃，不许静默穿透）")
        return self._insert(table, frame, "INSERT OR REPLACE")

    def append(self, table: str, frame: pd.DataFrame) -> int:
        """不按主键去重的追加（给没建主键的大表，比如 `model_score`）。列的校验和 `upsert` 相同。

        去重是调用方的责任：调用方必须在同一个事务里先查过哪些键已经有了。
        """
        self.upsert_check(table, frame)
        if frame.empty:
            return 0
        return self._insert(table, frame, "INSERT")

    def upsert_check(self, table: str, frame: pd.DataFrame) -> None:
        """`upsert` / `append` 共用的列校验（不写库）。"""
        if table not in schema.TABLE_FIELDS:
            raise DataError(f"未知的表: {table}（schema.TABLE_FIELDS 之外的表不许写）")
        expected = schema.TABLE_FIELDS[table]
        actual = self.table_columns(table)
        if not actual:
            raise DataError(f"表 {table} 不存在（用 WriteStore(read_only=False) 打开时会自动建表）")
        if actual != expected:
            raise DataError(f"{table} 的库内列与 schema.TABLE_FIELDS 不一致: {actual} != {expected}")
        missing = [column for column in expected if column not in frame.columns]
        extra = [column for column in frame.columns if column not in expected]
        if not frame.empty and (missing or extra):
            raise DataError(f"{table} 写入缺少列 {missing} / 含表外列 {extra}")

    def _insert(self, table: str, frame: pd.DataFrame, verb: str) -> int:
        ordered = frame.loc[:, list(schema.TABLE_FIELDS[table])]
        self.connection.register("_upsert_frame", ordered)
        try:
            self.connection.execute(f"{verb} INTO {table} BY NAME SELECT * FROM _upsert_frame")
        finally:
            self.connection.unregister("_upsert_frame")
        return len(ordered)

    # ----------------------------------------------------------------- 体检
    def row_count(self, table: str, *, trade_date: date | None = None) -> int:
        if trade_date is None:
            row = self.connection.execute(f"SELECT count(*) FROM {table}").fetchone()
        else:
            column = schema.TABLE_DATE_COLUMNS[table]
            row = self.connection.execute(
                f"SELECT count(*) FROM {table} WHERE {column} = ?", [trade_date]
            ).fetchone()
        return int(row[0]) if row else 0

    def coverage(self, *, tables: tuple[str, ...] | None = None) -> tuple[TableCoverage, ...]:
        """逐表的行数与首末日期。默认只报行情表。

        打分表（`model_score`）不算缺口：它不是从上游取的行情。想看它就显式传 `tables=`。
        """
        out: list[TableCoverage] = []
        for table in tables if tables is not None else schema.MARKET_TABLES:
            if not self.has_table(table):
                out.append(TableCoverage(table, 0, None, None))
                continue
            column = schema.TABLE_DATE_COLUMNS[table]
            row = self.connection.execute(
                f"SELECT count(*), min({column}), max({column}) FROM {table}"
            ).fetchone()
            out.append(
                TableCoverage(
                    table,
                    int(row[0]) if row else 0,
                    row[1] if row else None,
                    row[2] if row else None,
                )
            )
        return tuple(out)

    def open_dates(
        self, *, exchange: str = "SSE", start: date | None = None, end: date | None = None
    ) -> tuple[date, ...]:
        """本地日历里的开市日（升序）。日历不全时返回的自然也不全。"""
        clauses = ["exchange = ?", "is_open = 1"]
        params: list[object] = [exchange]
        if start is not None:
            clauses.append("cal_date >= ?")
            params.append(start)
        if end is not None:
            clauses.append("cal_date <= ?")
            params.append(end)
        rows = self.connection.execute(
            f"SELECT cal_date FROM trade_cal WHERE {' AND '.join(clauses)} ORDER BY cal_date",
            params,
        ).fetchall()
        return tuple(row[0] for row in rows)

    def dates_with_rows(self, table: str) -> dict[date, int]:
        """每张表按日期分组的行数。缺口检测的原料。"""
        column = schema.TABLE_DATE_COLUMNS[table]
        rows = self.connection.execute(
            f"SELECT {column}, count(*) FROM {table} GROUP BY {column} ORDER BY {column}"
        ).fetchall()
        return {row[0]: int(row[1]) for row in rows}

    def calendar_range(self, *, exchange: str = "SSE") -> tuple[date, date] | None:
        row = self.connection.execute(
            "SELECT min(cal_date), max(cal_date) FROM trade_cal WHERE exchange = ?", [exchange]
        ).fetchone()
        if not row or row[0] is None:
            return None
        return (row[0], row[1])
