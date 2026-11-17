"""`rebound/data/schema.py` 的自洽性：DDL 与那几张映射表必须逐列一致。

`rebound/data/schema.py` 的 docstring 写着"与下面的 DDL 逐列交叉校验，两边不许漂移"——
这里就是那个校验。它不连真实库，只建一个临时库，所以永远能跑。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rebound.data import schema
from rebound.data.db import WriteStore


@pytest.fixture
def store(tmp_path: Path) -> WriteStore:
    with WriteStore(tmp_path / "schema.duckdb") as opened:
        yield opened


def test_every_declared_table_is_created(store: WriteStore) -> None:
    assert set(schema.TABLE_ORDER) == set(schema.TABLE_FIELDS)
    for table in schema.TABLE_ORDER:
        assert store.has_table(table), table


def test_ddl_columns_match_table_fields_exactly(store: WriteStore) -> None:
    """两边漂移的代价：`upsert` 会在运行期才报"表外列 / 缺少列"。"""
    for table in schema.TABLE_ORDER:
        assert store.table_columns(table) == schema.TABLE_FIELDS[table], table


def test_primary_keys_and_date_columns_are_declared(store: WriteStore) -> None:
    for table in schema.TABLE_ORDER:
        assert schema.TABLE_PRIMARY_KEYS[table], table
        assert set(schema.TABLE_PRIMARY_KEYS[table]) <= set(schema.TABLE_FIELDS[table]), table
        assert schema.TABLE_DATE_COLUMNS[table] in schema.TABLE_FIELDS[table], table


def test_date_columns_are_real_dates_not_strings(store: WriteStore) -> None:
    """v1 把日期存成 VARCHAR，代价是 11 个入口各埋一次 ConversionException。"""
    rows = store.connection.execute(
        "SELECT table_name, column_name, data_type FROM information_schema.columns"
    ).fetchall()
    types = {(table, column): kind for table, column, kind in rows}
    for table in schema.TABLE_ORDER:
        column = schema.TABLE_DATE_COLUMNS[table]
        assert types[(table, column)] == "DATE", (table, column, types[(table, column)])


def test_the_table_groups_partition_the_order(store: WriteStore) -> None:
    """行情表和模型表不相交、合起来恰好是 `TABLE_ORDER`：新加的表要么归进一组，要么不建。"""
    groups = (schema.MARKET_TABLES, schema.MODEL_TABLES)
    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            assert set(groups[left]) & set(groups[right]) == set(), (left, right)
    assert schema.TABLE_ORDER == sum(groups, ())


def test_coverage_defaults_to_market_tables_only(store: WriteStore) -> None:
    """打分表不是"每个交易日都从上游取"的，报成缺口就永远不健康。"""
    assert {item.table for item in store.coverage()} == set(schema.MARKET_TABLES)
    both = store.coverage(tables=schema.TABLE_ORDER)
    assert {item.table for item in both} == set(schema.TABLE_ORDER)


def test_no_market_table_is_missing_from_the_reader_doc() -> None:
    """docstring 的"表 → 谁在读"那张表不许漏表：漏了就等于没人知道谁消费它。"""
    doc = schema.__doc__ or ""
    for table in schema.TABLE_ORDER:
        assert f"`{table}`" in doc, table


def test_duckdb_accepts_every_declared_column_name(store: WriteStore) -> None:
    """列名撞上 DuckDB 关键字时，建表能过但读写会炸 —— 这里逐个真解析一遍。"""
    for table in schema.TABLE_ORDER:
        columns = ", ".join(schema.TABLE_FIELDS[table])
        store.connection.execute(f"SELECT {columns} FROM {table} LIMIT 0")
