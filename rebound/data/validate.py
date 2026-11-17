"""写库前的校验：一份 DataFrame 全部通过才允许进库。

这是从 v1 `engine/ingest_tushare.py::_validated_snapshot_frame` 搬过来的纪律。
v1 靠它挡掉了「`daily_basic.volume_ratio` 含有非法数值」「`suspend_d 20260811`
缺少字段: ['suspend_date']」「`daily` 全市场覆盖率未达到 100%」这类真实事故。

v2 只做一处改动：**字符串日期在这里转成 `datetime.date`**，成为全项目唯一的
字符串日期入口。v1 把 `'20261014'` 一路带到 SQL 层，于是每个查询点都得自己 cast。

规矩只有一条：校验不过就抛，绝不"跳过这一列/这一行"继续。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np
import pandas as pd

from rebound.data.errors import ValidationError

# 上游（Tushare）给的是 `'20261014'`；也接受带连字符的写法，方便测试与人工造数。
_DATE_FORMATS: tuple[str, ...] = ("%Y%m%d", "%Y-%m-%d")


def coerce_date_column(frame: pd.DataFrame, column: str, *, endpoint: str) -> pd.DataFrame:
    """把一列字符串日期转成 `date`，转换前后非空计数必须相等。

    计数守卫是关键：`errors="coerce"` 会把"上游给了垃圾"变成静默的 NaT，
    而那正是最需要报出来的情况。
    """
    before = int(frame[column].notna().sum())
    parsed = None
    for fmt in _DATE_FORMATS:
        candidate = pd.to_datetime(frame[column], format=fmt, errors="coerce")
        if int(candidate.notna().sum()) == before:
            parsed = candidate
            break
    if parsed is None:
        bad = frame.loc[pd.to_datetime(frame[column], errors="coerce").isna(), column]
        raise ValidationError(
            f"{endpoint}.{column} 不是可解析的日期: {sorted(map(str, bad.unique()))[:5]}"
        )
    out = frame.copy()
    out[column] = parsed.dt.date
    return out


def validated_snapshot(
    frame: object,
    *,
    endpoint: str,
    required: Sequence[str],
    business_keys: Sequence[str],
    date_column: str | None = None,
    expected_date: date | None = None,
    numeric: Sequence[str] = (),
    nullable_numeric: Sequence[str] = (),
    nonnegative_numeric: Sequence[str] = (),
    allowed_values: Mapping[str, Sequence[object]] | None = None,
) -> pd.DataFrame:
    """校验一份上游快照，通过则返回列已按 `required` 排好、日期已转成 `date` 的副本。

    十项检查，逐条对应一类真实事故：

    1. 非表 / 空表           -> 「返回空数据」（不要把"上游挂了"当成"今天没有行情"）
    2. 缺字段                -> 「缺少字段」（v1 的 `suspend_d` 事故）
    3. 日期列不可解析        -> 转换计数守卫
    4. 日期不是要的那一天    -> 「返回了其他交易日数据」
    5. 业务键为空            -> 「.ts_code 不可为空」
    6. 业务键重复            -> 「返回重复业务键」
    7. 数值列不可转数值      -> 「包含非法数值」（v1 的 `volume_ratio` 事故）
    8. 数值列出现 NaN        -> 除非显式列入 `nullable_numeric`（保留上游真实缺失）
    9. 数值列出现 inf        -> 「包含非法数值」
    10. 非负列出现负数       -> 「包含非法数值」
    11. 枚举列出现未定义取值 -> 「出现未定义取值」（`is_open` 只许 0/1）
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValidationError(f"{endpoint} 返回空数据")

    out = frame.copy()
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise ValidationError(f"{endpoint} 缺少字段: {sorted(missing)}")

    if date_column is not None:
        out = coerce_date_column(out, date_column, endpoint=endpoint)
        if expected_date is not None:
            other = sorted({value for value in out[date_column] if value != expected_date})
            if other:
                raise ValidationError(
                    f"{endpoint} {expected_date} 返回了其他交易日数据: {other[:5]}"
                )

    for key in business_keys:
        blank = out[key].isna() | (out[key].astype("string").str.strip() == "")
        if bool(blank.any()):
            raise ValidationError(f"{endpoint}.{key} 不可为空（{int(blank.sum())} 行）")

    duplicated = int(out.duplicated(subset=list(business_keys)).sum())
    if duplicated:
        raise ValidationError(f"{endpoint} 返回重复业务键（{duplicated} 行）")

    nullable = set(nullable_numeric)
    for column in numeric:
        raw = out[column]
        converted = pd.to_numeric(raw, errors="coerce")
        if bool((raw.notna() & converted.isna()).any()):
            bad = raw.loc[raw.notna() & converted.isna()]
            raise ValidationError(
                f"{endpoint}.{column} 包含非法数值: {sorted(map(str, bad.unique()))[:5]}"
            )
        if column not in nullable:
            hole = int(converted.isna().sum())
            if hole:
                raise ValidationError(f"{endpoint}.{column} 存在缺失值（{hole} 行）")
        finite = converted.dropna().to_numpy(dtype="float64")
        if finite.size and not bool(np.isfinite(finite).all()):
            raise ValidationError(f"{endpoint}.{column} 包含非法数值（inf）")
        if column in set(nonnegative_numeric) and finite.size and bool((finite < 0).any()):
            raise ValidationError(f"{endpoint}.{column} 包含非法数值（负数）")
        out[column] = converted

    for column, permitted in (allowed_values or {}).items():
        unexpected = sorted(
            set(out[column].dropna().unique()) - set(permitted), key=str
        )
        if unexpected:
            raise ValidationError(
                f"{endpoint}.{column} 出现未定义取值: {unexpected[:5]}"
                f"（允许 {sorted(permitted, key=str)}）"
            )

    return out.loc[:, list(required)]
