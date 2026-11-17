"""DuckDB 只读访问层：内核唯一的数据出入口。

两条纪律：

1. **日期一律是 `datetime.date`**。库里一律用 `DATE` 列，这一层是日期类型唯一的转换边界。
2. **只读优先**。`read_only=True` 是默认值，写库需要显式声明意图。

这一层不做任何业务判断：没有归一化、没有补零，只把表读成 DataFrame。唯一的例外是可见窗口
（`visible_through`）：设了它之后，读路径不许越过那一天（防前视）。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import TracebackType

import duckdb
import pandas as pd

# 截面快照：只取真正用到的列。
_SNAPSHOT_SQL = """
    SELECT d.ts_code, d.trade_date, d.close, d.pct_chg, d.amount,
           b.name, b.industry
    FROM daily d
    LEFT JOIN stock_basic b ON d.ts_code = b.ts_code
    WHERE d.trade_date = ?
"""

_SESSIONS_SQL = """
    SELECT cal_date FROM trade_cal
    WHERE exchange = ? AND is_open = 1 AND cal_date <= ?
    ORDER BY cal_date DESC LIMIT ?
"""


class LookaheadBlocked(RuntimeError):
    """请求的日期超出了允许可见的范围。

    **拒绝，而不是改写。** 遇到 `requested > visible_through` 直接抛异常，绝不静默把日期
    改成"能看的那一天"——静默改写会让一份前视的报告看起来完全正常。这个异常存在
    就是为了让那种错误变成一次崩溃，而不是一个数字。
    """


class Store:
    """一个 DuckDB 文件的只读句柄。

    用 `with` 管理生命周期；`close()` 幂等，可以放心放进 teardown。
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        read_only: bool = True,
        visible_through: date | None = None,
    ) -> None:
        self.path = Path(db_path)
        self.read_only = bool(read_only)
        #: 可见窗口的上界（含）。`None` = 不设限（实时轴）。
        #:
        #: 这是 `as_of` 时间纪律的**唯一落点**：设了它之后，读行情的路径（`snapshot`、
        #: 特征面板的 `ensure_visible`）都不允许越过这一天，把"后来才发生的事"挡住。
        self.visible_through = visible_through
        self._con = duckdb.connect(str(self.path), read_only=self.read_only)

    def ensure_visible(self, requested: date) -> date:
        """确认 `requested` 在可见范围内，否则抛 `LookaheadBlocked`。

        为什么是异常而不是 `min(requested, visible_through)`：后者会把
        "我要 2026-10-14 的数据"悄悄变成"给你 2026-09-16 的"，而调用方完全
        不知道。那样一份带前视的结论会以正常的样子产出，且**没有任何症状**。
        """
        if self.visible_through is None or requested <= self.visible_through:
            return requested
        raise LookaheadBlocked(
            f"请求 {requested} 超出了可见窗口（最多可见到 {self.visible_through}）。"
            "这是防前视闸门：验证轴只允许看到 as_of 之前的数据，"
            "越界的读必须失败，不能静默改写日期。"
        )

    def _clamp(self, requested: date) -> date:
        """把"取到哪天为止"压到可见范围内。**内部用，调用方看不到它。**"""
        if self.visible_through is None:
            return requested
        return min(requested, self.visible_through)

    # ------------------------------------------------------------- 生命周期
    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None  # type: ignore[assignment]

    def __enter__(self) -> Store:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            raise RuntimeError("Store 已关闭")
        return self._con

    # ----------------------------------------------------------------- 读取
    #
    # 每个读方法都先过 `_clamp`：可见窗口设了上界之后，**没有任何一条读路径
    # 能越过它**。纪律写在这里一次，而不是指望每个特征各自记得别去看未来。
    def snapshot(self, as_of: date) -> pd.DataFrame:
        """某交易日的截面（`daily` 左连 `stock_basic`）。

        `LEFT JOIN` 是有意的：缺 `stock_basic` 行的票必须能被下游看见并显式排除，
        而不是静默消失。
        """
        return self.connection.execute(_SNAPSHOT_SQL, [self._clamp(as_of)]).df()

    def latest_trade_date(self) -> date | None:
        """本地 `daily` 里最大的交易日（不校验行数）。

        **不受可见窗口影响**：这个方法回答的是"这个库最远有哪天"，属于对库本身的
        事实查询，不是"我能看到哪天"。用可见窗口去限制它会让新鲜度闸门误判——
        它会以为库比实际短，然后去补一段已经有的数据。
        """
        row = self.connection.execute("SELECT max(trade_date) FROM daily").fetchone()
        return row[0] if row and row[0] is not None else None

    def sessions(self, end: date, count: int, *, exchange: str = "SSE") -> tuple[date, ...]:
        """截至 `end`（含）的最近 `count` 个开市日，升序返回。

        交易日历是**权威**：目标交易日只能从这里取，不能从"本地行情最大日期"推，
        否则数据落后时幂等键会漂移。

        **刻意不受可见窗口限制**：日历是"哪天开市"的公共知识，不是行情数据——
        知道 10 月 20 日开市，不等于看到了 10 月 20 日的行情。
        """
        rows = self.connection.execute(_SESSIONS_SQL, [exchange, end, int(count)]).fetchall()
        return tuple(sorted(row[0] for row in rows))

    def has_table(self, name: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]
        ).fetchone()
        return row is not None
