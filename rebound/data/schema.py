"""整库的表结构：**唯一的 DDL 来源**（整份仓库里只有这里有 CREATE TABLE）。

只建"今天就有消费者"的表，每张表都要能说出谁在读它：

| 表 | 谁在读 |
|---|---|
| `trade_cal`   | `Store.sessions()` —— 交易日历是权威，信号日只能从这里取 |
| `stock_basic` | 股票池、选股 —— 名称与行业（每个行业最多 2 只靠它） |
| `daily`       | 特征、标签、回测 —— 日线行情 |
| `daily_limit` | 特征（涨停）与回测（一字板买不进、跌停卖不出） |
| `adj_factor`  | 特征与标签 —— 复权价 = 价 × 因子；不复权，除权缺口会被当成暴跌 |
| `namechange`  | 股票池 —— ST 按**当时的名称**判断（生效日不晚于当天的最后一个名字），没有记录才用现在的名称 |
| `model_score` | 选股、本金测算、纸面跟踪 —— 每天全股票池的打分和名次（按股票池版本、重训频率分开）；已有的日子不改写 |

表分两组：`MARKET_TABLES`（`rebound.data.sync` 写的行情快照）和 `MODEL_TABLES`（`rebound model score`
写的逐日打分）。缺口检测与覆盖率只看行情：打分不是"每个交易日都该从上游取到"的东西。

日期列一律 `DATE`，不是 `VARCHAR`。日期存成 `'20261014'` 会让每个查询点都要自己 cast，
早晚有一处忘了。这一层是日期类型的唯一定义处。
"""

from __future__ import annotations

# 表 -> 列（**顺序即 DDL 里的顺序**，写入时按此重排）。
# 与下面的 DDL 逐列交叉校验，两边不许漂移。
TABLE_FIELDS: dict[str, tuple[str, ...]] = {
    "trade_cal": ("exchange", "cal_date", "is_open"),
    "stock_basic": ("ts_code", "symbol", "name", "area", "industry", "market", "list_date"),
    "daily": (
        "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "pct_chg",
        "vol", "amount",
    ),
    "daily_limit": ("ts_code", "trade_date", "up_limit", "down_limit"),
    "adj_factor": ("ts_code", "trade_date", "adj_factor"),
    "namechange": ("ts_code", "name", "start_date", "end_date", "ann_date", "change_reason"),
    # --- 模型输出 ---
    "model_score": ("trade_date", "ts_code", "variant", "cadence", "score", "rank", "fit_date", "model_id",
                    "created_at"),
}

# 业务主键：幂等写入（INSERT OR REPLACE）靠它去重。
TABLE_PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "trade_cal": ("exchange", "cal_date"),
    "stock_basic": ("ts_code",),
    "daily": ("ts_code", "trade_date"),
    "daily_limit": ("ts_code", "trade_date"),
    "adj_factor": ("ts_code", "trade_date"),
    # 一个生效日一行；同一天生效、公告日不同的几行，同步时只留公告日最晚的。
    "namechange": ("ts_code", "start_date"),
    # 一天一只票、一版股票池、一种重训频率一行（业务键；建表不带主键，理由见 DDL）。
    "model_score": ("trade_date", "ts_code", "variant", "cadence"),
}

# 表 -> 它代表"哪一天"的列。缺口检测与覆盖率都按这列算。
TABLE_DATE_COLUMNS: dict[str, str] = {
    "trade_cal": "cal_date",
    "stock_basic": "list_date",
    "daily": "trade_date",
    "daily_limit": "trade_date",
    "adj_factor": "trade_date",
    "namechange": "start_date",
    "model_score": "trade_date",
}

# 行情快照：`rebound.data.sync` 写，缺口检测与新鲜度闸门只看这一组。
MARKET_TABLES: tuple[str, ...] = (
    "trade_cal", "stock_basic", "daily", "daily_limit", "adj_factor", "namechange",
)

# 模型输出：`rebound model score` 写。不是行情（不算缺口）。
MODEL_TABLES: tuple[str, ...] = ("model_score",)

# 建表顺序：先日历，再基础信息，然后是行情、模型输出。
TABLE_ORDER: tuple[str, ...] = MARKET_TABLES + MODEL_TABLES

_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS trade_cal (
        exchange  VARCHAR NOT NULL,
        cal_date  DATE    NOT NULL,
        is_open   INTEGER NOT NULL,
        PRIMARY KEY (exchange, cal_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stock_basic (
        ts_code   VARCHAR NOT NULL,
        symbol    VARCHAR,
        "name"    VARCHAR,
        area      VARCHAR,
        industry  VARCHAR,
        market    VARCHAR,
        list_date DATE,
        PRIMARY KEY (ts_code)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily (
        ts_code    VARCHAR NOT NULL,
        trade_date DATE    NOT NULL,
        open       DOUBLE,
        high       DOUBLE,
        low        DOUBLE,
        "close"    DOUBLE,
        pre_close  DOUBLE,
        pct_chg    DOUBLE,
        vol        DOUBLE,
        amount     DOUBLE,
        PRIMARY KEY (ts_code, trade_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_limit (
        ts_code    VARCHAR NOT NULL,
        trade_date DATE    NOT NULL,
        up_limit   DOUBLE,
        down_limit DOUBLE,
        PRIMARY KEY (ts_code, trade_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS adj_factor (
        ts_code    VARCHAR NOT NULL,
        trade_date DATE    NOT NULL,
        adj_factor DOUBLE,
        PRIMARY KEY (ts_code, trade_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS namechange (
        ts_code       VARCHAR NOT NULL,
        "name"        VARCHAR,
        start_date    DATE    NOT NULL,
        end_date      DATE,
        ann_date      DATE,
        change_reason VARCHAR,
        PRIMARY KEY (ts_code, start_date)
    )
    """,
    # 不建主键：两版 × 三种频率 × 2015 年以来每天 ≈ 2400 万行，主键索引要常驻内存约 1 GB。
    # 唯一性由唯一的写入口 `rebound.model.score._write` 保证：同一个事务里先查已有的
    # （日, 版本, 频率），有就整天不写（`WriteStore.append`，不是 INSERT OR REPLACE）。
    """
    CREATE TABLE IF NOT EXISTS model_score (
        trade_date DATE    NOT NULL,
        ts_code    VARCHAR NOT NULL,
        variant    VARCHAR NOT NULL,
        cadence    VARCHAR NOT NULL,
        score      DOUBLE  NOT NULL,
        "rank"     INTEGER NOT NULL,
        fit_date   DATE    NOT NULL,
        model_id   VARCHAR NOT NULL,
        created_at VARCHAR NOT NULL
    )
    """,
)


def ensure_schema(connection) -> None:  # noqa: ANN001 - duckdb.DuckDBPyConnection
    """建齐全部表。幂等：全部 `IF NOT EXISTS`，已存在不动。"""
    for statement in _DDL:
        connection.execute(statement)
