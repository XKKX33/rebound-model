"""数据层：Tushare -> DuckDB。

契约：

- **增量**：只拉缺失的交易日，已有的跳过（`already_present`）。
- **幂等**：同一天重复同步不产生重复行（表级主键 + `INSERT OR REPLACE`）。
- **报缺口**：同步结束时必须能说出"哪张表缺哪些天"，而不是静默跳过。
- **`None` 不写成 0**：上游真缺失就留空，不替下游猜。

还有一道**闸门**（`freshness.py`）：选股前必须确认本地数据已经是最新的，
否则会安静地拿几天前的数据选股。这条要求是代码，不是纪律。

模块分工：

    schema.py         整库表结构（唯一的 DDL 来源）
    errors.py         DataError / NetworkError（重试有意义）/ ValidationError（重试无意义）
    db.py             WriteStore：读写唯一入口、事务、upsert 列校验
    validate.py       写库前校验（一份 DataFrame 全过才进库）
    tushare_client.py 端点封装 + 限流退避
    sync.py           幂等增量同步 + 缺口审计
    history_import.py 从另一个库搬历史（重叠日逐行核对，对不上就拒绝）
    freshness.py      选股前的"数据是不是最新"闸门

**本包刻意不做 `from ... import *` 式的重导出**：调用方都按子模块导入
（`from rebound.data import freshness as fresh`、`from rebound.data.db import WriteStore`），
包级再抄一份导出只会形成第二个必须同步维护的名字清单。
"""

from __future__ import annotations

__all__: list[str] = []
