"""数据层的异常：三类失败必须能被调用方分开处理。

v1 的教训是把三类混成一句 `RuntimeError`：网络重试失败、上游返回的数据本身
是坏的、以及本地写入冲突，最后只能靠错误文本正则去猜。分开之后，重试策略
和"该不该把这一天的数据写进库"才能各自正确决定。
"""

from __future__ import annotations


class DataError(RuntimeError):
    """数据层一切失败的基类。"""


class NetworkError(DataError):
    """拿不到数据：重试耗尽、超时、上游返回错误。重试有意义。"""


class ValidationError(DataError):
    """拿到了数据，但数据本身不合法（缺字段、非法数值、重复业务键）。

    重试没有意义：同一个上游再问一次还是同一份坏数据。必须显式失败，
    绝不允许"校验不过就跳过这一列/这一行"。
    """
