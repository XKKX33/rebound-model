"""AI 多角色辩论层（7 个角色，协议 v4）。

技术架构：Python 负责只读候选快照、协议指纹与检查点持久化；
Node 侧车（`debate_agent/`）负责调用大模型并发跑完 7 个角色，通过 stdin/stdout NDJSON 通信。
"""

from __future__ import annotations

__all__: list[str] = []
