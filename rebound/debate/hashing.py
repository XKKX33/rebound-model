"""规范化 JSON 与内容寻址哈希（与 `debate_agent/src/contracts.ts` 对齐）。"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

__all__ = ["canonicalize", "fingerprint", "stable_json"]


def canonicalize(value: Any) -> Any:
    """将任意结构折成键序稳定、无 NaN、无 numpy 标量的 JSON 友好形态。"""
    if isinstance(value, dict):
        return {str(k): canonicalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonicalize(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return canonicalize(value.item())
    return value


def stable_json(value: Any) -> str:
    """紧凑、键序稳定、非 ASCII 不转义的 JSON。"""
    return json.dumps(
        canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()
