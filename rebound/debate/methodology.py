"""从 `docs/debate-methodology.md` 加载辩论方法论正文与 7 个角色的职责简报。

提示词只在这里维护唯一来源；缺失、重复或未知角色一律在加载期报错。
"""

from __future__ import annotations

import re
from pathlib import Path

from rebound import config as app_config
from rebound.debate.contracts import AGENT_ROLES, ContractError, Methodology

__all__ = ["METHODOLOGY_RELATIVE", "methodology_path", "load_methodology"]

METHODOLOGY_RELATIVE = Path("docs") / "debate-methodology.md"

_GAP = r"[ \t]*\r?\n(?:[ \t]*\r?\n)*"
_ROLE_BLOCK = re.compile(
    r"^###[ \t]+(?P<role>\S+)" + _GAP + r"```text\r?\n(?P<body>.*?)\r?\n```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
_BODY_BLOCK = re.compile(
    r"^##[ \t]+方法论正文" + _GAP + r"```text\r?\n(?P<body>.*?)\r?\n```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


def methodology_path(root: Path | None = None) -> Path:
    default = app_config.ROOT / METHODOLOGY_RELATIVE
    if root is None:
        return default
    candidate = Path(root) / METHODOLOGY_RELATIVE
    return candidate if candidate.exists() or not default.exists() else default


def load_methodology(path: Path | None = None, *, root: Path | None = None) -> Methodology:
    """读取并校验方法论文档与 7 个角色简报。"""
    target = Path(path) if path is not None else methodology_path(root)
    if not target.exists():
        raise ContractError(f"方法论文件不存在：{target}")
    text = target.read_text(encoding="utf-8")

    body = _BODY_BLOCK.search(text)
    if body is None:
        raise ContractError(f"{target} 里找不到 `## 方法论正文` 下的 ```text 代码块")

    briefs: dict[str, str] = {}
    for match in _ROLE_BLOCK.finditer(text):
        role = match.group("role")
        if role not in AGENT_ROLES:
            continue
        if role in briefs:
            raise ContractError(f"{target} 里角色 {role!r} 的职责写了两次")
        briefs[role] = match.group("body").strip()

    missing = [role for role in AGENT_ROLES if role not in briefs]
    if missing:
        raise ContractError(f"{target} 缺少这些角色的 `### 角色名` + ```text 职责块：{missing}")

    return Methodology(text=body.group("body").strip(), role_briefs=briefs)
