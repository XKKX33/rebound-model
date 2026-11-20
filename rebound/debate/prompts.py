"""提示词单一规则装载、完整 System Prompt 装配与单股实际 User JSON 构造。

规则唯一来源：`rebound/debate/prompt_rules.json`（与 `debate_agent/src/prompts.ts` 共享）。
方法论正文与 7 个角色职责唯一来源：`docs/debate-methodology.md`（历史运行优先读冻结的 `request.json`）。
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Mapping

from rebound import config as app_config
from rebound.debate import chat
from rebound.debate.contracts import (
    AGENT_ROLES,
    ANALYST_ROLES,
    ContractError,
    Methodology,
    expected_final_count,
)
from rebound.debate.methodology import METHODOLOGY_RELATIVE, load_methodology

__all__ = [
    "PROMPT_RULES_PATH",
    "build_debate_prompt_sections",
    "build_debate_system_prompt",
    "build_role_user_payload",
    "estimate_tokens",
    "inspect_prompts",
    "load_prompt_rules",
]

PROMPT_RULES_PATH = Path(__file__).resolve().parent / "prompt_rules.json"
_ANALYST_SET = frozenset(ANALYST_ROLES)


@functools.lru_cache(maxsize=1)
def load_prompt_rules() -> dict[str, Any]:
    """读取共享的提示词规则配置（与 Node 侧车共源，缓存于内存）。"""
    raw = PROMPT_RULES_PATH.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ContractError(f"prompt_rules.json 顶层必须是对象：{PROMPT_RULES_PATH}")
    return data


def estimate_tokens(text: str) -> int:
    """粗略估算中英混排文本的 Token 数（仅用于 UI 参考展示）。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, round(cjk * 0.65 + other * 0.28))


def build_debate_prompt_sections(role: str, methodology: Methodology) -> list[dict[str, str]]:
    """按与 `debate_agent/src/prompts.ts` 完全一致的顺序拆分角色 system prompt 各段落。"""
    rules = load_prompt_rules()
    brief = methodology.role_briefs.get(role)
    if not brief:
        raise ContractError(f"方法论里缺少角色职责: {role}")
    output_schema = rules["output_schema"].get(role)
    if not output_schema:
        raise ContractError(f"未定义输出格式的角色: {role}")
    length_rule = rules["length_rules"].get(role)
    if not length_rule:
        raise ContractError(f"未定义长度约束的角色: {role}")
    constraints = rules["role_constraints"]
    common_rules: list[str] = list(rules["common_rules"])

    sections: list[dict[str, str]] = [
        {
            "key": "methodology",
            "title": "共享方法论正文",
            "kind": "shared",
            "body": methodology.text.strip(),
            "rendered": methodology.text.strip(),
        },
        {
            "key": "role_brief",
            "title": "你的职责（角色预置提示词）",
            "kind": "preset",
            "body": brief.strip(),
            "rendered": f"## 你的职责\n{brief.strip()}",
        },
    ]

    if role in _ANALYST_SET:
        c = constraints["analyst"]
        sections.append(
            {
                "key": "constraint",
                "title": c["title"],
                "kind": "rule",
                "body": c["body"],
                "rendered": f"## {c['title']}\n{c['body']}",
            }
        )
    elif role == "risk_chair":
        c = constraints["risk_chair"]
        sections.append(
            {
                "key": "constraint",
                "title": c["title"],
                "kind": "rule",
                "body": c["body"],
                "rendered": f"## {c['title']}\n{c['body']}",
            }
        )
    elif role == "final_pick":
        c = constraints["final_pick"]
        sections.append(
            {
                "key": "constraint",
                "title": c["title"],
                "kind": "rule",
                "body": c["body"],
                "rendered": f"## {c['title']}\n{c['body']}",
            }
        )

    sections.append(
        {
            "key": "output_schema",
            "title": "输出格式",
            "kind": "rule",
            "body": output_schema,
            "rendered": f"## 输出格式\n{output_schema}",
        }
    )
    sections.append(
        {
            "key": "length_rule",
            "title": "长度",
            "kind": "rule",
            "body": length_rule,
            "rendered": f"## 长度\n{length_rule}",
        }
    )
    bans_body = "\n".join(f"- {item}" for item in common_rules)
    sections.append(
        {
            "key": "common_rules",
            "title": "禁令",
            "kind": "rule",
            "body": bans_body,
            "rendered": f"## 禁令\n{bans_body}",
        }
    )
    return sections


def build_debate_system_prompt(role: str, methodology: Methodology) -> str:
    """组装与 `debate_agent/src/prompts.ts` 逐字一致的完整 system prompt。"""
    sections = build_debate_prompt_sections(role, methodology)
    return "\n\n".join(sec["rendered"] for sec in sections)


def _methodology_from_request(request: Mapping[str, Any] | None) -> Methodology | None:
    if not isinstance(request, Mapping):
        return None
    raw = request.get("methodology")
    if not isinstance(raw, Mapping):
        return None
    text = raw.get("text")
    briefs = raw.get("role_briefs")
    if not isinstance(text, str) or not isinstance(briefs, Mapping):
        return None
    try:
        return Methodology(
            text=text,
            role_briefs={str(k): str(v) for k, v in briefs.items()},
        )
    except ContractError:
        return None


def _build_roster_from_request(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_roster = request.get("roster")
    if isinstance(raw_roster, list) and len(raw_roster) > 0:
        return [dict(item) for item in raw_roster if isinstance(item, Mapping)]
    out: list[dict[str, Any]] = []
    for cand in request.get("candidates") or ():
        if not isinstance(cand, Mapping):
            continue
        features = cand.get("features") if isinstance(cand.get("features"), Mapping) else {}
        model = cand.get("model") if isinstance(cand.get("model"), Mapping) else {}
        flags = cand.get("flags") if isinstance(cand.get("flags"), list) else []
        out.append(
            {
                "ts_code": cand.get("ts_code"),
                "name": cand.get("name"),
                "industry": cand.get("industry"),
                "rank": model.get("rank"),
                "pct": model.get("pct"),
                "r5": features.get("r5"),
                "r20": features.get("r20"),
                "from_low60": features.get("from_low60"),
                "dev_ma20": features.get("dev_ma20"),
                "vol5": features.get("vol5"),
                "flags": [f.get("code") for f in flags if isinstance(f, Mapping) and f.get("code")],
            }
        )
    return out


def _clean_analyst_views(stock_data: Mapping[str, Any] | None) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(stock_data, Mapping):
        return [], list(ANALYST_ROLES)
    raw = stock_data.get("analysts")
    by_role: dict[str, dict[str, Any]] = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping) and isinstance(item.get("role"), str):
                by_role[item["role"]] = dict(item)
    elif isinstance(raw, Mapping):
        for role_id, item in raw.items():
            if isinstance(item, Mapping):
                by_role[str(role_id)] = dict(item)
    ordered: list[dict[str, Any]] = []
    missing: list[str] = []
    for role_id in ANALYST_ROLES:
        if role_id in by_role:
            ordered.append(by_role[role_id])
        else:
            missing.append(role_id)
    return ordered, missing


def _clean_debate_rounds(
    stock_data: Mapping[str, Any] | None, required_rounds: tuple[str, ...]
) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(stock_data, Mapping):
        return [], list(required_rounds)
    raw = stock_data.get("debate")
    by_round: dict[str, dict[str, Any]] = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping) and isinstance(item.get("round"), str):
                by_round[item["round"]] = dict(item)
    elif isinstance(raw, Mapping):
        for rname, item in raw.items():
            if isinstance(item, Mapping):
                by_round[str(rname)] = dict(item)
    ordered: list[dict[str, Any]] = []
    missing: list[str] = []
    for rname in required_rounds:
        if rname in by_round:
            ordered.append(by_round[rname])
        else:
            missing.append(rname)
    return ordered, missing


def build_role_user_payload(
    role: str,
    request: Mapping[str, Any],
    *,
    ts_code: str | None = None,
    stock_data: Mapping[str, Any] | None = None,
    all_stocks: Mapping[str, Mapping[str, Any]] | None = None,
    result_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """按 `debate_agent/src/workflow.ts` 完全相同的结构构造某角色实际收到的 User JSON。"""
    if role not in AGENT_ROLES:
        raise ValueError(f"未知辩论角色：{role}")

    context = request.get("context")
    candidates = [c for c in (request.get("candidates") or ()) if isinstance(c, Mapping)]

    if role == "final_pick":
        by_code = {str(c.get("ts_code")): c for c in candidates if c.get("ts_code")}
        verdicts: list[Mapping[str, Any]] = []
        if isinstance(result_data, Mapping) and isinstance(result_data.get("verdicts"), list):
            verdicts = [v for v in result_data["verdicts"] if isinstance(v, Mapping)]
        elif isinstance(all_stocks, Mapping):
            for st in all_stocks.values():
                if isinstance(st, Mapping) and isinstance(st.get("verdict"), Mapping):
                    verdicts.append(st["verdict"])

        rows: list[dict[str, Any]] = []
        for verdict in verdicts:
            code = str(verdict.get("ts_code") or "")
            cand = by_code.get(code)
            if cand is None:
                continue
            model = cand.get("model") if isinstance(cand.get("model"), Mapping) else {}
            flags = cand.get("flags") if isinstance(cand.get("flags"), list) else []
            rows.append(
                {
                    "ts_code": cand.get("ts_code"),
                    "name": cand.get("name"),
                    "industry": cand.get("industry"),
                    "model_rank": model.get("rank"),
                    "model_pct": model.get("pct"),
                    "in_model_picks": model.get("in_model_picks"),
                    "verdict": verdict.get("verdict"),
                    "score": verdict.get("score"),
                    "thesis": verdict.get("thesis"),
                    "action": verdict.get("action"),
                    "risks": verdict.get("risks", []),
                    "flags": [f.get("code") for f in flags if isinstance(f, Mapping) and f.get("code")],
                }
            )
        rows.sort(
            key=lambda item: (
                -(float(item["score"]) if isinstance(item.get("score"), (int, float)) else -1.0),
                int(item["model_rank"]) if isinstance(item.get("model_rank"), int) else 999999,
            )
        )
        limits = request.get("limits") if isinstance(request.get("limits"), Mapping) else {}
        max_per_ind = int(limits.get("max_per_industry", 2))
        target_final = int(limits.get("final", 6))
        expected = (
            expected_final_count(target_final, [str(r.get("industry") or "") for r in rows], max_per_ind)
            if rows
            else 0
        )
        payload: dict[str, Any] = {
            "context": context,
            "final_count": expected,
            "max_per_industry": max_per_ind,
            "model_picks": list(request.get("model_picks") or ()),
            "candidates": rows,
        }
        missing_deps = [] if rows else ["risk_chair（尚无已辩完股票的风控结论）"]
        wire = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return {
            "role": role,
            "ts_code": "-",
            "stock_name": "全批终选",
            "ready": len(missing_deps) == 0,
            "missing_deps": missing_deps,
            "top_keys": list(payload.keys()),
            "char_count": len(wire),
            "est_tokens": estimate_tokens(wire),
            "payload": payload,
        }

    # 个股 6 个角色
    clean_code = (ts_code or "").strip()
    cand = next((c for c in candidates if str(c.get("ts_code")) == clean_code), None)
    if cand is None and not clean_code and candidates:
        cand = candidates[0]
        clean_code = str(cand.get("ts_code") or "")
    if cand is None:
        raise ValueError(f"候选池中找不到股票：{ts_code!r}")

    effective_stock = stock_data
    if effective_stock is None and isinstance(all_stocks, Mapping):
        effective_stock = all_stocks.get(clean_code)

    payload = {"context": context, "candidate": cand}
    missing_deps = []

    if role in ("bull", "bear", "bull_counter", "risk_chair"):
        analysts, missing_analysts = _clean_analyst_views(effective_stock)
        payload["analyst_views"] = analysts
        missing_deps.extend(missing_analysts)

    if role == "bear":
        rounds, missing_rounds = _clean_debate_rounds(effective_stock, ("bull",))
        payload["debate"] = rounds
        missing_deps.extend(missing_rounds)
    elif role == "bull_counter":
        rounds, missing_rounds = _clean_debate_rounds(effective_stock, ("bull", "bear"))
        payload["debate"] = rounds
        missing_deps.extend(missing_rounds)
    elif role == "risk_chair":
        rounds, missing_rounds = _clean_debate_rounds(effective_stock, ("bull", "bear", "bull_counter"))
        payload["debate"] = rounds
        missing_deps.extend(missing_rounds)
        payload["roster"] = _build_roster_from_request(request)

    wire = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return {
        "role": role,
        "ts_code": clean_code,
        "stock_name": str(cand.get("name") or clean_code),
        "ready": len(missing_deps) == 0,
        "missing_deps": missing_deps,
        "top_keys": list(payload.keys()),
        "char_count": len(wire),
        "est_tokens": estimate_tokens(wire),
        "payload": payload,
    }


def inspect_prompts(
    *,
    settings: app_config.Settings,
    frozen_request: Mapping[str, Any] | None = None,
    source_preference: str = "auto",
) -> dict[str, Any]:
    """返回全部 7 个辩论 Agent 与对话 Agent 的预置提示词、完整 System Prompt 与版本差异状态。"""
    workspace_meth = load_methodology(root=settings.root)
    frozen_meth = _methodology_from_request(frozen_request)
    has_frozen = frozen_meth is not None

    methodology_changed = False
    changed_roles: list[str] = []
    if frozen_meth is not None:
        methodology_changed = frozen_meth.text.strip() != workspace_meth.text.strip()
        for role in AGENT_ROLES:
            if frozen_meth.role_briefs.get(role, "").strip() != workspace_meth.role_briefs.get(role, "").strip():
                changed_roles.append(role)
    differs_from_workspace = methodology_changed or len(changed_roles) > 0

    pref = (source_preference or "auto").strip().lower()
    if pref == "workspace" or frozen_meth is None:
        active_source = "workspace"
        active_meth = workspace_meth
    else:
        active_source = "frozen"
        active_meth = frozen_meth

    rules = load_prompt_rules()
    role_meta_map: Mapping[str, Any] = rules.get("role_meta", {})

    debate_roles: list[dict[str, Any]] = []
    for role in AGENT_ROLES:
        sections = build_debate_prompt_sections(role, active_meth)
        full_prompt = "\n\n".join(sec["rendered"] for sec in sections)
        meta = role_meta_map.get(role, {})
        constraint_sec = next((s for s in sections if s["key"] == "constraint"), None)
        debate_roles.append(
            {
                "id": role,
                "name": meta.get("name", role),
                "phase": meta.get("phase", ""),
                "phase_label": meta.get("phase_label", ""),
                "role_brief": active_meth.role_briefs[role].strip(),
                "constraint": (
                    {"title": constraint_sec["title"], "body": constraint_sec["body"]}
                    if constraint_sec is not None
                    else None
                ),
                "output_schema": rules["output_schema"][role],
                "length_rule": rules["length_rules"][role],
                "common_rules": list(rules["common_rules"]),
                "input_visibility": {
                    "summary": meta.get("input_summary", ""),
                    "visible_fields": list(meta.get("visible_fields", ())),
                    "hidden_note": meta.get("hidden_note", ""),
                },
                "sections": sections,
                "system_prompt": full_prompt,
                "char_count": len(full_prompt),
                "est_tokens": estimate_tokens(full_prompt),
                "changed_from_workspace": role in changed_roles,
            }
        )

    chat_roles: list[dict[str, Any]] = []
    for preset in chat.CHAT_ROLE_PRESETS:
        pid = preset["id"]
        identity, uses_brief = chat.build_chat_role_identity(pid, active_meth)
        base_prompt = f"{active_meth.text.strip()}\n\n{identity}"
        chat_roles.append(
            {
                "id": pid,
                "name": preset["name"],
                "desc": preset["desc"],
                "uses_methodology_role_brief": uses_brief,
                "preset_identity": identity,
                "system_prompt_base": base_prompt,
                "char_count": len(base_prompt),
                "est_tokens": estimate_tokens(base_prompt),
            }
        )

    chat_tools: list[dict[str, Any]] = []
    for tool in chat.TOOL_DEFINITIONS:
        fn = tool.get("function") if isinstance(tool, Mapping) else None
        if not isinstance(fn, Mapping):
            continue
        tname = str(fn.get("name") or "")
        chat_tools.append(
            {
                "name": tname,
                "label": chat.TOOL_LABELS.get(tname, tname),
                "description": str(fn.get("description") or ""),
                "parameters": fn.get("parameters") or {},
            }
        )

    meth_text = active_meth.text.strip()
    return {
        "source": active_source,
        "has_frozen": has_frozen,
        "differs_from_workspace": differs_from_workspace,
        "methodology_changed": methodology_changed,
        "changed_roles": changed_roles,
        "methodology_path": str(METHODOLOGY_RELATIVE).replace("\\", "/"),
        "rules_path": "rebound/debate/prompt_rules.json",
        "methodology": {
            "text": meth_text,
            "char_count": len(meth_text),
            "est_tokens": estimate_tokens(meth_text),
        },
        "debate_roles": debate_roles,
        "chat_roles": chat_roles,
        "chat_tools": chat_tools,
    }
