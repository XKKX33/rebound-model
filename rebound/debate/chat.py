"""AI 辩论复盘对话 Agent（档位 B：预置方法论 + 6 个只读工具调用 + 逐字流式思考 + 落盘持久化）。

设计纪律：
- 提示词底座与辩论系统共用唯一来源 `docs/debate-methodology.md`；
- 所有工具均为只读（读 `request.json`、`stocks/*.json`、`result.json`、`final.json` 与只读 DuckDB），绝不修改任何已冻结账本；
- 会话记录持久化到 `<run_dir>/chat_sessions.json`（或未落盘演练的 `<db>.debate/_chats/<as_of>.json`），按原子写落盘；
- 复用 `config/settings.yaml` 与 `.env` 中的大模型配置，支持 OpenAI 兼容 SSE `reasoning_content` + `content` + `tool_calls` 流式解析。
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import duckdb

from rebound import clock
from rebound import config as app_config
from rebound.debate import runs
from rebound.debate.methodology import load_methodology
from rebound.debate.sidecar import resolve_api_key

__all__ = [
    "CHAT_ROLE_PRESETS",
    "ChatStore",
    "ChatToolbox",
    "TOOL_DEFINITIONS",
    "run_chat_turn",
]

_MAX_TOOL_ROUNDS = 6
_MAX_HISTORY_MESSAGES = 24
_MAX_SESSIONS_PER_STORE = 30
_SAFE_CODE = re.compile(r"^[0-9A-Za-z._-]+$")
_SAFE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SAFE_HASH = re.compile(r"^[0-9a-f]{12}$")

CHAT_ROLE_PRESETS: tuple[dict[str, str], ...] = (
    {
        "id": "research_copilot",
        "name": "投研复盘助手",
        "desc": "全局复盘视角：解读终选逻辑、跨股对比、排查风险、生成次日开盘清单",
    },
    {
        "id": "risk_chair",
        "name": "风控主席 (risk_chair)",
        "desc": "以风控拍板视角回应看多/看空与 0~100 打分依据",
    },
    {
        "id": "final_pick",
        "name": "终选人 (final_pick)",
        "desc": "以全局终选视角解释行业上限约束与保留/换出取舍",
    },
    {
        "id": "model_analyst",
        "name": "模型解读员 (model_analyst)",
        "desc": "侧重 XGBoost 排名、特征分位与模型盲区拆解",
    },
    {
        "id": "chart_analyst",
        "name": "走势分析员 (chart_analyst)",
        "desc": "只看 20 日量价形态、均线支撑压力与 MACD 动能",
    },
    {
        "id": "bull",
        "name": "多方辩手 (bull)",
        "desc": "从做多角度深挖 T+1 买进、持有约 5 日的看多依据",
    },
    {
        "id": "bear",
        "name": "空方辩手 (bear)",
        "desc": "从风控回避角度继续挑刺结构缺陷与买入障碍",
    },
    {
        "id": "bull_counter",
        "name": "多方反驳手 (bull_counter)",
        "desc": "针对空方质疑做逐条反驳与风险权重评估",
    },
)

_ROLE_PRESET_IDS = {item["id"] for item in CHAT_ROLE_PRESETS}

TOOL_LABELS: dict[str, str] = {
    "get_stock_debate": "读取个股完整数据与 6 轮辩论",
    "compare_candidates": "横向对比多只候选股",
    "screen_candidates": "筛选/排序全部送辩候选",
    "query_stock_history": "查询 DuckDB 历史日线",
    "check_post_signal_returns": "核验信号日事后 T+1→T+N 收益",
    "list_debate_runs": "查看历史辩论运行记录",
}

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_debate",
            "description": (
                "读取当前辩论运行中任意一只候选股的完整档案，包括 28 个特征、全池分位、近 20 日前复权日线 bars、"
                "行业统计、flags 以及全部 6 轮结构化辩论原文（model_analyst、chart_analyst、bull、bear、bull_counter、risk_chair）。"
                "支持传股票代码（如 002245.SZ 或 002245）或中文简称（如 蔚蓝锂芯）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {"type": "string", "description": "股票代码或中文名称"},
                    "include_bars": {"type": "boolean", "description": "是否包含近 20 日逐日 K 线，默认 true"},
                    "include_features": {"type": "boolean", "description": "是否包含完整 28 特征与分位，默认 true"},
                },
                "required": ["ts_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_candidates",
            "description": (
                "横向对比 2~6 只送辩候选股的模型名次、风控结论与评分、核心量价特征、flags、"
                "以及各自多空辩论要点，适合回答对比类问题。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_codes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要对比的股票代码或简称列表（2~6 只）",
                    },
                    "include_debate_arguments": {
                        "type": "boolean",
                        "description": "是否附带各自 bull/bear/bull_counter 发言，默认 true",
                    },
                },
                "required": ["ts_codes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "screen_candidates",
            "description": (
                "对当前全部送辩候选（如 20 只）按风控方向、行业、flags 或任意特征字段"
                "（如 score、model_rank、pct_chg、r5、r20、r60、from_low60、dev_ma20、vol5、vol20、amp20、dif_ratio、hist_ratio）"
                "进行筛选与排序，适合全局统计与排名问题。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string", "description": "可选过滤：看多 或 看空"},
                    "industry": {"type": "string", "description": "可选行业名称过滤（模糊包含）"},
                    "flag": {"type": "string", "description": "可选风险旗标过滤，如 limit_down_recent、limit_up_today"},
                    "sort_by": {"type": "string", "description": "排序字段，默认 score"},
                    "descending": {"type": "boolean", "description": "是否降序，默认 true"},
                    "limit": {"type": "integer", "description": "返回条数上限，默认 20"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_stock_history",
            "description": (
                "只读查询 DuckDB 数据库的日线行情（daily 表），可查看任意主板股票在指定日期及之前最多 60 根历史日线。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {"type": "string", "description": "股票代码或简称"},
                    "as_of": {"type": "string", "description": "截止交易日 YYYY-MM-DD，缺省取当前运行信号日"},
                    "bars": {"type": "integer", "description": "返回最近多少根日线（5~60，默认 30）"},
                },
                "required": ["ts_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_post_signal_returns",
            "description": (
                "只读查询 DuckDB，核验某信号日之后是否已有后续交易日行情；若有，计算 T+1 开盘买、T+N 开盘卖的事后涨跌幅"
                "（同时检查 T+1 是否一字涨停买不进）；若该日已是库内最新交易日，则明确返回尚无后续行情。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_codes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要核验的股票代码列表；缺省自动核验当前 AI 终选前 6 与模型原前 6",
                    },
                    "as_of": {"type": "string", "description": "信号日 YYYY-MM-DD，缺省取当前运行信号日"},
                    "hold_days": {"type": "integer", "description": "持有交易日数（默认 5，即 T+1 开盘买、T+5 开盘卖）"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_debate_runs",
            "description": "列出当前数据库下已保存的历史 AI 辩论运行记录及终选名单摘要。",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "最多返回多少条，默认 10"},
                },
            },
        },
    },
]


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp = path.with_name(f"{path.stem}.tmp-{os.getpid()}-{threading.get_ident()}{path.suffix}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class ChatStore:
    """将对话会话按运行目录持久化到 `chat_sessions.json`（不触碰冻结账本）。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    @classmethod
    def for_run(
        cls,
        db_path: Path,
        *,
        variant: str,
        cadence: str,
        as_of: str,
        hash_prefix: str | None = None,
    ) -> "ChatStore":
        clean_date = as_of if _SAFE_DATE.fullmatch(as_of or "") else "general"
        clean_hash = hash_prefix if _SAFE_HASH.fullmatch(hash_prefix or "") else ""
        if clean_hash and clean_date != "general":
            run_folder = runs.debate_root(db_path) / f"{variant}-{cadence}" / clean_date / clean_hash
            if (run_folder / "request.json").is_file():
                return cls(run_folder / "chat_sessions.json")
        target = runs.debate_root(db_path) / "_chats" / f"{variant}-{cadence}-{clean_date}.json"
        return cls(target)

    def _load_unlocked(self) -> dict[str, Any]:
        raw = _read_json(self.path)
        if isinstance(raw, dict) and isinstance(raw.get("sessions"), list):
            return raw
        return {"version": 1, "sessions": []}

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            data = self._load_unlocked()
            sessions = list(data["sessions"])
        sessions.sort(key=lambda s: str(s.get("updated_at", "")), reverse=True)
        return sessions

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._load_unlocked()
            for item in data["sessions"]:
                if item.get("id") == session_id:
                    return json.loads(json.dumps(item, ensure_ascii=False))
        return None

    def create_session(
        self,
        *,
        role_preset: str = "research_copilot",
        title: str = "新对话",
    ) -> dict[str, Any]:
        if role_preset not in _ROLE_PRESET_IDS:
            role_preset = "research_copilot"
        now = clock.beijing_now().isoformat(timespec="seconds")
        session = {
            "id": f"chat-{uuid.uuid4().hex[:10]}",
            "title": (title or "新对话").strip()[:48],
            "role_preset": role_preset,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        with self._lock:
            data = self._load_unlocked()
            data["sessions"].insert(0, session)
            data["sessions"] = data["sessions"][:_MAX_SESSIONS_PER_STORE]
            _write_atomic(self.path, data)
        return session

    def save_session(self, session: dict[str, Any]) -> dict[str, Any]:
        session["updated_at"] = clock.beijing_now().isoformat(timespec="seconds")
        with self._lock:
            data = self._load_unlocked()
            updated = False
            for idx, existing in enumerate(data["sessions"]):
                if existing.get("id") == session.get("id"):
                    data["sessions"][idx] = session
                    updated = True
                    break
            if not updated:
                data["sessions"].insert(0, session)
            data["sessions"] = data["sessions"][:_MAX_SESSIONS_PER_STORE]
            _write_atomic(self.path, data)
        return session

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            data = self._load_unlocked()
            before = len(data["sessions"])
            data["sessions"] = [s for s in data["sessions"] if s.get("id") != session_id]
            if len(data["sessions"]) == before:
                return False
            _write_atomic(self.path, data)
            return True


class ChatToolbox:
    """为对话 Agent 提供对当前运行 100% 辩论信息与 DuckDB 行情的只读查询能力。"""

    def __init__(
        self,
        db_path: Path,
        *,
        variant: str,
        cadence: str,
        as_of: str,
        hash_prefix: str | None = None,
        run_detail: Mapping[str, Any] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.variant = variant
        self.cadence = cadence
        self.as_of = as_of
        self.hash_prefix = hash_prefix or ""
        self.run_detail = dict(run_detail) if run_detail is not None else {}

        req = self.run_detail.get("request") or {}
        self._candidates: list[dict[str, Any]] = list(req.get("candidates") or [])
        self._by_code: dict[str, dict[str, Any]] = {c["ts_code"]: c for c in self._candidates}
        self._stocks: dict[str, dict[str, Any]] = dict(self.run_detail.get("stocks") or {})

        # 若 stocks 检查点不全，补充合并 result.json 中的分析师/辩论/裁决
        result = self.run_detail.get("result") or {}
        if result:
            analysts_by: dict[str, list[dict[str, Any]]] = {}
            debate_by: dict[str, list[dict[str, Any]]] = {}
            for a in result.get("analysts") or []:
                analysts_by.setdefault(a["ts_code"], []).append(a)
            for d in result.get("debate") or []:
                debate_by.setdefault(d["ts_code"], []).append(d)
            for v in result.get("verdicts") or []:
                code = v["ts_code"]
                if code not in self._stocks:
                    self._stocks[code] = {
                        "verdict": v,
                        "analysts": analysts_by.get(code, []),
                        "debate": debate_by.get(code, []),
                    }

    def resolve_code(self, raw: str) -> str | None:
        token = (raw or "").strip().upper()
        if not token:
            return None
        if token in self._by_code:
            return token
        for code, cand in self._by_code.items():
            if code.split(".")[0] == token or str(cand.get("name", "")).upper() == token:
                return code
        for code, cand in self._by_code.items():
            if token in str(cand.get("name", "")).upper():
                return code
        return None

    def stock_bundle(
        self,
        ts_code: str,
        *,
        include_bars: bool = True,
        include_features: bool = True,
    ) -> dict[str, Any] | None:
        code = self.resolve_code(ts_code)
        if code is None:
            return None
        cand = self._by_code[code]
        bundle = self._stocks.get(code) or {}
        out: dict[str, Any] = {
            "ts_code": code,
            "name": cand.get("name"),
            "industry": cand.get("industry"),
            "model": cand.get("model"),
            "industry_stats": cand.get("industry_stats"),
            "flags": cand.get("flags"),
        }
        if include_features:
            out["features"] = cand.get("features")
            out["feature_pct"] = cand.get("feature_pct")
        if include_bars:
            out["bars"] = cand.get("bars")
        out["analysts"] = bundle.get("analysts") or []
        out["debate"] = bundle.get("debate") or []
        out["verdict"] = bundle.get("verdict")
        return out

    def global_context_snapshot(self, attached_codes: Sequence[str] = ()) -> dict[str, Any]:
        """构造常驻提示词里的紧凑全局上下文（终选 + 20 股风控简表 + 聚焦个股完整明细）。"""
        req = self.run_detail.get("request") or {}
        final = self.run_detail.get("final")
        roster_summary: list[dict[str, Any]] = []
        for cand in self._candidates:
            code = cand["ts_code"]
            v = (self._stocks.get(code) or {}).get("verdict") or {}
            f = cand.get("features") or {}
            roster_summary.append(
                {
                    "model_rank": cand["model"]["rank"],
                    "ts_code": code,
                    "name": cand["name"],
                    "industry": cand["industry"],
                    "in_model_picks": cand["model"]["in_model_picks"],
                    "verdict": v.get("verdict"),
                    "score": v.get("score"),
                    "thesis": v.get("thesis"),
                    "action": v.get("action"),
                    "pct_chg": f.get("pct_chg"),
                    "r5": f.get("r5"),
                    "r20": f.get("r20"),
                    "from_low60": f.get("from_low60"),
                    "dev_ma20": f.get("dev_ma20"),
                    "vol5": f.get("vol5"),
                    "flags": [fl["code"] for fl in cand.get("flags") or []],
                }
            )

        attached_details: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in attached_codes[:3]:
            code = self.resolve_code(raw)
            if code and code not in seen:
                seen.add(code)
                detail = self.stock_bundle(code, include_bars=True, include_features=True)
                if detail is not None:
                    attached_details.append(detail)

        return {
            "as_of": self.as_of,
            "variant": self.variant,
            "cadence": self.cadence,
            "hash_prefix": self.hash_prefix,
            "context": req.get("context"),
            "model_picks": req.get("model_picks") or [],
            "final": final,
            "all_candidates_summary": roster_summary,
            "attached_stocks_full": attached_details,
        }

    def execute(self, name: str, arguments: Mapping[str, Any]) -> tuple[bool, str, Any]:
        """执行单个只读工具，返回 `(ok, concise_summary, data)`。"""
        try:
            if name == "get_stock_debate":
                return self._tool_get_stock_debate(arguments)
            if name == "compare_candidates":
                return self._tool_compare_candidates(arguments)
            if name == "screen_candidates":
                return self._tool_screen_candidates(arguments)
            if name == "query_stock_history":
                return self._tool_query_stock_history(arguments)
            if name == "check_post_signal_returns":
                return self._tool_check_post_signal_returns(arguments)
            if name == "list_debate_runs":
                return self._tool_list_debate_runs(arguments)
            return False, f"未知工具：{name}", {"error": f"未知工具：{name}"}
        except Exception as exc:
            return False, f"工具执行异常：{exc}", {"error": str(exc)}

    def _tool_get_stock_debate(self, args: Mapping[str, Any]) -> tuple[bool, str, Any]:
        raw = str(args.get("ts_code") or "").strip()
        inc_bars = bool(args.get("include_bars", True))
        inc_feat = bool(args.get("include_features", True))
        detail = self.stock_bundle(raw, include_bars=inc_bars, include_features=inc_feat)
        if detail is None:
            return False, f"未在当前送辩候选中找到 {raw}", {
                "error": f"{raw} 不在当前送辩候选内",
                "available_codes": [f"{c['ts_code']}({c['name']})" for c in self._candidates],
            }
        v = detail.get("verdict") or {}
        v_txt = f"{v.get('verdict', '未裁决')} {v.get('score', '')}".strip()
        bar_n = len(detail.get("bars") or [])
        summary = f"已读取 {detail['ts_code']} {detail['name']}（{v_txt} · {bar_n} 根日线 · {len(detail['analysts']) + len(detail['debate']) + (1 if v else 0)} 轮记录）"
        return True, summary, detail

    def _tool_compare_candidates(self, args: Mapping[str, Any]) -> tuple[bool, str, Any]:
        raw_codes = args.get("ts_codes") or []
        if not isinstance(raw_codes, list) or not raw_codes:
            return False, "ts_codes 不能为空", {"error": "ts_codes 必须是非空数组"}
        inc_debate = bool(args.get("include_debate_arguments", True))
        rows: list[dict[str, Any]] = []
        missing: list[str] = []
        key_feats = (
            "pct_chg", "r5", "r10", "r20", "r60", "from_low60", "from_high60",
            "dev_ma5", "dev_ma20", "ma20_slope5", "dd20", "vol5", "vol20", "amp20", "dif_ratio", "hist_ratio",
        )
        for raw in raw_codes[:6]:
            code = self.resolve_code(str(raw))
            if not code:
                missing.append(str(raw))
                continue
            cand = self._by_code[code]
            bundle = self._stocks.get(code) or {}
            f = cand.get("features") or {}
            fp = cand.get("feature_pct") or {}
            row: dict[str, Any] = {
                "ts_code": code,
                "name": cand["name"],
                "industry": cand["industry"],
                "model": cand["model"],
                "flags": cand.get("flags") or [],
                "industry_stats": cand.get("industry_stats"),
                "key_features": {k: f[k] for k in key_feats if k in f},
                "key_feature_pct": {k: fp[k] for k in key_feats if k in fp},
                "verdict": bundle.get("verdict"),
                "analysts": [
                    {"role": a["role"], "stance": a["stance"], "conclusion": a["conclusion"], "risks": a.get("risks", [])}
                    for a in (bundle.get("analysts") or [])
                ],
            }
            if inc_debate:
                row["debate"] = bundle.get("debate") or []
            rows.append(row)
        summary = f"已横向对比 {len(rows)} 只股票：" + "、".join(f"{r['name']}({r['ts_code']})" for r in rows)
        return True, summary, {"compared": rows, "missing": missing}

    def _tool_screen_candidates(self, args: Mapping[str, Any]) -> tuple[bool, str, Any]:
        verdict_f = str(args.get("verdict") or "").strip()
        ind_f = str(args.get("industry") or "").strip()
        flag_f = str(args.get("flag") or "").strip()
        sort_by = str(args.get("sort_by") or "score").strip()
        desc = bool(args.get("descending", True))
        limit = max(1, min(50, int(args.get("limit") or 20)))

        rows: list[dict[str, Any]] = []
        for cand in self._candidates:
            code = cand["ts_code"]
            v = (self._stocks.get(code) or {}).get("verdict") or {}
            f = cand.get("features") or {}
            flags = [fl["code"] for fl in cand.get("flags") or []]
            if verdict_f and v.get("verdict") != verdict_f:
                continue
            if ind_f and ind_f not in str(cand.get("industry", "")):
                continue
            if flag_f and flag_f not in flags:
                continue
            rows.append(
                {
                    "ts_code": code,
                    "name": cand["name"],
                    "industry": cand["industry"],
                    "model_rank": cand["model"]["rank"],
                    "model_pct": cand["model"]["pct"],
                    "in_model_picks": cand["model"]["in_model_picks"],
                    "verdict": v.get("verdict"),
                    "score": v.get("score"),
                    "thesis": v.get("thesis"),
                    "flags": flags,
                    "pct_chg": f.get("pct_chg"),
                    "r5": f.get("r5"),
                    "r20": f.get("r20"),
                    "r60": f.get("r60"),
                    "from_low60": f.get("from_low60"),
                    "dev_ma20": f.get("dev_ma20"),
                    "vol5": f.get("vol5"),
                    "vol20": f.get("vol20"),
                    "amp20": f.get("amp20"),
                    "dif_ratio": f.get("dif_ratio"),
                    "hist_ratio": f.get("hist_ratio"),
                    "sort_value": (
                        v.get("score")
                        if sort_by == "score"
                        else cand["model"]["rank"]
                        if sort_by in ("rank", "model_rank")
                        else f.get(sort_by)
                    ),
                }
            )

        def sort_key(row: dict[str, Any]) -> tuple[int, float]:
            val = row.get("sort_value")
            if isinstance(val, (int, float)):
                return (0, -float(val) if desc else float(val))
            return (1, 0.0)

        rows.sort(key=sort_key)
        rows = rows[:limit]
        summary = f"筛出 {len(rows)} 只候选（按 {sort_by} {'降序' if desc else '升序'}）"
        return True, summary, {"sort_by": sort_by, "descending": desc, "count": len(rows), "rows": rows}

    def _resolve_db_code(self, con: duckdb.DuckDBPyConnection, raw: str) -> str | None:
        in_cand = self.resolve_code(raw)
        if in_cand:
            return in_cand
        token = (raw or "").strip().upper()
        if not token:
            return None
        row = con.execute(
            "SELECT ts_code FROM stock_basic WHERE UPPER(ts_code)=? OR UPPER(name)=? OR split_part(UPPER(ts_code),'.',1)=? LIMIT 1",
            [token, token, token],
        ).fetchone()
        return str(row[0]) if row else (token if _SAFE_CODE.fullmatch(token) else None)

    def _tool_query_stock_history(self, args: Mapping[str, Any]) -> tuple[bool, str, Any]:
        if not self.db_path.is_file():
            return False, "数据库文件不存在", {"error": "数据库文件不存在"}
        raw = str(args.get("ts_code") or "").strip()
        as_of = str(args.get("as_of") or self.as_of or "").strip()
        n_bars = max(5, min(60, int(args.get("bars") or 30)))

        con = duckdb.connect(str(self.db_path), read_only=True)
        try:
            code = self._resolve_db_code(con, raw)
            if not code:
                return False, f"库内找不到股票 {raw}", {"error": f"库内找不到股票 {raw}"}
            sql = (
                "SELECT CAST(trade_date AS VARCHAR) AS d, open, high, low, close, pct_chg, amount "
                "FROM daily WHERE ts_code = ? "
                + ("AND CAST(trade_date AS VARCHAR) <= ? " if as_of else "")
                + "ORDER BY trade_date DESC LIMIT ?"
            )
            params: list[Any] = [code, as_of, n_bars] if as_of else [code, n_bars]
            rows = con.execute(sql, params).fetchall()
            rows.reverse()
            bars = [
                {
                    "date": str(r[0])[:10],
                    "open": round(float(r[1]), 2),
                    "high": round(float(r[2]), 2),
                    "low": round(float(r[3]), 2),
                    "close": round(float(r[4]), 2),
                    "pct_chg": round(float(r[5]), 2) if r[5] is not None else None,
                    "amount_yi": round(float(r[6]) / 100_000.0, 2) if r[6] is not None else None,
                }
                for r in rows
            ]
            summary = f"已从 DuckDB 读取 {code} 最近 {len(bars)} 根日线" + (f"（截至 {as_of}）" if as_of else "")
            return True, summary, {"ts_code": code, "as_of": as_of, "bars": bars}
        finally:
            con.close()

    def _tool_check_post_signal_returns(self, args: Mapping[str, Any]) -> tuple[bool, str, Any]:
        if not self.db_path.is_file():
            return False, "数据库文件不存在", {"error": "数据库文件不存在"}
        as_of = str(args.get("as_of") or self.as_of or "").strip()
        hold_days = max(1, min(20, int(args.get("hold_days") or 5)))
        raw_codes = args.get("ts_codes")
        codes: list[str] = []
        if isinstance(raw_codes, list) and raw_codes:
            for item in raw_codes[:12]:
                c = self.resolve_code(str(item)) or str(item).strip().upper()
                if c and c not in codes:
                    codes.append(c)
        else:
            final = self.run_detail.get("final") or {}
            for p in final.get("picks") or []:
                if p["ts_code"] not in codes:
                    codes.append(p["ts_code"])
            for m in final.get("model_picks") or (self.run_detail.get("request") or {}).get("model_picks") or []:
                if m["ts_code"] not in codes:
                    codes.append(m["ts_code"])

        con = duckdb.connect(str(self.db_path), read_only=True)
        try:
            max_day_row = con.execute("SELECT MAX(CAST(trade_date AS VARCHAR)) FROM daily").fetchone()
            latest_db_day = str(max_day_row[0])[:10] if max_day_row and max_day_row[0] else ""
            future_days = [
                str(r[0])[:10]
                for r in con.execute(
                    "SELECT DISTINCT CAST(trade_date AS VARCHAR) AS d FROM daily "
                    "WHERE CAST(trade_date AS VARCHAR) > ? ORDER BY d ASC LIMIT ?",
                    [as_of, hold_days + 2],
                ).fetchall()
            ]
            if not future_days:
                summary = f"库内最新交易日为 {latest_db_day}，{as_of} 之后尚无未来行情"
                return True, summary, {
                    "as_of": as_of,
                    "latest_db_day": latest_db_day,
                    "available_future_days": 0,
                    "message": summary,
                    "results": [],
                }

            buy_day = future_days[0]
            sell_day = future_days[min(hold_days - 1, len(future_days) - 1)]
            out_rows: list[dict[str, Any]] = []
            for code in codes:
                b = con.execute(
                    "SELECT open, high, low, close, pct_chg FROM daily WHERE ts_code=? AND CAST(trade_date AS VARCHAR)=?",
                    [code, buy_day],
                ).fetchone()
                s = con.execute(
                    "SELECT open, close FROM daily WHERE ts_code=? AND CAST(trade_date AS VARCHAR)=?",
                    [code, sell_day],
                ).fetchone()
                if not b or not s or not b[0]:
                    out_rows.append({"ts_code": code, "status": "停牌或缺数据"})
                    continue
                buy_open = float(b[0])
                sell_open = float(s[0])
                one_word = abs(float(b[0]) - float(b[1])) < 1e-6 and abs(float(b[1]) - float(b[2])) < 1e-6 and abs(float(b[2]) - float(b[3])) < 1e-6
                t1_pct = round(float(b[4]), 2) if b[4] is not None else None
                ret_pct = round((sell_open / buy_open - 1.0) * 100.0, 2)
                out_rows.append(
                    {
                        "ts_code": code,
                        "name": (self._by_code.get(code) or {}).get("name", ""),
                        "buy_day": buy_day,
                        "buy_open": round(buy_open, 2),
                        "t1_pct_chg": t1_pct,
                        "sell_day": sell_day,
                        "sell_open": round(sell_open, 2),
                        "open_to_open_pct": ret_pct,
                        "t1_one_word_flat": one_word,
                    }
                )
            summary = f"已核验 {len(out_rows)} 只股票在 {buy_day}→{sell_day} 的开盘到开盘收益"
            return True, summary, {
                "as_of": as_of,
                "latest_db_day": latest_db_day,
                "buy_day": buy_day,
                "sell_day": sell_day,
                "actual_hold_days": min(hold_days, len(future_days)),
                "results": out_rows,
            }
        finally:
            con.close()

    def _tool_list_debate_runs(self, args: Mapping[str, Any]) -> tuple[bool, str, Any]:
        limit = max(1, min(30, int(args.get("limit") or 10)))
        series_dir = runs.debate_root(self.db_path) / f"{self.variant}-{self.cadence}"
        if not series_dir.is_dir():
            return True, "当前数据库下暂无历史辩论运行", {"runs": []}
        days = sorted((d.name for d in series_dir.iterdir() if d.is_dir()), reverse=True)
        items: list[dict[str, Any]] = []
        for day_str in days:
            if len(items) >= limit:
                break
            try:
                from datetime import date as _date

                _, entries = runs.list_runs(
                    self.db_path, variant=self.variant, cadence=self.cadence, as_of=_date.fromisoformat(day_str)
                )
            except Exception:
                continue
            for e in entries:
                final = e.get("final") or {}
                items.append(
                    {
                        "as_of": day_str,
                        "hash_prefix": e["hash_prefix"],
                        "completed": e["completed"],
                        "official": e["official"],
                        "picks": [f"{p['ts_code']} {p['name']}({p['verdict']}{int(p['score'])})" for p in final.get("picks") or []],
                    }
                )
                if len(items) >= limit:
                    break
        summary = f"找到 {len(items)} 条历史辩论记录"
        return True, summary, {"runs": items}


def build_chat_system_prompt(
    settings: app_config.Settings,
    *,
    role_preset: str,
    context_snapshot: Mapping[str, Any],
) -> str:
    """组装对话 Agent 的完整 system prompt（唯一方法论底座 + 角色身份 + 实时常驻上下文）。"""
    meth = load_methodology(root=settings.root)
    parts: list[str] = [meth.text.strip()]

    if role_preset == "research_copilot" or role_preset not in meth.role_briefs:
        parts.append(
            "## 你的对话身份：反弹模型投研复盘助手\n"
            "你正在 Web UI 对话面板中协助研究员深度复核本次 AI 多角色辩论的选股结果。\n"
            "1. 回答要直接、有条理、结论先行，善用简短清单和对照表，不要套话。\n"
            "2. 你默认已看到当前运行的全景摘要（终选前 6、原模型前 6、全部 20 只风控简表）和当前聚焦个股的完整明细。\n"
            "3. 当用户提到任意其他候选股、需要逐字核对某只票的 20 日 K 线或 6 轮辩论原文、需要跨股筛选排序或核验历史事后收益时，**主动调用只读工具获取精确数据后再回答**，不要凭猜测作答。\n"
            "4. 严格遵守方法论纪律：只使用上下文和工具返回的数字与事实，绝对不要编造新闻、公告、业绩、资金流、盘口或输入外信息；引用指标时带上字段路径（例如 `features.r20`、`bars[-1].close`、`flags.limit_down_recent`），方便用户在界面悬停核对原值。"
        )
    else:
        role_brief = meth.role_briefs[role_preset].strip()
        parts.append(
            f"## 你的对话身份：{role_preset}\n"
            "你正在 Web UI 对话面板中以该角色的视角接受研究员的追问与拷问。\n"
            f"原角色职责如下：\n{role_brief}\n\n"
            "注意：在对话模式下请用清晰自然的中文 Markdown 回答（不必强套单行 JSON 外壳），如需核查任意候选股完整 K 线、特征或前序辩论原文，请主动调用只读工具查证后再作答，绝不编造外部信息。"
        )

    snapshot_json = json.dumps(context_snapshot, ensure_ascii=False, separators=(",", ":"))
    parts.append(f"## 当前页面常驻运行上下文（JSON）\n{snapshot_json}")
    return "\n\n".join(parts)


@dataclass
class StreamChunk:
    thinking: str = ""
    text: str = ""
    tool_calls_delta: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


StreamFn = Callable[
    [app_config.Settings, str, list[dict[str, Any]], list[dict[str, Any]], threading.Event],
    Iterator[StreamChunk],
]


def openai_stream_chat(
    settings: app_config.Settings,
    api_key: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    stop_event: threading.Event,
) -> Iterator[StreamChunk]:
    """使用标准库发起 OpenAI 兼容流式 Chat Completions（支持 reasoning_content + tool_calls）。"""
    llm = settings.debate.llm
    base = llm.base_url.rstrip("/")
    url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"

    payload: dict[str, Any] = {
        "model": llm.model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": llm.temperature,
        "max_tokens": llm.max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    effort = (llm.reasoning_effort or "").strip().lower()
    if effort and effort not in ("off", "none"):
        payload["reasoning_effort"] = effort

    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    timeout_s = max(30, min(settings.debate.call_timeout_seconds, 600))
    try:
        resp = urllib.request.urlopen(req, timeout=timeout_s)
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        raise RuntimeError(f"上游模型返回 HTTP {exc.code}：{body or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"连接大模型网关失败：{exc.reason}") from exc

    with resp:
        for raw_line in resp:
            if stop_event.is_set():
                return
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                return
            try:
                obj = json.loads(data_str)
            except ValueError:
                continue

            usage_obj = obj.get("usage")
            usage_out: dict[str, int] | None = None
            if isinstance(usage_obj, dict):
                usage_out = {
                    "input_tokens": int(usage_obj.get("prompt_tokens") or 0),
                    "output_tokens": int(usage_obj.get("completion_tokens") or 0),
                }

            choices = obj.get("choices") or []
            if not choices:
                if usage_out:
                    yield StreamChunk(usage=usage_out)
                continue

            choice = choices[0]
            delta = choice.get("delta") or {}
            thinking = delta.get("reasoning_content") or delta.get("reasoning") or ""
            text = delta.get("content") or ""
            tc_delta = delta.get("tool_calls")
            finish_reason = choice.get("finish_reason")
            yield StreamChunk(
                thinking=str(thinking) if thinking else "",
                text=str(text) if text else "",
                tool_calls_delta=tc_delta if isinstance(tc_delta, list) else None,
                finish_reason=str(finish_reason) if finish_reason else None,
                usage=usage_out,
            )


def _merge_tool_calls(acc: dict[int, dict[str, Any]], deltas: list[dict[str, Any]]) -> None:
    for item in deltas:
        idx = int(item.get("index", 0))
        slot = acc.setdefault(
            idx,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if item.get("id"):
            slot["id"] = str(item["id"])
        fn = item.get("function") or {}
        if fn.get("name"):
            slot["function"]["name"] += str(fn["name"])
        if fn.get("arguments"):
            slot["function"]["arguments"] += str(fn["arguments"])


def run_chat_turn(
    *,
    settings: app_config.Settings,
    store: ChatStore,
    toolbox: ChatToolbox,
    session_id: str | None,
    user_message: str,
    role_preset: str = "research_copilot",
    attached_codes: Sequence[str] = (),
    stop_event: threading.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """执行一轮带只读工具循环的流式对话，逐帧产出 `(event_name, payload)` 并持久化落盘。"""
    text_in = (user_message or "").strip()
    if not text_in:
        yield ("chat.error", {"error": "消息内容不能为空"})
        return

    preset = role_preset if role_preset in _ROLE_PRESET_IDS else "research_copilot"
    stop = stop_event or threading.Event()

    session = store.get_session(session_id) if session_id else None
    if session is None:
        title = text_in.replace("\n", " ")[:28]
        session = store.create_session(role_preset=preset, title=title)
    else:
        session["role_preset"] = preset
        if not session.get("messages") and session.get("title") in ("", "新对话"):
            session["title"] = text_in.replace("\n", " ")[:28]

    now = clock.beijing_now().isoformat(timespec="seconds")
    user_entry = {
        "id": f"msg-{uuid.uuid4().hex[:10]}",
        "role": "user",
        "content": text_in,
        "attached_codes": list(attached_codes),
        "role_preset": preset,
        "created_at": now,
    }
    session["messages"].append(user_entry)
    store.save_session(session)

    asst_msg_id = f"msg-{uuid.uuid4().hex[:10]}"
    yield (
        "chat.start",
        {
            "session_id": session["id"],
            "session_title": session["title"],
            "message_id": asst_msg_id,
            "role_preset": preset,
        },
    )

    # 准备 API Key（若注入测试 stream_fn 则免读密钥）
    api_key = ""
    if stream_fn is None:
        try:
            api_key = resolve_api_key(settings)
        except Exception as exc:
            yield ("chat.error", {"error": str(exc)})
            return

    ctx_snapshot = toolbox.global_context_snapshot(attached_codes)
    system_prompt = build_chat_system_prompt(settings, role_preset=preset, context_snapshot=ctx_snapshot)

    wire_messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for msg in session["messages"][-_MAX_HISTORY_MESSAGES:]:
        if msg["role"] in ("user", "assistant") and msg.get("content"):
            wire_messages.append({"role": msg["role"], "content": msg["content"]})

    full_thinking_parts: list[str] = []
    full_text_parts: list[str] = []
    recorded_tools: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0

    caller = stream_fn or openai_stream_chat

    try:
        for _round in range(_MAX_TOOL_ROUNDS):
            if stop.is_set():
                break
            round_thinking = ""
            round_text = ""
            tool_acc: dict[int, dict[str, Any]] = {}
            finish_reason: str | None = None

            for chunk in caller(settings, api_key, wire_messages, TOOL_DEFINITIONS, stop):
                if stop.is_set():
                    break
                if chunk.thinking:
                    round_thinking += chunk.thinking
                    full_thinking_parts.append(chunk.thinking)
                    yield ("chat.thinking", {"session_id": session["id"], "message_id": asst_msg_id, "delta": chunk.thinking})
                if chunk.text:
                    round_text += chunk.text
                    full_text_parts.append(chunk.text)
                    yield ("chat.delta", {"session_id": session["id"], "message_id": asst_msg_id, "delta": chunk.text})
                if chunk.tool_calls_delta:
                    _merge_tool_calls(tool_acc, chunk.tool_calls_delta)
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                if chunk.usage:
                    total_input_tokens += chunk.usage.get("input_tokens", 0)
                    total_output_tokens += chunk.usage.get("output_tokens", 0)

            if stop.is_set():
                break

            ordered_calls = [tool_acc[i] for i in sorted(tool_acc.keys()) if tool_acc[i]["function"]["name"]]
            if not ordered_calls:
                break

            # 追加 assistant tool_calls 帧到本轮临时消息列表
            asst_wire: dict[str, Any] = {
                "role": "assistant",
                "content": round_text or "",
                "tool_calls": [
                    {
                        "id": tc["id"] or f"call_{idx}",
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"] or "{}",
                        },
                    }
                    for idx, tc in enumerate(ordered_calls)
                ],
            }
            wire_messages.append(asst_wire)

            # 逐个执行只读工具并向前端流式推送进度步骤
            for idx, tc in enumerate(ordered_calls):
                call_id = tc["id"] or f"call_{idx}"
                fn_name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"] or "{}"
                try:
                    parsed_args = json.loads(raw_args)
                    if not isinstance(parsed_args, dict):
                        parsed_args = {}
                except ValueError:
                    parsed_args = {}

                label = TOOL_LABELS.get(fn_name, fn_name)
                yield (
                    "chat.tool_start",
                    {
                        "session_id": session["id"],
                        "message_id": asst_msg_id,
                        "call_id": call_id,
                        "name": fn_name,
                        "label": label,
                        "arguments": parsed_args,
                    },
                )
                t0 = time.perf_counter()
                ok, summary, data = toolbox.execute(fn_name, parsed_args)
                elapsed_ms = int((time.perf_counter() - t0) * 1000)

                record = {
                    "call_id": call_id,
                    "name": fn_name,
                    "label": label,
                    "arguments": parsed_args,
                    "ok": ok,
                    "elapsed_ms": elapsed_ms,
                    "summary": summary,
                }
                recorded_tools.append(record)
                yield (
                    "chat.tool_done",
                    {
                        "session_id": session["id"],
                        "message_id": asst_msg_id,
                        **record,
                    },
                )
                wire_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": fn_name,
                        "content": json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str),
                    }
                )

            if finish_reason == "stop":
                break
    except Exception as exc:
        yield ("chat.error", {"session_id": session["id"], "message_id": asst_msg_id, "error": str(exc)})
        return

    final_text = "".join(full_text_parts).strip()
    final_thinking = "".join(full_thinking_parts).strip()
    assistant_entry = {
        "id": asst_msg_id,
        "role": "assistant",
        "content": final_text or ("（已中止生成）" if stop.is_set() else "（模型未返回正文）"),
        "thinking": final_thinking,
        "tool_calls": recorded_tools,
        "role_preset": preset,
        "stopped": bool(stop.is_set()),
        "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        },
        "created_at": clock.beijing_now().isoformat(timespec="seconds"),
    }
    session["messages"].append(assistant_entry)
    store.save_session(session)

    event_name = "chat.stopped" if stop.is_set() else "chat.done"
    yield (
        event_name,
        {
            "session_id": session["id"],
            "session_title": session["title"],
            "message": assistant_entry,
            "usage": assistant_entry["usage"],
        },
    )
