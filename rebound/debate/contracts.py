"""与辩论侧车（`debate_agent/`）之间的协议契约（协议 v4）。

两语言两端（Python 与 `debate_agent/src/contracts.ts`）必须同步维护同一套字段形状与常量；
`tests/test_debate_contracts.py` 会在测试期做跨语言交叉核对，防止静默分叉。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from rebound.debate.hashing import canonicalize, fingerprint, stable_json

__all__ = [
    "AGENT_ROLES",
    "ANALYST_ROLES",
    "DEBATE_ENTRY_ROUNDS",
    "DEBATE_ROUNDS",
    "FINAL_PICK_ROLE",
    "FLAG_CODES",
    "MAX_CALL_TIMEOUT_SECONDS",
    "MAX_CANDIDATES",
    "MAX_TOKENS_LIMIT",
    "PROTOCOL_VERSION",
    "STANCES",
    "VERDICTS",
    "AnalystView",
    "Candidate",
    "CandidateBar",
    "CandidateFlag",
    "CandidateModel",
    "ContractError",
    "DebateEntry",
    "DebatedResult",
    "Event",
    "FailedStock",
    "FinalPick",
    "FinalizedResult",
    "IndustryStats",
    "LlmSettings",
    "Methodology",
    "ModelPickItem",
    "PoolSummary",
    "RiskVerdict",
    "RunContext",
    "RunLimits",
    "RunRequest",
    "Usage",
    "build_roster",
    "canonicalize",
    "expected_final_count",
    "fingerprint",
    "input_fingerprint",
    "parse_debated",
    "parse_event",
    "parse_finalized",
    "stable_json",
    "stock_checkpoint",
    "verify_debated",
    "verify_final",
]

PROTOCOL_VERSION = 4

ANALYST_ROLES: tuple[str, ...] = ("model_analyst", "chart_analyst")
DEBATE_ROUNDS: tuple[str, ...] = ("bull", "bear", "bull_counter", "risk_chair")
FINAL_PICK_ROLE = "final_pick"
AGENT_ROLES: tuple[str, ...] = (*ANALYST_ROLES, *DEBATE_ROUNDS, FINAL_PICK_ROLE)
DEBATE_ENTRY_ROUNDS: tuple[str, ...] = ("bull", "bear", "bull_counter")

VERDICTS: tuple[str, ...] = ("看多", "看空")
STANCES: tuple[str, ...] = ("bull", "bear", "neutral")
FLAG_CODES: tuple[str, ...] = (
    "limit_up_today",
    "one_word_limit_up",
    "limit_up_streak",
    "limit_down_recent",
    "missing_features",
    "industry_crowded",
)

MAX_CANDIDATES = 50
MAX_TOKENS_LIMIT = 32768
MAX_CALL_TIMEOUT_SECONDS = 10_800

_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "run.started",
        "stock.started",
        "role.started",
        "role.delta",
        "role.completed",
        "role.failed",
        "stock.completed",
        "stock.failed",
        "run.partial",
        "run.completed",
        "usage.updated",
        "debated",
        "finalized",
        "valid",
        "health",
        "error",
    }
)


class ContractError(RuntimeError):
    """侧车请求、响应或检查点不符合协议契约，或回传的令牌对不上。"""


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    repeated: list[str] = []
    for value in values:
        if value in seen and value not in repeated:
            repeated.append(value)
        seen.add(value)
    return repeated


def _require_non_neg_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{where} 必须是非负整数，实际是 {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class Methodology:
    """方法论正文 + 7 个角色的职责简报，全部来自 `docs/debate-methodology.md`。"""

    text: str
    role_briefs: Mapping[str, str]

    def __post_init__(self) -> None:
        missing = [role for role in AGENT_ROLES if role not in self.role_briefs]
        unknown = [role for role in self.role_briefs if role not in AGENT_ROLES]
        if missing or unknown:
            raise ContractError(
                f"methodology.role_briefs 必须恰好覆盖 {len(AGENT_ROLES)} 个角色："
                f"缺少 {missing}，多出 {unknown}"
            )
        empty = [role for role in AGENT_ROLES if not self.role_briefs[role].strip()]
        if empty:
            raise ContractError(f"methodology.role_briefs 里这些角色的职责是空的：{empty}")

    def payload(self) -> dict[str, Any]:
        return {"text": self.text, "role_briefs": dict(self.role_briefs)}


@dataclass(frozen=True, slots=True)
class LlmSettings:
    """侧车构造大模型请求用的配置。**不含密钥**。"""

    id: str
    base_url: str
    reasoning_effort: str
    max_tokens: int
    temperature: float

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RunLimits:
    candidates: int
    final: int
    max_per_industry: int
    concurrency: int
    call_timeout_seconds: int

    def __post_init__(self) -> None:
        if not 0 < self.candidates <= MAX_CANDIDATES:
            raise ContractError(f"limits.candidates 必须在 1..{MAX_CANDIDATES}，实际是 {self.candidates}")
        if not 0 < self.final <= self.candidates:
            raise ContractError(
                f"limits.final 必须在 1..limits.candidates({self.candidates})，实际是 {self.final}"
            )
        if not 0 < self.max_per_industry <= MAX_CANDIDATES:
            raise ContractError(f"limits.max_per_industry 必须在 1..{MAX_CANDIDATES}，实际是 {self.max_per_industry}")
        if not 0 < self.concurrency <= MAX_CANDIDATES:
            raise ContractError(f"limits.concurrency 必须在 1..{MAX_CANDIDATES}，实际是 {self.concurrency}")
        if not 0 < self.call_timeout_seconds <= MAX_CALL_TIMEOUT_SECONDS:
            raise ContractError(
                f"limits.call_timeout_seconds 必须在 1..{MAX_CALL_TIMEOUT_SECONDS}，"
                f"实际是 {self.call_timeout_seconds}"
            )

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PoolSummary:
    median_pct_chg: float
    median_r5: float
    median_r20: float
    up_ratio: float

    def payload(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RunContext:
    as_of: str
    model_label: str
    variant: str
    cadence: str
    fit_date: str
    model_id: str
    scores_stored: bool
    pool_size: int
    candidates: int
    final: int
    max_per_industry: int
    pool: PoolSummary
    model_source: str | None = None
    feature_set: str | None = None

    def payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "as_of": self.as_of,
            "model_label": self.model_label,
            "variant": self.variant,
            "cadence": self.cadence,
            "fit_date": self.fit_date,
            "model_id": self.model_id,
            "scores_stored": self.scores_stored,
            "pool_size": self.pool_size,
            "candidates": self.candidates,
            "final": self.final,
            "max_per_industry": self.max_per_industry,
            "pool": self.pool.payload(),
        }
        if self.model_source is not None:
            out["model_source"] = self.model_source
        if self.feature_set is not None:
            out["feature_set"] = self.feature_set
        return out


@dataclass(frozen=True, slots=True)
class ModelPickItem:
    ts_code: str
    name: str
    industry: str
    rank: int
    debated: bool

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateModel:
    rank: int
    pct: float
    score: float
    in_model_picks: bool

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateBar:
    date: str
    open: float
    high: float
    low: float
    close: float
    pct_chg: float
    amount_yi: float
    limit: str | None
    vol5: float | None = None
    dif_ratio: float | None = None
    hist_ratio: float | None = None

    def payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "date": self.date,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "pct_chg": self.pct_chg,
            "amount_yi": self.amount_yi,
        }
        if self.vol5 is not None:
            out["vol5"] = self.vol5
        if self.dif_ratio is not None:
            out["dif_ratio"] = self.dif_ratio
        if self.hist_ratio is not None:
            out["hist_ratio"] = self.hist_ratio
        out["limit"] = self.limit
        return out


@dataclass(frozen=True, slots=True)
class IndustryStats:
    pool_count: int
    top_count: int
    rank_in_industry: int
    median_r5: float | None
    median_r20: float | None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateFlag:
    code: str
    text: str

    def __post_init__(self) -> None:
        if self.code not in FLAG_CODES:
            raise ContractError(f"未知的 flag code：{self.code!r}")

    def payload(self) -> dict[str, str]:
        return {"code": self.code, "text": self.text}


@dataclass(frozen=True, slots=True)
class Candidate:
    ts_code: str
    name: str
    industry: str
    model: CandidateModel
    features: Mapping[str, float]
    feature_pct: Mapping[str, float]
    bars: tuple[CandidateBar, ...]
    industry_stats: IndustryStats
    flags: tuple[CandidateFlag, ...]
    extra_features: Mapping[str, float] | None = None
    extra_feature_pct: Mapping[str, float] | None = None
    explain: Mapping[str, Any] | None = None
    peer_models: Mapping[str, Mapping[str, Any]] | None = None

    def payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ts_code": self.ts_code,
            "name": self.name,
            "industry": self.industry,
            "model": self.model.payload(),
            "features": dict(self.features),
            "feature_pct": dict(self.feature_pct),
            "bars": [bar.payload() for bar in self.bars],
            "industry_stats": self.industry_stats.payload(),
            "flags": [flag.payload() for flag in self.flags],
        }
        if self.extra_features is not None:
            out["extra_features"] = dict(self.extra_features)
        if self.extra_feature_pct is not None:
            out["extra_feature_pct"] = dict(self.extra_feature_pct)
        if self.explain is not None:
            out["explain"] = dict(self.explain)
        if self.peer_models is not None:
            out["peer_models"] = {k: dict(v) for k, v in self.peer_models.items()}
        return out


def build_roster(candidates: Sequence[Candidate]) -> tuple[dict[str, Any], ...]:
    """构造全部送辩候选的横向对照简表，供风控主席（含断点续跑场景）使用。"""
    return tuple(
        {
            "ts_code": c.ts_code,
            "name": c.name,
            "industry": c.industry,
            "rank": c.model.rank,
            "pct": c.model.pct,
            "r5": c.features.get("r5"),
            "r20": c.features.get("r20"),
            "from_low60": c.features.get("from_low60"),
            "dev_ma20": c.features.get("dev_ma20"),
            "vol5": c.features.get("vol5"),
            "flags": [flag.code for flag in c.flags],
        }
        for c in candidates
    )


@dataclass(frozen=True, slots=True)
class RunRequest:
    """一次辩论的完整输入。`candidate_hash` / `input_hash` 是不透明令牌。"""

    run_id: str
    as_of: str
    candidate_hash: str
    input_hash: str
    limits: RunLimits
    methodology: Methodology
    llm: LlmSettings
    context: RunContext
    model_picks: tuple[ModelPickItem, ...]
    candidates: tuple[Candidate, ...]
    roster: tuple[Mapping[str, Any], ...] | None = None

    def __post_init__(self) -> None:
        codes = [item.ts_code for item in self.candidates]
        if len(codes) != len(set(codes)):
            raise ContractError(f"candidates 里有重复的 ts_code：{_duplicates(codes)}")
        if not self.candidates:
            raise ContractError("candidates 不能为空")
        if len(self.candidates) > self.limits.candidates:
            raise ContractError(
                f"candidates 有 {len(self.candidates)} 只，超过 limits.candidates={self.limits.candidates}"
            )
        for item in self.candidates:
            if not item.bars:
                raise ContractError(f"{item.ts_code} 的 bars 不能为空")
            if item.bars[-1].date != self.as_of:
                raise ContractError(
                    f"{item.ts_code} 最后一根 bar 日期({item.bars[-1].date})必须等于 as_of({self.as_of})"
                )

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(item.ts_code for item in self.candidates)

    def payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "run_id": self.run_id,
            "as_of": self.as_of,
            "candidate_hash": self.candidate_hash,
            "input_hash": self.input_hash,
            "limits": self.limits.payload(),
            "methodology": self.methodology.payload(),
            "llm": self.llm.payload(),
            "context": self.context.payload(),
            "model_picks": [item.payload() for item in self.model_picks],
            "candidates": [item.payload() for item in self.candidates],
        }
        if self.roster is not None:
            out["roster"] = [dict(item) for item in self.roster]
        return out

    def to_line(self) -> str:
        """一行紧凑 JSON，供侧车 stdin 读取。"""
        return json.dumps(self.payload(), ensure_ascii=False, separators=(",", ":"))


def input_fingerprint(
    *,
    as_of: str,
    limits: RunLimits,
    methodology: Methodology,
    llm: LlmSettings,
    context: RunContext,
    model_picks: Sequence[ModelPickItem],
    candidates: Sequence[Candidate],
) -> str:
    """除 `run_id` 和两个哈希本身以外的整个请求的指纹（第 3.8 节）。"""
    return fingerprint(
        {
            "protocol_version": PROTOCOL_VERSION,
            "as_of": as_of,
            "limits": limits.payload(),
            "methodology": methodology.payload(),
            "llm": llm.payload(),
            "context": context.payload(),
            "model_picks": [item.payload() for item in model_picks],
            "candidates": [item.payload() for item in candidates],
        }
    )


@dataclass(frozen=True, slots=True)
class AnalystView:
    ts_code: str
    role: str
    stance: str
    conclusion: str
    risks: tuple[str, ...]
    citations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DebateEntry:
    ts_code: str
    round: str
    argument: str
    citations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RiskVerdict:
    ts_code: str
    verdict: str
    score: float
    thesis: str
    risks: tuple[str, ...]
    action: str
    citations: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "ts_code": self.ts_code,
            "verdict": self.verdict,
            "score": self.score,
            "thesis": self.thesis,
            "risks": list(self.risks),
            "action": self.action,
            "citations": list(self.citations),
        }


@dataclass(frozen=True, slots=True)
class FinalPick:
    ts_code: str
    rank: int
    reason: str


@dataclass(frozen=True, slots=True)
class FailedStock:
    ts_code: str
    role: str
    error: str


@dataclass(frozen=True, slots=True)
class Usage:
    calls: int
    failed_calls: int
    input_tokens: int
    output_tokens: int

    @classmethod
    def from_payload(cls, raw: Mapping[str, Any]) -> Usage:
        if not isinstance(raw, Mapping):
            raise ContractError(f"usage 必须是对象，实际是 {type(raw).__name__}")
        return cls(
            **{
                name: _require_non_neg_int(raw.get(name), f"usage.{name}")
                for name in ("calls", "failed_calls", "input_tokens", "output_tokens")
            }
        )

    def payload(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in ("calls", "failed_calls", "input_tokens", "output_tokens")}

    def plus(self, other: Usage) -> Usage:
        return Usage(**{name: getattr(self, name) + getattr(other, name) for name in self.payload()})


@dataclass(frozen=True, slots=True)
class DebatedResult:
    run_id: str
    candidate_hash: str
    input_hash: str
    candidate_codes: tuple[str, ...]
    analysts: tuple[AnalystView, ...]
    debate: tuple[DebateEntry, ...]
    verdicts: tuple[RiskVerdict, ...]
    failed_stocks: tuple[FailedStock, ...]
    usage: Usage


@dataclass(frozen=True, slots=True)
class FinalizedResult:
    final_count: int
    picks: tuple[FinalPick, ...]
    reason: str
    usage: Usage


@dataclass(frozen=True, slots=True)
class Event:
    type: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    @property
    def known(self) -> bool:
        return self.type in _EVENT_TYPES

    def describe(self) -> str:
        parts: list[str] = []
        for key in ("ts_code", "role", "attempt", "kind", "failed", "debated", "candidates"):
            if key in self.payload:
                parts.append(f"{key}={self.payload[key]}")
        if self.type == "role.failed" and "error" in self.payload:
            parts.append(str(self.payload["error"]))
        return f"{self.type} {' '.join(parts)}".strip()


# 在契约类定义完成后再重导出解析与校验函数，保持对外导入路径不变且避免循环导入。
from rebound.debate.verify import (  # noqa: E402
    expected_final_count,
    parse_debated,
    parse_event,
    parse_finalized,
    stock_checkpoint,
    verify_debated,
    verify_final,
)
