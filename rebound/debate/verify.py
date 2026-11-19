"""侧车 NDJSON 事件与终选结果校验（协议 v4）。"""

from __future__ import annotations

import json
import math
from typing import Any, Iterable, Mapping, Sequence

from rebound.debate.contracts import (
    ANALYST_ROLES,
    DEBATE_ENTRY_ROUNDS,
    STANCES,
    VERDICTS,
    AnalystView,
    ContractError,
    DebateEntry,
    DebatedResult,
    Event,
    FailedStock,
    FinalPick,
    FinalizedResult,
    RiskVerdict,
    RunRequest,
    Usage,
)

__all__ = [
    "expected_final_count",
    "parse_debated",
    "parse_event",
    "parse_finalized",
    "stock_checkpoint",
    "verify_debated",
    "verify_final",
]


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} 必须是对象，实际是 {type(value).__name__}")
    return value


def _sequence(value: Any, where: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContractError(f"{where} 必须是数组，实际是 {type(value).__name__}")
    return list(value)


def _text(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{where} 必须是字符串，实际是 {type(value).__name__}")
    if not allow_empty and not value.strip():
        raise ContractError(f"{where} 不能是空串")
    return value


def _int(value: Any, where: str, *, low: int, high: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{where} 必须是整数，实际是 {type(value).__name__}")
    if value < low or (high is not None and value > high):
        bound = f"[{low}, {high}]" if high is not None else f">= {low}"
        raise ContractError(f"{where} 必须在 {bound}，实际是 {value}")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where} 必须是数字，实际是 {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"{where} 必须是有限数字，实际是 {value!r}")
    return number


def _text_list(value: Any, where: str) -> tuple[str, ...]:
    return tuple(
        _text(item, f"{where}[{index}]", allow_empty=True)
        for index, item in enumerate(_sequence(value, where))
    )


def _checked_choice(value: Any, where: str, allowed: tuple[str, ...]) -> str:
    text = _text(value, where)
    if text not in allowed:
        raise ContractError(f"{where} 必须是 {list(allowed)} 之一，实际是 {text!r}")
    return text


def _bounded_score(value: Any, where: str) -> float:
    score = _number(value, where)
    if not 0.0 <= score <= 100.0:
        raise ContractError(f"{where} 必须在 0..100，实际是 {score}")
    return score


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    repeated: list[str] = []
    for value in values:
        if value in seen and value not in repeated:
            repeated.append(value)
        seen.add(value)
    return repeated


def _parse_analyst(raw: Any, where: str) -> AnalystView:
    item = _mapping(raw, where)
    return AnalystView(
        ts_code=_text(item.get("ts_code"), f"{where}.ts_code"),
        role=_checked_choice(item.get("role"), f"{where}.role", ANALYST_ROLES),
        stance=_checked_choice(item.get("stance"), f"{where}.stance", STANCES),
        conclusion=_text(item.get("conclusion"), f"{where}.conclusion"),
        risks=_text_list(item.get("risks"), f"{where}.risks"),
        citations=_text_list(item.get("citations"), f"{where}.citations"),
    )


def _parse_debate_entry(raw: Any, where: str) -> DebateEntry:
    item = _mapping(raw, where)
    return DebateEntry(
        ts_code=_text(item.get("ts_code"), f"{where}.ts_code"),
        round=_checked_choice(item.get("round"), f"{where}.round", DEBATE_ENTRY_ROUNDS),
        argument=_text(item.get("argument"), f"{where}.argument"),
        citations=_text_list(item.get("citations"), f"{where}.citations"),
    )


def _parse_verdict(raw: Any, where: str) -> RiskVerdict:
    item = _mapping(raw, where)
    return RiskVerdict(
        ts_code=_text(item.get("ts_code"), f"{where}.ts_code"),
        verdict=_checked_choice(item.get("verdict"), f"{where}.verdict", VERDICTS),
        score=_bounded_score(item.get("score"), f"{where}.score"),
        thesis=_text(item.get("thesis"), f"{where}.thesis"),
        risks=_text_list(item.get("risks"), f"{where}.risks"),
        action=_text(item.get("action"), f"{where}.action"),
        citations=_text_list(item.get("citations"), f"{where}.citations"),
    )


def _check_stock_bundle(
    code: str,
    analysts: Sequence[AnalystView],
    debate: Sequence[DebateEntry],
    where: str,
) -> None:
    if (
        any(item.ts_code != code for item in analysts)
        or len(analysts) != len(ANALYST_ROLES)
        or {a.role for a in analysts} != set(ANALYST_ROLES)
    ):
        raise ContractError(f"{where} 股票 {code} 缺少或重复了分析师观点")
    if (
        any(item.ts_code != code for item in debate)
        or len(debate) != len(DEBATE_ENTRY_ROUNDS)
        or {d.round for d in debate} != set(DEBATE_ENTRY_ROUNDS)
    ):
        raise ContractError(f"{where} 股票 {code} 缺少或重复了辩论记录")


def stock_checkpoint(payload: Mapping[str, Any]) -> dict[str, Any]:
    """将 `stock.completed` 里的单只股票产物校验成完整检查点。"""
    body = _mapping(payload, "stock.completed")
    verdict = _parse_verdict(body.get("verdict"), "stock.completed.verdict")
    analysts = tuple(
        _parse_analyst(raw, f"stock.completed.analysts[{index}]")
        for index, raw in enumerate(_sequence(body.get("analysts"), "stock.completed.analysts"))
    )
    debate = tuple(
        _parse_debate_entry(raw, f"stock.completed.debate[{index}]")
        for index, raw in enumerate(_sequence(body.get("debate"), "stock.completed.debate"))
    )
    _check_stock_bundle(verdict.ts_code, analysts, debate, "stock.completed")
    return {"verdict": verdict, "analysts": analysts, "debate": debate}


def parse_event(line: str) -> Event | None:
    """解析侧车 stdout 的一行 NDJSON；空行返回 None，非 JSON 立即报错。"""
    text = line.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContractError(f"侧车 stdout 出现了非 JSON 的一行：{text[:200]!r}") from exc
    body = _mapping(payload, "事件")
    kind = _text(body.get("type"), "事件.type")
    return Event(type=kind, payload=body)


def parse_debated(payload: Mapping[str, Any]) -> DebatedResult:
    body = _mapping(payload, "debated")
    analysts = tuple(
        _parse_analyst(item, f"analysts[{index}]")
        for index, item in enumerate(_sequence(body.get("analysts"), "debated.analysts"))
    )
    debate = tuple(
        _parse_debate_entry(item, f"debate[{index}]")
        for index, item in enumerate(_sequence(body.get("debate"), "debated.debate"))
    )
    verdicts = tuple(
        _parse_verdict(item, f"verdicts[{index}]")
        for index, item in enumerate(_sequence(body.get("verdicts"), "debated.verdicts"))
    )
    failed = tuple(
        FailedStock(
            ts_code=_text(item.get("ts_code"), f"failed_stocks[{index}].ts_code"),
            role=_text(item.get("role"), f"failed_stocks[{index}].role", allow_empty=True),
            error=_text(item.get("error"), f"failed_stocks[{index}].error", allow_empty=True),
        )
        for index, raw in enumerate(_sequence(body.get("failed_stocks"), "debated.failed_stocks"))
        for item in (_mapping(raw, f"failed_stocks[{index}]"),)
    )
    return DebatedResult(
        run_id=_text(body.get("run_id"), "debated.run_id"),
        candidate_hash=_text(body.get("candidate_hash"), "debated.candidate_hash"),
        input_hash=_text(body.get("input_hash"), "debated.input_hash"),
        candidate_codes=_text_list(body.get("candidate_codes"), "debated.candidate_codes"),
        analysts=analysts,
        debate=debate,
        verdicts=verdicts,
        failed_stocks=failed,
        usage=Usage.from_payload(_mapping(body.get("usage"), "debated.usage")),
    )


def parse_finalized(payload: Mapping[str, Any]) -> FinalizedResult:
    body = _mapping(payload, "finalized")
    picks = tuple(
        FinalPick(
            ts_code=_text(item.get("ts_code"), f"finalized.picks[{index}].ts_code"),
            rank=_int(item.get("rank"), f"finalized.picks[{index}].rank", low=1),
            reason=_text(item.get("reason"), f"finalized.picks[{index}].reason"),
        )
        for index, raw in enumerate(_sequence(body.get("picks"), "finalized.picks"))
        for item in (_mapping(raw, f"finalized.picks[{index}]"),)
    )
    return FinalizedResult(
        final_count=_int(body.get("final_count"), "finalized.final_count", low=1),
        picks=picks,
        reason=_text(body.get("reason"), "finalized.reason"),
        usage=Usage.from_payload(_mapping(body.get("usage"), "finalized.usage")),
    )


def expected_final_count(target_final: int, industries: Iterable[str], max_per_industry: int) -> int:
    """按行业可容纳上限收缩后的期望终选名额（第 3.6 节）。"""
    counts: dict[str, int] = {}
    total = 0
    for industry in industries:
        counts[industry] = counts.get(industry, 0) + 1
        total += 1
    feasible = sum(min(count, max_per_industry) for count in counts.values())
    return min(target_final, total, feasible)


def verify_debated(result: DebatedResult, request: RunRequest) -> None:
    """断言 `--debate` 返回的令牌、代码顺序、角色完整性与分拆划分无误。"""
    if result.run_id != request.run_id:
        raise ContractError(f"侧车回传的 run_id={result.run_id!r} 与下发的 {request.run_id!r} 不一致")
    if result.candidate_hash != request.candidate_hash:
        raise ContractError(
            f"侧车回传的 candidate_hash={result.candidate_hash!r} 与下发的 {request.candidate_hash!r} 不一致"
        )
    if result.input_hash != request.input_hash:
        raise ContractError(
            f"侧车回传的 input_hash={result.input_hash!r} 与下发的 {request.input_hash!r} 不一致"
        )
    if result.candidate_codes != request.codes:
        raise ContractError(
            f"侧车回传的 candidate_codes 与下发顺序不一致（下发 {len(request.codes)} 只，回传 {len(result.candidate_codes)} 只）"
        )

    pool = set(request.codes)
    verdict_codes = [v.ts_code for v in result.verdicts]
    failed_codes = [f.ts_code for f in result.failed_stocks]
    for label, codes in (
        ("verdicts", verdict_codes),
        ("failed_stocks", failed_codes),
        ("analysts", [a.ts_code for a in result.analysts]),
        ("debate", [d.ts_code for d in result.debate]),
    ):
        foreign = [code for code in codes if code not in pool]
        if foreign:
            raise ContractError(f"{label} 里出现送辩范围外的代码：{foreign}")

    for label, codes in (("verdicts", verdict_codes), ("failed_stocks", failed_codes)):
        dup = _duplicates(codes)
        if dup:
            raise ContractError(f"{label} 里出现重复代码：{dup}")

    debated_set, failed_set = set(verdict_codes), set(failed_codes)
    overlap = sorted(debated_set & failed_set)
    if overlap:
        raise ContractError(f"同时出现在 verdicts 和 failed_stocks 里：{overlap}")
    if debated_set | failed_set != pool:
        missing = sorted(pool - (debated_set | failed_set))
        raise ContractError(f"辩完与失败之和不等于下发候选，缺少：{missing}")

    analysts_by_code: dict[str, list[AnalystView]] = {code: [] for code in debated_set}
    for view in result.analysts:
        if view.ts_code not in debated_set:
            raise ContractError(f"analysts 含未辩完股票 {view.ts_code!r}")
        analysts_by_code[view.ts_code].append(view)

    debate_by_code: dict[str, list[DebateEntry]] = {code: [] for code in debated_set}
    for entry in result.debate:
        if entry.ts_code not in debated_set:
            raise ContractError(f"debate 含未辩完股票 {entry.ts_code!r}")
        debate_by_code[entry.ts_code].append(entry)

    for code in verdict_codes:
        _check_stock_bundle(code, analysts_by_code[code], debate_by_code[code], "debated")


def verify_final(
    finalized: FinalizedResult,
    verdicts: Sequence[RiskVerdict],
    request: RunRequest,
) -> None:
    """校验 `--finalize` 名单：名额按 3.6 重算、名次从 1 连续、代码均已辩完且不重复、行业不超限。"""
    debated_codes = {item.ts_code for item in verdicts}
    if len(debated_codes) != len(verdicts):
        raise ContractError("终选输入 verdicts 里有重复代码")
    industry_by_code = {item.ts_code: item.industry for item in request.candidates}
    unknown = [code for code in debated_codes if code not in industry_by_code]
    if unknown:
        raise ContractError(f"verdicts 里有请求范围外的代码：{unknown}")

    expected = expected_final_count(
        request.limits.final,
        [industry_by_code[item.ts_code] for item in verdicts],
        request.limits.max_per_industry,
    )
    if finalized.final_count != expected or len(finalized.picks) != expected:
        raise ContractError(
            f"终选名额应为 {expected}，收到 final_count={finalized.final_count}、picks={len(finalized.picks)}"
        )

    ranks = [pick.rank for pick in finalized.picks]
    if ranks != list(range(1, expected + 1)):
        raise ContractError(f"终选名次必须从 1 连续递增到 {expected}，实际是 {ranks}")

    seen_codes: set[str] = set()
    industry_counts: dict[str, int] = {}
    for pick in finalized.picks:
        if pick.ts_code not in debated_codes:
            raise ContractError(f"终选名单里有未辩完或候选池外的代码：{pick.ts_code!r}")
        if pick.ts_code in seen_codes:
            raise ContractError(f"终选名单里有重复代码：{pick.ts_code!r}")
        seen_codes.add(pick.ts_code)
        industry = industry_by_code[pick.ts_code]
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if industry_counts[industry] > request.limits.max_per_industry:
            raise ContractError(
                f"终选名单行业「{industry}」有 {industry_counts[industry]} 只，超过上限 {request.limits.max_per_industry}"
            )
