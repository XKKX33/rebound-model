"""辩论流程编排：只读快照 → 检查点续跑 → `--debate` → `--finalize` → 冻结结果。"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping

from rebound import clock
from rebound import config as app_config
from rebound.data.errors import DataError
from rebound.debate import runs, snapshot
from rebound.debate.contracts import (
    AnalystView,
    DebateEntry,
    DebatedResult,
    Event,
    FailedStock,
    FinalizedResult,
    RiskVerdict,
    RunRequest,
    Usage,
    build_roster,
    expected_final_count,
    stock_checkpoint,
)
from rebound.debate.sidecar import Sidecar

__all__ = [
    "DebateError",
    "dry_run",
    "run_debate",
]


class DebateError(DataError):
    """辩论未能给出有效名单（辩完数量不足、侧车故障、校验失败），对应退出码 5。"""


def _format_score(score: float) -> str:
    return str(int(score)) if score == int(score) else f"{score:g}"


def dry_run(
    db_path: Path,
    *,
    settings: app_config.Settings,
    as_of: date | None = None,
    variant: str | None = None,
    cadence: str | None = None,
    model_source: str | None = None,
    sidecar_client: Any = None,
) -> tuple[snapshot.Snapshot, Mapping[str, Any]]:
    """只组装请求并调用侧车 `--validate`，不调模型、不写运行目录。"""
    snap = snapshot.build_request(
        db_path,
        settings=settings,
        as_of=as_of,
        variant=variant,
        cadence=cadence,
        model_source=model_source,
    )
    client = sidecar_client if sidecar_client is not None else Sidecar(settings)
    receipt = client.validate(snap.request)
    return snap, receipt


def _shortage_reason(
    *,
    requested_final: int,
    actual_final: int,
    debated_count: int,
    submitted_count: int,
) -> str | None:
    if actual_final >= requested_final:
        return None
    if debated_count < requested_final:
        return f"辩完 {debated_count} 只（送辩 {submitted_count} 只），少于目标 {requested_final} 只"
    return f"辩完候选行业集中，按同行业上限约束最多选出 {actual_final} 只"


def _build_final_payload(
    request: RunRequest,
    *,
    verdicts: tuple[RiskVerdict, ...],
    failed_stocks: tuple[FailedStock, ...],
    finalized: FinalizedResult,
    usage: Usage,
    completed_at: str,
) -> dict[str, Any]:
    cand_by_code = {c.ts_code: c for c in request.candidates}
    verdict_by_code = {v.ts_code: v for v in verdicts}
    ai_picks = [
        {
            "rank": pick.rank,
            "ts_code": pick.ts_code,
            "name": cand_by_code[pick.ts_code].name,
            "industry": cand_by_code[pick.ts_code].industry,
            "model_rank": cand_by_code[pick.ts_code].model.rank,
            "verdict": verdict_by_code[pick.ts_code].verdict,
            "score": verdict_by_code[pick.ts_code].score,
            "reason": pick.reason,
        }
        for pick in finalized.picks
    ]
    model_picks = [item.payload() for item in request.model_picks]
    ai_codes = {item["ts_code"] for item in ai_picks}
    model_codes = {item["ts_code"] for item in model_picks}
    overlap_codes = [item["ts_code"] for item in ai_picks if item["ts_code"] in model_codes]

    return {
        "as_of": request.as_of,
        "variant": request.context.variant,
        "cadence": request.context.cadence,
        "model_id": request.context.model_id,
        "run_id": request.run_id,
        "input_hash": request.input_hash,
        "candidate_hash": request.candidate_hash,
        "submitted_count": len(request.candidates),
        "debated_count": len(verdicts),
        "failed_count": len(failed_stocks),
        "failed_stocks": [asdict(item) for item in failed_stocks],
        "final_count": finalized.final_count,
        "requested_final": request.limits.final,
        "shortage_reason": _shortage_reason(
            requested_final=request.limits.final,
            actual_final=finalized.final_count,
            debated_count=len(verdicts),
            submitted_count=len(request.candidates),
        ),
        "picks": ai_picks,
        "reason": finalized.reason,
        "model_picks": model_picks,
        "overlap_count": len(overlap_codes),
        "overlap_codes": overlap_codes,
        "usage": usage.payload(),
        "completed_at": completed_at,
    }


def _build_result_payload(
    request: RunRequest,
    *,
    analysts: tuple[AnalystView, ...],
    debate: tuple[DebateEntry, ...],
    verdicts: tuple[RiskVerdict, ...],
    failed_stocks: tuple[FailedStock, ...],
    finalized: FinalizedResult,
    usage: Usage,
    completed_at: str,
) -> dict[str, Any]:
    return {
        "protocol_version": 4,
        "run_id": request.run_id,
        "as_of": request.as_of,
        "input_hash": request.input_hash,
        "candidate_hash": request.candidate_hash,
        "candidate_codes": list(request.codes),
        "analysts": [asdict(item) for item in analysts],
        "debate": [asdict(item) for item in debate],
        "verdicts": [item.payload() for item in verdicts],
        "final": {
            "final_count": finalized.final_count,
            "picks": [asdict(item) for item in finalized.picks],
            "reason": finalized.reason,
        },
        "failed_stocks": [asdict(item) for item in failed_stocks],
        "usage": usage.payload(),
        "completed_at": completed_at,
    }


def _merge_bundles(
    request: RunRequest,
    restored: Mapping[str, dict[str, Any]],
    debated: DebatedResult | None,
) -> tuple[
    tuple[AnalystView, ...],
    tuple[DebateEntry, ...],
    tuple[RiskVerdict, ...],
    tuple[FailedStock, ...],
]:
    bundles: dict[str, dict[str, Any]] = {}
    for code, cp in restored.items():
        bundles[code] = {
            "verdict": cp["verdict"],
            "analysts": list(cp["analysts"]),
            "debate": list(cp["debate"]),
        }
    if debated is not None:
        analysts_by: dict[str, list[AnalystView]] = {}
        debate_by: dict[str, list[DebateEntry]] = {}
        for item in debated.analysts:
            analysts_by.setdefault(item.ts_code, []).append(item)
        for item in debated.debate:
            debate_by.setdefault(item.ts_code, []).append(item)
        for item in debated.verdicts:
            bundles[item.ts_code] = {
                "verdict": item,
                "analysts": analysts_by[item.ts_code],
                "debate": debate_by[item.ts_code],
            }

    analysts_all: list[AnalystView] = []
    debate_all: list[DebateEntry] = []
    verdicts_all: list[RiskVerdict] = []
    for code in request.codes:
        bundle = bundles.get(code)
        if bundle is None:
            continue
        verdicts_all.append(bundle["verdict"])
        analysts_all.extend(bundle["analysts"])
        debate_all.extend(bundle["debate"])
    failed = debated.failed_stocks if debated is not None else ()
    return tuple(analysts_all), tuple(debate_all), tuple(verdicts_all), failed


def run_debate(
    db_path: Path,
    *,
    settings: app_config.Settings,
    as_of: date | None = None,
    variant: str | None = None,
    cadence: str | None = None,
    model_source: str | None = None,
    sidecar_client: Any = None,
    log: Callable[[str], None] | None = None,
    on_event: Callable[[Event], None] | None = None,
    stop: threading.Event | None = None,
) -> dict[str, Any]:
    """执行一次完整的 AI 辩论（命中已完成目录则直接返回冻结结果，中断后支持按票续跑）。"""
    say = log or (lambda _line: None)
    snap = snapshot.build_request(
        db_path,
        settings=settings,
        as_of=as_of,
        variant=variant,
        cadence=cadence,
        model_source=model_source,
    )
    req = snap.request
    rdir = runs.run_dir(
        db_path,
        variant=req.context.variant,
        cadence=req.context.cadence,
        as_of=req.as_of,
        input_hash=req.input_hash,
        model_source=req.context.model_source,
    )
    final_path = rdir / "final.json"
    if final_path.is_file():
        cached_final = json.loads(final_path.read_text(encoding="utf-8"))
        return {"cached": True, "run_dir": str(rdir), "final": cached_final}

    runs.freeze_request(rdir, req)
    restored = runs.load_checkpoints(rdir, req)
    base_usage = runs.load_usage(rdir)
    remaining = tuple(c for c in req.candidates if c.ts_code not in restored)
    total_count = len(req.candidates)
    done_count = len(restored)
    name_by_code = {c.ts_code: c.name for c in req.candidates}
    client = sidecar_client if sidecar_client is not None else Sidecar(settings)
    if on_event is not None:
        on_event(
            Event(
                "run.prepared",
                {
                    "run_id": req.run_id,
                    "as_of": req.as_of,
                    "variant": req.context.variant,
                    "cadence": req.context.cadence,
                    "hash_prefix": req.input_hash[:12],
                    "run_dir": str(rdir),
                    "total_count": total_count,
                    "restored_count": done_count,
                },
            )
        )

    def make_hook(usage_base: Usage) -> Callable[[Event], None]:
        nonlocal done_count
        last_usage = usage_base

        def on_sidecar_event(event: Event) -> None:
            nonlocal done_count, last_usage
            if event.type != "role.delta":
                runs.append_event(rdir, {"type": event.type, **event.payload})
            if "usage" in event.payload:
                last_usage = usage_base.plus(Usage.from_payload(event.payload["usage"]))
                runs.save_usage(rdir, last_usage)
            if event.type == "stock.completed":
                code = str(event.payload["ts_code"])
                stock = event.payload["stock"]
                runs.save_stock_checkpoint(rdir, code, stock)
                cp = stock_checkpoint(stock)
                done_count += 1
                say(
                    f"[辩论] ({done_count}/{total_count}) {code} {name_by_code.get(code, '')}："
                    f"{cp['verdict'].verdict} {_format_score(cp['verdict'].score)}"
                )
            elif event.type == "stock.failed":
                code = str(event.payload.get("ts_code", ""))
                err = str(event.payload.get("error", ""))
                say(f"[辩论] {code} {name_by_code.get(code, '')} 没辩完：{err}")
            if on_event is not None:
                on_event(event)

        return on_sidecar_event

    debated: DebatedResult | None = None
    if remaining:
        full_roster = build_roster(req.candidates) if len(remaining) < len(req.candidates) else None
        sub_req = RunRequest(
            run_id=req.run_id,
            as_of=req.as_of,
            candidate_hash=snapshot_hash(remaining),
            input_hash=req.input_hash,
            limits=req.limits,
            methodology=req.methodology,
            llm=req.llm,
            context=req.context,
            model_picks=req.model_picks,
            candidates=remaining,
            roster=full_roster,
        )
        debated = client.debate(sub_req, on_event=make_hook(base_usage), stop=stop)
        base_usage = base_usage.plus(debated.usage)
        runs.save_usage(rdir, base_usage)
        _ensure_checkpoints_saved(rdir, debated)

    analysts, debate_entries, verdicts, failed_stocks = _merge_bundles(req, restored, debated)
    if len(verdicts) < settings.debate.min_debated:
        raise DebateError(
            f"辩完 {len(verdicts)} 只，少于 min_debated={settings.debate.min_debated}，"
            f"不出名单（已保留检查点，重跑同一命令可续跑）"
        )

    finalized = client.finalize(req, verdicts, on_event=make_hook(base_usage), stop=stop)
    total_usage = base_usage.plus(finalized.usage)
    runs.save_usage(rdir, total_usage)

    completed_at = clock.beijing_now().isoformat(timespec="seconds")
    result_payload = _build_result_payload(
        req,
        analysts=analysts,
        debate=debate_entries,
        verdicts=verdicts,
        failed_stocks=failed_stocks,
        finalized=finalized,
        usage=total_usage,
        completed_at=completed_at,
    )
    final_payload = _build_final_payload(
        req,
        verdicts=verdicts,
        failed_stocks=failed_stocks,
        finalized=finalized,
        usage=total_usage,
        completed_at=completed_at,
    )
    runs.freeze_result(rdir, result_payload)
    runs.freeze_final(rdir, final_payload)
    if on_event is not None:
        on_event(
            Event(
                "run.frozen",
                {
                    "run_id": req.run_id,
                    "as_of": req.as_of,
                    "variant": req.context.variant,
                    "cadence": req.context.cadence,
                    "hash_prefix": req.input_hash[:12],
                    "run_dir": str(rdir),
                    "final": final_payload,
                },
            )
        )
    return {"cached": False, "run_dir": str(rdir), "final": final_payload}


def snapshot_hash(candidates: tuple[Any, ...]) -> str:
    from rebound.debate.contracts import fingerprint

    return fingerprint([item.payload() for item in candidates])


def _ensure_checkpoints_saved(rdir: Path, debated: DebatedResult) -> None:
    """兜底落盘：即使事件钩子未收到某只票的 `stock.completed`，也把已返回结论写进检查点。"""
    stocks_dir = rdir / "stocks"
    analysts_by: dict[str, list[dict[str, Any]]] = {}
    debate_by: dict[str, list[dict[str, Any]]] = {}
    for item in debated.analysts:
        analysts_by.setdefault(item.ts_code, []).append(asdict(item))
    for item in debated.debate:
        debate_by.setdefault(item.ts_code, []).append(asdict(item))
    for item in debated.verdicts:
        if (stocks_dir / f"{item.ts_code}.json").is_file():
            continue
        runs.save_stock_checkpoint(
            rdir,
            item.ts_code,
            {
                "verdict": item.payload(),
                "analysts": analysts_by.get(item.ts_code, []),
                "debate": debate_by.get(item.ts_code, []),
            },
        )
