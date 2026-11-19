"""辩论服务与运行目录测试（`rebound/debate/service.py`、`runs.py`）：假侧车端到端、断点续跑与冻结语义。"""

from __future__ import annotations

import json
import shutil
import threading
from datetime import date
from pathlib import Path

import pytest
from conftest import TEST_FIRST_FIT

from rebound import config as app_config
from rebound.debate import contracts, runs, service
from rebound.debate.sidecar import SidecarStopped
from rebound.model.artifacts import models_root
from rebound.model.score import fill_scores

DAY = date(2023, 3, 15)


def _copy_world(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    shutil.copytree(models_root(src), models_root(dst))
    return dst


@pytest.fixture(scope="module")
def scored_pristine(trained_world_pristine: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = _copy_world(trained_world_pristine, tmp_path_factory.mktemp("debate-svc") / "market.duckdb")
    fill_scores(path, first_fit=TEST_FIRST_FIT)
    return path


@pytest.fixture
def world(scored_pristine: Path, tmp_path: Path) -> Path:
    return _copy_world(scored_pristine, tmp_path / "market.duckdb")


class FakeSidecar:
    """可控假侧车：模拟全量成功、部分失败、中途停止以及 `--finalize`。"""

    def __init__(
        self,
        *,
        fail_codes: frozenset[str] = frozenset(),
        stop_after: int | None = None,
    ) -> None:
        self.fail_codes = fail_codes
        self.stop_after = stop_after
        self.debate_requests: list[contracts.RunRequest] = []
        self.finalize_calls: list[tuple[str, ...]] = []
        self.validated: list[contracts.RunRequest] = []

    def validate(self, request: contracts.RunRequest) -> dict:
        self.validated.append(request)
        return {"type": "valid", "protocol_version": 4, "candidates": len(request.candidates)}

    def debate(
        self,
        request: contracts.RunRequest,
        *,
        on_event=None,
        stop: threading.Event | None = None,
    ) -> contracts.DebatedResult:
        self.debate_requests.append(request)
        if on_event is not None:
            on_event(contracts.Event("run.started", {"run_id": request.run_id, "candidates": len(request.candidates)}))

        analysts: list[contracts.AnalystView] = []
        debate_entries: list[contracts.DebateEntry] = []
        verdicts: list[contracts.RiskVerdict] = []
        failed: list[contracts.FailedStock] = []
        calls = 0

        for idx, cand in enumerate(request.candidates):
            code = cand.ts_code
            if self.stop_after is not None and idx >= self.stop_after:
                raise SidecarStopped("模拟用户中断")
            if code in self.fail_codes:
                failed.append(contracts.FailedStock(code, "bear", "超时"))
                if on_event is not None:
                    on_event(contracts.Event("stock.failed", {"ts_code": code, "error": "bear 重试用尽（超时）"}))
                continue

            stock_analysts = [
                contracts.AnalystView(code, role, "bull", f"{role} 结论", ("风险",), ("features.r5",))
                for role in contracts.ANALYST_ROLES
            ]
            stock_debate = [
                contracts.DebateEntry(code, rnd, f"{rnd} 论点", ("bars[-1].close",))
                for rnd in contracts.DEBATE_ENTRY_ROUNDS
            ]
            score = float(95 - cand.model.rank)
            verdict = contracts.RiskVerdict(code, "看多", score, "核心判断", ("风险",), "开盘关注", ("features.r20",))
            analysts.extend(stock_analysts)
            debate_entries.extend(stock_debate)
            verdicts.append(verdict)
            calls += 6

            if on_event is not None:
                usage_now = contracts.Usage(calls=calls, failed_calls=len(failed), input_tokens=calls * 10, output_tokens=calls * 5)
                on_event(contracts.Event("usage.updated", {"usage": usage_now.payload()}))
                on_event(
                    contracts.Event(
                        "stock.completed",
                        {
                            "ts_code": code,
                            "verdict": verdict.verdict,
                            "score": verdict.score,
                            "usage": usage_now.payload(),
                            "stock": {
                                "verdict": verdict.payload(),
                                "analysts": [
                                    {
                                        "ts_code": a.ts_code, "role": a.role, "stance": a.stance,
                                        "conclusion": a.conclusion, "risks": list(a.risks),
                                        "citations": list(a.citations),
                                    }
                                    for a in stock_analysts
                                ],
                                "debate": [
                                    {
                                        "ts_code": d.ts_code, "round": d.round, "argument": d.argument,
                                        "citations": list(d.citations),
                                    }
                                    for d in stock_debate
                                ],
                            },
                        },
                    )
                )

        usage = contracts.Usage(calls=calls, failed_calls=len(failed), input_tokens=calls * 10, output_tokens=calls * 5)
        result = contracts.DebatedResult(
            run_id=request.run_id,
            candidate_hash=request.candidate_hash,
            input_hash=request.input_hash,
            candidate_codes=request.codes,
            analysts=tuple(analysts),
            debate=tuple(debate_entries),
            verdicts=tuple(verdicts),
            failed_stocks=tuple(failed),
            usage=usage,
        )
        contracts.verify_debated(result, request)
        return result

    def finalize(
        self,
        request: contracts.RunRequest,
        verdicts: tuple[contracts.RiskVerdict, ...],
        *,
        on_event=None,
        stop: threading.Event | None = None,
    ) -> contracts.FinalizedResult:
        self.finalize_calls.append(tuple(v.ts_code for v in verdicts))
        by_code = {c.ts_code: c for c in request.candidates}
        ordered = sorted(verdicts, key=lambda v: (-v.score, by_code[v.ts_code].model.rank))
        expected = contracts.expected_final_count(
            request.limits.final,
            [by_code[v.ts_code].industry for v in verdicts],
            request.limits.max_per_industry,
        )
        seen_ind: dict[str, int] = {}
        picks: list[contracts.FinalPick] = []
        for item in ordered:
            ind = by_code[item.ts_code].industry
            if seen_ind.get(ind, 0) >= request.limits.max_per_industry:
                continue
            seen_ind[ind] = seen_ind.get(ind, 0) + 1
            picks.append(contracts.FinalPick(item.ts_code, len(picks) + 1, "风控评分最高"))
            if len(picks) >= expected:
                break
        finalized = contracts.FinalizedResult(
            final_count=expected,
            picks=tuple(picks),
            reason="综合风控分与行业上限选出",
            usage=contracts.Usage(calls=1, failed_calls=0, input_tokens=50, output_tokens=20),
        )
        contracts.verify_final(finalized, verdicts, request)
        if on_event is not None:
            on_event(contracts.Event("usage.updated", {"usage": finalized.usage.payload()}))
        return finalized


def test_full_run_freezes_files_and_skips_rerun(world: Path, settings) -> None:
    fake = FakeSidecar()
    logs: list[str] = []
    first = service.run_debate(world, settings=settings, as_of=DAY, sidecar_client=fake, log=logs.append)
    assert first["cached"] is False
    rdir = Path(first["run_dir"])
    for filename in ("request.json", "usage.json", "events.ndjson", "result.json", "final.json"):
        assert (rdir / filename).is_file()
    assert len(list((rdir / "stocks").glob("*.json"))) == settings.debate.candidates
    assert len(fake.debate_requests) == 1
    assert len(fake.finalize_calls) == 1
    assert any("[辩论] (1/20)" in line for line in logs)
    assert first["final"]["usage"]["calls"] == 20 * 6 + 1

    # 已完成的运行再次调用直接返回冻结结果，不调模型、不改写文件
    mtime = (rdir / "final.json").stat().st_mtime_ns
    second = service.run_debate(world, settings=settings, as_of=DAY, sidecar_client=FakeSidecar())
    assert second["cached"] is True
    assert second["final"] == first["final"]
    assert (rdir / "final.json").stat().st_mtime_ns == mtime


def test_insufficient_debated_refuses_final_and_resume_sends_only_remaining(world: Path, settings) -> None:
    # 送辩 20 只，前 10 只成功，后 10 只失败（< min_debated=12）-> 不写 final.json，检查点保留
    from rebound.debate import snapshot

    snap = snapshot.build_request(world, settings=settings, as_of=DAY)
    fail_tail = frozenset(snap.request.codes[10:])
    fake1 = FakeSidecar(fail_codes=fail_tail)
    with pytest.raises(service.DebateError, match="少于 min_debated=12"):
        service.run_debate(world, settings=settings, as_of=DAY, sidecar_client=fake1)

    rdir = runs.run_dir(world, variant="nocap", cadence="M", as_of=DAY.isoformat(), input_hash=snap.request.input_hash)
    assert (rdir / "request.json").is_file()
    assert not (rdir / "final.json").exists() and not (rdir / "result.json").exists()
    assert len(list((rdir / "stocks").glob("*.json"))) == 10
    assert fake1.finalize_calls == []

    # 续跑：只发剩下的 10 只，但终选必须覆盖全部 20 只
    fake2 = FakeSidecar()
    resumed = service.run_debate(world, settings=settings, as_of=DAY, sidecar_client=fake2)
    assert resumed["cached"] is False
    assert fake2.debate_requests[0].codes == snap.request.codes[10:]
    assert fake2.debate_requests[0].limits.candidates == 20
    assert fake2.finalize_calls == [snap.request.codes]
    assert resumed["final"]["debated_count"] == 20
    assert resumed["final"]["usage"]["calls"] == 10 * 6 + 10 * 6 + 1


def test_stop_midway_keeps_checkpoints_without_writing_final(world: Path, settings) -> None:
    fake = FakeSidecar(stop_after=3)
    with pytest.raises(SidecarStopped):
        service.run_debate(world, settings=settings, as_of=DAY, sidecar_client=fake)

    _, entries = runs.list_runs(world, variant="nocap", cadence="M", as_of=DAY)
    assert len(entries) == 1
    assert entries[0]["completed"] is False and entries[0]["debated_count"] == 3


def test_changing_methodology_creates_new_run_dir_and_tracks_earliest_as_official(
    world: Path, tmp_path: Path, settings,
) -> None:
    first = service.run_debate(world, settings=settings, as_of=DAY, sidecar_client=FakeSidecar())

    # 拷贝配置与修改后的方法论到临时项目根
    proj = tmp_path / "proj"
    (proj / "config").mkdir(parents=True)
    (proj / "docs").mkdir(parents=True)
    base_cfg = (Path(app_config.ROOT) / "config" / "settings.yaml").read_text(encoding="utf-8")
    (proj / "config" / "settings.yaml").write_text(base_cfg, encoding="utf-8")
    meth_text = (Path(app_config.ROOT) / "docs" / "debate-methodology.md").read_text(encoding="utf-8")
    (proj / "docs" / "debate-methodology.md").write_text(
        meth_text.replace("反弹模型名单复核方法论", "反弹模型名单复核方法论（修订）"),
        encoding="utf-8",
    )
    edited_settings = app_config.load_settings(proj / "config" / "settings.yaml")
    second = service.run_debate(world, settings=edited_settings, as_of=DAY, sidecar_client=FakeSidecar())

    assert first["run_dir"] != second["run_dir"]
    day_str, entries = runs.list_runs(world, variant="nocap", cadence="M", as_of=DAY)
    assert day_str == DAY.isoformat() and len(entries) == 2
    assert sum(1 for e in entries if e["official"]) == 1
    assert entries[0]["run_dir"] == first["run_dir"] and entries[0]["official"] is True
