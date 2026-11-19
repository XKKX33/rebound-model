"""AI 辩论本地 Web UI 服务测试（`rebound/debate/web.py`）：静态资源、历史回放、SSE 流式 delta、互斥与安全。"""

from __future__ import annotations

import json
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from conftest import TEST_FIRST_FIT

from rebound import config as app_config
from rebound.debate import contracts, service, web
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
    path = _copy_world(trained_world_pristine, tmp_path_factory.mktemp("debate-web") / "market.duckdb")
    fill_scores(path, first_fit=TEST_FIRST_FIT)
    return path


@pytest.fixture
def world(scored_pristine: Path, tmp_path: Path) -> Path:
    return _copy_world(scored_pristine, tmp_path / "data" / "market.duckdb")


class StreamingFakeSidecar:
    """模拟带思考与正文逐字流的假侧车，用于验证 SSE `role.delta`、续跑 `roster` 与不写盘语义。"""

    def __init__(self, *, delay: float = 0.0, fail_first_code: str | None = None) -> None:
        self.delay = delay
        self.fail_first_code = fail_first_code
        self.seen_requests: list[contracts.RunRequest] = []

    def debate(
        self,
        request: contracts.RunRequest,
        *,
        on_event=None,
        stop: threading.Event | None = None,
    ) -> contracts.DebatedResult:
        self.seen_requests.append(request)
        if on_event is not None:
            on_event(contracts.Event("run.started", {"run_id": request.run_id, "candidates": len(request.candidates)}))

        analysts: list[contracts.AnalystView] = []
        debate_entries: list[contracts.DebateEntry] = []
        verdicts: list[contracts.RiskVerdict] = []
        failed: list[contracts.FailedStock] = []
        calls = 0

        for idx, cand in enumerate(request.candidates):
            code = cand.ts_code
            if stop is not None and stop.is_set():
                from rebound.debate.sidecar import SidecarStopped

                raise SidecarStopped("测试停止")
            if self.fail_first_code and code == self.fail_first_code:
                self.fail_first_code = None
                failed.append(contracts.FailedStock(code, "risk_chair", "模拟超时"))
                if on_event is not None:
                    on_event(contracts.Event("stock.failed", {"ts_code": code, "error": "模拟超时"}))
                continue

            if on_event is not None:
                on_event(contracts.Event("stock.started", {"ts_code": code, "index": idx + 1, "total": len(request.candidates)}))
                on_event(contracts.Event("role.started", {"ts_code": code, "role": "model_analyst", "attempt": 1}))
                on_event(contracts.Event("role.delta", {"ts_code": code, "role": "model_analyst", "attempt": 1, "channel": "thinking", "delta": "正在核对量价分位"}))
                on_event(contracts.Event("role.delta", {"ts_code": code, "role": "model_analyst", "attempt": 1, "channel": "text", "delta": '{"stance":"bull"}'}))
            if self.delay > 0:
                time.sleep(self.delay)

            stock_analysts = [
                contracts.AnalystView(code, role, "bull", f"{role} 结论", ("风险",), ("features.r5",))
                for role in contracts.ANALYST_ROLES
            ]
            stock_debate = [
                contracts.DebateEntry(code, rnd, f"{rnd} 论点", ("bars[-1].close",))
                for rnd in contracts.DEBATE_ENTRY_ROUNDS
            ]
            score = float(90 - cand.model.rank)
            verdict = contracts.RiskVerdict(code, "看多", score, "核心判断", ("风险",), "开盘观察", ("features.r20",))
            analysts.extend(stock_analysts)
            debate_entries.extend(stock_debate)
            verdicts.append(verdict)
            calls += 6

            if on_event is not None:
                on_event(contracts.Event("role.completed", {"ts_code": code, "role": "model_analyst", "attempt": 1, "data": {"role": "model_analyst", "stance": "bull", "conclusion": "model_analyst 结论", "risks": ["风险"], "citations": ["features.r5"]}}))
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
                                        "ts_code": a.ts_code,
                                        "role": a.role,
                                        "stance": a.stance,
                                        "conclusion": a.conclusion,
                                        "risks": list(a.risks),
                                        "citations": list(a.citations),
                                    }
                                    for a in stock_analysts
                                ],
                                "debate": [
                                    {
                                        "ts_code": d.ts_code,
                                        "round": d.round,
                                        "argument": d.argument,
                                        "citations": list(d.citations),
                                    }
                                    for d in stock_debate
                                ],
                            },
                        },
                    )
                )

        return contracts.DebatedResult(
            run_id=request.run_id,
            candidate_hash=request.candidate_hash,
            input_hash=request.input_hash,
            candidate_codes=request.codes,
            analysts=tuple(analysts),
            debate=tuple(debate_entries),
            verdicts=tuple(verdicts),
            failed_stocks=tuple(failed),
            usage=contracts.Usage(calls=calls, failed_calls=len(failed), input_tokens=calls * 10, output_tokens=calls * 5),
        )

    def finalize(self, request: contracts.RunRequest, verdicts, *, on_event=None, stop=None) -> contracts.FinalizedResult:
        if on_event is not None:
            on_event(contracts.Event("role.started", {"ts_code": "-", "role": "final_pick", "attempt": 1}))
            on_event(contracts.Event("role.delta", {"ts_code": "-", "role": "final_pick", "attempt": 1, "channel": "thinking", "delta": "汇总全部风控打分"}))
        by_code = {c.ts_code: c for c in request.candidates}
        ordered = sorted(verdicts, key=lambda v: (-v.score, by_code[v.ts_code].model.rank))
        picks: list[contracts.FinalPick] = []
        ind_counts: dict[str, int] = {}
        for v in ordered:
            ind = by_code[v.ts_code].industry
            if ind_counts.get(ind, 0) >= request.limits.max_per_industry:
                continue
            ind_counts[ind] = ind_counts.get(ind, 0) + 1
            picks.append(contracts.FinalPick(v.ts_code, len(picks) + 1, "按评分入选"))
            if len(picks) >= request.limits.final:
                break
        return contracts.FinalizedResult(
            final_count=len(picks),
            picks=tuple(picks),
            reason="测试终选理由",
            usage=contracts.Usage(calls=1, failed_calls=0, input_tokens=100, output_tokens=50),
        )


def _make_settings(world: Path) -> app_config.Settings:
    base = app_config.load_settings()
    return app_config.Settings(
        data=base.data,
        sync=base.sync,
        model=base.model,
        pick=base.pick,
        debate=app_config.DebateConfig(
            candidates=8,
            min_debated=6,
            bars=10,
            concurrency=2,
            call_timeout_seconds=60,
            llm=base.debate.llm,
        ),
        root=world.parent.parent,
        sections=base.sections,
    )


def _http_json(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_web_static_meta_and_no_secret_leak(world: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "ak-super-secret-never-appear-in-web-ui"
    monkeypatch.setenv("WORKBENCH_AI_API_KEY", secret)
    settings = _make_settings(world)
    server, _hub = web.create_server(settings, default_db=world, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(f"{base_url}/", timeout=5) as resp:
            html = resp.read().decode("utf-8")
            assert resp.status == 200
            assert "AI 辩论复核台" in html

        with urllib.request.urlopen(f"{base_url}/static/app.js", timeout=5) as resp:
            assert resp.status == 200
            assert "EventSource" in resp.read().decode("utf-8")

        # 路径穿越防护
        code, _ = _http_json(f"{base_url}/static/..%2Fweb.py")
        assert code in (403, 404)

        status, meta = _http_json(f"{base_url}/api/meta")
        assert status == 200
        assert meta["config"]["api_key_configured"] is True
        assert secret not in json.dumps(meta, ensure_ascii=False)
        assert DAY.isoformat() in meta["trading_days"]

        # 演练预览接口
        status, dry = _http_json(
            f"{base_url}/api/dry-run",
            method="POST",
            payload={"db": str(world), "as_of": DAY.isoformat()},
        )
        assert status == 200
        assert len(dry["request"]["candidates"]) == 8
    finally:
        server.shutdown()
        server.server_close()


def test_web_live_run_sse_deltas_and_history_detail(world: Path) -> None:
    settings = _make_settings(world)
    fake = StreamingFakeSidecar(delay=0.01)
    server, _hub = web.create_server(
        settings,
        default_db=world,
        host="127.0.0.1",
        port=0,
        sidecar_factory=lambda _s: fake,
    )
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    sse_events: list[tuple[str, dict[str, Any]]] = []
    sse_ready = threading.Event()
    sse_done = threading.Event()

    def listen_sse() -> None:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/stream")
        with urllib.request.urlopen(req, timeout=15) as resp:
            event_name = ""
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    parsed = json.loads(line[5:].strip())
                    sse_events.append((event_name, parsed["payload"]))
                    if event_name == "status":
                        sse_ready.set()
                    if event_name == "job.completed":
                        sse_done.set()
                        return

    sse_thread = threading.Thread(target=listen_sse, daemon=True)
    sse_thread.start()
    try:
        assert sse_ready.wait(timeout=5), "SSE 未及时建立"
        base_url = f"http://127.0.0.1:{port}"
        status, started = _http_json(
            f"{base_url}/api/runs/start",
            method="POST",
            payload={"db": str(world), "as_of": DAY.isoformat()},
        )
        assert status == 202 and started["started"] is True
        assert sse_done.wait(timeout=15), "后台辩论任务未在预期内完成"

        names = [name for name, _ in sse_events]
        assert "run.prepared" in names
        assert "role.delta" in names
        assert "run.frozen" in names
        assert "job.completed" in names

        deltas = [p for name, p in sse_events if name == "role.delta"]
        assert any(d["channel"] == "thinking" and "正在核对量价分位" in d["delta"] for d in deltas)
        assert any(d["role"] == "final_pick" and d["channel"] == "thinking" for d in deltas)

        # 验证 /api/meta 与 /api/run 返回完整冻结结果与内存思考流
        status, meta = _http_json(f"{base_url}/api/meta")
        assert status == 200 and len(meta["runs"]) == 1
        hp = meta["runs"][0]["hash_prefix"]

        status, detail = _http_json(f"{base_url}/api/run?db={urllib.parse.quote(str(world))}&as_of={DAY.isoformat()}&hash={hp}")
        assert status == 200
        assert detail["completed"] is True
        assert detail["official"] is True
        assert len(detail["final"]["picks"]) == 6
        assert any(v["thinking"] for v in detail["live_streams"].values())
        # role.delta 不写入磁盘 events.ndjson
        assert all(ev.get("type") != "role.delta" for ev in detail["events"])
    finally:
        server.shutdown()
        server.server_close()
        sse_thread.join(timeout=2)


def test_resume_preserves_full_roster(world: Path) -> None:
    """续跑仅剩部分股票时，侧车收到的 sub_req.roster 仍覆盖全部送辩候选。"""
    settings = _make_settings(world)
    first_snap = service.snapshot.build_request(world, settings=settings, as_of=DAY)
    total_codes = list(first_snap.request.codes)

    # 第一次只辩完前 2 只后触发中断
    class PartialSidecar(StreamingFakeSidecar):
        def debate(self, request, *, on_event=None, stop=None):
            self.seen_requests.append(request)
            from rebound.debate.sidecar import SidecarStopped

            for idx, cand in enumerate(request.candidates[:2]):
                code = cand.ts_code
                stock_analysts = [
                    contracts.AnalystView(code, role, "bull", "结论", ("风险",), ("features.r5",))
                    for role in contracts.ANALYST_ROLES
                ]
                stock_debate = [
                    contracts.DebateEntry(code, rnd, "论点", ("bars[-1].close",))
                    for rnd in contracts.DEBATE_ENTRY_ROUNDS
                ]
                verdict = contracts.RiskVerdict(code, "看多", 80.0, "判断", ("风险",), "关注", ("features.r20",))
                if on_event is not None:
                    on_event(
                        contracts.Event(
                            "stock.completed",
                            {
                                "ts_code": code,
                                "verdict": verdict.verdict,
                                "score": verdict.score,
                                "usage": contracts.Usage(6 * (idx + 1), 0, 10, 5).payload(),
                                "stock": {
                                    "verdict": verdict.payload(),
                                    "analysts": [
                                        {
                                            "ts_code": a.ts_code,
                                            "role": a.role,
                                            "stance": a.stance,
                                            "conclusion": a.conclusion,
                                            "risks": list(a.risks),
                                            "citations": list(a.citations),
                                        }
                                        for a in stock_analysts
                                    ],
                                    "debate": [
                                        {
                                            "ts_code": d.ts_code,
                                            "round": d.round,
                                            "argument": d.argument,
                                            "citations": list(d.citations),
                                        }
                                        for d in stock_debate
                                    ],
                                },
                            },
                        )
                    )
            raise SidecarStopped("模拟中途断开")

    with pytest.raises(Exception):
        service.run_debate(world, settings=settings, as_of=DAY, sidecar_client=PartialSidecar())

    resumed_sidecar = StreamingFakeSidecar()
    service.run_debate(world, settings=settings, as_of=DAY, sidecar_client=resumed_sidecar)
    assert len(resumed_sidecar.seen_requests) == 1
    sub_req = resumed_sidecar.seen_requests[0]
    assert len(sub_req.candidates) == len(total_codes) - 2
    assert sub_req.roster is not None
    assert [row["ts_code"] for row in sub_req.roster] == total_codes
