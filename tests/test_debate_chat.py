"""AI 投研复盘对话（档位 B：`rebound/debate/chat.py` + `web.py` `/api/chat/*`）完整单元测试。"""

from __future__ import annotations

import json
import shutil
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Iterator

import duckdb
import pytest
from conftest import TEST_FIRST_FIT

from rebound import config as app_config
from rebound.debate import chat, runs, service, web
from rebound.model.artifacts import models_root
from rebound.model.score import fill_scores
from tests.test_debate_web import StreamingFakeSidecar

DAY = date(2023, 3, 15)


def _copy_world(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    shutil.copytree(models_root(src), models_root(dst))
    return dst


@pytest.fixture(scope="module")
def scored_pristine(trained_world_pristine: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = _copy_world(trained_world_pristine, tmp_path_factory.mktemp("debate-chat") / "market.duckdb")
    fill_scores(path, first_fit=TEST_FIRST_FIT)
    return path


@pytest.fixture
def world(scored_pristine: Path, tmp_path: Path) -> Path:
    return _copy_world(scored_pristine, tmp_path / "data" / "market.duckdb")


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
        root=app_config.load_settings().root,
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


def _read_sse_post(url: str, payload: dict[str, Any]) -> tuple[int, list[tuple[str, dict[str, Any]]]]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events: list[tuple[str, dict[str, Any]]] = []
    with urllib.request.urlopen(req, timeout=15) as resp:
        ev_name = ""
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if line.startswith("event:"):
                ev_name = line[6:].strip()
            elif line.startswith("data:"):
                parsed = json.loads(line[5:].strip())
                events.append((ev_name, parsed.get("payload") or {}))
        return resp.status, events


def test_chat_toolbox_all_six_readonly_tools(world: Path) -> None:
    """验证 6 个只读工具对冻结辩论账本与 DuckDB 的 100% 访问能力。"""
    settings = _make_settings(world)
    out = service.run_debate(world, settings=settings, as_of=DAY, sidecar_client=StreamingFakeSidecar())
    hp = out["final"]["input_hash"][:12]

    hub = web.DebateHub(settings, default_db=world)
    toolbox = hub.build_chat_toolbox(db=str(world), as_of=DAY.isoformat(), hash_prefix=hp)

    snap = toolbox.global_context_snapshot()
    assert len(snap["all_candidates_summary"]) == 8
    assert snap["final"]["final_count"] == 6

    first = snap["all_candidates_summary"][0]
    second = snap["all_candidates_summary"][1]

    # 1. get_stock_debate：支持完整代码、裸 6 位代码、中文简称，且对未知代码优雅报错
    ok1, summary1, data1 = toolbox.execute("get_stock_debate", {"ts_code": first["ts_code"]})
    assert ok1 is True and first["ts_code"] in summary1
    assert len(data1["analysts"]) == 2 and len(data1["debate"]) == 3 and data1["verdict"] is not None

    bare_code = first["ts_code"].split(".")[0]
    ok1b, _, data1b = toolbox.execute("get_stock_debate", {"ts_code": bare_code, "include_bars": False})
    assert ok1b is True and data1b["ts_code"] == first["ts_code"] and "bars" not in data1b

    ok1c, _, data1c = toolbox.execute("get_stock_debate", {"ts_code": first["name"]})
    assert ok1c is True and data1c["ts_code"] == first["ts_code"]

    ok_bad, _, data_bad = toolbox.execute("get_stock_debate", {"ts_code": "999999.SH"})
    assert ok_bad is False and "available_codes" in data_bad

    # 2. compare_candidates
    ok2, summary2, data2 = toolbox.execute(
        "compare_candidates",
        {"ts_codes": [first["ts_code"], second["ts_code"], "000000.SZ"]},
    )
    assert ok2 is True and len(data2["compared"]) == 2
    assert data2["missing"] == ["000000.SZ"]
    assert first["name"] in summary2

    # 3. screen_candidates
    ok3, _, data3 = toolbox.execute("screen_candidates", {"verdict": "看多", "sort_by": "vol5", "descending": True, "limit": 5})
    assert ok3 is True and len(data3["rows"]) == 5
    vals = [r["sort_value"] for r in data3["rows"] if isinstance(r["sort_value"], (int, float))]
    assert vals == sorted(vals, reverse=True)

    # 4. query_stock_history
    ok4, _, data4 = toolbox.execute("query_stock_history", {"ts_code": first["ts_code"], "bars": 12})
    assert ok4 is True and len(data4["bars"]) == 12
    assert data4["bars"][-1]["date"] <= DAY.isoformat()

    # 5. check_post_signal_returns（有未来行情 vs 已是最新交易日）
    ok5, _, data5 = toolbox.execute("check_post_signal_returns", {"ts_codes": [first["ts_code"]], "hold_days": 5})
    assert ok5 is True
    assert data5["buy_day"] > DAY.isoformat()
    assert len(data5["results"]) == 1 and "open_to_open_pct" in data5["results"][0]

    con = duckdb.connect(str(world), read_only=True)
    try:
        latest_day = str(con.execute("SELECT MAX(CAST(trade_date AS VARCHAR)) FROM daily").fetchone()[0])[:10]
    finally:
        con.close()
    ok5b, summary5b, data5b = toolbox.execute("check_post_signal_returns", {"as_of": latest_day})
    assert ok5b is True and data5b["available_future_days"] == 0 and "尚无未来行情" in summary5b

    # 6. list_debate_runs
    ok6, _, data6 = toolbox.execute("list_debate_runs", {"limit": 5})
    assert ok6 is True and len(data6["runs"]) == 1
    assert data6["runs"][0]["hash_prefix"] == hp


def test_chat_http_tool_loop_streaming_persistence_and_immutability(
    world: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """端到端测试 /api/chat/*：多轮工具调用、SSE 思考/正文流、落盘持久化与冻结账本零改动。"""
    secret = "sk-never-leak-in-chat-ui"
    monkeypatch.setenv("WORKBENCH_AI_API_KEY", secret)

    settings = _make_settings(world)
    out = service.run_debate(world, settings=settings, as_of=DAY, sidecar_client=StreamingFakeSidecar())
    run_dir = Path(out["run_dir"])
    hp = out["final"]["input_hash"][:12]
    target_code = out["final"]["picks"][1]["ts_code"]
    attached_code = out["final"]["picks"][0]["ts_code"]

    # 记录冻结文件原始字节，确保对话绝不触碰
    frozen_before = {
        name: (run_dir / name).read_bytes()
        for name in ("request.json", "result.json", "final.json")
    }

    captured_calls: list[list[dict[str, Any]]] = []

    def fake_stream_fn(
        _settings: app_config.Settings,
        _api_key: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        _stop: threading.Event,
    ) -> Iterator[chat.StreamChunk]:
        captured_calls.append(json.loads(json.dumps(messages, ensure_ascii=False)))
        assert len(tools) == 9
        if len(captured_calls) == 1:
            # 第 1 轮：流式思考 + 分片 tool_calls（同时查两类只读工具）
            yield chat.StreamChunk(thinking="先调工具核查候选明细与量比排名。")
            yield chat.StreamChunk(
                tool_calls_delta=[
                    {
                        "index": 0,
                        "id": "call_stock_1",
                        "type": "function",
                        "function": {"name": "get_stock_debate", "arguments": f'{{"ts_code":"{target_code}"'},
                    }
                ]
            )
            yield chat.StreamChunk(
                tool_calls_delta=[
                    {"index": 0, "function": {"arguments": "}"}},
                    {
                        "index": 1,
                        "id": "call_screen_2",
                        "type": "function",
                        "function": {"name": "screen_candidates", "arguments": '{"sort_by":"vol5","limit":3}'},
                    },
                ],
                finish_reason="tool_calls",
                usage={"input_tokens": 1200, "output_tokens": 60},
            )
        else:
            # 第 2 轮：校验工具结果已回传到 messages，然后逐字流式输出最终研报结论
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            assert len(tool_msgs) == 2
            assert target_code in tool_msgs[0]["content"]
            yield chat.StreamChunk(thinking="已核对工具数据，开始组织结论。")
            yield chat.StreamChunk(text=f"### 核查结论\n- 标的 `{target_code}` 已核实 `features.r5` 与风控裁决。")
            yield chat.StreamChunk(
                text="\n- 量比前 3 筛查完成。",
                finish_reason="stop",
                usage={"input_tokens": 1800, "output_tokens": 90},
            )

    server, _hub = web.create_server(
        settings,
        default_db=world,
        host="127.0.0.1",
        port=0,
        chat_stream_fn=fake_stream_fn,
    )
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{port}"
        qs = urllib.parse.urlencode({"db": str(world), "as_of": DAY.isoformat(), "hash": hp})

        # 初始会话为空且返回 8 角色与 6 工具清单
        status, initial = _http_json(f"{base_url}/api/chat/sessions?{qs}")
        assert status == 200
        assert initial["sessions"] == []
        assert len(initial["roles"]) == 8
        assert len(initial["tools"]) == 9
        assert secret not in json.dumps(initial, ensure_ascii=False)

        # 空消息在发流前直接返回 400
        bad_status, bad_body = _http_json(
            f"{base_url}/api/chat/stream",
            method="POST",
            payload={"db": str(world), "as_of": DAY.isoformat(), "hash": hp, "message": "   "},
        )
        assert bad_status == 400 and "不能为空" in bad_body["error"]

        # 发起流式对话
        st_code, sse_events = _read_sse_post(
            f"{base_url}/api/chat/stream",
            {
                "db": str(world),
                "as_of": DAY.isoformat(),
                "hash": hp,
                "message": f"帮我核查 {target_code} 并筛出量比前 3",
                "role_preset": "research_copilot",
                "attached_codes": [attached_code],
            },
        )
        assert st_code == 200
        ev_names = [name for name, _ in sse_events]
        assert ev_names[0] == "chat.start"
        assert "chat.thinking" in ev_names
        assert ev_names.count("chat.tool_start") == 2
        assert ev_names.count("chat.tool_done") == 2
        assert "chat.delta" in ev_names
        assert ev_names[-1] == "chat.done"

        done_payload = sse_events[-1][1]
        session_id = done_payload["session_id"]
        assert done_payload["usage"] == {"input_tokens": 3000, "output_tokens": 150}
        assert target_code in done_payload["message"]["content"]
        assert len(done_payload["message"]["tool_calls"]) == 2
        assert all(tc["ok"] is True for tc in done_payload["message"]["tool_calls"])

        # 验证 system prompt 注入了方法论与挂载个股
        sys_prompt = captured_calls[0][0]["content"]
        assert "docs/debate-methodology.md" not in sys_prompt or "反弹模型" in sys_prompt
        assert "六步" in sys_prompt or "风控主席" in sys_prompt
        assert attached_code in sys_prompt

        # 验证落盘到 <run_dir>/chat_sessions.json 且可重启重载
        store_file = run_dir / "chat_sessions.json"
        assert store_file.is_file()
        status, listed = _http_json(f"{base_url}/api/chat/sessions?{qs}")
        assert status == 200 and len(listed["sessions"]) == 1
        assert len(listed["sessions"][0]["messages"]) == 2

        # 验证冻结账本字节完全未变
        for name, before_bytes in frozen_before.items():
            assert (run_dir / name).read_bytes() == before_bytes

        # 创建新会话并删除旧会话
        status, created = _http_json(
            f"{base_url}/api/chat/sessions",
            method="POST",
            payload={"db": str(world), "as_of": DAY.isoformat(), "hash": hp, "role_preset": "risk_chair", "title": "风控追问"},
        )
        assert status == 201 and created["session"]["role_preset"] == "risk_chair"

        del_qs = urllib.parse.urlencode({"db": str(world), "as_of": DAY.isoformat(), "hash": hp, "session_id": session_id})
        status, deleted = _http_json(f"{base_url}/api/chat/sessions?{del_qs}", method="DELETE")
        assert status == 200 and deleted["deleted"] is True

        status, after_del = _http_json(f"{base_url}/api/chat/sessions?{qs}")
        assert len(after_del["sessions"]) == 1 and after_del["sessions"][0]["id"] == created["session"]["id"]
    finally:
        server.shutdown()
        server.server_close()


def test_dry_run_chat_uses_preview_cache_without_polluting_run_dirs(world: Path) -> None:
    """演练预览状态下对话也能读取全部 8 只预览候选，且聊天落盘到 `_chats/` 不污染运行目录。"""
    settings = _make_settings(world)

    def echo_stream(
        _settings: app_config.Settings,
        _api_key: str,
        messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
        _stop: threading.Event,
    ) -> Iterator[chat.StreamChunk]:
        # 验证预览候选已经注入 system prompt
        assert "all_candidates_summary" in messages[0]["content"]
        yield chat.StreamChunk(text="演练快照已读取完毕。", finish_reason="stop", usage={"input_tokens": 100, "output_tokens": 10})

    server, _hub = web.create_server(
        settings,
        default_db=world,
        host="127.0.0.1",
        port=0,
        chat_stream_fn=echo_stream,
    )
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{port}"
        st, dry = _http_json(
            f"{base_url}/api/dry-run",
            method="POST",
            payload={"db": str(world), "as_of": DAY.isoformat()},
        )
        assert st == 200
        preview_code = dry["request"]["candidates"][0]["ts_code"]

        st_code, events = _read_sse_post(
            f"{base_url}/api/chat/stream",
            {
                "db": str(world),
                "as_of": DAY.isoformat(),
                "hash": dry["hash_prefix"],
                "message": f"看看 {preview_code}",
                "attached_codes": [preview_code],
            },
        )
        assert st_code == 200 and events[-1][0] == "chat.done"

        # 演练目录下不应创建假的 <variant>-<cadence>/<as_of>/<hash> 目录
        fake_run_dir = runs.debate_root(world) / f"{dry['variant']}-{dry['cadence']}" / DAY.isoformat() / dry["hash_prefix"]
        assert not fake_run_dir.exists()
        # 应安全落盘在 _chats/
        chat_fallback = runs.debate_root(world) / "_chats" / f"{dry['variant']}-{dry['cadence']}-{DAY.isoformat()}.json"
        assert chat_fallback.is_file()
    finally:
        server.shutdown()
        server.server_close()
