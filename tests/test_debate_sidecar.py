"""辩论侧车管理与看门狗测试：假脚本子进程测看门狗/协议异常，真侧车测 --health 与 --validate。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from test_debate_contracts import _debated_payload, make_request

from rebound.debate import contracts, sidecar

ROOT = Path(__file__).resolve().parents[1]


def _client(settings, monkeypatch, script: Path, *, stall: float) -> sidecar.Sidecar:
    monkeypatch.setattr(sidecar.Sidecar, "_stall_seconds", lambda self: stall, raising=False)
    monkeypatch.setattr(sidecar.Sidecar, "_environment", lambda self: {"PYTHONUTF8": "1"}, raising=False)
    return sidecar.Sidecar(settings, node=sys.executable, entry=script, api_key="test-key")


def _script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "fake_sidecar.py"
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# 看门狗与子进程异常（使用 Python 假侧车，不依赖 node）
# --------------------------------------------------------------------------- #


def test_silent_process_is_killed_as_timeout(tmp_path: Path, settings, monkeypatch) -> None:
    entry = _script(tmp_path, "import time\ntime.sleep(60)\n")
    client = _client(settings, monkeypatch, entry, stall=0.8)
    with pytest.raises(sidecar.SidecarTimeout, match="没有任何输出"):
        client._invoke([], stdin_line=None, timeout=None)


def test_chatty_slow_process_survives_beyond_stall_threshold(tmp_path: Path, settings, monkeypatch) -> None:
    entry = _script(
        tmp_path,
        "import time\n"
        "for i in range(6):\n"
        "    print('{\"type\":\"role.started\"}', flush=True)\n"
        "    time.sleep(0.3)\n",
    )
    client = _client(settings, monkeypatch, entry, stall=0.8)
    lines, _, code = client._invoke([], stdin_line=None, timeout=None)
    assert code == 0 and len(lines) == 6


def test_stderr_counts_as_activity(tmp_path: Path, settings, monkeypatch) -> None:
    entry = _script(
        tmp_path,
        "import sys, time\n"
        "for i in range(6):\n"
        "    print('busy', i, file=sys.stderr, flush=True)\n"
        "    time.sleep(0.3)\n"
        "print('{\"type\":\"health\"}', flush=True)\n",
    )
    client = _client(settings, monkeypatch, entry, stall=0.8)
    lines, errors, code = client._invoke([], stdin_line=None, timeout=None)
    assert code == 0 and len(errors) == 6 and len(lines) == 1


def test_stop_event_and_callback_error_kill_child_promptly(tmp_path: Path, settings, monkeypatch) -> None:
    entry = _script(
        tmp_path,
        "import time\nprint('{\"type\":\"run.started\"}', flush=True)\ntime.sleep(60)\n",
    )
    client = _client(settings, monkeypatch, entry, stall=30.0)
    stop = threading.Event()

    def stop_soon() -> None:
        time.sleep(0.2)
        stop.set()

    threading.Thread(target=stop_soon, daemon=True).start()
    started = time.monotonic()
    with pytest.raises(sidecar.SidecarStopped):
        client._invoke([], stdin_line=None, timeout=None, stop=stop)
    assert time.monotonic() - started < 5

    def fail_hook(_event) -> None:
        raise OSError("磁盘满")

    with pytest.raises(OSError, match="磁盘满"):
        client._invoke([], stdin_line=None, timeout=None, on_event=fail_hook)


def test_fake_sidecar_runs_debate_and_finalize_round_trip(tmp_path: Path, settings, monkeypatch) -> None:
    req = make_request()
    debated = _debated_payload(req)
    finalized = {
        "type": "finalized",
        "final_count": 2,
        "picks": [
            {"ts_code": req.codes[0], "rank": 1, "reason": "最好"},
            {"ts_code": req.codes[1], "rank": 2, "reason": "次优"},
        ],
        "reason": "整体排序",
        "usage": {"calls": 1, "failed_calls": 0, "input_tokens": 10, "output_tokens": 5},
    }
    entry = _script(
        tmp_path,
        "import json, sys\n"
        f"DEBATED = {json.dumps(debated, ensure_ascii=False)}\n"
        f"FINALIZED = {json.dumps(finalized, ensure_ascii=False)}\n"
        "line = sys.stdin.readline()\n"
        "if '--debate' in sys.argv:\n"
        "    print(json.dumps({'type':'run.started','candidates':3}, ensure_ascii=False), flush=True)\n"
        "    print(json.dumps(DEBATED, ensure_ascii=False), flush=True)\n"
        "elif '--finalize' in sys.argv:\n"
        "    print(json.dumps(FINALIZED, ensure_ascii=False), flush=True)\n",
    )
    client = _client(settings, monkeypatch, entry, stall=10.0)
    events: list[str] = []
    res = client.debate(req, on_event=lambda e: events.append(e.type))
    assert events == ["run.started"]
    assert res.run_id == req.run_id and len(res.verdicts) == 3

    fin = client.finalize(req, res.verdicts)
    assert fin.final_count == 2 and [p.ts_code for p in fin.picks] == list(req.codes[:2])


def test_crashing_or_garbled_sidecar_raises_sidecar_error(tmp_path: Path, settings, monkeypatch) -> None:
    req = make_request()
    crash = _script(tmp_path, "import sys\nprint('oops', file=sys.stderr)\nsys.exit(2)\n")
    with pytest.raises(sidecar.SidecarError, match="退出码 2"):
        _client(settings, monkeypatch, crash, stall=5.0).debate(req)

    garbled = _script(tmp_path, "print('not json at all', flush=True)\n")
    with pytest.raises(contracts.ContractError, match="非 JSON"):
        _client(settings, monkeypatch, garbled, stall=5.0).debate(req)


def test_stall_threshold_exceeds_call_timeout(settings) -> None:
    client = sidecar.Sidecar(settings, api_key="x")
    assert client._stall_seconds() > float(settings.debate.call_timeout_seconds)


# --------------------------------------------------------------------------- #
# 真实 Node 侧车交叉回环（仅在 node 与 node_modules 存在时运行）
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    shutil.which("node") is None or not (sidecar.SIDECAR_DIR / "node_modules").is_dir(),
    reason="需要本机 Node 与 debate_agent/node_modules",
)
class TestRealSidecarRoundTrip:
    def test_health_and_validate_accept_valid_request_and_reject_bad_one(self, settings) -> None:
        client = sidecar.Sidecar(settings)
        health = client.health()
        assert health["ok"] is True and health["protocol_version"] == contracts.PROTOCOL_VERSION

        req = make_request()
        valid = client.validate(req)
        assert valid["protocol_version"] == contracts.PROTOCOL_VERSION
        assert valid["run_id"] == req.run_id
        assert valid["roles"] == sorted(contracts.AGENT_ROLES)

        broken = req.payload()
        broken["limits"]["final"] = contracts.MAX_CANDIDATES + 1
        done = subprocess.run(
            ["node", str(sidecar.SIDECAR_ENTRY), "--validate"],
            cwd=sidecar.SIDECAR_DIR,
            input=json.dumps(broken, ensure_ascii=False),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env={**os.environ},
        )
        assert done.returncode != 0
        last = json.loads(done.stdout.strip().splitlines()[-1])
        assert last["type"] == "error" and "limits.final" in last["message"]
