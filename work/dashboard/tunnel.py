"""Keep the current temporary sharing address tied to an actual health check."""
from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import re
import shutil
import ssl
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from paths import PROJECT

SHARING = PROJECT / "work/sharing"
STATE_FILE = SHARING / "public_state.json"
ACCESS_FILE = Path(__file__).with_name("share_access.json")
URL_PATTERN = re.compile(r"https://[a-z0-9.-]+\.lhr\.life\b", re.I)


def save_state(state: dict) -> None:
    state["checked_at"] = datetime.now(timezone.utc).isoformat()
    pending = STATE_FILE.with_suffix(".tmp")
    pending.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(STATE_FILE)


def health(url: str, authorization: str) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/api/health", headers={"Authorization": authorization},
    )
    with urllib.request.urlopen(request, timeout=8, context=ssl.create_default_context()) as response:
        result = json.loads(response.read(4096))
    if result.get("status") != "ready":
        raise RuntimeError("看板尚未就绪")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8880)
    arguments = parser.parse_args()
    SHARING.mkdir(parents=True, exist_ok=True)
    access = json.loads(ACCESS_FILE.read_text(encoding="utf-8"))
    authorization = "Basic " + base64.b64encode(
        f"{access['username']}:{access['password']}".encode(),
    ).decode()
    health(f"http://127.0.0.1:{arguments.port}", authorization)
    executable = shutil.which("ssh")
    if not executable:
        raise SystemExit("未找到 Windows OpenSSH 客户端。")
    command = [
        executable, "-T", "-o", "ConnectTimeout=15", "-o", "ExitOnForwardFailure=yes",
        "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={SHARING / 'localhostrun_known_hosts'}",
        "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
        f"-R80:127.0.0.1:{arguments.port}", "nokey@localhost.run",
    ]
    events: queue.Queue[str] = queue.Queue()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    def collect() -> None:
        for line in process.stdout:
            events.put(line.strip())

    threading.Thread(target=collect, daemon=True).start()
    state = {
        "url": None, "status": "connecting", "provider": "localhost.run",
        "temporary": True, "domain_rotates": True,
        "manager_pid": os.getpid(), "ssh_pid": process.pid,
    }
    save_state(state)
    next_check = float("inf")
    announced_url = None
    try:
        while process.poll() is None or not events.empty():
            try:
                line = events.get(timeout=1)
            except queue.Empty:
                line = ""
            match = URL_PATTERN.search(line)
            if match:
                address = match.group(0)
                if address != state["url"]:
                    state.update(url=address, status="checking")
                    state.pop("error", None)
                    state.pop("last_health", None)
                    save_state(state)
                    next_check = time.monotonic() + 2
            if line and re.search(r"burst|free tier|removed|disconnect|error|denied", line, re.I):
                print(line[:400], flush=True)
            if line and "removed" in line.lower():
                state["status"] = "unavailable"
                state.pop("last_health", None)
                save_state(state)
                next_check = time.monotonic() + 5
            if state["url"] and time.monotonic() >= next_check:
                try:
                    result = health(state["url"], authorization)
                    state.update(status="ready", last_health=result)
                    state.pop("error", None)
                    if announced_url != state["url"]:
                        print(f"可分享地址：{state['url']}", flush=True)
                        announced_url = state["url"]
                except Exception as error:
                    was_ready = state["status"] == "ready"
                    state.update(status="unavailable", error=str(error)[:200])
                    state.pop("last_health", None)
                    if was_ready:
                        print("临时公网暂时无法访问；本机 8875 看板继续运行。", flush=True)
                save_state(state)
                next_check = time.monotonic() + (30 if state["status"] == "ready" else 10)
    except KeyboardInterrupt:
        pass
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        state["status"] = "stopped"
        state.pop("last_health", None)
        save_state(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
