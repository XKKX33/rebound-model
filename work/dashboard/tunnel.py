"""Supervise the temporary Cloudflare sharing connection and verified entry."""
from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import re
import msvcrt
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
CLIENT = SHARING / "runtime/cloudflared.exe"
URL_PATTERN = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com\b", re.I)


def save_state(state: dict) -> None:
    state["checked_at"] = datetime.now(timezone.utc).isoformat()
    pending = STATE_FILE.with_suffix(".tmp")
    pending.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(STATE_FILE)


def health(url: str, authorization: str) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/__share/health", headers={"Authorization": authorization},
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        result = json.loads(response.read(4096))
    if result.get("status") != "ready" or result.get("transport") != "websocket-v1":
        raise RuntimeError("共享网关尚未就绪或尚未更新流式转发")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8880)
    arguments = parser.parse_args()
    SHARING.mkdir(parents=True, exist_ok=True)
    if not CLIENT.is_file():
        raise SystemExit("未找到已校验的 Cloudflare 客户端，位置为 work/sharing/runtime/cloudflared.exe。")
    access = json.loads(ACCESS_FILE.read_text(encoding="utf-8"))
    authorization = "Basic " + base64.b64encode(
        f"{access['username']}:{access['password']}".encode(),
    ).decode()
    health(f"http://127.0.0.1:{arguments.port}", authorization)
    lock_path = SHARING / ".tunnel.lock"
    lock_path.touch(exist_ok=True)
    lock = lock_path.open("r+b")
    if lock_path.stat().st_size == 0:
        lock.write(b"1")
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close()
        raise SystemExit("共享连接已在运行，请查看 work/sharing/public_state.json 中的当前地址。")
    state = {
        "url": None, "status": "connecting", "provider": "Cloudflare Quick Tunnel",
        "temporary": True, "domain_rotates": False, "changes_when_recreated": True,
        "transport": "websocket-v1", "manager_pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(), "restarts": 0,
    }
    process = None
    backoff = 3
    try:
        while True:
            events: queue.Queue[str] = queue.Queue()
            command = [
                str(CLIENT), "tunnel", "--no-autoupdate", "--protocol", "http2",
                "--url", f"http://127.0.0.1:{arguments.port}", "--loglevel", "info",
            ]
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if state.get("url"):
                state["previous_url"] = state["url"]
            state.update(url=None, status="connecting", connector_pid=process.pid)
            state.pop("error", None)
            state.pop("last_health", None)
            save_state(state)

            def collect(source=process.stdout, target=events):
                for line in source:
                    target.put(line.strip())

            threading.Thread(target=collect, daemon=True).start()
            next_check = time.monotonic() + 2
            announced_url = None
            while process.poll() is None or not events.empty():
                try:
                    line = events.get(timeout=1)
                except queue.Empty:
                    line = ""
                match = URL_PATTERN.search(line)
                if match:
                    address = match.group(0)
                    if address != state.get("url"):
                        state.update(url=address, status="checking")
                        save_state(state)
                        next_check = time.monotonic() + 2
                if line and any(marker in line for marker in (" ERR ", " WRN ", "Registered tunnel connection")):
                    print(line[:600], flush=True)
                if state.get("url") and time.monotonic() >= next_check:
                    try:
                        result = health(state["url"], authorization)
                        state.update(status="ready", last_health=result)
                        state.pop("error", None)
                        state.pop("failure_since", None)
                        backoff = 3
                        if announced_url != state["url"]:
                            print(f"可分享地址：{state['url']}", flush=True)
                            announced_url = state["url"]
                    except Exception as error:
                        if state.get("status") == "ready":
                            print("公网连接暂时中断，客户端正在重新连接；本机看板继续运行。", flush=True)
                        state.update(status="unavailable", error=str(error)[:180])
                        state.setdefault("failure_since", datetime.now(timezone.utc).isoformat())
                        state.pop("last_health", None)
                    save_state(state)
                    next_check = time.monotonic() + (30 if state["status"] == "ready" else 10)
            state.update(status="reconnecting", last_connector_exit=process.returncode)
            state.pop("last_health", None)
            state["restarts"] += 1
            save_state(state)
            print(f"共享客户端已退出，将在 {backoff} 秒后恢复。重建连接后地址可能改变。", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
    except KeyboardInterrupt:
        pass
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        state["status"] = "stopped"
        state.pop("last_health", None)
        save_state(state)
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
