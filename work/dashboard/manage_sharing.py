"""Start or reuse only this project's sharing gateway and tunnel."""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil

from paths import PROJECT

DIRECTORY = Path(__file__).resolve().parent
SHARING = PROJECT / "work/sharing"
STATE = SHARING / "public_state.json"


def auth():
    access = json.loads((DIRECTORY / "share_access.json").read_text(encoding="utf-8"))
    return "Basic " + base64.b64encode(f"{access['username']}:{access['password']}".encode()).decode()


def check(url, authorization=None, transport=False):
    request = urllib.request.Request(url, headers={"Authorization": authorization} if authorization else {})
    with urllib.request.urlopen(request, timeout=5) as response:
        payload = json.loads(response.read(4096))
    return payload.get("status") == "ready" and (
        not transport or payload.get("transport") == "websocket-v1"
    )


def process_matches(pid, script):
    try:
        if not pid or int(pid) <= 0:
            return None
        process = psutil.Process(int(pid))
        expected = (DIRECTORY / script).resolve()
        if any(Path(arg).resolve() == expected for arg in process.cmdline() if arg.lower().endswith(".py")):
            return process
    except (TypeError, ValueError, psutil.NoSuchProcess):
        return None
    except psutil.AccessDenied as error:
        raise PermissionError(f"Windows 拒绝读取共享进程 {pid}；尚未关闭任何服务。") from error
    return None


def spawn(script, label, arguments=(), environment=None):
    output = (SHARING / f"{label}.log").open("ab")
    errors = (SHARING / f"{label}-error.log").open("ab")
    try:
        return subprocess.Popen(
            [sys.executable, "-B", "-X", "utf8", str(DIRECTORY / script), *arguments],
            cwd=PROJECT, env=environment or os.environ.copy(),
            stdin=subprocess.DEVNULL, stdout=output, stderr=errors,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    finally:
        output.close()
        errors.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8880)
    args = parser.parse_args()
    SHARING.mkdir(parents=True, exist_ok=True)
    if not check("http://127.0.0.1:8875/api/health"):
        raise RuntimeError("8875 看板未就绪，请先运行“重启看板.cmd”。")
    gateway_url = f"http://127.0.0.1:{args.port}/__share/health"
    gateway_ready = False
    try:
        gateway_ready = check(gateway_url, auth(), True)
    except (OSError, ValueError):
        pass
    if not gateway_ready:
        for connection in psutil.net_connections(kind="tcp"):
            if connection.status != psutil.CONN_LISTEN or connection.laddr.port != args.port:
                continue
            process = process_matches(connection.pid, "share.py")
            if process is None:
                raise RuntimeError(f"端口 {args.port} 已由其他程序使用，未关闭该程序。")
            process.terminate()
            process.wait(timeout=10)
        environment = os.environ.copy()
        environment.update(
            DASHBOARD_SHARE_PORT=str(args.port), DASHBOARD_SHARE_UPSTREAM="http://127.0.0.1:8875",
            PYTHONDONTWRITEBYTECODE="1",
        )
        spawn("share.py", "gateway", environment=environment)
        for _ in range(15):
            time.sleep(1)
            try:
                if check(gateway_url, auth(), True):
                    gateway_ready = True
                    break
            except (OSError, ValueError):
                pass
    if not gateway_ready:
        raise RuntimeError("共享网关启动失败，请查看 work/sharing/gateway-error.log。")

    current = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}
    manager = process_matches(current.get("manager_pid", 0), "tunnel.py")
    if manager is None:
        started = spawn("tunnel.py", "tunnel", ("--port", str(args.port)))
        manager_pid = started.pid
    else:
        manager_pid = manager.pid
    deadline = time.monotonic() + 50
    while time.monotonic() < deadline:
        if STATE.is_file():
            current = json.loads(STATE.read_text(encoding="utf-8"))
            if current.get("manager_pid") == manager_pid and current.get("status") == "ready":
                if check(current["url"] + "/__share/health", auth(), True):
                    print(f"可分享地址：{current['url']}/")
                    print("账号为 team，密码仍在 work/dashboard/share_access.json。")
                    print("地址显示后，可关闭这个命令窗口：共享服务已在后台独立运行。")
                    print("电脑须保持开机、联网且不休眠；关闭后台服务、休眠或关机都会影响访问。")
                    print("重复开启共享会复用现有连接。")
                    return 0
        time.sleep(1)
    print("共享连接正在建立或恢复，尚未通过公网核验。")
    print("当前状态与地址见 work/sharing/public_state.json；再次开启共享会复用这条连接。")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, psutil.Error) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
