"""Stop only verified dashboard processes and wait for their actual exit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import psutil

DIRECTORY = Path(__file__).resolve().parent
PORT_SCRIPTS = {8765: "server.py", 8767: "start_debate_ui.py", 8875: "server.py"}


def listeners() -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for entry in psutil.net_connections(kind="tcp"):
        if entry.status == psutil.CONN_LISTEN and entry.laddr.port in PORT_SCRIPTS:
            if entry.pid is None:
                raise RuntimeError(f"无法取得端口 {entry.laddr.port} 的进程身份，未关闭服务。")
            result.setdefault(entry.laddr.port, set()).add(entry.pid)
    return result


def script_matches(arguments: list[str], expected: Path) -> bool:
    for argument in arguments:
        if not argument.lower().endswith(".py"):
            continue
        try:
            if Path(argument).resolve() == expected:
                return True
        except (OSError, ValueError):
            continue
    return False


def inspect() -> list[dict]:
    records = []
    for port, process_ids in listeners().items():
        expected = DIRECTORY / PORT_SCRIPTS[port]
        for process_id in process_ids:
            try:
                process = psutil.Process(process_id)
                if not script_matches(process.cmdline(), expected):
                    raise RuntimeError(
                        f"端口 {port} 的进程 {process_id} 不属于本目录的 {expected.name}，未关闭服务。"
                    )
                records.append({"pid": process_id, "port": port, "created": process.create_time()})
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied as error:
                raise PermissionError(
                    f"Windows 拒绝读取端口 {port} 的进程 {process_id}。"
                    "如需管理其他账户的服务，请以管理员身份运行。"
                ) from error
    return records


def stop(records: list[dict], legacy_only: bool) -> dict:
    stopped = []
    for record in records:
        if legacy_only and record["port"] == 8875:
            continue
        process_id, port = record["pid"], record["port"]
        try:
            process = psutil.Process(process_id)
            if process.create_time() != record["created"] or not script_matches(
                process.cmdline(), DIRECTORY / PORT_SCRIPTS[port],
            ):
                raise RuntimeError(f"端口 {port} 的进程身份已改变，已停止重启。")
            process.terminate()
            process.wait(timeout=15)
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as error:
            raise PermissionError(
                f"Windows 拒绝终止端口 {port} 的进程 {process_id}。"
                "请以管理员身份运行；未删除生成目录。"
            ) from error
        except psutil.TimeoutExpired as error:
            raise RuntimeError(
                f"端口 {port} 的进程 {process_id} 在 15 秒内未完成退出，未删除生成目录。"
            ) from error
        stopped.append({"pid": process_id, "port": port})
    remaining = listeners()
    expected_empty = (8765, 8767) if legacy_only else PORT_SCRIPTS
    if any(remaining.get(port) for port in expected_empty):
        raise RuntimeError("目标端口仍有监听进程，未删除生成目录。")
    return {
        "stopped": stopped,
        "current": [record for record in records if record["port"] == 8875] if legacy_only else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-only", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    arguments = parser.parse_args()
    try:
        records = inspect()
        result = {"listeners": records} if arguments.inspect_only else stop(records, arguments.legacy_only)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (psutil.Error, OSError, RuntimeError) as error:
        print(json.dumps({"error": str(error), "type": type(error).__name__}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
