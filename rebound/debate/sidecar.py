"""将辩论侧车（`debate_agent/`）当作一次性子进程调用，走 stdin/stdout NDJSON。

通道纪律：
    stdin   一行 JSON，然后立刻关闭
    stdout  只走协议：每条进度一行 JSON，末行为 `debated`、`finalized`、`valid`、`health` 或 `error`
    stderr  只放给人看的日志
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from rebound import config as app_config
from rebound.data.errors import DataError
from rebound.debate.contracts import (
    ContractError,
    DebatedResult,
    Event,
    FinalizedResult,
    RiskVerdict,
    RunRequest,
    parse_debated,
    parse_event,
    parse_finalized,
    verify_debated,
    verify_final,
)

__all__ = [
    "MIN_NODE_VERSION",
    "SIDECAR_DIR",
    "SIDECAR_ENTRY",
    "Sidecar",
    "SidecarError",
    "SidecarStopped",
    "SidecarTimeout",
    "SidecarUnavailable",
    "node_version_ok",
    "resolve_api_key",
]

SIDECAR_DIR = app_config.ROOT / "debate_agent"
SIDECAR_ENTRY = SIDECAR_DIR / "src" / "main.ts"
MIN_NODE_VERSION = (22, 19, 0)

_API_KEY_VAR = "DEBATE_LLM_API_KEY"
_KILL_GRACE_SECONDS = 2.0
_STDERR_TAIL = 12
_HARD_CAP_SECONDS = 3 * 3600.0
_STALL_SLACK_SECONDS = 60.0
_WATCH_INTERVAL_SECONDS = 1.0
_TERMINAL_EVENTS = frozenset({"debated", "finalized", "result"})

EventHook = Callable[[Event], None]


class SidecarError(DataError):
    """侧车跑完了但没给出可用结果（崩溃、协议行缺失、形状不符）。"""


class SidecarTimeout(SidecarError):
    """停滞超时或达到硬上限被杀进程。"""


class SidecarStopped(SidecarError):
    """收到停止信号或用户 Ctrl+C，父进程已杀掉子进程。"""


class SidecarUnavailable(SidecarError):
    """起不来：node 不在或版本不够、入口文件不存在、依赖没装或密钥缺失。"""


def node_version_ok(raw: str) -> bool:
    """校验 Node 版本字符串（如 `v24.14.0`）是否满足 `>= 22.19.0`。"""
    match = re.match(r"^\s*v?(\d+)\.(\d+)\.(\d+)", raw)
    if match is None:
        return False
    version = tuple(int(part) for part in match.groups())
    return version >= MIN_NODE_VERSION


def resolve_api_key(settings: app_config.Settings) -> str:
    """从项目 `.env` 与环境变量读取大模型密钥；已有同名环境变量优先。"""
    import dotenv

    name = settings.debate.llm.api_key_env
    dotenv.load_dotenv(settings.root / ".env", override=False)
    key = os.environ.get(name, "").strip()
    if not key:
        raise SidecarUnavailable(
            f"环境变量 {name} 未设置：把密钥写进 {settings.root / '.env'}（该文件已在 .gitignore 里）"
        )
    return key


def _resolve_sidecar_dir(settings: app_config.Settings, entry: Path | None) -> tuple[Path, Path]:
    if entry is not None:
        resolved = Path(entry)
        # 假脚本测试时直接以其所在目录为 cwd，不强求 node_modules
        return resolved.parent.parent if resolved.parent.name == "src" else resolved.parent, resolved
    candidate_dir = settings.root / "debate_agent"
    sidecar_dir = candidate_dir if candidate_dir.is_dir() else SIDECAR_DIR
    return sidecar_dir, sidecar_dir / "src" / "main.ts"


class Sidecar:
    """侧车的一次调用封装，每次执行起一个独立子进程。"""

    def __init__(
        self,
        settings: app_config.Settings,
        *,
        node: str = "node",
        entry: Path | None = None,
        api_key: str | None = None,
    ) -> None:
        self._settings = settings
        self._node_command = node
        self._sidecar_dir, self._entry = _resolve_sidecar_dir(settings, entry)
        self._custom_entry = entry is not None
        self._api_key = api_key

    def preflight(self) -> None:
        """花钱调用前检查 node 可执行文件、Node 版本、入口与依赖目录。"""
        if shutil.which(self._node_command) is None:
            raise SidecarUnavailable(
                f"找不到 {self._node_command!r}。辩论侧车需要 Node >= 22.19；"
                f"装好后运行 `cd {self._sidecar_dir} && npm install`。"
            )
        if not self._entry.exists():
            raise SidecarUnavailable(f"侧车入口不存在：{self._entry}")
        if not self._custom_entry and not (self._sidecar_dir / "node_modules").exists():
            raise SidecarUnavailable(
                f"侧车依赖未安装（{self._sidecar_dir / 'node_modules'} 不存在）："
                f"运行 `cd {self._sidecar_dir} && npm install`"
            )

    def _environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        if self._api_key is None:
            self._api_key = resolve_api_key(self._settings)
        environment[_API_KEY_VAR] = self._api_key
        return environment

    def _stall_seconds(self) -> float:
        """停滞阈值：`call_timeout_seconds * 1.5 + 60`，给长思考加一次重试留出余量。"""
        return float(self._settings.debate.call_timeout_seconds) * 1.5 + _STALL_SLACK_SECONDS

    def health(self, *, timeout: float = 30.0) -> Mapping[str, Any]:
        """`node src/main.ts --health`：验证 Node 版本、语法与依赖可加载，不碰网络。"""
        self.preflight()
        body, errors, code = self._invoke(["--health"], stdin_line=None, timeout=timeout, need_key=False)
        if code != 0:
            raise SidecarError(f"侧车 --health 退出码 {code}{self._stderr_suffix(errors)}")
        if not body:
            raise SidecarError("侧车 --health 没有输出任何一行")
        event = parse_event(body[-1])
        if event is None or event.type != "health":
            raise SidecarError(f"侧车 --health 的输出不是 health 事件：{body[-1][:200]!r}")
        node_ver = str(event.payload.get("node", ""))
        if not node_version_ok(node_ver):
            raise SidecarUnavailable(f"Node 版本过低（收到 {node_ver!r}，要求 >= v22.19.0）")
        return event.payload

    def validate(self, request: RunRequest, *, timeout: float = 60.0) -> Mapping[str, Any]:
        """`node src/main.ts --validate`：校验请求体跨语言形状，不碰网络。"""
        self.preflight()
        body, error_lines, code = self._invoke(
            ["--validate"], stdin_line=request.to_line(), timeout=timeout, need_key=False
        )
        for line in body:
            event = parse_event(line)
            if event is None:
                continue
            if event.type == "valid":
                return event.payload
            if event.type == "error":
                raise SidecarError(
                    f"侧车拒收这份请求：{event.payload.get('message', '未给出原因')}"
                    f"{self._stderr_suffix(error_lines)}"
                )
        raise SidecarError(
            f"侧车 --validate 退出码 {code}，没有给出 valid 事件{self._stderr_suffix(error_lines)}"
        )

    def debate(
        self,
        request: RunRequest,
        *,
        on_event: EventHook | None = None,
        timeout: float | None = None,
        stop: threading.Event | None = None,
    ) -> DebatedResult:
        """`--debate`：跑完请求中全部候选的 6 个个股角色并校验返回。"""
        self.preflight()
        lines, error_lines, code = self._invoke(
            ["--debate"],
            stdin_line=request.to_line(),
            timeout=timeout,
            on_event=on_event,
            stop=stop,
        )
        payload, failure = _extract_terminal(lines, "debated")
        if payload is None:
            reason = failure or f"侧车退出码 {code}，没有给出 debated 事件"
            raise SidecarError(f"辩论失败：{reason}{self._stderr_suffix(error_lines)}")
        result = parse_debated(payload)
        verify_debated(result, request)
        return result

    def finalize(
        self,
        request: RunRequest,
        verdicts: Sequence[RiskVerdict],
        *,
        on_event: EventHook | None = None,
        timeout: float | None = None,
        stop: threading.Event | None = None,
    ) -> FinalizedResult:
        """`--finalize`：基于全部辩完的风控结论，执行一次终选并校验返回。"""
        self.preflight()
        envelope = {
            "request": request.payload(),
            "verdicts": [item.payload() for item in verdicts],
        }
        lines, error_lines, code = self._invoke(
            ["--finalize"],
            stdin_line=json.dumps(envelope, ensure_ascii=False, separators=(",", ":")),
            timeout=timeout,
            on_event=on_event,
            stop=stop,
        )
        payload, failure = _extract_terminal(lines, "finalized")
        if payload is None:
            reason = failure or f"侧车退出码 {code}，没有给出 finalized 事件"
            raise SidecarError(f"终选失败：{reason}{self._stderr_suffix(error_lines)}")
        finalized = parse_finalized(payload)
        verify_final(finalized, verdicts, request)
        return finalized

    def _invoke(
        self,
        extra_args: Sequence[str],
        *,
        stdin_line: str | None,
        timeout: float | None,
        on_event: EventHook | None = None,
        stop: threading.Event | None = None,
        need_key: bool = True,
    ) -> tuple[list[str], list[str], int]:
        command = [self._node_command, str(self._entry), *extra_args]
        env = self._environment() if need_key else dict(os.environ)
        cwd = str(self._sidecar_dir) if self._sidecar_dir.is_dir() else None
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise SidecarUnavailable(f"启动侧车失败（{command[0]}）：{exc}") from exc

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        activity = _Activity()
        callback_errors: list[BaseException] = []
        threads = (
            threading.Thread(
                target=_drain,
                args=(process.stdout, stdout_lines, on_event, activity, callback_errors),
                name="sidecar-stdout",
                daemon=True,
            ),
            threading.Thread(
                target=_drain,
                args=(process.stderr, stderr_lines, None, activity),
                name="sidecar-stderr",
                daemon=True,
            ),
        )
        for thread in threads:
            thread.start()

        if process.stdin is not None:
            try:
                if stdin_line is not None:
                    process.stdin.write(stdin_line + "\n")
                process.stdin.close()
            except OSError:
                pass

        stall = self._stall_seconds() if timeout is None else None
        started = time.monotonic()
        reason: str | None = None
        stopped = False
        interrupted = False
        code = -1

        try:
            while True:
                if callback_errors:
                    _kill_process(process)
                    break
                try:
                    code = process.wait(timeout=_WATCH_INTERVAL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    if stop is not None and stop.is_set():
                        stopped = True
                        _kill_process(process)
                        break
                    elapsed = time.monotonic() - started
                    if timeout is not None and elapsed >= timeout:
                        reason = f"超过 {timeout:.0f} 秒仍未结束"
                    elif stall is not None:
                        quiet = activity.quiet_for()
                        if quiet >= stall:
                            reason = f"已经 {quiet:.0f} 秒没有任何输出（判活阈值 {stall:.0f} 秒），看起来卡住了"
                        elif elapsed >= _HARD_CAP_SECONDS:
                            reason = f"跑了 {elapsed / 60:.0f} 分钟仍未结束，超过硬上限"
                    if reason is not None:
                        _kill_process(process)
                        break
        except KeyboardInterrupt:
            interrupted = True
            _kill_process(process)

        for thread in threads:
            thread.join(timeout=_KILL_GRACE_SECONDS)

        if interrupted:
            raise SidecarStopped("收到中断信号，已停止侧车，已完成股票检查点保留。")
        if callback_errors:
            raise callback_errors[0]
        if stopped:
            raise SidecarStopped("侧车已按请求停止，未完成股票不会保存为结论。")
        if reason is not None:
            raise SidecarTimeout(f"侧车{reason}，已杀进程。{self._stderr_suffix(stderr_lines)}")
        return stdout_lines, stderr_lines, code

    @staticmethod
    def _stderr_suffix(error_lines: Sequence[str]) -> str:
        tail = [line.rstrip() for line in error_lines[-_STDERR_TAIL:] if line.strip()]
        if not tail:
            return ""
        return "\n侧车日志：\n  " + "\n  ".join(tail)


def _kill_process(process: subprocess.Popen[str]) -> None:
    process.kill()
    try:
        process.wait(timeout=_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:  # pragma: no cover
        pass


def _extract_terminal(lines: Sequence[str], terminal_type: str) -> tuple[Mapping[str, Any] | None, str | None]:
    payload: Mapping[str, Any] | None = None
    failure: str | None = None
    for line in lines:
        event = parse_event(line)
        if event is None:
            continue
        if event.type == terminal_type:
            payload = event.payload
        elif event.type == "error":
            failure = str(event.payload.get("message", "侧车未给出原因"))
    return payload, failure


class _Activity:
    __slots__ = ("_lock", "_last")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last = time.monotonic()

    def touch(self) -> None:
        with self._lock:
            self._last = time.monotonic()

    def quiet_for(self) -> float:
        with self._lock:
            return time.monotonic() - self._last


def _drain(
    stream: Any,
    sink: list[str],
    on_event: EventHook | None,
    activity: _Activity | None = None,
    callback_errors: list[BaseException] | None = None,
) -> None:
    if stream is None:
        return
    for line in stream:
        sink.append(line)
        if activity is not None:
            activity.touch()
        if on_event is None:
            continue
        try:
            event = parse_event(line)
        except ContractError:
            continue
        if event is not None and event.type not in _TERMINAL_EVENTS:
            try:
                on_event(event)
            except BaseException as exc:
                if callback_errors is None:
                    raise
                callback_errors.append(exc)
                break
    stream.close()
