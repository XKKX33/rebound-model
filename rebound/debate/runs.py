"""运行目录、冻结文件与断点续跑检查点（第 3.8 节）。

目录布局：
    <db_stem>.debate/<variant>-<cadence>/<YYYY-MM-DD>/<input_hash 前 12 位>/
      ├─ request.json           完整请求（无密钥），首次写入后冻结
      ├─ stocks/<ts_code>.json  单只辩完股票检查点（原子写）
      ├─ events.ndjson          侧车进度事件（追加写）
      ├─ usage.json             各次尝试累计用量（原子写）
      ├─ result.json            全部辩论与终选细节，冻结
      └─ final.json             展示用最终名单，冻结
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from rebound.debate.contracts import (
    ContractError,
    RunRequest,
    Usage,
    stock_checkpoint,
)

__all__ = [
    "LEGACY_MODEL_SOURCE",
    "append_event",
    "debate_root",
    "freeze_final",
    "freeze_request",
    "freeze_result",
    "list_runs",
    "load_checkpoints",
    "load_usage",
    "run_dir",
    "save_stock_checkpoint",
    "save_usage",
    "series_name",
]

LEGACY_MODEL_SOURCE = "return_model"
_SAFE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SAFE_CODE = re.compile(r"^[0-9A-Za-z._-]+$")
_SAFE_HASH_PREFIX = re.compile(r"^[0-9a-f]{12}$")


def debate_root(db_path: Path) -> Path:
    path = Path(db_path)
    return path.with_name(f"{path.stem}.debate")


def series_name(variant: str, cadence: str, model_source: str | None = None) -> str:
    """返回 `<db>.debate/` 下的系列目录名：原版模型保持 `<variant>-<cadence>`，新模型按 `<variant>-<cadence>-<model>` 分目录。"""
    if not _SAFE_CODE.fullmatch(variant) or not _SAFE_CODE.fullmatch(cadence):
        raise ContractError(f"非法的版本或频率：{variant!r}/{cadence!r}")
    clean_model = (model_source or "").strip()
    if not clean_model or clean_model == LEGACY_MODEL_SOURCE:
        return f"{variant}-{cadence}"
    if not _SAFE_CODE.fullmatch(clean_model):
        raise ContractError(f"非法的模型名：{clean_model!r}")
    return f"{variant}-{cadence}-{clean_model}"


def run_dir(
    db_path: Path,
    *,
    variant: str,
    cadence: str,
    as_of: str,
    input_hash: str,
    model_source: str | None = None,
) -> Path:
    if not _SAFE_DATE.fullmatch(as_of):
        raise ContractError(f"非法的运行日期：{as_of!r}")
    prefix = input_hash[:12]
    if not _SAFE_HASH_PREFIX.fullmatch(prefix):
        raise ContractError(f"非法的 input_hash 前缀：{prefix!r}")
    return debate_root(db_path) / series_name(variant, cadence, model_source) / as_of / prefix


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _dump_json(value)
    tmp = path.with_name(f"{path.stem}.tmp-{os.getpid()}-{threading.get_ident()}{path.suffix}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _freeze(path: Path, value: Any) -> bool:
    """首次写入冻结；文件已存在返回 False 不覆盖。"""
    text = _dump_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
        return True
    except FileExistsError:
        return False


def freeze_request(rdir: Path, request: RunRequest) -> None:
    """冻结 `request.json`；若已存在则校验核心哈希一致，防止续跑错位。"""
    path = rdir / "request.json"
    if _freeze(path, request.payload()):
        return
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContractError(f"已冻结的 request.json 读不了：{exc}") from exc
    if not isinstance(existing, dict) or existing.get("input_hash") != request.input_hash:
        raise ContractError(f"运行目录 {rdir} 已有不匹配的 request.json，拒绝覆盖")


def freeze_result(rdir: Path, payload: Mapping[str, Any]) -> None:
    if not _freeze(rdir / "result.json", dict(payload)):
        raise ContractError(f"运行目录 {rdir} 已有 result.json，已完成运行不允许改写")


def freeze_final(rdir: Path, payload: Mapping[str, Any]) -> None:
    if not _freeze(rdir / "final.json", dict(payload)):
        raise ContractError(f"运行目录 {rdir} 已有 final.json，已完成运行不允许改写")


def save_stock_checkpoint(rdir: Path, ts_code: str, stock_payload: Mapping[str, Any]) -> None:
    """校验单只股票检查点完整性后原子落盘。"""
    if not _SAFE_CODE.fullmatch(ts_code):
        raise ContractError(f"非法的股票代码：{ts_code!r}")
    cp = stock_checkpoint(stock_payload)
    if cp["verdict"].ts_code != ts_code:
        raise ContractError(f"检查点代码错位：期望 {ts_code}，收到 {cp['verdict'].ts_code}")
    _write_atomic(rdir / "stocks" / f"{ts_code}.json", dict(stock_payload))


def load_checkpoints(rdir: Path, request: RunRequest) -> dict[str, dict[str, Any]]:
    """按请求顺序读出全部已通过校验的单只股票检查点。"""
    stocks_dir = rdir / "stocks"
    if not stocks_dir.is_dir():
        return {}
    allowed = set(request.codes)
    found: dict[str, dict[str, Any]] = {}
    for file in sorted(stocks_dir.glob("*.json")):
        if ".tmp-" in file.name:
            continue
        code = file.stem
        if code not in allowed:
            raise ContractError(f"检查点 {file.name} 不在本次送辩候选范围内")
        try:
            raw = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ContractError(f"检查点 {file} 损坏：{exc}") from exc
        cp = stock_checkpoint(raw)
        if cp["verdict"].ts_code != code:
            raise ContractError(f"检查点 {file.name} 内含代码 {cp['verdict'].ts_code}")
        found[code] = cp
    return {code: found[code] for code in request.codes if code in found}


def save_usage(rdir: Path, usage: Usage) -> None:
    _write_atomic(rdir / "usage.json", usage.payload())


def load_usage(rdir: Path) -> Usage:
    path = rdir / "usage.json"
    if not path.is_file():
        return Usage(0, 0, 0, 0)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContractError(f"用量文件 {path} 损坏：{exc}") from exc
    return Usage.from_payload(raw)


def append_event(rdir: Path, event_payload: Mapping[str, Any]) -> None:
    rdir.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(event_payload), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    with (rdir / "events.ndjson").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def list_runs(
    db_path: Path,
    *,
    variant: str,
    cadence: str,
    as_of: date | None = None,
    model_source: str | None = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    """列出某一天的全部辩论运行；最早完成的一次标为 `official=True`（当日正式结果）。"""
    series_dir = debate_root(db_path) / series_name(variant, cadence, model_source)
    if not series_dir.is_dir():
        return None, []
    if as_of is not None:
        day_str = as_of.isoformat()
    else:
        dated = sorted(
            d.name for d in series_dir.iterdir()
            if d.is_dir() and _SAFE_DATE.fullmatch(d.name) and any(d.iterdir())
        )
        if not dated:
            return None, []
        day_str = dated[-1]

    day_dir = series_dir / day_str
    if not day_dir.is_dir():
        return day_str, []

    completed: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for folder in sorted(day_dir.iterdir()):
        if not folder.is_dir():
            continue
        final_path = folder / "final.json"
        if final_path.is_file():
            final_data = json.loads(final_path.read_text(encoding="utf-8"))
            result_path = folder / "result.json"
            result_data = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
            completed.append(
                {
                    "hash_prefix": folder.name,
                    "run_dir": str(folder),
                    "completed": True,
                    "official": False,
                    "completed_at": str(final_data.get("completed_at") or ""),
                    "final": final_data,
                    "result": result_data,
                }
            )
        elif (folder / "request.json").is_file():
            req_data = json.loads((folder / "request.json").read_text(encoding="utf-8"))
            stocks_dir = folder / "stocks"
            done = (
                sum(1 for f in stocks_dir.glob("*.json") if ".tmp-" not in f.name)
                if stocks_dir.is_dir()
                else 0
            )
            incomplete.append(
                {
                    "hash_prefix": folder.name,
                    "run_dir": str(folder),
                    "completed": False,
                    "official": False,
                    "completed_at": "",
                    "run_id": req_data.get("run_id"),
                    "submitted_count": len(req_data.get("candidates", [])),
                    "debated_count": done,
                }
            )

    completed.sort(key=lambda item: (item["completed_at"], item["hash_prefix"]))
    if completed:
        completed[0]["official"] = True
    return day_str, [*completed, *incomplete]
