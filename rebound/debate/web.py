"""AI 辩论本地 Web UI 服务（仅标准库，默认监听 127.0.0.1）。

提供：
- 历史辩论回放与当日快照演练预览（不花 Token）
- 后台启动 / 断点续跑 / 优雅停止（复用 `service.run_debate` 与 `threading.Event`）
- SSE 实时推送：工作流节点进度、单股 6 步流水态、以及 `role.delta` 逐字思考与正文流
"""

from __future__ import annotations

import json
import mimetypes
import queue
import re
import threading
import time
import webbrowser
from dataclasses import replace
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

import duckdb

from rebound import clock
from rebound import config as app_config
from rebound import params as model_params
from rebound.data.errors import DataError
from rebound.debate import chat, ml_bridge, prompts, runs, service, snapshot
from rebound.debate.contracts import ContractError, Event
from rebound.debate.sidecar import SidecarError, SidecarStopped, resolve_api_key

__all__ = [
    "DebateHub",
    "WEB_UI_DIR",
    "create_server",
    "serve_ui",
]

WEB_UI_DIR = Path(__file__).resolve().parent / "web_ui"
_SAFE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SAFE_HASH = re.compile(r"^[0-9a-f]{12}$")
_HEARTBEAT_SECONDS = 15.0
_MAX_STREAM_CHARS = 80_000


class BusyError(RuntimeError):
    """已有辩论任务在运行，拒绝重复发起。"""


class NotFoundError(RuntimeError):
    """请求的运行或资源不存在。"""


def _safe_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def _api_key_ready(settings: app_config.Settings) -> bool:
    try:
        resolve_api_key(settings)
        return True
    except SidecarError:
        return False


def _read_json_file(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_events_file(path: Path, *, limit: int = 2000) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
    return out[-limit:]


def _list_trading_days(db_path: Path, limit: int = 120) -> list[str]:
    if not db_path.is_file():
        return []
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            "SELECT DISTINCT CAST(trade_date AS VARCHAR) AS d "
            "FROM daily ORDER BY d DESC LIMIT ?",
            [limit],
        ).fetchall()
        return [str(r[0])[:10] for r in rows if r and r[0] is not None]
    except Exception:
        return []
    finally:
        con.close()


class DebateHub:
    """管理本地 Web UI 的历史查询、单任务后台运行与 SSE 实时事件分发。"""

    def __init__(
        self,
        settings: app_config.Settings,
        *,
        default_db: Path | None = None,
        sidecar_factory: Callable[[app_config.Settings], Any] | None = None,
        chat_stream_fn: chat.StreamFn | None = None,
    ) -> None:
        self.settings = settings
        self.default_db = self._pick_default_db(default_db)
        self.sidecar_factory = sidecar_factory
        self.chat_stream_fn = chat_stream_fn
        self._lock = threading.Lock()
        self._subscribers: set[queue.Queue[tuple[str, dict[str, Any]]]] = set()
        self._active: dict[str, Any] | None = None
        self._stop_event: threading.Event | None = None
        self._worker: threading.Thread | None = None
        self._stream_run_key: str | None = None
        self._live_streams: dict[str, dict[str, Any]] = {}
        self._active_roles: dict[str, dict[str, Any]] = {}
        self._chat_stores: dict[Path, chat.ChatStore] = {}
        self._dry_run_cache: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

    def _pick_default_db(self, explicit: Path | None) -> Path:
        if explicit is not None:
            return Path(explicit).resolve()
        configured = self.settings.data.db_path.resolve()
        if configured.is_file():
            return configured
        data_dir = self.settings.root / "data"
        if data_dir.is_dir():
            dbs = sorted(data_dir.glob("*.duckdb"), key=lambda p: p.stat().st_mtime, reverse=True)
            if dbs:
                return dbs[0].resolve()
        return configured

    def list_dbs(self) -> list[dict[str, Any]]:
        seen: dict[str, Path] = {}
        if self.default_db.is_file():
            seen[str(self.default_db).lower()] = self.default_db
        configured = self.settings.data.db_path.resolve()
        if configured.is_file():
            seen.setdefault(str(configured).lower(), configured)
        data_dir = self.settings.root / "data"
        if data_dir.is_dir():
            for path in sorted(data_dir.glob("*.duckdb")):
                resolved = path.resolve()
                seen.setdefault(str(resolved).lower(), resolved)
        items: list[dict[str, Any]] = []
        for path in seen.values():
            try:
                label = str(path.relative_to(self.settings.root)).replace("\\", "/")
            except ValueError:
                label = path.name
            items.append({"path": str(path), "label": label, "name": path.name})
        return items

    def resolve_db(self, raw: str | None) -> Path:
        if not raw or not raw.strip():
            return self.default_db
        allowed = {str(Path(item["path"]).resolve()).lower(): Path(item["path"]).resolve() for item in self.list_dbs()}
        candidate = Path(raw.strip())
        if not candidate.is_absolute():
            candidate = (self.settings.root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        key = str(candidate).lower()
        if key in allowed:
            return allowed[key]
        raise ValueError(f"不允许或不存在的数据库路径：{raw}")

    def _resolve_vc(self, variant: str | None, cadence: str | None) -> tuple[str, str]:
        v = (variant or self.settings.model.variant).strip()
        c = (cadence or self.settings.model.cadence).strip()
        if v not in model_params.VARIANTS:
            raise ValueError(f"不支持的股票池版本：{v}")
        if c not in model_params.CADENCES:
            raise ValueError(f"不支持的重训频率：{c}")
        return v, c

    def _resolve_model_source(self, raw: str | None) -> str:
        return ml_bridge.validate_model_source(raw)

    def _run_key(
        self,
        db_path: Path,
        variant: str,
        cadence: str,
        as_of: str,
        hash_prefix: str,
        model_source: str | None = None,
    ) -> str:
        series = runs.series_name(variant, cadence, model_source)
        return f"{str(db_path).lower()}|{series}|{as_of}|{hash_prefix}"

    def list_all_runs(
        self,
        db_path: Path,
        *,
        variant: str,
        cadence: str,
        model_source: str | None = None,
    ) -> list[dict[str, Any]]:
        ms = self._resolve_model_source(model_source)
        series_dir = runs.debate_root(db_path) / runs.series_name(variant, cadence, ms)
        if not series_dir.is_dir():
            return []
        dates = sorted(
            (d.name for d in series_dir.iterdir() if d.is_dir() and _SAFE_DATE.fullmatch(d.name)),
            reverse=True,
        )
        summaries: list[dict[str, Any]] = []
        for day_str in dates:
            _, entries = runs.list_runs(
                db_path, variant=variant, cadence=cadence, as_of=date.fromisoformat(day_str), model_source=ms
            )
            for entry in entries:
                folder = Path(entry["run_dir"])
                req = _read_json_file(folder / "request.json") or {}
                usage = _read_json_file(folder / "usage.json") or {}
                llm = req.get("llm") if isinstance(req.get("llm"), dict) else {}
                limits = req.get("limits") if isinstance(req.get("limits"), dict) else {}
                ctx = req.get("context") if isinstance(req.get("context"), dict) else {}
                final = entry.get("final") if entry.get("completed") else None
                summaries.append(
                    {
                        "as_of": day_str,
                        "variant": variant,
                        "cadence": cadence,
                        "model_source": ms,
                        "model_label": ctx.get("model_label") or ml_bridge.ML_MODEL_META.get(ms, {}).get("name", ""),
                        "feature_set": ctx.get("feature_set") or ("F28" if ms == runs.LEGACY_MODEL_SOURCE else "F69"),
                        "hash_prefix": entry["hash_prefix"],
                        "run_id": (final or req).get("run_id", ""),
                        "completed": bool(entry["completed"]),
                        "official": bool(entry["official"]),
                        "completed_at": entry.get("completed_at") or "",
                        "submitted_count": (
                            final.get("submitted_count") if final else entry.get("submitted_count", len(req.get("candidates", [])))
                        ),
                        "debated_count": (
                            final.get("debated_count") if final else entry.get("debated_count", 0)
                        ),
                        "failed_count": final.get("failed_count", 0) if final else 0,
                        "final_count": final.get("final_count", 0) if final else 0,
                        "overlap_count": final.get("overlap_count") if final else None,
                        "model": llm.get("id", ""),
                        "reasoning_effort": llm.get("reasoning_effort", ""),
                        "concurrency": limits.get("concurrency"),
                        "usage": (final.get("usage") if final else usage) or {},
                    }
                )
        return summaries

    def active_snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            if self._active is None:
                return None
            return {
                **self._active,
                "active_roles": {k: dict(v) for k, v in self._active_roles.items()},
            }

    def meta(
        self,
        *,
        db: str | None = None,
        variant: str | None = None,
        cadence: str | None = None,
        model_source: str | None = None,
    ) -> dict[str, Any]:
        db_path = self.resolve_db(db)
        v, c = self._resolve_vc(variant, cadence)
        ms = self._resolve_model_source(model_source)
        dbs = self.list_dbs()
        run_list = self.list_all_runs(db_path, variant=v, cadence=c, model_source=ms)
        if ml_bridge.is_ml_model(ms) and ml_bridge.is_ml_available():
            days = ml_bridge.get_ml_dates(ms, limit=260)
        else:
            days = _list_trading_days(db_path)
        for item in run_list:
            if item["as_of"] not in days:
                days.append(item["as_of"])
        days = sorted(set(days), reverse=True)

        sources = ml_bridge.list_model_sources()
        for src in sources:
            sid = src["id"]
            src["run_count"] = len(self.list_all_runs(db_path, variant=v, cadence=c, model_source=sid))

        return {
            "dbs": dbs,
            "selected_db": str(db_path),
            "variant": v,
            "cadence": c,
            "model_source": ms,
            "ml_available": ml_bridge.is_ml_available(),
            "model_sources": sources,
            "variants": list(model_params.VARIANTS),
            "cadences": list(model_params.CADENCES),
            "config": {
                "model": self.settings.debate.llm.model,
                "base_url": self.settings.debate.llm.base_url,
                "reasoning_effort": self.settings.debate.llm.reasoning_effort,
                "temperature": self.settings.debate.llm.temperature,
                "max_tokens": self.settings.debate.llm.max_tokens,
                "candidates": self.settings.debate.candidates,
                "min_debated": self.settings.debate.min_debated,
                "bars": self.settings.debate.bars,
                "concurrency": self.settings.debate.concurrency,
                "call_timeout_seconds": self.settings.debate.call_timeout_seconds,
                "final": self.settings.pick.top_n,
                "max_per_industry": self.settings.pick.max_per_industry,
                "api_key_env": self.settings.debate.llm.api_key_env,
                "api_key_configured": _api_key_ready(self.settings),
            },
            "trading_days": days,
            "runs": run_list,
            "active": self.active_snapshot(),
        }

    def get_run_detail(
        self,
        *,
        db: str | None,
        variant: str | None,
        cadence: str | None,
        as_of: str,
        hash_prefix: str,
        model_source: str | None = None,
    ) -> dict[str, Any]:
        if not _SAFE_DATE.fullmatch(as_of) or not _SAFE_HASH.fullmatch(hash_prefix):
            raise ValueError("非法的运行日期或哈希前缀")
        db_path = self.resolve_db(db)
        v, c = self._resolve_vc(variant, cadence)
        ms = self._resolve_model_source(model_source)
        folder = runs.debate_root(db_path) / runs.series_name(v, c, ms) / as_of / hash_prefix
        if not folder.is_dir():
            raise NotFoundError(f"运行目录不存在：{as_of}/{hash_prefix}")

        request_data = _read_json_file(folder / "request.json")
        if request_data is None:
            raise NotFoundError("运行目录缺少 request.json")
        usage_data = _read_json_file(folder / "usage.json") or {
            "calls": 0,
            "failed_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        result_data = _read_json_file(folder / "result.json")
        final_data = _read_json_file(folder / "final.json")
        events_data = _read_events_file(folder / "events.ndjson")

        stocks: dict[str, Any] = {}
        stocks_dir = folder / "stocks"
        if stocks_dir.is_dir():
            for file in sorted(stocks_dir.glob("*.json")):
                if ".tmp-" in file.name:
                    continue
                loaded = _read_json_file(file)
                if isinstance(loaded, dict):
                    stocks[file.stem] = loaded

        # 从同一天列表里核对是否为当日正式结果
        _, day_entries = runs.list_runs(
            db_path, variant=v, cadence=c, as_of=date.fromisoformat(as_of), model_source=ms
        )
        official = any(e["hash_prefix"] == hash_prefix and e["official"] for e in day_entries)

        run_key = self._run_key(db_path, v, c, as_of, hash_prefix, ms)
        with self._lock:
            is_running = bool(
                self._active
                and self._active.get("running")
                and self._stream_run_key == run_key
            )
            live_streams = (
                {k: dict(val) for k, val in self._live_streams.items()}
                if self._stream_run_key == run_key
                else {}
            )
            active_roles = (
                {k: dict(val) for k, val in self._active_roles.items()}
                if is_running
                else {}
            )

        return {
            "db": str(db_path),
            "as_of": as_of,
            "variant": v,
            "cadence": c,
            "model_source": ms,
            "hash_prefix": hash_prefix,
            "completed": final_data is not None,
            "official": official,
            "is_running": is_running,
            "request": request_data,
            "stocks": stocks,
            "result": result_data,
            "final": final_data,
            "usage": (final_data.get("usage") if final_data else usage_data) or usage_data,
            "events": events_data,
            "live_streams": live_streams,
            "active_roles": active_roles,
        }

    def dry_run(
        self,
        *,
        db: str | None,
        as_of: str,
        variant: str | None = None,
        cadence: str | None = None,
        model_source: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not _SAFE_DATE.fullmatch(as_of):
            raise ValueError(f"非法日期：{as_of}")
        db_path = self.resolve_db(db)
        v, c = self._resolve_vc(variant, cadence)
        ms = self._resolve_model_source(model_source)
        eff = self._apply_overrides(overrides)
        snap = snapshot.build_request(
            db_path,
            settings=eff,
            as_of=date.fromisoformat(as_of),
            variant=v,
            cadence=c,
            model_source=ms,
        )
        out = {
            "db": str(db_path),
            "as_of": snap.request.as_of,
            "variant": snap.request.context.variant,
            "cadence": c,
            "model_source": ms,
            "hash_prefix": snap.request.input_hash[:12],
            "request": snap.request.payload(),
        }
        with self._lock:
            self._dry_run_cache[(str(db_path).lower(), snap.request.context.variant, c, ms, snap.request.as_of)] = out
        return out

    def _apply_overrides(self, overrides: Mapping[str, Any] | None) -> app_config.Settings:
        if not overrides:
            return self.settings
        debate = self.settings.debate
        llm = debate.llm
        if "reasoning_effort" in overrides and overrides["reasoning_effort"] is not None:
            effort = str(overrides["reasoning_effort"]).strip()
            if effort not in app_config.REASONING_EFFORTS:
                raise ValueError(f"reasoning_effort 必须是 {sorted(app_config.REASONING_EFFORTS)} 之一")
            llm = replace(llm, reasoning_effort=effort)
        if "concurrency" in overrides and overrides["concurrency"] is not None:
            conc = int(overrides["concurrency"])
            if not 1 <= conc <= 16:
                raise ValueError("concurrency 必须在 1~16 之间")
            debate = replace(debate, concurrency=conc)
        if "candidates" in overrides and overrides["candidates"] is not None:
            cands = int(overrides["candidates"])
            if not self.settings.pick.top_n <= cands <= 50:
                raise ValueError(f"candidates 必须在 {self.settings.pick.top_n}~50 之间")
            min_deb = min(debate.min_debated, cands)
            debate = replace(debate, candidates=cands, min_debated=min_deb)
        debate = replace(debate, llm=llm)
        return replace(self.settings, debate=debate)

    def start_run(
        self,
        *,
        db: str | None,
        as_of: str,
        variant: str | None = None,
        cadence: str | None = None,
        model_source: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not _SAFE_DATE.fullmatch(as_of):
            raise ValueError(f"非法日期：{as_of}")
        db_path = self.resolve_db(db)
        v, c = self._resolve_vc(variant, cadence)
        ms = self._resolve_model_source(model_source)
        if ml_bridge.is_ml_model(ms):
            v = "nocap"
        eff = self._apply_overrides(overrides)

        stop_event = threading.Event()
        started_at = clock.beijing_now().isoformat(timespec="seconds")
        with self._lock:
            if self._active and self._active.get("running"):
                raise BusyError("已有辩论任务正在运行，请先等待完成或点击停止。")
            self._stop_event = stop_event
            self._active = {
                "running": True,
                "stage": "preparing",
                "db": str(db_path),
                "as_of": as_of,
                "variant": v,
                "cadence": c,
                "model_source": ms,
                "run_id": "",
                "hash_prefix": "",
                "started_at": started_at,
                "stopping": False,
                "error": None,
                "model": eff.debate.llm.model,
                "reasoning_effort": eff.debate.llm.reasoning_effort,
                "concurrency": eff.debate.concurrency,
            }
            self._stream_run_key = None
            self._live_streams = {}
            self._active_roles = {}
            active_copy = dict(self._active)

        self._publish("job.started", active_copy)

        def worker() -> None:
            try:
                client = self.sidecar_factory(eff) if self.sidecar_factory is not None else None
                out = service.run_debate(
                    db_path,
                    settings=eff,
                    as_of=date.fromisoformat(as_of),
                    variant=v,
                    cadence=c,
                    model_source=ms,
                    sidecar_client=client,
                    on_event=self._handle_service_event,
                    stop=stop_event,
                )
                final = out["final"]
                hash_prefix = str(final.get("input_hash", ""))[:12]
                with self._lock:
                    if self._active is not None:
                        self._active["running"] = False
                        self._active["stage"] = "completed"
                        self._active["run_id"] = final.get("run_id", "")
                        self._active["hash_prefix"] = hash_prefix
                        self._active_roles = {}
                    snapshot_active = dict(self._active) if self._active else {}
                self._publish(
                    "job.completed",
                    {
                        **snapshot_active,
                        "cached": bool(out.get("cached")),
                        "run_dir": out.get("run_dir", ""),
                        "final": final,
                    },
                )
            except SidecarStopped as exc:
                with self._lock:
                    if self._active is not None:
                        self._active["running"] = False
                        self._active["stage"] = "stopped"
                        self._active["error"] = str(exc)
                        self._active_roles = {}
                    snapshot_active = dict(self._active) if self._active else {}
                self._publish("job.stopped", snapshot_active)
            except (DataError, ContractError, ValueError, OSError, RuntimeError) as exc:
                with self._lock:
                    if self._active is not None:
                        self._active["running"] = False
                        self._active["stage"] = "error"
                        self._active["error"] = str(exc)
                        self._active_roles = {}
                    snapshot_active = dict(self._active) if self._active else {}
                self._publish("job.error", snapshot_active)

        thread = threading.Thread(target=worker, name="debate-web-worker", daemon=True)
        with self._lock:
            self._worker = thread
        thread.start()
        return active_copy

    def stop_run(self) -> dict[str, Any]:
        with self._lock:
            if not self._active or not self._active.get("running") or self._stop_event is None:
                return {"stopping": False, "message": "当前没有正在运行的辩论任务"}
            self._active["stopping"] = True
            self._stop_event.set()
            payload = dict(self._active)
        self._publish("job.stopping", payload)
        return {"stopping": True, "active": payload}

    def _handle_service_event(self, event: Event) -> None:
        etype = event.type
        data = dict(event.payload)
        now = time.time()

        with self._lock:
            if etype == "run.prepared" and self._active is not None:
                hp = str(data.get("hash_prefix", ""))
                self._active["run_id"] = str(data.get("run_id", ""))
                self._active["hash_prefix"] = hp
                self._active["stage"] = "debating"
                self._stream_run_key = self._run_key(
                    Path(self._active["db"]),
                    self._active["variant"],
                    self._active["cadence"],
                    self._active["as_of"],
                    hp,
                    self._active.get("model_source"),
                )
            elif etype == "role.started":
                code = str(data.get("ts_code", "-"))
                role = str(data.get("role", ""))
                attempt = int(data.get("attempt", 1))
                if role == "final_pick" and self._active is not None:
                    self._active["stage"] = "finalizing"
                self._active_roles[code] = {"role": role, "attempt": attempt, "started_at": now}
                self._live_streams[f"{code}::{role}"] = {
                    "ts_code": code,
                    "role": role,
                    "attempt": attempt,
                    "thinking": "",
                    "text": "",
                    "started_at": now,
                    "completed": False,
                }
            elif etype == "role.delta":
                code = str(data.get("ts_code", "-"))
                role = str(data.get("role", ""))
                channel = str(data.get("channel", "text"))
                delta = str(data.get("delta", ""))
                key = f"{code}::{role}"
                entry = self._live_streams.get(key)
                if entry is None:
                    entry = {
                        "ts_code": code,
                        "role": role,
                        "attempt": int(data.get("attempt", 1)),
                        "thinking": "",
                        "text": "",
                        "started_at": now,
                        "completed": False,
                    }
                    self._live_streams[key] = entry
                if channel in ("thinking", "text") and len(entry[channel]) < _MAX_STREAM_CHARS:
                    entry[channel] += delta
            elif etype == "role.completed":
                code = str(data.get("ts_code", "-"))
                role = str(data.get("role", ""))
                entry = self._live_streams.get(f"{code}::{role}")
                if entry is not None:
                    entry["completed"] = True
                if role in ("risk_chair", "final_pick"):
                    self._active_roles.pop(code, None)
            elif etype in ("stock.completed", "stock.failed"):
                code = str(data.get("ts_code", ""))
                self._active_roles.pop(code, None)
            elif etype == "run.frozen":
                self._active_roles.clear()

            if self._active is not None:
                data.setdefault("hash_prefix", self._active.get("hash_prefix", ""))
                data.setdefault("as_of", self._active.get("as_of", ""))
                data.setdefault("variant", self._active.get("variant", ""))
                data.setdefault("cadence", self._active.get("cadence", ""))
                data.setdefault("model_source", self._active.get("model_source", ""))

        self._publish(etype, data)

    def subscribe(self) -> queue.Queue[tuple[str, dict[str, Any]]]:
        q: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue(maxsize=1024)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: queue.Queue[tuple[str, dict[str, Any]]]) -> None:
        with self._lock:
            self._subscribers.discard(q)

    def _publish(self, event_name: str, payload: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait((event_name, payload))
            except queue.Full:
                # 慢消费者丢弃旧的 delta 保活，不阻塞后端辩论线程
                try:
                    q.get_nowait()
                    q.put_nowait((event_name, payload))
                except queue.Empty:
                    pass

    # ------------------------------------------------------------------
    # AI 投研复盘对话（档位 B：只读工具 + 落盘持久化 + 流式思考）
    # ------------------------------------------------------------------

    def chat_store(
        self,
        *,
        db: str | None = None,
        variant: str | None = None,
        cadence: str | None = None,
        as_of: str | None = None,
        hash_prefix: str | None = None,
        model_source: str | None = None,
    ) -> chat.ChatStore:
        db_path = self.resolve_db(db)
        v, c = self._resolve_vc(variant, cadence)
        ms = self._resolve_model_source(model_source)
        store = chat.ChatStore.for_run(
            db_path,
            variant=v,
            cadence=c,
            as_of=(as_of or "").strip(),
            hash_prefix=(hash_prefix or "").strip() or None,
            model_source=ms,
        )
        key = store.path.resolve()
        with self._lock:
            existing = self._chat_stores.get(key)
            if existing is None:
                self._chat_stores[key] = store
                return store
            return existing

    def list_chat_sessions(
        self,
        *,
        db: str | None = None,
        variant: str | None = None,
        cadence: str | None = None,
        as_of: str | None = None,
        hash_prefix: str | None = None,
        model_source: str | None = None,
    ) -> dict[str, Any]:
        store = self.chat_store(
            db=db, variant=variant, cadence=cadence, as_of=as_of, hash_prefix=hash_prefix, model_source=model_source
        )
        return {
            "roles": list(chat.CHAT_ROLE_PRESETS),
            "tools": [{"name": k, "label": v} for k, v in chat.TOOL_LABELS.items()],
            "sessions": store.list_sessions(),
            "store_name": store.path.name,
        }

    def create_chat_session(
        self,
        *,
        db: str | None = None,
        variant: str | None = None,
        cadence: str | None = None,
        as_of: str | None = None,
        hash_prefix: str | None = None,
        model_source: str | None = None,
        role_preset: str = "research_copilot",
        title: str = "新对话",
    ) -> dict[str, Any]:
        store = self.chat_store(
            db=db, variant=variant, cadence=cadence, as_of=as_of, hash_prefix=hash_prefix, model_source=model_source
        )
        return store.create_session(role_preset=role_preset, title=title)

    def delete_chat_session(
        self,
        *,
        db: str | None = None,
        variant: str | None = None,
        cadence: str | None = None,
        as_of: str | None = None,
        hash_prefix: str | None = None,
        model_source: str | None = None,
        session_id: str,
    ) -> bool:
        if not session_id or not session_id.strip():
            raise ValueError("缺少 session_id")
        store = self.chat_store(
            db=db, variant=variant, cadence=cadence, as_of=as_of, hash_prefix=hash_prefix, model_source=model_source
        )
        return store.delete_session(session_id.strip())

    def _resolve_run_detail_optional(
        self,
        *,
        db: str | None = None,
        variant: str | None = None,
        cadence: str | None = None,
        as_of: str | None = None,
        hash_prefix: str | None = None,
        model_source: str | None = None,
    ) -> tuple[Path, str, str, str, str, str, dict[str, Any] | None]:
        db_path = self.resolve_db(db)
        v, c = self._resolve_vc(variant, cadence)
        ms = self._resolve_model_source(model_source)
        if ml_bridge.is_ml_model(ms):
            v = "nocap"
        clean_date = (as_of or "").strip()
        clean_hash = (hash_prefix or "").strip()
        run_detail: dict[str, Any] | None = None

        if _SAFE_DATE.fullmatch(clean_date) and _SAFE_HASH.fullmatch(clean_hash):
            try:
                run_detail = self.get_run_detail(
                    db=str(db_path),
                    variant=v,
                    cadence=c,
                    as_of=clean_date,
                    hash_prefix=clean_hash,
                    model_source=ms,
                )
            except NotFoundError:
                run_detail = None

        if run_detail is None and _SAFE_DATE.fullmatch(clean_date):
            with self._lock:
                cached_dry = self._dry_run_cache.get((str(db_path).lower(), v, c, ms, clean_date))
            if cached_dry is not None:
                run_detail = {
                    "request": cached_dry.get("request"),
                    "stocks": {},
                    "result": None,
                    "final": None,
                }
        return db_path, v, c, ms, clean_date, clean_hash, run_detail

    def get_prompts(
        self,
        *,
        db: str | None = None,
        variant: str | None = None,
        cadence: str | None = None,
        as_of: str | None = None,
        hash_prefix: str | None = None,
        model_source: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        """返回 7 个辩论角色与对话角色的预置提示词及完整 System Prompt。"""
        _db_path, v, c, ms, clean_date, clean_hash, run_detail = self._resolve_run_detail_optional(
            db=db, variant=variant, cadence=cadence, as_of=as_of, hash_prefix=hash_prefix, model_source=model_source
        )
        frozen_request = run_detail.get("request") if isinstance(run_detail, dict) else None
        out = prompts.inspect_prompts(
            settings=self.settings,
            frozen_request=frozen_request,
            source_preference=(source or "auto"),
        )
        out["as_of"] = clean_date
        out["hash_prefix"] = clean_hash
        out["variant"] = v
        out["cadence"] = c
        out["model_source"] = ms
        return out

    def get_prompt_payload(
        self,
        *,
        db: str | None = None,
        variant: str | None = None,
        cadence: str | None = None,
        as_of: str | None = None,
        hash_prefix: str | None = None,
        model_source: str | None = None,
        role: str,
        ts_code: str | None = None,
    ) -> dict[str, Any]:
        """构造指定角色与个股实际送入大模型的 User JSON（以及对话常驻上下文预览）。"""
        clean_role = (role or "").strip()
        if not clean_role:
            raise ValueError("缺少 role 参数")

        if clean_role.startswith("chat:"):
            preset = clean_role[len("chat:") :].strip() or "research_copilot"
            toolbox = self.build_chat_toolbox(
                db=db,
                variant=variant,
                cadence=cadence,
                as_of=as_of,
                hash_prefix=hash_prefix,
                model_source=model_source,
            )
            attached = [ts_code.strip()] if (ts_code and ts_code.strip()) else []
            ctx_snapshot = toolbox.global_context_snapshot(attached)
            full_prompt = chat.build_chat_system_prompt(
                self.settings, role_preset=preset, context_snapshot=ctx_snapshot
            )
            wire = json.dumps(ctx_snapshot, ensure_ascii=False, separators=(",", ":"))
            return {
                "role": clean_role,
                "ts_code": attached[0] if attached else "",
                "stock_name": "对话常驻上下文",
                "ready": True,
                "missing_deps": [],
                "top_keys": list(ctx_snapshot.keys()),
                "char_count": len(wire),
                "est_tokens": prompts.estimate_tokens(wire),
                "payload": ctx_snapshot,
                "full_system_prompt": full_prompt,
                "full_system_prompt_chars": len(full_prompt),
                "full_system_prompt_est_tokens": prompts.estimate_tokens(full_prompt),
            }

        _db_path, _v, _c, _ms, _clean_date, _clean_hash, run_detail = self._resolve_run_detail_optional(
            db=db, variant=variant, cadence=cadence, as_of=as_of, hash_prefix=hash_prefix, model_source=model_source
        )
        request_obj = run_detail.get("request") if isinstance(run_detail, dict) else None
        if not isinstance(request_obj, Mapping):
            raise NotFoundError("尚未加载任何运行快照或演练数据，无法生成单股实际 User JSON")
        return prompts.build_role_user_payload(
            clean_role,
            request_obj,
            ts_code=ts_code,
            all_stocks=run_detail.get("stocks") if isinstance(run_detail, dict) else None,
            result_data=run_detail.get("result") if isinstance(run_detail, dict) else None,
        )

    def build_chat_toolbox(
        self,
        *,
        db: str | None = None,
        variant: str | None = None,
        cadence: str | None = None,
        as_of: str | None = None,
        hash_prefix: str | None = None,
        model_source: str | None = None,
    ) -> chat.ChatToolbox:
        db_path, v, c, ms, clean_date, clean_hash, run_detail = self._resolve_run_detail_optional(
            db=db, variant=variant, cadence=cadence, as_of=as_of, hash_prefix=hash_prefix, model_source=model_source
        )

        return chat.ChatToolbox(
            db_path,
            variant=v,
            cadence=c,
            as_of=clean_date,
            hash_prefix=clean_hash,
            model_source=ms,
            run_detail=run_detail,
        )

    def prepare_chat_turn(
        self,
        *,
        db: str | None = None,
        variant: str | None = None,
        cadence: str | None = None,
        as_of: str | None = None,
        hash_prefix: str | None = None,
        model_source: str | None = None,
        session_id: str | None = None,
        message: str,
        role_preset: str = "research_copilot",
        attached_codes: Sequence[str] = (),
        reasoning_effort: str | None = None,
        stop_event: threading.Event | None = None,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """校验参数并返回对话 SSE 事件迭代器（便于在发送 HTTP 200 前拦截 400 错误）。"""
        text_in = (message or "").strip()
        if not text_in:
            raise ValueError("消息内容不能为空")
        eff = self._apply_overrides({"reasoning_effort": reasoning_effort} if reasoning_effort else None)
        store = self.chat_store(
            db=db, variant=variant, cadence=cadence, as_of=as_of, hash_prefix=hash_prefix, model_source=model_source
        )
        toolbox = self.build_chat_toolbox(
            db=db, variant=variant, cadence=cadence, as_of=as_of, hash_prefix=hash_prefix, model_source=model_source
        )
        return chat.run_chat_turn(
            settings=eff,
            store=store,
            toolbox=toolbox,
            session_id=(session_id or "").strip() or None,
            user_message=text_in,
            role_preset=(role_preset or "research_copilot").strip(),
            attached_codes=list(attached_codes or ()),
            stop_event=stop_event,
            stream_fn=self.chat_stream_fn,
        )

    # ------------------------------------------------------------------
    # 多模型对比与归因只读查询（XGBoost F69 / 随机森林 F69 / LSTM S+F69）
    # ------------------------------------------------------------------

    def get_ml_overview(self) -> dict[str, Any]:
        out = ml_bridge.get_leaderboard()
        out["feature_catalog"] = ml_bridge.get_feature_catalog()
        return out

    def get_ml_nav(
        self,
        *,
        rule: str | None = None,
        period: str | None = None,
        start_year: int | None = None,
    ) -> dict[str, Any]:
        return ml_bridge.get_nav_compare(
            rule=(rule or ml_bridge.MAIN_BACKTEST_RULE).strip(),
            period=(period or "test").strip(),
            start_year=start_year,
        )

    def get_ml_consensus(
        self,
        *,
        as_of: str | None = None,
        top_n: int = 6,
        with_outcome: bool = False,
    ) -> dict[str, Any]:
        return ml_bridge.get_day_consensus(
            (as_of or "").strip(),
            top_n=max(1, min(20, int(top_n))),
            max_per_industry=self.settings.pick.max_per_industry,
            with_outcome=with_outcome,
        )

    def get_ml_explain(
        self,
        *,
        as_of: str,
        ts_code: str,
        model_id: str | None = None,
        top: int = 8,
    ) -> dict[str, Any]:
        if not as_of or not ts_code:
            raise ValueError("缺少 as_of 或 ts_code 参数")
        return ml_bridge.get_stock_explain_all(
            as_of.strip(),
            ts_code.strip(),
            model_id=(model_id or "").strip() or None,
            top=max(1, min(8, int(top))),
            top_n=self.settings.pick.top_n,
            max_per_industry=self.settings.pick.max_per_industry,
        )


def _make_handler(hub: DebateHub) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
            body = _safe_json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_static(self, rel_path: str) -> None:
            clean = rel_path.lstrip("/") or "index.html"
            target = (WEB_UI_DIR / clean).resolve()
            if target != WEB_UI_DIR and WEB_UI_DIR not in target.parents:
                self._send_json({"error": "非法静态资源路径"}, HTTPStatus.FORBIDDEN)
                return
            if not target.is_file():
                self._send_json({"error": "资源不存在"}, HTTPStatus.NOT_FOUND)
                return
            body = target.read_bytes()
            ctype, _ = mimetypes.guess_type(target.name)
            if target.suffix == ".js":
                ctype = "application/javascript"
            elif target.suffix == ".css":
                ctype = "text/css"
            elif target.suffix == ".html":
                ctype = "text/html"
            ctype = ctype or "application/octet-stream"
            if ctype.startswith(("text/", "application/javascript")):
                ctype = f"{ctype}; charset=utf-8"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _read_body_json(self) -> dict[str, Any]:
            length_raw = self.headers.get("Content-Length", "0")
            try:
                length = int(length_raw)
            except ValueError as exc:
                raise ValueError("Content-Length 非法") from exc
            if length <= 0:
                return {}
            if length > 1_048_576:
                raise ValueError("请求体过大")
            raw = self.rfile.read(length)
            parsed = json.loads(raw.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("请求体必须是 JSON 对象")
            return parsed

        def _handle_sse(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            q = hub.subscribe()
            try:
                init_payload = {"active": hub.active_snapshot()}
                self._write_sse("status", init_payload)
                while True:
                    try:
                        event_name, payload = q.get(timeout=_HEARTBEAT_SECONDS)
                        self._write_sse(event_name, payload)
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return
            finally:
                hub.unsubscribe(q)

        def _write_sse(self, event_name: str, payload: dict[str, Any]) -> None:
            encoded = json.dumps(
                {"type": event_name, "payload": payload},
                ensure_ascii=False,
                default=str,
            )
            frame = f"event: {event_name}\ndata: {encoded}\n\n".encode("utf-8")
            self.wfile.write(frame)
            self.wfile.flush()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            arg = lambda key: query.get(key, [None])[0]

            try:
                if path == "/" or path == "/index.html":
                    self._send_static("index.html")
                    return
                if path.startswith("/static/"):
                    self._send_static(path[len("/static/"):])
                    return
                if path == "/api/meta":
                    self._send_json(
                        hub.meta(
                            db=arg("db"),
                            variant=arg("variant"),
                            cadence=arg("cadence"),
                            model_source=arg("model_source"),
                        )
                    )
                    return
                if path == "/api/run":
                    as_of = arg("as_of")
                    hash_prefix = arg("hash")
                    if not as_of or not hash_prefix:
                        self._send_json({"error": "缺少 as_of 或 hash 参数"}, HTTPStatus.BAD_REQUEST)
                        return
                    detail = hub.get_run_detail(
                        db=arg("db"),
                        variant=arg("variant"),
                        cadence=arg("cadence"),
                        as_of=as_of,
                        hash_prefix=hash_prefix,
                        model_source=arg("model_source"),
                    )
                    self._send_json(detail)
                    return
                if path == "/api/stream":
                    self._handle_sse()
                    return
                if path == "/api/prompts":
                    self._send_json(
                        hub.get_prompts(
                            db=arg("db"),
                            variant=arg("variant"),
                            cadence=arg("cadence"),
                            as_of=arg("as_of"),
                            hash_prefix=arg("hash"),
                            model_source=arg("model_source"),
                            source=arg("source"),
                        )
                    )
                    return
                if path == "/api/prompts/payload":
                    role_arg = arg("role")
                    if not role_arg:
                        self._send_json({"error": "缺少 role 参数"}, HTTPStatus.BAD_REQUEST)
                        return
                    self._send_json(
                        hub.get_prompt_payload(
                            db=arg("db"),
                            variant=arg("variant"),
                            cadence=arg("cadence"),
                            as_of=arg("as_of"),
                            hash_prefix=arg("hash"),
                            model_source=arg("model_source"),
                            role=role_arg,
                            ts_code=arg("ts_code"),
                        )
                    )
                    return
                if path == "/api/ml/overview":
                    self._send_json(hub.get_ml_overview())
                    return
                if path == "/api/ml/nav":
                    sy_raw = (arg("start_year") or "").strip()
                    start_year_val: int | None = None
                    if sy_raw:
                        try:
                            start_year_val = int(sy_raw)
                        except ValueError as exc:
                            raise ValueError(f"非法起步年份：{sy_raw}") from exc
                    self._send_json(
                        hub.get_ml_nav(
                            rule=arg("rule"),
                            period=arg("period"),
                            start_year=start_year_val,
                        )
                    )
                    return
                if path == "/api/ml/consensus":
                    top_raw = arg("top_n")
                    top_n = int(top_raw) if top_raw and top_raw.isdigit() else 6
                    with_out = (arg("with_outcome") or "").strip().lower() in ("1", "true", "yes")
                    self._send_json(
                        hub.get_ml_consensus(as_of=arg("as_of"), top_n=top_n, with_outcome=with_out)
                    )
                    return
                if path == "/api/ml/explain":
                    as_of_arg = arg("as_of") or ""
                    code_arg = arg("ts_code") or ""
                    if not as_of_arg or not code_arg:
                        self._send_json({"error": "缺少 as_of 或 ts_code 参数"}, HTTPStatus.BAD_REQUEST)
                        return
                    top_raw = arg("top")
                    top_k = int(top_raw) if top_raw and top_raw.isdigit() else 8
                    self._send_json(
                        hub.get_ml_explain(
                            as_of=as_of_arg,
                            ts_code=code_arg,
                            model_id=arg("model_id"),
                            top=top_k,
                        )
                    )
                    return
                if path == "/api/chat/sessions":
                    self._send_json(
                        hub.list_chat_sessions(
                            db=arg("db"),
                            variant=arg("variant"),
                            cadence=arg("cadence"),
                            as_of=arg("as_of"),
                            hash_prefix=arg("hash"),
                            model_source=arg("model_source"),
                        )
                    )
                    return
                self._send_json({"error": "未找到接口"}, HTTPStatus.NOT_FOUND)
            except NotFoundError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except (DataError, ContractError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.UNPROCESSABLE_ENTITY)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                body = self._read_body_json()
                if path == "/api/dry-run":
                    as_of = str(body.get("as_of") or "").strip()
                    result = hub.dry_run(
                        db=body.get("db"),
                        as_of=as_of,
                        variant=body.get("variant"),
                        cadence=body.get("cadence"),
                        model_source=body.get("model_source"),
                        overrides=body.get("overrides"),
                    )
                    self._send_json(result)
                    return
                if path == "/api/runs/start":
                    as_of = str(body.get("as_of") or "").strip()
                    active = hub.start_run(
                        db=body.get("db"),
                        as_of=as_of,
                        variant=body.get("variant"),
                        cadence=body.get("cadence"),
                        model_source=body.get("model_source"),
                        overrides=body.get("overrides"),
                    )
                    self._send_json({"started": True, "active": active}, HTTPStatus.ACCEPTED)
                    return
                if path == "/api/runs/stop":
                    self._send_json(hub.stop_run())
                    return
                if path == "/api/chat/sessions":
                    session = hub.create_chat_session(
                        db=body.get("db"),
                        variant=body.get("variant"),
                        cadence=body.get("cadence"),
                        as_of=body.get("as_of"),
                        hash_prefix=body.get("hash"),
                        model_source=body.get("model_source"),
                        role_preset=str(body.get("role_preset") or "research_copilot"),
                        title=str(body.get("title") or "新对话"),
                    )
                    self._send_json({"session": session}, HTTPStatus.CREATED)
                    return
                if path == "/api/chat/stream":
                    attached = body.get("attached_codes")
                    if not isinstance(attached, list):
                        attached = []
                    stop_ev = threading.Event()
                    stream = hub.prepare_chat_turn(
                        db=body.get("db"),
                        variant=body.get("variant"),
                        cadence=body.get("cadence"),
                        as_of=body.get("as_of"),
                        hash_prefix=body.get("hash"),
                        model_source=body.get("model_source"),
                        session_id=body.get("session_id"),
                        message=str(body.get("message") or ""),
                        role_preset=str(body.get("role_preset") or "research_copilot"),
                        attached_codes=[str(x) for x in attached],
                        reasoning_effort=body.get("reasoning_effort"),
                        stop_event=stop_ev,
                    )
                    self.close_connection = True
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()
                    client_gone = False
                    for ev_name, payload in stream:
                        if client_gone:
                            continue
                        try:
                            self._write_sse(ev_name, payload)
                        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                            stop_ev.set()
                            client_gone = True
                    return
                self._send_json({"error": "未找到接口"}, HTTPStatus.NOT_FOUND)
            except BusyError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except (DataError, ContractError, SidecarError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.UNPROCESSABLE_ENTITY)

        def do_DELETE(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            arg = lambda key: query.get(key, [None])[0]
            try:
                if path == "/api/chat/sessions":
                    sid = arg("session_id") or ""
                    deleted = hub.delete_chat_session(
                        db=arg("db"),
                        variant=arg("variant"),
                        cadence=arg("cadence"),
                        as_of=arg("as_of"),
                        hash_prefix=arg("hash"),
                        model_source=arg("model_source"),
                        session_id=sid,
                    )
                    self._send_json({"deleted": deleted, "session_id": sid})
                    return
                self._send_json({"error": "未找到接口"}, HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except (DataError, ContractError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.UNPROCESSABLE_ENTITY)

    return Handler


class _ReuseThreadingServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def create_server(
    settings: app_config.Settings,
    *,
    default_db: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    sidecar_factory: Callable[[app_config.Settings], Any] | None = None,
    chat_stream_fn: chat.StreamFn | None = None,
) -> tuple[ThreadingHTTPServer, DebateHub]:
    hub = DebateHub(
        settings,
        default_db=default_db,
        sidecar_factory=sidecar_factory,
        chat_stream_fn=chat_stream_fn,
    )
    server = _ReuseThreadingServer((host, port), _make_handler(hub))
    return server, hub


def serve_ui(
    settings: app_config.Settings,
    *,
    default_db: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> int:
    server, hub = create_server(settings, default_db=default_db, host=host, port=port)
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}/"
    print(f"反弹模型 AI 辩论 Web UI 已启动：{url}", flush=True)
    print(f"默认数据库：{hub.default_db}", flush=True)
    print("按 Ctrl+C 停止服务。", flush=True)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭 Web UI 服务...", flush=True)
    finally:
        hub.stop_run()
        server.server_close()
    return 0
