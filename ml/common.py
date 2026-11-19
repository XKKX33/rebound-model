"""数据与模型部分的公共工具：路径、配置、只读数据库连接、实验登记、计时与内存。

所有脚本都从仓库根目录运行：`python -X utf8 -m ml.<脚本名>`。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import duckdb
import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]          # rebound-model/
ML = ROOT / "ml"


def load_config() -> dict[str, Any]:
    with open(ML / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()
DB_PATH = ROOT / CFG["paths"]["db"]
DATA_OUT = ROOT / CFG["paths"]["data_out"]
RESULTS = ROOT / CFG["paths"]["results"]
REGISTRY = RESULTS / "registry.jsonl"


def result_dir(name: str) -> Path:
    """某一步的结果目录，例如 result_dir('p1_1_data_quality')。"""
    path = RESULTS / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    return DATA_OUT


def connect() -> duckdb.DuckDBPyConnection:
    """研究库的只读连接（总原则 4：研究库只读）。"""
    return duckdb.connect(str(DB_PATH), read_only=True)


def rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


@contextmanager
def timed(label: str, log: list[str] | None = None) -> Iterator[None]:
    """计时并打印内存，`log` 不为空时把这行也记进去。"""
    t0 = time.time()
    yield
    line = f"[{label}] {time.time() - t0:.1f}s，进程内存 {rss_gb():.2f} GB"
    print(line, flush=True)
    if log is not None:
        log.append(line)


def file_fingerprint(path: Path, nbytes: int = 1 << 20) -> str:
    """大文件的快速指纹：文件大小 + 首尾各 1MB 的 sha256（够用来判断"是不是同一份数据"）。"""
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
        f.seek(max(0, size - nbytes))
        h.update(f.read(nbytes))
    return h.hexdigest()[:16]


def register(step: str, **info: Any) -> None:
    """往 ml/results/registry.jsonl 追加一条实验 / 产物记录（总原则 5：可复现）。"""
    RESULTS.mkdir(parents=True, exist_ok=True)
    row = {"time": datetime.now().isoformat(timespec="seconds"), "step": step, **info}
    with open(REGISTRY, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_md(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"写出 {path.relative_to(ROOT)}")


def upsert_section(path: Path, heading: str, body: str) -> None:
    """Markdown 文件里“以 heading 开头的那一节”整体替换为 heading + body；没有就加到文件末尾。

    一节的范围：从 heading 那一行到下一个 `## ` / `### ` 标题或 `---` 分隔线之前。只动这一节，前后的内容保持原样
    （避免某个脚本重写自己的结果时，把后面其他任务的记录一起删掉）。
    """
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    block = heading.rstrip() + "\n\n" + body.strip() + "\n"
    i = text.find(heading)
    if i < 0:
        path.write_text(text.rstrip() + "\n\n" + block, encoding="utf-8")
        return
    j = len(text)
    for mark in ("\n## ", "\n### ", "\n---"):
        k = text.find(mark, i + len(heading))
        if k >= 0:
            j = min(j, k)
    path.write_text(text[:i] + block + text[j:], encoding="utf-8")
