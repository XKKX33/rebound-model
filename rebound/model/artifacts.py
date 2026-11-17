"""反弹模型的模型文件：每个准入池版本、每个重训日一个目录。

    <库名>.models/return_model/<版本>/<重训日>/
        manifest.json   训练区间、行数、特征指纹、标签口径、准入池、参数、种子、线程、数据指纹、文件哈希
        seed0.ubj …     每个种子一个模型（xgboost 的 UBJSON）

模型目录挨着它用的库（`data/market.duckdb` → `data/market.models/`）：换一个库就是另一套模型，
不会拿 A 库训的模型给 B 库打分。按月重训的每个重训日都存一份，每年、每季度重训共用这些文件，
训过的不再训。

`model_id` = manifest（去掉生成时间）的哈希。种子和线程数固定时，同样的数据训出来的模型文件
逐字节相同，所以 `model_id` 也相同。加载时重算 `model_id`、核对每个文件的哈希，对不上就拒绝：
被改过的模型不能悄悄拿去打分。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import xgboost as xgb

from rebound.data.errors import DataError

STRATEGY = "return_model"
MANIFEST = "manifest.json"
#: 不进 `model_id` 的键：生成时间每次都不一样，不代表模型不同。
_VOLATILE = ("created_at", "model_id")
#: manifest 里的模型文件名只能是这个样子：路径是从 manifest 里读出来拼的，不能指到目录外面去。
_FILE_NAME = re.compile(r"seed[0-9]+[.]ubj")
#: 临时目录改名的重试间隔（秒）：Windows 上刚写完的文件可能被杀毒、索引短暂占着。
_RENAME_WAITS = (0.2, 0.5, 1.0, 2.0, 4.0)


class ModelError(DataError):
    """模型文件缺失、被改过，或者和要求的口径对不上。"""


def models_root(db_path: Path) -> Path:
    db_path = Path(db_path)
    return db_path.with_name(f"{db_path.stem}.models") / STRATEGY


@dataclass(frozen=True)
class Block:
    """一次重训：准入池版本、重训日、目录、manifest。"""

    variant: str
    fit_date: date
    path: Path
    manifest: dict[str, Any]

    @property
    def model_id(self) -> str:
        return str(self.manifest["model_id"])


def model_id(manifest: dict[str, Any]) -> str:
    body = {key: value for key, value in manifest.items() if key not in _VOLATILE}
    blob = json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def list_blocks(db_path: Path, variant: str) -> list[Block]:
    """这个版本训好的全部重训日，按日期升序。写到一半的临时目录不算。"""
    root = models_root(db_path) / variant
    if not root.is_dir():
        return []
    blocks = []
    for child in sorted(root.iterdir()):
        try:
            fit_date = date.fromisoformat(child.name)
        except ValueError:
            continue  # `<重训日>.tmp-<pid>`：训练中途断了留下的
        path = child / MANIFEST
        if not path.is_file():
            raise ModelError(f"{child} 里没有 {MANIFEST}：不是训练写出来的目录，删掉它再训练")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelError(f"{path} 读不出来（{exc}）：文件坏了，删掉 {child} 再训练") from exc
        blocks.append(Block(variant, fit_date, child, manifest))
    return blocks


def write_block(root: Path, manifest: dict[str, Any], boosters: list[tuple[str, xgb.Booster]]) -> Block:
    """先写进临时目录、写完再改名：中途断了不会留下半截的重训日。"""
    final = root / manifest["variant"] / manifest["fit_date"]
    tmp = final.with_name(f"{final.name}.tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    files = {}
    for name, booster in boosters:
        booster.save_model(tmp / name)
        files[name] = hashlib.sha256((tmp / name).read_bytes()).hexdigest()
    body = {**manifest, "files": files}
    body = {**body, "model_id": model_id(body), "created_at": datetime.now().isoformat(timespec="seconds")}
    (tmp / MANIFEST).write_text(json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    for wait in (*_RENAME_WAITS, None):
        try:
            os.replace(tmp, final)
            break
        except PermissionError:
            if wait is None:
                raise
            time.sleep(wait)
    return Block(manifest["variant"], date.fromisoformat(manifest["fit_date"]), final, body)


def load_models(block: Block) -> list[xgb.Booster]:
    """核对 `model_id` 和每个文件的哈希，再按文件名顺序加载。"""
    manifest = block.manifest
    if model_id(manifest) != manifest.get("model_id"):
        raise ModelError(f"{block.path}：manifest 的内容和它记的 model_id 对不上（被改过）；删掉这个目录重新训练")
    boosters = []
    for name, digest in sorted(manifest["files"].items()):
        if not _FILE_NAME.fullmatch(name):
            raise ModelError(f"{block.path}：manifest 里的文件名 {name!r} 不是 seedN.ubj；删掉这个目录重新训练")
        path = block.path / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ModelError(f"{path}：模型文件缺失或被改过（哈希和 manifest 对不上）；删掉 {block.path} 重新训练")
        booster = xgb.Booster()
        booster.load_model(path)
        boosters.append(booster)
    return boosters
