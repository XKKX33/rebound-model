"""读取 P1 的产物：主面板、新特征、特征集合。后面的训练、评估都从这里取数据。"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from ml.common import RESULTS, data_dir
from rebound.features import FEATURES

F28: tuple[str, ...] = tuple(FEATURES)


@lru_cache(maxsize=1)
def f69() -> tuple[str, ...]:
    """原 28 个 + 去重后保留的 41 个新特征（1.4 的 final_features.txt）。"""
    path = RESULTS / "p1_4_features" / "final_features.txt"
    return tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def feature_set(name: str) -> tuple[str, ...]:
    if name == "F28":
        return F28
    if name == "F69":
        return f69()
    raise ValueError(f"未知的特征集 {name}（可选 F28 / F69）")


def load_panel(columns: list[str] | None = None) -> pd.DataFrame:
    """主面板（1.2）：键、28 个原特征、标签列。"""
    return pd.read_parquet(data_dir() / "panel_nocap.parquet", columns=columns)


def load_features(name: str) -> pd.DataFrame:
    """某个特征集的特征矩阵，与主面板逐行对齐。"""
    cols = feature_set(name)
    old = [c for c in cols if c in F28]
    new = [c for c in cols if c not in F28]
    parts = [load_panel(old)]
    if new:
        parts.append(pd.read_parquet(data_dir() / "features_extra.parquet", columns=new))
    x = pd.concat(parts, axis=1)
    return x[list(cols)]
