"""共享测试夹具。

反弹模型的测试都跑在一个合成的"小世界"上（33 只主板票、2020-06 到 2023-05 的工作日，随机游走的价格），
几秒训完、几秒打完分，不需要网络和真实行情。
"""

from __future__ import annotations

import shutil
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rebound import config
from rebound.data import schema

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def settings() -> config.Settings:
    return config.load_settings()


# --------------------------------------------------------------------------- #
# 反弹模型的小世界：33 只主板票、2020-06 到 2023-05 的工作日
# --------------------------------------------------------------------------- #

#: 小世界的开市日（工作日都开市），到 2023-05-31 为止。
MODEL_SESSIONS: tuple[date, ...] = tuple(
    d for d in (date(2020, 6, 1) + timedelta(days=i) for i in range(1095)) if d.weekday() < 5
)
#: 小世界的第一个重训日（正式是 2015-01）：按月 2022-09 ~ 2023-05 九次，按季从 2022-10 起，按年只有 2023-01。
TEST_FIRST_FIT = date(2022, 9, 1)
#: 测试里的模型只训 8 棵浅树，几秒训完；正式参数在 `rebound.params.PARAMS`。
TINY_PARAMS = {"n_estimators": 8, "max_depth": 2, "learning_rate": 0.3, "subsample": 0.8,
               "colsample_bytree": 0.8, "tree_method": "hist"}
#: 当时的名称：600029 在 2022-11-01 戴帽（现在叫 *ST股票），600000 在 2022-03-01 摘帽（之前是 ST）。
MODEL_NAMES = (
    ("600000.SH", "ST股票0", date(2019, 1, 2), date(2022, 2, 28), date(2018, 12, 28), "*ST"),
    ("600000.SH", "股票0", date(2022, 3, 1), None, date(2022, 2, 25), "撤销ST"),
    ("600029.SH", "股票29", date(2010, 1, 4), date(2022, 10, 31), date(2010, 1, 4), "其他"),
    ("600029.SH", "*ST股票", date(2022, 11, 1), None, date(2022, 10, 28), "*ST"),
)


def build_model_world(path: Path) -> Path:
    """随机游走的价格，偶尔收盘涨停；成交额都过准入线，单价有时跌破 2 元下限。
    后 3 只（600030 ~ 600032）单价在 70 元上下：不限价版有、限 70 元版没有。"""
    from rebound.data.db import WriteStore

    rng = np.random.default_rng(7)
    rows = []
    for k in range(33):
        code, close = f"60{k:04d}.SH", (8.0 + k * 0.5 if k < 30 else (85.0, 66.0, 120.0)[k - 30])
        for i, day in enumerate(MODEL_SESSIONS):
            pre, move = close, float(np.clip(rng.normal(0.0005, 0.022), -0.095, 0.095))
            close = round(pre * 1.1, 2) if i % 41 == k % 41 else round(pre * (1 + move), 2)
            open_ = round(pre * (1 + move * 0.4), 2)
            rows.append((code, day, open_, max(open_, close), min(open_, close), close, pre,
                         (close / pre - 1) * 100, 1e4, float(rng.uniform(1e5, 6e5))))
    daily = pd.DataFrame(rows, columns=list(schema.TABLE_FIELDS["daily"]))
    with WriteStore(path) as store:
        store.upsert("trade_cal", pd.DataFrame({"exchange": "SSE", "cal_date": MODEL_SESSIONS, "is_open": 1}))
        store.upsert("stock_basic", pd.DataFrame({
            "ts_code": [f"60{k:04d}.SH" for k in range(33)], "symbol": [f"60{k:04d}" for k in range(33)],
            "name": [f"股票{k}" for k in range(29)] + ["*ST股票"] + [f"股票{k}" for k in range(30, 33)], "area": "",
            "industry": [f"行业{k % 6}" for k in range(33)], "market": "主板", "list_date": date(2000, 1, 4)}))
        store.upsert("namechange", pd.DataFrame(list(MODEL_NAMES), columns=list(schema.TABLE_FIELDS["namechange"])))
        store.upsert("daily", daily)
        store.upsert("daily_limit", daily[["ts_code", "trade_date"]].assign(
            up_limit=(daily.pre_close * 1.1).round(2), down_limit=(daily.pre_close * 0.9).round(2)))
        store.upsert("adj_factor", daily[["ts_code", "trade_date"]].assign(adj_factor=1.0))
    return path


@pytest.fixture(scope="session")
def model_world_pristine(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_model_world(tmp_path_factory.mktemp("model-world") / "market.duckdb")


@pytest.fixture
def model_world(model_world_pristine: Path, tmp_path: Path) -> Path:
    """每个测试一份自己的库（训练、打分会往里写）。"""
    target = tmp_path / "market.duckdb"
    shutil.copy2(model_world_pristine, target)
    return target


@pytest.fixture(scope="session")
def trained_world_pristine(model_world_pristine: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    from rebound.model.train import train

    path = tmp_path_factory.mktemp("trained-world") / "market.duckdb"
    shutil.copy2(model_world_pristine, path)
    train(path, params=TINY_PARAMS, first_fit=TEST_FIRST_FIT)
    return path


@pytest.fixture
def trained_world(trained_world_pristine: Path, tmp_path: Path) -> Path:
    """训好小模型（两个版本 × 9 个按月重训日）的库，连模型目录一起复制。"""
    from rebound.model.artifacts import models_root

    # 自己一个子目录：和 `model_world` 同时用时两个库（连同模型目录）不会互相覆盖
    target = tmp_path / "trained" / "market.duckdb"
    target.parent.mkdir()
    shutil.copy2(trained_world_pristine, target)
    shutil.copytree(models_root(trained_world_pristine), models_root(target))
    return target
