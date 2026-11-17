"""每日选股：某个交易日收盘后，按反弹模型的分数出第二天开盘要买的名单。

    分数        这一天、这个股票池版本、这种重训频率存好的全池分数（`rebound model score` 写的）；
                没有就当场用对应重训日的模型算（同一套特征、同一批模型文件），不写库
    重训日      不晚于这一天的最后一个重训日（每年 / 每季度 / 每月）；它的模型没训就拒绝，提示先训练
    排序        分数从高到低，同分按代码
    行业去重    在前 candidate_limit 只里，每个行业最多 max_per_industry 只，取前 top_n

存好的分数是记录：有就用它（全池就是当时打分的那些票），以后复盘看到的也是这一份。当场算的和存好的
出自同一个模型、同一套特征，只差特征的 float32 末位（逐年整段算和只算一天，窗口的起点不同）。

只读：不联网、不写库。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from rebound import params as model_params
from rebound.data.errors import DataError
from rebound.features import FEATURES, eligible, feature_panel
from rebound.model.artifacts import Block, ModelError, list_blocks, load_models
from rebound.model.train import active_fit, check_current, fit_days, open_store
from rebound.store import Store

__all__ = ["DayPick", "SelectError", "dedup_top", "pick_day"]


class SelectError(DataError):
    """这一天选不了股：没有行情、股票池是空的、存好的分数和模型对不上。"""


@dataclass(frozen=True, slots=True)
class DayPick:
    """一天的选股结果。

    `pool` 是全池按分数降序（同分按代码）的排名，列：ts_code、name、industry、close、amount、score、
    rank（1 起）、pct（全池分位，越大越好）；`picks` 是行业去重后取的前 top_n 行，列同上。
    """

    day: date
    variant: str
    cadence: str
    fit_date: date
    model_id: str
    #: True = 分数来自库里存好的 `model_score`；False = 当场算的（库里还没有这一天）。
    stored: bool
    #: 存好的全池和今天重算的股票池差几只（多半是后来补了改名记录）；以存好的为准。
    differ: int
    pool: pd.DataFrame
    picks: pd.DataFrame


def dedup_top(industries: Sequence[str], *, max_per_industry: int, top_n: int) -> list[int]:
    """按给定顺序走一遍，每个行业最多 `max_per_industry` 只，取够 `top_n` 就停。返回入选的位置。

    同一题材行情里，前排常被同一个行业挤满；分散到不同行业是选股口径的一部分。
    行业为空的票算作同一个"空行业"，同样受上限约束。
    """
    chosen: list[int] = []
    taken: dict[str, int] = {}
    for at, industry in enumerate(industries):
        if taken.get(industry, 0) >= max_per_industry:
            continue
        chosen.append(at)
        taken[industry] = taken.get(industry, 0) + 1
        if len(chosen) >= top_n:
            break
    return chosen


def _block(store: Store, db_path: Path, day: date, variant: str, cadence: str) -> Block:
    """这一天该用的重训日：不晚于它的最后一个；那个重训日按计划该有、但模型没训，就拒绝。"""
    fit = active_fit(day, fit_days(list(store.sessions(day, 1_000_000)), cadence))
    if fit is None:
        raise ModelError(f"{day} 早于反弹模型的第一个重训日 {model_params.FIRST_FIT}，没有模型可用")
    blocks = {b.fit_date: b for b in list_blocks(db_path, variant)}
    if fit not in blocks:
        raise ModelError(f"{model_params.VARIANT_NAMES[variant]} 重训日 {fit} 的模型还没训练"
                         f"（{model_params.CADENCE_NAMES[cadence]}重训，{day} 要用它）：先跑 rebound model train")
    check_current(blocks[fit])
    return blocks[fit]


def _stored(store: Store, day: date, variant: str, cadence: str) -> pd.DataFrame:
    if not store.has_table("model_score"):
        return pd.DataFrame(columns=["ts_code", "score", "model_id"])
    return store.connection.execute(
        "SELECT ts_code, score, model_id FROM model_score WHERE trade_date = ? AND variant = ? AND cadence = ?",
        [day, variant, cadence]).df()


def _rank(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["score", "ts_code"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    return out.assign(rank=np.arange(1, len(out) + 1), pct=out["score"].rank(pct=True, method="average"))


def _latest_day(store: Store, as_of: date | None) -> date:
    if as_of is None:
        day = store.latest_trade_date()
        if day is None:
            raise SelectError("库里没有日线行情：先 rebound sync")
        return day
    if store.snapshot(as_of).empty:
        raise SelectError(f"{as_of} 本地没有日线行情（不是开市日，或者还没同步）")
    return as_of


def _scored_pool(store: Store, db_path: Path, day: date, variant: str, cadence: str) -> tuple[pd.DataFrame, Block, bool, int]:
    block = _block(store, db_path, day, variant, cadence)
    panel = feature_panel(store, day, day)
    panel = panel.assign(ts_code=panel["ts_code"].astype(str), name=panel["name"].astype(object),
                         industry=panel["industry"].astype(object))
    today = panel[eligible(panel, model_params.VARIANTS[variant], min_bars=model_params.MIN_BARS)]
    stored = _stored(store, day, variant, cadence)
    if stored.empty:
        if today.empty:
            raise SelectError(f"{day} {model_params.VARIANT_NAMES[variant]}股票池是空的")
        x = today.loc[:, list(FEATURES)].to_numpy(np.float32)
        score = np.mean([b.inplace_predict(x).astype(np.float32) for b in load_models(block)], axis=0)
        return today.assign(score=score.astype(np.float64)), block, False, 0
    repeated = int(stored["ts_code"].duplicated().sum())
    if repeated:
        raise SelectError(f"model_score 里 {day} {model_params.VARIANT_NAMES[variant]}"
                          f"{model_params.CADENCE_NAMES[cadence]}重训有 {repeated} 只票的分写了不止一行"
                          "（两个写库任务同时补过分）：删掉这一天这条序列的分再跑 rebound model score")
    ids = sorted(set(stored["model_id"]))
    if ids != [block.model_id]:
        raise SelectError(f"model_score 里 {day} 的分数是模型 {ids} 打的，这天该用 {block.model_id}"
                          f"（{block.path}）：模型被换过，先查清再选股")
    pool = panel.merge(stored[["ts_code", "score"]], on="ts_code", validate="1:1")
    if len(pool) != len(stored):
        raise SelectError(f"model_score 里 {day} 有 {len(stored) - len(pool)} 只票在行情里找不到：库在打分之后变过")
    differ = len(set(stored["ts_code"]) ^ set(today["ts_code"]))
    return pool, block, True, differ


def pick_day(db_path: Path, *, as_of: date | None = None, variant: str = model_params.DEFAULT_VARIANT,
             cadence: str = model_params.DEFAULT_CADENCE, top_n: int = 6, max_per_industry: int = 2,
             candidate_limit: int = 260) -> DayPick:
    """`as_of` 收盘后的名单（不填 = 库里最后一个交易日）。只读。"""
    if variant not in model_params.VARIANTS:
        raise SelectError(f"股票池版本只能是 {sorted(model_params.VARIANTS)}，收到 {variant!r}")
    if cadence not in model_params.CADENCES:
        raise SelectError(f"重训频率只能是 {sorted(model_params.CADENCES)}，收到 {cadence!r}")
    with open_store(db_path) as store:
        day = _latest_day(store, as_of)
        scored, block, stored, differ = _scored_pool(store, db_path, day, variant, cadence)
    columns = ["ts_code", "name", "industry", "close", "amount", "score"]
    pool = _rank(scored.loc[:, columns])
    head = pool.head(int(candidate_limit))
    industries = ["" if pd.isna(x) else str(x) for x in head["industry"]]
    chosen = dedup_top(industries, max_per_industry=int(max_per_industry), top_n=int(top_n))
    return DayPick(day=day, variant=variant, cadence=cadence, fit_date=block.fit_date, model_id=block.model_id,
                   stored=stored, differ=differ, pool=pool, picks=head.iloc[chosen].reset_index(drop=True))
