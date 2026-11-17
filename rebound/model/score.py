"""`rebound model score`：反弹模型每天给全准入池打分，写进 `model_score`。

- 两个准入池版本（不限价、限 70 元）× 三种重训频率（每年、每季度、每月）= 六条序列。
- 第一次跑：补齐第一个重训日（2015-01）以来的每一天；之后每次只补库里还没有分的日子。
- **已经有分的日子不改写**：那是当时实际用的分。`fit_date`、`model_id` 记着是哪次重训的模型打的。
- 每天用这种频率"不晚于它的最后一个重训日"的模型；那个重训日按计划该有、但还没训，就拒绝，提示先训练。
- 写之前一次查完：要用的模型口径对不对；库里已有的分是不是现在这些模型打的（重训过就不是，混着用会让本金测算
  悄悄用两套口径）。有一处不对就一行都不写。
- 名次：当天分数从高到低，同分按代码；每条序列各排各的。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from rebound.data.db import WriteStore
from rebound import params as model_params
from rebound.features import FEATURES, eligible, feature_panel
from rebound.store import Store
from rebound.model.artifacts import Block, ModelError, list_blocks, load_models
from rebound.model.train import active_fit, check_current, data_sessions, fit_days, open_store, year_spans


@dataclass(frozen=True)
class ScoreReport:
    days: int = 0   # 补了几个交易日（任何一条序列补了就算）
    rows: int = 0   # 写了几行（六条序列合计）

    def as_dict(self) -> dict[str, int]:
        return {"days": self.days, "rows": self.rows}


def score_frame(rows: pd.DataFrame, variant: str, cadence: str, days: dict[date, date],
                blocks: dict[date, Block], cache: dict[tuple[str, date], list]) -> pd.DataFrame:
    """`rows`（这个版本准入池里的特征行）逐日用对应重训日的模型打分、排名，排成 `model_score` 的列。

    分数 = 3 个种子的预测（float32）取平均，和十年实验 `train_cadence.assemble` 同一个算法，存成 DOUBLE 不再变。
    """
    trade_days, inverse = np.unique(rows["trade_date"].dt.date.to_numpy(), return_inverse=True)
    fit_of_day = np.array([days[d] for d in trade_days], dtype=object)
    x = rows.loc[:, list(FEATURES)].to_numpy(np.float32)
    score = np.empty(len(rows), dtype=np.float32)
    for when in sorted(set(fit_of_day)):
        at = np.flatnonzero(np.isin(inverse, np.flatnonzero(fit_of_day == when)))
        key = (variant, when)
        if key not in cache:
            check_current(blocks[when])
            cache[key] = load_models(blocks[when])
            for booster in cache[key]:
                booster.set_param({"nthread": model_params.THREADS})
        score[at] = np.mean([b.inplace_predict(x[at]).astype(np.float32) for b in cache[key]], axis=0)
    fit = fit_of_day[inverse]
    out = pd.DataFrame({
        "trade_date": trade_days[inverse], "ts_code": rows["ts_code"].astype(str).to_numpy(), "variant": variant,
        "cadence": cadence, "score": score.astype(np.float64), "fit_date": fit,
        "model_id": np.array([blocks[f].model_id for f in fit_of_day], dtype=object)[inverse],
    }).sort_values(["trade_date", "score", "ts_code"], ascending=[True, False, True], kind="mergesort")
    out["rank"] = out.groupby("trade_date").cumcount().to_numpy() + 1
    out["created_at"] = datetime.now().isoformat(timespec="seconds")
    return out.reset_index(drop=True)


def fill_scores(
    db_path: Path,
    *,
    first_fit: date | None = None,
    log: Callable[[str], None] | None = None,
) -> ScoreReport:
    """把六条序列缺分的日子补上。`first_fit` 只给测试用。"""
    say = log or (lambda _line: None)
    with Store(db_path) as store:
        sessions = data_sessions(store)
        done = _stored(store)
        used = _used(store)
    blocks = {v: {b.fit_date: b for b in list_blocks(db_path, v)} for v in model_params.VARIANTS}
    pending: dict[tuple[str, str], dict[date, date]] = {}
    for variant in model_params.VARIANTS:
        for cadence in model_params.CADENCES:
            fits = fit_days(sessions, cadence, first_fit=first_fit)
            days = {}
            for day in sessions:
                fit = active_fit(day, fits)
                if fit is None or (day, variant, cadence) in done:
                    continue
                if fit not in blocks[variant]:
                    raise ModelError(f"{model_params.VARIANT_NAMES[variant]} 重训日 {fit} 还没训练"
                                     f"（{model_params.CADENCE_NAMES[cadence]}重训的 {day} 要用它）：先跑 rebound model train")
                days[day] = fit
            pending[(variant, cadence)] = days
    _check_blocks(pending, blocks, used)
    every = sorted({d for days in pending.values() for d in days})
    report = ScoreReport()
    for lo, hi in year_spans(every[0], every[-1]) if every else []:
        span = {d: f for days in pending.values() for d, f in days.items() if lo <= d <= hi}
        if not span:
            continue
        with open_store(db_path) as store:
            panel = feature_panel(store, min(span), max(span))
        written, wrote_days, cache = 0, set(), {}
        for (variant, cadence), days in pending.items():
            want = {d: f for d, f in days.items() if lo <= d <= hi}
            if not want:
                continue
            mask = eligible(panel, model_params.VARIANTS[variant], min_bars=model_params.MIN_BARS)
            rows = panel[mask & panel["trade_date"].isin(pd.DatetimeIndex([pd.Timestamp(d) for d in want])).to_numpy()]
            if not rows.empty:
                frame = score_frame(rows, variant, cadence, want, blocks[variant], cache)
                written += _write(db_path, frame)
                wrote_days.update(frame["trade_date"])
        report = ScoreReport(days=report.days + len(wrote_days), rows=report.rows + written)
        say(f"[打分] {lo.year} 年：{len(wrote_days)} 个交易日，写入 {written} 行")
        del panel, cache
    return report


def _check_blocks(pending: dict[tuple[str, str], dict[date, date]], blocks: dict[str, dict[date, Block]],
                  used: set[tuple[str, date, str]]) -> None:
    """写之前把要用的模型一次查完：口径不对、或者库里的分是被换掉的模型打的，就一行都不写。

    逐年查会在第几年才撞上，前面几年已经写进去了。
    """
    for variant, fit in sorted({(v, f) for (v, _c), days in pending.items() for f in days.values()}):
        check_current(blocks[variant][fit])
    replaced = sorted((v, f, m) for v, f, m in used if (block := blocks.get(v, {}).get(f)) is None or block.model_id != m)
    if replaced:
        v, f, m = replaced[0]
        now = blocks.get(v, {}).get(f)
        raise ModelError(f"model_score 里有 {len(replaced)} 组分是换掉的模型打的（例如{model_params.VARIANT_NAMES[v]} 重训日 {f}："
                         f"库里是 {m}，现在{'没有这个模型' if now is None else '是 ' + now.model_id}）。分不改写，"
                         "混着两个模型的分会让本金测算悄悄用两套口径：先删掉这些重训日打的分再跑 rebound model score")


def _used(store: Store) -> set[tuple[str, date, str]]:
    if not store.has_table("model_score"):
        return set()
    return set(store.connection.execute("SELECT DISTINCT variant, fit_date, model_id FROM model_score").fetchall())


def _stored(store: Store | WriteStore, where: str = "", params: list | None = None) -> set[tuple[date, str, str]]:
    if not store.has_table("model_score"):
        return set()
    sql = f"SELECT DISTINCT trade_date, variant, cadence FROM model_score {where}"
    return set(store.connection.execute(sql, params or []).fetchall())


def _write(db_path: Path, frame: pd.DataFrame) -> int:
    """写进去之前在同一个事务里再查一遍：已经有分的（日, 版本, 频率）一行都不写。"""
    variant, cadence = frame["variant"].iat[0], frame["cadence"].iat[0]
    with WriteStore(db_path) as store, store.transaction():
        done = _stored(store, "WHERE variant = ? AND cadence = ? AND trade_date BETWEEN ? AND ?",
                       [variant, cadence, frame["trade_date"].min(), frame["trade_date"].max()])
        fresh = frame[[(d, variant, cadence) not in done for d in frame["trade_date"]]]
        return store.append("model_score", fresh) if not fresh.empty else 0
