"""`rebound model train`：反弹模型每月第一个交易日重训一次，两个准入池版本各一套，模型文件全部存本地。

- 重训日：2015-01 起每个月的第一个开市日（`model_params.FIRST_FIT`）。每年、每季度重训用的是其中
  1 月、1/4/7/10 月的那几次，同一个重训日的模型三种频率共用，所以只按月训，训过的不再训。
- 每次重训只用实际卖出日早于重训日的行（扩张窗口：从库里第一天起）；标签是这些行在当天各自准入池里的
  百分位（`model_labels.percentile(known=...)`，修掉了研究版的轻微泄漏）。
- xgboost 参数、3 个种子、线程数都固定（`model_params`），同样的数据训两次，模型文件逐字节相同。
  训练行分块送进 `QuantileDMatrix`（和十年实验 `train_cadence.py` 同法），不另拷一份整个训练集。
- 已经训好的重训日跳过；口径（参数、种子、准入池、特征）变了就报错，要先删掉那个目录。
  这样旧模型不会被悄悄换掉，换掉要有人明确动手。
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import sys
import time
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from rebound.data.errors import DataError
from rebound import params as model_params
from rebound.features import FEATURES, eligible, feature_panel, fingerprint
from rebound.labels import (
    CARRY_MAX, MODEL_CELL, carry_trades, check_matrix, empty_matrix, fill_matrix, percentile,
)
from rebound.store import Store
from rebound.universe import NON_MAIN_PREFIXES
from rebound.model.artifacts import STRATEGY, Block, ModelError, list_blocks, models_root, write_block

MANIFEST_VERSION = 2
#: 按年算特征时 DuckDB 的上限：全历史的窗口函数不设限会按物理内存的 80% 去用，和别的进程抢。
DUCKDB_LIMITS = "SET memory_limit = '1GB'; SET threads = 8"
#: 送进 QuantileDMatrix 的分块窗口（行数）：和十年实验相同。分位点草图和分块有关，改了就是另一个模型。
CHUNK = 250_000
_NON_MAIN = "^(" + "|".join(NON_MAIN_PREFIXES) + ")"


def open_store(db_path: Path) -> Store:
    """只读打开，并设好 DuckDB 的内存和线程上限（训练、打分都用它）。"""
    store = Store(db_path)
    store.connection.execute(DUCKDB_LIMITS)
    return store


def data_sessions(store: Store) -> list[date]:
    """库里第一天到最后一天之间的开市日（日历往后排的日子不算）。"""
    first, last = store.connection.execute("SELECT min(trade_date), max(trade_date) FROM daily").fetchone()
    if last is None:
        raise DataError("库里没有日线行情：先 rebound sync 或 rebound data import-history")
    return [d for d in store.sessions(last, 1_000_000) if d >= first]


def year_spans(first: date, last: date) -> list[tuple[date, date]]:
    """按自然年切的区间：切点固定，同样的库每次切法相同，特征也就逐位相同。"""
    return [(max(first, date(y, 1, 1)), min(last, date(y, 12, 31))) for y in range(first.year, last.year + 1)]


def fit_days(sessions: list[date], cadence: str, *, first_fit: date | None = None) -> list[date]:
    """这种频率的重训日：选中月份的第一个开市日，不早于 `first_fit`（和十年实验 `fit_days` 同法）。"""
    start = first_fit or model_params.FIRST_FIT
    months = model_params.CADENCES[cadence]
    firsts: dict[tuple[int, int], date] = {}
    for day in sorted(sessions):
        firsts.setdefault((day.year, day.month), day)
    return sorted(day for (_, month), day in firsts.items() if month in months and day >= start)


def active_fit(day: date, fits: list[date]) -> date | None:
    """`day` 用哪一次重训的模型：不晚于它的最后一次。"""
    at = bisect_right(fits, day)
    return fits[at - 1] if at else None


@dataclass(frozen=True)
class TrainingRows:
    """全历史里至少在一个版本的准入池里的行（按日期排好，同日按代码）：日下标、股下标、28 个特征、
    各版本是否在池里、o1o5 标签。"""

    dates: pd.DatetimeIndex
    codes: np.ndarray
    di: np.ndarray
    ji: np.ndarray
    x: np.ndarray
    member: dict[str, np.ndarray]
    ret: np.ndarray
    exit_day: np.ndarray


def load_rows(db_path: Path, *, log: Callable[[str], None] | None = None) -> TrainingRows:
    """按年算特征（省内存），逐年填进价格矩阵；算完 o1o5 标签就放掉矩阵，只留准入池的行。"""
    say = log or (lambda _line: None)
    with open_store(db_path) as store:
        sessions = data_sessions(store)
        codes = np.array([c for (c,) in store.connection.execute(
            f"SELECT DISTINCT ts_code FROM daily WHERE NOT regexp_matches(ts_code, '{_NON_MAIN}') ORDER BY 1").fetchall()])
        matrix = empty_matrix(pd.DatetimeIndex([pd.Timestamp(d) for d in sessions]), codes)
        di_parts, ji_parts, feats, member = [], [], [], {v: [] for v in model_params.VARIANTS}
        for lo, hi in year_spans(sessions[0], sessions[-1]):
            t0 = time.time()
            panel = feature_panel(store, lo, hi)
            fill_matrix(matrix, panel)
            masks = {v: eligible(panel, u, min_bars=model_params.MIN_BARS) for v, u in model_params.VARIANTS.items()}
            keep = np.flatnonzero(np.logical_or.reduce(list(masks.values())))
            rows = panel.iloc[keep]
            di = matrix.dates.get_indexer(pd.DatetimeIndex(rows["trade_date"]))
            cat = rows["ts_code"].astype("category")
            ji = pd.Index(codes).get_indexer(cat.cat.categories.astype(str))[cat.cat.codes.to_numpy()]
            order = np.lexsort((ji, di))  # 按日、同日按代码：和十年实验按日稳定排序后的行序相同
            di_parts.append(di[order])
            ji_parts.append(ji[order])
            feats.append(rows.loc[:, list(FEATURES)].to_numpy(np.float32)[order])
            for v, mask in masks.items():
                member[v].append(mask[keep][order])
            say(f"[特征] {lo.year} 年：{len(panel)} 行，准入池 " + "、".join(
                f"{model_params.VARIANT_NAMES[v]} {int(m.sum())}" for v, m in masks.items()) + f" 行（{time.time() - t0:.0f}s）")
            del panel, rows
    check_matrix(matrix)
    di = np.concatenate(di_parts).astype(np.int64, copy=False)
    ji = np.concatenate(ji_parts).astype(np.int64, copy=False)
    del di_parts, ji_parts  # 拼好就放掉分年的那份（约 70 MB）
    if (np.diff(di) < 0).any():
        raise DataError("训练行没有按日期排好（按年拼接出了错）")
    labels = carry_trades(matrix, di, ji, MODEL_CELL)
    dates = matrix.dates
    del matrix
    gc.collect()
    x = np.empty((len(di), len(FEATURES)), dtype=np.float32)
    at = 0
    while feats:
        part = feats.pop(0)
        x[at: at + len(part)] = part
        at += len(part)
    return TrainingRows(dates=dates, codes=codes, di=di, ji=ji, x=x,
                        member={v: np.concatenate(m) for v, m in member.items()},
                        ret=labels["ret"], exit_day=labels["exit_day"])


@dataclass(frozen=True)
class TrainingSet:
    """一次重训的训练数据。`prefix`：这个版本准入池里信号日早于重训日的行（按日期排好）；
    `keep`：其中参加训练的（重训日前已卖出、收益算得出）；`y`：和 `prefix` 对齐的标签，不参加的是 NaN。"""

    prefix: np.ndarray
    keep: np.ndarray
    y: np.ndarray

    @property
    def rows(self) -> np.ndarray:
        return self.prefix[self.keep]


def training_set(rows: TrainingRows, variant: str, fit: int, *, research_labels: bool = False) -> TrainingSet:
    """重训日（日期下标 `fit`）的训练数据。

    正式口径：百分位只在"重训日前已经卖出"的行里排，没卖出的行不影响别人的标签（实盘重训时只知道这些）。
    `research_labels=True` 是十年实验的口径（百分位在当天整个准入池里排，含重训日后才卖出的行，
    有一点前视），只给逐位核对用。
    """
    pool = np.flatnonzero(rows.member[variant])
    prefix = pool[: int(np.searchsorted(rows.di[pool], fit))]
    ret = rows.ret[prefix]
    keep = np.isfinite(ret) & (rows.exit_day[prefix] >= 0) & (rows.exit_day[prefix] < fit)
    y = percentile(ret, rows.di[prefix], known=None if research_labels else keep)
    return TrainingSet(prefix=prefix, keep=keep, y=np.where(keep, y, np.nan))


class _Batches(xgb.DataIter):
    """训练行分块送进 QuantileDMatrix，不把整个训练集再拷一份。

    分块和十年实验 `train_cadence._Batches` 一样：在 `prefix` 上每 `CHUNK` 行一个窗口，送窗口里参加训练的行
    （QuantileDMatrix 的分位点草图和分块有关，分法相同，同样的数据才训出同样的模型）。
    """

    def __init__(self, x: np.ndarray, data: TrainingSet) -> None:
        self._x, self._data, self._at = x, data, 0
        super().__init__()

    def next(self, input_data: Callable[..., None]) -> bool:
        prefix, keep = self._data.prefix, self._data.keep
        while self._at < len(prefix):
            stop = min(self._at + CHUNK, len(prefix))
            at = self._at + np.flatnonzero(keep[self._at:stop])
            self._at = stop
            if len(at):
                input_data(data=self._x[prefix[at]], label=self._data.y[at])
                return True
        return False

    def reset(self) -> None:
        self._at = 0


def _digest(x: np.ndarray, data: TrainingSet) -> str:
    """训练数据的指纹（特征 + 标签，按送进模型的顺序），分块算，不拷整份。"""
    h = hashlib.sha256()
    for lo in range(0, len(data.prefix), CHUNK):
        at = lo + np.flatnonzero(data.keep[lo: lo + CHUNK])
        h.update(np.ascontiguousarray(x[data.prefix[at]]).tobytes())
        h.update(data.y[at].astype(np.float64).tobytes())
    return h.hexdigest()


def booster_params(params: dict[str, Any], seed: int) -> dict[str, Any]:
    """`XGBRegressor(**params, random_state=seed, n_jobs=THREADS)` 的原生写法（和十年实验相同）。"""
    return {"objective": "reg:squarederror", "max_depth": params["max_depth"], "eta": params["learning_rate"],
            "subsample": params["subsample"], "colsample_bytree": params["colsample_bytree"],
            "tree_method": params["tree_method"], "seed": seed, "nthread": model_params.THREADS}


def no_power_throttling() -> bool:
    """Windows 会把后台进程限速、挪到能效核上，xgboost 的线程只用上几个核（十年实验实测）。
    只关本进程的执行速度限速，不改电源计划，进程退出就失效。"""
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    state = State(1, 0x1, 0)  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED：管这一项，设为不限速
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.SetProcessInformation.restype = wintypes.BOOL
        return bool(kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), 4, ctypes.byref(state),
                                                   ctypes.sizeof(state)))
    except (AttributeError, OSError, ctypes.ArgumentError):
        return False


def spec(variant: str, fit_date: date, params: dict[str, Any]) -> dict[str, Any]:
    """一次重训的口径：和已有的对不上就不能跳过，也不能悄悄覆盖。"""
    return {
        "version": MANIFEST_VERSION, "strategy": STRATEGY, "variant": variant, "fit_date": fit_date.isoformat(),
        "label": {"entry": MODEL_CELL.entry, "x": MODEL_CELL.x, "exit": MODEL_CELL.exit, "carry_max": CARRY_MAX,
                  "target": "当天准入池里的百分位，只在重训日前已卖出的行里排"},
        "window": "扩张窗口：从库里第一天到重训日前已卖出的行",
        "features": {"names": list(FEATURES), "fingerprint": fingerprint()},
        "universe": dataclasses.asdict(model_params.VARIANTS[variant]), "min_bars": model_params.MIN_BARS,
        "params": dict(params), "seeds": list(model_params.SEEDS), "threads": model_params.THREADS,
        "chunk": CHUNK, "xgboost": xgb.__version__,
    }


#: 这些键变了，旧模型打的分就不能再用（特征、准入池、标签不同了）。参数、种子变了只是另一个模型。
CURRENT_KEYS = ("version", "strategy", "variant", "fit_date", "label", "features", "universe", "min_bars")
#: 已训的重训日能不能跳过：再加上参数和种子（换了就是另一个模型）。线程数、分块、xgboost 版本和说明文字不比：
#: 那是在哪台机器上、怎么训的，不改变这个模型该是什么；manifest 里照样记着。
TRAINED_KEYS = (*CURRENT_KEYS, "params", "seeds")


def _comparable(key: str, value: Any) -> Any:
    """标签口径里的 `target` 是给人看的说明：改个措辞不该让 284 个重训日全部作废。"""
    if key == "label" and isinstance(value, dict):
        return {k: v for k, v in value.items() if k != "target"}
    return value


def _changed(block: Block, want: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    return sorted(key for key in keys if _comparable(key, block.manifest.get(key)) != _comparable(key, want[key]))


def check_current(block: Block) -> None:
    """打分、选股之前：这次重训的特征、准入池、标签口径必须和现在的代码一致。"""
    changed = _changed(block, spec(block.variant, block.fit_date, block.manifest.get("params", {})), CURRENT_KEYS)
    if changed:
        raise ModelError(f"{block.path} 是按别的口径训的（{'、'.join(changed)} 不同）：要重训就先删掉这个目录")


@dataclass(frozen=True)
class TrainReport:
    trained: tuple[tuple[str, date, str], ...] = ()   # (版本, 重训日, model_id)
    skipped: tuple[tuple[str, date, str], ...] = ()

    def as_dict(self) -> dict[str, object]:
        row = lambda item: {"variant": item[0], "fit_date": item[1].isoformat(), "model_id": item[2]}  # noqa: E731
        return {"trained": [row(t) for t in self.trained], "skipped": [row(s) for s in self.skipped]}


def train(
    db_path: Path,
    *,
    variants: tuple[str, ...] | None = None,
    params: dict[str, Any] | None = None,
    first_fit: date | None = None,
    log: Callable[[str], None] | None = None,
) -> TrainReport:
    """把每个版本到期的重训日都训好（库里有行情的每月第一个开市日都算到期），已训的跳过。

    `first_fit` 只给测试用（小世界没有 2015 年的行情）。
    """
    say = log or (lambda _line: None)
    variants = tuple(variants or model_params.VARIANTS)
    params = dict(model_params.PARAMS if params is None else params)
    with Store(db_path) as store:
        fits = fit_days(data_sessions(store), "M", first_fit=first_fit)
    if not fits:
        raise DataError(f"库里的行情还没到第一个重训日 {first_fit or model_params.FIRST_FIT}：没有可训的")
    todo: dict[str, list[date]] = {}
    skipped, stale = [], []
    for variant in variants:
        existing = {block.fit_date: block for block in list_blocks(db_path, variant)}
        for fit in fits:
            if fit in existing:
                changed = _changed(existing[fit], spec(variant, fit, params), TRAINED_KEYS)
                if changed:
                    stale.append(f"{existing[fit].path}（{'、'.join(changed)} 不同）")
                skipped.append((variant, fit, existing[fit].model_id))
            else:
                todo.setdefault(variant, []).append(fit)
    if stale:  # 一次列全：一个一个报，要来回跑几百次
        raise ModelError(f"{len(stale)} 个已训的重训日是按别的口径训的，要重训就先删掉这些目录：" + "；".join(stale[:5])
                         + (f"；……共 {len(stale)} 个" if len(stale) > 5 else ""))
    if not todo:
        return TrainReport(skipped=tuple(skipped))

    no_power_throttling()
    rows = load_rows(db_path, log=say)
    root, trained, total, t_start = models_root(db_path), [], sum(len(v) for v in todo.values()), time.time()
    for variant, days in todo.items():
        for fit in days:
            at = int(rows.dates.searchsorted(pd.Timestamp(fit)))
            data = training_set(rows, variant, at)
            used = data.rows
            if len(used) == 0:
                raise DataError(f"{variant} {fit}：重训日之前没有已卖出的准入池行，训不了")
            t0 = time.time()
            matrix = xgb.QuantileDMatrix(_Batches(rows.x, data), nthread=model_params.THREADS)
            boosters = [(f"seed{seed}.ubj", xgb.train(booster_params(params, seed), matrix,
                                                     num_boost_round=params["n_estimators"]))
                        for seed in model_params.SEEDS]
            del matrix
            manifest = {**spec(variant, fit, params), "train": {
                "first_signal": rows.dates[rows.di[used].min()].date().isoformat(),
                "last_signal": rows.dates[rows.di[used].max()].date().isoformat(),
                "last_exit": rows.dates[rows.exit_day[used].max()].date().isoformat(),
                "rows": int(len(used)), "days": int(len(np.unique(rows.di[used]))),
                "data_sha256": _digest(rows.x, data),
            }}
            block = write_block(root, manifest, boosters)
            trained.append((variant, fit, block.model_id))
            done = len(trained)
            eta = (time.time() - t_start) / done * (total - done) / 60
            say(f"[训练 {done}/{total}] {model_params.VARIANT_NAMES[variant]} {fit}：{len(used)} 行"
                f"（{time.time() - t0:.0f}s，还要约 {eta:.0f} 分钟），model_id {block.model_id}")
            del boosters, data, used
            gc.collect()
    return TrainReport(trained=tuple(trained), skipped=tuple(skipped))