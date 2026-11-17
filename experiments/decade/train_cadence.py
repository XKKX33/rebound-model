"""按年 / 半年 / 季 / 月重训反弹模型，给 2015 年起每个准入池股票日打分。

标签、特征、参数、种子与 p0p1 的 train_daily.py 相同；只有重训的时点不同。
每次重训用扩张窗口：此前全部"实际卖出日早于重训日"的行。打分覆盖到下一次重训之前。
同一天的重训各频率共用（按年的重训日也是按季、按月的重训日）。
分两步，省内存：先算标签（要行情矩阵，算完就放掉），再训练（只要特征）。
训练时行按日期排好，训练行分批送进 xgboost 的 QuantileDMatrix，不把整份训练集再拷一份（本机空闲内存常不到 1 GB）。
每个（重训日, 种子）的打分存 fits/，中断后重跑只算缺的。输出 scores10_<频率>.parquet（di, ji, score）。
用法：python -X utf8 train_cadence.py [--cadences Y H Q M] [--threads 16] [--extra 列名 ...] [--window-years N]
--extra：在 28 个特征后面加 DATA/ind_features.parquet 里的这些列（补充登记七；和标签逐行对齐过）。
--window-years：滚动窗口，每次重训只用之前 N 年的信号日（补充登记八）。
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

from common import DATA, PANEL, axes, load_market10, register, row_index, signal_range
from engine import BASELINE, trades
from stage_data import FEATURES

CADENCES = {"Y": (1,), "H": (1, 7), "Q": (1, 4, 7, 10), "M": tuple(range(1, 13))}
SEEDS = (0, 1, 2)
PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              tree_method="hist")
LABELS, FITS = DATA / "labels10_o1o5.parquet", DATA / "fits"
#: 准入池、剔除 < 2 元。读的时候单线程（use_threads=False），保证几次读出来的行序一样
FILTERS = [("eligible", "==", True), ("below2", "==", False)]
#: 每批送进 QuantileDMatrix 的行数（约 28 MB）
CHUNK = 250_000
#: --train-from 时，训练行少于这么多的重训日跳过（例如起点后才几周的数据）。从头扩张的窗口不设下限。
MIN_TRAIN = 200_000
#: --extra 给的额外特征列（在 FEATURES 之后），和它们所在的文件（ind_features.py 写的，行和标签一一对应）
EXTRA: tuple[str, ...] = ()
EXTRA_FILE = DATA / "ind_features.parquet"
#: --window-years：滚动窗口，训练只用重训日之前这么多年的信号日（补充登记八）；None 是从头扩张
WINDOW_YEARS: int | None = None


def fit_days(dates: pd.DatetimeIndex, months: tuple[int, ...], d_first: int, d_last: int) -> list[int]:
    """每个选中月份的第一个开市日（下标），落在信号日范围里的。"""
    days = {int(dates.searchsorted(pd.Timestamp(y, mo, 1)))
            for y in range(dates[d_first].year, dates[d_last].year + 1) for mo in months}
    return sorted(d for d in days if d_first <= d <= d_last)


def score_end(dates: pd.DatetimeIndex, f: int) -> int:
    """一次重训的打分范围 [f, score_end)：往后一年（每种频率的两次重训之间都不超过一年）。"""
    return min(int(dates.searchsorted(dates[f] + pd.DateOffset(years=1))), len(dates))


def block_ends(fits: list[int], n_days: int) -> list[int]:
    """每次重训的打分用到下一次重训之前；最后一次用到数据末尾。"""
    return [*fits[1:], n_days]


def build_labels() -> int:
    """o1o5 在 carry 口径下的收益，取当天准入池（剔除 < 2 元）里的百分位。和 train_daily.py 同一算法。"""
    m, low, _ = load_market10(raw=False)
    t = pq.read_table(PANEL, columns=["ts_code", "trade_date", "eligible", "below2"], filters=FILTERS, use_threads=False,
                      read_dictionary=["ts_code"]).to_pandas()
    di, ji = row_index(t.ts_code, t.trade_date, m.codes, m.dates)
    del t
    lab = trades(m, di, ji, BASELINE, "carry")
    y = pd.Series(lab["ret"]).groupby(di).rank(pct=True).to_numpy()
    pd.DataFrame({"di": di.astype(np.int32), "ji": ji.astype(np.int32), "y": y,
                  "exit_day": lab["exit_day"].astype(np.int32)}).to_parquet(LABELS, index=False)
    del m, low, lab
    gc.collect()
    return len(di)

def panel_key() -> str:
    """面板的指纹（行数、准入行数、首尾日、接缝核对）：面板重建过，标签和缓存的打分都要重算。"""
    meta = json.loads((DATA / "panel10.meta.json").read_text(encoding="utf-8"))
    key = {k: meta[k] for k in ("rows", "eligible_rows", "first", "last", "seam")}
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:8]


def load_features() -> tuple[np.ndarray, ...]:
    """特征矩阵和标签，按日期排好：训练行落在一段前缀里，打分行是一段连续区间，都不用另拷。返回 x, y, exit_day, di, ji。

    一列一列读、按排好的顺序写进去，边读边放；和标签逐行对齐过。
    """
    lab = pd.read_parquet(LABELS)
    codes, dates = axes()
    t = pq.read_table(PANEL, columns=["ts_code", "trade_date"], filters=FILTERS, read_dictionary=["ts_code"],
                      use_threads=False).to_pandas()
    di, ji = row_index(t.ts_code, t.trade_date, codes, dates)
    del t
    if len(di) != len(lab) or (di != lab.di.to_numpy()).any() or (ji != lab.ji.to_numpy()).any():
        raise SystemExit(f"特征和标签的行对不上：删掉 {LABELS.name} 重算")
    if EXTRA:
        key = pq.read_table(EXTRA_FILE, columns=["di", "ji"])
        if key.num_rows != len(di) or (key.column("di").to_numpy() != di).any() or (key.column("ji").to_numpy() != ji).any():
            raise SystemExit(f"{EXTRA_FILE.name} 和标签的行对不上：重跑 ind_features.py --force")
        del key
    order = np.argsort(di, kind="stable")
    y, exit_day = lab.y.to_numpy()[order], lab.exit_day.to_numpy()[order]
    di, ji = di[order].astype(np.int32), ji[order].astype(np.int32)
    del lab
    x = np.empty((len(order), len(FEATURES) + len(EXTRA)), dtype=np.float32)
    for k, name in enumerate(FEATURES):
        x[:, k] = pq.read_table(PANEL, columns=[name], filters=FILTERS, use_threads=False).column(0).to_numpy()[order]
    for k, name in enumerate(EXTRA, len(FEATURES)):
        x[:, k] = pq.read_table(EXTRA_FILE, columns=[name]).column(0).to_numpy()[order]
    return x, y, exit_day, di, ji


def fit_name(day: pd.Timestamp, n_train: int, n_test: int, key: str, train_from: str | None = None) -> str:
    # "batches"：按日期排好、分批喂给 QuantileDMatrix；和最早整份拷贝的那版缓存分开。train_from、EXTRA 只在给了时进指纹
    parts = [key, PARAMS, FEATURES, n_train, n_test, "batches"] + ([train_from] if train_from else [])
    parts += [list(EXTRA)] if EXTRA else []
    parts += [f"window{WINDOW_YEARS}"] if WINDOW_YEARS else []
    tag = hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8]
    return f"o1o5_{day:%Y%m%d}_{tag}"


def spans(f: int, dates: pd.DatetimeIndex, y, exit_day, di, a: int = 0) -> tuple[int, np.ndarray, int, int]:
    """重训日 f（行已按日期排好）：训练行在 [a, n) 里、是实际卖出日早于 f 的（keep，长 n − a）；
    打分行是 [lo, hi) = [f, 往后一年)。a = 0 是从头起的扩张窗口；--train-from 时 a 是那天的第一行。"""
    n = int(np.searchsorted(di, f))  # 信号日早于 f 的行；卖出日至少晚 5 个开市日，训练行都在这里面
    keep = np.isfinite(y[a:n]) & (exit_day[a:n] >= 0) & (exit_day[a:n] < f)
    return n, keep, n, int(np.searchsorted(di, score_end(dates, f)))


def window_start(f: int, dates: pd.DatetimeIndex, di: np.ndarray, a: int) -> int:
    """重训日 f 的训练行从哪一行起：--window-years 时是 f 往前 N 年那天的第一行（滚动窗口），但不早于 a；
    没给就是 a（从头扩张，或 --train-from 的起点）。行已按日期排好。"""
    if not WINDOW_YEARS:
        return a
    first = int(dates.searchsorted(dates[f] - pd.DateOffset(years=WINDOW_YEARS)))
    return max(a, int(np.searchsorted(di, first)))


class _Batches(xgb.DataIter):
    """把前缀里要训练的行分批送进 QuantileDMatrix，不把整份训练集再拷一份（本机内存紧）。"""

    def __init__(self, x: np.ndarray, y: np.ndarray, keep: np.ndarray):
        self._x, self._y, self._keep, self._at = x, y, keep, 0
        super().__init__()

    def next(self, input_data) -> bool:
        while self._at < len(self._keep):
            stop = min(self._at + CHUNK, len(self._keep))
            rows = self._at + np.flatnonzero(self._keep[self._at:stop])
            self._at = stop
            if len(rows):
                input_data(data=self._x[rows], label=self._y[rows])
                return True
        return False

    def reset(self) -> None:
        self._at = 0


def no_power_throttling() -> bool:
    """Windows 会把后台进程限速、挪到能效核上（本机是 8 个性能核 + 8 个能效核）：xgboost 的 16 个线程只用上 3~5 个核。

    只对本进程关掉执行速度限速（SetProcessInformation / ProcessPowerThrottling），不改电源计划，进程退出就失效。
    """
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    speed = 0x1  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED
    state = State(1, speed, 0)  # 管执行速度这一项，状态设为"不限速"
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.SetProcessInformation.restype = wintypes.BOOL
        ok = kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state))
    except (AttributeError, OSError, ctypes.ArgumentError):
        return False  # 老版本 Windows 没有这个接口：照常训练，只是可能慢
    return bool(ok)


def booster_params(seed: int, threads: int) -> dict:
    """和 XGBRegressor(**PARAMS, random_state=seed, n_jobs=threads) 同一组参数（原生接口的写法）。"""
    return {"objective": "reg:squarederror", "max_depth": PARAMS["max_depth"], "eta": PARAMS["learning_rate"],
            "subsample": PARAMS["subsample"], "colsample_bytree": PARAMS["colsample_bytree"],
            "tree_method": PARAMS["tree_method"], "seed": seed, "nthread": threads}


def run_fits(days: list[int], x, y, exit_day, di, dates, threads: int, key: str, train_from: str | None = None,
             a: int = 0, min_train: int = 0) -> int:
    FITS.mkdir(exist_ok=True)
    new = 0
    for f in days:
        af = window_start(f, dates, di, a)
        n, keep, lo, hi = spans(f, dates, y, exit_day, di, af)
        if int(keep.sum()) < min_train:
            print(f"fit {dates[f].date()}: train {int(keep.sum())} < {min_train}, skip", flush=True)
            continue
        name = fit_name(dates[f], int(keep.sum()), hi - lo, key, train_from)
        todo = [s for s in SEEDS if not (FITS / f"{name}_s{s}.npy").is_file()]
        if not todo:
            continue
        data = xgb.QuantileDMatrix(_Batches(x[af:n], y[af:n], keep), nthread=threads)
        for seed in todo:
            t0 = time.time()
            booster = xgb.train(booster_params(seed, threads), data, num_boost_round=PARAMS["n_estimators"])
            np.save(FITS / f"{name}_s{seed}.npy", booster.inplace_predict(x[lo:hi]).astype(np.float32))
            new += 1
            print(f"fit {dates[f].date()} seed {seed}: train {int(keep.sum())}, test {hi - lo} ({time.time() - t0:.0f}s)",
                  flush=True)
            del booster
        del data
        gc.collect()
    return new


def assemble(cadence: str, days: list[int], y, exit_day, di, ji, dates, key: str, train_from: str | None = None,
             a: int = 0, min_train: int = 0) -> tuple[str, int]:
    """拼一种频率的打分：每次重训的模型管到下一次重训之前，3 个种子取平均。训练行太少跳过的那几段不打分。"""
    score = np.full(len(di), np.nan, dtype=np.float32)
    for f, stop in zip(days, block_ends(days, len(dates))):
        if stop > score_end(dates, f):
            raise SystemExit(f"{cadence} 在 {dates[f].date()} 的打分区间超出了一年")
        _, train_keep, lo, hi = spans(f, dates, y, exit_day, di, window_start(f, dates, di, a))
        if int(train_keep.sum()) < min_train:
            continue
        name = fit_name(dates[f], int(train_keep.sum()), hi - lo, key, train_from)
        preds = np.mean([np.load(FITS / f"{name}_s{s}.npy") for s in SEEDS], axis=0)
        end = int(np.searchsorted(di, stop))
        if len(preds) != hi - lo or end > hi:
            raise SystemExit(f"{cadence} 在 {dates[f].date()} 的缓存打分和打分区间对不上")
        score[lo:end] = preds[:end - lo]
    keep = np.isfinite(score)
    suffix = f"_from{train_from.replace('-', '')}" if train_from else ""
    path = DATA / f"scores10_{cadence}{suffix}.parquet"
    pd.DataFrame({"di": di[keep], "ji": ji[keep], "score": score[keep]}).to_parquet(path, index=False)
    return path.name, int(keep.sum())


def main() -> None:
    global LABELS, EXTRA, WINDOW_YEARS
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadences", nargs="+", choices=list(CADENCES), default=list(CADENCES))
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--extra", nargs="+", default=[], help="ind_features.parquet 里要加进模型的列（补充登记七）")
    ap.add_argument("--window-years", type=int, default=None,
                    help="滚动窗口：每次重训只用之前这么多年的信号日（补充登记八）；要配 DECADE_DATA 指到另一个目录")
    ap.add_argument("--train-from", default=None,
                    help="只用这天（YYYY-MM-DD）以后的信号日训练（固定起点，不是从头扩张）；训练行不到 MIN_TRAIN 的重训日跳过。"
                         "输出 scores10_<频率>_from<日期>.parquet")
    args = ap.parse_args()
    if args.window_years and DATA.resolve() == Path(__file__).resolve().parent:
        raise SystemExit("--window-years 要配 DECADE_DATA 指到另一个目录（例如 win/w2），不能覆盖原版的打分")
    EXTRA, WINDOW_YEARS = tuple(args.extra), args.window_years
    start, key = time.time(), panel_key()
    print(f"power throttling off: {no_power_throttling()}", flush=True)
    LABELS = DATA / f"labels10_o1o5_{key}.parquet"
    if not LABELS.is_file():
        print(f"labels: {build_labels()} rows ({time.time() - start:.0f}s)", flush=True)
    _, dates = axes()
    d_first, d_last = signal_range(dates)
    x, y, exit_day, di, ji = load_features()
    print(f"features {x.shape} ({time.time() - start:.0f}s)", flush=True)
    a = int(np.searchsorted(di, dates.searchsorted(pd.Timestamp(args.train_from)))) if args.train_from else 0
    min_train = MIN_TRAIN if args.train_from else 0
    for cadence in args.cadences:
        days = fit_days(dates, CADENCES[cadence], d_first, d_last)
        new = run_fits(days, x, y, exit_day, di, dates, args.threads, key, args.train_from, a, min_train)
        name, scored = assemble(cadence, days, y, exit_day, di, ji, dates, key, args.train_from, a, min_train)
        register("train", {"cadence": cadence, "fit_days": [str(dates[f].date()) for f in days], "seeds": SEEDS,
                           "params": PARAMS, "features": [*FEATURES, *EXTRA], "purge": "exit_day < fit day",
                           "window": (f"rolling {WINDOW_YEARS}y" if WINDOW_YEARS
                                      else f"from {args.train_from}" if args.train_from else "expanding"),
                           "panel": key}, {"new_fits": new, "scored": scored})
        print(f"{cadence}: {len(days)} fit days, {new} new fits -> {name} ({time.time() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
