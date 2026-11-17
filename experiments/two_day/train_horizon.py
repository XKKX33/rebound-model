"""换一个标签的买卖时点重训反弹模型（o1o2 = T+1 开盘买、T+2 开盘卖；o1o3；o1o5），每月重训，给 2015 年起的准入池打分。

除了标签的买卖时点，其余和十年实验 train_cadence.py 完全相同：28 个特征、参数、3 个种子取平均、扩张窗口、
训练行要"实际卖出日早于重训日"、百分位在当天准入池（剔除 < 2 元）里排（十年实验的口径）。直接调用它的函数。
--device cuda：在 GPU 上训（hist 是同一个算法，浮点和抽样不逐位相同；本机 CPU 被别的会话占着）。
每个（重训日, 种子）的打分存 <变体>/fits/，中断后重跑只算缺的。输出 <变体>/scores_<时点>_<设备>_M.parquet。
用法（run_all.py 会设好 DECADE_DATA）：python -X utf8 train_horizon.py --cell o1o2 [--device cuda] [--threads 8]
  --check-labels：用这里的标签代码重算 o1o5，必须和十年实验存的标签逐行一样（证明只换了买卖时点）。
  --max-fits N：冒烟，只训最后 N 个重训日（训练行最多的），不拼打分。
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

from horizon import cell_of, data_dir  # noqa: I001  先 import：把十年实验和 p0p1 的目录加进 sys.path
import train_cadence as tc
from common import PANEL, axes, load_market10, register, row_index, signal_range
from engine import Cell, Market, trades
from stage_data import FEATURES

CADENCE = "M"


def label_frame(m: Market, di: np.ndarray, ji: np.ndarray, cell: Cell) -> pd.DataFrame:
    """逐笔收益（carry 口径）→ 当天这些行里的百分位。和十年实验 train_cadence.build_labels 同一算法，只是时点可换。"""
    lab = trades(m, di, ji, cell, "carry")
    y = pd.Series(lab["ret"]).groupby(di).rank(pct=True).to_numpy()
    return pd.DataFrame({"di": di.astype(np.int32), "ji": ji.astype(np.int32), "y": y,
                         "exit_day": lab["exit_day"].astype(np.int32)})


def build_labels(cell: Cell, path: Path) -> int:
    m, _, _ = load_market10(raw=False)
    t = pq.read_table(PANEL, columns=["ts_code", "trade_date", "eligible", "below2"], filters=tc.FILTERS,
                      use_threads=False, read_dictionary=["ts_code"]).to_pandas()
    di, ji = row_index(t.ts_code, t.trade_date, m.codes, m.dates)
    del t
    frame = label_frame(m, di, ji, cell)
    del m
    gc.collect()
    frame.to_parquet(path, index=False)
    return len(frame)


def labels_path(out: Path, cell: str, key: str) -> Path:
    return out / f"labels10_{cell}_{key}.parquet"


def same_frames(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """逐列逐行相等（浮点列 NaN 对 NaN 算相等）。"""
    return list(a.columns) == list(b.columns) and len(a) == len(b) and all(
        np.array_equal(a[c].to_numpy(), b[c].to_numpy(), equal_nan=a[c].dtype.kind == "f") for c in a.columns)


def check_labels(out: Path, key: str) -> None:
    stored = labels_path(out, "o1o5", key)
    if not stored.is_file():
        raise SystemExit(f"没有十年实验的 o1o5 标签 {stored.name}：先跑 run_all.py 链接输入")
    tmp = out / f"labels10_o1o5_{key}.check.parquet"
    n = build_labels(cell_of("o1o5"), tmp)
    same = same_frames(pd.read_parquet(stored), pd.read_parquet(tmp))
    tmp.unlink()
    if not same:
        raise SystemExit("这里算的 o1o5 标签和十年实验存的不一样：标签代码有出入，停")
    print(f"labels o1o5 check: {n} rows identical to {stored.name}", flush=True)


def fit_name(cell: str, device: str, day: pd.Timestamp, n_train: int, n_test: int, key: str) -> str:
    parts = [key, tc.PARAMS, FEATURES, n_train, n_test, "batches", cell, device]
    tag = hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8]
    return f"{cell}_{device}_{day:%Y%m%d}_{tag}"


def booster_params(seed: int, threads: int, device: str) -> dict:
    """十年实验的参数；只多了 device。"""
    return {**tc.booster_params(seed, threads), "device": device}


def run_fits(days: list[int], x, y, exit_day, di, dates, cell: str, device: str, threads: int, key: str,
             fits: Path) -> int:
    fits.mkdir(exist_ok=True)
    new = 0
    for f in days:
        n, keep, lo, hi = tc.spans(f, dates, y, exit_day, di, 0)
        name = fit_name(cell, device, dates[f], int(keep.sum()), hi - lo, key)
        todo = [s for s in tc.SEEDS if not (fits / f"{name}_s{s}.npy").is_file()]
        if not todo:
            continue
        data = xgb.QuantileDMatrix(tc._Batches(x[:n], y[:n], keep), nthread=threads)
        for seed in todo:
            t0 = time.time()
            booster = xgb.train(booster_params(seed, threads, device), data, num_boost_round=tc.PARAMS["n_estimators"])
            booster.set_param({"device": "cpu"})  # 打分的特征在内存里，在 CPU 上预测（结果同一棵树）
            np.save(fits / f"{name}_s{seed}.npy", booster.inplace_predict(x[lo:hi]).astype(np.float32))
            new += 1
            print(f"fit {dates[f].date()} seed {seed}: train {int(keep.sum())}, test {hi - lo} ({time.time() - t0:.0f}s)",
                  flush=True)
            del booster
        del data
        gc.collect()
    return new


def assemble(days: list[int], y, exit_day, di, ji, dates, cell: str, device: str, key: str, out: Path) -> tuple[str, int]:
    """每次重训的模型管到下一次重训之前，3 个种子取平均（和十年实验 train_cadence.assemble 同一做法）。"""
    score = np.full(len(di), np.nan, dtype=np.float32)
    for f, stop in zip(days, tc.block_ends(days, len(dates))):
        if stop > tc.score_end(dates, f):
            raise SystemExit(f"{dates[f].date()} 的打分区间超出了一年")
        _, keep, lo, hi = tc.spans(f, dates, y, exit_day, di, 0)
        name = fit_name(cell, device, dates[f], int(keep.sum()), hi - lo, key)
        preds = np.mean([np.load(out / "fits" / f"{name}_s{s}.npy") for s in tc.SEEDS], axis=0)
        end = int(np.searchsorted(di, stop))
        if len(preds) != hi - lo or end > hi:
            raise SystemExit(f"{dates[f].date()} 的缓存打分和打分区间对不上")
        score[lo:end] = preds[:end - lo]
    ok = np.isfinite(score)
    path = out / f"scores_{cell}_{device}_{CADENCE}.parquet"
    pd.DataFrame({"di": di[ok], "ji": ji[ok], "score": score[ok]}).to_parquet(path, index=False)
    return path.name, int(ok.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="o1o2")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-fits", type=int, default=0)
    ap.add_argument("--check-labels", action="store_true")
    args = ap.parse_args()
    out, start, key = data_dir(), time.time(), tc.panel_key()
    print(f"power throttling off: {tc.no_power_throttling()}", flush=True)
    if args.check_labels:
        check_labels(out, key)
        return
    tc.LABELS = labels_path(out, args.cell, key)
    if not tc.LABELS.is_file():
        print(f"labels {args.cell}: {build_labels(cell_of(args.cell), tc.LABELS)} rows ({time.time() - start:.0f}s)",
              flush=True)
    _, dates = axes()
    d_first, d_last = signal_range(dates)
    x, y, exit_day, di, ji = tc.load_features()
    print(f"features {x.shape} ({time.time() - start:.0f}s)", flush=True)
    days = tc.fit_days(dates, tc.CADENCES[CADENCE], d_first, d_last)
    if args.max_fits:
        new = run_fits(days[-args.max_fits:], x, y, exit_day, di, dates, args.cell, args.device, args.threads, key,
                       out / "fits")
        print(f"smoke: {new} new fits ({time.time() - start:.0f}s)", flush=True)
        return
    new = run_fits(days, x, y, exit_day, di, dates, args.cell, args.device, args.threads, key, out / "fits")
    name, scored = assemble(days, y, exit_day, di, ji, dates, args.cell, args.device, key, out)
    register("train_horizon", {"cell": args.cell, "device": args.device, "cadence": CADENCE, "seeds": tc.SEEDS,
                               "params": tc.PARAMS, "features": list(FEATURES), "purge": "exit_day < fit day",
                               "window": "expanding", "panel": key}, {"new_fits": new, "scored": scored})
    print(f"{args.cell} {args.device}: {len(days)} fit days, {new} new fits -> {name} ({time.time() - start:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
