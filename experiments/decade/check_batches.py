"""核对：训练改成"按日期排好、分批喂 QuantileDMatrix"之后，打分和原来"整份拷贝训练集"的那版一致。

拿 2015-01-05 那次重训：原来的 3 个种子（中断前存下的缓存）和新的 3 个种子，各取平均后比相关系数和排名重合度。
参照：原来两个种子之间的相关系数（换种子本身会带来多大差别）。
用法：python -X utf8 check_batches.py
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from common import DATA, axes, signal_range
from stage_data import FEATURES
from train_cadence import FITS, PARAMS, SEEDS, fit_days, fit_name, panel_key, score_end


def main() -> None:
    key = panel_key()
    lab = pd.read_parquet(DATA / f"labels10_o1o5_{key}.parquet")
    _, dates = axes()
    d_first, d_last = signal_range(dates)
    f = fit_days(dates, (1,), d_first, d_last)[0]
    di, y, exit_day = lab.di.to_numpy(), lab.y.to_numpy(), lab.exit_day.to_numpy()
    n_train = int((np.isfinite(y) & (exit_day >= 0) & (exit_day < f)).sum())
    old_test = np.flatnonzero((di >= f) & (di < score_end(dates, f)))
    old_tag = hashlib.sha256(json.dumps([key, PARAMS, FEATURES, n_train, len(old_test)]).encode()).hexdigest()[:8]
    old = [np.load(FITS / f"o1o5_{dates[f]:%Y%m%d}_{old_tag}_s{s}.npy") for s in SEEDS]
    new = [np.load(FITS / f"{fit_name(dates[f], n_train, len(old_test), key)}_s{s}.npy") for s in SEEDS]
    order = np.argsort(di, kind="stable")
    lo = int(np.searchsorted(di[order], f))
    old = [p[np.searchsorted(old_test, order[lo:lo + len(old_test)])] for p in old]  # 换成新的行序
    a, b = np.mean(old, axis=0), np.mean(new, axis=0)
    days = di[order][lo:lo + len(old_test)]
    rank_corr = pd.DataFrame({"d": days, "a": a, "b": b}).groupby("d").apply(
        lambda g: g.a.rank().corr(g.b.rank()), include_groups=False).mean()
    print(f"2015-01-05: {len(a)} rows; corr(old mean, new mean) {np.corrcoef(a, b)[0, 1]:.4f}; "
          f"daily rank corr {rank_corr:.4f}; corr(old seed0, old seed1) {np.corrcoef(old[0], old[1])[0, 1]:.4f}")


if __name__ == "__main__":
    main()
