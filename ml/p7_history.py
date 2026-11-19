"""P7 长期回测（补充分析，PREREG 补充登记十一）：固定超参数，2015 ~ 2026 每年第一个交易日重训，给当年准入池全部股票打分，再用仓库模拟器连续回测。

不改变任何正式判定、不用来挑模型。结果分三段解读：2015 ~ 2020 仅供参考，2021 ~ 2022 偏乐观（挑参数用的年份），2023 ~ 2026 是真正的样本外。

两个阶段：
    python -X utf8 -m ml.p7_history train      # 逐年重训、打分（约 2.5 小时，可续跑：已完成的模型跳过）
    python -X utf8 -m ml.p7_history analyze    # 一致性核对、逐年 IC、回测、作者对照、候选名单修正前后对比、报告
产物：分数和明细在 data/ml/history/（不进 Git）；汇总的表、图、报告在 ml/results/p7_history/。
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from ml import evaluate as ev
from ml.common import RESULTS, data_dir, register, result_dir

OUT = "p7_history"
START_YEAR = 2015
TRAIN_ORDER = ["xgb_F69", "xgb_repro_F28", "xgb_F28", "xgb_F69_s5", "xgb_F69_s4", "rf_F69"]
NAMES = {"xgb_F69": "XGBoost F69", "xgb_F69_s5": "XGBoost F69（敏感性：第 5 组参数）", "xgb_F69_s4": "XGBoost F69（敏感性：第 4 组参数）",
         "xgb_F28": "XGBoost F28", "xgb_repro_F28": "作者原模型 F28", "rf_F69": "随机森林 F69", "ens_top3": "集成 top3"}


def hist_dir():
    d = data_dir() / "history"
    d.mkdir(exist_ok=True)
    return d


def make_model(mid: str):
    """冻结配置（PREREG 冻结记录、补充登记十一），每次调用返回一个全新的模型。"""
    from ml.models import RandomForest, XGB
    from ml.p3_4_random_forest import chosen_params
    from ml.p3_5_xgboost import chosen, params_of
    if mid == "xgb_repro_F28":
        return XGB(name="xgb_repro")
    if mid == "rf_F69":
        return RandomForest(params=chosen_params("F69"), seeds=(0, 1, 2), name="rf_F69")
    if mid in ("xgb_F69", "xgb_F28"):
        c = chosen(mid[-3:])
        cfg = {k: c[k] for k in ("max_depth", "learning_rate", "colsample_bytree")}
        return XGB(params=params_of(cfg, c["最终阶段树数"]), seeds=(0, 1, 2), name=mid)
    if mid == "xgb_F69_s5":
        return XGB(params=params_of({"max_depth": 6, "learning_rate": 0.03, "colsample_bytree": 0.5}, 312), seeds=(0, 1, 2), name=mid)
    if mid == "xgb_F69_s4":
        return XGB(params=params_of({"max_depth": 4, "learning_rate": 0.1, "colsample_bytree": 0.8}, 116), seeds=(0, 1, 2), name=mid)
    raise ValueError(mid)


def feature_set(mid: str) -> str:
    return "F28" if mid.endswith("F28") else "F69"


def _log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(result_dir(OUT) / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def train() -> None:
    """逐年重训、给当年准入池全部行打分。每个模型存一个文件；有了就跳过（可续跑）。"""
    from ml.dataset import F28
    from ml.preprocess import Preprocessor
    from ml.train_eval import check_keys, keys_fingerprint, load
    keys, x = load("F69")
    ref = keys_fingerprint(keys)
    if ref != keys_fingerprint(ev.load_keys()):
        raise SystemExit("两次读到的 keys 不一致，先排查数据文件")
    day = keys["di"].to_numpy().copy()
    stages = ev.history_refits(keys, START_YEAR)
    for mid in TRAIN_ORDER:
        path = hist_dir() / f"scores_{mid}.parquet"
        if path.exists():
            continue
        xs = x[list(F28)] if feature_set(mid) == "F28" else x
        parts, t_all = [], time.time()
        for st in stages:
            t0 = time.time()
            m = make_model(mid)
            fit = st.fit_rows
            y = ev.train_labels(keys, fit)
            pre = Preprocessor(m.preprocess).fit(xs[fit], day[fit])
            X_fit, _ = pre.transform(xs[fit], day[fit])
            m.fit(X_fit, y)
            del X_fit
            X_y, _ = pre.transform(xs[st.eval_rows], day[st.eval_rows])
            S = m.predict_seeds(X_y)
            if not np.isfinite(S).all():
                raise RuntimeError(f"{mid} {st.name}：分数里有空值或无穷")
            check_keys(keys, ref, f"{mid} {st.name} 年重训之后")
            k = keys.loc[st.eval_rows, ["trade_date", "ts_code"]].reset_index(drop=True)
            df = k.assign(score=S.mean(axis=1).astype(np.float32), **{f"s{j}": S[:, j].astype(np.float32) for j in range(S.shape[1])})
            parts.append(df)
            _log(f"{mid} {st.name} 年：训练 {int(fit.sum()):,} 行，打分 {len(df):,} 行，{time.time() - t0:.0f}s")
        pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
        _log(f"{mid} 完成，{time.time() - t_all:.0f}s")
        register("P7_history_train", model=mid, seconds=round(time.time() - t_all, 1))


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "analyze"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "train":
        train()
    else:
        from ml.p7_history_analyze import analyze
        analyze()
