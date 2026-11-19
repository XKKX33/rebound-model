"""任务 3.6：在主环境为 LSTM 准备训练行和评估行（.npy），给 dl-study 环境的 `ml/lstm_train.py` 读。

和其他模型完全同一套划分和标签（`ml/evaluate.py`）：
- fit_j / fit_d：训练行的股票下标（对应序列张量的第 0 维）和信号日下标；fit_y：训练标签（只在已卖出的行里排的百分位）；
- eval_j / eval_d / eval_y：评估行（eval_y = 当天收益百分位 label_pct，只在选择阶段用于早停）；
- fit_day / eval_day：交易日下标（按天组批、按天算 IC）；
- fit_f / eval_f：F69 手工特征，dense 预处理（截面排名 + 市场状态 z 分数 + 缺失指示，只用训练行拟合），float16。
行的顺序 = 主面板里的顺序（按日期、代码），评估行和 `keys[stage.eval_rows]` 一一对应。
测试段按年重训（PREREG 补充登记十）：`final_<年>` 的训练行 = 那一年第一个交易日的重训行，评估行 = 全部测试行
（调度脚本只取这一年的分数拼接；2023 年那份完整分数就是“只训练一次”的对照）。

运行：python -X utf8 -m ml.lstm_prep select|final_2023|final_2024|final_2025|final_2026
"""

from __future__ import annotations

import json
import sys

import numpy as np

from ml import evaluate as ev
from ml.common import data_dir
from ml.dataset import load_features
from ml.preprocess import Preprocessor
from ml.seq_data import WINDOW, load_X


def prepare(stage: str) -> dict:
    keys = ev.load_keys()
    plan = ev.protocol_a(keys)
    if stage == "select":
        st = plan.select
    else:                                   # "final_<年>"：测试段按年重训（PREREG 补充登记十），评估行 = 全部测试行
        year = stage.split("_", 1)[1]
        ref = next(r for r in plan.refits if r.name == year)
        st = ev.Stage(stage, ref.fit_di, ref.fit_rows, plan.final.eval_rows)
    _, _, _, panel_j = load_X()
    di = keys["di"].to_numpy()
    if di.min() < WINDOW - 1:
        raise RuntimeError("有样本的 60 天窗口越过了第一天")
    out = data_dir() / "lstm" / stage
    out.mkdir(parents=True, exist_ok=True)
    fit, evr = st.fit_rows, st.eval_rows
    arrays = {
        "fit_j": panel_j[fit], "fit_d": di[fit].astype(np.int32), "fit_day": di[fit].astype(np.int32),
        "fit_y": ev.train_labels(keys, fit).astype(np.float32),
        "eval_j": panel_j[evr], "eval_d": di[evr].astype(np.int32), "eval_day": di[evr].astype(np.int32),
        "eval_y": keys.loc[evr, "label_pct"].to_numpy(np.float32),
    }
    x = load_features("F69")
    pre = Preprocessor("dense").fit(x[fit], di[fit])
    arrays["fit_f"], names = pre.transform(x[fit], di[fit])
    arrays["eval_f"], _ = pre.transform(x[evr], di[evr])
    del x
    for k in ("fit_f", "eval_f"):
        arrays[k] = arrays[k].astype(np.float16)
    for k, v in arrays.items():
        np.save(out / f"{k}.npy", v)
    meta = {"stage": stage, "fit_di": st.fit_di, "fit_rows": int(fit.sum()), "eval_rows": int(evr.sum()),
            "f_names": names, "f_dim": len(names)}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {out}：训练 {meta['fit_rows']:,} 行，评估 {meta['eval_rows']:,} 行，手工特征 {len(names)} 维")
    return meta


if __name__ == "__main__":
    prepare(sys.argv[1] if len(sys.argv) > 1 else "select")
