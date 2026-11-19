"""任务 3.6：在主环境为 LSTM 准备训练行和评估行（.npy），给 dl-study 环境的 `ml/lstm_train.py` 读。

和其他模型完全同一套划分和标签（`ml/evaluate.py`）：
- fit_j / fit_d：训练行的股票下标（对应序列张量的第 0 维）和信号日下标；fit_y：训练标签（只在已卖出的行里排的百分位）；
- eval_j / eval_d / eval_y：评估行（eval_y = 当天收益百分位 label_pct，只在选择阶段用于早停）；
- fit_day / eval_day：交易日下标（按天组批、按天算 IC）；
- fit_f / eval_f：F69 手工特征，dense 预处理（截面排名 + 市场状态 z 分数 + 缺失指示，只用训练行拟合），float16。
行的顺序 = 主面板里的顺序（按日期、代码），评估行和 `keys[stage.eval_rows]` 一一对应。
测试段按年重训（PREREG 补充登记十）：`final_<年>` 的训练行 = 那一年第一个交易日的重训行，评估行 = 全部测试行
（调度脚本只取这一年的分数拼接；2023 年那份完整分数就是“只训练一次”的对照）。

运行：python -X utf8 -m ml.lstm_prep select|final_2023|final_2024|final_2025|final_2026|hist_2015 ... hist_2026
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


def _dense_once(x, di, fit, evr):
    pre = Preprocessor("dense").fit(x[fit], di[fit])
    f_fit, names = pre.transform(x[fit], di[fit])
    f_eval, _ = pre.transform(x[evr], di[evr])
    return pre, f_fit, f_eval, names


def _bad_columns(x, rows, f, names) -> list[str]:
    """原始数据在这些行里有非空值、转换后却整列为 0 的列（偶发计算错误的症状，见 PREREG 冻结后的运行记录、补充登记十三）。"""
    bad = []
    for k, c in enumerate(names):
        if c.startswith("na_") or not len(f):
            continue
        if x.loc[rows, c].notna().any() and not np.any(f[:, k]):
            bad.append(c)
    return bad


def stable_dense(x, di, fit, evr, stage: str, max_runs: int = 8):
    """dense 预处理反复算，直到有两次结果逐元素完全相同、预处理参数有限、而且没有整列变成 0 的列，才采用；最多算 max_runs 次。
    每次不一致都记进 data/ml/debug/anomalies.log。

    2026-11-19 发现：正式测试段 LSTM S+F69 的 final_2023 / final_2026 训练特征里 breadth5 整列为 0、final_2025 评估特征里
    mkt_vol20 整列为 0（同样的代码重算正常），属于本项目反复出现、原因未查明的偶发计算错误（像是内存出错，只在大计算量时偶尔出现）。
    加这道防护后才重跑（PREREG 冻结后的运行记录）。
    """
    import time
    from ml.common import data_dir

    def problems_of(c) -> list[str]:
        out = []
        if not all(np.isfinite(v) for v in list(c[0].mean_.values()) + list(c[0].std_.values())):
            out.append("预处理参数有空值")
        for k, rows, name in ((1, fit, "fit_f"), (2, evr, "eval_f")):
            if bad := _bad_columns(x, rows, c[k], c[3]):
                out.append(f"{name} 整列为 0：{bad}")
        return out

    def same(a, b) -> bool:
        return (a[3] == b[3] and a[0].mean_ == b[0].mean_ and a[0].std_ == b[0].std_
                and all(a[k].shape == b[k].shape and np.array_equal(a[k], b[k], equal_nan=True) for k in (1, 2)))

    good: list = []
    for run in range(1, max_runs + 1):
        c = _dense_once(x, di, fit, evr)
        probs = problems_of(c)
        match = next((g for g in good if same(g, c)), None) if not probs else None
        if match is not None:
            if run > 2:
                print(f"{stage}：第 {run} 次才得到两次一致的结果", flush=True)
            return c
        if probs or good:
            msg = "；".join(probs) if probs else "和之前的结果不一致"
            d = data_dir() / "debug"
            d.mkdir(exist_ok=True)
            with open(d / "anomalies.log", "a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} lstm_prep {stage} 第 {run} 次：{msg}" + "\n")
            print(f"{stage}：dense 预处理第 {run} 次异常（{msg}）", flush=True)
        if not probs:
            good.append(c)
    raise RuntimeError(f"{stage}：dense 预处理算了 {max_runs} 次都没有两次一致的结果，停止")


def prepare(stage: str) -> dict:
    keys = ev.load_keys()
    plan = ev.protocol_a(keys)
    if stage == "select":
        st = plan.select
    elif stage == "deploy":                 # 交付用的模型（PREREG 补充登记十三第 4 条）：截至数据最后一天已卖出的全部行；评估行 = 最后 5 个交易日（只做流程核对）
        last = int(keys["di"].max())
        st = ev.Stage(stage, last + 1, ev.fit_rows(keys, last + 1), (keys["di"] > last - 5).to_numpy())
    elif stage.startswith("hist_"):         # 长期回测（PREREG 补充登记十一、十三）：评估行 = 这一年准入池全部行
        year = stage.split("_", 1)[1]
        st = next(r for r in ev.history_refits(keys) if r.name == year)
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
    pre, arrays["fit_f"], arrays["eval_f"], names = stable_dense(x, di, fit, evr, stage)
    del x
    if stage == "deploy":
        import joblib
        joblib.dump(pre, out / "preprocess.joblib")
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
