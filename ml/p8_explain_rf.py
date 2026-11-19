"""P8：随机森林 F69 的逐只股票解释（PREREG 补充登记十三第 3 条），供 `api.explain` 使用。

方法：路径分解（Saabas）。一棵回归树从根走到叶子，每一步分裂把“当前节点的平均标签”变成“子节点的平均标签”，
这个变化记到这一步用的特征上；一只股票的分数 = 全部树根节点平均值（基线）+ 各特征贡献之和（逐行精确成立，本脚本会核对）。
森林的贡献 = 所有树（3 个种子 × 300 棵）的平均。和 XGBoost 用的 TreeSHAP 读法相同（加分 / 减分），但不是 SHAP：
靠近树根的特征会被多记一些。没用精确 TreeSHAP 的原因：900 棵深树 × 160 万行的计算量太大（见补充登记十三的说明）。

实现：每棵树先算好“走到每个节点时各特征的累计贡献”表（节点数 × 特征数），打分时 `apply` 得到每行落到的叶子，直接查表相加。

按测试段的方式重训（2023 ~ 2026 每年第一个交易日，冻结配置，种子 0 ~ 2），重训分数必须与正式分数 `rf_F69__final` 一致（这里会核对），
测试段每一行保存“基线 + 贡献最大的 8 个特征（名称、特征值、贡献）”到 `data/ml/explain/rf_F69.parquet`，格式与 `xgb_F69.parquet` 相同。

运行：python -X utf8 -m ml.p8_explain_rf          # 约 30 分钟
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ml import evaluate as ev
from ml.common import data_dir, register

TOP = 8
CHUNK = 50_000


def path_table(tree, n_features: int) -> np.ndarray:
    """[节点数, 特征数]：从根走到这个节点，路径上每个特征累计带来的平均标签变化。"""
    t = tree.tree_
    value = t.value[:, 0, 0].astype(np.float64)
    table = np.zeros((t.node_count, n_features), dtype=np.float64)
    stack = [0]
    while stack:
        node = stack.pop()
        f = t.feature[node]
        for child in (t.children_left[node], t.children_right[node]):
            if child == -1:
                continue
            table[child] = table[node]
            table[child, f] += value[child] - value[node]
            stack.append(child)
    return table


def forest_contrib(forests, X: np.ndarray) -> tuple[np.ndarray, float]:
    """全部森林（种子）所有树的平均贡献 [n, 特征数] 和基线（所有树根节点值的平均）。"""
    trees = [e for f in forests for e in f.estimators_]
    base = float(np.mean([e.tree_.value[0, 0, 0] for e in trees]))
    out = np.zeros((len(X), X.shape[1]), dtype=np.float64)
    for f in forests:
        tables = [path_table(e, X.shape[1]) for e in f.estimators_]
        for lo in range(0, len(X), CHUNK):
            xb = X[lo:lo + CHUNK]
            leaves = f.apply(xb)                                    # [n, 300]
            acc = np.zeros((len(xb), X.shape[1]), dtype=np.float64)
            for k, tab in enumerate(tables):
                acc += tab[leaves[:, k]]
            out[lo:lo + CHUNK] += acc
    return out / len(trees), base


def main() -> None:
    from ml.models import RandomForest
    from ml.p3_4_random_forest import chosen_params
    from ml.preprocess import Preprocessor
    from ml.train_eval import load
    t0 = time.time()
    keys, x = load("F69")
    plan = ev.protocol_a(keys)
    day = keys["di"].to_numpy()
    final_sc = pd.read_parquet(data_dir() / "scores" / "rf_F69__final.parquet")
    test_pos = np.flatnonzero(plan.final.eval_rows)
    rows_out, checks = [], []
    for st in plan.refits:
        fit, evr = st.fit_rows, st.eval_rows
        m = RandomForest(params=chosen_params("F69"), seeds=(0, 1, 2), name="rf_F69")
        pre = Preprocessor(m.preprocess).fit(x[fit], day[fit])
        Xf, names = pre.transform(x[fit], day[fit])
        m.fit(Xf, ev.train_labels(keys, fit))
        del Xf
        Xe, _ = pre.transform(x[evr], day[evr])
        score = m.predict(Xe)
        diff = float(np.max(np.abs(final_sc["score"].to_numpy()[evr[test_pos]] - score)))
        contrib, base = forest_contrib(m.forests, Xe)
        gap = float(np.max(np.abs(base + contrib.sum(axis=1) - score)))
        checks.append((st.name, diff, gap))
        if diff > 1e-5 or gap > 1e-5:
            raise RuntimeError(f"{st.name}：重训分数与正式分数差 {diff:.2e}，或分解不完备 {gap:.2e}")
        k = keys[evr]
        order = np.argsort(-np.abs(contrib), axis=1)[:, :TOP]
        r = np.arange(len(Xe))[:, None]
        out = pd.DataFrame({"trade_date": k["trade_date"].to_numpy(), "ts_code": k["ts_code"].to_numpy(),
                            "score": score.astype(np.float32), "base": np.float32(base)})
        for j in range(TOP):
            out[f"f{j + 1}"] = pd.Categorical.from_codes(order[:, j], categories=names)
            out[f"x{j + 1}"] = Xe[r[:, 0], order[:, j]].astype(np.float32)
            out[f"v{j + 1}"] = contrib[r[:, 0], order[:, j]].astype(np.float32)
        rows_out.append(out)
        print(f"{st.name}：训练 {int(fit.sum()):,} 行，解释 {len(Xe):,} 行，分数差 {diff:.1e}，完备性误差 {gap:.1e}，{time.time() - t0:.0f}s", flush=True)
    exp_dir = data_dir() / "explain"
    exp_dir.mkdir(exist_ok=True)
    pd.concat(rows_out, ignore_index=True).to_parquet(exp_dir / "rf_F69.parquet", index=False)
    register("P8_explain_rf", checks=checks, seconds=round(time.time() - t0, 1))
    print(f"写出 {exp_dir / 'rf_F69.parquet'}")


if __name__ == "__main__":
    main()
