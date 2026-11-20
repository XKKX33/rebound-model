"""任务 4.4（选做）重要性的时间稳定性 + 为 `api.explain` 预先算好测试段的 SHAP。

- 按测试段的方式（PREREG 补充登记十）重训 XGBoost F69：2023、2024、2025、2026 年第一个交易日各一次（冻结配置，161 棵 × 种子 0 ~ 2），
  再加上选择阶段的模型（训练 2013 ~ 2020）。每个模型在它负责的那一年（选择阶段：验证段）的全部行上算 SHAP（3 个种子平均）。
- 时间稳定性：各模型的平均 |SHAP| 排名之间的 Spearman 相关，以及前 10 名的变化。
- 给接口用：测试段每一行保存“基线值 + 贡献最大的 8 个特征（名称、特征值、SHAP）”到 `data/ml/explain/xgb_F69.parquet`。
  重训得到的分数应与 `data/ml/scores/xgb_F69__final.parquet` 一致（同一配置、同样的种子；这里会核对）。

运行：python -X utf8 -m ml.p4_4_shap_stability
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ml import evaluate as ev
from ml.common import data_dir, register, result_dir, write_md

OUT = "p4_4_shap_stability"
TOP = 8


def main() -> None:
    import xgboost as xgb
    from ml.api import F28_INFO
    from ml.dataset import F28
    from ml.features_extra import feature_table
    from ml.models import XGB
    from ml.p3_5_xgboost import chosen, params_of
    from ml.preprocess import Preprocessor
    from ml.train_eval import load
    t0 = time.time()
    keys, x = load("F69")
    plan = ev.protocol_a(keys)
    day = keys["di"].to_numpy()
    c = chosen("F69")
    cfg = {k: c[k] for k in ("max_depth", "learning_rate", "colsample_bytree")}
    meta = feature_table().set_index("特征")
    stages = [("选择阶段（2021 ~ 2022）", plan.select.fit_rows, plan.select.eval_rows)] + \
             [(r.name, r.fit_rows, r.eval_rows) for r in plan.refits]
    mean_abs, rows_out, check = {}, [], []
    final_sc = pd.read_parquet(data_dir() / "scores" / "xgb_F69__final.parquet")
    test_pos = np.flatnonzero(plan.final.eval_rows)
    for name, fit, evr in stages:
        y = ev.train_labels(keys, fit)
        pre = Preprocessor("tree").fit(x[fit], day[fit])
        Xf, names = pre.transform(x[fit], day[fit])
        Xe, _ = pre.transform(x[evr], day[evr])
        m = XGB(params=params_of(cfg, c["最终阶段树数"]), seeds=(0, 1, 2), name="xgb_F69").fit(Xf, y)
        del Xf
        dm = xgb.DMatrix(Xe)
        shap = np.mean([b.predict(dm, pred_contribs=True, iteration_range=(0, n)) for b, n in zip(m.boosters, m.best_rounds)], axis=0)
        score = m.predict(Xe)
        mean_abs[name] = pd.Series(np.abs(shap[:, :-1]).mean(axis=0), index=names)
        if name != "选择阶段（2021 ~ 2022）":
            mine = evr[test_pos]                                         # 这一年的行在测试行里的位置
            check.append(float(np.max(np.abs(final_sc["score"].to_numpy()[mine] - score))))
            k = keys[evr]
            order = np.argsort(-np.abs(shap[:, :-1]), axis=1)[:, :TOP]
            r = np.arange(len(Xe))[:, None]
            out = pd.DataFrame({"trade_date": k["trade_date"].to_numpy(), "ts_code": k["ts_code"].to_numpy(),
                                "score": score.astype(np.float32), "base": shap[:, -1].astype(np.float32)})
            for j in range(TOP):
                out[f"f{j + 1}"] = pd.Categorical.from_codes(order[:, j], categories=names)
                out[f"x{j + 1}"] = Xe[r[:, 0], order[:, j]].astype(np.float32)
                out[f"v{j + 1}"] = shap[r[:, 0], order[:, j]].astype(np.float32)
            rows_out.append(out)
        print(f"{name}：训练 {int(fit.sum()):,} 行，SHAP {len(Xe):,} 行，{time.time() - t0:.0f}s", flush=True)
    exp_dir = data_dir() / "explain"
    exp_dir.mkdir(exist_ok=True)
    pd.concat(rows_out, ignore_index=True).to_parquet(exp_dir / "xgb_F69.parquet", index=False)

    out = result_dir(OUT)
    M = pd.DataFrame(mean_abs)
    M.to_csv(out / "mean_abs_shap.csv", encoding="utf-8-sig")
    rank = M.rank(ascending=False)
    corr = rank.corr(method="spearman").round(3)
    cn = {f: (F28_INFO[f][0] if f in F28 else meta.loc[f, "中文名"]) for f in M.index}
    top10 = pd.DataFrame({col: [f"{f}（{cn[f]}）" for f in M[col].sort_values(ascending=False).index[:10]] for col in M.columns},
                         index=range(1, 11))
    share = (M / M.sum()).copy()
    grp = {f: ("原特征" if f in F28 else meta.loc[f, "组"]) for f in M.index}
    by_group = share.groupby(pd.Series(grp)).sum().mul(100).round(1)
    moves = rank.max(axis=1) - rank.min(axis=1)
    big = rank.loc[M.mean(axis=1).sort_values(ascending=False).index[:20]].astype(int)
    big.index = [f"{f}（{cn[f]}）" for f in big.index]
    register("4.4_shap_stability", max_score_diff=max(check), seconds=round(time.time() - t0, 1))
    _figure(out, M, cn)
    notes_path = out / "notes.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "（读法待补）"
    write_md(out / "report.md", f"""
# 4.4 重要性的时间稳定性（选做）

- 代码：`ml/p4_4_shap_stability.py`；结果 `mean_abs_shap.csv`
- 模型：XGBoost F69 冻结配置（161 棵 × 3 个种子），选择阶段 1 次 + 测试段按年重训 4 次；每个模型在它负责的那一年上算 SHAP（3 个种子平均）
- 核对：按年重训得到的分数和测试段正式分数（`xgb_F69__final`）的最大差 {max(check):.2e}（同一配置、同样的种子，应为 0 或浮点误差）
- 附带产物：`data/ml/explain/xgb_F69.parquet`，测试段每一行的基线值和贡献最大的 {TOP} 个特征，供 `api.explain` 使用

## 一、各模型平均 |SHAP| 排名之间的 Spearman 相关（69 个特征）

{corr.to_markdown()}

## 二、各年的前 10 名

{top10.to_markdown()}

## 三、平均 |SHAP| 前 20 个特征在各模型里的名次

{big.to_markdown()}

## 四、按组的 |SHAP| 占比（%）

{by_group.to_markdown()}

![各年重要性](fig1_yearly_importance.png)

## 五、读法

{notes}
""")


def _figure(out, M, cn) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    top = M.mean(axis=1).sort_values(ascending=False).index[:15]
    fig, ax = plt.subplots(figsize=(10, 6))
    for f in top:
        ax.plot(range(M.shape[1]), M.loc[f], marker="o", label=f"{f}（{cn[f]}）")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(["选择阶段\n（2021 ~ 2022）"] + [f"{c} 年重训" for c in M.columns[1:]])
    ax.set_ylabel("平均 |SHAP|")
    ax.set_title("前 15 个特征的重要性随时间的变化（XGBoost F69）")
    ax.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out / "fig1_yearly_importance.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
