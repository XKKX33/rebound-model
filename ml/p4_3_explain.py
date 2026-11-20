"""任务 4.3 解释性（树模型部分；LSTM 的积分梯度见 `ml/lstm_ig.py`）。

解释的对象：按冻结配置在选择阶段（训练 2013 ~ 2020）重训的模型，在验证段（2021 ~ 2022）上解释——
- XGBoost 调参版 F69（树深 4、学习率 0.1、列抽样 0.5、161 棵、种子 0 ~ 2，和测试段用的配置完全相同）：
  gain 重要性；验证段排列重要性；SHAP（xgboost 自带的精确 TreeSHAP，`pred_contribs`）总览和依赖图；
- 随机森林 F69（冻结配置，种子 0）：不纯度重要性、验证段排列重要性，和 XGBoost 对照；
- 一棵深度 3 的浅决策树（只为直观说明，拟合 XGBoost F69 的分数，看它的主干逻辑）。

排列重要性：把一个特征（或一组特征）在**同一天内**随机打乱（保持每天的分布不变，只破坏它和收益的对应），看验证段逐日 IC 下降多少。
同一特征的原值和排名版本信息相同，所以另做“按组打乱”（一组特征一起打乱），结果更好解读。

运行：python -X utf8 -m ml.p4_3_explain
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ml import evaluate as ev
from ml.common import RESULTS, data_dir, register, result_dir, write_md
from ml.dataset import F28

OUT = "p4_3_explain"
N_SHAP = 60000          # SHAP 用的验证段抽样行数
SEED = 20261119


def _setup():
    from ml.models import RandomForest, XGB
    from ml.p3_4_random_forest import chosen_params
    from ml.p3_5_xgboost import chosen, params_of
    from ml.preprocess import Preprocessor
    from ml.train_eval import load
    keys, x = load("F69")
    st = ev.protocol_a(keys).select
    day = keys["di"].to_numpy()
    y = ev.train_labels(keys, st.fit_rows)
    pre = Preprocessor("tree").fit(x[st.fit_rows], day[st.fit_rows])
    Xf, names = pre.transform(x[st.fit_rows], day[st.fit_rows])
    Xv, _ = pre.transform(x[st.eval_rows], day[st.eval_rows])
    c = chosen("F69")
    cfg = {k: c[k] for k in ("max_depth", "learning_rate", "colsample_bytree")}
    xgbm = XGB(params=params_of(cfg, c["最终阶段树数"]), seeds=(0, 1, 2), name="xgb_F69").fit(Xf, y)
    rf = RandomForest(params=chosen_params("F69"), seeds=(0,), name="rf_F69").fit(Xf, y)
    kv = keys[st.eval_rows].reset_index(drop=True)
    return names, Xf, y, Xv, kv, xgbm, rf


def _ic(score, kv) -> float:
    return float(ev.daily_rank_ic(ev._frame(score, kv)).mean())


def permutation(model, Xv, kv, names, groups: dict[str, list[str]], rng) -> pd.DataFrame:
    """按天内打乱：单个特征、按组。返回 IC 下降。"""
    base = _ic(model.predict(Xv), kv)
    day = kv["di"].to_numpy()
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    ends = np.r_[starts[1:], len(day)]

    def shuffled(cols: list[int]) -> np.ndarray:
        X = Xv.copy()
        for a, b in zip(starts, ends):
            p = rng.permutation(b - a) + a
            X[a:b][:, cols] = Xv[p][:, cols]
        return X

    rows = []
    for g, members in groups.items():
        idx = [names.index(f) for f in members]
        rows.append({"对象": g, "类型": "组", "特征数": len(idx), "IC 下降": base - _ic(model.predict(shuffled(idx)), kv)})
    for i, f in enumerate(names):
        rows.append({"对象": f, "类型": "单个特征", "特征数": 1, "IC 下降": base - _ic(model.predict(shuffled([i])), kv)})
    out = pd.DataFrame(rows)
    out.attrs["base_ic"] = base
    return out


def run() -> dict:
    import xgboost as xgb
    from sklearn.tree import DecisionTreeRegressor
    from ml.api import F28_INFO
    from ml.features_extra import feature_table
    from ml.p4_2_ablation import groups as new_groups
    out = result_dir(OUT)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    names, Xf, y, Xv, kv, xgbm, rf = _setup()
    print(f"模型训练完成 {time.time() - t0:.0f}s；验证 IC：XGB {_ic(xgbm.predict(Xv), kv):.4f}，RF {_ic(rf.predict(Xv), kv):.4f}", flush=True)
    meta = feature_table().set_index("特征")
    cn = {f: (F28_INFO[f][0] if f in F28 else meta.loc[f, "中文名"]) for f in names}
    grp = {f: ("原特征" if f in F28 else meta.loc[f, "组"]) for f in names}
    groups = {"原特征（F28）": list(F28), **{f"{k} {meta.loc[v[0], '组'][2:]}": v for k, v in new_groups().items()}}

    # 1. gain 重要性（3 个种子平均，归一化到和为 1）
    gain = pd.DataFrame({f"s{i}": pd.Series(b.get_score(importance_type="total_gain")) for i, b in enumerate(xgbm.boosters)})
    gain.index = [names[int(k[1:])] for k in gain.index]
    gain = gain.reindex(names).fillna(0).mean(axis=1)
    gain = gain / gain.sum()
    # 2. 随机森林不纯度重要性
    imp_rf = pd.Series(rf.forests[0].feature_importances_, index=names)
    # 3. 排列重要性
    perm_x = permutation(xgbm, Xv, kv, names, groups, rng)
    print(f"XGB 排列重要性完成 {time.time() - t0:.0f}s", flush=True)
    perm_r = permutation(rf, Xv, kv, names, groups, rng)
    print(f"RF 排列重要性完成 {time.time() - t0:.0f}s", flush=True)
    # 4. SHAP（验证段抽样，3 个种子的 SHAP 取平均——模型分数是 3 个种子的平均，SHAP 可加）
    pick = np.sort(rng.choice(len(Xv), N_SHAP, replace=False))
    dm = xgb.DMatrix(Xv[pick])
    (data_dir() / "explain").mkdir(exist_ok=True)               # SHAP 抽样较大（约 30 MB），放 data/（不进 Git）
    shap = np.mean([b.predict(dm, pred_contribs=True, iteration_range=(0, n)) for b, n in zip(xgbm.boosters, xgbm.best_rounds)], axis=0)
    np.save(data_dir() / "explain" / "shap_sample.npy", shap.astype(np.float32))
    np.save(data_dir() / "explain" / "shap_sample_X.npy", Xv[pick].astype(np.float32))
    mean_abs = pd.Series(np.abs(shap[:, :-1]).mean(axis=0), index=names)
    # 5. 浅决策树：拟合 XGBoost 的分数（训练段抽 20 万行），看主干逻辑
    sub = rng.choice(len(Xf), 200000, replace=False)
    Xs = Xf[sub].copy()                                   # sklearn 的单棵树不接受缺失值：用训练抽样的中位数填
    med = np.nanmedian(Xs, axis=0)
    Xs = np.where(np.isnan(Xs), med[None, :], Xs)
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=2000, random_state=0).fit(Xs, xgbm.predict(Xf[sub]))
    Xv_med = np.where(np.isnan(Xv), med[None, :], Xv)
    tree_ic = _ic(tree.predict(Xv_med), kv)

    imp = pd.DataFrame({"特征": names, "中文名": [cn[f] for f in names], "分组": [grp[f] for f in names],
                        "XGB gain 占比": gain.to_numpy(), "XGB |SHAP| 均值": mean_abs.to_numpy(),
                        "XGB 排列：IC 下降": perm_x[perm_x["类型"] == "单个特征"]["IC 下降"].to_numpy(),
                        "RF 不纯度占比": imp_rf.to_numpy(),
                        "RF 排列：IC 下降": perm_r[perm_r["类型"] == "单个特征"]["IC 下降"].to_numpy()})
    imp.to_csv(out / "importance.csv", index=False, encoding="utf-8-sig")
    g = perm_x[perm_x["类型"] == "组"][["对象", "特征数", "IC 下降"]].rename(columns={"IC 下降": "XGB：IC 下降"})
    g["RF：IC 下降"] = perm_r[perm_r["类型"] == "组"]["IC 下降"].to_numpy()
    g.to_csv(out / "group_permutation.csv", index=False, encoding="utf-8-sig")
    info = {"base_ic_xgb": perm_x.attrs["base_ic"], "base_ic_rf": perm_r.attrs["base_ic"], "tree_ic": tree_ic,
            "n_shap": N_SHAP, "seconds": round(time.time() - t0, 1)}
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    register("4.3_explain", **info)
    _figures(out, imp, shap, Xv[pick], names, cn, tree, med)
    return info


def _figures(out, imp, shap, X, names, cn, tree, med) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 图 1：SHAP 总览（前 20 个特征，点的颜色 = 特征值在当天截面的高低）
    top = imp.sort_values("XGB |SHAP| 均值", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(9, 8))
    rng = np.random.default_rng(0)
    show = rng.choice(len(X), 6000, replace=False)
    for k, f in enumerate(top["特征"][::-1]):
        j = names.index(f)
        v = X[show, j]
        r = pd.Series(v).rank(pct=True).to_numpy()
        ax.scatter(shap[show, j], k + rng.uniform(-0.3, 0.3, len(show)), c=r, cmap="coolwarm", s=3, alpha=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{f}（{cn[f]}）" for f in top["特征"][::-1]], fontsize=8)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("SHAP 值（对分数的贡献；红 = 特征值高，蓝 = 特征值低）")
    ax.set_title("XGBoost F69 的 SHAP 总览\n（验证段抽样 6000 行，平均 |SHAP| 前 20 个特征）")
    fig.tight_layout()
    fig.savefig(out / "fig1_shap_summary.png", dpi=130)
    plt.close(fig)

    # 图 2：依赖图（3 个）
    pairs = [("min_ret20", "mkt_vol20"), ("ovn20", "mkt_ret20"), ("turn_f", "mkt_vol20")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (f, c) in zip(axes, pairs):
        j, jc = names.index(f), names.index(c)
        v = X[show, j]
        lo, hi = np.nanpercentile(v, [1, 99])
        m = np.isfinite(v) & (v >= lo) & (v <= hi)
        sc = ax.scatter(v[m], shap[show, j][m], c=X[show, jc][m], cmap="viridis", s=4, alpha=0.6)
        fig.colorbar(sc, ax=ax, label=f"{c}（{cn[c]}）")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel(f"{f}（{cn[f]}）")
        ax.set_ylabel("SHAP 值")
    fig.suptitle("SHAP 依赖图：横轴 = 特征值，纵轴 = 对分数的贡献，颜色 = 市场状态（看个股特征的作用是否随市场环境变化）")
    fig.tight_layout()
    fig.savefig(out / "fig2_shap_dependence.png", dpi=130)
    plt.close(fig)

    # 图 3：XGBoost 与随机森林的重要性对照（排列重要性）
    t = imp.sort_values("XGB 排列：IC 下降", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    yy = np.arange(len(t))
    ax.barh(yy - 0.2, t["XGB 排列：IC 下降"], height=0.4, label="XGBoost")
    ax.barh(yy + 0.2, t["RF 排列：IC 下降"], height=0.4, label="随机森林")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{f}（{c}）" for f, c in zip(t["特征"], t["中文名"])], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("天内打乱这个特征后，验证段 IC 下降多少")
    ax.set_title("排列重要性：XGBoost F69 与随机森林 F69（XGBoost 前 20 个）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig3_permutation.png", dpi=130)
    plt.close(fig)

    # 图 4：浅决策树
    fig, ax = plt.subplots(figsize=(18, 7))
    plot_tree(tree, feature_names=[f"{f}\n{cn[f]}" for f in names], filled=True, rounded=True, fontsize=8, ax=ax,
              impurity=False, precision=3)
    ax.set_title("深度 3 的浅决策树：拟合 XGBoost F69 的分数（value = 该叶子的平均分数，越高越值得买）")
    fig.tight_layout()
    fig.savefig(out / "fig4_shallow_tree.png", dpi=120)
    plt.close(fig)


def redraw_summary() -> None:
    """只重画图 1（SHAP 总览），读已保存的 SHAP 抽样，不重新训练。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    out = result_dir(OUT)
    imp = pd.read_csv(out / "importance.csv")
    shap, X = np.load(data_dir() / "explain" / "shap_sample.npy"), np.load(data_dir() / "explain" / "shap_sample_X.npy")
    names, cn = imp["特征"].tolist(), dict(zip(imp["特征"], imp["中文名"]))
    top = imp.sort_values("XGB |SHAP| 均值", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(9, 8))
    rng = np.random.default_rng(0)
    show = rng.choice(len(X), 6000, replace=False)
    for k, f in enumerate(top["特征"][::-1]):
        j = names.index(f)
        r = pd.Series(X[show, j]).rank(pct=True).to_numpy()
        ax.scatter(shap[show, j], k + rng.uniform(-0.3, 0.3, len(show)), c=r, cmap="coolwarm", s=3, alpha=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{f}（{cn[f]}）" for f in top["特征"][::-1]], fontsize=8)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("SHAP 值（对分数的贡献；红 = 特征值高，蓝 = 特征值低）")
    ax.set_title("XGBoost F69 的 SHAP 总览\n（验证段抽样 6000 行，平均 |SHAP| 前 20 个特征）")
    fig.tight_layout()
    fig.savefig(out / "fig1_shap_summary.png", dpi=130)
    plt.close(fig)


def report() -> None:
    out = result_dir(OUT)
    imp = pd.read_csv(out / "importance.csv")
    g = pd.read_csv(out / "group_permutation.csv")
    info = json.loads((out / "info.json").read_text(encoding="utf-8"))
    rk = imp[["XGB gain 占比", "XGB |SHAP| 均值", "XGB 排列：IC 下降", "RF 不纯度占比", "RF 排列：IC 下降"]].rank(ascending=False)
    corr = rk.corr(method="spearman").round(2)
    top = imp.sort_values("XGB |SHAP| 均值", ascending=False).head(25).copy()
    for c in ("XGB gain 占比", "RF 不纯度占比"):
        top[c] = (top[c] * 100).round(2)
    for c in ("XGB |SHAP| 均值", "XGB 排列：IC 下降", "RF 排列：IC 下降"):
        top[c] = top[c].round(4)
    g = g.copy()
    for c in ("XGB：IC 下降", "RF：IC 下降"):
        g[c] = g[c].round(4)
    lstm_path = RESULTS / "p4_3_explain" / "lstm_ig.md"
    lstm = lstm_path.read_text(encoding="utf-8") if lstm_path.exists() else "（LSTM 积分梯度见 `ml/lstm_ig.py`，结果待补）"
    notes_path = out / "notes.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "（读法待补）"
    write_md(out / "report.md", f"""
# 4.3 解释性

- 代码：`ml/p4_3_explain.py`（树模型）、`ml/lstm_ig.py`（LSTM 积分梯度）；结果 `importance.csv`、`group_permutation.csv`
- 解释的对象：按冻结配置在选择阶段（训练 2013 ~ 2020）重训的 XGBoost F69（161 棵 × 3 个种子）和随机森林 F69（种子 0），在验证段（2021 ~ 2022）上解释。
  这两个模型的验证 IC：XGBoost {info['base_ic_xgb']:.4f}、随机森林 {info['base_ic_rf']:.4f}
- 排列重要性：把特征在同一天内随机打乱，看验证段逐日 IC 下降多少（下降越多越重要；负数 = 打乱后反而更好）
- SHAP：xgboost 自带的精确 TreeSHAP，验证段抽样 {info['n_shap']:,} 行，3 个种子平均

## 一、按组的排列重要性（一组特征一起打乱）

{g.to_markdown(index=False)}

## 二、单个特征（按 XGBoost 平均 |SHAP| 排序的前 25 个；占比单位 %）

{top.drop(columns=["分组"]).to_markdown(index=False)}

**五种重要性排名之间的 Spearman 相关**（69 个特征）

{corr.to_markdown()}

![SHAP 总览](fig1_shap_summary.png)

![SHAP 依赖图](fig2_shap_dependence.png)

![排列重要性](fig3_permutation.png)

## 三、浅决策树（深度 3，拟合 XGBoost F69 的分数）

这棵树只有 8 个叶子，在验证段上的 IC 是 {info['tree_ic']:.4f}（XGBoost 本身 {info['base_ic_xgb']:.4f}）。

![浅决策树](fig4_shallow_tree.png)

## 四、LSTM：积分梯度

{lstm}

## 五、读法

{notes}
""")


if __name__ == "__main__":
    import sys
    if "--fig1" in sys.argv:
        redraw_summary()
    elif "--report" not in sys.argv:
        run()
    report()
