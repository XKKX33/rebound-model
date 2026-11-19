"""任务 6.3 图表：报告和看板用的 7 类图，统一放到 `ml/results/figures/`。只读已落盘的结果，不训练。

1. fig1_leaderboard.png    排行榜：测试 IC（全段）条形图 + 95% 区间，叉号 = 验证 IC
2. fig2_monthly_ic.png     逐月 IC 时间序列（验证段 + 测试段）
3. fig3_deciles.png        10 组平均超额（测试段）
4. fig4_nav.png            测试期净值：模型 vs 准入池等权 vs 中证 1000 指数（DB-K30-b8，20 万）
5. fig5_ablation.png       消融瀑布图（4.2，验证段）
6. fig6_shap_summary.png、fig6_shap_dependence.png   SHAP（复制自 4.3）
7. fig7_ic_vs_excess.png   IC 与组合超额散点（复制自 P5）

运行：python -X utf8 -m ml.p6_figures
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from ml.common import RESULTS, connect, load_config

OUT = RESULTS / "figures"
KEY = {"xgb_repro_F28": "仓库原模型 F28", "xgb_F28": "XGBoost F28", "xgb_F69": "XGBoost F69", "rf_F69": "随机森林 F69",
       "lstm_S": "LSTM-S", "lstm_SF69": "LSTM-S+F69", "ens_family": "集成 family"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _daily(mid: str, stage: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / "p3_models" / mid / f"{stage}_daily.csv", index_col=0, parse_dates=True)


def fig1(plt) -> None:
    lb = pd.read_csv(RESULTS / "leaderboard.csv")
    lb = lb[lb["model_id"] != "random"].sort_values("测试 IC 全段")
    ci = lb["测试 IC 全段 区间"].str.strip("[]").str.split(",").map(lambda v: (float(v[0]), float(v[1])))
    lo, hi = ci.map(lambda v: v[0]), ci.map(lambda v: v[1])
    lab = lb["模型"].str.replace("（3.9）", "").str.replace("（3.10）", "") + " " + lb["特征集"].where(~lb["特征集"].isin(["—", "r5"]), "")
    color = lb["类别"].map({"参照": "#9aa5b1", "模型": "#2a5d9f", "任务形式": "#c98a2b", "集成": "#2a7f62"})
    y = np.arange(len(lb))
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.barh(y, lb["测试 IC 全段"], color=color, xerr=[lb["测试 IC 全段"] - lo, hi - lb["测试 IC 全段"]], capsize=3,
            error_kw={"ecolor": "#555", "lw": 0.8})
    ax.scatter(lb["验证 IC（用于选择，偏乐观）"], y, marker="x", color="#c0392b", zorder=3, label="验证 IC（用于选择，偏乐观）")
    ax.set_yticks(y)
    ax.set_yticklabels(lab)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("测试期（2023-01 ~ 2026-10，按年重训）逐日 Rank IC 均值，误差线 = 按月整块重抽 95% 区间")
    ax.set_title("总排行榜：灰 = 参照，蓝 = 三个模型，橙 = 任务形式对比，绿 = 集成")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_leaderboard.png", dpi=130)
    plt.close(fig)


def fig2(plt) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for mid, name in KEY.items():
        d = pd.concat([_daily(mid, "select")["ic"], _daily(mid, "final")["ic"]])
        m = d.groupby(d.index.to_period("M")).mean()
        ax.plot(m.index.to_timestamp(), m.values, lw=1.4 if mid in ("xgb_F69", "xgb_repro_F28") else 0.9,
                ls="--" if mid == "xgb_repro_F28" else "-", label=name)
    cut = pd.Timestamp(load_config()["split"]["test_start"])
    ax.axvline(cut, color="k", lw=1)
    ax.text(cut, ax.get_ylim()[1], " 测试段开始（之前是验证段）", va="top", fontsize=8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("当月逐日 Rank IC 均值")
    ax.set_title("逐月 IC：验证段（2021 ~ 2022，模型用 2013 ~ 2020 训练）与测试段（2023 起，每年重训）")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_monthly_ic.png", dpi=130)
    plt.close(fig)


def fig3(plt) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    mids = ["xgb_repro_F28", "xgb_F69", "lstm_S", "lstm_SF69"]
    w = 0.2
    for k, mid in enumerate(mids):
        g = np.array(json.loads((RESULTS / "p3_models" / mid / "final.json").read_text(encoding="utf-8"))["groups"]["全段"]) * 100
        ax.bar(np.arange(1, 11) + (k - 1.5) * w, g, width=w, label=KEY[mid])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("按模型分数每天分成 10 组（10 = 分数最高）")
    ax.set_ylabel("5 日平均超额（%，相对当天准入池平均）")
    ax.set_title("10 组平均超额（测试期）：分数越高，之后 5 天的超额越高")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_deciles.png", dpi=130)
    plt.close(fig)


def fig4(plt) -> None:
    nav = pd.read_csv(RESULTS / "p5_backtest" / "nav_DB-K30-b8.csv", index_col=0, parse_dates=True)
    bench = load_config().get("benchmark_index", "000852.SH")
    with connect() as con:
        idx = con.execute("SELECT trade_date, close FROM index_daily WHERE ts_code = ? AND trade_date BETWEEN ? AND ? ORDER BY 1",
                          [bench, nav.index[0].date(), nav.index[-1].date()]).df()
    idx["trade_date"] = pd.to_datetime(idx["trade_date"])
    ind = idx.set_index("trade_date")["close"].reindex(nav.index).ffill()
    fig, ax = plt.subplots(figsize=(11, 5))
    for mid in ("xgb_repro_F28", "xgb_F28", "xgb_F69", "rf_F69", "lstm_SF69", "ens_family"):
        ax.plot(nav.index, nav[mid], lw=1.6 if mid == "xgb_F69" else 1.0, ls="--" if mid == "xgb_repro_F28" else "-", label=KEY[mid])
    ax.plot(nav.index, nav["准入池等权"], color="k", lw=1.2, ls=":", label="准入池等权（不可交易、不扣费）")
    ax.plot(nav.index, ind / ind.iloc[0], color="#888", lw=1.2, label=f"中证 1000 指数（{bench}）")
    ax.axhline(1, color="k", lw=0.4)
    ax.set_ylabel("净值（测试期第一天 = 1）")
    ax.set_title("测试期净值：DB-K30-b8，20 万本金，真实费用（仓库模拟器）")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_nav.png", dpi=130)
    plt.close(fig)


def fig5(plt) -> None:
    from ml.features_extra import GROUPS
    t = pd.read_csv(RESULTS / "p4_2_ablation" / "ablation.csv").set_index("配置")["验证 IC"]
    f69 = json.loads((RESULTS / "p3_models" / "xgb_F69" / "select.json").read_text(encoding="utf-8"))["summary"]["全部"]["ic_mean"]
    base = t["F28（同参数）"]
    inc = {f"{g} {GROUPS[g]}": t[f"F28 + {g}"] - base for g in GROUPS}
    inc = dict(sorted(inc.items(), key=lambda kv: -kv[1]))
    overlap = f69 - base - sum(inc.values())
    labels = ["F28（同参数）"] + list(inc) + ["重叠（各组信息重复）", "F69"]
    fig, ax = plt.subplots(figsize=(12, 5))
    cur = base
    ax.bar(0, base, color="#9aa5b1")
    for k, (name, v) in enumerate(list(inc.items()) + [("重叠", overlap)], start=1):
        ax.bar(k, v, bottom=cur, color="#2a7f62" if v >= 0 else "#c0392b")
        ax.text(k, cur + max(v, 0) + 0.0002, f"{v:+.4f}", ha="center", fontsize=8)
        cur += v
    ax.bar(len(labels) - 1, f69, color="#2a5d9f")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(min(base, f69, cur) - 0.006, max(base, f69, base + sum(max(v, 0) for v in inc.values())) + 0.003)
    ax.set_ylabel("验证段 IC")
    ax.set_title("消融瀑布图（验证段，同一套 XGBoost 参数）：从 F28 出发，每组单独加进去的增量；最后一格是各组信息重复造成的抵消")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_ablation.png", dpi=130)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    plt = _plt()
    for f in (fig1, fig2, fig3, fig4, fig5):
        f(plt)
        print(f"{f.__name__} 完成", flush=True)
    for src, dst in ((RESULTS / "p4_3_explain" / "fig1_shap_summary.png", "fig6_shap_summary.png"),
                     (RESULTS / "p4_3_explain" / "fig2_shap_dependence.png", "fig6_shap_dependence.png"),
                     (RESULTS / "p5_backtest" / "fig2_ic_vs_excess.png", "fig7_ic_vs_excess.png")):
        shutil.copyfile(src, OUT / dst)
    print(f"写出 {OUT}")


if __name__ == "__main__":
    main()
