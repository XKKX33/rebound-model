"""任务 3.11 总排行榜：把所有模型的验证段（用于选择）和测试段（只跑一次）结果汇总到一张表。

只读已经落盘的结果（`ml/results/p3_models/<model_id>/select.json / final.json` 和逐日序列），不训练。
测试段要等所有选择冻结（PREREG 冻结记录）后，用 `train_eval --final`、`p3_6_lstm --final`、`p3_10_ensemble --final` 跑完才有
（测试段按年重训，PREREG 补充登记十；`final_once` 是只训练一次的对照）。

输出：`ml/results/leaderboard.csv`、`leaderboard.json`、`leaderboard.md`（含 PREREG 补充登记四的 4 个主要比较的正式判定）。

运行：python -X utf8 -m ml.p3_11_leaderboard
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ml import evaluate as ev
from ml.common import RESULTS, write_md

#: model_id → (显示名, 类别, 特征集)。顺序即排行榜里的顺序（参照在前）。
MODELS = {
    "random": ("随机打分", "参照", "—"),
    "rev_r5": ("反转因子 −r5", "参照", "r5"),
    "xgb_repro_F28": ("XGBoost 复现基准", "参照", "F28"),
    "rf_F28": ("随机森林", "模型", "F28"),
    "rf_F69": ("随机森林", "模型", "F69"),
    "xgb_F28": ("XGBoost 调参版", "模型", "F28"),
    "xgb_F69": ("XGBoost 调参版", "模型", "F69"),
    "lstm_S": ("LSTM", "模型", "S"),
    "lstm_SF69": ("LSTM", "模型", "S+F69"),
    "xgb_clf_F69": ("XGBoost 二分类（3.9）", "任务形式", "F69"),
    "xgb_rank_F69": ("XGBoost 排序（3.9）", "任务形式", "F69"),
    "ens_top3": ("集成：XGB F69 + RF F69 + XGB F28（3.10）", "集成", "—"),
    "ens_family": ("集成：XGB F69 + RF F69 + LSTM-S+F69（3.10）", "集成", "—"),
}
#: PREREG 补充登记四：事先指定的 4 个主要比较（A − B）
PRIMARY = [
    ("rf_F69", "xgb_F69", "RQ1：随机森林-F69 vs XGBoost-F69（Bagging vs Boosting）"),
    ("lstm_S", "xgb_F69", "RQ2：LSTM-S vs XGBoost-F69"),
    ("lstm_SF69", "lstm_S", "RQ2：LSTM-S+F69 vs LSTM-S"),
    ("xgb_F69", "xgb_F28", "RQ3：XGBoost-F69 vs XGBoost-F28"),
]


def _load(mid: str, stage: str):
    d = RESULTS / "p3_models" / mid
    if not (d / f"{stage}.json").exists():
        return None, None
    info = json.loads((d / f"{stage}.json").read_text(encoding="utf-8"))
    daily = pd.read_csv(d / f"{stage}_daily.csv", index_col=0, parse_dates=True)
    return info, daily


def _part_days() -> dict[str, pd.DatetimeIndex]:
    keys = ev.load_keys()
    st = ev.protocol_a(keys).final
    dates = keys["trade_date"].to_numpy()
    return {name: pd.DatetimeIndex(np.unique(dates[m])) for name, m in st.parts.items()}


def build() -> pd.DataFrame:
    parts = _part_days()
    rows = []
    for mid, (name, kind, fs) in MODELS.items():
        sel, _ = _load(mid, "select")
        fin, fdaily = _load(mid, "final")
        if sel is None:
            continue
        s = sel["summary"]["全部"]
        row = {"model_id": mid, "模型": name, "类别": kind, "特征集": fs,
               "验证 IC（用于选择，偏乐观）": s["ic_mean"], "验证前30超额": s["top30"]}
        if fin is not None:
            for p in ("T1", "T2", "全段"):
                f = fin["summary"][p]
                lo, hi = ev.month_ci(fdaily.loc[fdaily.index.intersection(parts[p]), "ic"])
                row[f"测试 IC {p}"] = f["ic_mean"]
                row[f"测试 IC {p} 区间"] = f"[{lo:.4f}, {hi:.4f}]"
            f = fin["summary"]["全段"]
            row.update({"测试 ICIR": f["icir"], "测试 NW t": f["nw_t"], "测试前6超额": f["top6"], "测试前30超额": f["top30"],
                        "测试多空": f["long_short"], "测试10组单调性": f["monotonic"]})
            for y in sorted(k for k in fin["summary"] if k.isdigit()):
                row[f"测试 IC {y}"] = fin["summary"][y]["ic_mean"]
            once, odaily = _load(mid, "final_once")
            if once is not None:                       # 对照：只训练一次（PREREG 补充登记十），看不重训会过时多少
                row["对照：只训练一次 测试 IC 全段"] = once["summary"]["全段"]["ic_mean"]
                row["对照：只训练一次 测试 IC T2"] = once["summary"]["T2"]["ic_mean"]
            seed_ic = fin["seed_ic"]
            row.update({"测试种子标准差": float(np.std(seed_ic, ddof=1)) if len(seed_ic) > 1 else 0.0,
                        "测试种子极差": fin["seed_spread"], "训练秒（最终阶段）": (fin.get("seconds") or {}).get("fit")})
        rows.append(row)
    return pd.DataFrame(rows)


def primary() -> pd.DataFrame:
    out = []
    for a, b, label in PRIMARY:
        ia, da = _load(a, "final")
        ib, db = _load(b, "final")
        if ia is None or ib is None:
            continue
        for col, unit, met in (("ic", 1, "逐日 IC"), ("top30", 100, "前 30 超额（%）")):
            c = ev.compare(da[col], db[col])
            out.append({"比较": label, "指标": met, "差": round(c["diff"] * unit, 4),
                        "95% 区间": f"[{c['ci'][0] * unit:+.4f}, {c['ci'][1] * unit:+.4f}]", "NW t": round(c["nw_t"], 2),
                        "判定": ev.verdict(c, ia["seed_spread"], ib["seed_spread"]) if col == "ic" else ""})
    return pd.DataFrame(out)


#: 其他比较（只描述，不做正式判定）：A − B
OTHERS = [
    ("xgb_F69", "xgb_repro_F28", "XGBoost F69 − 仓库原模型（全部改进的总效果）"),
    ("xgb_F28", "xgb_repro_F28", "XGBoost F28 − 仓库原模型（只调参）"),
    ("rf_F28", "xgb_F28", "随机森林 F28 − XGBoost F28"),
    ("lstm_SF69", "xgb_F69", "LSTM-S+F69 − XGBoost F69"),
    ("rf_F69", "rf_F28", "随机森林 F69 − F28（新特征）"),
    ("ens_top3", "xgb_F69", "集成 top3 − XGBoost F69"),
    ("ens_family", "xgb_F69", "集成 family − XGBoost F69"),
    ("xgb_rank_F69", "xgb_F69", "排序 − 回归（3.9）"),
    ("xgb_clf_F69", "xgb_F69", "二分类 − 回归（3.9）"),
    ("lstm_S", "rev_r5", "LSTM-S − 反转因子"),
]


def others() -> pd.DataFrame:
    out = []
    for a, b, label in OTHERS:
        _, da = _load(a, "final")
        _, db = _load(b, "final")
        if da is None or db is None:
            continue
        for col, unit, met in (("ic", 1, "逐日 IC"), ("top30", 100, "前 30 超额（%）")):
            c = ev.compare(da[col], db[col])
            out.append({"比较": label, "指标": met, "差": round(c["diff"] * unit, 4),
                        "95% 区间": f"[{c['ci'][0] * unit:+.4f}, {c['ci'][1] * unit:+.4f}]"})
    return pd.DataFrame(out)


def retrain_effect() -> pd.DataFrame:
    """按年重训 − 只训练一次：逐日 IC 差（全段、T2），看不重训会过时多少。"""
    parts = _part_days()
    out = []
    for mid, (name, kind, fs) in MODELS.items():
        _, d1 = _load(mid, "final")
        _, d0 = _load(mid, "final_once")
        if d1 is None or d0 is None or kind == "参照" and mid != "xgb_repro_F28":
            continue
        row = {"model_id": mid, "模型": f"{name} {fs}".strip()}
        for p in ("全段", "T2", "2026"):
            days = d1.index.intersection(parts[p])
            c = ev.compare(d1.loc[days, "ic"], d0.loc[days, "ic"])
            row[f"{p}：IC 差"] = round(c["diff"], 4)
            row[f"{p}：区间"] = f"[{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]"
        out.append(row)
    return pd.DataFrame(out)


def figures(lb: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    out = RESULTS / "p3_11_leaderboard"
    out.mkdir(exist_ok=True)
    d = lb[lb["model_id"] != "random"].copy()
    lo = d["测试 IC 全段 区间"].str.strip("[]").str.split(",").map(lambda v: float(v[0]))
    hi = d["测试 IC 全段 区间"].str.strip("[]").str.split(",").map(lambda v: float(v[1]))
    lab = d["模型"].str.replace("（3.9）", "").str.replace("（3.10）", "") + " " + d["特征集"].where(~d["特征集"].isin(["—", "r5"]), "")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.errorbar(d["测试 IC 全段"], y, xerr=[d["测试 IC 全段"] - lo, hi - d["测试 IC 全段"]], fmt="o", color="#2a5d9f",
                ecolor="#9fb6d6", capsize=3, label="测试 IC（全段）与 95% 区间")
    ax.scatter(d["验证 IC（用于选择，偏乐观）"], y, marker="x", color="#c0392b", label="验证 IC（用于选择）", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(lab)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("逐日 Rank IC 均值")
    ax.set_title("验证 IC 与测试 IC（测试段按年重训）")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1_valid_vs_test.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    years = [c for c in lb.columns if c.startswith("测试 IC 20")]
    for mid, style in (("xgb_repro_F28", "k--"), ("rf_F69", "-"), ("xgb_F28", "-"), ("xgb_F69", "-"), ("lstm_S", "-"),
                       ("lstm_SF69", "-"), ("ens_family", ":")):
        r = lb[lb["model_id"] == mid].iloc[0]
        ax.plot([c[-4:] for c in years], [r[c] for c in years], style, marker="o", label=mid)
    ax.set_ylabel("当年逐日 IC 均值")
    ax.set_title("测试段逐年 IC（2026 年只到 10 月）")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig2_yearly_ic.png", dpi=130)
    plt.close(fig)


def main() -> None:
    lb = build()
    lb.to_csv(RESULTS / "leaderboard.csv", index=False, encoding="utf-8-sig")
    (RESULTS / "leaderboard.json").write_text(lb.to_json(orient="records", force_ascii=False, indent=1), encoding="utf-8")
    pr = primary()
    ot = others()
    rt = retrain_effect()
    figures(lb)
    notes_path = RESULTS / "p3_11_leaderboard" / "notes.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "（读法待补）"
    show = lb.copy()
    for c in show.columns:
        if show[c].dtype.kind == "f":
            show[c] = show[c].round(4)
    write_md(RESULTS / "leaderboard.md", f"""
# 总排行榜（任务 3.11）

- 验证 IC 用于选择，偏乐观；测试段（2023-01 起）在所有选择冻结后只跑一次（冻结记录见 `ml/PREREG.md`）。
- 测试段的正式分数是**按年重训**的：每年第一个交易日用此前全部数据重训一次，参数不变（PREREG 补充登记十）。
  “对照：只训练一次”两列是 2023 年那次训练的模型给整个测试段打的分，用来看不重训会过时多少，不用于判定。
- 区间：按月整块重抽 95% 区间。超额、多空是小数（0.003 = 0.3%）。

{show.to_markdown(index=False)}

## 主要比较（PREREG 补充登记四，测试段全段，正式判定）

判定规则（PREREG 2.4）：区间下限 > 0 且差值 > 两者种子极差中较大的 → 更好；只有前者 → 可能更好；区间跨 0 → 分不出。

{pr.to_markdown(index=False) if len(pr) else "（测试段还没跑）"}

## 其他比较（只描述，不做判定）

{ot.to_markdown(index=False) if len(ot) else ""}

## 按年重训 − 只训练一次（逐日 IC 差）

{rt.to_markdown(index=False) if len(rt) else ""}

![验证 vs 测试](p3_11_leaderboard/fig1_valid_vs_test.png)

![逐年 IC](p3_11_leaderboard/fig2_yearly_ic.png)

## 读法

{notes}
""")
    print(f"写出 {RESULTS / 'leaderboard.md'}：{len(lb)} 个模型")


if __name__ == "__main__":
    main()
