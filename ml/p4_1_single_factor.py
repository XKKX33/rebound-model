"""任务 4.1 单因子检验：F69 的 69 个特征逐个在测试期算 Rank IC、ICIR、Newey-West t 和逐年表现，并和 PREREG 第一节登记的预期方向对照。

- 每个特征：每天在有值的行里算 Spearman(特征, 收益)，再按时间平均（`ml/evaluate.py` 的同一套函数）；
- 显著：Newey-West |t| > 3（Harvey-Liu-Zhu 的多重检验门槛，PREREG 第一节）；
- 判定（只对 41 个新特征中有方向的）：显著且方向一致 → 假设得到支持；显著但方向相反 → 与假设相反（必须讨论）；不显著 → 无证据。
  预期方向为 `?` 的只报告方向；`交互`（市场状态，同一天所有股票取值相同）单独不能区分股票，不做检验；
  28 个原特征没有登记方向，只描述。
- 同时给出验证段（2021 ~ 2022）的 IC 作对照，看方向是否稳定。

运行：python -X utf8 -m ml.p4_1_single_factor
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml import evaluate as ev
from ml.common import register, result_dir, write_md
from ml.dataset import F28, load_features

OUT = "p4_1_single_factor"
T_GATE = 3.0


def factor_ic(values: np.ndarray, keys: pd.DataFrame, rows: np.ndarray) -> pd.Series:
    """一个特征在 rows 上的逐日 Rank IC（只用有值的行；当天有效行 < 30 的天跳过）。"""
    v = values[rows]
    ok = np.isfinite(v)
    k = keys[rows].reset_index(drop=True)[ok].reset_index(drop=True)
    if len(k) == 0:
        return pd.Series(dtype=float)
    return ev.daily_rank_ic(ev._frame(v[ok], k))


def verdict(sign: str, ic: float, t: float) -> str:
    if sign == "交互":
        return "不检验（市场状态）"
    if not np.isfinite(t):
        return "无法计算"
    sig = abs(t) > T_GATE
    if sign in ("+", "-"):
        if not sig:
            return "无证据"
        return "假设得到支持" if (ic > 0) == (sign == "+") else "与假设相反"
    if sign == "?":
        return ("显著为正" if ic > 0 else "显著为负") if sig else "不显著"
    return ("显著为正" if ic > 0 else "显著为负") if sig else "不显著"      # 原特征：没有登记方向


def run() -> pd.DataFrame:
    from ml.api import F28_INFO
    from ml.features_extra import feature_table
    keys = ev.load_keys()
    plan = ev.protocol_a(keys)
    test, valid = plan.final.eval_rows, plan.select.eval_rows
    parts = {k: v for k, v in plan.final.parts.items()}
    x = load_features("F69")
    meta = feature_table().set_index("特征")
    rows = []
    years = keys["trade_date"].dt.year.to_numpy()
    for i, f in enumerate(x.columns):
        vals = x[f].to_numpy(np.float64)
        new = f not in F28
        sign = meta.loc[f, "预期方向"] if new else ""
        row = {"特征": f, "来源": "新特征" if new else "原特征",
               "组": meta.loc[f, "组"] if new else "仓库原特征",
               "中文名": meta.loc[f, "中文名"] if new else F28_INFO[f][0], "预期方向": sign,
               "测试期覆盖率": float(np.isfinite(vals[test]).mean())}
        if sign == "交互":
            row["判定"] = verdict(sign, np.nan, np.nan)
            rows.append(row)
            continue
        ic = factor_ic(vals, keys, test)
        s = ev.ic_summary(ic)
        row.update({"测试 IC": s["ic_mean"], "ICIR": s["icir"], "IC>0 天数比例": s["ic_pos"], "NW t": s["nw_t"],
                    "NW t（滞后 5）": s["nw_t_lag5"], "NW t（滞后 20）": s["nw_t_lag20"]})
        for p in ("T1", "T2"):
            d = ic.loc[ic.index.intersection(pd.DatetimeIndex(keys.loc[parts[p], "trade_date"].unique()))]
            row[f"IC {p}"] = float(d.mean())
        for y in sorted(set(years[test])):
            d = ic[ic.index.year == y]
            row[f"IC {y}"] = float(d.mean())
        row["验证 IC（2021 ~ 2022）"] = float(factor_ic(vals, keys, valid).mean())
        row["判定"] = verdict(sign, s["ic_mean"], s["nw_t"])
        rows.append(row)
        print(f"[{i + 1}/{len(x.columns)}] {f}: IC {s['ic_mean']:+.4f}  t {s['nw_t']:+.1f}  {row['判定']}", flush=True)
    del x
    tab = pd.DataFrame(rows)
    out = result_dir(OUT)
    tab.to_csv(out / "single_factor.csv", index=False, encoding="utf-8-sig")
    register("4.1_single_factor", n=len(tab), verdicts=tab["判定"].value_counts().to_dict())
    return tab


def report(tab: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    out = result_dir(OUT)

    t = tab.dropna(subset=["NW t"]).sort_values("NW t")
    colors = {"假设得到支持": "#2a7f62", "与假设相反": "#c0392b", "无证据": "#b0b0b0"}
    c = [colors.get(v, "#4a6fa5" if abs(tt) > T_GATE else "#d5dbe5") for v, tt in zip(t["判定"], t["NW t"])]
    fig, ax = plt.subplots(figsize=(8, 15))
    ax.barh(t["特征"] + "（" + t["中文名"] + "）", t["NW t"], color=c)
    for x0 in (-T_GATE, T_GATE):
        ax.axvline(x0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("测试期 Rank IC 的 Newey-West t（虚线 = ±3）")
    ax.set_title("单因子检验（测试期 2023-01 ~ 2026-10）\n"
                 "新特征：绿 = 与登记方向一致，红 = 相反，深灰 = 不显著\n"
                 "原特征 / 方向未定：蓝 = 显著，浅灰 = 不显著", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig1_tstat.png", dpi=130)
    plt.close(fig)

    def fmt(d):
        d = d.copy()
        for col in d.columns:
            if d[col].dtype.kind == "f":
                d[col] = d[col].round(2 if "t" in col or "比例" in col or "覆盖" in col else 4)
        return d.to_markdown(index=False)

    new = tab[tab["来源"] == "新特征"]
    counts = new["判定"].value_counts()
    cols = ["特征", "中文名", "组", "预期方向", "测试 IC", "NW t", "IC T1", "IC T2", "验证 IC（2021 ~ 2022）", "判定"]
    years = [c for c in tab.columns if c.startswith("IC 20")]
    yearly = tab.dropna(subset=["NW t"])[["特征", "中文名"] + years].copy()
    sign_flip = tab.dropna(subset=["NW t"])
    flip = sign_flip[np.sign(sign_flip["测试 IC"]) != np.sign(sign_flip["验证 IC（2021 ~ 2022）"])]
    notes_path = out / "notes.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "（读法待补）"
    write_md(out / "report.md", f"""
# 4.1 单因子检验（测试期）

- 代码：`ml/p4_1_single_factor.py`；结果表 `single_factor.csv`（全部列，含覆盖率、ICIR、NW t 的滞后 5 / 20 敏感性、逐年 IC）
- 做法：每个特征每天在有值的行里算 Spearman(特征, T+1 开盘买 T+5 开盘卖的收益)，再按时间平均；测试期 = 2023-01-03 ~ 2026-10-16
- 显著：Newey-West |t| > 3（滞后 10 天）；判定规则在看数据之前写在 `ml/PREREG.md` 第一节
- 这是“单个特征本身”的预测力，不涉及任何模型；IC 为负的特征对模型同样有用（模型会反着用）

## 一、新特征：和事先登记的方向对照

41 个新特征的判定：{"、".join(f"{k} {v} 个" for k, v in counts.items())}。

{fmt(new[cols])}

## 二、原特征（仓库 28 个，没有登记方向，只描述）

{fmt(tab[tab["来源"] == "原特征"][[c for c in cols if c not in ("组", "预期方向")]])}

## 三、逐年 IC

{fmt(yearly)}

**测试期和验证期方向相反的特征**（{len(flip)} 个）：{"、".join(flip["特征"]) if len(flip) else "无"}

![t 值](fig1_tstat.png)

## 四、读法

{notes}
""")


def main() -> None:
    tab = run()
    report(tab)


if __name__ == "__main__":
    import sys
    if "--report" in sys.argv:
        report(pd.read_csv(result_dir(OUT) / "single_factor.csv"))
    else:
        main()
