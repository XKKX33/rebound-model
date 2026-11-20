"""任务 3.10（选做）简单集成：成员分数当天换成百分位名次，等权平均（组合事先写在 PREREG 补充登记八）。

不训练：只读成员已落盘的分数 `data/ml/scores/<model_id>__<阶段>.parquet`，用统一的 `score_and_save` 评估和落盘。
各种子分别集成（成员的第 j 个种子一起平均），用来算种子极差。

运行：python -X utf8 -m ml.p3_10_ensemble            # 选择阶段（验证段）+ 报告
      python -X utf8 -m ml.p3_10_ensemble --final    # 测试段（按年重训 + 只训练一次的对照），成员的测试段分数都跑完后才用（任务 3.11）
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from ml import evaluate as ev
from ml.common import RESULTS, data_dir, result_dir, write_md
from ml.train_eval import score_and_save

OUT = "p3_10_ensemble"
ENSEMBLES = {
    "ens_top3": ["xgb_F69", "rf_F69", "xgb_F28"],
    "ens_family": ["xgb_F69", "rf_F69", "lstm_SF69"],
}
NAMES = {"xgb_F69": "XGBoost F69", "rf_F69": "随机森林 F69", "xgb_F28": "XGBoost F28", "lstm_SF69": "LSTM-S+F69"}


def _rank(score: np.ndarray, day: np.ndarray) -> np.ndarray:
    """当天截面百分位名次（平均名次处理并列），(0, 1]。"""
    s = pd.Series(score)
    return s.groupby(day).rank(pct=True).to_numpy()


def build(stage: str, keys=None) -> dict:
    keys = ev.load_keys() if keys is None else keys
    plan = ev.protocol_a(keys)
    st = plan.select if stage == "select" else plan.final
    k_eval = keys[st.eval_rows].reset_index(drop=True)
    day = k_eval["di"].to_numpy()
    out = {}
    for eid, members in ENSEMBLES.items():
        seed_cols = None
        cols = []
        for m in members:
            sc = pd.read_parquet(data_dir() / "scores" / f"{m}__{stage}.parquet")
            if not (sc["ts_code"].to_numpy() == k_eval["ts_code"].to_numpy()).all():
                raise RuntimeError(f"{m} 的分数行和评估行对不上")
            seeds = sorted(c for c in sc.columns if c[:1] == "s" and c[1:].isdigit())
            seed_cols = len(seeds) if seed_cols is None else min(seed_cols, len(seeds))
            cols.append(sc[seeds].to_numpy())
        S = np.column_stack([np.mean([_rank(c[:, j], day) for c in cols], axis=0) for j in range(seed_cols)])
        info = {"feature_set": "集成", "model": {"name": eid, "members": members, "method": "当天百分位名次等权平均"},
                "complexity": len(members), "seconds": {"fit": 0.0}}
        out[eid] = score_and_save(eid, stage, st, keys, S, info, save=True)
        print(f"[{stage}] {eid}：完成")
    return out


def report() -> None:
    out = result_dir(OUT)
    load = lambda m: (json.loads((RESULTS / "p3_models" / m / "select.json").read_text(encoding="utf-8")),  # noqa: E731
                      pd.read_csv(RESULTS / "p3_models" / m / "select_daily.csv", index_col=0, parse_dates=True))
    ids = list(ENSEMBLES) + sorted({m for ms in ENSEMBLES.values() for m in ms}, key=list(NAMES).index)
    data = {m: load(m) for m in ids}
    rows = []
    for m in ids:
        info, _ = data[m]
        s = info["summary"]["全部"]
        rows.append({"模型": m if m in ENSEMBLES else NAMES[m], "成员": " + ".join(NAMES[x] for x in ENSEMBLES[m]) if m in ENSEMBLES else "—",
                     "验证 IC": round(s["ic_mean"], 4), "ICIR": round(s["icir"], 3), "NW t": round(s["nw_t"], 1),
                     "前6 超额%": round(s["top6"] * 100, 3), "前30 超额%": round(s["top30"] * 100, 3), "多空%": round(s["long_short"] * 100, 3),
                     "各种子 IC": " / ".join(f"{v:.4f}" for v in info["seed_ic"]), "种子极差": round(info["seed_spread"], 4)})
    tab = pd.DataFrame(rows)
    members = sorted({m for ms in ENSEMBLES.values() for m in ms}, key=list(NAMES).index)
    corr = pd.DataFrame({NAMES[m]: data[m][1]["ic"] for m in members}).corr().round(3)
    cmp_rows = []
    for eid, ms in ENSEMBLES.items():
        best = max(ms, key=lambda m: data[m][0]["summary"]["全部"]["ic_mean"])
        for col, unit, met in (("ic", 1, "逐日 IC"), ("top30", 100, "前 30 超额（%）")):
            c = ev.compare(data[eid][1][col], data[best][1][col])
            row = {"比较": f"{eid} − 最好的成员（{NAMES[best]}）", "指标": met, "差": round(c["diff"] * unit, 4),
                   "95% 区间": f"[{c['ci'][0] * unit:+.4f}, {c['ci'][1] * unit:+.4f}]"}
            if col == "ic":
                row["判定"] = ev.verdict(c, data[eid][0]["seed_spread"], data[best][0]["seed_spread"])
            cmp_rows.append(row)
    cmp_tab = pd.DataFrame(cmp_rows).fillna("")
    notes_path = out / "notes.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "（读法待补）"
    write_md(out / "report.md", f"""
# 3.10 简单集成（选做，验证段）

- 代码：`ml/p3_10_ensemble.py`；组合事先写在 `ml/PREREG.md` 补充登记八
- 方法：每个成员的分数当天换成百分位名次，等权平均；不训练、不调权；两个组合都进测试段，只描述
- 产物：分数 `data/ml/scores/ens_top3__select.parquet`、`ens_family__select.parquet`；指标 `ml/results/p3_models/ens_*/`

## 一、结果

{tab.to_markdown(index=False)}

**成员逐日 IC 的相关系数**（越低，集成越可能有用）

{corr.to_markdown()}

**集成 − 最好的单个成员（按月整块重抽 95% 区间；仅供参考）**

{cmp_tab.to_markdown(index=False)}

## 二、读法

{notes}
""")


def main() -> None:
    if "--final" in sys.argv:
        keys = ev.load_keys()
        build("final", keys)          # 测试段按年重训的分数（正式）
        build("final_once", keys)     # 只训练一次的分数（对照，PREREG 补充登记十）
        return
    if "--report" not in sys.argv:
        build("select")
    report()


if __name__ == "__main__":
    main()
