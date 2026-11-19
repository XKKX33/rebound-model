"""任务 4.2 按组消融：哪类新信息有用（XGBoost 调参版，只看验证段）。

- 加一组：F28 + 某一组新特征（A ~ H），和 F28 比，看增量 IC；
- 去一组：F69 − 某一组新特征，和 F69 比，看损失；
- 所有消融用**同一套参数**（XGBoost 调参版 F69 选定的：树深 4、学习率 0.1、列抽样 0.5、行抽样 0.8），树数最多 2000、
  按验证段 IC 早停，3 个种子（0、1、2）分数取平均。这样差别只来自特征，不来自参数。
  基准“F69”就是 `xgb_F69` 的选择阶段结果；“F28（同参数）”另跑一次（`xgb_F28` 是按 F28 自己调的参数，不能直接比）。
- 比较：逐日 IC 差的均值 + 按月整块重抽 95% 区间（`ml/evaluate.compare`）。只在验证段做，不用于改动任何已冻结的模型；
  测试段只报两个最终模型（`xgb_F28`、`xgb_F69`）的结果，见总排行榜。

运行：python -X utf8 -m ml.p4_2_ablation             # 可续跑（已完成的组跳过）
      python -X utf8 -m ml.p4_2_ablation --report
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from ml import evaluate as ev
from ml.common import RESULTS, register, result_dir, write_md
from ml.dataset import F28
from ml.models import XGB

OUT = "p4_2_ablation"


def groups() -> dict[str, list[str]]:
    from ml.dataset import f69
    from ml.features_extra import GROUPS, feature_table
    t = feature_table().set_index("特征")
    out: dict[str, list[str]] = {}
    for f in f69():
        if f not in F28:
            out.setdefault(t.loc[f, "组"].split()[0], []).append(f)
    return {g: out[g] for g in GROUPS if g in out}


def model() -> XGB:
    from ml.p3_5_xgboost import EARLY_METRIC, PATIENCE, chosen, params_of
    c = chosen("F69")
    cfg = {k: c[k] for k in ("max_depth", "learning_rate", "colsample_bytree")}
    return XGB(params=params_of(cfg), seeds=(0, 1, 2), early_stop=PATIENCE, early_metric=EARLY_METRIC, name="ablation")


def configs() -> dict[str, list[str]]:
    from ml.dataset import f69
    g = groups()
    cfg = {"F28（同参数）": list(F28)}
    cfg.update({f"F28 + {k}": list(F28) + v for k, v in g.items()})
    cfg.update({f"F69 − {k}": [f for f in f69() if f not in v] for k, v in g.items()})
    return cfg


def run(log) -> None:
    from ml.train_eval import load, run as run_model
    out = result_dir(OUT)
    path = out / "ablation.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["配置"])
    rows = done.to_dict("records")
    keys, x = load("F69")
    for name, cols in configs().items():
        if (done["配置"] == name).any():
            continue
        t0 = time.time()
        r = run_model(f"abl_{name}", "select", model=model(), feature_set=name, keys=keys, x=x[cols], save=False)
        s = r["summary"]["全部"]
        r["daily"].to_csv(out / f"daily_{len(rows):02d}.csv", encoding="utf-8-sig")
        rows.append({"配置": name, "特征数": len(cols), "验证 IC": s["ic_mean"], "ICIR": s["icir"], "前6 超额%": s["top6"] * 100,
                     "前30 超额%": s["top30"] * 100, "各种子 IC": " / ".join(f"{v:.4f}" for v in r["seed_ic"]),
                     "种子极差": r["seed_spread"], "树数": " / ".join(map(str, r["model"]["best_rounds"])),
                     "日序列文件": f"daily_{len(rows):02d}.csv", "秒": round(time.time() - t0, 1)})
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        log(f"{name}（{len(cols)} 个特征）：验证 IC {s['ic_mean']:.4f}，种子 {np.round(r['seed_ic'], 4).tolist()}，{time.time() - t0:.0f}s")
        register("4.2_ablation", config=name, n_features=len(cols), valid_ic=s["ic_mean"])


def report() -> None:
    out = result_dir(OUT)
    tab = pd.read_csv(out / "ablation.csv")
    daily = {r["配置"]: pd.read_csv(out / r["日序列文件"], index_col=0, parse_dates=True) for _, r in tab.iterrows()}
    f69 = json.loads((RESULTS / "p3_models" / "xgb_F69" / "select.json").read_text(encoding="utf-8"))
    daily["F69"] = pd.read_csv(RESULTS / "p3_models" / "xgb_F69" / "select_daily.csv", index_col=0, parse_dates=True)
    s69 = f69["summary"]["全部"]
    base_row = {"配置": "F69（= xgb_F69）", "特征数": 69, "验证 IC": s69["ic_mean"], "ICIR": s69["icir"], "前6 超额%": s69["top6"] * 100,
                "前30 超额%": s69["top30"] * 100, "各种子 IC": " / ".join(f"{v:.4f}" for v in f69["seed_ic"]),
                "种子极差": f69["seed_spread"]}
    names = {g: f"{g} {n}" for g, n in __import__("ml.features_extra", fromlist=["GROUPS"]).GROUPS.items()}
    gsz = {g: len(v) for g, v in groups().items()}
    rows = []
    for _, r in tab.iterrows():
        name = r["配置"]
        if name.startswith("F28 + "):
            g, base = name[-1], "F28（同参数）"
            sign = 1
        elif name.startswith("F69 − "):
            g, base = name[-1], "F69"
            sign = -1
        else:
            continue
        c = ev.compare(daily[name]["ic"], daily[base]["ic"])
        t30 = ev.compare(daily[name]["top30"], daily[base]["top30"])
        rows.append({"组": names[g], "特征数": gsz[g], "做法": "加到 F28 上" if sign > 0 else "从 F69 去掉",
                     "IC 变化": round(c["diff"], 4), "95% 区间": f"[{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]",
                     "区间不含 0": "是" if c["ci"][0] > 0 or c["ci"][1] < 0 else "",
                     "前30 超额变化（%）": round(t30["diff"] * 100, 3)})
    eff = pd.DataFrame(rows)
    add = eff[eff["做法"] == "加到 F28 上"].set_index("组")
    drop = eff[eff["做法"] == "从 F69 去掉"].set_index("组")
    summary = pd.DataFrame({"加到 F28 上的 IC 增量": add["IC 变化"], "从 F69 去掉的 IC 损失": -drop["IC 变化"],
                            "加到 F28 上的前30 超额变化（%）": add["前30 超额变化（%）"]}).sort_values("加到 F28 上的 IC 增量", ascending=False)
    show = pd.concat([pd.DataFrame([base_row]), tab.drop(columns=["日序列文件", "秒"])], ignore_index=True)
    for c in ("验证 IC", "ICIR", "种子极差"):
        show[c] = show[c].astype(float).round(4)
    for c in ("前6 超额%", "前30 超额%"):
        show[c] = show[c].astype(float).round(3)
    notes_path = out / "notes.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "（读法待补）"
    write_md(out / "report.md", f"""
# 4.2 按组消融（验证段）

- 代码：`ml/p4_2_ablation.py`；结果 `ablation.csv`、每个配置的逐日序列 `daily_*.csv`
- 模型：XGBoost，所有配置用同一套参数（`xgb_F69` 选定的：树深 4、学习率 0.1、列抽样 0.5、行抽样 0.8），按验证段 IC 早停，3 个种子平均
- 只在验证段（2021 ~ 2022）上做，只用于回答“哪类信息有用”，不改动任何已冻结的模型

## 一、每组的作用

{summary.round(4).to_markdown()}

**加一组 / 去一组的详细比较**（按月整块重抽 95% 区间）

{eff.to_markdown(index=False)}

## 二、全部配置

{show.to_markdown(index=False)}

## 三、读法

{notes}
""")


def main() -> None:
    if "--report" not in sys.argv:
        out = result_dir(OUT)

        def log(msg):
            line = f"{time.strftime('%H:%M:%S')} {msg}"
            print(line, flush=True)
            with open(out / "run.log", "a", encoding="utf-8") as f:
                f.write(line + "\n")

        run(log)
    report()


if __name__ == "__main__":
    main()
