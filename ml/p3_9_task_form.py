"""任务 3.9 任务形式对比：XGBoost、F69，只改“学什么”——回归 / 二分类 / 排序（设置见 PREREG 补充登记七）。

回答“既然最后只用排名，直接学排名会不会更好”。
- 回归 `xgb_F69`：3.5 的结果，不重跑；
- 二分类 `xgb_clf_F69`：是否进当天前 20%（训练标签 > 0.8）；
- 排序 `xgb_rank_F69`：`rank:pairwise`，同一交易日为一组，成对抽样 "mean"（每行 1 对）。
参数全部沿用 `xgb_F69` 的选定参数，各自用验证段 IC 早停，3 个种子；最终阶段树数 = 3 个种子早停树数的平均。

运行：python -X utf8 -m ml.p3_9_task_form            # 训练分类、排序两个版本 + 报告
      python -X utf8 -m ml.p3_9_task_form --report   # 只重写报告
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from ml import evaluate as ev
from ml.common import RESULTS, data_dir, register, result_dir, write_md
from ml.models import XGB

OUT = "p3_9_task_form"
IDS = {"reg": "xgb_F69", "clf": "xgb_clf_F69", "rank": "xgb_rank_F69"}
NAMES = {"reg": "回归（收益百分位）", "clf": "二分类（是否进前 20%）", "rank": "排序（rank:pairwise）"}
RANK_EXTRA = {"lambdarank_pair_method": "mean", "lambdarank_num_pair_per_sample": 1, "ndcg_exp_gain": False}


def _chosen() -> dict:
    path = RESULTS / OUT / "chosen.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def model_for(obj: str, stage: str) -> XGB:
    """给 train_eval.CONFIGS 用。参数 = xgb_F69 的选定参数；选择阶段早停，最终阶段用早停树数的平均。"""
    from ml.p3_5_xgboost import EARLY_METRIC, PATIENCE, chosen, params_of
    c = chosen("F69")
    cfg = {k: c[k] for k in ("max_depth", "learning_rate", "colsample_bytree")}
    extra = RANK_EXTRA if obj == "rank" else {}
    if stage == "final":
        n = _chosen()[obj]["最终阶段树数"]
        return XGB(params={**params_of(cfg, n), **extra}, seeds=(0, 1, 2), objective=obj, name=IDS[obj])
    return XGB(params={**params_of(cfg), **extra}, seeds=(0, 1, 2), objective=obj, early_stop=PATIENCE,
               early_metric=EARLY_METRIC, name=IDS[obj])


def run_select(log) -> None:
    from ml.train_eval import load, run
    keys, x = load("F69")
    ch = _chosen()
    for obj in ("clf", "rank"):
        if obj in ch:                     # 已经跑完的形式不重跑（可续跑）
            continue
        t0 = time.time()
        r = run(IDS[obj], "select", model=model_for(obj, "select"), feature_set="F69", keys=keys, x=x, save=True)
        trees = r["model"]["best_rounds"]
        ch[obj] = {"各种子早停树数": trees, "最终阶段树数": int(round(float(np.mean(trees))))}
        (result_dir(OUT) / "chosen.json").write_text(json.dumps(ch, ensure_ascii=False, indent=1), encoding="utf-8")
        s = r["summary"]["全部"]
        log(f"[{obj}] 验证 IC {s['ic_mean']:.4f}，各种子 {np.round(r['seed_ic'], 4).tolist()}，树数 {trees}，"
            f"{time.time() - t0:.0f}s")
        register("3.9_task_form", objective=obj, valid_ic=s["ic_mean"], best_rounds=trees)
    from ml.p3_5_xgboost import chosen
    ch["reg"] = {"各种子早停树数": chosen("F69")["各种子树数"], "最终阶段树数": chosen("F69")["最终阶段树数"]}
    (result_dir(OUT) / "chosen.json").write_text(json.dumps(ch, ensure_ascii=False, indent=1), encoding="utf-8")


def report() -> None:
    out = result_dir(OUT)
    ch = _chosen()
    keys = ev.load_keys()
    st = ev.protocol_a(keys).select
    k_eval = keys[st.eval_rows].reset_index(drop=True)
    infos, res = {}, {}
    for obj, mid in IDS.items():
        infos[obj] = json.loads((RESULTS / "p3_models" / mid / "select.json").read_text(encoding="utf-8"))
        score = pd.read_parquet(data_dir() / "scores" / f"{mid}__select.parquet")["score"].to_numpy()
        res[obj] = ev.evaluate(score, k_eval, auc=True, ndcg=True)   # 三种形式都算 AUC 和 NDCG，便于横向比较

    rows = []
    for obj in IDS:
        s, info = res[obj]["summary"]["全部"], infos[obj]
        rows.append({"形式": NAMES[obj], "model_id": IDS[obj], "验证 IC": round(s["ic_mean"], 4), "ICIR": round(s["icir"], 3),
                     "NW t": round(s["nw_t"], 1), "前6 超额%": round(s["top6"] * 100, 3), "前30 超额%": round(s["top30"] * 100, 3),
                     "多空%": round(s["long_short"] * 100, 3), "10组单调性": round(s["monotonic"], 3),
                     "AUC（前20%）": round(s["auc"], 4), "NDCG@30": round(s["ndcg30"], 4),
                     "各种子 IC": " / ".join(f"{v:.4f}" for v in info["seed_ic"]), "种子极差": round(info["seed_spread"], 4),
                     "树数": " / ".join(map(str, ch[obj]["各种子早停树数"] or [])), "最终阶段树数": ch[obj]["最终阶段树数"],
                     "训练秒": info["seconds"]["fit"]})
    tab = pd.DataFrame(rows)

    groups = pd.DataFrame({NAMES[o]: res[o]["groups"]["全部"] * 100 for o in IDS}).T.round(3)
    groups.columns = range(1, 11)
    yearly = pd.DataFrame({IDS[o]: res[o]["daily"]["ic"].groupby(res[o]["daily"].index.year).mean() for o in IDS}).round(4)

    cmp_rows = []
    for a in ("clf", "rank"):
        for col, unit, label in (("ic", 1, "逐日 IC"), ("top30", 100, "前 30 超额（%）"), ("auc", 1, "AUC"), ("ndcg", 1, "NDCG@30")):
            c = ev.compare(res[a]["daily"][col], res["reg"]["daily"][col])
            row = {"比较": f"{NAMES[a]} − 回归", "指标": label, "差": round(c["diff"] * unit, 4),
                   "95% 区间": f"[{c['ci'][0] * unit:+.4f}, {c['ci'][1] * unit:+.4f}]"}
            if col == "ic":
                row["判定"] = ev.verdict(c, infos[a]["seed_spread"], infos["reg"]["seed_spread"])
            cmp_rows.append(row)
    cmp_tab = pd.DataFrame(cmp_rows).fillna("")
    corr = pd.DataFrame({NAMES[o]: res[o]["daily"]["ic"] for o in IDS}).corr().round(3)

    notes_path = out / "notes.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "（读法待补）"
    text = f"""
# 3.9 任务形式对比：回归 / 二分类 / 排序（验证段）

- 代码：`ml/p3_9_task_form.py`、`ml/models.py`（`XGB` 的 objective）；设置见 `ml/PREREG.md` 补充登记七
- 运行：`python -X utf8 -m ml.p3_9_task_form`；也可用统一入口 `python -X utf8 -m ml.train_eval xgb_clf_F69 xgb_rank_F69`
- 产物：`chosen.json`；分数 `data/ml/scores/xgb_clf_F69__select.parquet`、`xgb_rank_F69__select.parquet`
- 所有数字都只来自验证段（2021 ~ 2022）；测试段等所有选择冻结后统一跑（任务 3.11）

## 一、设置

只改“学什么”，其余和 XGBoost 调参版 F69 完全相同：F69 特征、选定参数（树深 4、学习率 0.1、列抽样 0.5、行抽样 0.8）、
树数最多 2000 且用验证段 IC 早停、3 个种子。

- **回归**：拟合收益百分位（现行做法，即 3.5 的 `xgb_F69`，没有重跑）；
- **二分类**：是否进当天前 20%（训练标签 > 0.8），`binary:logistic`；
- **排序**：`rank:pairwise`，每个交易日是一组，同一天里随机配对、学“谁比谁好”（`lambdarank_pair_method="mean"`，每行 1 对）。

参数是按回归调的，没有给分类和排序单独调参，所以这个比较对回归略有利。

## 二、结果

{tab.to_markdown(index=False)}

**分 10 组的平均超额（%，第 10 组 = 分数最高）**

{groups.to_markdown()}

**逐年 IC**

{yearly.to_markdown()}

**和回归比（按月整块重抽 95% 区间；IC 一行按 PREREG 2.4 给出判定，仅供参考，正式判定在测试段）**

{cmp_tab.to_markdown(index=False)}

**三种形式逐日 IC 的相关系数**（越接近 1，说明三者每天的好坏越同步，差别主要是整体水平）

{corr.to_markdown()}

## 三、读法

{notes}
"""
    write_md(out / "report.md", text)


def main() -> None:
    if "--report" not in sys.argv:
        out = result_dir(OUT)

        def log(msg):
            line = f"{time.strftime('%H:%M:%S')} {msg}"
            print(line, flush=True)
            with open(out / "run.log", "a", encoding="utf-8") as f:
                f.write(line + "\n")

        run_select(log)
    report()


if __name__ == "__main__":
    main()
