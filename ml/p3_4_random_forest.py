"""任务 3.4 随机森林：在验证段上调参（F28、F69 分别调），选出最优后跑 3 个种子并落盘。

网格在运行前写进了 `ml/results/tuning.md`（原 12 组 + 冻结前补的 6 组，补充登记九）。流程（PREREG 2.5、补充登记四）：
1. 每个特征集：18 组配置各训练一次（种子 0），只看验证段 Rank IC；每跑完一组就追加到 `tuning_<特征集>.csv`（中断后可续跑）；
2. 按选择规则挑最优（验证 IC 最高；差不到 0.002 算打平，打平选预期叶子数更少的）；
3. 最优配置跑 3 个种子（0、1、2），分数取平均，用统一入口落盘：model_id = `rf_F28`、`rf_F69`；
4. 写报告 `ml/results/p3_4_random_forest/report.md`，并把结果追加到 `ml/results/tuning.md`。

运行：python -X utf8 -m ml.p3_4_random_forest            # 全流程（约 1.5 小时）
      python -X utf8 -m ml.p3_4_random_forest --report   # 只根据已有结果重写报告
"""

from __future__ import annotations

import itertools
import json
import sys
import time

import numpy as np
import pandas as pd

from ml import evaluate as ev
from ml.common import RESULTS, register, result_dir, upsert_section, write_md
from ml.models import RF_FIXED, RandomForest
from ml.train_eval import load, run

GRID = [{"max_samples": ms, "max_features": mf, "min_samples_leaf": leaf}
        for ms, mf, leaf in itertools.product((0.05, 0.15), (0.2, 0.4), (50, 200, 1000))]
#: 冻结前补的 6 组（第 13 ~ 18 组，PREREG 补充登记九）：原网格的最好结果在“叶子小、抽样多”的边上，往外补
GRID += [{"max_samples": ms, "max_features": mf, "min_samples_leaf": leaf}
         for ms, mf, leaf in [(0.15, 0.2, 20), (0.15, 0.4, 20), (0.3, 0.2, 50), (0.3, 0.4, 50), (0.3, 0.2, 20), (0.3, 0.4, 20)]]
SETS = ("F28", "F69")
OUT = "p3_4_random_forest"


def tune(fs: str, keys, x, log) -> pd.DataFrame:
    path = result_dir(OUT) / f"tuning_{fs}.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    for i, cfg in enumerate(GRID):
        if len(done) and ((done["组"] == i + 1).any()):
            continue
        t0 = time.time()
        r = run(f"rf_{fs}_trial{i + 1}", "select", model=RandomForest(params=cfg, seeds=(0,)), feature_set=fs,
                keys=keys, x=x, save=False)
        s = r["summary"]["全部"]
        rows.append({"组": i + 1, **cfg, "验证 IC": s["ic_mean"], "ICIR": s["icir"], "NW t": s["nw_t"],
                     "前6 超额%": s["top6"] * 100, "前30 超额%": s["top30"] * 100,
                     "平均叶子数": r["model"]["mean_leaves"][0], "complexity": cfg["max_samples"] / cfg["min_samples_leaf"],
                     "训练秒": r["seconds"]["fit"], "总秒": round(time.time() - t0, 1)})
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        log(f"[{fs}] 第 {i + 1}/{len(GRID)} 组 {cfg}：验证 IC {s['ic_mean']:.4f}，训练 {r['seconds']['fit']:.0f}s")
        register("3.4_rf_trial", feature_set=fs, trial=i + 1, params=cfg, valid_ic=s["ic_mean"], fit_seconds=r["seconds"]["fit"])
    return pd.DataFrame(rows)


def choose(tab: pd.DataFrame) -> dict:
    best = ev.pick_best([{"valid_ic": r["验证 IC"], "complexity": r["complexity"], "组": int(r["组"])} for _, r in tab.iterrows()])
    row = tab[tab["组"] == best["组"]].iloc[0]
    return {"组": int(row["组"]), "max_samples": float(row["max_samples"]), "max_features": float(row["max_features"]),
            "min_samples_leaf": int(row["min_samples_leaf"]), "验证 IC（种子 0）": float(row["验证 IC"]),
            "最高验证 IC": float(tab["验证 IC"].max()), "打平的组": [int(g) for g in tab.loc[tab["验证 IC"] >= tab["验证 IC"].max() - ev.TIE_IC, "组"]]}


def chosen_params(fs: str) -> dict:
    """已选定的参数（给 train_eval.CONFIGS 用）。"""
    c = json.loads((RESULTS / OUT / "chosen.json").read_text(encoding="utf-8"))[fs]
    return {k: c[k] for k in ("max_samples", "max_features", "min_samples_leaf")}


def report() -> None:
    out = result_dir(OUT)
    chosen = json.loads((out / "chosen.json").read_text(encoding="utf-8"))
    tabs = {fs: pd.read_csv(out / f"tuning_{fs}.csv") for fs in SETS}
    finals = {fs: json.loads((RESULTS / "p3_models" / f"rf_{fs}" / "select.json").read_text(encoding="utf-8")) for fs in SETS}
    daily = {m: pd.read_csv(RESULTS / "p3_models" / m / "select_daily.csv", index_col=0, parse_dates=True)
             for m in ("rf_F28", "rf_F69", "xgb_repro_F28", "rev_r5")}
    repro = json.loads((RESULTS / "p3_models" / "xgb_repro_F28" / "select.json").read_text(encoding="utf-8"))

    def fmt(tab):
        t = tab.drop(columns=["complexity", "总秒"]).copy()
        for c in ("验证 IC", "ICIR"):
            t[c] = t[c].round(4)
        for c in ("NW t", "前6 超额%", "前30 超额%"):
            t[c] = t[c].round(2)
        t["平均叶子数"] = t["平均叶子数"].round(0).astype(int)
        t["训练秒"] = t["训练秒"].round(0).astype(int)
        return t.to_markdown(index=False)

    rows = []
    for name, info in (("随机森林 F28", finals["F28"]), ("随机森林 F69", finals["F69"]), ("XGBoost 复现 F28（参照）", repro)):
        s = info["summary"]["全部"]
        rows.append({"模型": name, "验证 IC": round(s["ic_mean"], 4), "ICIR": round(s["icir"], 3), "NW t": round(s["nw_t"], 1),
                     "前6 超额%": round(s["top6"] * 100, 3), "前30 超额%": round(s["top30"] * 100, 3),
                     "多空%": round(s["long_short"] * 100, 3), "10组单调性": round(s["monotonic"], 3),
                     "各种子 IC": " / ".join(f"{v:.4f}" for v in info["seed_ic"]), "种子极差": round(info["seed_spread"], 4),
                     "训练秒": info["seconds"]["fit"]})
    final_tab = pd.DataFrame(rows)
    cmps = {
        "随机森林 F69 − 随机森林 F28": ev.compare(daily["rf_F69"]["ic"], daily["rf_F28"]["ic"]),
        "随机森林 F28 − XGBoost 复现 F28": ev.compare(daily["rf_F28"]["ic"], daily["xgb_repro_F28"]["ic"]),
        "随机森林 F28 − 反转因子": ev.compare(daily["rf_F28"]["ic"], daily["rev_r5"]["ic"]),
    }
    cmp_tab = pd.DataFrame([{"比较（逐日 IC 差）": k, "差": round(v["diff"], 4), "95% 区间": f"[{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]"}
                            for k, v in cmps.items()])
    yearly = pd.DataFrame({m: daily[m]["ic"].groupby(daily[m].index.year).mean() for m in ("rf_F28", "rf_F69", "xgb_repro_F28")}).round(4)
    groups = pd.DataFrame({fs: pd.Series(finals[fs]["groups"]["全部"], index=range(1, 11)) * 100 for fs in SETS}).T.round(3)

    def pick_text(fs):
        c = chosen[fs]
        tie = c["打平的组"]
        why = (f"第 {c['组']} 组验证 IC 最高" if len(tie) == 1 else
               f"第 {'、'.join(map(str, tie))} 组和最高值相差不到 0.002，按规则取预期叶子数最少（最简单）的第 {c['组']} 组")
        return (f"- **{fs}**：{why}——`max_samples={c['max_samples']}`、`max_features={c['max_features']}`、"
                f"`min_samples_leaf={c['min_samples_leaf']}`（种子 0 的验证 IC {c['验证 IC（种子 0）']:.4f}）")

    notes_path = out / "notes.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "（读法待补）"
    text = f"""
# 3.4 随机森林（验证段）

- 代码：`ml/models.py` 的 `RandomForest`（scikit-learn `RandomForestRegressor`）、`ml/p3_4_random_forest.py`（调参 + 最终训练）；单元测试 `ml/tests/test_models.py::test_random_forest`
- 运行：`python -X utf8 -m ml.p3_4_random_forest`；最终模型也可以单独用统一入口 `python -X utf8 -m ml.train_eval rf_F28 rf_F69` 复现
- 产物：`tuning_F28.csv`、`tuning_F69.csv`（每组的结果）、`chosen.json`（选定的参数）；分数 `data/ml/scores/rf_F28__select.parquet`、`rf_F69__select.parquet`
- 所有数字都只来自验证段（2021 ~ 2022）；测试段等所有模型的选择冻结后统一跑（任务 3.11）

## 一、设置

- 固定：{RF_FIXED['n_estimators']} 棵树、有放回抽样（bootstrap）、{RF_FIXED['n_jobs']} 线程；原值输入（缺失值由 sklearn 处理）；目标 = 训练标签（收益百分位）。
- 调参：`max_samples`（每棵树抽多少比例的训练行）× `max_features`（每次分裂看多少比例的特征）× `min_samples_leaf`（叶子最少样本数），
  原网格 12 组；原网格最好的结果落在边上，冻结前又往外补了 6 组（第 13 ~ 18 组：叶子 20、每棵树抽 30%，PREREG 补充登记九），共 18 组。
  F28、F69 各调一遍；网格都在运行前写进了 `ml/results/tuning.md`。调参只用种子 0；原网格 12 组时的选择见 `chosen_grid12.json`。
- 为什么要抽样（`max_samples`）：训练段 192 万行，每棵树都用全量又深又慢；而且 Bagging 本来就是“每棵树看一部分数据，再平均”，
  抽得少一些，树和树之间更不相关，平均后方差反而更小。

## 二、调参结果（种子 0）

### F28（28 个原特征）

{fmt(tabs['F28'])}

### F69（加入新特征）

{fmt(tabs['F69'])}

### 选定

{pick_text('F28')}
{pick_text('F69')}

## 三、最终模型（选定参数 × 3 个种子，分数取平均）

{final_tab.to_markdown(index=False)}

**分 10 组的平均超额（%，第 10 组 = 分数最高）**

{groups.to_markdown()}

**逐年 IC**

{yearly.to_markdown()}

**比较（按月整块重抽 95% 区间；仅供参考，正式判定在测试段，见 PREREG 2.4）**

{cmp_tab.to_markdown(index=False)}

## 四、读法

{notes}
"""
    write_md(out / "report.md", text)
    upsert_section(RESULTS / "tuning.md", "### 3.4 补网格后：结果（用于选择）", f"""
共试 {len(tabs['F28']) + len(tabs['F69'])} 组（F28、F69 各 {len(GRID)} 组），详细见 `ml/results/p3_4_random_forest/report.md` 和 `tuning_F28.csv`、`tuning_F69.csv`。

| 特征集 | 最高验证 IC（种子 0） | 与最高相差 < 0.002 的组 | 选定（最简单的那组） | 选定组的验证 IC（种子 0） | 3 个种子平均后的验证 IC |
|---|---|---|---|---|---|
""" + "\n".join(
        f"| {fs} | {chosen[fs]['最高验证 IC']:.4f} | {'、'.join(map(str, chosen[fs]['打平的组']))} | 第 {chosen[fs]['组']} 组：`max_samples={chosen[fs]['max_samples']}`、`max_features={chosen[fs]['max_features']}`、`min_samples_leaf={chosen[fs]['min_samples_leaf']}` | {chosen[fs]['验证 IC（种子 0）']:.4f} | {finals[fs]['summary']['全部']['ic_mean']:.4f} |"
        for fs in SETS))


def main() -> None:
    if "--report" in sys.argv:
        report()
        return
    out = result_dir(OUT)
    log_path = out / "run.log"

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    chosen_path = out / "chosen.json"
    chosen = json.loads(chosen_path.read_text(encoding="utf-8")) if chosen_path.exists() else {}
    for fs in SETS:
        keys, x = load(fs)
        tab = tune(fs, keys, x, log)
        chosen[fs] = choose(tab)
        chosen_path.write_text(json.dumps(chosen, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"[{fs}] 选定 {chosen[fs]}")
        r = run(f"rf_{fs}", "select", model=RandomForest(params=chosen_params(fs), seeds=(0, 1, 2), name=f"rf_{fs}"),
                feature_set=fs, keys=keys, x=x, save=True)
        log(f"[{fs}] 最终 3 种子：验证 IC {r['summary']['全部']['ic_mean']:.4f}，各种子 {np.round(r['seed_ic'], 4).tolist()}，训练 {r['seconds']['fit']:.0f}s")
        del keys, x
    report()


if __name__ == "__main__":
    main()
