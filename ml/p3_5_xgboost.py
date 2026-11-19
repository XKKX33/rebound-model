"""任务 3.5 XGBoost 调参版：在验证段上调参（F28、F69 分别调），选出最优后跑 3 个种子并落盘。

网格在运行前写进了 `ml/results/tuning.md`（12 组，树数用早停）。
第一轮早停看验证段平方损失，停得太早、选出的调参版不如复现基准；第二轮（本文件现行设置）早停改看验证段逐日 IC，
网格不变（PREREG 补充登记六）。第一轮结果保留为 `round1_rmse_*` 和模型 `xgb_rmse_F28`、`xgb_rmse_F69`。
流程（PREREG 2.5）：
1. 每个特征集：12 组配置各训练一次（种子 0，早停），只看验证段 Rank IC；每跑完一组就追加到 `tuning_<特征集>.csv`（可续跑）；
2. 选择规则：验证 IC 最高；差不到 0.002 算打平，打平选“实际树数 × 2^树深”更小的；
3. 最优配置跑 3 个种子（0、1、2，各自早停），分数取平均，用统一入口落盘：model_id = `xgb_F28`、`xgb_F69`；
   最终阶段（测试段）重训时，树数取 3 个种子早停树数的平均（四舍五入），不再看任何验证数据；
4. 写报告 `ml/results/p3_5_xgboost/report.md`，并把结果追加到 `ml/results/tuning.md`。

运行：python -X utf8 -m ml.p3_5_xgboost            # 全流程
      python -X utf8 -m ml.p3_5_xgboost --report   # 只根据已有结果重写报告
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
from ml.models import XGB
from rebound import params as repo_params
from ml.train_eval import load, run

MAX_TREES, PATIENCE, EARLY_METRIC = 2000, 100, "ic"
GRID = [{"max_depth": d, "learning_rate": lr, "colsample_bytree": cs}
        for d, lr, cs in itertools.product((4, 6, 8), (0.03, 0.1), (0.5, 0.8))]
SETS = ("F28", "F69")
OUT = "p3_5_xgboost"


def params_of(cfg: dict, n_trees: int = MAX_TREES) -> dict:
    """完整的 xgboost 参数：复现基准的参数，改掉调的那几项。"""
    return {**repo_params.PARAMS, **cfg, "n_estimators": n_trees}


def tune(fs: str, keys, x, log) -> pd.DataFrame:
    path = result_dir(OUT) / f"tuning_{fs}.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    for i, cfg in enumerate(GRID):
        if len(done) and (done["组"] == i + 1).any():
            continue
        t0 = time.time()
        model = XGB(params=params_of(cfg), seeds=(0,), early_stop=PATIENCE, early_metric=EARLY_METRIC, name=f"xgb_{fs}_trial{i + 1}")
        r = run(model.name, "select", model=model, feature_set=fs, keys=keys, x=x, save=False)
        s = r["summary"]["全部"]
        trees = r["model"]["best_rounds"][0]
        rows.append({"组": i + 1, **cfg, "早停树数": trees, "验证 IC": s["ic_mean"], "ICIR": s["icir"], "NW t": s["nw_t"],
                     "前6 超额%": s["top6"] * 100, "前30 超额%": s["top30"] * 100,
                     "complexity": trees * 2 ** cfg["max_depth"], "训练秒": r["seconds"]["fit"], "总秒": round(time.time() - t0, 1)})
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        log(f"[{fs}] 第 {i + 1}/12 组 {cfg}：早停 {trees} 棵，验证 IC {s['ic_mean']:.4f}，训练 {r['seconds']['fit']:.0f}s")
        register("3.5_xgb_trial", feature_set=fs, trial=i + 1, params=cfg, best_rounds=trees, valid_ic=s["ic_mean"],
                 fit_seconds=r["seconds"]["fit"])
    return pd.DataFrame(rows)


def choose(tab: pd.DataFrame) -> dict:
    best = ev.pick_best([{"valid_ic": r["验证 IC"], "complexity": r["complexity"], "组": int(r["组"])} for _, r in tab.iterrows()])
    row = tab[tab["组"] == best["组"]].iloc[0]
    return {"组": int(row["组"]), "max_depth": int(row["max_depth"]), "learning_rate": float(row["learning_rate"]),
            "colsample_bytree": float(row["colsample_bytree"]), "早停树数（种子 0）": int(row["早停树数"]),
            "验证 IC（种子 0）": float(row["验证 IC"]), "最高验证 IC": float(tab["验证 IC"].max()),
            "打平的组": [int(g) for g in tab.loc[tab["验证 IC"] >= tab["验证 IC"].max() - ev.TIE_IC, "组"]]}


def chosen(fs: str) -> dict:
    return json.loads((RESULTS / OUT / "chosen.json").read_text(encoding="utf-8"))[fs]


def model_for(fs: str, stage: str) -> XGB:
    """给 train_eval.CONFIGS 用：选择阶段 3 个种子各自早停；最终阶段用 3 个种子早停树数的平均，不早停。"""
    c = chosen(fs)
    cfg = {k: c[k] for k in ("max_depth", "learning_rate", "colsample_bytree")}
    if stage == "final":
        return XGB(params=params_of(cfg, c["最终阶段树数"]), seeds=(0, 1, 2), name=f"xgb_{fs}")
    return XGB(params=params_of(cfg), seeds=(0, 1, 2), early_stop=PATIENCE, early_metric=EARLY_METRIC, name=f"xgb_{fs}")


def report() -> None:
    out = result_dir(OUT)
    ch = json.loads((out / "chosen.json").read_text(encoding="utf-8"))
    tabs = {fs: pd.read_csv(out / f"tuning_{fs}.csv") for fs in SETS}
    load_info = lambda m: json.loads((RESULTS / "p3_models" / m / "select.json").read_text(encoding="utf-8"))  # noqa: E731
    models = {"XGBoost 调参版 F28": "xgb_F28", "XGBoost 调参版 F69": "xgb_F69",
              "第一轮（损失早停）F28": "xgb_rmse_F28", "第一轮（损失早停）F69": "xgb_rmse_F69",
              "XGBoost 复现 F28（参照）": "xgb_repro_F28", "随机森林 F28": "rf_F28", "随机森林 F69": "rf_F69"}
    round1 = {fs: pd.read_csv(out / f"round1_rmse_tuning_{fs}.csv") for fs in SETS}
    infos = {k: load_info(m) for k, m in models.items()}
    daily = {m: pd.read_csv(RESULTS / "p3_models" / m / "select_daily.csv", index_col=0, parse_dates=True) for m in models.values()}

    def fmt(tab):
        t = tab.drop(columns=["complexity", "总秒"]).copy()
        for c in ("验证 IC", "ICIR"):
            t[c] = t[c].round(4)
        for c in ("NW t", "前6 超额%", "前30 超额%"):
            t[c] = t[c].round(2)
        t["训练秒"] = t["训练秒"].round(0).astype(int)
        return t.to_markdown(index=False)

    rows = []
    for name, info in infos.items():
        s = info["summary"]["全部"]
        rows.append({"模型": name, "验证 IC": round(s["ic_mean"], 4), "ICIR": round(s["icir"], 3), "NW t": round(s["nw_t"], 1),
                     "前6 超额%": round(s["top6"] * 100, 3), "前30 超额%": round(s["top30"] * 100, 3),
                     "多空%": round(s["long_short"] * 100, 3), "10组单调性": round(s["monotonic"], 3),
                     "各种子 IC": " / ".join(f"{v:.4f}" for v in info["seed_ic"]), "种子极差": round(info["seed_spread"], 4),
                     "树数": " / ".join(map(str, info["model"].get("best_rounds", []))) or "—", "训练秒": info["seconds"]["fit"]})
    final_tab = pd.DataFrame(rows)
    pairs = [("xgb_F69", "xgb_F28", "XGBoost 调参版 F69 − F28（新特征）"),
             ("xgb_F28", "xgb_repro_F28", "XGBoost 调参版 F28 − 复现基准（调参的作用）"),
             ("xgb_F69", "rf_F69", "XGBoost 调参版 F69 − 随机森林 F69（Boosting − Bagging）"),
             ("xgb_F28", "rf_F28", "XGBoost 调参版 F28 − 随机森林 F28")]
    cmp_rows = []
    for a, b, label in pairs:
        for col, unit in (("ic", 1), ("top30", 100)):
            c = ev.compare(daily[a][col], daily[b][col])
            cmp_rows.append({"比较": label, "指标": "逐日 IC" if col == "ic" else "前 30 超额（%）", "差": round(c["diff"] * unit, 4),
                             "95% 区间": f"[{c['ci'][0] * unit:+.4f}, {c['ci'][1] * unit:+.4f}]"})
    cmp_tab = pd.DataFrame(cmp_rows)
    yearly = pd.DataFrame({m: daily[m]["ic"].groupby(daily[m].index.year).mean() for m in models.values()}).round(4)
    groups = pd.DataFrame({fs: pd.Series(infos[f"XGBoost 调参版 {fs}"]["groups"]["全部"], index=range(1, 11)) * 100 for fs in SETS}).T.round(3)

    def pick_text(fs):
        c = ch[fs]
        tie = c["打平的组"]
        why = (f"第 {c['组']} 组验证 IC 最高" if len(tie) == 1 else
               f"第 {'、'.join(map(str, tie))} 组和最高值相差不到 0.002，按规则取“树数 × 2^树深”最小（最简单）的第 {c['组']} 组")
        return (f"- **{fs}**：{why}——`max_depth={c['max_depth']}`、`learning_rate={c['learning_rate']}`、"
                f"`colsample_bytree={c['colsample_bytree']}`，种子 0 早停在 {c['早停树数（种子 0）']} 棵（验证 IC {c['验证 IC（种子 0）']:.4f}）；"
                f"3 个种子早停树数 {c['各种子树数']}，最终阶段重训用 **{c['最终阶段树数']}** 棵")

    notes_path = out / "notes.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "（读法待补）"
    text = f"""
# 3.5 XGBoost 调参版（验证段）

- 代码：`ml/models.py` 的 `XGB`（早停）、`ml/p3_5_xgboost.py`（调参 + 最终训练）
- 运行：`python -X utf8 -m ml.p3_5_xgboost`；最终模型也可以用统一入口 `python -X utf8 -m ml.train_eval xgb_F28 xgb_F69` 复现
- 产物：`tuning_F28.csv`、`tuning_F69.csv`、`chosen.json`；分数 `data/ml/scores/xgb_F28__select.parquet`、`xgb_F69__select.parquet`
- 所有数字都只来自验证段（2021 ~ 2022）；测试段等所有模型的选择冻结后统一跑（任务 3.11）

## 一、设置

- 和复现基准相同：`reg:squarederror` 拟合收益百分位，原值输入，`subsample=0.8`，hist，16 线程。
- 调参：`max_depth` × `learning_rate` × `colsample_bytree` 共 12 组；树数最多 {MAX_TREES}，**验证段逐日 IC 连续 {PATIENCE} 轮不提高就早停**。
  网格在运行前写进了 `ml/results/tuning.md`。调参只用种子 0。
- **一共跑了两轮**：第一轮早停看验证段平方损失（XGBoost 的默认做法），结果大多在 15 ~ 300 棵树就停了，
  选出的调参版（F28 0.0929、F69 0.0950）反而不如不调参的复现基准（0.0946），种子之间也差得多。原因是早停标准和选择标准不一致：
  平方损失不再下降时，排序能力还在提高。第二轮把早停改为看验证段 IC（和 LSTM 的早停规则一致），网格不变。
  改动写在 `ml/PREREG.md` 补充登记六；第一轮结果全部保留（第五节）。下面第二、三节是第二轮。

## 二、调参结果（第二轮，早停看 IC，种子 0）

### F28

{fmt(tabs['F28'])}

### F69

{fmt(tabs['F69'])}

### 选定

{pick_text('F28')}
{pick_text('F69')}

## 三、最终模型（选定参数 × 3 个种子，各自早停，分数取平均）

{final_tab.to_markdown(index=False)}

**分 10 组的平均超额（%，第 10 组 = 分数最高）**

{groups.to_markdown()}

**逐年 IC**

{yearly.to_markdown()}

**比较（按月整块重抽 95% 区间；仅供参考，正式判定在测试段，见 PREREG 2.4）**

{cmp_tab.to_markdown(index=False)}

## 四、读法

{notes}

## 五、第一轮（早停看平方损失，不参与选择）

### F28

{fmt(round1['F28'])}

### F69

{fmt(round1['F69'])}

第一轮选出的配置跑 3 个种子后的结果见第三节表中“第一轮（损失早停）”两行；完整报告 `round1_rmse_report.md`。
"""
    write_md(out / "report.md", text)
    upsert_section(RESULTS / "tuning.md", "### 3.5 第二轮：结果（2026-11-18，用于选择）", f"""
第二轮共试 {len(tabs['F28']) + len(tabs['F69'])} 组（F28、F69 各 12 组）；连同第一轮，XGBoost 调参版共试 48 组。详细见 `ml/results/p3_5_xgboost/report.md`。

| 特征集 | 最高验证 IC（种子 0） | 与最高相差 < 0.002 的组 | 选定（最简单的那组） | 选定组的验证 IC（种子 0） | 3 个种子平均后的验证 IC | 最终阶段树数 |
|---|---|---|---|---|---|---|
""" + "\n".join(
        f"| {fs} | {ch[fs]['最高验证 IC']:.4f} | {'、'.join(map(str, ch[fs]['打平的组']))} | 第 {ch[fs]['组']} 组：深 {ch[fs]['max_depth']}、学习率 {ch[fs]['learning_rate']}、列抽样 {ch[fs]['colsample_bytree']} | {ch[fs]['验证 IC（种子 0）']:.4f} | {infos[f'XGBoost 调参版 {fs}']['summary']['全部']['ic_mean']:.4f} | {ch[fs]['最终阶段树数']} |"
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

    path = out / "chosen.json"
    ch = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for fs in SETS:
        keys, x = load(fs)
        tab = tune(fs, keys, x, log)
        ch[fs] = choose(tab)
        path.write_text(json.dumps(ch, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"[{fs}] 选定 {ch[fs]}")
        r = run(f"xgb_{fs}", "select", model=model_for(fs, "select"), feature_set=fs, keys=keys, x=x, save=True)
        ch[fs]["各种子树数"] = r["model"]["best_rounds"]
        ch[fs]["最终阶段树数"] = int(round(float(np.mean(r["model"]["best_rounds"]))))
        path.write_text(json.dumps(ch, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"[{fs}] 最终 3 种子：验证 IC {r['summary']['全部']['ic_mean']:.4f}，各种子 {np.round(r['seed_ic'], 4).tolist()}，"
            f"树数 {r['model']['best_rounds']}，训练 {r['seconds']['fit']:.0f}s")
        del keys, x
    report()


if __name__ == "__main__":
    main()
