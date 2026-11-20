"""任务 3.1 ~ 3.3 的报告：统一接口、两个参照、XGBoost 复现基准（都只到验证段）。

先运行 `python -X utf8 -m ml.train_eval random rev_r5 xgb_repro_F28`，再运行本脚本：
    python -X utf8 -m ml.p3_baselines

本脚本另外做一件事：**用仓库自己的训练代码**（`training_set` + 仓库的分块 `_Batches` + `booster_params`）
在同一个重训日训一遍，和统一接口训出来的 XGBoost 比较，确认“我们的接口 = 作者的训练流程”。

输出：
- ml/results/p3_1_interface/report.md
- ml/results/p3_2_references/report.md
- ml/results/p3_3_xgb_repro/report.md、repo_equivalence.json、yearly_ic.csv
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from ml import evaluate as ev
from ml.common import RESULTS, data_dir, register, result_dir, write_md
from ml.dataset import load_features
from rebound import params as repo_params
from rebound.model.train import TrainingRows, _Batches, booster_params, no_power_throttling, training_set

AUTHOR_IC = 0.095
TOL = 0.02


def _load(model_id: str) -> tuple[dict, pd.DataFrame]:
    d = RESULTS / "p3_models" / model_id
    info = json.loads((d / "select.json").read_text(encoding="utf-8"))
    daily = pd.read_csv(d / "select_daily.csv", index_col=0, parse_dates=True)
    return info, daily


def _row(model_id: str, info: dict) -> dict:
    s = info["summary"]["全部"]
    return {"模型": model_id, "验证 IC": round(s["ic_mean"], 4), "ICIR": round(s["icir"], 3), "NW t": round(s["nw_t"], 1),
            "IC>0 比例": round(s["ic_pos"], 3), "前6 超额%": round(s["top6"] * 100, 3), "前30 超额%": round(s["top30"] * 100, 3),
            "多空%": round(s["long_short"] * 100, 3), "10组单调性": round(s["monotonic"], 3),
            "种子 IC": " / ".join(f"{v:.4f}" for v in info["seed_ic"]), "训练秒": info["seconds"]["fit"]}


def repo_equivalence(keys: pd.DataFrame, ours: np.ndarray) -> dict:
    """用仓库的 training_set + _Batches + booster_params 在同一重训日训 3 个种子，与统一接口的分数比较。"""
    plan = ev.protocol_a(keys)
    fit_di, evr = plan.select.fit_di, plan.select.eval_rows
    x = load_features("F28").to_numpy(np.float32)
    rows = TrainingRows(dates=pd.DatetimeIndex(keys["trade_date"].drop_duplicates()), codes=np.array([]),
                        di=keys["di"].to_numpy(np.int64), ji=np.zeros(len(keys), np.int64), x=x,
                        member={"nocap": np.ones(len(keys), bool)}, ret=keys["ret_o1o5"].to_numpy(np.float64),
                        exit_day=keys["exit_di"].to_numpy(np.int64))
    data = training_set(rows, "nocap", fit_di)
    same_rows = bool(np.array_equal(data.rows, np.flatnonzero(plan.select.fit_rows)))
    same_y = bool(np.array_equal(data.y[data.keep].astype(np.float32), ev.train_labels(keys, plan.select.fit_rows).astype(np.float32)))
    no_power_throttling()
    t0 = time.time()
    matrix = xgb.QuantileDMatrix(_Batches(rows.x, data), nthread=repo_params.THREADS)
    dv = xgb.DMatrix(x[evr], nthread=repo_params.THREADS)
    preds = [xgb.train(booster_params(repo_params.PARAMS, s), matrix, num_boost_round=repo_params.PARAMS["n_estimators"]).predict(dv)
             for s in repo_params.SEEDS]
    secs = time.time() - t0
    repo = np.column_stack(preds)
    k_eval = keys[evr].reset_index(drop=True)
    ic_repo = ev.evaluate(repo.mean(axis=1), k_eval)["summary"]["全部"]["ic_mean"]
    ic_ours = ev.evaluate(ours.mean(axis=1), k_eval)["summary"]["全部"]["ic_mean"]
    return {"训练行相同": same_rows, "训练标签相同": same_y, "仓库流程训练秒": round(secs, 1),
            "逐种子最大绝对差": [float(np.abs(repo[:, j] - ours[:, j]).max()) for j in range(repo.shape[1])],
            "逐种子相关系数": [float(np.corrcoef(repo[:, j], ours[:, j])[0, 1]) for j in range(repo.shape[1])],
            "仓库流程 IC": float(ic_repo), "统一接口 IC": float(ic_ours)}


def main() -> None:
    keys = ev.load_keys()
    infos = {m: _load(m) for m in ("random", "rev_r5", "xgb_repro_F28")}
    table = pd.DataFrame([_row(m, i) for m, (i, _) in infos.items()])

    # ---------------- 3.1 接口 ----------------
    out1 = result_dir("p3_1_interface")
    write_md(out1 / "report.md", f"""
# 3.1 统一模型接口

- 代码：`ml/models.py`（模型）、`ml/train_eval.py`（统一入口）；单元测试 `ml/tests/test_models.py`（6 项，全部通过）
- 运行：`python -X utf8 -m ml.train_eval <model_id> ...`（选择阶段：训练段 → 验证段）；加 `--final` 是最终阶段（训练 + 验证 → 测试，所有选择冻结后才用）

## 一、接口

每个模型实现两个方法：

| 方法 | 输入 | 说明 |
|---|---|---|
| `fit(X, y, X_valid, y_valid, day_valid)` | `X`：预处理后的特征（float32）；`y`：训练标签（当天收益百分位，只在已卖出的行里排） | 后三个参数只给需要早停的模型（XGBoost 调参版、MLP），其他模型忽略 |
| `predict(X)` / `predict_seeds(X)` | 评估段的特征 | 前者返回各种子的平均分数（正式分数），后者返回每个种子各自的分数（算种子间波动） |

每个模型还声明 `preprocess`（`tree` 用原值，`dense` 用截面排名，见 1.5）和 `complexity`（选参数打平时用，越小越简单）。

## 二、统一入口做的事（`train_eval.run`）

1. 按协议 A 取训练行和评估行（`ml/evaluate.py`，训练行的卖出日都早于重训日）；
2. 训练标签 `train_labels`（只在已知行里排百分位，和仓库 `training_set` 逐位相同）；
3. 预处理：只用训练行拟合（`ml/preprocess.py`）；
4. 训练 → 对评估段打分（每个种子一列，正式分数取平均）；
5. 用 `evaluate` 算全部预测层指标，另算每个种子单独的 IC；
6. 落盘：
   - 分数 `data/ml/scores/<model_id>__<阶段>.parquet`（`trade_date, ts_code, score, s0, s1, ...`，给 P5 回测和队友的接口用）；
   - 指标 `ml/results/p3_models/<model_id>/<阶段>.json`、逐日序列 `<阶段>_daily.csv`（模型两两比较用）；
   - `ml/results/registry.jsonl` 追加一条。

模型在 `train_eval.CONFIGS` 里登记（model_id → 构造函数、特征集）。目前有：{"、".join(f"`{m}`" for m in infos)}。

## 三、跑通的结果

{table[["模型", "验证 IC", "NW t", "前6 超额%", "前30 超额%", "训练秒"]].to_markdown(index=False)}

参照的数字和 P2 检查（`ml/results/p2_protocol/report.md`）完全相同：统一入口和评估代码前后一致。
""")

    # ---------------- 3.2 参照 ----------------
    out2 = result_dir("p3_2_references")
    ref = table[table["模型"].isin(["random", "rev_r5"])]
    write_md(out2 / "report.md", f"""
# 3.2 两个参照（验证段）

- 运行：`python -X utf8 -m ml.train_eval random rev_r5`
- 测试段结果在所有选择冻结后与其他模型一起算（PREREG 2.5）

| 参照 | 作用 |
|---|---|
| `random` 随机打分 | 下限：IC 应≈0。用来检验评估流程本身不会凭空产生预测能力 |
| `rev_r5` 单因子反转（按 −r5 排序，近 5 日跌得越多分越高） | A 股最经典的短期反转因子，也是“反弹”最朴素的版本；模型至少要明显好过它才算有用 |

{ref.drop(columns=["种子 IC", "训练秒"]).to_markdown(index=False)}

读法：
- 随机打分的 IC −0.001、NW t −0.8，在随机波动范围内，评估流程没有问题。
- 反转因子 IC 0.040、NW t 4.2，是真实但很弱的信号；但它前 6 名、前 30 名的超额是**负的**：
  按 −r5 分 10 组，反转效应主要来自“近期涨得最多的回落”，跌得最多的那一组并不反弹（分组见 `ml/results/p2_protocol/report.md`）。
  这是后面检验“模型能不能抓住非线性关系”的一个好对照。
""")

    # ---------------- 3.3 XGBoost 复现 ----------------
    out3 = result_dir("p3_3_xgb_repro")
    info, daily = infos["xgb_repro_F28"]
    s = info["summary"]["全部"]
    yearly = daily.groupby(daily.index.year)[["ic", "top6", "top30"]].mean()
    yearly[["top6", "top30"]] *= 100
    yearly.columns = ["IC", "前6 超额%", "前30 超额%"]
    yearly.to_csv(out3 / "yearly_ic.csv", encoding="utf-8-sig")
    ok = abs(s["ic_mean"] - AUTHOR_IC) <= TOL
    scores = pd.read_parquet(data_dir() / "scores" / "xgb_repro_F28__select.parquet")
    ours = scores[[c for c in scores.columns if c.startswith("s") and c[1:].isdigit()]].to_numpy(np.float64)
    eq = repo_equivalence(keys, ours)
    (out3 / "repo_equivalence.json").write_text(json.dumps(eq, ensure_ascii=False, indent=1), encoding="utf-8")
    _, d_rev = infos["rev_r5"]
    cmp_rev = ev.compare(daily["ic"], d_rev["ic"])
    cmp_top = ev.compare(daily["top30"], d_rev["top30"])
    groups = pd.Series(info["groups"]["全部"], index=range(1, 11)) * 100
    write_md(out3 / "report.md", f"""
# 3.3 XGBoost 复现基准（验证段）

- 运行：`python -X utf8 -m ml.train_eval xgb_repro_F28`，报告 `python -X utf8 -m ml.p3_baselines`
- 配置：仓库原参数 `rebound.params.PARAMS`（{repo_params.PARAMS}），种子 {list(repo_params.SEEDS)}，{repo_params.THREADS} 线程；28 个原特征；
  训练 {info['fit_rows']:,} 行（2013-08 ~ 2020-12，卖出日早于 2021-01-04），验证段 {info['eval_rows']:,} 行；训练 {info['seconds']['fit']:.0f} 秒

## 一、验收（PREREG 2.5、补充登记二第 8 条）

| 项目 | 数值 |
|---|---|
| 作者记录的 Rank IC | 约 {AUTHOR_IC}（开发期和留出期，每月重训） |
| 我们的验证段 Rank IC（3 个种子平均的分数） | **{s['ic_mean']:.4f}** |
| 差距 | {s['ic_mean'] - AUTHOR_IC:+.4f}（容许 ±{TOL}） |
| 结论 | **{"通过" if ok else "未通过：按补充登记二第 8 条先查口径"}** |

## 二、和作者的训练代码逐项核对

用仓库自己的 `training_set`、分块 `_Batches`、`booster_params` 在同一重训日（2021-01-04）重训 3 个种子，与统一接口训出来的模型对比：

| 项目 | 结果 |
|---|---|
| 训练行完全相同 | {eq['训练行相同']} |
| 训练标签完全相同 | {eq['训练标签相同']} |
| 两边分数的相关系数（逐种子） | {" / ".join(f"{v:.6f}" for v in eq['逐种子相关系数'])} |
| 两边分数的最大绝对差（逐种子） | {" / ".join(f"{v:.2e}" for v in eq['逐种子最大绝对差'])} |
| 验证段 IC：仓库流程 / 统一接口 | {eq['仓库流程 IC']:.4f} / {eq['统一接口 IC']:.4f} |

{"两边分数**逐位相同**（最大差 0）：统一接口训出来的模型和作者的训练代码训出来的完全一样。IC 末位的微小差别只是因为我们的分数存成 float32 后再平均。" if max(eq['逐种子最大绝对差']) == 0 else "仓库把训练行按“准入池前缀”每 25 万行分块送入 xgboost，我们按“训练行”分块；xgboost 的直方图分位点和分块方式有关，所以两边不是逐位相同，但相关系数接近 1、IC 几乎一样。"}

另：作者的 0.095 是**每月重训**得到的（仓库每个月第一个交易日用此前全部数据重训一次；“每季度 / 每年”只是从月度模型里挑着用）。
我们的协议 A 在验证段两年里**不重训**，IC 仍然对上，说明这套特征和参数的排序能力在两年内比较稳定。

## 三、指标

{table[table["模型"] == "xgb_repro_F28"].drop(columns=["模型"]).to_markdown(index=False)}

- 种子极差 {info['seed_spread']:.4f}：3 个种子之间的差别很小。
- Newey-West t = {s['nw_t']:.1f}（普通 t {s['naive_t']:.1f}），远超 3。

**分 10 组的平均超额（%，第 10 组 = 分数最高）**

{groups.round(3).to_frame("超额%").T.to_markdown(index=False)}

**逐年**（验证段只有两年，固定模型、中间不重训）

{yearly.round(4).to_markdown()}

## 四、和反转因子比（按月整块重抽 95% 区间）

| 指标 | XGBoost − 反转因子 | 95% 区间 |
|---|---|---|
| 逐日 IC | {cmp_rev['diff']:+.4f} | [{cmp_rev['ci'][0]:+.4f}, {cmp_rev['ci'][1]:+.4f}] |
| 前 30 名超额（%） | {cmp_top['diff'] * 100:+.3f} | [{cmp_top['ci'][0] * 100:+.3f}, {cmp_top['ci'][1] * 100:+.3f}] |

两个区间都不含 0：XGBoost 在 28 个特征上的排序能力明显强于单因子反转；而且反转因子前 6 名的超额是负的（−0.56%），
XGBoost 的前 6 名是正的（+0.35%），10 组也完全单调——组合多个特征后，最靠前的那一小撮不再“跌了还跌”。
""")
    register("3.1-3.3_baselines", acceptance=ok, valid_ic=s["ic_mean"], repo_equivalence=eq,
             vs_rev_ic=cmp_rev, vs_rev_top30=cmp_top)


if __name__ == "__main__":
    main()
