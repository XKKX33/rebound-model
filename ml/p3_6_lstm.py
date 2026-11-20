"""任务 3.6 LSTM：调参（S、S+F69 两个版本分别调）→ 选定后跑 3 个种子 → 用统一的评估落盘。

分工：主环境（本文件）准备数据（`ml/lstm_prep.py`）、调度、评估、写报告；训练在 dl-study 环境（`ml/lstm_train.py`），
两边只通过 .npy 交换。网格在运行前写进 `ml/results/tuning.md`。规则见 PREREG 补充登记三：
调参只用种子 0；每个 epoch 末算验证段 IC，连续 5 个 epoch 不提高就停；选择看验证段 Rank IC（由 `ml/evaluate.py` 计算）；
打平（差 < 0.002）选参数个数更少的；选定后跑 3 个种子，最终阶段的 epoch 数 = 3 个种子最佳 epoch 的平均（四舍五入）。

第二轮（现行，PREREG 补充登记九）：网格不变，另加学习率余弦衰减（20 个 epoch 一个周期，最多 20 个 epoch）和权重的指数滑动平均（0.999），
见 `ROUND2`。第一轮（无这两项）的结果保留为 `round1_*` 和模型 `lstm_r1_S`、`lstm_r1_SF69`，不参与选择。

model_id：`lstm_S`（只用 60 天序列）、`lstm_SF69`（序列 + F69 手工特征）。

运行：python -X utf8 -m ml.p3_6_lstm              # 选择阶段全流程
      python -X utf8 -m ml.p3_6_lstm --report     # 只重写报告
      python -X utf8 -m ml.p3_6_lstm --final      # 测试段（按年重训），所有选择冻结后才用（任务 3.11）
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from ml import evaluate as ev
from ml.common import RESULTS, ROOT, data_dir, register, result_dir, upsert_section, write_md
from ml.train_eval import score_and_save

DL_PY = r"D:\Anaconda\envs\dl-study\python.exe"
OUT = "p3_6_lstm"
VARIANTS = {"S": "lstm_S", "SF": "lstm_SF69"}
#: 第二轮的训练方法（PREREG 补充登记九）：加到每一组配置上；最多 20 个 epoch（= 余弦周期）
ROUND2 = {"ema": 0.999, "tmax": 20}
MAX_EPOCHS = 20
VARIANT_NAMES = {"S": "LSTM-S（只用序列）", "SF": "LSTM-S+F69（序列 + 手工特征）"}
#: 事先列出的 12 组（PREREG 补充登记三）：隐藏维度、层数、dropout、学习率、权重衰减、每个 epoch 抽的交易日比例
GRID = [
    {"hidden": 32, "layers": 1, "dropout": 0.0, "lr": 1e-3, "wd": 0.0, "frac": 0.5},
    {"hidden": 64, "layers": 1, "dropout": 0.0, "lr": 1e-3, "wd": 0.0, "frac": 0.5},
    {"hidden": 128, "layers": 1, "dropout": 0.0, "lr": 1e-3, "wd": 0.0, "frac": 0.5},
    {"hidden": 64, "layers": 2, "dropout": 0.1, "lr": 1e-3, "wd": 0.0, "frac": 0.5},
    {"hidden": 128, "layers": 2, "dropout": 0.1, "lr": 1e-3, "wd": 0.0, "frac": 0.5},
    {"hidden": 64, "layers": 1, "dropout": 0.0, "lr": 3e-4, "wd": 0.0, "frac": 0.5},
    {"hidden": 64, "layers": 2, "dropout": 0.1, "lr": 3e-4, "wd": 0.0, "frac": 0.5},
    {"hidden": 64, "layers": 1, "dropout": 0.0, "lr": 1e-3, "wd": 1e-4, "frac": 0.5},
    {"hidden": 64, "layers": 2, "dropout": 0.3, "lr": 1e-3, "wd": 1e-4, "frac": 0.5},
    {"hidden": 64, "layers": 1, "dropout": 0.0, "lr": 1e-3, "wd": 0.0, "frac": 0.25},
    {"hidden": 64, "layers": 2, "dropout": 0.1, "lr": 1e-3, "wd": 0.0, "frac": 1.0},
    {"hidden": 128, "layers": 2, "dropout": 0.3, "lr": 3e-4, "wd": 1e-4, "frac": 0.5},
]


def train(stage: str, variant: str, cfg: dict, seeds: str, out_name: str, epochs: int | None = None,
          data: str | None = None, save_model: bool = False) -> tuple[np.ndarray, dict]:
    out = data_dir() / "lstm" / "runs" / out_name
    cmd = [DL_PY, "-X", "utf8", str(ROOT / "ml" / "lstm_train.py"), "--stage", stage, "--variant", variant,
           "--cfg", json.dumps({**cfg, **ROUND2}), "--seeds", seeds, "--max-epochs", str(MAX_EPOCHS), "--out", str(out)]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if data is not None:
        cmd += ["--data", data]
    if save_model:
        cmd += ["--save-model"]
    # 笔记本显卡同时给其他程序用，偶尔会临时显存不足（第 5 组第一次就是这样，单独重跑峰值只有 4.6 GB）：
    # 遇到 out of memory 等 60 秒、同一种子重来，最多 3 次；其他错误直接报。
    env = {**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    for attempt in range(3):
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=env)
        if r.returncode == 0 or "out of memory" not in r.stderr:
            break
        time.sleep(60)
    if r.returncode != 0:
        raise RuntimeError(f"LSTM 训练失败：{r.stderr[-2000:]}")
    return np.load(str(out) + ".npy"), json.loads(open(str(out) + ".json", encoding="utf-8").read())


def tune(variant: str, keys, st, log) -> pd.DataFrame:
    path = result_dir(OUT) / f"tuning_{variant}.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    for i, cfg in enumerate(GRID):
        if len(done) and (done["组"] == i + 1).any():
            continue
        S, meta = train("select", variant, cfg, "0", f"{variant}_trial{i + 1}")
        run0 = meta["runs"][0]
        r = score_and_save(f"lstm_{variant}_trial{i + 1}", "select", st, keys, S, {}, save=False)
        s = r["summary"]["全部"]
        rows.append({"组": i + 1, **cfg, "参数个数": run0["params"], "最佳 epoch": run0["best_epoch"], "跑了 epoch": run0["epochs_run"],
                     "验证 IC": s["ic_mean"], "ICIR": s["icir"], "NW t": s["nw_t"], "前6 超额%": s["top6"] * 100,
                     "前30 超额%": s["top30"] * 100, "训练秒": run0["seconds"], "显存 GB": meta["peak_mem_gb"]})
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        log(f"[{variant}] 第 {i + 1}/12 组 {cfg}：最佳 epoch {run0['best_epoch']}/{run0['epochs_run']}，验证 IC {s['ic_mean']:.4f}，{run0['seconds']:.0f}s")
        register("3.6_lstm_trial", variant=variant, trial=i + 1, cfg=cfg, valid_ic=s["ic_mean"], best_epoch=run0["best_epoch"],
                 seconds=run0["seconds"])
    return pd.DataFrame(rows)


def choose(tab: pd.DataFrame) -> dict:
    best = ev.pick_best([{"valid_ic": r["验证 IC"], "complexity": r["参数个数"], "组": int(r["组"])} for _, r in tab.iterrows()])
    row = tab[tab["组"] == best["组"]].iloc[0]
    return {"组": int(row["组"]), "cfg": GRID[int(row["组"]) - 1], "参数个数": int(row["参数个数"]),
            "验证 IC（种子 0）": float(row["验证 IC"]), "最高验证 IC": float(tab["验证 IC"].max()),
            "打平的组": [int(g) for g in tab.loc[tab["验证 IC"] >= tab["验证 IC"].max() - ev.TIE_IC, "组"]]}


def run_select(log) -> None:
    keys = ev.load_keys()
    st = ev.protocol_a(keys).select
    if not (data_dir() / "lstm" / "select" / "meta.json").exists():
        from ml.lstm_prep import prepare
        prepare("select")
    path = result_dir(OUT) / "chosen.json"
    ch = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for v in VARIANTS:
        tab = tune(v, keys, st, log)
        ch[v] = choose(tab)
        path.write_text(json.dumps(ch, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"[{v}] 选定 {ch[v]}")
        S, meta = train("select", v, ch[v]["cfg"], "0,1,2", f"{VARIANTS[v]}__select")
        epochs = [r["best_epoch"] for r in meta["runs"]]
        info = {"feature_set": "S" if v == "S" else "S+F69", "model": {"name": VARIANTS[v], "cfg": ch[v]["cfg"], "seeds": [0, 1, 2],
                "best_epochs": epochs, "params": meta["runs"][0]["params"], "peak_mem_gb": meta["peak_mem_gb"]},
                "complexity": meta["runs"][0]["params"], "seconds": {"fit": round(sum(r["seconds"] for r in meta["runs"]), 1)}}
        r = score_and_save(VARIANTS[v], "select", st, keys, S, info, save=True)
        ch[v]["各种子最佳 epoch"] = epochs
        ch[v]["最终阶段 epoch"] = int(round(float(np.mean(epochs))))
        path.write_text(json.dumps(ch, ensure_ascii=False, indent=1), encoding="utf-8")
        (result_dir(OUT) / f"{VARIANTS[v]}_curves.json").write_text(
            json.dumps([r_["curve"] for r_ in meta["runs"]], ensure_ascii=False), encoding="utf-8")
        log(f"[{v}] 最终 3 种子：验证 IC {r['summary']['全部']['ic_mean']:.4f}，各种子 {np.round(r['seed_ic'], 4).tolist()}，最佳 epoch {epochs}")


def run_final(variants: tuple[str, ...] = ("S", "SF")) -> None:
    """测试段：按年重训（PREREG 补充登记十）。只在所有选择冻结后（任务 3.11）调用。

    每年第一个交易日用此前全部已卖出的行重训（参数、epoch 数固定，不看任何测试数据），模型给全部测试行打分；
    正式分数（`final`）每年只取这一年的行拼起来；2023 年那次的完整分数是“只训练一次”的对照（`final_once`）。
    """
    from ml.lstm_prep import prepare
    from ml.train_eval import check_keys, keys_fingerprint
    keys = ev.load_keys()
    ref = keys_fingerprint(keys)
    plan = ev.protocol_a(keys)
    test_pos = np.flatnonzero(plan.final.eval_rows)
    ch = json.loads((RESULTS / OUT / "chosen.json").read_text(encoding="utf-8"))
    for st in plan.refits:
        if not (data_dir() / "lstm" / f"final_{st.name}" / "meta.json").exists():
            prepare(f"final_{st.name}")
    for v, mid in VARIANTS.items():
        if v not in variants:
            continue
        S_final, S_once, refits = None, None, []
        for st in plan.refits:
            S, meta = train("final", v, ch[v]["cfg"], "0,1,2", f"{mid}__final_{st.name}", epochs=ch[v]["最终阶段 epoch"],
                            data=f"final_{st.name}")
            if S_final is None:
                S_final, S_once = np.full(S.shape, np.nan), S.copy()
            mine = st.eval_rows[test_pos]
            S_final[mine] = S[mine]
            refits.append({"year": st.name, "fit_rows": int(st.fit_rows.sum()), "eval_rows": int(mine.sum()),
                           "seconds": round(sum(r["seconds"] for r in meta["runs"]), 1), "peak_mem_gb": meta["peak_mem_gb"]})
            print(f"  [{mid}] {st.name} 年重训完成", flush=True)
        if not np.isfinite(S_final).all():
            raise RuntimeError("按年重训的分数没有覆盖全部测试行，或有无穷值")
        check_keys(keys, ref, f"{mid} 保存之前")
        base = {"feature_set": "S" if v == "S" else "S+F69", "complexity": meta["runs"][0]["params"],
                "model": {"name": mid, "cfg": {**ch[v]["cfg"], **ROUND2}, "epochs": ch[v]["最终阶段 epoch"]}}
        score_and_save(mid, "final_once", plan.final, keys, S_once,
                       {**base, "protocol": "只训练一次（对照）", "seconds": {"fit": refits[0]["seconds"]}}, save=True)
        score_and_save(mid, "final", plan.final, keys, S_final,
                       {**base, "protocol": "测试段按年重训（PREREG 补充登记十）", "refits": refits,
                        "seconds": {"fit": round(sum(r["seconds"] for r in refits), 1)}}, save=True)


def report() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    out = result_dir(OUT)
    ch = json.loads((out / "chosen.json").read_text(encoding="utf-8"))
    tabs = {v: pd.read_csv(out / f"tuning_{v}.csv") for v in VARIANTS}
    load_info = lambda m: json.loads((RESULTS / "p3_models" / m / "select.json").read_text(encoding="utf-8"))  # noqa: E731
    models = {"LSTM-S": "lstm_S", "LSTM-S+F69": "lstm_SF69", "第一轮 LSTM-S": "lstm_r1_S", "第一轮 LSTM-S+F69": "lstm_r1_SF69",
              "XGBoost 调参版 F69": "xgb_F69", "XGBoost 调参版 F28": "xgb_F28",
              "随机森林 F69": "rf_F69", "XGBoost 复现 F28（参照）": "xgb_repro_F28", "反转因子（参照）": "rev_r5"}
    infos = {k: load_info(m) for k, m in models.items()}
    daily = {m: pd.read_csv(RESULTS / "p3_models" / m / "select_daily.csv", index_col=0, parse_dates=True) for m in models.values()}

    def fmt(tab):
        t = tab.copy()
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
                     "各种子 IC": " / ".join(f"{v:.4f}" for v in info["seed_ic"]), "种子极差": round(info["seed_spread"], 4)})
    final_tab = pd.DataFrame(rows)
    pairs = [("lstm_S", "xgb_F69", "LSTM-S − XGBoost 调参版 F69（RQ2）"), ("lstm_SF69", "lstm_S", "LSTM-S+F69 − LSTM-S（加手工特征）"),
             ("lstm_SF69", "xgb_F69", "LSTM-S+F69 − XGBoost 调参版 F69"), ("lstm_S", "rev_r5", "LSTM-S − 反转因子"),
             ("lstm_S", "lstm_r1_S", "LSTM-S 第二轮 − 第一轮"), ("lstm_SF69", "lstm_r1_SF69", "LSTM-S+F69 第二轮 − 第一轮")]
    cmp_rows = []
    for a, b, label in pairs:
        for col, unit in (("ic", 1), ("top30", 100)):
            c = ev.compare(daily[a][col], daily[b][col])
            cmp_rows.append({"比较": label, "指标": "逐日 IC" if col == "ic" else "前 30 超额（%）", "差": round(c["diff"] * unit, 4),
                             "95% 区间": f"[{c['ci'][0] * unit:+.4f}, {c['ci'][1] * unit:+.4f}]"})
    cmp_tab = pd.DataFrame(cmp_rows)
    yearly = pd.DataFrame({m: daily[m]["ic"].groupby(daily[m].index.year).mean() for m in models.values()}).round(4)
    round1 = {v: pd.read_csv(out / f"round1_tuning_{v}.csv") for v in VARIANTS}
    r1_ch = json.loads((out / "round1_chosen.json").read_text(encoding="utf-8"))

    # 图 2：两轮的验证曲线（选定配置的 3 个种子），看第二轮的稳定化做法是否压住了噪声
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    noise = {}
    for ax, v in zip(axes, VARIANTS):
        for tag, path, style in (("第一轮", out / f"round1_{VARIANTS[v]}_curves.json", "--"), ("第二轮", out / f"{VARIANTS[v]}_curves.json", "-")):
            curves = json.loads(path.read_text(encoding="utf-8"))
            jumps = []
            for sd, c in enumerate(curves):
                ics = [r["valid_ic"] for r in c]
                jumps += list(np.abs(np.diff(ics)))
                ax.plot([r["epoch"] for r in c], ics, style, marker="o", ms=2, lw=1, color=f"C{sd}",
                        label=f"{tag} 种子 {sd}")
            noise[(v, tag)] = float(np.mean(jumps))
        ax.set_title(f"{VARIANT_NAMES[v]}：虚线第一轮，实线第二轮")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("验证段逐日 IC 均值")
    fig.tight_layout()
    fig.savefig(out / "fig2_round1_vs_round2.png", dpi=130)
    plt.close(fig)
    noise_tab = pd.DataFrame([{"版本": VARIANT_NAMES[v], "第一轮：相邻 epoch 验证 IC 平均跳动": round(noise[(v, "第一轮")], 4),
                               "第二轮：相邻 epoch 验证 IC 平均跳动": round(noise[(v, "第二轮")], 4)} for v in VARIANTS])
    groups = pd.DataFrame({k: pd.Series(infos[k]["groups"]["全部"], index=range(1, 11)) * 100 for k in ("LSTM-S", "LSTM-S+F69")}).T.round(3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, v in zip(axes, VARIANTS):
        curves = json.loads((out / f"{VARIANTS[v]}_curves.json").read_text(encoding="utf-8"))
        for s, c in enumerate(curves):
            ax.plot([r["epoch"] for r in c], [r["valid_ic"] for r in c], marker="o", ms=3, label=f"种子 {s}")
        ax.axhline(infos["XGBoost 调参版 F69"]["summary"]["全部"]["ic_mean"], color="gray", ls="--", lw=1, label="XGBoost 调参版 F69")
        ax.set_title(f"{VARIANT_NAMES[v]}：每个 epoch 的验证 IC")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("验证段逐日 IC 均值")
    fig.tight_layout()
    fig.savefig(out / "fig1_valid_ic_curves.png", dpi=130)
    plt.close(fig)

    def pick_text(v):
        c = ch[v]
        tie = c["打平的组"]
        why = (f"第 {c['组']} 组验证 IC 最高" if len(tie) == 1 else
               f"第 {'、'.join(map(str, tie))} 组和最高值相差不到 0.002，按规则取参数最少（最简单）的第 {c['组']} 组")
        return (f"- **{VARIANT_NAMES[v]}**：{why}——{c['cfg']}，{c['参数个数']:,} 个参数（种子 0 验证 IC {c['验证 IC（种子 0）']:.4f}）；"
                f"3 个种子最佳 epoch {c['各种子最佳 epoch']}，最终阶段训练 **{c['最终阶段 epoch']}** 个 epoch")

    notes_path = out / "notes.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "（读法待补）"
    text = f"""
# 3.6 LSTM（验证段，第二轮）

- 代码：`ml/lstm_prep.py`（主环境准备数据）、`ml/lstm_train.py`（dl-study 环境训练，只依赖 numpy、torch）、`ml/p3_6_lstm.py`（调度、评估、报告）
- 运行：`python -X utf8 -m ml.p3_6_lstm`
- 产物：`tuning_S.csv`、`tuning_SF.csv`、`chosen.json`、`fig1_valid_ic_curves.png`；分数 `data/ml/scores/lstm_S__select.parquet`、`lstm_SF69__select.parquet`
- 所有数字都只来自验证段（2021 ~ 2022）；测试段等所有模型的选择冻结后统一跑（任务 3.11）

## 一、设置

- 输入：每个样本是这只股票过去 60 个交易日 × 11 个原始日线通道（1.6）；S+F69 版本再把 69 个手工特征（dense 预处理，71 维）
  拼到 LSTM 最后一步的隐状态上。
- 结构：LSTM（1 ~ 2 层）→ 最后一步隐状态 → dropout →（拼手工特征）→ 全连接 → ReLU → 全连接 → 一个分数。
- 训练：均方误差，目标 = 训练标签 − 0.5；Adam；每个 epoch 随机抽一部分交易日，每批由完整的交易日组成（≤ 4096 行）；
  每个 epoch 末算验证段逐日 IC，连续 5 个 epoch 不提高就停（第一轮最多 30 个，第二轮最多 20 个），取 IC 最高的 epoch。
- 调参：事先列出的 12 组（隐藏维度、层数、dropout、学习率、权重衰减、每 epoch 抽样比例），S 和 S+F69 各调一遍，只用种子 0。
- **一共跑了两轮**，网格相同。第一轮验证曲线噪声很大（相邻 epoch 的验证 IC 差 0.005 ~ 0.02），早停取的是噪声的最高点，
  最终阶段固定 epoch 数时停在哪里也靠运气。第二轮（本报告第二、三节）加了两项常规的稳定化做法（PREREG 补充登记九）：
  **学习率余弦衰减**（第 1 个 epoch 用设定值，第 20 个 epoch 降到接近 0，最多 20 个 epoch）和**权重的指数滑动平均**
  （衰减 0.999、前几百步预热，验证、早停、输出都用平均权重）。第二轮替换第一轮是事先定的，不按两轮谁高来挑；第一轮见第五节。
- 说明：显卡上的 LSTM（cuDNN）训练不是逐位可复现的，同一种子重跑结果会有很小的差别；种子之间的差别见下表。

## 二、调参结果（种子 0）

### LSTM-S（只用序列）

{fmt(tabs['S'])}

### LSTM-S+F69（序列 + 手工特征）

{fmt(tabs['SF'])}

### 选定

{pick_text('S')}
{pick_text('SF')}

## 三、最终模型（选定参数 × 3 个种子，分数取平均）与其他模型对照

{final_tab.to_markdown(index=False)}

**分 10 组的平均超额（%，第 10 组 = 分数最高）**

{groups.to_markdown()}

**逐年 IC**

{yearly.to_markdown()}

**比较（按月整块重抽 95% 区间；仅供参考，正式判定在测试段，见 PREREG 2.4）**

{cmp_tab.to_markdown(index=False)}

![验证 IC 曲线](fig1_valid_ic_curves.png)

**两轮的验证曲线对比**（选定配置的 3 个种子）

{noise_tab.to_markdown(index=False)}

![两轮对比](fig2_round1_vs_round2.png)

## 四、读法

{notes}

## 五、第一轮（无学习率衰减和权重平均，不参与选择）

### LSTM-S

{fmt(round1['S'])}

### LSTM-S+F69

{fmt(round1['SF'])}

第一轮选定：S 第 {r1_ch['S']['组']} 组、S+F69 第 {r1_ch['SF']['组']} 组；3 个种子的结果见第三节表中“第一轮”两行。完整报告 `round1_report.md`。
"""
    write_md(out / "report.md", text)
    upsert_section(RESULTS / "tuning.md", "### 3.6 第二轮：结果（用于选择）", f"""
第二轮共试 {len(tabs["S"]) + len(tabs["SF"])} 组（S、S+F69 各 12 组），连同第一轮 LSTM 共试 48 组，详细见 `ml/results/p3_6_lstm/report.md`。

| 版本 | 最高验证 IC（种子 0） | 与最高相差 < 0.002 的组 | 选定（最简单的那组） | 选定组的验证 IC（种子 0） | 3 个种子平均后的验证 IC | 最终阶段 epoch |
|---|---|---|---|---|---|---|
""" + "\n".join(
        f"| {VARIANT_NAMES[v]} | {ch[v]['最高验证 IC']:.4f} | {'、'.join(map(str, ch[v]['打平的组']))} | 第 {ch[v]['组']} 组：{ch[v]['cfg']} | {ch[v]['验证 IC（种子 0）']:.4f} | {infos['LSTM-S' if v == 'S' else 'LSTM-S+F69']['summary']['全部']['ic_mean']:.4f} | {ch[v]['最终阶段 epoch']} |"
        for v in VARIANTS))


def main() -> None:
    if "--report" in sys.argv:
        report()
        return
    if "--final-sf" in sys.argv:            # 只重跑 S+F69（2026-11-19 偶发计算错误后，见 PREREG 补充登记十三）
        run_final(("SF",))
        return
    if "--final" in sys.argv:
        run_final()
        return
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
