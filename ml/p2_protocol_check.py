"""任务 P2：在真实数据上检查评估协议（划分、间隔、训练标签、评估代码）。

不训练任何模型，**不碰测试段的收益**：
- 划分：各段的日期、天数、行数，训练行被间隔清掉多少，训练行最晚卖出日 vs 评估段第一天；
- 训练标签：和仓库 `training_set` 的规则一致（单元测试已逐位核对），这里看分布；
- 评估代码自检（只用训练段和验证段）：
  - 完美打分（分数 = 真实收益）：IC 应为 1，前 K 名超额是上限；
  - 随机打分：IC 应≈0，Newey-West t 应在 ±2 左右；
  - 单因子反转 `-r5`：演示一个真实但弱的信号，并对比普通 t 和 Newey-West t（正式结果见任务 3.2）。

运行：python -X utf8 -m ml.p2_protocol_check
输出：ml/results/p2_protocol/report.md、split.csv、protocol_b.csv、sanity.csv
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ml import evaluate as ev
from ml.common import CFG, register, result_dir, write_md
from ml.dataset import load_panel


def _span(keys: pd.DataFrame, rows: np.ndarray) -> tuple[str, str, int, int]:
    d = keys.loc[rows, "trade_date"]
    return d.min().date().isoformat(), d.max().date().isoformat(), int(d.nunique()), int(rows.sum())


def main() -> None:
    out = result_dir("p2_protocol")
    keys = ev.load_keys()
    dates = keys.drop_duplicates("di").set_index("di")["trade_date"].sort_index()
    plan = ev.protocol_a(keys)

    # ---- 一、划分 ----
    s = CFG["split"]
    rows = []
    for stage, seg_start, label in ((plan.select, s["train_start"], "选择阶段：训练 → 验证"),
                                    (plan.final, s["train_start"], "最终阶段：训练 + 验证 → 测试")):
        fit_date = dates[stage.fit_di]
        before = (keys["trade_date"] < fit_date).to_numpy() & keys["ret_o1o5"].notna().to_numpy()
        purged = before & ~stage.fit_rows
        f0, f1, fd, fn = _span(keys, stage.fit_rows)
        last_exit = dates[int(keys.loc[stage.fit_rows, "exit_di"].max())]
        rows.append({"阶段": label + "（训练行）", "重训日": fit_date.date(), "信号日": f"{f0} ~ {f1}", "天数": fd, "行数": fn,
                     "被间隔清掉的行": int(purged.sum()),
                     "被清掉的信号日": f"{keys.loc[purged, 'trade_date'].min().date()} ~ {keys.loc[purged, 'trade_date'].max().date()}",
                     "训练行最晚卖出日": last_exit.date(), "最晚卖出日 < 重训日": bool(last_exit < fit_date)})
        for name, m in (stage.parts or {"验证": stage.eval_rows}).items():
            e0, e1, ed, en = _span(keys, m)
            rows.append({"阶段": f"  评估段 {name}", "信号日": f"{e0} ~ {e1}", "天数": ed, "行数": en})
    split = pd.DataFrame(rows)
    split["被间隔清掉的行"] = split["被间隔清掉的行"].astype("Int64")
    split.to_csv(out / "split.csv", index=False, encoding="utf-8-sig")

    pb = []
    for st in ev.protocol_b(keys):
        e0, e1, ed, en = _span(keys, st.eval_rows)
        pb.append({"测试年": st.name, "重训日": dates[st.fit_di].date(), "训练行数": int(st.fit_rows.sum()),
                   "训练行最晚卖出日": dates[int(keys.loc[st.fit_rows, "exit_di"].max())].date(),
                   "评估天数": ed, "评估行数": en, "备注": "超参数在这两年上选，不算严格样本外" if st.name in ("2021", "2022") else ""})
    pb = pd.DataFrame(pb)
    pb.to_csv(out / "protocol_b.csv", index=False, encoding="utf-8-sig")

    # 训练标签：百分位分布（应近似均匀）
    y = ev.train_labels(keys, plan.select.fit_rows)
    yq = np.quantile(y, [0, 0.1, 0.5, 0.9, 1])

    # ---- 二、评估代码自检：只用训练段和验证段 ----
    r5 = load_panel(["r5"])["r5"].to_numpy(np.float64)
    train_eval = ev.eval_rows(keys, s["train_start"], s["train_end"])
    sanity, secs, rev_groups = [], {}, {}
    for seg, m in (("训练段", train_eval), ("验证段", plan.select.eval_rows)):
        k = keys[m].reset_index(drop=True)
        rng = np.random.default_rng(CFG["random_state"])
        for name, score in (("完美打分（分数 = 收益）", k["ret_o1o5"].to_numpy()),
                            ("随机打分", rng.normal(size=len(k))),
                            ("单因子反转 −r5", -r5[m])):
            t0 = time.time()
            res = ev.evaluate(score, k, auc=True, ndcg=True)
            secs[(seg, name)] = time.time() - t0
            if name.startswith("单因子"):
                rev_groups[seg] = res["groups"]["全部"] * 100
            r = res["summary"]["全部"]
            sanity.append({"段": seg, "打分": name, "天数": r["days"], "IC 均值": r["ic_mean"], "ICIR": r["icir"],
                           "IC>0 比例": r["ic_pos"], "普通 t": r["naive_t"], "NW t（滞后 10）": r["nw_t"],
                           "NW t（滞后 5）": r["nw_t_lag5"], "NW t（滞后 20）": r["nw_t_lag20"],
                           "前6 超额%": r["top6"] * 100, "前30 超额%": r["top30"] * 100,
                           "多空%": r["long_short"] * 100, "10组单调性": r["monotonic"],
                           "AUC": r["auc"], "NDCG@30": r["ndcg30"]})
    sanity = pd.DataFrame(sanity)
    sanity.to_csv(out / "sanity.csv", index=False, encoding="utf-8-sig")
    show = sanity.copy()
    for c in show.columns[3:]:
        show[c] = show[c].map(lambda v: "—" if not np.isfinite(v) else f"{v:.3f}")

    rev_tab = pd.DataFrame(rev_groups).T.round(3)
    rev_tab.columns = [f"第{c}组" for c in rev_tab.columns]
    rev_tab.to_csv(out / "reversal_groups.csv", encoding="utf-8-sig")
    rev = sanity[(sanity["打分"] == "单因子反转 −r5") & (sanity["段"] == "验证段")].iloc[0]
    rnd = sanity[sanity["打分"] == "随机打分"]
    text = f"""
# P2 评估协议

- 协议的文字版：`ml/PREREG.md` 第二节（时间划分、指标、显著性、选择规则；每条注明沿用仓库还是新增、为什么）
- 代码：`ml/evaluate.py`；单元测试 `ml/tests/test_evaluate.py`（13 项，全部通过）
- 本检查：`python -X utf8 -m ml.p2_protocol_check`；产物 `split.csv`、`protocol_b.csv`、`sanity.csv`、`reversal_groups.csv`
- **没有训练任何模型，没有用到测试段的收益**（测试段只数了天数和行数）

## 一、主协议 A 的划分（真实数据）

{split.astype(object).where(split.notna(), "").to_markdown(index=False)}

读法：
- “被间隔清掉的行”就是任务清单里的“段与段之间隔 5 个交易日”：重训日前最后几天的信号，到重训日还没卖出（T+5 才卖、或顺延），
  它们的收益在重训时还不知道，所以不进训练。规则和仓库 `training_set` 完全一样（单元测试 `test_train_labels_match_repo_training_set` 逐位核对）。
- “训练行最晚卖出日 < 重训日”两行都是 True：训练用到的每一个收益，在重训那天之前都已经实现，没有用到评估段的信息。
- 测试段分 T1、T2 两段：作者挑特征和参数的时期（2022-09 ~ 2024-06）与 T1 重叠，见 PREREG 2.1。

训练标签（选择阶段）的分布：最小 {yq[0]:.4f}、10% 分位 {yq[1]:.3f}、中位数 {yq[2]:.3f}、90% 分位 {yq[3]:.3f}、最大 {yq[4]:.3f}
——每天的百分位，近似 0 ~ 1 均匀分布，符合定义。

## 二、协议 B（按年滚动，稳健性用）

{pb.to_markdown(index=False)}

## 三、评估代码自检（只用训练段、验证段）

{show.to_markdown(index=False)}

结论：
1. **完美打分**：IC = 1、10 组完全单调、NDCG = 1，前 6 名超额就是“事后选最好的 6 只”的上限，说明 IC、分组、Top-K、NDCG 的计算方向和口径正确。
2. **随机打分**：IC 均值 {rnd['IC 均值'].iloc[0]:.4f}（训练段）、{rnd['IC 均值'].iloc[1]:.4f}（验证段），NW t 分别为 {rnd['NW t（滞后 10）'].iloc[0]:.2f}、{rnd['NW t（滞后 10）'].iloc[1]:.2f}，
   AUC≈0.5，都在随机波动范围内，说明评估流程本身没有引入虚假的预测能力（比如拿到了未来信息）。
3. **单因子反转 −r5**（演示；正式结果在任务 3.2）：验证段 IC {rev['IC 均值']:.4f}，普通 t = {rev['普通 t']:.1f}，
   Newey-West t = {rev['NW t（滞后 10）']:.1f}（滞后 5 / 20：{rev['NW t（滞后 5）']:.1f} / {rev['NW t（滞后 20）']:.1f}）。
   NW t 明显小于普通 t：标签是 5 日收益，相邻几天的 IC 正相关，普通 t 检验高估了显著性，这正是协议里用 Newey-West 的原因。
4. **一个值得注意的现象**（不是程序错误）：−r5 的 IC 和多空收益为正，但前 6 / 前 30 名的超额是负的、AUC 和 NDCG 略低于随机。
   按 −r5 分 10 组的平均超额（%，第 1 组 = 近 5 天涨得最多，第 10 组 = 跌得最多）：

{rev_tab.to_markdown()}

   反转效应主要来自第 1 组（近期涨得最多的股票之后明显跑输），第 2 ~ 9 组随跌幅递增，但跌得**最多**的第 10 组并不反弹，
   其中最极端的前 6 / 前 30 只反而跑输——关系在两端不是线性的。
   这说明只看 IC 不够，必须同时看 Top-K（协议里两者都报），也预示了非线性模型可能有用武之地（RQ1）。
   完美打分和随机打分的结果证明这些指标本身算得没错（单元测试也和 scipy / sklearn / statsmodels 逐位核对过）。

完美打分每天的 IC 都是 1，标准差为 0，所以 ICIR 和 t 值没有定义（表中记为 —）。

## 四、给后续任务的用法

```python
from ml import evaluate as ev
keys = ev.load_keys()
plan = ev.protocol_a(keys)
st = plan.select                                  # 选择阶段（最终阶段：plan.final）
y = ev.train_labels(keys, st.fit_rows)            # 训练标签
# 模型在 X[st.fit_rows] 上用 y 训练，对 X[st.eval_rows] 打分 → score
res = ev.evaluate(score, keys[st.eval_rows])      # 最终阶段加 parts=分段后的布尔数组，见 p2 代码
ev.compare(res_a["daily"]["ic"], res_b["daily"]["ic"])   # 两个模型比较：差值 + 按月重抽区间
```

一次完整评估（含 AUC、NDCG；训练段约 190 万行、验证段约 70 万行）最多耗时 {max(secs.values()):.1f} 秒，评估不是瓶颈。
"""
    write_md(out / "report.md", text)
    register("P2_protocol", select_fit=str(dates[plan.select.fit_di].date()), final_fit=str(dates[plan.final.fit_di].date()),
             select_train_rows=int(plan.select.fit_rows.sum()), final_train_rows=int(plan.final.fit_rows.sum()),
             valid_rows=int(plan.select.eval_rows.sum()), test_rows=int(plan.final.eval_rows.sum()),
             sanity=sanity.round(4).to_dict("records"))


if __name__ == "__main__":
    main()
