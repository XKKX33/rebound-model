"""P8 交付版报告（只读 ml/results/delivery/ 里已存的结果）：python -X utf8 -m ml.p8_delivery_report"""

from __future__ import annotations

import json

import pandas as pd

from ml.common import result_dir, write_md
from ml.p7_history_analyze import MAIN, RULES, SEGMENTS
from ml.p7_history_report import STRATEGY
from ml.p8_delivery import BENCH, M3, NAMES, OUT


def _seg_table(res: pd.DataFrame, rule: str) -> pd.DataFrame:
    t = res[res["规则"] == rule].set_index("model_id").reindex(M3)
    rows = []
    for mid, r in t.iterrows():
        row = {"模型": NAMES[mid]}
        for seg in SEGMENTS:
            ci = r[f"{seg}｜超额区间"]
            row[seg] = f"{r[f'{seg}｜年化%']:+.1f}% / 超额 {r[f'{seg}｜超额%']:+.1f}% [{ci[0]:+.1f}, {ci[1]:+.1f}] / 回撤 {r[f'{seg}｜最大回撤%']:.1f}%"
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt_lb(lb: pd.DataFrame) -> pd.DataFrame:
    t = lb.drop(columns=["model_id"]).copy()
    for c in t.columns:
        if c.endswith("区间"):
            t[c] = t[c].map(lambda v: f"[{v[0]:.4f}, {v[1]:.4f}]" if isinstance(v, list) else v)
    return t.set_index("模型").T


def report() -> None:
    out = result_dir(OUT)
    lb = pd.DataFrame(json.loads((out / "leaderboard.json").read_text(encoding="utf-8")))
    ic_cmp = pd.read_csv(out / "test_ic_compare.csv")
    port = pd.read_csv(out / "test_portfolio.csv")
    pdiff = pd.read_csv(out / "test_portfolio_diff.csv")
    tyic = pd.read_csv(out / "test_yearly_ic.csv", index_col=0)
    cons = pd.read_csv(out / "lstm_consistency.csv")
    res = pd.DataFrame(json.loads((out / "long_results.json").read_text(encoding="utf-8")))
    years = pd.read_csv(out / "long_yearly.csv")
    lyic = pd.read_csv(out / "long_yearly_ic.csv", index_col=0)
    licmp = pd.read_csv(out / "long_ic_compare.csv")
    stress = pd.read_csv(out / "long_stress.csv")
    starts = pd.read_csv(out / "long_starts.csv")
    hs = json.loads((out / "long_holding_stats.json").read_text(encoding="utf-8"))
    nav = pd.read_csv(out / f"long_nav_{MAIN}.csv", index_col=0, parse_dates=True)

    my = years[years["规则"] == MAIN]
    yr_ret = my.pivot(index="年", columns="model_id", values="收益%")[M3].rename(columns=NAMES)
    for c in BENCH:
        s = nav[c].groupby(nav.index.year).last()
        yr_ret[c] = ((s / s.shift(1).fillna(1.0) - 1) * 100).round(1)
    yr_ex = my.pivot(index="年", columns="model_id", values="超额%")[M3].rename(columns=NAMES)
    yr_dd = my.pivot(index="年", columns="model_id", values="最大回撤%")[M3].rename(columns=NAMES)
    rule_rows = []
    for mid in M3:
        row = {"模型": NAMES[mid]}
        for rule in RULES:
            r = res[(res["model_id"] == mid) & (res["规则"] == rule)].iloc[0]
            row[rule] = f"{r['全期 2015 ~ 2026｜年化%']:+.1f}% / 超额 {r['全期 2015 ~ 2026｜超额%']:+.1f}% / 回撤 {r['全期 2015 ~ 2026｜最大回撤%']:.1f}%"
        rule_rows.append(row)
    cost = res[res["规则"] == MAIN].set_index("model_id").reindex(M3)[["换手（倍/年）", "费用占比（%/年）", "平均持有天数", "平均持股"]]
    cost.index = [NAMES[m] for m in cost.index]

    st_secs = []
    for rule in ("DB-K30-b8", "FH-K6-h4"):
        d = starts[starts["规则"] == rule]
        summ = d.groupby("模型").agg(年化中位数=("年化%", "median"), 年化最差=("年化%", "min"), 超额中位数=("年化超额%", "median"),
                                     超额最差=("年化超额%", "min"), 回撤中位数=("最大回撤%", "median"), 回撤最差=("最大回撤%", "min")).reindex(list(NAMES.values()))
        piv = d.pivot(index="起步年", columns="模型", values="年化%")[list(NAMES.values())]
        piv.insert(0, "准入池同期年化", d.drop_duplicates("起步年").set_index("起步年")["准入池同期年化%"])
        dd = d.pivot(index="起步年", columns="模型", values="最大回撤%")[list(NAMES.values())]
        st_secs.append(f"""**{rule}（20 万）**：按起点汇总

{summ.round(1).to_markdown()}

年化收益（%）

{piv.round(1).to_markdown()}

最大回撤（%）

{dd.round(1).to_markdown()}
""")

    def stats_md(mid):
        by_year, top, ind = (pd.DataFrame(x) for x in hs[mid])
        return f"""**{NAMES[mid]}**：逐年交易统计（胜率、平均单笔收益按扣费后计，%）

{by_year.to_markdown(index=False)}

持有天数最多的 15 只股票

{top.to_markdown(index=False)}

每年持仓最多的 3 个行业（按持有天数占比）

{ind.to_markdown(index=False)}
"""

    notes_path = out / "notes.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "（读法待补）"
    write_md(out / "report.md", f"""
# 交付版结果：三个 F69 模型（XGBoost、随机森林、LSTM S+F69）

- 交付范围见 `ml/PREREG.md` 补充登记十三：交付物只保留这三个模型；其他模型的结果作为过程记录留在各步目录，不在这里出现。
- 代码：`ml/p8_delivery.py`（计算）、`ml/p8_delivery_report.py`（本报告）；LSTM 长期回测的训练：`ml/p7_history_lstm.py`。
- **第一部分是正式数字**（测试段 2023-01 ~ 2026-10，冻结后只跑一次，每年初重训）；**第二部分是补充分析**（2015 ~ 2026 长期回测，
  2015 ~ 2020 仅供参考，2021 ~ 2022 偏乐观，2023 ~ 2026 与测试段相同）。
- 这三个模型是看过测试结果之后才定为交付范围的，测试段数字因此略偏乐观。

## 读法

{notes}

## 一、测试段（正式）

### 1. 排行榜

IC = 逐日 Rank IC 均值；超额、多空是小数（0.003 = 0.3%，5 天持有期）。验证 IC 是挑参数用的，偏乐观。

{_fmt_lb(lb).to_markdown()}

### 2. 模型两两之间的 IC 差（按月整块重抽 95% 区间）

{ic_cmp.to_markdown(index=False)}

### 3. 逐年 IC

{tyic.to_markdown()}

### 4. 组合回测（P5 基准口径：仓库模拟器、真实费用、现行印花税）

{port.to_markdown(index=False)}

**组合超额差**（逐日收益差 × 242，按月整块重抽 95% 区间，%）

{pdiff.to_markdown(index=False)}

## 二、长期回测 2015 ~ 2026（补充分析）

做法与 P7 完全相同（补充登记十一）：冻结参数，每年第一个交易日用此前全部已卖出的样本重训（3 个种子），给当年准入池全部股票打分，
20 万本金连续回测到 2026-10，印花税按历史税率。

**LSTM 长期分数与测试段正式分数的一致性**（GPU 训练不能逐位复现，2023 ~ 2026）

{cons.to_markdown(index=False)}

### 1. 分段汇总（{MAIN}：年化 / 年化超额 [区间] / 最大回撤）

{_seg_table(res, MAIN).to_markdown(index=False)}

**三条规则的全期结果**（年化 / 年化超额 / 最大回撤）

{pd.DataFrame(rule_rows).to_markdown(index=False)}

**换手与费用（{MAIN}，全期）**

{cost.round(2).to_markdown()}

![长期净值与回撤](fig1_long_nav.png)

### 2. 逐年（{MAIN}）

**逐年收益（%）**，附准入池等权和指数

{yr_ret.round(1).to_markdown()}

**逐年超额（%，相对准入池等权）**

{yr_ex.round(1).to_markdown()}

**逐年最大回撤（%）**

{yr_dd.round(1).to_markdown()}

![逐年超额](fig2_yearly_excess.png)

### 3. 逐年 IC（只在有有效收益的行上算）

{lyic.to_markdown()}

**模型之间的 IC 差（分段，按月整块重抽 95% 区间）**

{licmp.to_markdown(index=False)}

![逐年 IC](fig3_yearly_ic.png)

### 4. 极端行情（{MAIN}）

{stress.to_markdown(index=False)}

### 5. 换起点（2015 ~ 2024 每年第一个交易日起步，各跑到 2026-10）

年化按“起步日到 2026-10-23、每年 242 个交易日”折算，和上面分段表的口径略有不同。

{''.join(st_secs)}

### 6. 月度超额

全部规则的月度收益在 `long_monthly_<规则>.csv`（%）。

![月度超额](fig4_monthly_excess.png)

### 7. 选了哪些股票

{stats_md('xgb_F69')}

{stats_md('rf_F69')}

{stats_md('lstm_SF69')}

明细（不进 Git，在 `data/ml/history/`）：`trades_<模型>__<规则>.parquet`、`holdings_<模型>__<规则>.parquet`、`picks_<模型>.parquet`。

## 三、交易策略说明
{STRATEGY}
""")
    print(f"写出 {out / 'report.md'}")


if __name__ == "__main__":
    report()
