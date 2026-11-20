"""P7 长期回测的报告（只读 ml/results/p7_history/ 里已存的结果）：python -X utf8 -m ml.p7_history_report"""

from __future__ import annotations

import json

import pandas as pd

from ml.common import result_dir, write_md
from ml.p7_history import NAMES, OUT
from ml.p7_history_analyze import MAIN, MODELS, RULES, SEGMENTS

STRATEGY = """
**DB-K30-b8（20 万本金，主规则）**：每天收盘后，模型给准入池里全部股票打分排名。第二天：
1. 先卖：持仓里排名掉出前 240 名（30 × 8）的股票，在持有满 5 天后的尾盘卖出（跌停卖不出、停牌就继续拿着，记为“被卡住”）；
2. 再补：按当天排名从高往低，把空位补满到 30 只，开盘买入，每只分到的钱大致相等；同一个行业最多 2 只；
   涨停买不进就顺延到下一名；一手（100 股）都买不起的跳过；
3. 排名还在前 240 名以内的持仓继续拿着，不必每天换。目的是减少换手和手续费。

**FH-K6-h4（20 万本金，最朴素的对照）**：资金分成 4 份轮流用。每天用其中一份，开盘买入当天排名前 6 的股票（同样每个行业最多 2 只），
持有到第 5 天尾盘卖出。这条规则不是搜出来的。

**DB-K3-b16（2 万本金，集中）**：同 DB-K30-b8，但最多持 3 只，排名掉出前 48 名才卖。

**费用（三条规则相同）**：佣金万五（每笔最低 5 元）、过户费十万分之一（双边）、印花税卖出时收（2023-08-28 之前 0.1%，之后 0.05%）、买入按一手取整。
不计冲击成本。**买卖时点**：信号日 T 收盘后出名单，T+1 开盘买，持有期最后一天尾盘卖（仓库现行口径）。
"""

LEAKS = """
| 渠道 | 影响到哪 | 对应的检查 |
|---|---|---|
| 超参数是在 2013 ~ 2020 训练、2021 ~ 2022 验证上选的 | 2015 ~ 2022 | 参数敏感性：换成验证 IC 最高的第 5 组、较低的第 4 组，看结果变不变（第三、六节） |
| 41 个新特征是看过全样本的探索性分析后设计的 | 全期 | 无法消除；新特征的方向是在计算任何统计量之前登记的（PREREG 第一节） |
| 作者的 28 个特征和原参数是在 2022-09 ~ 2024-06 上挑的 | 全期（作者原模型、F28 模型） | 无法消除，如实说明 |
| DB-K30-b8 是作者在 2015 ~ 2026 上从 150 条规则里挑的 | 用这条规则的组合结果 | 对照 FH-K6-h4（不是搜出来的，第七节） |
| 集成的成员是按验证段结果选的 | 集成 | 只描述 |
"""


def _seg_table(res: pd.DataFrame, rule: str, models: list[str] = MODELS) -> pd.DataFrame:
    t = res[res["规则"] == rule].set_index("model_id").reindex(models)
    rows = []
    for mid, r in t.iterrows():
        row = {"模型": NAMES[mid]}
        for seg in SEGMENTS:
            ci = r[f"{seg}｜超额区间"]
            row[seg] = f"{r[f'{seg}｜年化%']:+.1f}% / 超额 {r[f'{seg}｜超额%']:+.1f}% [{ci[0]:+.1f}, {ci[1]:+.1f}] / 回撤 {r[f'{seg}｜最大回撤%']:.1f}%"
        rows.append(row)
    return pd.DataFrame(rows)


def report() -> None:
    out = result_dir(OUT)
    res = pd.DataFrame(json.loads((out / "results.json").read_text(encoding="utf-8")))
    years = pd.read_csv(out / "yearly.csv")
    yic = pd.read_csv(out / "yearly_ic.csv", index_col=0)
    fix = pd.read_csv(out / "fix_check.csv")
    auth = pd.read_csv(out / "author_check.csv")
    cons = pd.read_csv(out / "consistency.csv")
    stress = pd.read_csv(out / "stress.csv")
    hs = json.loads((out / "holding_stats.json").read_text(encoding="utf-8"))["stats"]
    main_years = years[years["规则"] == MAIN]
    yr_ret = main_years.pivot(index="年", columns="model_id", values="收益%")[MODELS].rename(columns=NAMES)
    yr_ex = main_years.pivot(index="年", columns="model_id", values="超额%")[MODELS].rename(columns=NAMES)
    yr_dd = main_years.pivot(index="年", columns="model_id", values="最大回撤%")[["xgb_F69", "xgb_repro_F28", "rf_F69"]].rename(columns=NAMES)
    nav = pd.read_csv(out / f"nav_{MAIN}.csv", index_col=0, parse_dates=True)
    pool_y = nav["准入池等权"].groupby(nav.index.year).last()
    pool_y = (pool_y / pool_y.shift(1).fillna(1.0) - 1) * 100
    idx_y = {c: (nav[c].groupby(nav.index.year).last() / nav[c].groupby(nav.index.year).last().shift(1).fillna(1.0) - 1) * 100
             for c in ("沪深300", "中证500", "中证1000")}
    yr_ret.insert(0, "准入池等权", pool_y.round(1))
    for c, v in idx_y.items():
        yr_ret[c] = v.round(1)
    sens_tab = pd.concat([_seg_table(res, r, ["xgb_F69", "xgb_F69_s5", "xgb_F69_s4"]).assign(规则=r) for r in RULES], ignore_index=True)
    rule_rows = []
    for mid in MODELS:
        row = {"模型": NAMES[mid]}
        for rule in RULES:
            r = res[(res["model_id"] == mid) & (res["规则"] == rule)].iloc[0]
            row[rule] = f"{r['全期 2015 ~ 2026｜年化%']:+.1f}% / 超额 {r['全期 2015 ~ 2026｜超额%']:+.1f}% / 回撤 {r['全期 2015 ~ 2026｜最大回撤%']:.1f}%"
        rule_rows.append(row)
    cost_rows = res[res["规则"] == MAIN].set_index("model_id").reindex(MODELS)[["换手（倍/年）", "费用占比（%/年）", "平均持有天数", "平均持股"]]
    cost_rows.index = [NAMES[m] for m in cost_rows.index]
    notes_path = out / "notes.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "（读法待补）"

    def stats_md(mid):
        by_year, top, ind = (pd.DataFrame(x) for x in hs[mid])
        return f"""**{NAMES[mid]}：逐年交易统计**（胜率、平均单笔收益按扣费后计，%）

{by_year.to_markdown(index=False)}

**持有天数最多的 15 只股票**

{top.to_markdown(index=False)}

**每年持仓最多的 3 个行业（按持有天数占比）**

{ind.to_markdown(index=False)}
"""

    write_md(out / "report.md", f"""
# P7 长期回测：2015 ~ 2026（补充分析）

- 代码：`ml/p7_history.py`（逐年重训）、`ml/p7_history_analyze.py`（回测与分析）、`ml/p7_history_report.py`（本报告）；设置事先登记在 `ml/PREREG.md` 补充登记十一
- 做法：超参数全部沿用冻结记录；2015 ~ 2026 每年第一个交易日，用 2013-08 起、卖出日早于重训日的全部样本训练（3 个种子），给当年准入池**全部**股票打分；
  用仓库模拟器从 2015 年初连续回测到 2026-10（最后一个信号日 2026-10-16），20 万本金、真实费用、印花税按历史税率
- 结果文件：`results.json`（每个模型 × 规则 × 时段）、`yearly.csv`、`monthly_<规则>.csv`（每月收益，%）、`nav_<规则>.csv`、`yearly_ic.csv`、`stress.csv`、
  `author_check.csv`、`fix_check.csv`；交易明细、每日持仓、每日候选名单在 `data/ml/history/`（不进 Git）
- **这是补充分析，不改变任何正式判定。** 三段的性质：2015 ~ 2020 仅供参考，2021 ~ 2022 偏乐观（挑参数的年份），2023 ~ 2026 是真正的样本外

**“间接偷看”的渠道和对应的检查**
{LEAKS}

## 一、核对

**1. 和测试段正式分数一致**（2023 ~ 2026 有效行，同样的重训日、训练行、参数、种子）

{cons.to_markdown(index=False)}

**2. 候选名单修正前后**（2023 ~ 2026，{MAIN}，印花税口径与 P5 基准相同；PREREG 规定：差超过 1 个百分点则重做 P5 及其下游）

{fix.to_markdown(index=False)}

**3. 与作者十年实验逐年对照**：作者原模型 × 作者当时的规则 `FH-K6-next-o1o5`（每天买前 6 只、分批持有、第 5 天开盘卖）、20 万、历史印花税，
对照 `experiments/decade/REPORT.md` 的“按年重训”一栏

{auth.to_markdown(index=False)}

## 二、逐年 IC（预测层，只在有有效收益的行上算）

{yic.to_markdown()}

**模型之间的 IC 差（按月整块重抽 95% 区间）**

{pd.read_csv(out / 'ic_compare.csv').to_markdown(index=False) if (out / 'ic_compare.csv').exists() else ''}

![逐年 IC](fig4_yearly_ic.png)

## 三、组合：分段汇总（{MAIN}，年化 / 年化超额 [区间] / 最大回撤）

{_seg_table(res, MAIN).to_markdown(index=False)}

**换手与费用（{MAIN}，全期）**

{cost_rows.round(2).to_markdown()}

![净值与回撤](fig1_nav_drawdown.png)

## 四、逐年收益与超额（{MAIN}）

**逐年收益（%）**，附准入池等权和指数

{yr_ret.round(1).to_markdown()}

**逐年超额（%，相对准入池等权，逐日超额均值 × 242）**

{yr_ex.round(1).to_markdown()}

**逐年最大回撤（%）**

{yr_dd.round(1).to_markdown()}

![逐年超额](fig2_yearly_excess.png)

## 五、极端行情（{MAIN}）

{stress.to_markdown(index=False)}

## 六、参数敏感性（XGBoost F69 三组参数，三条规则）

{sens_tab.to_markdown(index=False)}

## 七、规则对照（全期：年化 / 年化超额 / 最大回撤）

{pd.DataFrame(rule_rows).to_markdown(index=False)}

## 八、选了哪些股票、怎么交易

{stats_md('xgb_F69')}

{stats_md('xgb_repro_F28')}

明细（不进 Git，在 `data/ml/history/`）：`trades_<模型>__<规则>.parquet`（每一笔交易：信号日、买入日、卖出日、代码、当时的名称、行业、买入时排名、
买卖价、扣费前后收益、持有天数、是否被卡住）、`holdings_<模型>__<规则>.parquet`（每天收盘后的持仓）、`picks_<模型>.parquet`（每天的前 30 名，每个行业最多 2 只）。

## 九、月度收益

全部模型、全部规则的月度收益在 `monthly_<规则>.csv`（%）。下图是 XGBoost F69 在 {MAIN} 下的月度收益和月度超额。

![月度收益](fig3_monthly_return.png)

## 九点五、换起点

2015 ~ 2024 每年第一个交易日起步的结果见 [starts.md](starts.md)（补充登记十二）。

## 十、交易策略说明
{STRATEGY}

## 十一、读法

{notes}
""")
    print(f"写出 {out / 'report.md'}")


if __name__ == "__main__":
    report()
