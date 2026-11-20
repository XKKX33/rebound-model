"""P9 补充：DB-K20-b16 与作者规则 DB-K30-b8 的全部回测对比（三个交付模型，20 万本金）。

DB-K20-b16 是看过验证段（2021 ~ 2022）和测试段（2023 ~ 2026）全部 150 条规则的结果后，唯一两段都进前 30 名的规则
（`ml/results/p9_rule_select/`）。它的选择用到了测试段，所以测试段的比较不是样本外；
**2015 ~ 2020 是选规则时没有用过的数据**（长期回测分数：每年初用此前数据重训），可以当独立检验。

做法与已有回测完全相同（不改任何设置，只换规则）：
1. 测试段 2023-01 ~ 2026-10：正式分数，费用同 P5（现行印花税）；
2. 长期回测 2015 ~ 2026：长期回测分数，印花税按历史税率，分段 / 逐年 / 极端行情同 P7；
3. 换起点 2015 ~ 2024：同 `ml/p7_starts.py`。
运行时核对：DB-K30-b8 的数字必须和已存结果（P5、delivery）一致。

输出 `ml/results/p9_rule_select/compare_*.csv` 和 compare.md。
运行：python -X utf8 -m ml.p9_rule_compare        # 约 5 分钟
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ml.common import DB_PATH, data_dir, result_dir, write_md
from ml.p5_backtest import to_market_index
from ml.p7_history_analyze import LAST_SIGNAL, SEGMENTS, STRESS, load_scores
from ml.p7_starts import STARTS
from ml.p8_delivery import M3, NAMES

OUT = "p9_rule_select"
RULES = ("DB-K30-b8", "DB-K20-b16")
TAG = {"DB-K30-b8": "作者规则", "DB-K20-b16": "两段平衡"}
CAPITAL = 200_000.0


def _check(got: float, want: float, what: str, tol: float = 0.05) -> None:
    if abs(got - want) > tol:
        raise RuntimeError(f"{what}：算出 {got}，已存结果 {want}，不一致")


def test_part() -> tuple[pd.DataFrame, pd.DataFrame]:
    from rebound.backtest import stats
    from rebound.backtest.costs import CostModel
    from rebound.backtest.market import candidates, load_markets, pool_daily_return
    from rebound.backtest.rules import rule_of
    from rebound.backtest.sim import simulate
    m = load_markets(DB_PATH, start=date(2022, 1, 1))["nocap"]
    pool, dates = pool_daily_return(m), m.dates
    first = pd.read_parquet(data_dir() / "scores" / "xgb_F69__final.parquet", columns=["trade_date"])["trade_date"]
    d_first, d_last = int(dates.get_indexer([first.min()])[0]), int(dates.get_indexer([first.max()])[0])
    idx = stats.scoring_days(d_first, d_last)
    ref = pd.read_csv(result_dir("p5_backtest") / "results.csv", encoding="utf-8-sig")
    rows, years = [], []
    for mid in M3:
        sc = pd.read_parquet(data_dir() / "scores" / f"{mid}__final.parquet", columns=["trade_date", "ts_code", "score"])
        cands, rank = candidates(m, to_market_index(m, sc, "score", mid))
        for rule in RULES:
            res = simulate(m, cands, rank, rule_of(rule), CostModel(capital=CAPITAL), d_first, d_last)
            summ = stats.summary(res, pool, dates, idx)
            per = stats.period(res.nav, pool, dates, idx)
            if rule == "DB-K30-b8":
                r0 = ref[(ref["model_id"] == mid) & (ref["规则"] == rule) & (ref["口径"] == "基准")].iloc[0]
                _check(per["excess"], r0["全段 超额%"], f"测试段 {mid} {rule} 超额")
            rows.append({"规则": f"{rule}（{TAG[rule]}）", "模型": NAMES[mid], "年化%": per["cagr"], "年化超额%": per["excess"],
                         "超额区间": f"[{per['excess_ci'][0]:+.1f}, {per['excess_ci'][1]:+.1f}]", "最大回撤%": per["max_dd"],
                         "β": per["beta"], "换手（倍/年）": summ["turnover"], "费用（%/年）": summ["cost_pct"],
                         "平均持有天数": summ["hold_days"], "平均持股": summ["avg_holdings"]})
            for y, v in summ["years"].items():
                years.append({"规则": rule, "模型": NAMES[mid], "年": int(y), "收益%": v["total"], "超额%": v["excess"]})
            print(f"测试段 {rule} {mid}：年化 {per['cagr']:+.1f}%，超额 {per['excess']:+.1f}%", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(years)


def long_part() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from rebound.backtest import stats
    from rebound.backtest.costs import CostModel
    from rebound.backtest.market import candidates, load_markets, pool_daily_return
    from rebound.backtest.rules import rule_of
    from rebound.backtest.sim import simulate
    m = load_markets(DB_PATH, start=date(2014, 1, 1))["nocap"]
    pool, dates = pool_daily_return(m), m.dates
    d_first = int(dates.searchsorted(pd.Timestamp("2015-01-01")))
    d_last = int(dates.get_indexer([pd.Timestamp(LAST_SIGNAL)])[0])
    d_end = len(dates) - 1
    idx = stats.scoring_days(d_first, d_last)
    yy = dates[idx].year
    ref_l = pd.DataFrame(pd.read_json(result_dir("delivery") / "long_results.json"))
    ref_s = pd.read_csv(result_dir("delivery") / "long_starts.csv", encoding="utf-8-sig")
    seg_rows, year_rows, stress_rows, start_rows = [], [], [], []
    for mid in M3:
        sc = load_scores(mid)
        cands, rank = candidates(m, to_market_index(m, sc[sc["trade_date"] <= pd.Timestamp(LAST_SIGNAL)], "score", mid))
        for rule in RULES:
            cost = CostModel(capital=CAPITAL, stamp_before=0.001)
            res = simulate(m, cands, rank, rule_of(rule), cost, d_first, d_last)
            summ = stats.summary(res, pool, dates, idx)
            row = {"规则": f"{rule}（{TAG[rule]}）", "模型": NAMES[mid], "换手（倍/年）": summ["turnover"], "费用（%/年）": summ["cost_pct"]}
            for seg, (y0, y1) in SEGMENTS.items():
                per = stats.period(res.nav, pool, dates, idx[(yy >= y0) & (yy <= y1)])
                row[f"{seg}｜年化%"], row[f"{seg}｜超额%"], row[f"{seg}｜最大回撤%"] = per["cagr"], per["excess"], per["max_dd"]
                row[f"{seg}｜超额区间"] = f"[{per['excess_ci'][0]:+.1f}, {per['excess_ci'][1]:+.1f}]"
            if rule == "DB-K30-b8":
                r0 = ref_l[(ref_l["model_id"] == mid) & (ref_l["规则"] == rule)].iloc[0]
                _check(row["全期 2015 ~ 2026｜超额%"], r0["全期 2015 ~ 2026｜超额%"], f"长期 {mid} {rule} 超额")
            seg_rows.append(row)
            for y, v in stats.by_year(res.nav, pool, dates, idx).items():
                year_rows.append({"规则": rule, "模型": NAMES[mid], "年": int(y), "收益%": v["total"], "超额%": v["excess"], "最大回撤%": v["max_dd"]})
            nav = pd.Series(res.nav[idx[0] - 1:idx[-1] + 1] / res.nav[idx[0] - 1], index=dates[idx[0] - 1:idx[-1] + 1])
            for name, (a, b) in STRESS.items():
                s = nav.loc[a:b]
                base = nav.loc[:pd.Timestamp(a)].iloc[-1] if (nav.index < pd.Timestamp(a)).any() else s.iloc[0]
                seg = pd.concat([pd.Series([base]), s.reset_index(drop=True)])
                stress_rows.append({"规则": rule, "模型": NAMES[mid], "时期": name, "区间收益%": round((s.iloc[-1] / base - 1) * 100, 1),
                                    "区间最大回撤%": round(float((seg / seg.cummax() - 1).min()) * 100, 1)})
            for y in STARTS:
                d0 = int(dates.searchsorted(pd.Timestamp(f"{y}-01-01")))
                r = simulate(m, cands, rank, rule_of(rule), CostModel(capital=CAPITAL, stamp_before=0.001), d0, d_last)
                w = stats.window(r.nav, pool, d0, d_end, dates)
                per = stats.period(r.nav, pool, dates, stats.scoring_days(d0, d_last))
                if rule == "DB-K30-b8":
                    r0 = ref_s[(ref_s["model_id"] == mid) & (ref_s["规则"] == rule) & (ref_s["起步年"] == y)].iloc[0]
                    _check(w["cagr"], r0["年化%"], f"换起点 {mid} {rule} {y} 年化")
                start_rows.append({"规则": rule, "模型": NAMES[mid], "起步年": y, "年化%": w["cagr"], "年化超额%": per["excess"],
                                   "最大回撤%": w["max_dd"], "期末资金（万元）": round(CAPITAL * w["final_x"] / 1e4, 1)})
            print(f"长期 {rule} {mid}：全期年化 {row['全期 2015 ~ 2026｜年化%']:+.1f}%，换起点完成", flush=True)
    return pd.DataFrame(seg_rows), pd.DataFrame(year_rows), pd.DataFrame(stress_rows), pd.DataFrame(start_rows)


def main() -> None:
    out = result_dir(OUT)
    test, test_years = test_part()
    seg, years, stress, starts = long_part()
    for name, t in (("test", test), ("test_yearly", test_years), ("long", seg), ("long_yearly", years), ("long_stress", stress), ("starts", starts)):
        t.to_csv(out / f"compare_{name}.csv", index=False, encoding="utf-8-sig")
    seg_cols = ["规则", "模型"] + [f"{s}｜{k}" for s in SEGMENTS for k in ("年化%", "超额%", "最大回撤%")]
    yr = years.pivot_table(index="年", columns=["模型", "规则"], values="收益%")
    st = starts.pivot_table(index="起步年", columns=["模型", "规则"], values="年化%")
    sdd = starts.pivot_table(index="起步年", columns=["模型", "规则"], values="最大回撤%")
    ss = stress.pivot_table(index="时期", columns=["模型", "规则"], values="区间收益%")
    summ = starts.groupby(["规则", "模型"]).agg(年化中位数=("年化%", "median"), 年化最差=("年化%", "min"), 回撤中位数=("最大回撤%", "median"),
                                              回撤最差=("最大回撤%", "min")).round(1)
    write_md(out / "compare.md", f"""
# DB-K20-b16 与作者规则 DB-K30-b8：全部回测对比

- 代码 `ml/p9_rule_compare.py`；两条规则用完全相同的分数、模拟器、费用，20 万本金。DB-K30-b8 的数字已核对与已存结果一致。
- DB-K20-b16 是看过验证段和测试段全部 150 条规则后选的（两段都进前 30 名的唯一一条），**测试段对它不是样本外**；
  **2015 ~ 2020 是选它时没有用过的数据**，可作独立检验（但该段本身仅供参考，见长期回测报告）。

## 1. 测试段 2023-01 ~ 2026-10（P5 口径）

{test.to_markdown(index=False)}

## 2. 长期回测 2015 ~ 2026：分段（年化 / 年化超额 / 最大回撤，%）

{seg[seg_cols].to_markdown(index=False)}

## 3. 逐年收益（%）

{yr.round(1).to_markdown()}

## 4. 极端行情：区间收益（%）

{ss.round(1).to_markdown()}

## 5. 换起点（2015 ~ 2024 每年起步，到 2026-10）

汇总

{summ.to_markdown()}

年化（%）

{st.round(1).to_markdown()}

最大回撤（%）

{sdd.round(1).to_markdown()}
""")
    print(f"写出 {out / 'compare.md'}", flush=True)


if __name__ == "__main__":
    main()
