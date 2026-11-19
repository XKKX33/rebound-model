"""P7 补充：长期回测换起点（PREREG 补充登记十二，描述性分析）。

分数沿用 P7（每年的模型不变），只改起步时间：2015 ~ 2024 每年第一个交易日拿 20 万现金起步，跑到 2026-10。
看每个起点的第一年收益、年化、最大回撤、翻了几倍、准入池同期年化和年化超额，避免一个起点、一次股灾左右整体印象。

运行：python -X utf8 -m ml.p7_starts         # 约 5 分钟（读行情 + 100 次模拟）
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ml.common import DB_PATH, register, result_dir, write_md
from ml.p7_history import NAMES, OUT
from ml.p7_history_analyze import LAST_SIGNAL, load_scores

STARTS = list(range(2015, 2025))
MODELS = ["xgb_F69", "xgb_F28", "xgb_repro_F28", "rf_F69", "ens_top3"]
RULES = {"DB-K30-b8": 200_000, "FH-K6-h4": 200_000}


def run(models: list[str] | None = None) -> pd.DataFrame:
    models = models or MODELS
    from rebound.backtest import stats
    from rebound.backtest.costs import CostModel
    from rebound.backtest.market import candidates, load_markets, pool_daily_return
    from rebound.backtest.rules import rule_of
    from rebound.backtest.sim import simulate
    from ml.p5_backtest import to_market_index
    m = load_markets(DB_PATH, start=date(2014, 1, 1))["nocap"]
    pool, dates = pool_daily_return(m), m.dates
    d_last = int(dates.get_indexer([pd.Timestamp(LAST_SIGNAL)])[0])
    d_end = len(dates) - 1
    rows = []
    for mid in models:
        sc = load_scores(mid)
        cands, rank = candidates(m, to_market_index(m, sc[sc["trade_date"] <= pd.Timestamp(LAST_SIGNAL)], "score", mid))
        for rule, capital in RULES.items():
            for y in STARTS:
                d0 = int(dates.searchsorted(pd.Timestamp(f"{y}-01-01")))
                res = simulate(m, cands, rank, rule_of(rule), CostModel(capital=float(capital), stamp_before=0.001), d0, d_last)
                w = stats.window(res.nav, pool, d0, d_end, dates)
                per = stats.period(res.nav, pool, dates, stats.scoring_days(d0, d_last))
                rows.append({"model_id": mid, "模型": NAMES[mid], "规则": rule, "起步年": y, "第一年收益%": w["first_year"],
                             "年化%": w["cagr"], "年化超额%": per["excess"], "最大回撤%": w["max_dd"], "翻了几倍": w["final_x"],
                             "准入池同期年化%": w["pool_cagr"], "持有年数": round((d_end - d0) / 242, 1)})
            print(f"{mid} {rule} 完成", flush=True)
    return pd.DataFrame(rows)


def report(t: pd.DataFrame) -> None:
    out = result_dir(OUT)
    t.to_csv(out / "starts.csv", index=False, encoding="utf-8-sig")
    secs = []
    for rule in RULES:
        d = t[t["规则"] == rule]
        piv = {c: d.pivot(index="起步年", columns="模型", values=c)[[NAMES[m] for m in MODELS]] for c in ("年化%", "年化超额%", "最大回撤%", "第一年收益%")}
        piv["年化%"].insert(0, "准入池同期年化", d.drop_duplicates("起步年").set_index("起步年")["准入池同期年化%"])
        summ = d.groupby("模型").agg(年化中位数=("年化%", "median"), 年化最差=("年化%", "min"), 超额中位数=("年化超额%", "median"),
                                     超额最差=("年化超额%", "min"), 回撤中位数=("最大回撤%", "median"), 回撤最差=("最大回撤%", "min"),
                                     每个起点都赚钱=("年化%", lambda s: "是" if (s > 0).all() else f"否（{(s <= 0).sum()} 个起点亏）"),
                                     每个起点都跑赢准入池=("年化超额%", lambda s: "是" if (s > 0).all() else f"否（{(s <= 0).sum()} 个）")).reindex([NAMES[m] for m in MODELS])
        secs.append(f"""
### {rule}（20 万本金）

**按起点汇总**（10 个起点：2015 ~ 2024 每年第一个交易日）

{summ.round(1).to_markdown()}

**年化收益（%）**

{piv['年化%'].round(1).to_markdown()}

**年化超额（%，相对准入池等权）**

{piv['年化超额%'].round(1).to_markdown()}

**最大回撤（%）**

{piv['最大回撤%'].round(1).to_markdown()}

**第一年收益（%）**（起步那一年的收益，看“刚开始就碰上坏年份”的风险）

{piv['第一年收益%'].round(1).to_markdown()}
""")
    notes_path = out / "starts_notes.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "（读法待补）"
    write_md(out / "starts.md", f"""
# P7 补充：换起点（PREREG 补充登记十二，描述性分析）

- 代码：`ml/p7_starts.py`；结果 `starts.csv`
- 分数沿用 P7（每年的模型不变），只改起步时间：2015 ~ 2024 每年第一个交易日拿 20 万现金起步，跑到 2026-10；印花税按历史税率
- 年化按“起步日收盘到 2026-10-23、每年 242 个交易日”折算，和 P7 主报告（按计分日数折算）口径略有不同，2015 年起步的数字会差零点几个百分点
  （如 XGBoost F69 主报告 19.9%、本表 19.4%）
- 起点越晚，持有期越短（2024 年起步约 2.8 年）；2023 年以前起步的结果包含“仅供参考”和“偏乐观”的年份
{''.join(secs)}
## 读法

{notes}
""")
    print(f"写出 {out / 'starts.md'}")


def main() -> None:
    t = run()
    register("P7_starts", n=len(t))
    report(t)


if __name__ == "__main__":
    import sys
    if "--report" in sys.argv:
        report(pd.read_csv(result_dir(OUT) / "starts.csv"))
    else:
        main()
