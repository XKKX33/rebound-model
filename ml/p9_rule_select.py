"""P9：只用验证段独立选择交易规则（PREREG 补充登记十四）。

1. 验证段（2021-01-04 ~ 2022-12-30，选择阶段模型的分数）：150 条规则 × 三个交付模型，20 万空仓起步，印花税按历史税率；
   三个模型的逐日超额取平均，用仓库 `stats.scores` 的 `smooth` 选法（网格近邻平滑后的日均超额）打分，取最高的一条。
2. 测试段（2023-01 ~ 2026-10，每年初重训的正式分数，费用同 P5）：选出的规则跑一次，与 DB-K30-b8、FH-K6-h4 并列；
   全部 150 条规则的测试段结果只作描述（看排名），不据此改选。

输出 `ml/results/p9_rule_select/`：valid.csv、test.csv、choice.json、report.md。
运行：python -X utf8 -m ml.p9_rule_select          # 约 2 分钟（900 次模拟）
"""

from __future__ import annotations

import json
import time
from datetime import date

import numpy as np
import pandas as pd

from ml.common import DB_PATH, data_dir, register, result_dir, write_md
from ml.p5_backtest import to_market_index
from ml.p8_delivery import M3, NAMES

OUT = "p9_rule_select"
CAPITAL = 200_000
REFS = ("DB-K30-b8", "FH-K6-h4")      # 作者规则（主）和不经挑选的对照


def _sim_all(m, phase: str, specs, cost) -> tuple[dict, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """每条规则 × 每个模型：逐日组合收益、逐日超额（计分日）和 period 统计。"""
    from rebound.backtest import stats
    from rebound.backtest.market import candidates, pool_daily_return
    from rebound.backtest.sim import simulate
    pool, dates = pool_daily_return(m), m.dates
    first = pd.read_parquet(data_dir() / "scores" / f"xgb_F69__{phase}.parquet", columns=["trade_date"])["trade_date"]
    d_first, d_last = int(dates.get_indexer([first.min()])[0]), int(dates.get_indexer([first.max()])[0])
    idx = stats.scoring_days(d_first, d_last)
    out = {}
    t0 = time.time()
    for mid in M3:
        sc = pd.read_parquet(data_dir() / "scores" / f"{mid}__{phase}.parquet", columns=["trade_date", "ts_code", "score"])
        cands, rank = candidates(m, to_market_index(m, sc, "score", mid))
        for i, spec in enumerate(specs):
            res = simulate(m, cands, rank, spec.rule, cost, d_first, d_last)
            ret = res.nav[idx] / res.nav[idx - 1] - 1
            per = stats.period(res.nav, pool, dates, idx)
            summ = stats.summary(res, pool, dates, idx)
            out[(spec.short, mid)] = {"ret": ret, "ex": ret - pool[idx], "per": per,
                                      "turnover": summ["turnover"], "cost_pct": summ["cost_pct"], "hold_days": summ["hold_days"]}
            if (i + 1) % 50 == 0:
                print(f"[{phase}] {mid} {i + 1}/{len(specs)}（{time.time() - t0:.0f}s）", flush=True)
    return out, idx, dates, pool


def _table(res: dict, specs) -> pd.DataFrame:
    rows = []
    for spec in specs:
        row = {"规则": spec.short, "类": spec.family, "K": spec.k, "h/b": spec.p, "说明": spec.plain}
        for mid in M3:
            p = res[(spec.short, mid)]["per"]
            row[f"{mid}｜年化%"] = p["cagr"]
            row[f"{mid}｜超额%"] = p["excess"]
            row[f"{mid}｜最大回撤%"] = p["max_dd"]
        row["三模型平均｜年化%"] = round(np.mean([row[f"{m}｜年化%"] for m in M3]), 2)
        row["三模型平均｜超额%"] = round(float(np.nanmean(np.mean([res[(spec.short, m)]["ex"] for m in M3], axis=0)) * 242 * 100), 2)
        row["三模型平均｜换手"] = round(np.mean([res[(spec.short, m)]["turnover"] for m in M3]), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    from rebound.backtest import stats
    from rebound.backtest.costs import CostModel
    from rebound.backtest.market import load_markets
    from rebound.backtest.rules import specs as all_specs
    specs = all_specs()
    out = result_dir(OUT)
    # ---- 1. 验证段选择 ----
    mv = load_markets(DB_PATH, start=date(2020, 1, 1))["nocap"]
    rv, idx_v, dates_v, _ = _sim_all(mv, "select", specs, CostModel(capital=float(CAPITAL), stamp_before=0.001))
    book = np.array([np.mean([rv[(s.short, m)]["ret"] for m in M3], axis=0) for s in specs])
    excess = np.array([np.mean([rv[(s.short, m)]["ex"] for m in M3], axis=0) for s in specs])
    ok = np.isfinite(book).all(axis=0) & np.isfinite(excess).all(axis=0)
    near = stats.neighbor_index([("all", s) for s in specs])
    score = stats.scores(book[:, ok], excess[:, ok], near)["smooth"]
    tv = _table(rv, specs)
    tv["选法得分（平滑后年化超额%）"] = np.round(score * 242 * 100, 2)
    tv["验证段排名"] = tv["选法得分（平滑后年化超额%）"].rank(ascending=False, method="min").astype(int)
    tv = tv.sort_values("验证段排名")
    chosen = tv.iloc[0]["规则"]
    tv.to_csv(out / "valid.csv", index=False, encoding="utf-8-sig")
    print(f"验证段选出：{chosen}", flush=True)
    # ---- 2. 测试段 ----
    mt = load_markets(DB_PATH, start=date(2022, 1, 1))["nocap"]
    rt, idx_t, dates_t, pool_t = _sim_all(mt, "final", specs, CostModel(capital=float(CAPITAL)))
    tt = _table(rt, specs)
    tt["测试段排名（三模型平均超额）"] = tt["三模型平均｜超额%"].rank(ascending=False, method="min").astype(int)
    tt = tt.sort_values("测试段排名（三模型平均超额）")
    tt.to_csv(out / "test.csv", index=False, encoding="utf-8-sig")
    focus = {}
    for rule in (chosen, *REFS):
        focus[rule] = {mid: {**rt[(rule, mid)]["per"], "turnover": rt[(rule, mid)]["turnover"], "cost_pct": rt[(rule, mid)]["cost_pct"],
                             "hold_days": rt[(rule, mid)]["hold_days"]} for mid in M3}
    choice = {"chosen": chosen, "valid_rank_of_refs": {r: int(tv.set_index("规则").loc[r, "验证段排名"]) for r in REFS},
              "test_rank": {r: int(tt.set_index("规则").loc[r, "测试段排名（三模型平均超额）"]) for r in (chosen, *REFS)},
              "n_rules": len(specs), "test": focus}
    (out / "choice.json").write_text(json.dumps(choice, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    register("P9_rule_select", chosen=chosen)
    report(tv, tt, choice)


def report(tv: pd.DataFrame, tt: pd.DataFrame, choice: dict) -> None:
    out = result_dir(OUT)
    chosen = choice["chosen"]
    rows = []
    for rule in (chosen, *REFS):
        tag = "验证段选出" if rule == chosen else ("作者规则" if rule == "DB-K30-b8" else "对照（未挑选）")
        for mid in M3:
            p = choice["test"][rule][mid]
            rows.append({"规则": f"{rule}（{tag}）", "模型": NAMES[mid], "年化%": p["cagr"], "年化超额%": p["excess"],
                         "超额区间": f"[{p['excess_ci'][0]:+.1f}, {p['excess_ci'][1]:+.1f}]", "最大回撤%": p["max_dd"],
                         "换手（倍/年）": p["turnover"], "费用（%/年）": p["cost_pct"], "平均持有天数": p["hold_days"]})
    cols_v = ["验证段排名", "规则", "说明", "选法得分（平滑后年化超额%）", "三模型平均｜超额%"] + [f"{m}｜超额%" for m in M3]
    cols_t = ["测试段排名（三模型平均超额）", "规则", "说明", "三模型平均｜超额%", "三模型平均｜年化%"] + [f"{m}｜超额%" for m in M3]
    write_md(out / "report.md", f"""
# P9：只用验证段独立选择交易规则（PREREG 补充登记十四）

- 代码：`ml/p9_rule_select.py`；结果 `valid.csv`（验证段 150 条）、`test.csv`（测试段 150 条，只作描述）、`choice.json`
- 选法（运行前写定）：验证段 2021 ~ 2022、三个交付模型的选择阶段分数，20 万空仓起步、印花税按历史税率；三个模型逐日超额取平均，
  仓库 `stats.scores` 的 `smooth` 选法（网格近邻平滑后的日均超额）取最高。测试段只用来报告，不改选。

## 结果

- **验证段选出：{chosen}**（{tv.iloc[0]['说明']}）
- 作者规则 DB-K30-b8 在验证段排第 {choice['valid_rank_of_refs']['DB-K30-b8']} / {choice['n_rules']}；FH-K6-h4 排第 {choice['valid_rank_of_refs']['FH-K6-h4']}
- 测试段排名（三模型平均年化超额，150 条中）：{chosen} 第 {choice['test_rank'][chosen]}，DB-K30-b8 第 {choice['test_rank']['DB-K30-b8']}，FH-K6-h4 第 {choice['test_rank']['FH-K6-h4']}

### 测试段 2023-01 ~ 2026-10（20 万，现行印花税，与 P5 口径相同）

{pd.DataFrame(rows).to_markdown(index=False)}

### 验证段前 15 名

{tv[cols_v].head(15).to_markdown(index=False)}

### 测试段前 15 名（只作描述）

{tt[cols_t].head(15).to_markdown(index=False)}
""")
    print(f"写出 {out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
