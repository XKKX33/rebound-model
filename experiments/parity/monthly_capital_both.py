"""20 万与 2 万推荐规则的逐月期末资金（正式库、不限价、最低 5 元）。只读，不改库。

20 万：每月重训 + DB-K30-b4（滚动样本外按信息比率推荐）
2 万：每月重训 + DB-K2-b6（换起点体检推荐）
输出 monthly_capital_both.json / .md。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from rebound.backtest.capital import CapitalRequest
from rebound.backtest.market import candidates, load_markets, model_scores
from rebound.backtest.rules import parse
from rebound.backtest.sim import simulate

LAB = Path(__file__).resolve().parent
DB = LAB.parents[1] / "data" / "market.duckdb"

CASES = [
    {"label": "20 万", "cached": "capital_formal_200000_nocap.json", "capital": 200_000.0, "rule": "DB-K30-b4"},
    {"label": "2 万", "cached": "capital_formal_20000_nocap.json", "capital": 20_000.0, "rule": "DB-K2-b6"},
]

# 行情和分数两份请求共用，只加载一次
cached0 = json.loads((LAB / CASES[0]["cached"]).read_text(encoding="utf-8"))
end = date.fromisoformat(cached0["today"])
req0 = CapitalRequest(capital=CASES[0]["capital"], variant="nocap", min_commission=5.0)
m = load_markets(DB, end=end)[req0.variant]
cands, rank = candidates(m, model_scores(DB, m, req0.variant, "M"))
d_first = int(m.dates.searchsorted(pd.Timestamp("2015-01-01")))
today = len(m.dates) - 1
d_last = today - 1
when = m.dates[d_first:today + 1]


def case_series(case: dict) -> dict:
    cached = json.loads((LAB / case["cached"]).read_text(encoding="utf-8"))
    req = CapitalRequest(capital=case["capital"], variant="nocap", min_commission=5.0)
    spec = parse(case["rule"])
    res = simulate(m, cands, rank, spec.rule, req.cost, d_first, d_last)
    nav = res.nav[d_first:today + 1]
    assert np.isfinite(nav).all(), f"{case['label']} 净值有空值"
    daily = pd.Series(nav, index=when)
    month_end = daily.groupby(daily.index.to_period("M")).last()
    single = next(r for r in cached["rows"] if r["cadence"] == "M" and r["name"] == case["rule"])
    assert abs(float(nav[-1]) - float(single["final"])) < 1.0, (float(nav[-1]), single["final"])

    rows = []
    for y in sorted({p.year for p in month_end.index}):
        vals = {p.month: float(month_end[p]) for p in month_end.index if p.year == y}
        seg = nav[when.year == y]
        peak = np.maximum.accumulate(seg)
        dd = float((seg / peak - 1).min() * 100)
        rows.append({
            "year": y,
            "months": {str(k): round(v, 2) for k, v in vals.items()},
            "end": round(float(month_end[month_end.index.year == y].iloc[-1]), 2),
            "ret": single["years"].get(str(y), {}).get("total"),
            "dd": round(dd, 1),
        })
    return {
        "label": case["label"],
        "rule": f"M {case['rule']}",
        "capital": case["capital"],
        "request": cached["request"],
        "start": str(when[0].date()),
        "end": str(when[-1].date()),
        "start_nav": float(nav[0]),
        "final": round(float(nav[-1]), 2),
        "cagr": single["cagr"],
        "excess": single["excess"],
        "excess_ci": single["excess_ci"],
        "max_dd": single["max_dd"],
        "rows": rows,
    }


out = {"generated_from": "current model_score (fingerprint 1707b0d30392, before delisting retrain)",
       "cases": [case_series(c) for c in CASES]}
(LAB / "monthly_capital_both.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def wan(x: float) -> str:
    return f"{x / 10000:,.2f}"


lines = ["# 20 万与 2 万推荐规则：逐月期末资金", "",
         f"- 正式库（不限价），佣金万五、每笔最低 5 元、印花税万五、过户费十万分之一；{out['cases'][0]['start']} ~ {out['cases'][0]['end']}。",
         "- 月末资金 = 当月最后一个交易日的账户总额（现金 + 持仓市值），单位万元。",
         "- 每月重训（M）；20 万规则 DB-K30-b4，2 万规则 DB-K2-b6。",
         "- 用的是当前库里的分数（特征指纹 1707b0d30392，退市整理期还没重新训；重训后数字会有小变化）。", ""]
for c in out["cases"]:
    lines += [f"## {c['label']}：{c['rule']}", "",
              f"- 起始 {c['start_nav']:,.0f} 元，期末 {c['final']:,.0f} 元（{c['final'] / c['start_nav']:.1f} 倍）；"
              f"年化 {c['cagr']:+.1f}%，超额 {c['excess']:+.1f}%（区间 [{c['excess_ci'][0]:+.1f}, {c['excess_ci'][1]:+.1f}]），最大回撤 {c['max_dd']:.1f}%。", "",
              "| 年 | 1月 | 2月 | 3月 | 4月 | 5月 | 6月 | 7月 | 8月 | 9月 | 10月 | 11月 | 12月 | 年末资金 | 全年 | 年内回撤 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in c["rows"]:
        cells = [wan(r["months"][str(mo)]) if str(mo) in r["months"] else "" for mo in range(1, 13)]
        lines.append(f"| {r['year']} | " + " | ".join(cells) +
                     f" | **{wan(r['end'])}** | {r['ret']:+.1f}% | {r['dd']:.1f}% |")
    lines += ["", "单位：万元；「全年」是缓存里按逐日收益算的全年收益。", ""]
(LAB / "monthly_capital_both.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

for c in out["cases"]:
    print(c["label"], c["rule"], "start", round(c["start_nav"], 2), "final", c["final"],
          "cagr", c["cagr"], "dd", c["max_dd"], "years", len(c["rows"]))
print("written", LAB / "monthly_capital_both.md")
