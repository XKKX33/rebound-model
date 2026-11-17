"""2 万本金、每月重训 + DB-K2-b6 的逐年逐月收益（正式库、不限价、最低 5 元）。只读，不改库。"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from rebound.backtest.capital import CapitalRequest
from rebound.backtest.market import candidates, load_markets, model_scores, pool_daily_return
from rebound.backtest.rules import parse
from rebound.backtest.sim import simulate

LAB = Path(__file__).resolve().parent
DB = LAB.parents[1] / "data" / "market.duckdb"
CACHED = LAB / "capital_formal_20000_nocap.json"
RULE = "DB-K2-b6"

req = CapitalRequest(capital=20_000.0, variant="nocap", min_commission=5.0)
cached = json.loads(CACHED.read_text(encoding="utf-8"))
end = date.fromisoformat(cached["today"])

m = load_markets(DB, end=end)[req.variant]
cands, rank = candidates(m, model_scores(DB, m, req.variant, "M"))
spec = parse(RULE)
d_first = int(m.dates.searchsorted(pd.Timestamp("2015-01-01")))
today = len(m.dates) - 1
d_last = today - 1
res = simulate(m, cands, rank, spec.rule, req.cost, d_first, d_last)

when = m.dates[d_first:today + 1]
nav = res.nav[d_first:today + 1]
start_nav = float(res.nav[d_first])
assert np.isfinite(nav).all(), "净值有空值"

daily = pd.Series(nav, index=when)
month_end = daily.groupby(daily.index.to_period("M")).last()
prev = month_end.shift(1)
prev.iloc[0] = start_nav
month_ret = (month_end / prev - 1) * 100

# 池子（准入池等权）同月收益
pool = pool_daily_return(m)
pdaily = pd.Series(np.nan_to_num(pool[d_first + 1:today + 1]), index=m.dates[d_first + 1:today + 1])
pool_nav = pd.concat([pd.Series([1.0], index=[when[0]]), (1 + pdaily).cumprod()])
pool_end = pool_nav.groupby(pool_nav.index.to_period("M")).last()
pprev = pool_end.shift(1)
pprev.iloc[0] = 1.0
pool_ret = (pool_end / pprev - 1) * 100

years = sorted({p.year for p in month_ret.index})
official = {r["cadence"]: r for r in cached["rows"] if r["name"] == RULE}
single = next(r for r in cached["rows"] if r["cadence"] == "M" and r["name"] == RULE)
off_years = single["years"]

# 年内最大回撤（按日）
def year_dd(year: int) -> float:
    seg = nav[(when.year == year)]
    peak = np.maximum.accumulate(seg)
    return float((seg / peak - 1).min() * 100)

rows = []
for y in years:
    vals = {p.month: month_ret[p] for p in month_ret.index if p.year == y}
    row = {"year": y, **{k: round(v, 1) for k, v in vals.items()}}
    row["total"] = round(float(np.prod([1 + v / 100 for v in vals.values()]) - 1) * 100, 1)
    row["pool"] = round(float(np.prod([1 + pool_ret[p] / 100 for p in pool_ret.index if p.year == y]) - 1) * 100, 1)
    row["dd"] = round(year_dd(y), 1)
    row["official"] = off_years.get(str(y), {}).get("total")
    row["end_wan"] = round(float(month_end[month_end.index.year == y].iloc[-1]) / 10000, 1)
    rows.append(row)

# 月度清单
listing = [{"month": str(p), "ret": round(float(month_ret[p]), 1),
            "pool": round(float(pool_ret.get(p, np.nan)), 1),
            "excess": round(float(month_ret[p] - pool_ret.get(p, np.nan)), 1)} for p in month_ret.index]
worst = sorted(listing, key=lambda r: r["ret"])[:10]
best = sorted(listing, key=lambda r: -r["ret"])[:10]
pos = sum(1 for r in listing if r["ret"] > 0)

out = {"rule": f"M {RULE}", "request": cached["request"], "start": str(when[0].date()),
       "end": str(when[-1].date()), "start_nav": start_nav, "final": round(float(nav[-1]) * 1.0, 2),
       "months": len(listing), "positive": pos, "rows": rows, "worst": worst, "best": best,
       "official": {k: single[k] for k in ("cagr", "excess", "max_dd", "final", "years")}}
(LAB / "monthly_20000_DB-K2-b6.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

# ---------------- markdown ----------------
lines = [f"# 2 万本金 · 每月重训 + {RULE}：逐年逐月收益", "",
         f"- 正式库（不限价），佣金万五、每笔最低 5 元、印花税万五、过户费十万分之一；{out['start']} ~ {out['end']}。",
         f"- 起始 {start_nav:,.0f} 元，期末 {out['final']:,.0f} 元（{out['final'] / start_nav:.1f} 倍）；共 {len(listing)} 个月，"
         f"{pos} 个月赚钱（{pos / len(listing) * 100:.0f}%）。",
         f"- 全期：年化 {single['cagr']:+.1f}%，超额 {single['excess']:+.1f}%，最大回撤 {single['max_dd']:.1f}%。",
         "- 月度收益按月末净值算；2015-01 含最初 5 个交易日的建仓期；2026-10 只到 10-30。"
         "「全年」是 12 个月复利，「口径核对」是缓存里按逐日收益算的全年（两者起点差 5 个交易日，2015 年会有小差别）。", "",
         "## 每年每月（%）", "",
         "| 年 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 全年 | 池子 | 年内回撤 | 口径核对 | 年末(万) |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in rows:
    cells = [f"{r.get(mo, 0.0):+.1f}" if mo in r else "" for mo in range(1, 13)]
    lines.append(f"| {r['year']} | " + " | ".join(cells) +
                 f" | **{r['total']:+.1f}** | {r['pool']:+.1f} | {r['dd']:.1f} | {r['official']:+.1f} | {r['end_wan']:,.1f} |")
lines += ["", "## 最差的 10 个月", "", "| 月份 | 当月 | 池子 | 差 |", "|---|---:|---:|---:|"]
lines += [f"| {r['month']} | {r['ret']:+.1f}% | {r['pool']:+.1f}% | {r['excess']:+.1f} |" for r in worst]
lines += ["", "## 最好的 10 个月", "", "| 月份 | 当月 | 池子 | 差 |", "|---|---:|---:|---:|"]
lines += [f"| {r['month']} | {r['ret']:+.1f}% | {r['pool']:+.1f}% | {r['excess']:+.1f} |" for r in best]
lines += ["", "## 全部月份", "", "| 月份 | 当月 | 池子 | 差 |", "|---|---:|---:|---:|"]
lines += [f"| {r['month']} | {r['ret']:+.1f}% | {r['pool']:+.1f}% | {r['excess']:+.1f} |" for r in listing]
(LAB / "monthly_20000_DB-K2-b6.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print("\n".join(lines[:40]))
print(f"\n... 完整表在 {LAB / 'monthly_20000_DB-K2-b6.md'}")
print(f"期末 {out['final']:,.0f} 元；缓存期末 {single['final']:,.0f} 元")
