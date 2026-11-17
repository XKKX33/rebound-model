"""2 万 V1+F20 的净值和分期收益：每年、每月收益，年末资产，期末资产。

V1 = 加行业特征的打分（ind/scores10_M.parquet），F20 = 热门股刚跌不买；规则 DB-K3-b16，每月重训，本金 2 万。
和 theme_eval_m.json 里 V1|M|2万推荐|F20 的数字对不上就报错。
输出 ind/v1f20_2w_detail.json、ind/v1f20_2w_daily.csv、ind/v1f20_2w_monthly.csv。
用法：DECADE_DATA=ind python -X utf8 theme_nav.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA, load_market10, signal_range
from decade_sim import period, scoring_days
from engine import candidates, pool_daily_return
from rule_grid import parse
from rule_search import cost_model
from sim import simulate
from theme_eval import CASES, drop, hot_breaking

HERE = Path(__file__).resolve().parent
OUT = HERE / "ind"
CAPITAL, CAD, NAME = CASES["2万推荐"]


def daily(nav: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """计分日的逐日收益；第一天相对建仓期末的净值（和 period 的口径一致）。"""
    return nav[idx] / np.r_[nav[idx[0] - 1], nav[idx][:-1]] - 1


def book(nav: np.ndarray, idx: np.ndarray, dates: pd.DatetimeIndex, freq: str) -> pd.DataFrame:
    """按自然年或自然月复利：收益、期末资产、段内最大回撤。"""
    ret = pd.Series(daily(nav, idx), index=dates[idx])
    key = ret.index.to_period("M") if freq == "M" else ret.index.year
    pos = pd.Series(np.arange(len(idx)), index=dates[idx])
    out = []
    for k, r in ret.groupby(key):
        first, last = idx[int(pos[r.index[0]])], idx[int(pos[r.index[-1]])]
        seg = nav[first - 1:last + 1] / nav[first - 1]
        out.append((k, (1 + r).prod() - 1, float(nav[last]), float((seg / np.maximum.accumulate(seg) - 1).min())))
    return pd.DataFrame(out, columns=["period", "ret", "nav_end", "max_dd"])


def main() -> None:
    if DATA.resolve() == HERE:
        raise SystemExit("DECADE_DATA 要指到 V1 的目录（ind）")
    m, low, _ = load_market10(raw=True)
    dates = m.dates
    d_first, d_last = signal_range(dates)
    mf = replace(m, eligible=m.eligible & ~low)
    idx = scoring_days(dates, d_first, d_last)
    scores = pd.read_parquet(DATA / f"scores10_{CAD}.parquet")
    cands, rank = candidates(m, scores[~low[scores.di.to_numpy(), scores.ji.to_numpy()]])
    flags = hot_breaking(m.codes, dates, m.eligible.shape)
    res = simulate(mf, drop(cands, flags["F20"]), rank, parse(NAME).rule, cost_model(CAPITAL, "real"), d_first, d_last)
    nav = res.nav

    days = period(nav, pool_daily_return(mf), dates, idx)
    days = {k: v for k, v in days.items() if k in ("days", "total", "cagr", "max_dd")}
    recorded = json.loads((OUT / "theme_eval_m.json").read_text(encoding="utf-8"))["jobs"][f"V1|{CAD}|2万推荐|F20"]["all"]
    for k in ("days", "total", "cagr", "max_dd"):
        if abs(days[k] - recorded[k]) > 0.011:
            raise SystemExit(f"和 theme_eval_m.json 对不上：{k} {days[k]} vs {recorded[k]}")

    yearly = book(nav, idx, dates, "Y")
    monthly = book(nav, idx, dates, "M")
    monthly["year"] = monthly.period.dt.year
    monthly["month"] = monthly.period.dt.month
    table = monthly.pivot(index="year", columns="month", values="ret")
    final = float(nav[idx[-1]])
    out = {"case": "V1+F20", "capital": CAPITAL, "cadence": CAD, "rule": NAME,
           "first": str(dates[idx[0]].date()), "last": str(dates[idx[-1]].date()),
           "all": days, "final_asset": round(final, 2), "multiple": round(final / CAPITAL, 2),
           "yearly": [{"year": int(r.period), "ret": round(float(r.ret) * 100, 2), "nav_end": round(float(r.nav_end), 2),
                       "max_dd": round(float(r.max_dd) * 100, 2)} for r in yearly.itertuples()],
           "monthly": [{"year": int(r.year), "month": int(r.month), "ret": round(float(r.ret) * 100, 2),
                        "nav_end": round(float(r.nav_end), 2)} for r in monthly.itertuples()]}
    (OUT / "v1f20_2w_detail.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame({"date": dates[idx], "nav": nav[idx], "ret": daily(nav, idx)}).to_csv(
        OUT / "v1f20_2w_daily.csv", index=False, float_format="%.6f")
    monthly[["year", "month", "ret", "nav_end", "max_dd"]].assign(ret=lambda d: d.ret * 100, max_dd=lambda d: d.max_dd * 100).to_csv(
        OUT / "v1f20_2w_monthly.csv", index=False, float_format="%.2f")

    print(f"{out['first']} ~ {out['last']}  {days['days']} 天  年化 {days['cagr']:+.1f}%  累计 {days['total']:+.1f}%  "
          f"期末 {final:,.0f} 元（{final / CAPITAL:.1f} 倍）  最大回撤 {days['max_dd']:.1f}%", flush=True)
    print("\n每年：")
    for r in out["yearly"]:
        print(f"  {r['year']}  {r['ret']:+8.2f}%  年末 {r['nav_end']:>12,.0f}  年内回撤 {r['max_dd']:6.2f}%", flush=True)
    print("\n每月（%）：", flush=True)
    show = table * 100
    print(show.round(2).to_string(), flush=True)
    print(f"\n-> {OUT / 'v1f20_2w_detail.json'}", flush=True)


if __name__ == "__main__":
    main()
