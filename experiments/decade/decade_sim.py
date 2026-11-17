"""十年持股模拟：4 种重训频率 × o1o5 的 18 组持股规则，本金 20 万，逐日模拟，再按年拆。

成本档：hist（主档，印花税按当时的税率，2023-08-28 之前 0.1%）、real（全期按现行 0.05%）；
prop（资金不受限、按比例收费）只给两条主规则。输出 sim10_results.json。
用法：python -X utf8 decade_sim.py
"""
from __future__ import annotations

import json
import time
from dataclasses import replace

import duckdb
import numpy as np
import pandas as pd

from common import DATA, DB, load_market10, register, signal_range
from train_cadence import no_power_throttling
from engine import BASELINE, candidates, month_ci, pool_daily_return
from grid import rules_for
from grid3 import SMALL
from sim import WARMUP, YEAR, CostModel, Rule, simulate

REAL = CostModel(capital=200_000.0)
TIERS = {"hist": replace(REAL, stamp_before=0.001), "real": REAL}
PROP = (REAL.commission + REAL.transfer, REAL.commission + REAL.stamp + REAL.transfer)
MAIN = ("FH-K6-next-o1o5", "FH-K2-next-o1o5-small")
CADENCES = ("Y", "H", "Q", "M")
INDEXES = {"000300.SH": "沪深300", "000905.SH": "中证500", "000852.SH": "中证1000", "000001.SH": "上证指数"}
P0P1_FROM = pd.Timestamp("2022-09-01")  # 和 p0p1 第三、四轮同一段，作对照


def rules() -> dict[str, Rule]:
    out = {r.name: r for r in rules_for(BASELINE)}
    out.update({Rule(r.kind, r.k, BASELINE, buffer=r.buffer).name + "-small": Rule(r.kind, r.k, BASELINE, buffer=r.buffer)
                for r in SMALL})
    return out


def period(nav: np.ndarray, pool: np.ndarray, dates: pd.DatetimeIndex, idx: np.ndarray) -> dict:
    """一段净值日 idx：总收益、年化、超额（年化 [按月重抽区间]）、β、α（年化）、最大回撤。"""
    ret = nav[idx] / nav[idx - 1] - 1
    bench = pool[idx]
    ok = np.isfinite(ret) & np.isfinite(bench)
    ret, bench, when = ret[ok], bench[ok], dates[idx][ok]
    lo, hi = month_ci(pd.Series(ret - bench, index=when))
    beta = float(np.cov(ret, bench)[0, 1] / np.var(bench, ddof=1)) if len(ret) > 2 else float("nan")
    seg = nav[idx[0] - 1:idx[-1] + 1] / nav[idx[0] - 1]
    years = len(idx) / YEAR
    return {"days": int(len(idx)), "total": round((float(seg[-1]) - 1) * 100, 2),
            "cagr": round((float(seg[-1]) ** (1 / years) - 1) * 100, 2),
            "excess": round(float((ret - bench).mean()) * YEAR * 100, 2),
            "excess_ci": [round(lo * YEAR * 100, 2), round(hi * YEAR * 100, 2)],
            "beta": round(beta, 3), "alpha": round(float((ret - beta * bench).mean()) * YEAR * 100, 2),
            "max_dd": round(float((seg / np.maximum.accumulate(seg) - 1).min()) * 100, 2)}


def scoring_days(dates: pd.DatetimeIndex, d_first: int, d_last: int) -> np.ndarray:
    """计分的净值日：跳过建仓期（和 p0p1 的 sim._days 同一规则）。"""
    return np.arange(d_first + 1 + WARMUP, d_last + 2)


def by_year(nav, pool, dates, idx) -> dict:
    years = dates[idx].year
    return {str(y): period(nav, pool, dates, idx[years == y]) for y in np.unique(years)}

def pool_nav(pool: np.ndarray, d_first: int) -> np.ndarray:
    nav = np.full(len(pool), np.nan)
    nav[d_first] = 1.0
    nav[d_first + 1:] = np.cumprod(1 + np.nan_to_num(pool[d_first + 1:]))
    return nav


def index_years(dates: pd.DatetimeIndex, idx: np.ndarray) -> dict:
    """各指数在同样的计分日上的逐年收益和全期年化（%）。"""
    con = duckdb.connect(str(DB), read_only=True)
    try:
        raw = con.execute("SELECT ts_code, CAST(trade_date AS TIMESTAMP) AS trade_date, close FROM index_daily").df()
    finally:
        con.close()
    years, out = dates[idx].year, {}
    for code, name in INDEXES.items():
        c = raw[raw.ts_code == code].set_index("trade_date").close.sort_index().reindex(dates).ffill().to_numpy()
        per = {str(y): round((c[idx[years == y][-1]] / c[idx[years == y][0] - 1] - 1) * 100, 2) for y in np.unique(years)}
        per["all"] = round(((c[idx[-1]] / c[idx[0] - 1]) ** (YEAR / len(idx)) - 1) * 100, 2)
        out[name] = per
    return out


def paired_diff(a: np.ndarray, b: np.ndarray, dates: pd.DatetimeIndex, idx: np.ndarray) -> dict:
    """两条净值逐日收益之差：全期年化 [按月重抽区间]，逐年年化，以及有几年 a 更好。"""
    d = (a[idx] / a[idx - 1] - 1) - (b[idx] / b[idx - 1] - 1)
    ok = np.isfinite(d)
    lo, hi = month_ci(pd.Series(d[ok], index=dates[idx][ok]))
    years = dates[idx].year
    per = {str(y): round(float(np.nanmean(d[years == y])) * YEAR * 100, 2) for y in np.unique(years)}
    return {"ann": round(float(d[ok].mean()) * YEAR * 100, 2), "ci": [round(lo * YEAR * 100, 2), round(hi * YEAR * 100, 2)],
            "years": per, "better_years": int(sum(v > 0 for v in per.values())), "n_years": len(per)}


def main() -> None:
    start = time.time()
    print(f"power throttling off: {no_power_throttling()}", flush=True)
    m, low, _ = load_market10(raw=True)
    dates = m.dates
    d_first, d_last = signal_range(dates)
    mf = replace(m, eligible=m.eligible & ~low)
    pool = pool_daily_return(mf)
    idx = scoring_days(dates, d_first, d_last)
    p0p1 = idx[dates[idx] >= P0P1_FROM]
    meta = json.loads((DATA / "panel10.meta.json").read_text(encoding="utf-8"))
    pnav = pool_nav(pool, d_first)
    out = {"capital": REAL.capital, "main": MAIN,
           "costs": {**{k: v.__dict__ for k, v in TIERS.items()}, "prop": {"bps": [round(x * 1e4, 2) for x in PROP]}},
           "context": {"pool": {"all": period(pnav, pool, dates, idx), "years": by_year(pnav, pool, dates, idx)},
                       "index": index_years(dates, idx), "pool_size": meta["pool_per_day_by_year"],
                       "first": str(dates[idx[0]].date()), "last": str(dates[idx[-1]].date())},
           "results": {}}
    navs, all_rules = {}, rules()
    for cadence in CADENCES:
        path = DATA / f"scores10_{cadence}.parquet"
        if not path.is_file():
            continue
        scores = pd.read_parquet(path)
        cands, rank = candidates(m, scores[~low[scores.di.to_numpy(), scores.ji.to_numpy()]])
        del scores
        found = out["results"][cadence] = {}
        for name, rule in all_rules.items():
            row = {}
            for tier, cost in {**TIERS, **({"prop": PROP} if name in MAIN else {})}.items():
                res = simulate(mf, cands, rank, rule, cost, d_first, d_last)
                row[tier] = {"all": period(res.nav, pool, dates, idx), "years": by_year(res.nav, pool, dates, idx),
                             "p0p1": period(res.nav, pool, dates, p0p1),
                             "final": round(float(res.nav[d_last + 1] / res.nav[d_first]) * REAL.capital, 0)}
                if res.cash is not None:
                    row[tier]["idle_cash_pct"] = round(float(np.nanmean(res.cash[idx] / res.nav[idx])) * 100, 1)
                navs[(cadence, name, tier)] = res.nav
            found[name] = row
            r = row["hist"]["all"]
            print(f"{cadence} {name:24s} cagr {r['cagr']:+6.1f} excess {r['excess']:+6.1f} beta {r['beta']:.2f} "
                  f"final {row['hist']['final']:>12,.0f} ({time.time() - start:.0f}s)", flush=True)
        del cands, rank
    out["cadence_vs_yearly"] = {f"{c}|{n}|{t}": paired_diff(navs[(c, n, t)], navs[("Y", n, t)], dates, idx)
                                for c in CADENCES[1:] if c in out["results"] and "Y" in out["results"]
                                for n in all_rules for t in TIERS}
    out["seconds"] = round(time.time() - start)
    (DATA / "sim10_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    register("sim", {"cadences": list(out["results"]), "rules": list(all_rules), "tiers": list(TIERS),
                     "capital": REAL.capital},
             {c: {n: out["results"][c][n]["hist"]["all"]["excess"] for n in MAIN} for c in out["results"]})
    print("-> sim10_results.json", flush=True)


if __name__ == "__main__":
    main()
