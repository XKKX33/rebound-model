"""补充登记七的比较：V1（加 9 个行业特征重训）vs V0（现在的模型），以及 F（热门股刚开始跌就不买）。

三套固定规则，真实账户、现行印花税、2015-01 起计分：
- 主：20 万每月重训 DB-K30-b8；
- 2 万每月重训 DB-K3-b16；
- 20 万每年重训 FH-K6-h4（原方案）。

--stage q：每季度版的中间检查。每月重训的两条规则改用每季度的打分比 V1 / V0；每年重训那条用每年的。
          F 用现在模型自己频率的打分。另写止损判断：主规则配对差的区间上限 < 0 就不训练每月版。
--stage m：每月版终版。
--self-check：V1 也读原版打分，所有"V1 − V0"的逐日差必须正好是 0（冒烟用），结果写 theme_eval_<stage>_selfcheck.json。
V0 读本目录的打分；V1 读 DECADE_DATA（ind/）里的。输出 DECADE_DATA/theme_eval_<stage>.json。
用法：DECADE_DATA=ind python -X utf8 theme_eval.py --stage q|m [--self-check]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import DATA, DB, PANEL, load_market10, signal_range
from decade_sim import period, scoring_days
from engine import candidates, month_ci, pool_daily_return
from industry_check import day_corr
from momentum_check import month_panel, regime
from rule_grid import parse
from rule_search import cost_model
from sim import YEAR, simulate
from train_cadence import no_power_throttling, panel_key

HERE = Path(__file__).resolve().parent
CASES = {"20万推荐": (200000, "M", "DB-K30-b8-o1o5"), "2万推荐": (20000, "M", "DB-K3-b16-o1o5"),
         "原方案": (200000, "Y", "FH-K6-next-o1o5")}
PRIMARY = "20万推荐"
QS = {"F20": 0.2, "F10": 0.1, "F30": 0.3}
DEV_END = pd.Timestamp("2025-12-31")
WINDOWS = {"2021-05~11": ("2021-05-01", "2021-11-30"), "2026-04~06": ("2026-04-01", "2026-06-30"),
           "2026-07": ("2026-07-01", "2026-07-31")}
TERCILES = ("反转月", "中间", "强势股月")


def breaking(di: np.ndarray, r20: np.ndarray, r60: np.ndarray, q: float) -> np.ndarray:
    """每行是不是"热门股刚开始跌"：前一段涨幅 (1 + r60) / (1 + r20) − 1（60 天前到 20 天前）排在当天这批行的前 q，
    且近 20 日涨幅 < 0。缺值不算。"""
    prior = (1 + r60) / (1 + r20) - 1
    pct = pd.Series(prior).groupby(di).rank(ascending=False, pct=True).to_numpy()  # 最强那只最小
    return (r20 < 0) & (pct <= q)


def hot_breaking(codes: np.ndarray, dates: pd.DatetimeIndex, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """F 的不买名单（日 × 股），排名范围是当天的准入池（剔除 < 2 元，和训练标签同一批行）。"""
    from common import row_index
    p = pq.read_table(PANEL, columns=["ts_code", "trade_date", "eligible", "below2", "r20", "r60"], read_dictionary=["ts_code"],
                      filters=[("eligible", "==", True), ("below2", "==", False)], use_threads=False).to_pandas()
    di, ji = row_index(p.pop("ts_code"), p.pop("trade_date"), codes, dates)
    r20, r60 = p.r20.to_numpy(np.float64), p.r60.to_numpy(np.float64)
    out = {}
    for label, q in QS.items():
        flag = np.zeros(shape, dtype=bool)
        hit = breaking(di, r20, r60, q)
        flag[di[hit], ji[hit]] = True
        out[label] = flag
    return out


def drop(cands: dict[int, np.ndarray], flag: np.ndarray) -> dict[int, np.ndarray]:
    """只改买入的候选顺序（不买名单剔掉）；卖出看的排名矩阵不动。"""
    return {d: c[~flag[d, c]] for d, c in cands.items()}


def daily_ret(nav: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return nav[idx] / nav[idx - 1] - 1


def regime_gap(ret: np.ndarray, pool: np.ndarray, when: pd.DatetimeIndex, cut: pd.Series, mask=None) -> dict:
    """按月复利的（组合 − 准入池），百分点，按当月是强势股月 / 中间 / 反转月取平均。"""
    keep = np.ones(len(ret), dtype=bool) if mask is None else mask
    s = pd.DataFrame({"b": ret[keep], "p": pool[keep]}, index=when[keep])
    mon = (1 + s).groupby(s.index.to_period("M")).prod() - 1
    gap = (mon.b - mon.p) * 100
    return {str(k): round(float(v.mean()), 2) for k, v in gap.groupby(cut.reindex(gap.index), observed=True)}


def windows(ret: np.ndarray, pool: np.ndarray, when: pd.DatetimeIndex) -> dict:
    out = {}
    for label, (a, b) in WINDOWS.items():
        k = (when >= a) & (when <= b)
        out[label] = {"book": round((float(np.prod(1 + ret[k])) - 1) * 100, 1),
                      "pool": round((float(np.prod(1 + pool[k])) - 1) * 100, 1)}
    return out


def paired(a: np.ndarray, b: np.ndarray, when: pd.DatetimeIndex, periods: dict[str, np.ndarray] | None = None) -> dict:
    """b − a 的逐日差：年化 [按月整块重抽 95% 区间]。默认分全期、2015–2025、2026；periods 可换成别的分段（名字 → 布尔掩码）。"""
    d = b - a
    out = {"max_abs_daily": float(np.abs(d).max())}
    if periods is None:
        periods = {"全期": np.ones(len(d), dtype=bool), "2015-2025": when <= DEV_END, "2026": when > DEV_END}
    for label, k in periods.items():
        lo, hi = month_ci(pd.Series(d[k], index=when[k]))
        out[label] = {"diff": round(float(d[k].mean()) * YEAR * 100, 2), "ci": [round(lo * YEAR * 100, 2), round(hi * YEAR * 100, 2)]}
    return out


def terciles() -> pd.Series:
    """每月（PeriodIndex）是反转月 / 中间 / 强势股月：上月末准入池按近 60 日涨幅分五档，这个月最强档 − 最弱档的三分位。"""
    con = duckdb.connect(str(DB), read_only=True)
    try:
        con.execute("SET memory_limit = '1500MB'")
        con.execute("SET threads = 4")
        basic = con.execute("SELECT ts_code, industry FROM stock_basic").df()
        reg = regime(month_panel(con), dict(zip(basic.ts_code, basic.industry.fillna(""))))
    finally:
        con.close()
    return pd.qcut(reg.wml60, 3, labels=list(TERCILES))


def ic_table(cad: str, v1: Path, cut: pd.Series, d0: int, dates: pd.DatetimeIndex) -> dict:
    """同一批行上 V0 / V1 打分的逐日秩相关（×100）和前 30 名的平均名次（百分位），按月平均。"""
    from industry_check import stats
    lab = pq.read_table(HERE / f"labels10_o1o5_{panel_key()}.parquet", columns=["di", "ji", "y"],
                        filters=[("di", ">=", d0)]).to_pandas()
    lab = lab[lab.y.notna()]
    for tag, base in (("V0", HERE), ("V1", v1)):
        s = pq.read_table(base / f"scores10_{cad}.parquet", filters=[("di", ">=", d0)]).to_pandas()
        lab = lab.merge(s.rename(columns={"score": tag}), on=["di", "ji"])
    di = lab.di.to_numpy()
    g = lab.groupby("di")
    yr = g.y.rank(pct=True).to_numpy()
    daily = {}
    for tag in ("V0", "V1"):
        daily[f"ic_{tag}"] = day_corr(di, g[tag].rank(pct=True).to_numpy(), yr) * 100
        top = g[tag].rank(ascending=False, method="first").to_numpy() <= 30
        daily[f"top30_{tag}"] = lab.y[top].groupby(di[top]).mean() * 100
    d = pd.DataFrame(daily)
    mon = d.groupby(dates[d.index.to_numpy()].to_period("M")).mean()
    by = cut.reindex(mon.index)
    return {"months": len(mon), "mean": {c: round(float(mon[c].mean()), 2) for c in mon.columns},
            "ic_diff": stats(mon.ic_V1 - mon.ic_V0), "top30_diff": stats(mon.top30_V1 - mon.top30_V0),
            "by_tercile": {c: {str(k): round(float(v.mean()), 2) for k, v in mon[c].groupby(by, observed=True)}
                           for c in mon.columns}}


def jobs(stage: str) -> list[tuple[str, str, str, str]]:
    """要模拟的（打分, 频率, 规则, 不买名单）。V1 / V0 在同一频率上比；F 用现在模型自己的频率。"""
    out = []
    for case, (_, cad, _) in CASES.items():
        cmp = "Q" if stage == "q" and cad == "M" else cad
        out += [("V0", cmp, case, ""), ("V1", cmp, case, ""), ("V0", cad, case, "")]
        out += [("V0", cad, case, f) for f in QS] + ([("V1", cad, case, "F20")] if stage == "m" else [])
    return list(dict.fromkeys(out))


def comparisons(stage: str) -> list[tuple[str, str, str, tuple, tuple]]:
    """（规则, 比什么, 频率, 基准, 新）。"""
    out = []
    for case, (_, cad, _) in CASES.items():
        cmp = "Q" if stage == "q" and cad == "M" else cad
        out.append((case, "V1", cmp, ("V0", cmp, case, ""), ("V1", cmp, case, "")))
        out += [(case, f, cad, ("V0", cad, case, ""), ("V0", cad, case, f)) for f in QS]
        if stage == "m":
            out += [(case, "V1+F20 vs V1", cad, ("V1", cad, case, ""), ("V1", cad, case, "F20")),
                    (case, "V1+F20 vs V0", cad, ("V0", cad, case, ""), ("V1", cad, case, "F20"))]
    return out


def key(job: tuple) -> str:
    return "|".join(x or "-" for x in job)


def bind(flags: dict, rank: np.ndarray, d_first: int, d_last: int) -> dict:
    """不买名单占准入池多少、占（现在模型）打分前 30 名多少（%）。"""
    r = rank[d_first:d_last + 1]
    scored, top = np.isfinite(r), r <= 30
    return {f: {"pool%": round(float(fl[d_first:d_last + 1][scored].mean() * 100), 2),
                "top30%": round(float(fl[d_first:d_last + 1][top].mean() * 100), 2)} for f, fl in flags.items()}


def summarize(nav, r, pool, pool_s, dates, idx, when, cut) -> dict:
    dev = when <= DEV_END
    return {"all": period(nav, pool, dates, idx), "dev": period(nav, pool, dates, idx[dev]),
            "y2026": period(nav, pool, dates, idx[~dev]), "regime_gap": regime_gap(r, pool_s, when, cut),
            "regime_gap_dev": regime_gap(r, pool_s, when, cut, dev), "windows": windows(r, pool_s, when)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("q", "m"), required=True)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if not args.self_check and DATA.resolve() == HERE:
        raise SystemExit("DECADE_DATA 要指到 V1 的目录（ind）")
    v1 = HERE if args.self_check else DATA
    start = time.time()
    print(f"power throttling off: {no_power_throttling()}", flush=True)
    m, low, _ = load_market10(raw=True)
    dates = m.dates
    d_first, d_last = signal_range(dates)
    mf = replace(m, eligible=m.eligible & ~low)
    pool = pool_daily_return(mf)
    idx = scoring_days(dates, d_first, d_last)
    when, pool_s = dates[idx], pool[idx]
    cut = terciles()
    flags = hot_breaking(m.codes, dates, m.eligible.shape)
    todo = jobs(args.stage)
    rets, results, binding = {}, {}, {}
    for src, cad in dict.fromkeys((j[0], j[1]) for j in todo):
        s = pd.read_parquet((HERE if src == "V0" else v1) / f"scores10_{cad}.parquet")
        cands, rank = candidates(m, s[~low[s.di.to_numpy(), s.ji.to_numpy()]])
        del s
        if src == "V0" and cad == "M":
            binding = bind(flags, rank, d_first, d_last)
        for job in (j for j in todo if j[:2] == (src, cad)):
            capital, _, name = CASES[job[2]]
            c = drop(cands, flags[job[3]]) if job[3] else cands
            res = simulate(mf, c, rank, parse(name).rule, cost_model(capital, "real"), d_first, d_last)
            rets[job] = daily_ret(res.nav, idx)
            results[key(job)] = s = summarize(res.nav, rets[job], pool, pool_s, dates, idx, when, cut)
            print(f"{key(job):24s} 年化 {s['all']['cagr']:+6.1f} 超额 {s['all']['excess']:+6.1f} 回撤 {s['all']['max_dd']:6.1f} "
                  f"({time.time() - start:.0f}s)", flush=True)
        del cands, rank
    comps = []
    for case, kind, cad, a, b in comparisons(args.stage):
        ra, rb = results[key(a)], results[key(b)]
        comps.append({"case": case, "kind": kind, "cadence": cad, **paired(rets[a], rets[b], when),
                      **{f"{g}_diff": {t: round(rb[g].get(t, np.nan) - ra[g].get(t, np.nan), 2) for t in TERCILES}
                         for g in ("regime_gap", "regime_gap_dev")},
                      "windows_diff": {w: round(rb["windows"][w]["book"] - ra["windows"][w]["book"], 1) for w in WINDOWS}})
        c = comps[-1]
        print(f"{case} {kind:14s} {cad} 全期 {c['全期']['diff']:+6.2f} {c['全期']['ci']}  2015-2025 {c['2015-2025']['diff']:+6.2f} "
              f"{c['2015-2025']['ci']}  2026 {c['2026']['diff']:+6.2f}", flush=True)
    if args.self_check:
        bad = [c for c in comps if c["kind"] == "V1" and c["max_abs_daily"] != 0]
        if bad:
            raise SystemExit(f"自检失败：V1 读原版打分，逐日差却不是 0：{[(c['case'], c['max_abs_daily']) for c in bad]}")
        print("自检通过：V1 = V0 时逐日差全是 0", flush=True)
    primary = next(c for c in comps if c["case"] == PRIMARY and c["kind"] == "V1")
    stop = args.stage == "q" and primary["全期"]["ci"][1] < 0
    ic = {cad: ic_table(cad, v1, cut, d_first, dates) for cad in (("Q", "Y") if args.stage == "q" else ("M", "Q", "Y"))}
    out = {"stage": args.stage, "self_check": args.self_check, "v1_dir": str(v1), "cases": CASES, "qs": QS,
           "jobs": results, "comparisons": comps, "binding": binding, "ic": ic, "stop": stop,
           "seconds": round(time.time() - start)}
    path = DATA / f"theme_eval_{args.stage}{'_selfcheck' if args.self_check else ''}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"止损：{stop}；打分 {json.dumps({k: v['mean'] for k, v in ic.items()}, ensure_ascii=False)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
