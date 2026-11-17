"""补充登记八的比较：滚动窗口 W2 / W3（每次重训只用之前 2 / 3 年）vs 现在的扩张窗口 V0（2013 年起全部历史）。

三套固定规则（同补充登记七），真实账户、现行印花税：20 万 DB-K30-b8（主）、2 万 DB-K3-b16、20 万 FH-K6-h4。
- 窗口的作用：同一重训频率下 W − V0 的逐日配对差。--stage y 只比每年重训；--stage m 再加每月重训。
- 用户的设计 vs 现在的推荐：每年重训的 W − 每月重训的 V0（两条 DB 规则；原方案本来就是每年重训）。
主指标从 2017-01 起算（2016 年以前 W2、2017 年以前 W3 的训练行和 V0 几乎一样）；全期、2026、逐年另报。
--self-check：W 也读原版打分，"窗口的作用"那几组的逐日差必须正好是 0（冒烟用）。
V0 读本目录的打分，W2 / W3 读 win/w2、win/w3。输出 win/window_eval_<stage>[_selfcheck].json。
用法：python -X utf8 window_eval.py --stage y|m [--self-check]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_market10, signal_range
from decade_sim import period, scoring_days
from engine import candidates, pool_daily_return
from rule_grid import parse
from rule_search import cost_model
from sim import YEAR, simulate
from theme_eval import CASES, TERCILES, daily_ret, ic_table, key, paired, summarize, terciles
from train_cadence import no_power_throttling

HERE = Path(__file__).resolve().parent
OUT = HERE / "win"
VARIANTS = {"W2": OUT / "w2", "W3": OUT / "w3"}
FROM = pd.Timestamp("2017-01-01")


def periods(when: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    return {"2017起": when >= FROM, "全期": np.ones(len(when), dtype=bool), "2026": when >= pd.Timestamp("2026-01-01")}


def by_year(d: np.ndarray, when: pd.DatetimeIndex) -> dict:
    s = pd.Series(d, index=when)
    return {str(y): round(float(v.mean()) * YEAR * 100, 2) for y, v in s.groupby(s.index.year)}


def comparisons(stage: str) -> list[dict]:
    """要比的（规则, 窗口, 比什么, 基准, 新）。kind：Y / M = 同频率下窗口的作用；"Y vs V0-M" = 用户的设计 vs 现在的推荐。"""
    out = []
    for case, (_, cad, _) in CASES.items():
        for w in VARIANTS:
            out.append({"case": case, "window": w, "kind": "Y", "a": ("V0", "Y", case), "b": (w, "Y", case)})
            if cad == "M":
                out.append({"case": case, "window": w, "kind": "Y vs V0-M", "a": ("V0", "M", case), "b": (w, "Y", case)})
                if stage == "m":
                    out.append({"case": case, "window": w, "kind": "M", "a": ("V0", "M", case), "b": (w, "M", case)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("y", "m"), required=True)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    dirs = {"V0": HERE, **{w: HERE if args.self_check else d for w, d in VARIANTS.items()}}
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
    comps = comparisons(args.stage)
    todo = list(dict.fromkeys(j for c in comps for j in (c["a"], c["b"])))
    rets, results = {}, {}
    for src, cad in dict.fromkeys((j[0], j[1]) for j in todo):
        s = pd.read_parquet(dirs[src] / f"scores10_{cad}.parquet")
        cands, rank = candidates(m, s[~low[s.di.to_numpy(), s.ji.to_numpy()]])
        del s
        for job in (j for j in todo if j[:2] == (src, cad)):
            capital, _, name = CASES[job[2]]
            res = simulate(mf, cands, rank, parse(name).rule, cost_model(capital, "real"), d_first, d_last)
            rets[job] = daily_ret(res.nav, idx)
            out = summarize(res.nav, rets[job], pool, pool_s, dates, idx, when, cut)
            out["since2017"] = period(res.nav, pool, dates, idx[when >= FROM])
            results[key(job)] = out
            print(f"{key(job):18s} 年化 {out['all']['cagr']:+6.1f} 超额 {out['all']['excess']:+6.1f} 回撤 {out['all']['max_dd']:6.1f} "
                  f"2017 起年化 {out['since2017']['cagr']:+6.1f} ({time.time() - start:.0f}s)", flush=True)
        del cands, rank
    rows = []
    for c in comps:
        ra, rb = results[key(c["a"])], results[key(c["b"])]
        rows.append({"case": c["case"], "window": c["window"], "kind": c["kind"], "a": key(c["a"]), "b": key(c["b"]),
                     **paired(rets[c["a"]], rets[c["b"]], when, periods(when)),
                     "years": by_year(rets[c["b"]] - rets[c["a"]], when),
                     "regime_gap_diff": {t: round(rb["regime_gap"].get(t, np.nan) - ra["regime_gap"].get(t, np.nan), 2)
                                         for t in TERCILES},
                     "windows_diff": {w: round(rb["windows"][w]["book"] - ra["windows"][w]["book"], 1) for w in rb["windows"]}})
        r = rows[-1]
        print(f"{r['case']} {r['window']} {r['kind']:10s} 2017 起 {r['2017起']['diff']:+6.2f} {r['2017起']['ci']}  "
              f"全期 {r['全期']['diff']:+6.2f}  2026 {r['2026']['diff']:+6.2f}", flush=True)
    if args.self_check:
        bad = [(r["case"], r["window"], r["kind"]) for r in rows if r["kind"] in ("Y", "M") and r["max_abs_daily"] != 0]
        if bad:
            raise SystemExit(f"自检失败：窗口版读原版打分，同频率的逐日差却不是 0：{bad}")
        print("自检通过：同频率比时逐日差全是 0", flush=True)
    d0 = int(dates.searchsorted(FROM))
    cads = ("Y",) if args.stage == "y" else ("Y", "M")
    ic = {f"{w}|{cad}": ic_table(cad, dirs[w], cut, d0, dates) for w in VARIANTS for cad in cads}
    OUT.mkdir(exist_ok=True)
    path = OUT / f"window_eval_{args.stage}{'_selfcheck' if args.self_check else ''}.json"
    payload = {"stage": args.stage, "self_check": args.self_check, "cases": CASES, "from": str(FROM.date()),
               "jobs": results, "comparisons": rows, "ic": ic, "seconds": round(time.time() - start)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
