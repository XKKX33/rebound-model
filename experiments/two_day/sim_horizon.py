"""一个模型的每月重训打分 × 150 条持股规则，2 万真实账户逐日模拟：全期（2015 年起）+ 换起点（2017 / 2019 / 2021 / 2023）。

模拟器、成本、规则网格、统计和十年实验的 rule_search.py / fresh_start.py 完全相同（直接调用）。多出来的一处：
准入池剔除退市整理期（当时名称带"退"，正式系统 2026-10-31 起的口径），选股和准入池基准都剔；--no-delist 不剔，
那样必须和十年实验存的结果逐项一样（--check-decade）。
输出 <变体>/sims/<模型>_<费用>_<印花税>[_nodelist].json（每条规则的全期数字和四个起点）和 .npz（全期逐日收益，配对比较用）。
算过的规则跳过，中断后重跑接着算。
用法（run_all.py 会设好 DECADE_DATA）：python -X utf8 sim_horizon.py --model o1o2-cuda [--fee min5|free5] [--tier real|hist]
  [--rules 规则名 ...] [--no-delist] [--check-decade]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from horizon import CAPITAL, DECADE, FRESH_YEARS, MODELS, data_dir, delisting_rows, fresh_stats  # noqa: I001
from common import DB, PANEL, load_market10, register, row_index, signal_range
from decade_sim import scoring_days
from engine import Market, candidates, pool_daily_return
from fresh_start import window
from rule_grid import parse, specs
from rule_search import cost_model, summary
from sim import simulate
from train_cadence import no_power_throttling

SAVE_EVERY = 10
#: --check-decade 核对的规则（取十年实验换起点文件里也有的那几条）
CHECK_RULES = ("DB-K2-b6-o1o5", "DB-K3-b16-o1o5", "DB-K10-b12-o1o5", "DB-K2-b4-o1o5", "DB-K3-b4-o1o5")


@dataclass
class Setup:
    m: Market
    pool: np.ndarray
    idx: np.ndarray
    d_first: int
    d_last: int
    starts: dict[int, int]
    excluded: int


def prepare(delist: bool) -> Setup:
    """行情、准入池（剔除 < 2 元；delist 时再剔退市整理期）、准入池基准、计分日、四个起点。"""
    m, low, _ = load_market10(raw=True)
    drop = low.copy()
    excluded = 0
    if delist:
        rows = delisting_rows(PANEL, DB)
        if len(rows):
            di, ji = row_index(pd.Categorical(rows.ts_code), rows.trade_date, m.codes, m.dates)
            drop[di, ji] = True
        excluded = len(rows)
    mf = replace(m, eligible=m.eligible & ~drop)
    dates = m.dates
    d_first, d_last = signal_range(dates)
    starts = {y: max(int(dates.searchsorted(pd.Timestamp(y, 1, 1))), d_first) for y in FRESH_YEARS}
    return Setup(mf, pool_daily_return(mf), scoring_days(dates, d_first, d_last), d_first, d_last, starts, excluded)


def picks(s: Setup, scores: pd.DataFrame) -> tuple:
    """打分里不在准入池的行（< 2 元、退市整理期）去掉，再排名。"""
    keep = s.m.eligible[scores.di.to_numpy(), scores.ji.to_numpy()]
    return candidates(s.m, scores[keep])


def run_rule(s: Setup, cands, rank, name: str, cost) -> tuple[dict, np.ndarray]:
    """一条规则：全期（2015 起）的数字和逐日收益，四个起点各自重新起步的数字。"""
    rule = parse(name).rule
    res = simulate(s.m, cands, rank, rule, cost, s.d_first, s.d_last)
    out = {"full": summary(res, s.pool, s.m.dates, s.idx), "fresh": {}}
    for year, d0 in s.starts.items():
        again = simulate(s.m, cands, rank, rule, cost, d0, s.d_last)
        out["fresh"][str(year)] = window(again.nav, s.pool, d0, s.d_last + 1, s.m.dates)
    out["fresh_stats"] = fresh_stats(out["fresh"])
    return out, res.nav[s.idx] / res.nav[s.idx - 1] - 1


def out_paths(out: Path, model: str, fee: str, tier: str, delist: bool) -> tuple[Path, Path]:
    stem = f"{model}_{fee}_{tier}" + ("" if delist else "_nodelist")
    return out / "sims" / f"{stem}.json", out / "sims" / f"{stem}.npz"


def save(js: Path, npz: Path, results: dict, rets: dict, s: Setup) -> None:
    js.parent.mkdir(parents=True, exist_ok=True)
    tmp = js.with_name(js.stem + ".tmp.json")
    tmp.write_text(json.dumps(results, ensure_ascii=False, indent=0), encoding="utf-8")
    os.replace(tmp, js)
    names = list(rets)
    tmpz = npz.with_name(npz.stem + ".tmp.npz")
    with open(tmpz, "wb") as f:
        np.savez(f, names=np.array(names), ret=np.vstack([rets[n] for n in names]), pool=s.pool[s.idx], days=s.idx,
                 dates=np.array(s.m.dates[s.idx], dtype="datetime64[D]"))
    os.replace(tmpz, npz)


def load_done(js: Path, npz: Path) -> tuple[dict, dict]:
    if not js.is_file() or not npz.is_file():
        return {}, {}
    z = np.load(npz, allow_pickle=False)
    rets = dict(zip(z["names"].tolist(), z["ret"]))
    results = {k: v for k, v in json.loads(js.read_text(encoding="utf-8")).items() if k in rets}
    return results, {k: rets[k] for k in results}


def check_decade(out: Path, results: dict) -> list[str]:
    """不剔退市整理期时，和十年实验存的全期结果（rules/20000/real_M.json）、换起点结果逐项比。返回对不上的项。"""
    src = DECADE / "nocap" if out.name == "nocap" else DECADE
    full = json.loads((src / "rules" / "20000" / "real_M.json").read_text(encoding="utf-8"))
    fresh_file = src / ("fresh_nocap_20000.json" if out.name == "nocap" else "fresh_20000_all.json")
    fresh = json.loads(fresh_file.read_text(encoding="utf-8"))
    bad = []
    for name, got in results.items():
        if json.loads(json.dumps(got["full"])) != full[name]:
            bad.append(f"{name} 全期")
        for year, value in got["fresh"].items():
            want = fresh.get(f"20000|M|{name}|min5|{year}")
            if want is not None and json.loads(json.dumps(value)) != want:
                bad.append(f"{name} {year} 起步")
            elif want is None:
                bad.append(f"{name} {year} 起步：十年实验没存")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--fee", choices=("min5", "free5"), default="min5")
    ap.add_argument("--tier", choices=("real", "hist"), default="real")
    ap.add_argument("--rules", nargs="*", default=None)
    ap.add_argument("--no-delist", action="store_true")
    ap.add_argument("--check-decade", action="store_true", help="不剔退市整理期，核对几条规则和十年实验逐项一样")
    args = ap.parse_args()
    out, start, model = data_dir(), time.time(), MODELS[args.model]
    delist = not (args.no_delist or args.check_decade)
    names = list(CHECK_RULES) if args.check_decade else (args.rules or [s.name for s in specs()])
    js, npz = out_paths(out, args.model, args.fee, args.tier, delist)
    results, rets = load_done(js, npz)
    todo = [n for n in names if n not in results]
    print(f"power throttling off: {no_power_throttling()}; {args.model} {args.fee} {args.tier} delist={delist}: "
          f"{len(todo)} of {len(names)} rules to run", flush=True)
    if todo:
        s = prepare(delist)
        print(f"delisting rows excluded: {s.excluded} ({time.time() - start:.0f}s)", flush=True)
        cost = cost_model(CAPITAL, args.tier)
        cost = replace(cost, min_commission=0.0) if args.fee == "free5" else cost
        cands, rank = picks(s, pd.read_parquet(out / model.scores))
        for n, name in enumerate(todo, 1):
            results[name], rets[name] = run_rule(s, cands, rank, name, cost)
            f, w = results[name]["full"], results[name]["fresh_stats"]
            print(f"{args.model} {name:22s} full cagr {f['cagr']:+6.1f} dd {f['max_dd']:6.1f} | fresh worst "
                  f"{w['worst_cagr']:+6.1f} ({n}/{len(todo)}, {time.time() - start:.0f}s)", flush=True)
            if n % SAVE_EVERY == 0 or n == len(todo):
                save(js, npz, results, rets, s)
        register("sim_horizon", {"model": args.model, "fee": args.fee, "tier": args.tier, "delist": delist,
                                 "rules": len(todo)}, {"excluded": s.excluded})
    if args.check_decade:
        bad = check_decade(out, {n: results[n] for n in names})
        if bad:
            raise SystemExit("和十年实验对不上：" + "；".join(bad))
        print(f"decade check: {len(names)} rules identical (full + fresh starts)", flush=True)
    print(f"done ({time.time() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
