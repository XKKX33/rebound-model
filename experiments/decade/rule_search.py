"""持股规则网格搜索：每种重训频率的打分 × rule_grid 的全部规则，真实账户逐日模拟。

成本：佣金万五（每笔最低 5 元）、过户费十万分之一、一手取整；印花税 real = 全期按现行 0.05%，
hist = 2023-08-28 之前 0.1%。每组规则存逐日收益（计分日上），选优在 rule_select.py 里做。
已经算过的组合会跳过，中断后重跑接着算。

用法：python -X utf8 rule_search.py [--capital 200000] [--tier real] [--cadences Y H Q M] [--finalists]
--finalists：只算 rule_select 选出的入围规则（读 select.json），用来补 hist 档。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from common import DATA, load_market10, register, signal_range
from decade_sim import by_year, period, scoring_days
from engine import candidates, pool_daily_return
from rule_grid import CADENCES, Spec, out_dir, parse, specs
from sim import YEAR, CostModel, simulate
from train_cadence import no_power_throttling

SAVE_EVERY = 25


def cost_model(capital: float, tier: str) -> CostModel:
    base = CostModel(capital=capital)
    return replace(base, stamp_before=0.001) if tier == "hist" else base


def summary(res, pool: np.ndarray, dates: pd.DatetimeIndex, idx: np.ndarray) -> dict:
    """一组规则的全期数字、逐年超额和收益，以及交易量、费用、空仓。"""
    nav = res.nav
    out = period(nav, pool, dates, idx)
    years = by_year(nav, pool, dates, idx)
    out["years"] = {y: {"total": v["total"], "excess": v["excess"]} for y, v in years.items()}
    out["final"] = round(float(nav[idx[-1]]), 0)
    trades = res.trades
    out["buys_per_day"] = round(len(trades) / len(idx), 2)
    out["hold_days"] = round(float((trades.exit_day - trades.entry_day).mean()), 1) if len(trades) else None
    span = nav[idx].mean()
    out["turnover"] = round(float(res.traded[idx].sum() / span / (len(idx) / YEAR)) / 2, 1)
    out["cost_pct"] = round(float(res.costs[idx].sum() / span / (len(idx) / YEAR)) * 100, 2)
    out["idle_cash_pct"] = round(float(np.nanmean(res.cash[idx] / nav[idx])) * 100, 1)
    out["avg_holdings"] = round(float(res.holdings[idx].mean()), 1)
    out["poor_buys"] = int(res.poor_buys)
    return out


def paths(capital: float, tier: str, cadence: str):
    d = out_dir(capital)
    return d / f"{tier}_{cadence}.npz", d / f"{tier}_{cadence}.json"


def load_done(capital: float, tier: str, cadence: str) -> tuple[list[str], list[np.ndarray], dict]:
    npz, js = paths(capital, tier, cadence)
    if not npz.is_file() or not js.is_file():
        return [], [], {}
    z = np.load(npz, allow_pickle=False)
    stats = json.loads(js.read_text(encoding="utf-8"))
    return list(z["names"]), list(z["ret"]), stats


def save(capital, tier, cadence, names, rets, stats, pool_idx, day_idx, dates) -> None:
    npz, js = paths(capital, tier, cadence)
    npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz.with_name(npz.stem + ".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez(f, names=np.array(names), ret=np.vstack(rets), pool=pool_idx, days=day_idx,
                 dates=np.array(dates[day_idx], dtype="datetime64[D]"))
    os.replace(tmp, npz)
    js.write_text(json.dumps(stats, ensure_ascii=False, indent=0), encoding="utf-8")


def wanted(args) -> dict[str, list[Spec]]:
    if not args.finalists:
        return {c: specs() for c in args.cadences}
    chosen = json.loads((out_dir(args.capital) / "select.json").read_text(encoding="utf-8"))["finalists"]
    out: dict[str, list[Spec]] = {}
    for c, name in chosen:
        out.setdefault(c, []).append(parse(name))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=200_000.0)
    ap.add_argument("--tier", choices=("real", "hist"), default="real")
    ap.add_argument("--cadences", nargs="+", default=list(CADENCES))
    ap.add_argument("--finalists", action="store_true")
    args = ap.parse_args()
    start = time.time()
    todo: dict[str, list[Spec]] = {}
    for cadence, want in wanted(args).items():
        if not (DATA / f"scores10_{cadence}.parquet").is_file():
            print(f"{cadence}: 没有 scores10_{cadence}.parquet，跳过", flush=True)
            continue
        done = set(load_done(args.capital, args.tier, cadence)[0])
        left = [s for s in want if s.name not in done]
        if left:
            todo[cadence] = left
    if not todo:
        print("全部算过了", flush=True)
        return
    print(f"power throttling off: {no_power_throttling()}", flush=True)
    m, low, _ = load_market10(raw=True)
    dates = m.dates
    d_first, d_last = signal_range(dates)
    mf = replace(m, eligible=m.eligible & ~low)
    pool = pool_daily_return(mf)
    idx = scoring_days(dates, d_first, d_last)
    cost = cost_model(args.capital, args.tier)
    for cadence, left in todo.items():
        names, rets, stats = load_done(args.capital, args.tier, cadence)
        scores = pd.read_parquet(DATA / f"scores10_{cadence}.parquet")
        cands, rank = candidates(m, scores[~low[scores.di.to_numpy(), scores.ji.to_numpy()]])
        del scores
        for n, spec in enumerate(left, 1):
            res = simulate(mf, cands, rank, spec.rule, cost, d_first, d_last)
            names.append(spec.name)
            rets.append(res.nav[idx] / res.nav[idx - 1] - 1)
            stats[spec.name] = s = summary(res, pool, dates, idx)
            if n % SAVE_EVERY == 0 or n == len(left):
                save(args.capital, args.tier, cadence, names, rets, stats, pool[idx], idx, dates)
            print(f"{args.tier} {cadence} {spec.name:22s} excess {s['excess']:+6.1f} cagr {s['cagr']:+6.1f} "
                  f"dd {s['max_dd']:6.1f} ({n}/{len(left)}, {time.time() - start:.0f}s)", flush=True)
        del cands, rank
    register("rule_search", {"capital": args.capital, "tier": args.tier, "finalists": args.finalists,
                             "n_specs": {c: len(v) for c, v in todo.items()}}, {})
    print(f"done ({time.time() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
