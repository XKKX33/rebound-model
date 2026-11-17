"""换手当天的买卖时点：十年回测（正式库、每月重训的分数、正式模拟器的规则、真实账户口径）。

用法：python -X utf8 run_sched.py            核对 → 推荐规则的各种时点（results.json）→ 2 万、20 万规则网格（grid_*.json）
     python -X utf8 run_sched.py --no-grid  只跑前两步
定死的口径（看结果之前）：
- 规则：2 万 = 每月 DB-K2-b8（正式推荐），参照 DB-K2-b6（上一版推荐）；20 万 = 每月 DB-K30-b6（正式推荐）。
- 账户：佣金万五、每笔最低 5 元，印花税现行万五，过户费十万分之一，一手取整（和 rebound capital 默认相同）。
- 时间：2015-01 第一个开市日起到最后一个打分日；换起点 2017 / 2019 / 2021 / 2023 年第一个开市日。
- 2 万看四个起点的最差年化（正式的小本金选法），20 万看全期超额。时点见 sched_sim.SCHEDULES；"现状"另加
  9:30 成交的滑点 slip（买单要靠同一个开盘卖出的钱时按 开盘价 × (1 + slip) 成交），算出各方案和现状打平的 slip。
- 先核对：现状、slip = 0 和正式模拟器的逐日净值、成交逐位相同，对不上就停。
- 网格（2 万）：DB，K ∈ {1, 2, 3, 4, 6} × 9 个缓冲，每种时点按正式小本金选法（四个起点都赚、最差起点年化最高）重新挑。
- 网格（20 万，2026-11-01 为"永远半仓"加）：DB，K ∈ {8, 10, 15, 20, 30} × 9 个缓冲，现状 / 半仓 / 尾盘卖（前一晚定）
  各按正式选法挑（全期信息比率最高；它在某个起点亏钱就改挑四个起点都赚、最差起点年化最高的）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from rebound.model.train import open_store  # noqa: E402
from rebound.backtest import stats  # noqa: E402
from rebound.backtest.capital import FRESH_YEARS, SIGNAL_FROM, CapitalRequest  # noqa: E402
from rebound.backtest.market import Market, candidates, load_markets, model_scores, pool_daily_return  # noqa: E402
from rebound.backtest.rules import DB_B, Spec, parse  # noqa: E402
from rebound.backtest.sim import YEAR, simulate  # noqa: E402

from sched_sim import SCHEDULES, Schedule, reserve_slots, simulate_sched  # noqa: E402

DB_PATH = ROOT / "data" / "market.duckdb"
VARIANT, CADENCE = "nocap", "M"
RULES = {20_000: ("DB-K2-b8", "DB-K2-b6"), 200_000: ("DB-K30-b6",)}
SLIPS = (-0.002, 0.0, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02)
GRID_K = (1, 2, 3, 4, 6)
GRID_SLIPS = (0.002, 0.005)  # 网格里"现状"另加的两档滑点
KEEP = ("cagr", "excess", "excess_ci", "max_dd", "final", "hold_days", "turnover", "cost_pct", "idle_cash_pct",
        "avg_holdings", "buys_per_day")
OUT, GRID_OUT, PICKS_OUT = HERE / "results.json", HERE / "grid_20000.json", HERE / "grid_20000_picks.json"
GRID_FIELDS = ("cagr", "excess", "max_dd", "worst", "fresh", "turnover", "hold_days")
#: 20 万网格：正式推荐落在 K ≥ 8 这一片；只比现状、半仓、尾盘卖（前一晚定）三种时点。
BIG_K, BIG_SCHEDULES = (8, 10, 15, 20, 30), ("now", "half", "close_sell_prev_rank")
BIG_OUT, BIG_PICKS_OUT = HERE / "grid_200000.json", HERE / "grid_200000_picks.json"


def say(line: str) -> None:
    print(line, flush=True)


@dataclass(frozen=True)
class Ctx:
    m: Market
    cands: dict
    rank: np.ndarray
    pool: np.ndarray
    idx: np.ndarray
    d_first: int
    d_last: int
    today: int
    starts: dict


def last_scored() -> date:
    with open_store(DB_PATH) as store:
        rows = store.connection.execute("SELECT cadence, max(trade_date) FROM model_score WHERE variant = ? GROUP BY 1",
                                        [VARIANT]).fetchall()
    lasts = {d for _, d in rows}
    if len(lasts) != 1:
        raise SystemExit(f"三种重训频率打分到的日子不一样：{rows}")
    return lasts.pop()


def load() -> Ctx:
    """和 rebound capital 同一套：最后一个能成交的信号日是昨天，计分到今天。"""
    m = load_markets(DB_PATH, end=last_scored(), log=say)[VARIANT]
    dates, today = m.dates, len(m.dates) - 1
    d_first, d_last = int(dates.searchsorted(SIGNAL_FROM)), today - 1
    cands, rank = candidates(m, model_scores(DB_PATH, m, VARIANT, CADENCE))
    starts = {y: max(int(dates.searchsorted(pd.Timestamp(y, 1, 1))), d_first) for y in FRESH_YEARS}
    return Ctx(m, cands, rank, pool_daily_return(m), stats.scoring_days(d_first, d_last), d_first, d_last, today,
               starts)


def run(c: Ctx, rule, cost, sched: Schedule, slip: float = 0.0) -> tuple[dict, np.ndarray]:
    """全期数字 + 四个起点的年化；另报买单里靠同一腿卖出的钱的比例，和这部分一年的成交额（相对净值，倍）。"""
    reserve = reserve_slots(rule.k, sched.keep_cash)
    res, ledger = simulate_sched(c.m, c.cands, c.rank, rule, cost, c.d_first, c.d_last, sched, slip, reserve)
    s = stats.summary(res, c.pool, c.m.dates, c.idx)
    out = {k: s[k] for k in KEEP}
    seen = ledger[ledger.day.isin(c.idx)]
    funded = seen[seen.funded]
    out["funded_share"] = round(float(seen.funded.mean()) * 100, 1) if len(seen) else None
    out["funded_turnover"] = round(float((funded.value / funded.nav_prev).sum()) / (len(c.idx) / YEAR), 1)
    got = {}
    for y, d0 in c.starts.items():
        again, _ = simulate_sched(c.m, c.cands, c.rank, rule, cost, d0, c.d_last, sched, slip, reserve)
        got[str(y)] = stats.window(again.nav, c.pool, d0, c.today, c.m.dates)["cagr"]
    out["fresh"], out["worst"] = got, min(got.values())
    return out, res.nav[c.idx] / res.nav[c.idx - 1] - 1
def check(c: Ctx, rule, cost) -> None:
    """现状、slip = 0 必须和正式模拟器的开盘卖规则逐位相同（逐日净值、每笔成交）。
    2026-11-01 起正式规则的 `.rule` 是尾盘卖：这里按原来的开盘卖时点比（simulate_sched 不看 cell，只看 sched）。"""
    rule = replace(rule, cell=replace(rule.cell, exit="open"))
    base = simulate(c.m, c.cands, c.rank, rule, cost, c.d_first, c.d_last)
    mine, _ = simulate_sched(c.m, c.cands, c.rank, rule, cost, c.d_first, c.d_last, SCHEDULES[0])
    if not (np.array_equal(base.nav, mine.nav, equal_nan=True) and base.trades.equals(mine.trades)):
        raise SystemExit(f"{rule.name}：现状时点和正式模拟器对不上")


def cached_row(capital: int, short: str) -> dict | None:
    """正式本金测算最新的缓存里同一本金、同一条规则（每月重训）的那一行：只作交叉核对。"""
    files = sorted(DB_PATH.with_name(f"{DB_PATH.stem}.capital").glob("*.json"), key=lambda f: f.stat().st_mtime,
                   reverse=True)
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        r = d.get("request", {})
        if d.get("version") != 2 or r.get("capital") != capital or r.get("min_commission") != 5.0 \
                or r.get("variant") != VARIANT:
            continue
        hit = [row for row in d["rows"] if row["cadence"] == CADENCE and row["name"] == short]
        return hit[0] if hit else None
    return None


def paired(a: np.ndarray, b: np.ndarray, when: pd.DatetimeIndex) -> dict:
    """两条逐日收益之差的年化（百分点），按月重抽 95% 区间。"""
    d = a - b
    lo, hi = stats.month_ci(pd.Series(d, index=when))
    return {"diff": round(float(d.mean()) * YEAR * 100, 2),
            "ci": [round(lo * YEAR * 100, 2), round(hi * YEAR * 100, 2)]}


def breakeven(curve: list[tuple[float, float]], target: float) -> float | str:
    """现状加多少滑点（%）才和 target 打平：在 slip 曲线上线性插值；整条曲线都高于 / 低于 target 就报界外。"""
    for (s0, v0), (s1, v1) in zip(curve, curve[1:]):
        if v0 == target:
            return round(s0 * 100, 2)
        if (v0 - target) * (v1 - target) < 0:
            return round((s0 + (target - v0) * (s1 - s0) / (v1 - v0)) * 100, 2)
    if curve[-1][1] == target:
        return round(curve[-1][0] * 100, 2)
    return f">{curve[-1][0] * 100:g}" if all(v > target for _, v in curve) else f"<{curve[0][0] * 100:g}"


def save(path: Path, obj) -> None:
    tmp = path.with_name(path.stem + ".tmp.json")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)
def table(rows: dict, slip_rows: dict) -> None:
    for sched in SCHEDULES:
        r = rows[sched.name]
        tail = f"  对现状 {r['vs_now']['diff']} {r['vs_now']['ci']}  打平滑点 {r['breakeven_slip']}" if "vs_now" in r else ""
        say(f"  {sched.name:<22} 年化 {r['cagr']:>6}  超额 {r['excess']:>6}  回撤 {r['max_dd']:>6}  "
            f"最差起点 {r['worst']:>5} {list(r['fresh'].values())}  靠卖出的钱 {r['funded_share']}%{tail}")
    for s, r in slip_rows.items():
        say(f"  now+{s * 100:g}%{'':<14} 年化 {r['cagr']:>6}  超额 {r['excess']:>6}  回撤 {r['max_dd']:>6}  "
            f"最差起点 {r['worst']:>5} {list(r['fresh'].values())}")


def stage1(c: Ctx) -> None:
    out = {"last": str(c.m.dates[c.today].date()), "slips": list(SLIPS),
           "schedules": {s.name: s.label for s in SCHEDULES}, "rules": {}}
    when = c.m.dates[c.idx]
    for capital, names in RULES.items():
        cost = CapitalRequest(capital=float(capital)).cost
        for name in names:
            t0 = time.time()
            rule = parse(name).rule
            check(c, rule, cost)
            rows, rets = {}, {}
            for sched in SCHEDULES:
                rows[sched.name], rets[sched.name] = run(c, rule, cost, sched)
            slip_rows = {s: rows["now"] if s == 0 else run(c, rule, cost, SCHEDULES[0], s)[0] for s in SLIPS}
            curves = {metric: [(s, slip_rows[s][metric]) for s in SLIPS] for metric in ("excess", "worst")}
            for sched in SCHEDULES[1:]:
                r = rows[sched.name]
                r["vs_now"] = paired(rets[sched.name], rets["now"], when)
                r["breakeven_slip"] = {metric: breakeven(curve, r[metric]) for metric, curve in curves.items()}
            got = {"cagr": rows["now"]["cagr"], "fresh": rows["now"]["fresh"]}
            ref = cached_row(capital, parse(name).short)
            if ref is not None and ref.get("fresh"):
                prod = {"cagr": ref["cagr"], "fresh": {y: v["cagr"] for y, v in ref["fresh"]["starts"].items()}}
                got = {**got, "production_cache": prod, "same_as_cache": prod == got}
            out["rules"][f"{capital}|{name}"] = {"schedules": rows, "slip": {f"{s:g}": slip_rows[s] for s in SLIPS},
                                                 "cross_check": got}
            save(OUT, out)
            say(f"[时点] {capital} {name}（{time.time() - t0:.0f}s）和正式缓存一致：{got.get('same_as_cache')}")
            table(rows, slip_rows)
def grid(c: Ctx, capital: int = 20_000, configs: list | None = None, out: Path = GRID_OUT,
         picks_out: Path = PICKS_OUT) -> None:
    """小本金：每种时点在 45 条 DB 规则里按正式小本金选法（四个起点都赚、最差起点年化最高）重新挑。
    默认 2 万，每种时点，现状另加两档滑点。算过的跳过，中断后接着算。"""
    cost = CapitalRequest(capital=float(capital)).cost
    if configs is None:
        configs = [(s.name, s, 0.0) for s in SCHEDULES] + [(f"now+{x * 100:g}%", SCHEDULES[0], x) for x in GRID_SLIPS]
    done = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    t0 = time.time()
    for key, sched, slip in configs:
        for k in GRID_K:
            for b in DB_B:
                spec = Spec("DB", k, b)
                tag = f"{key}|{spec.short}"
                if tag not in done:
                    r, _ = run(c, spec.rule, cost, sched, slip)
                    done[tag] = {f: r[f] for f in GRID_FIELDS}
            save(out, done)
        say(f"[网格 {capital}] {key} 算完（{time.time() - t0:.0f}s）")
    picks = {}
    for key, _, _ in configs:
        mine = {t.split("|", 1)[1]: v for t, v in done.items() if t.split("|", 1)[0] == key}
        robust = {n: v for n, v in mine.items() if v["worst"] > 0}
        best = max(robust, key=lambda n: robust[n]["worst"]) if robust else None
        top = sorted(mine.items(), key=lambda kv: -kv[1]["worst"])[:5]
        picks[key] = {"pick": best, "row": mine.get(best), "robust_count": len(robust),
                      "top5": [[n, v["worst"], v["cagr"]] for n, v in top]}
        row = mine.get(best) or {}
        say(f"  {key:<22} 选 {best}  最差起点 {row.get('worst')}  全期年化 {row.get('cagr')}  "
            f"四个起点都赚的 {len(robust)}/{len(mine)}")
    save(picks_out, picks)


def grid_big(c: Ctx) -> None:
    """20 万：三种时点各自在 45 条 DB 规则（K ∈ BIG_K × 9 个缓冲）里按正式选法挑：全期信息比率最高的那条，
    它在某个起点亏钱就改挑四个起点都赚、最差起点年化最高的那条。算过的跳过。"""
    cost = CapitalRequest(capital=200_000.0).cost
    done = json.loads(BIG_OUT.read_text(encoding="utf-8")) if BIG_OUT.is_file() else {}
    by_name, bench, t0 = {s.name: s for s in SCHEDULES}, c.pool[c.idx], time.time()
    for key in BIG_SCHEDULES:
        for k in BIG_K:
            for b in DB_B:
                spec = Spec("DB", k, b)
                tag = f"{key}|{spec.short}"
                if tag not in done:
                    r, ret = run(c, spec.rule, cost, by_name[key])
                    ex = ret - bench
                    sd = float(ex.std(ddof=1))
                    done[tag] = {**{f: r[f] for f in GRID_FIELDS}, "idle_cash_pct": r["idle_cash_pct"],
                                 "ir": float(ex.mean() / sd * YEAR ** 0.5) if sd > 0 else float("-inf")}
            save(BIG_OUT, done)
        say(f"[网格 20 万] {key} 算完（{time.time() - t0:.0f}s）")
    picks = {}
    for key in BIG_SCHEDULES:
        mine = {t.split("|", 1)[1]: v for t, v in done.items() if t.split("|", 1)[0] == key}
        best = max(mine, key=lambda n: mine[n]["ir"])
        robust = {n: v for n, v in mine.items() if v["worst"] > 0}
        steady = max(robust, key=lambda n: robust[n]["worst"]) if robust else None
        pick = best if mine[best]["worst"] > 0 else steady
        picks[key] = {"pick": pick, "row": mine.get(pick), "best_ir": best, "steadiest": steady,
                      "robust_count": len(robust)}
        row = mine.get(pick) or {}
        say(f"  {key:<22} 选 {pick}（信息比率最高 {best}）  年化 {row.get('cagr')}  超额 {row.get('excess')}  "
            f"回撤 {row.get('max_dd')}  最差起点 {row.get('worst')}  现金 {row.get('idle_cash_pct')}%")
    save(BIG_PICKS_OUT, picks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-grid", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    c = load()
    say(f"[行情] 载入完（{time.time() - t0:.0f}s）：{c.m.dates[c.d_first].date()} ~ {c.m.dates[c.today].date()}")
    stage1(c)
    if not args.no_grid:
        grid(c)
        grid_big(c)
    say(f"[完] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
