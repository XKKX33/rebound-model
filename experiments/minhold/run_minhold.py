"""十年回测，每月重训的打分 × DB 规则 × 最少持有天数 N，真实账户逐日模拟（PREREG.md）。

全期（2015 年起）+ 换起点（2017 / 2019 / 2021 / 2023 年第一个开市日用同样的本金重新起步）。
准入池剔除收盘 < 2 元和退市整理期（当时名称带"退"），选股和准入池基准都剔；--no-delist 不剔。
其余（成本、计分日、统计、换起点的算法）直接调用十年实验的 rule_search.summary / fresh_start.window。
输出 <变体>/sims/<本金>_<费用>[_nodelist].json（每组 "N|规则名" 的数字）和 .npz（全期逐日收益，配对比较用）。
算过的组合跳过，中断后重跑接着算。

用法（run_all.py 会设好 DECADE_DATA）：
  python -X utf8 run_minhold.py --check                      N = 1、不剔退市整理期，和十年实验存的结果逐项比
  python -X utf8 run_minhold.py --capital 20000 200000 --rules primary|grid [--ns 1 4 ...] [--fee min5|free5] [--fresh]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DECADE = HERE.parent / "decade"
_data = Path(os.environ.get("DECADE_DATA", "")).resolve() if os.environ.get("DECADE_DATA") else None
if _data is None or _data.parent != HERE or _data.name not in ("nocap", "cap70"):
    raise SystemExit(f"先设 DECADE_DATA 指到 {HERE} 下的 nocap 或 cap70（run_all.py 会设好）")
sys.path.insert(0, str(DECADE))

from common import DATA, DB, PANEL, load_market10, register, row_index, signal_range  # noqa: E402
from decade_sim import scoring_days  # noqa: E402
from engine import Market, candidates, month_ci, pool_daily_return  # noqa: E402
from fresh_start import window  # noqa: E402
from rule_grid import parse, specs  # noqa: E402
from rule_search import cost_model, summary  # noqa: E402
from train_cadence import no_power_throttling  # noqa: E402

from minhold import FRESH_YEARS, LABEL_HOLD, NS, PRIMARY, hold_mix, rest_of_window, simulate_minhold  # noqa: E402

import duckdb  # noqa: E402

SAVE_EVERY = 20
DELIST = "退"


@dataclass
class Setup:
    m: Market
    drop: np.ndarray        # 日 × 股：不进准入池的（< 2 元，delist 时再加退市整理期）
    pool: np.ndarray
    idx: np.ndarray
    d_first: int
    d_last: int
    starts: dict[int, int]
    excluded: dict


def delisting_rows() -> pd.DataFrame:
    """准入池里当时名称带"退"的股票日。当时名称和十年实验 panel10.py 同一算法：namechange 里生效日不晚于当天的
    最后一个名字（同一生效日取公告最晚的），没有记录才用现在的名称。"""
    con = duckdb.connect()
    try:
        con.execute("SET memory_limit='1GB'; SET threads=4")
        con.execute(f"ATTACH '{DB.as_posix()}' AS db (READ_ONLY)")
        return con.execute(f"""
            WITH names AS (
                SELECT ts_code, name, CAST(start_date AS TIMESTAMP) AS start_ts FROM (
                    SELECT *, row_number() OVER (PARTITION BY ts_code, start_date ORDER BY ann_date DESC NULLS LAST, name) AS k
                    FROM db.namechange) WHERE k = 1
            ), r AS (
                SELECT CAST(ts_code AS VARCHAR) AS ts_code, CAST(trade_date AS TIMESTAMP) AS trade_date
                FROM read_parquet('{PANEL.as_posix()}') WHERE eligible
            )
            SELECT r.ts_code, r.trade_date FROM r
            ASOF LEFT JOIN names n ON r.ts_code = n.ts_code AND r.trade_date >= n.start_ts
            LEFT JOIN db.stock_basic b ON r.ts_code = b.ts_code
            WHERE coalesce(regexp_matches(coalesce(n.name, b.name), '{DELIST}'), false)
            ORDER BY 1, 2""").df()
    finally:
        con.close()


def prepare(delist: bool) -> Setup:
    m, low, _ = load_market10(raw=True)
    drop, excluded = low, {"delist_rows": 0, "delist_new": 0, "delist_codes": 0}
    if delist:
        rows = delisting_rows()
        if len(rows):
            di, ji = row_index(pd.Categorical(rows.ts_code), rows.trade_date, m.codes, m.dates)
            excluded = {"delist_rows": len(rows), "delist_new": int((~low[di, ji]).sum()),
                        "delist_codes": int(rows.ts_code.nunique())}
            drop = low.copy()
            drop[di, ji] = True
    m = replace(m, eligible=m.eligible & ~drop)
    dates = m.dates
    d_first, d_last = signal_range(dates)
    starts = {y: max(int(dates.searchsorted(pd.Timestamp(y, 1, 1))), d_first) for y in FRESH_YEARS}
    return Setup(m, drop, pool_daily_return(m), scoring_days(dates, d_first, d_last), d_first, d_last, starts,
                 excluded)


def picks(s: Setup) -> tuple:
    """每月重训的打分去掉不进准入池的行（十年实验 rule_search 只去 < 2 元；这里再去退市整理期），再排名。"""
    scores = pd.read_parquet(DATA / "scores10_M.parquet")
    keep = ~s.drop[scores.di.to_numpy(), scores.ji.to_numpy()]
    return candidates(s.m, scores[keep])


def rest_summary(m: Market, trades: pd.DataFrame) -> dict:
    """诊断：持有不到 4 天就卖掉的成交，拿到买入后第 4 天开盘会多赚还是少赚（相对同期准入池，%）。"""
    r = rest_of_window(m, trades, LABEL_HOLD)
    if r.empty:
        return {"n": 0}
    gap = pd.Series(r.gap.to_numpy(), index=m.dates[r.exit_day.to_numpy()])
    lo, hi = month_ci(gap)
    held = r.exit_day - r.entry_day
    return {"n": int(len(r)), "share_of_trades": round(len(r) / len(trades) * 100, 1),
            "gap": round(float(r.gap.mean()) * 100, 3), "ci": [round(lo * 100, 3), round(hi * 100, 3)],
            "stock": round(float(r.stock.mean()) * 100, 3), "pool": round(float(r.pool.mean()) * 100, 3),
            "beat_pool": round(float((r.gap > 0).mean()) * 100, 1),
            "by_hold": {str(h): {"n": int(len(g)), "gap": round(float(g.gap.mean()) * 100, 3)}
                        for h, g in r.groupby(held)}}


def run_one(s: Setup, cands, rank, name: str, n_hold: int, cost, fresh: bool, diag: bool) -> tuple[dict, np.ndarray]:
    rule = parse(name).rule
    res = simulate_minhold(s.m, cands, rank, rule, cost, s.d_first, s.d_last, min_hold=n_hold)
    out = {"full": summary(res, s.pool, s.m.dates, s.idx), "hold_mix": hold_mix(res.trades)}
    if diag:
        out["rest"] = rest_summary(s.m, res.trades)
    if fresh:
        out["fresh"] = {}
        for year, d0 in s.starts.items():
            again = simulate_minhold(s.m, cands, rank, rule, cost, d0, s.d_last, min_hold=n_hold)
            out["fresh"][str(year)] = window(again.nav, s.pool, d0, s.d_last + 1, s.m.dates)
        worst = min(v["cagr"] for v in out["fresh"].values())
        out["fresh_stats"] = {"worst_cagr": worst, "all_positive": worst > 0}
    return out, res.nav[s.idx] / res.nav[s.idx - 1] - 1


def out_paths(capital: int, fee: str, delist: bool) -> tuple[Path, Path]:
    stem = f"{capital}_{fee}" + ("" if delist else "_nodelist")
    return DATA / "sims" / f"{stem}.json", DATA / "sims" / f"{stem}.npz"


def save(js: Path, npz: Path, results: dict, rets: dict, s: Setup) -> None:
    js.parent.mkdir(parents=True, exist_ok=True)
    tmp = js.with_name(js.stem + ".tmp.json")
    tmp.write_text(json.dumps(results, ensure_ascii=False, indent=0), encoding="utf-8")
    os.replace(tmp, js)
    keys = list(rets)
    tmpz = npz.with_name(npz.stem + ".tmp.npz")
    with open(tmpz, "wb") as f:
        np.savez(f, keys=np.array(keys), ret=np.vstack([rets[k] for k in keys]), pool=s.pool[s.idx], days=s.idx,
                 dates=np.array(s.m.dates[s.idx], dtype="datetime64[D]"))
    os.replace(tmpz, npz)


def load_done(js: Path, npz: Path) -> tuple[dict, dict]:
    if not js.is_file() or not npz.is_file():
        return {}, {}
    z = np.load(npz, allow_pickle=False)
    rets = dict(zip(z["keys"].tolist(), z["ret"]))
    results = {k: v for k, v in json.loads(js.read_text(encoding="utf-8")).items() if k in rets}
    return results, {k: rets[k] for k in results}


def rule_list(capital: int, which: list[str]) -> list[str]:
    """primary：这种本金的两条主规则；grid：90 条 DB 规则（主规则排前面）；也可以直接给规则名。"""
    if which == ["primary"]:
        return list(PRIMARY[capital])
    if which == ["grid"]:
        db = [sp.name for sp in specs() if sp.family == "DB"]
        return list(PRIMARY[capital]) + [n for n in db if n not in PRIMARY[capital]]
    known = {sp.name for sp in specs() if sp.family == "DB"}
    bad = [n for n in which if n not in known]
    if bad:
        raise SystemExit(f"不是 DB 网格里的规则：{bad}")
    return which


def fee_model(capital: int, fee: str):
    cost = cost_model(capital, "real")
    return replace(cost, min_commission=0.0) if fee == "free5" else cost


def check(start: float) -> None:
    """N = 1、不剔退市整理期：四条主规则的全期数字和逐日收益（2 万、20 万）、2 万的四个起点，和十年实验存的逐项一样。"""
    s = prepare(delist=False)
    cands, rank = picks(s)
    src = DECADE / "nocap" if DATA.name == "nocap" else DECADE
    fresh_file = src / ("fresh_nocap_20000.json" if DATA.name == "nocap" else "fresh_20000_all.json")
    stored_fresh = json.loads(fresh_file.read_text(encoding="utf-8"))
    names = sorted({n for pair in PRIMARY.values() for n in pair})
    bad, compared = [], 0
    for capital in PRIMARY:
        stored = json.loads((src / "rules" / str(capital) / "real_M.json").read_text(encoding="utf-8"))
        z = np.load(src / "rules" / str(capital) / "real_M.npz")
        if not (np.array_equal(z["days"], s.idx) and np.array_equal(z["pool"], s.pool[s.idx])):
            bad.append(f"{capital} 计分日或准入池基准")
        stored_ret = dict(zip(z["names"].tolist(), z["ret"]))
        for name in names:
            want_fresh = {y: stored_fresh.get(f"{capital}|M|{name}|min5|{y}") for y in FRESH_YEARS}
            got, ret = run_one(s, cands, rank, name, 1, fee_model(capital, "min5"),
                               fresh=all(v is not None for v in want_fresh.values()), diag=False)
            if json.loads(json.dumps(got["full"])) != stored[name]:
                bad.append(f"{capital} {name} 全期")
            if not np.array_equal(ret, stored_ret[name]):
                bad.append(f"{capital} {name} 逐日收益")
            compared += 1
            for y, want in want_fresh.items():
                if want is not None:
                    compared += 1
                    if json.loads(json.dumps(got["fresh"][str(y)])) != want:
                        bad.append(f"{capital} {name} {y} 起步")
            print(f"check {capital} {name}: cagr {got['full']['cagr']:+.2f} vs {stored[name]['cagr']:+.2f} "
                  f"({time.time() - start:.0f}s)", flush=True)
    if bad:
        raise SystemExit("和十年实验对不上：" + "；".join(bad))
    print(f"decade check ({DATA.name}): {compared} items identical", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--capital", type=int, nargs="+", default=[20000, 200000])
    ap.add_argument("--rules", nargs="+", default=["primary"])
    ap.add_argument("--ns", type=int, nargs="+", default=list(NS))
    ap.add_argument("--fee", choices=("min5", "free5"), default="min5")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--no-delist", action="store_true")
    args = ap.parse_args()
    start = time.time()
    print(f"power throttling off: {no_power_throttling()}; variant {DATA.name}", flush=True)
    if args.check:
        check(start)
        return
    delist = not args.no_delist
    s = prepare(delist)
    print(f"excluded {s.excluded} ({time.time() - start:.0f}s)", flush=True)
    cands, rank = picks(s)
    total = 0
    for capital in args.capital:
        cost = fee_model(capital, args.fee)
        js, npz = out_paths(capital, args.fee, delist)
        results, rets = load_done(js, npz)
        names = rule_list(capital, args.rules)
        todo = [(n, name) for n in args.ns for name in names
                if f"{n}|{name}" not in results or (args.fresh and "fresh" not in results[f"{n}|{name}"])]
        print(f"{capital} {args.fee}: {len(todo)} of {len(args.ns) * len(names)} to run", flush=True)
        for i, (n, name) in enumerate(todo, 1):
            key = f"{n}|{name}"
            diag = n == 1 and name in PRIMARY.get(capital, ())
            results[key], rets[key] = run_one(s, cands, rank, name, n, cost, args.fresh, diag)
            f = results[key]["full"]
            w = results[key].get("fresh_stats")
            print(f"{capital} N={n:<2d} {name:18s} cagr {f['cagr']:+6.1f} exc {f['excess']:+6.1f} dd {f['max_dd']:6.1f} "
                  f"hold {f['hold_days']} cost {f['cost_pct']:5.2f}"
                  + (f" | worst start {w['worst_cagr']:+6.1f}" if w else "")
                  + f" ({i}/{len(todo)}, {time.time() - start:.0f}s)", flush=True)
            if i % SAVE_EVERY == 0 or i == len(todo):
                save(js, npz, results, rets, s)
        total += len(todo)
    register("run_minhold", {"capital": args.capital, "rules": args.rules, "ns": args.ns, "fee": args.fee,
                             "fresh": args.fresh, "delist": delist, "runs": total}, s.excluded)
    print(f"done ({time.time() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
