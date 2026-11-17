"""正式的本金测算代码（rebound/backtest）vs 十年实验：喂十年实验的打分，逐项比结果。两边的库和文件都只读。

1. 模拟：rebound.backtest.market.load_markets 读十年合成库，喂 decade/scores10_M.parquet（限 70 元）、
   nocap/scores10_M.parquet（不限价），抽一组规则，2 万、20 万本金，和 rules/<本金>/real_M.json 逐项比。
2. 选法：用十年实验存下的逐日收益（rules/<本金>/real_*.npz），跑正式的 walk_forward / best_of，和 select.json 比。
3. 换起点：几条规则 2017/2019/2021/2023 起步，和 fresh_20000_all.json 比。
用法：python -X utf8 -B capital_check.py   输出 capital_check.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from rebound.backtest import stats  # noqa: E402
from rebound.backtest.costs import CostModel  # noqa: E402
from rebound.backtest.market import candidates, load_markets, pool_daily_return  # noqa: E402
from rebound.backtest.rules import parse, rule_of  # noqa: E402
from rebound.backtest.sim import simulate  # noqa: E402

DECADE_DB = ROOT / "data/research-2013-2026.duckdb"
FORK2 = ROOT / "experiments/decade"
SOURCES = {"cap70": FORK2, "nocap": FORK2 / "nocap"}
RULES = ["FH-K1-next-o1o2", "FH-K6-next-o1o5", "FH-K2-next-o1o10", "FH-K10-next-o1o3", "FH-K20-next-o1o5",
         "DB-K1-b1-o1o5", "DB-K3-b16-o1o5", "DB-K2-b6-o1o5", "DB-K10-b12-o1o5", "DB-K30-b8-o1o5", "DB-K4-b1.5-o1o5"]
FRESH = ["DB-K3-b16-o1o5", "DB-K2-b6-o1o5", "DB-K10-b12-o1o5", "FH-K6-next-o1o5"]
CADENCES = ("Y", "H", "Q", "M")
OUT = HERE / "capital_check.json"
RESULT: dict = {}
T0 = time.time()


def say(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [{time.time() - T0:5.0f}s] {msg}", flush=True)


def save() -> None:
    OUT.write_text(json.dumps(RESULT, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def their_scores(variant: str, cadence: str, m) -> pd.DataFrame:
    src = SOURCES[variant]
    axes = json.loads((src / "axes.json").read_text(encoding="utf-8"))
    codes, dates = np.array(axes["codes"]), pd.DatetimeIndex(pd.to_datetime(axes["dates"]))
    s = pd.read_parquet(src / f"scores10_{cadence}.parquet")
    di = m.dates.get_indexer(dates)[s.di.to_numpy()]
    ji = m.code_index(codes[s.ji.to_numpy()])
    if (di < 0).any() or (ji < 0).any():
        raise SystemExit(f"{variant} {cadence}：十年实验的打分有日期或代码不在正式矩阵的轴上")
    return pd.DataFrame({"di": di, "ji": ji, "score": s.score.to_numpy()})


def same(a: dict, b: dict) -> list[str]:
    a = json.loads(json.dumps(a))
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def check_sims(markets) -> None:
    for variant, m in markets.items():
        d_first, d_last = int(m.dates.searchsorted(pd.Timestamp("2015-01-01"))), len(m.dates) - 1 - 5
        idx = stats.scoring_days(d_first, d_last)
        pool = pool_daily_return(m)
        cands, rank = candidates(m, their_scores(variant, "M", m))
        for capital in (20000, 200000):
            path = SOURCES[variant] / "rules" / str(capital) / "real_M.json"
            if not path.is_file():
                continue
            theirs = json.loads(path.read_text(encoding="utf-8"))
            for name in RULES:
                t0 = time.time()
                res = simulate(m, cands, rank, rule_of(name), CostModel(capital=float(capital)), d_first, d_last)
                mine = stats.summary(res, pool, m.dates, idx)
                diff = same(mine, theirs[name])
                RESULT.setdefault("sims", {})[f"{variant}|{capital}|{name}"] = {
                    "same": not diff, "differ": diff, "seconds": round(time.time() - t0, 1),
                    "final_mine": mine["final"], "final_theirs": theirs[name]["final"]}
                say(f"{variant} {capital} {name}: {'same' if not diff else diff} ({time.time() - t0:.1f}s)")
                save()


def check_fresh(markets) -> None:
    m = markets["cap70"]
    d_first, d_last = int(m.dates.searchsorted(pd.Timestamp("2015-01-01"))), len(m.dates) - 1 - 5
    pool = pool_daily_return(m)
    cands, rank = candidates(m, their_scores("cap70", "M", m))
    theirs = json.loads((FORK2 / "fresh_20000_all.json").read_text(encoding="utf-8"))
    for name in FRESH:
        for y in (2017, 2019, 2021, 2023):
            d0 = max(int(m.dates.searchsorted(pd.Timestamp(y, 1, 1))), d_first)
            res = simulate(m, cands, rank, rule_of(name), CostModel(capital=20000.0), d0, d_last)
            mine = stats.window(res.nav, pool, d0, d_last + 1, m.dates)
            key = f"20000|M|{name}|min5|{y}"
            diff = same(mine, theirs[key])
            RESULT.setdefault("fresh", {})[key] = {"same": not diff, "differ": diff}
            say(f"fresh {key}: {'same' if not diff else diff}")
            save()


def check_select() -> None:
    for capital in (20000, 200000):
        folder = FORK2 / "rules" / str(capital)
        configs, rets, pool, dates = [], [], None, None
        for cadence in CADENCES:
            z = np.load(folder / f"real_{cadence}.npz")
            pool, dates = z["pool"], pd.DatetimeIndex(z["dates"])
            for name, r in zip(z["names"], z["ret"]):
                configs.append((cadence, parse(str(name), legacy_ok=True)))  # 十年实验存的是开盘卖的老名字
                rets.append(r)
        book = np.vstack(rets)
        excess = book - pool
        near = stats.neighbor_index(configs)
        theirs = json.loads((folder / "select.json").read_text(encoding="utf-8"))
        walked = {}
        for picker in stats.PICKERS:
            w = stats.walk_forward(book, excess, dates, near, picker)
            mine = {**stats.series_stats(w["book"], w["excess"], w["dates"]),
                    "picks": {y: [configs[i][0], configs[i][1].legacy] for y, i in w["picks"].items()}}
            walked[picker] = mine
            ref = {k: theirs["walk_forward"][picker][k] for k in mine}
            diff = same(mine, ref)
            RESULT.setdefault("select", {})[f"{capital}|{picker}"] = {"same": not diff, "differ": diff}
            say(f"select {capital} {picker}: {'same' if not diff else diff}")
        RESULT["select"][f"{capital}|best_picker"] = {"same": stats.best_of(walked) == theirs["best_picker"],
                                                      "mine": stats.best_of(walked), "theirs": theirs["best_picker"]}
        save()


def main() -> None:
    check_select()
    markets = load_markets(DECADE_DB, log=say)
    say("markets loaded")
    check_sims(markets)
    check_fresh(markets)
    bad = [k for part in ("sims", "fresh", "select") for k, v in RESULT.get(part, {}).items() if not v["same"]]
    RESULT["summary"] = {"checked": sum(len(RESULT.get(p, {})) for p in ("sims", "fresh", "select")), "differ": bad}
    save()
    say(f"done: {RESULT['summary']}")


if __name__ == "__main__":
    main()
