"""诊断：正式推荐规则按现状时点的买单，买入价（不复权开盘价）多少元、一档报价（0.01 元）占百分之几。

用法：python -X utf8 diag_tick.py   输出 diag_tick.json
"""
from __future__ import annotations

import numpy as np

from run_sched import HERE, RULES, SCHEDULES, CapitalRequest, load, parse, save, say, simulate_sched

OUT = HERE / "diag_tick.json"
TICK = 0.01


def main() -> None:
    c = load()
    out = {}
    for capital, names in RULES.items():
        cost = CapitalRequest(capital=float(capital)).cost
        for name in names:
            _, ledger = simulate_sched(c.m, c.cands, c.rank, parse(name).rule, cost, c.d_first, c.d_last, SCHEDULES[0])
            seen = ledger[ledger.day.isin(c.idx)]
            px = c.m.raw["open"][seen.day.to_numpy(), seen.j.to_numpy()]
            years = c.m.dates[seen.day.to_numpy()].year
            row = {}
            for label, keep in (("all", np.ones(len(px), bool)), ("2024+", years >= 2024)):
                p = px[keep]
                row[label] = {"n": int(len(p)), "median_px": round(float(np.median(p)), 2),
                              "under_5": round(float((p < 5).mean()) * 100, 1),
                              "under_10": round(float((p < 10).mean()) * 100, 1),
                              "tick_pct_mean": round(float((TICK / p).mean()) * 100, 3),
                              "tick_pct_median": round(float(np.median(TICK / p)) * 100, 3)}
            out[f"{capital}|{name}"] = row
            say(f"{capital} {name}: {row}")
    save(OUT, out)


if __name__ == "__main__":
    main()
