"""best_summary.json 里每类挑出的规则（和全期年化最高的那条）：从 2015 年一路做到最后一天的期末金额（网格里没存）。
顺带核对：年化、最差起点和网格里存的一样。用法：python -X utf8 finals.py   输出 best_finals.json
"""
from __future__ import annotations

import json

from run_sched import HERE, SCHEDULES, CapitalRequest, load, parse, run, save, say
from sched_sim import hybrid


def sched_of(key: str):
    if key.startswith("hybrid_r"):
        return hybrid(int(key[len("hybrid_r"):]))
    return {s.name: s for s in SCHEDULES}[key]


def main() -> None:
    c = load()
    cost = CapitalRequest(capital=20_000.0).cost
    summary = json.loads((HERE / "best_summary.json").read_text(encoding="utf-8"))
    out = {}
    for key, row in summary.items():
        if key.startswith("fh_"):
            continue
        for name, ref in ((row.get("pick"), row.get("row")), (row["top_cagr"], row["top_row"])):
            if name is None or f"{key}|{name}" in out:
                continue
            r, _ = run(c, parse(name).rule, cost, sched_of(key))
            same = (r["cagr"], r["worst"]) == (ref["cagr"], ref["worst"])
            out[f"{key}|{name}"] = {"final": r["final"], "cagr": r["cagr"], "worst": r["worst"],
                                    "max_dd": r["max_dd"], "same_as_grid": same}
            say(f"{key} {name}: {out[f'{key}|{name}']}")
    save(HERE / "best_finals.json", out)


if __name__ == "__main__":
    main()
