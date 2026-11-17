"""事后拆分（看过网格结果才加，只解释、不改结论）：N 对 N = 1，按每组最多几只（K）分组，
另看 N = 1 时四个起点都赚的那些规则。只读 <变体>/sims/ 的结果，不重跑模拟。输出 by_k.json。
用法：python -X utf8 by_k.py
"""
from __future__ import annotations

import json

import numpy as np

from minhold_report import DB_RULES, HERE, SHOW_NS, Sims
from rule_grid import parse

GROUPS = {"K 1–3": (1, 3), "K 4–8": (4, 8), "K 10–30": (10, 30)}


def split(s: Sims, n: int) -> dict:
    out = {}
    robust = [r for r in DB_RULES if s.robust(1, r)]
    sets = {name: [r for r in DB_RULES if lo <= parse(r).k <= hi] for name, (lo, hi) in GROUPS.items()}
    sets["N = 1 时四个起点都赚的"] = robust
    for name, rules in sets.items():
        rules = [r for r in rules if s.has(n, r) and s.has(1, r)]
        d_exc = np.array([s.excess(n, r) - s.excess(1, r) for r in rules])
        d_worst = np.array([s.worst(n, r) - s.worst(1, r) for r in rules])
        base = np.array([s.excess(1, r) for r in rules])
        out[name] = {"rules": len(rules), "better": int((d_exc > 0).sum()), "d_excess": round(float(np.median(d_exc)), 1),
                     "d_worst": round(float(np.median(d_worst)), 1), "base_excess": round(float(np.median(base)), 1)}
    return out


def main() -> None:
    out = {}
    for capital in (20000, 200000):
        s = Sims("nocap", capital)
        out[str(capital)] = {str(n): split(s, n) for n in SHOW_NS if n != 1}
    (HERE / "by_k.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for capital, rows in out.items():
        for n in ("2", "4"):
            for name, v in rows[n].items():
                print(capital, f"N={n}", name, v)


if __name__ == "__main__":
    main()
