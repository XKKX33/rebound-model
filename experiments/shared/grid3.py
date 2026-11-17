"""第三轮（PREREG 补充登记三）：准入池加 2 元下限、按下限重训的打分、2 万本金的真实账户成本，重跑 G1。

36 组配置 = 原 27 组 + 小资金组 9 组（只用 o1o5）。成本档：
  real（G1 主档）、high（全部费率 ×1.5，G1 第二个条件）；只报告：free5（免五）、hist（减半前按 0.1% 印花税）、
  prop（资金不受限，按比例收费、不取整）。
cycle_wave 对照：已有的每日过门槛池剔掉 < 2 元的票，按 FH、同样的 K 跑（主网格用 K=6）。
用法：python -X utf8 grid3.py → p3_results.json
"""
from __future__ import annotations

import json
from copy import copy

import numpy as np
import pandas as pd

from engine import CELLS, HERE, attach_raw, candidates, load_market, matched_pool, pool_daily_return, pool_legs, register
from floor import below_floor
from grid import add_matched, cycle_candidates, pool_stats, rules_for
from sim import CostModel, Rule, _days, paired, simulate, summarize

REAL = CostModel()
TIERS = {"real": REAL, "high": REAL.scaled(1.5), "free5": CostModel(min_commission=0.0),
         "hist": CostModel(stamp_before=0.001), "prop": (REAL.commission + REAL.transfer,
                                                         REAL.commission + REAL.stamp + REAL.transfer)}
SMALL = [*(Rule("FH", k, None) for k in (1, 2, 3)), *(Rule("DB", k, None, buffer=b) for k in (2, 3, 4) for b in (2, 4))]


def passes(row: dict, key: str = "excess") -> bool:
    """G1（补充登记三）：开发期 real 档年化超额的区间下限 > 0，且 high 档点估计 > 0。"""
    return row["real"]["dev"][key]["ci"][0] > 0 and row["high"]["dev"][key]["ann"] > 0


def extras(m, res, d_first: int, d_last: int) -> dict:
    idx = _days(m, d_first, d_last)["all"]
    out = {"blocked_buys": res.blocked_buys, "poor_buys": res.poor_buys, "stuck_days": res.stuck_days,
           "poor_event_pct": round(res.poor_events / res.buy_events * 100, 1) if res.buy_events else None}
    if res.cash is not None:
        out["idle_cash_pct"] = round(float(np.nanmean(res.cash[idx] / res.nav[idx])) * 100, 1)
        out["final_nav"] = round(float(res.nav[d_last + 1]), 0)
    return out


def main() -> None:
    m = attach_raw(load_market())
    d_first, d_last = m.signal_range
    low = below_floor(m)
    mf = copy(m)
    mf.eligible = m.eligible & ~low
    pool, legs = pool_daily_return(mf), pool_legs(mf)
    cyc, _, _ = cycle_candidates(m)
    cyc = {d: order[~low[d, order]] for d, order in cyc.items()}
    cells = {c.name: c for c in CELLS}
    chosen = json.loads((HERE / "p1_cells.json").read_text(encoding="utf-8"))
    out = {"tiers": {k: (v.__dict__ if isinstance(v, CostModel) else {"prop_bps": [round(x * 1e4, 2) for x in v]})
                     for k, v in TIERS.items()}, "cycle": {}, "rules": {}}
    cycle_real = {}

    def cycle_run(cell, k: int):
        key = f"cycle-FH-K{k}-{cell.name}"
        if key not in out["cycle"]:
            out["cycle"][key] = {}
            for tier, cost in TIERS.items():
                res = simulate(mf, cyc, None, Rule("FH", k, cell), cost, d_first, d_last)
                out["cycle"][key][tier] = {**summarize(mf, res, pool, d_first, d_last), **extras(mf, res, d_first, d_last)}
                if tier == "real":
                    cycle_real[key] = res
        return key

    configs = [(cells[n], r, "") for n in chosen for r in rules_for(cells[n])]
    configs += [(cells["o1o5"], Rule(r.kind, r.k, cells["o1o5"], buffer=r.buffer), "-small") for r in SMALL]
    for cell, rule, group in configs:
        scores = pd.read_parquet(HERE / f"scores_{cell.name}_floor2.parquet")
        cands, rank = candidates(m, scores[~low[scores.di.to_numpy(), scores.ji.to_numpy()]])
        ref = cycle_run(cell, rule.k if rule.k < 6 else 6)
        bench = matched_pool(legs, rule.kind, cell)
        row = {}
        for tier, cost in TIERS.items():
            res = simulate(mf, cands, rank, rule, cost, d_first, d_last)
            row[tier] = {**add_matched(mf, res, summarize(mf, res, pool, d_first, d_last), bench, d_first, d_last),
                         **extras(mf, res, d_first, d_last)}
            if tier == "real":
                row[tier]["vs_cycle"] = paired(mf, res, cycle_real[ref], d_first, d_last)
                row[tier]["vs_cycle_ref"] = ref
        row["g1_pass"], row["g1_pass_matched"] = passes(row), passes(row, "matched")
        name = rule.name + group
        out["rules"][name] = row
        register("P3-grid", {"rule": name, "tiers": out["tiers"]},
                 {t: row[t]["dev"]["excess"] for t in ("real", "high")} | {"g1_pass": row["g1_pass"]})
        r = row["real"]
        print(f"{name:26s} dev real {r['dev']['excess']['ann']:+6.1f} {r['dev']['excess']['ci']} high "
              f"{row['high']['dev']['excess']['ann']:+6.1f} | hold {r['hold']['excess']['ann']:+6.1f} | all "
              f"{r['all']['excess']['ann']:+6.1f} | cagr {r['all']['cagr']:+6.1f} final {r.get('final_nav')} "
              f"idle {r.get('idle_cash_pct')}% poor-days {r.get('poor_event_pct')}% | pass {row['g1_pass']}", flush=True)
    passed = {k: v for k, v in out["rules"].items() if v["g1_pass"]}
    best = max(passed, key=lambda k: passed[k]["real"]["dev"]["excess"]["ann"]) if passed else None
    out["g1"] = {"passed": sorted(passed), "selected": best,
                 "selected_hold": None if best is None else {t: out["rules"][best][t]["hold"]["excess"] for t in TIERS},
                 "matched_passed": sorted(k for k, v in out["rules"].items() if v["g1_pass_matched"])}
    out["pool"] = pool_stats(mf, pool, d_first, d_last)
    (HERE / "p3_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    register("P3-G1", {"n_configs": len(out["rules"]), "rule": "dev real CI low > 0 and dev high > 0"}, out["g1"])
    print(json.dumps(out["g1"], ensure_ascii=False))


if __name__ == "__main__":
    main()
