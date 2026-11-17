"""2 万本金：能执行的最好（尾盘卖、前一晚定，DB-K2-b6）对理想情况的最好（开盘先卖后买、回测价，DB-K2-b8），
两边都是按正式小本金选法挑出的那条。读两边的 detail_*.json（detail_rule.py），并列逐年、逐月、换起点；
另报同一条规则换时点、全期年化最高的那条、9:30 贵 0.1% 的数字（best_finals / results / grid）。
用法：python -X utf8 compare_ideal.py   输出 compare_ideal.json、compare_monthly.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SIDES = {"exec": "detail_DB-K2-b6_close_sell_prev_rank.json", "ideal": "detail_DB-K2-b8_now.json"}


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{x:.1f}".replace("-", "−")


def main() -> None:
    a, b = (load(f) for f in SIDES.values())
    years = []
    for ra, rb in zip(a["yearly"], b["yearly"]):
        assert ra["key"] == rb["key"]
        years.append({"year": ra["key"][0], "exec_ret": ra["ret"] * 100, "exec_nav": ra["nav"],
                      "ideal_ret": rb["ret"] * 100, "ideal_nav": rb["nav"], "gap": (ra["ret"] - rb["ret"]) * 100,
                      "exec_dd": ra["dd"] * 100, "ideal_dd": rb["dd"] * 100, "pool": ra["pool"] * 100})
    months = pd.DataFrame([{"年": ra["key"][0], "月": ra["key"][1], "能执行%": round(ra["ret"] * 100, 2),
                            "理想%": round(rb["ret"] * 100, 2), "差%": round((ra["ret"] - rb["ret"]) * 100, 2)}
                           for ra, rb in zip(a["monthly"], b["monthly"])])
    months.to_csv(HERE / "compare_monthly.csv", index=False, encoding="utf-8-sig")
    stat = {}
    for key, d in (("exec", a), ("ideal", b)):
        mr = pd.Series([r["ret"] for r in d["monthly"]]) * 100
        stat[key] = {"final": d["row"]["final"], "cagr": d["row"]["cagr"], "max_dd": d["row"]["max_dd"],
                     "worst": d["row"]["worst"], "fresh": {y: v["cagr"] for y, v in d["fresh"].items()},
                     "fresh_end": {y: v["year_end"]["2026"] for y, v in d["fresh"].items()},
                     "drawdown": d["drawdown"], "months_up": round(float((mr > 0).mean()) * 100, 1),
                     "best_month": round(float(mr.max()), 1), "worst_month": round(float(mr.min()), 1),
                     "buys": d["buys"], "fees": round(sum(r["fees"] for r in d["yearly"])),
                     "hold_days": d["row"]["hold_days"], "turnover": d["row"]["turnover"]}
    grid = json.loads((HERE / "grid_20000.json").read_text(encoding="utf-8"))
    extra = {t: {f: grid[t][f] for f in ("cagr", "max_dd", "worst", "fresh")}
             for t in ("now|DB-K2-b6", "close_sell_prev_rank|DB-K2-b8", "now|DB-K3-b4", "close_sell_prev_rank|DB-K2-b4",
                       "now+0.2%|DB-K2-b8")}
    finals = load("best_finals.json")
    res = load("results.json")["rules"]["20000|DB-K2-b8"]["slip"]["0.001"]
    extra["now+0.1%|DB-K2-b8"] = {f: res[f] for f in ("cagr", "max_dd", "worst", "fresh", "final")}
    out = {"stat": stat, "years": years, "extra": extra,
           "finals": {k: finals[k] for k in ("now|DB-K3-b4", "close_sell_prev_rank|DB-K2-b4")},
           "worst_gap_months": months.nsmallest(6, "差%").to_dict("records"),
           "best_gap_months": months.nlargest(4, "差%").to_dict("records")}
    (HERE / "compare_ideal.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("| 年 | 能执行 收益 | 能执行 年末 | 理想 收益 | 理想 年末 | 差（百分点） |")
    for r in years:
        lab = f"{r['year']}（到 10-30）" if r["year"] == 2026 else str(r["year"])
        print(f"| {lab} | {pct(r['exec_ret'])}% | {r['exec_nav']:,.0f} | {pct(r['ideal_ret'])}% | {r['ideal_nav']:,.0f} | "
              f"{pct(r['gap'])} |")
    print(json.dumps(stat, ensure_ascii=False, indent=1))
    print(json.dumps(extra, ensure_ascii=False))
    print(json.dumps(out["finals"], ensure_ascii=False))
    print(months.nsmallest(6, "差%").to_string(index=False))
    print(months.nlargest(4, "差%").to_string(index=False))
    corr = months[["能执行%", "理想%"]].corr().iloc[0, 1]
    print("月收益相关系数", round(float(corr), 3), " 差为正的月份", round(float((months["差%"] > 0).mean()) * 100, 1), "%")


if __name__ == "__main__":
    main()
