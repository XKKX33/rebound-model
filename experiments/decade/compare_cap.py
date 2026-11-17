"""限 70 元 vs 不限价：同一组规则逐日配对比账户收益（不比超额，因为两版的准入池基准不一样）。
读 rules/<本金>/（限价）和 nocap/rules/<本金>/（不限价）的 real 档，输出 nocap/cap_compare_<本金>.{json,md}。
用法：python -X utf8 compare_cap.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import register
from engine import month_ci
from rule_grid import CADENCE_NAMES, CADENCES, REFERENCE
from rule_report import ci, short
from sim import YEAR

HERE = Path(__file__).resolve().parent
NOCAP = HERE / "nocap"
VERSIONS = {"capped": HERE, "nocap": NOCAP}


def load(base: Path, capital: int) -> tuple[dict, np.ndarray, pd.DatetimeIndex, dict]:
    """一个版本：{(频率, 规则名): 逐日收益}、准入池逐日收益、计分日、select.json。"""
    books, pool, dates = {}, None, None
    for c in CADENCES:
        path = base / "rules" / str(capital) / f"real_{c}.npz"
        if path.is_file():
            z = np.load(path)
            pool, dates = z["pool"], pd.DatetimeIndex(z["dates"])
            books.update({(c, str(n)): r for n, r in zip(z["names"], z["ret"])})
    sel = base / "rules" / str(capital) / "select.json"
    return books, pool, dates, json.loads(sel.read_text(encoding="utf-8")) if sel.is_file() else {}


def cagr(r: np.ndarray) -> float:
    return round((float(np.prod(1 + r)) ** (YEAR / len(r)) - 1) * 100, 2)


def pair(key, a: dict, b: dict, dates: pd.DatetimeIndex) -> dict | None:
    """同一组规则：两版各自的年化收益，以及逐日收益差（不限价 − 限价）的年化 [按月重抽区间]。"""
    if key not in a or key not in b:
        return None
    d = b[key] - a[key]
    lo, hi = month_ci(pd.Series(d, index=dates))
    return {"cadence": key[0], "name": key[1], "capped_cagr": cagr(a[key]), "nocap_cagr": cagr(b[key]),
            "diff": round(float(d.mean()) * YEAR * 100, 2), "ci": [round(lo * YEAR * 100, 2), round(hi * YEAR * 100, 2)]}


def compare(capital: int) -> dict | None:
    (a, pool_a, dates_a, sel_a), (b, pool_b, dates_b, sel_b) = (load(v, capital) for v in VERSIONS.values())
    if not a or not b:
        return None
    if not dates_a.equals(dates_b):
        raise SystemExit("两版的计分日不一样")
    common = sorted(set(a) & set(b), key=lambda k: (CADENCES.index(k[0]), k[1]))
    by_cadence = {}
    for c in CADENCES:
        keys = [k for k in common if k[0] == c]
        if keys:
            g = np.array([cagr(b[k]) - cagr(a[k]) for k in keys])
            d = np.array([(b[k] - a[k]).mean() * YEAR * 100 for k in keys])
            by_cadence[c] = {"n": len(keys), "nocap_better": round(float((g > 0).mean()), 3),
                             "median_cagr_gap": round(float(np.median(g)), 2), "median_daily_diff": round(float(np.median(d)), 2)}
    picks = {}
    for version, sel in (("capped", sel_a), ("nocap", sel_b)):
        if sel:
            best = sel["best_picker"]
            r = sel["recommended"][best]
            picks[version] = {"picker": best, "rule": [r["cadence"], r["name"]], "walk": sel["walk_forward"][best]}
    keys = {tuple(p["rule"]) for p in picks.values()} | {REFERENCE}
    return {"capital": capital, "pool_cagr": {"capped": cagr(pool_a), "nocap": cagr(pool_b)},
            "cadences": {"capped": sorted({k[0] for k in a}, key=CADENCES.index), "nocap": sorted({k[0] for k in b}, key=CADENCES.index)},
            "by_cadence": by_cadence, "picks": picks,
            "pairs": [p for p in (pair(k, a, b, dates_a) for k in sorted(keys)) if p is not None]}


def render(cmp: dict) -> str:
    names = {"capped": "限 70 元", "nocap": "不限价"}
    out = [f"# 限 70 元 vs 不限价（本金 {cmp['capital'] / 10000:g} 万）", "",
           f"- 两版算了的重训频率：限 70 元 {'、'.join(CADENCE_NAMES[c] for c in cmp['cadences']['capped'])}；"
           f"不限价 {'、'.join(CADENCE_NAMES[c] for c in cmp['cadences']['nocap'])}。",
           f"- 准入池等权的年化收益：限 70 元 {cmp['pool_cagr']['capped']:+.1f}%，不限价 {cmp['pool_cagr']['nocap']:+.1f}%。"
           "两版的超额是各比各的池子，所以下面只比账户收益。", ""]
    if cmp["picks"]:
        out += ["## 各自滚动检验最好的选法和推荐规则", "",
                "| 版本 | 选法 | 推荐规则 | 滚动样本外超额 [区间] | 样本外年化 | 样本外回撤 |", "|---|---|---|---|---|---|"]
        for v, p in cmp["picks"].items():
            w = p["walk"]
            out.append(f"| {names[v]} | {p['picker']} | {CADENCE_NAMES[p['rule'][0]]} {short(p['rule'][1])} | "
                       f"{w['excess']:+.1f} {ci(w['ci'])} | {w['cagr']:+.1f}% | {w['max_dd']:.1f}% |")
        out.append("")
    out += ["## 同一组规则，两版的账户收益", "", "所有规则一起看：不限价的年化收益比限 70 元高的比例、差的中位数。", "",
            "| 重训 | 组数 | 不限价更好的比例 | 年化收益差中位数 | 逐日收益差中位数（年化） |", "|---|---|---|---|---|"]
    out += [f"| {CADENCE_NAMES[c]} | {v['n']} | {v['nocap_better'] * 100:.0f}% | {v['median_cagr_gap']:+.1f} | {v['median_daily_diff']:+.1f} |"
            for c, v in cmp["by_cadence"].items()]
    out += ["", "几条关键规则（两版的推荐、原方案），逐日配对：", "",
            "| 重训 | 规则 | 限 70 元年化 | 不限价年化 | 逐日收益差（年化）[区间] |", "|---|---|---|---|---|"]
    out += [f"| {CADENCE_NAMES[p['cadence']]} | {short(p['name'])} | {p['capped_cagr']:+.1f}% | {p['nocap_cagr']:+.1f}% | "
            f"{p['diff']:+.1f} {ci(p['ci'])} |" for p in cmp["pairs"]]
    out += ["", "两版的规则网格、成本、选法都一样，只差准入池（和训练用的池子）里有没有 70 元以上的票；< 2 元两版都剔除。", ""]
    return "\n".join(out)


def main() -> None:
    done = {}
    for capital in (200_000, 20_000):
        cmp = compare(capital)
        if cmp is None:
            print(f"{capital}: 有一版还没结果，跳过", flush=True)
            continue
        (NOCAP / f"cap_compare_{capital}.json").write_text(json.dumps(cmp, ensure_ascii=False, indent=1), encoding="utf-8")
        (NOCAP / f"cap_compare_{capital}.md").write_text(render(cmp), encoding="utf-8")
        done[capital] = {c: v["median_cagr_gap"] for c, v in cmp["by_cadence"].items()}
        print(f"{capital}: -> {NOCAP / f'cap_compare_{capital}.md'}", flush=True)
    register("compare_cap", {"versions": list(VERSIONS)}, done)


if __name__ == "__main__":
    main()
