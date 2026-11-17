"""持股规则搜索的中文报告：rules/<本金>/select.json（+ 入围规则的 hist 档）→ rules/<本金>/REPORT.md。
用法：python -X utf8 rule_report.py [--capital 200000]
"""
from __future__ import annotations

import argparse
import json

from rule_grid import CADENCE_NAMES, CADENCES, out_dir

PICKER_NAMES = {"smooth": "平滑后超额（登记时定的）", "raw": "不平滑的超额", "ir": "信息比率", "growth": "平滑后复利增长"}
BENCH_NAMES = {"hindsight": "事后看全期最好的一组", "average": "全部组的平均", "reference": "原方案（每年重训 + FH-K6）"}
CONCENTRATED = 3.5  # 平均持股不超过这么多只，算集中持仓


def money(x: float) -> str:
    return f"{x / 10000:,.1f} 万"


def ci(v) -> str:
    return f"[{v[0]:+.1f}, {v[1]:+.1f}]"


def short(name: str) -> str:
    """FH-K6-next-o1o5 → FH-K6-h4，DB-K10-b4-o1o5 → DB-K10-b4。"""
    family, k, mid, cell = name.split("-")
    return f"FH-{k}-h{int(cell[3:]) - 1}" if family == "FH" else f"DB-{k}-{mid}"


def rule_line(r: dict) -> str:
    return f"{CADENCE_NAMES[r['cadence']]}重训 + {short(r['name'])}（{r['plain']}）"


def runs(picks: dict) -> str:
    """逐年选中的规则，连续相同的年份合并，例如"2017–2026：每半年 DB-K30-b8"。"""
    out, prev, first, last = [], None, None, None
    for y, (c, name) in [*picks.items(), (None, (None, None))]:
        if (c, name) != prev:
            if prev is not None:
                out.append(f"{first if first == last else f'{first}–{last}'}：{CADENCE_NAMES[prev[0]]} {short(prev[1])}")
            prev, first = (c, name), y
        last = y
    return "；".join(out)


def table(rows: list[dict], lead: list[str] | None = None, lead_name: str = "") -> list[str]:
    head = f"| {lead_name or '#'} | 重训 | 规则 | 怎么拿 | 超额 [区间] | 信息比率 | 年化 | 最大回撤 | 最差整年超额 | 超额为正 | 每天买 | 拿几天 | 年费用 | 期末 |"
    out = [head, "|" + "---|" * 14]
    for i, r in enumerate(rows):
        first = lead[i] if lead else str(i + 1)
        out.append(f"| {first} | {CADENCE_NAMES[r['cadence']]} | {short(r['name'])} | {r['plain']} | "
                   f"{r['excess']:+.1f} {ci(r['excess_ci'])} | {r['ir']:.2f} | {r['cagr']:+.1f}% | {r['max_dd']:.1f}% | "
                   f"{r['worst_year']:+.1f} | {r['years_positive']} | {r['buys_per_day']:.2f} | {r['hold_days']} | "
                   f"{r['cost_pct']:.1f}% | {money(r['final'])} |")
    return out


def conclusion(sel: dict) -> list[str]:
    best = sel["best_picker"]
    pick, walk, smooth = sel["recommended"][best], sel["walk_forward"][best], sel["recommended"]["smooth"]
    span = f"{min(walk['years'])}–{max(walk['years'])}"
    out = ["## 结论", "",
           f"- 滚动检验（{span} 每年年初只用之前的数据选规则，拼成样本外）里最好的选法是**{PICKER_NAMES[best]}**："
           f"样本外超额 {walk['excess']:+.1f} {ci(walk['ci'])}，年化 {walk['cagr']:+.1f}%，最大回撤 {walk['max_dd']:.1f}%。"
           f"逐年选中的规则：{runs(walk['picks'])}。"]
    if "vs_reference" in walk:
        out.append(f"  - 同一段比原方案多 {walk['vs_reference']['ann']:+.1f} {ci(walk['vs_reference']['ci'])}（逐日配对）。")
    out.append(f"- 用全部数据按这个选法，推荐 **{rule_line(pick)}**：全期超额 {pick['excess']:+.1f} {ci(pick['excess_ci'])}，"
               f"年化 {pick['cagr']:+.1f}%，最大回撤 {pick['max_dd']:.1f}%，本金 {money(sel['capital'])} 变成 {money(pick['final'])}；"
               f"每天买 {pick['buys_per_day']:.2f} 只，每只拿 {pick['hold_days']} 天，"
               f"费用每年约 {pick['cost_pct']:.1f}%。")
    if best != "smooth":
        w0 = sel["walk_forward"]["smooth"]
        out += [f"- 登记时定的选法（平滑后超额）推荐 {rule_line(smooth)}：全期超额 {smooth['excess']:+.1f}，但最大回撤 "
                f"{smooth['max_dd']:.1f}%；它的滚动样本外只有 {w0['excess']:+.1f} {ci(w0['ci'])}，年化 {w0['cagr']:+.1f}%，"
                f"回撤 {w0['max_dd']:.1f}%。不推荐。",
                "  - 选法是看了滚动结果才定的（四选一），样本外数字会偏乐观一点。"]
    crowd = [r for r in sel["table"]["smooth"] if r["avg_holdings"] <= CONCENTRATED]
    if crowd:
        out.append(f"- 按超额排的前 {len(sel['table']['smooth'])} 名里，{len(crowd)} 组平均只拿 ≤ {CONCENTRATED:g} 只，"
                   f"最大回撤 {min(r['max_dd'] for r in crowd):.0f}% ~ {max(r['max_dd'] for r in crowd):.0f}%。"
                   "集中持仓的超额靠少数几只票，哪组领先每年都在换，事先挑不中；分散的组超额低一些但年年稳，事先挑得中。")
    for key in ("mean", "ir"):
        p = sel["pbo"][key]
        out.append(f"- 过拟合概率 PBO（按{'超额' if key == 'mean' else '信息比率'}挑）= {p['pbo']:.2f}：样本内第一的那组，"
                   f"样本内超额 {p['is_best']:+.1f}，到样本外平均 {p['oos_of_is_best']:+.1f}（所有组样本外的中位数 {p['oos_median_all']:+.1f}）。"
                   "PBO 0.5 左右说明挑出来的和随手拿一组差不多，越接近 0 越可信。")
    rz = sel["reality"]["zero"]
    out.append(f"- 多重比较（现实检验）：{rz['n']} 组里最好那组超额 {rz['stat']:+.1f}，p = {rz['p']:.3f}；"
               f"校正后仍显著 > 0 的 {rz['n_above']} 组。")
    if "reference" in sel["reality"]:
        rr = sel["reality"]["reference"]
        out.append(f"  - 跑赢原方案：p = {rr['p']:.3f}，校正后显著跑赢的 {rr['n_above']} 组（门槛 {rr['crit']:+.1f}）。")
    out.append("- 同一段对照：" + "；".join(f"{BENCH_NAMES[k]} 超额 {b['excess']:+.1f}、年化 {b['cagr']:+.1f}%、回撤 {b['max_dd']:.1f}%"
                                       for k, b in sel["benchmarks"].items()) + "。")
    return out + [""]


def pickers(sel: dict) -> list[str]:
    out = ["## 四种选法的对比", "",
           "| 选法 | 全期推荐 | 全期超额 | 全期回撤 | 滚动样本外超额 [区间] | 样本外年化 | 样本外回撤 | 样本外比原方案 |",
           "|---|---|---|---|---|---|---|---|"]
    for p, name in PICKER_NAMES.items():
        r, w = sel["recommended"][p], sel["walk_forward"][p]
        vs = f"{w['vs_reference']['ann']:+.1f} {ci(w['vs_reference']['ci'])}" if "vs_reference" in w else "—"
        out.append(f"| {name} | {CADENCE_NAMES[r['cadence']]} {short(r['name'])} | {r['excess']:+.1f} | {r['max_dd']:.1f}% | "
                   f"{w['excess']:+.1f} {ci(w['ci'])} | {w['cagr']:+.1f}% | {w['max_dd']:.1f}% | {vs} |")
    out += ["", "逐年：每年年初选中的规则，和它这一年的超额。", "",
            "| 年份 | " + " | ".join(n.split("（")[0] for n in PICKER_NAMES.values()) + " |", "|---|" + "---|" * len(PICKER_NAMES)]
    for y in sel["walk_forward"]["smooth"]["years"]:
        cells = []
        for p in PICKER_NAMES:
            w = sel["walk_forward"][p]
            c, n = w["picks"][y]
            cells.append(f"{c} {short(n)} {w['years'][y]:+.1f}")
        out.append(f"| {y} | " + " | ".join(cells) + " |")
    return out + [""]


def cadences(sel: dict) -> list[str]:
    out = ["## 重训频率", "", "每种频率里打分最高的规则（两种选法）：", ""]
    rows, lead = [], []
    for p in ("ir", "smooth"):
        for c, r in sel["by_cadence"][p].items():
            rows.append(r)
            lead.append(PICKER_NAMES[p].split("（")[0])
    out += [*table(rows, lead, "选法"), "", "同一条规则、换重训频率，相对每年重训的全期超额差（所有网格点）：", "",
            "| 频率 | 规则类 | 组数 | 中位数 | 四分位 | 比每年重训好的比例 |", "|---|---|---|---|---|---|"]
    for key, v in sel["cadence_effect"].items():
        c, fam = key.split("|")
        out.append(f"| {CADENCE_NAMES[c]} | {'全部' if fam == 'all' else fam} | {v['n']} | {v['median']:+.1f} | "
                   f"[{v['q25']:+.1f}, {v['q75']:+.1f}] | {v['positive'] * 100:.0f}% |")
    d = sel["detail"][sel["best_picker"]]
    if "vs_yearly" in d:
        out.append(f"\n推荐规则比同一条规则每年重训：逐日配对差 {d['vs_yearly']['ann']:+.1f} {ci(d['vs_yearly']['ci'])}。")
    return out + [""]


def heatmap(sel: dict) -> list[str]:
    c = sel["recommended"][sel["best_picker"]]["cadence"]
    out = [f"## 网格全貌（{CADENCE_NAMES[c]}重训）", ""]
    for fam, axis in (("FH", "h"), ("DB", "b")):
        h = sel["heatmaps"].get(f"{c}|{fam}")
        if h is None:
            continue
        for key, what, fmt in (("excess", "全期年化超额", "{:+.1f}"), ("ir", "信息比率", "{:.2f}")):
            out += [f"{fam} 的{what}（行 = K，列 = {'持有天数 h' if fam == 'FH' else '缓冲倍数 b'}）：", "",
                    f"| K \\ {axis} | " + " | ".join(f"{p:g}" for p in h["p"]) + " |", "|---|" + "---|" * len(h["p"])]
            out += [f"| {k} | " + " | ".join("" if v is None else fmt.format(v) for v in line) + " |"
                    for k, line in zip(h["k"], h[key])]
            out.append("")
    return out


def finalists(sel: dict, hist: dict) -> list[str]:
    known = [*sel["table"]["smooth"], *sel["table"]["ir"], *sel["recommended"].values(), sel["hindsight"],
             *([sel["reference"]] if sel["reference"] else [])]
    rows = {(r["cadence"], r["name"]): r for r in known}
    out = ["## 入围规则：现行印花税 vs 当时的印花税", "",
           "real = 全期按现行 0.05%；hist = 2023-08-28 之前按当时的 0.1%。往后交易用的是 real 的税率。", "",
           "| 重训 | 规则 | real 超额 | real 年化 | real 回撤 | hist 超额 | hist 年化 | hist 回撤 |", "|---|---|---|---|---|---|---|---|"]
    for c, name in sel["finalists"]:
        r, h = rows.get((c, name)), hist.get((c, name))
        if r is None:
            continue
        tail = f"{h['excess']:+.1f} | {h['cagr']:+.1f}% | {h['max_dd']:.1f}%" if h else "（还没算） | | "
        out.append(f"| {CADENCE_NAMES[c]} | {short(name)} | {r['excess']:+.1f} | {r['cagr']:+.1f}% | {r['max_dd']:.1f}% | {tail} |")
    return out + [""]


def detail(sel: dict) -> list[str]:
    best = sel["best_picker"]
    pick, d, ref = sel["recommended"][best], sel["detail"][best], sel["reference"]
    out = [f"## 推荐规则逐年（{rule_line(pick)}）", ""]
    ref_years = sel.get("reference_years") or {}
    out += ["| 年份 | 收益 | 超额 | 原方案收益 | 原方案超额 |", "|---|---|---|---|---|"]
    for y, v in d["years"].items():
        rv = ref_years.get(y)
        tail = f"{rv['total']:+.1f} | {rv['excess']:+.1f}" if rv else " | "
        out.append(f"| {y} | {v['total']:+.1f} | {v['excess']:+.1f} | {tail} |")
    if "vs_reference" in d and ref:
        out.append(f"\n全期比原方案（{rule_line(ref)}，超额 {ref['excess']:+.1f}）：逐日配对差 "
                   f"{d['vs_reference']['ann']:+.1f} {ci(d['vs_reference']['ci'])}。")
    return out + [""]


FRESH_ROWS = 12  # 每种佣金最多列几条（按最差起点的年化排），推荐规则总是列出


def fresh_section(fresh: dict, capital: float, picked: set[str] = frozenset()) -> list[str]:
    """换起点重新起步（fresh_*.json）：同一本金从不同年份起步、拿到最后的年化 / 最大回撤。按最差起点排。"""
    rows: dict[tuple, dict] = {}
    for key, v in fresh.items():
        cap, cadence, name, fee, year = key.split("|")
        if float(cap) == capital:
            rows.setdefault((cadence, name, fee), {})[int(year)] = v
    if not rows:
        return []
    years = sorted({y for v in rows.values() for y in v})
    pool = {y: next(v[y]["pool_cagr"] for v in rows.values() if y in v) for y in years}
    out = ["## 换个年份重新起步", "",
           "上面的全期和滚动数字都来自 2015 年起步的同一个账户：账户先涨大了，5 元最低佣金的占比就小了。"
           "这里同样的本金从不同年份重新起步、一直拿到最后，看小本金是不是被最低佣金拖垮（免五 = 没有最低佣金）。"
           f"按 2017–2023 几个起点里最差的年化排，每种佣金列前 {FRESH_ROWS} 条（推荐规则总是列出）。"
           "这是样本内的体检排名，没有滚动样本外的估计。", "",
           "| 重训 | 规则 | 佣金 | " + " | ".join(f"{y} 年起：年化 / 回撤" for y in years) + " |", "|---|---|---|" + "---|" * len(years)]
    def worst(v: dict) -> float:
        """2017–2023 几个起点里最差的年化（2025 年起步只有不到两年，不算）。"""
        core = [x["cagr"] for y, x in v.items() if 2017 <= y <= 2023]
        return min(core) if core else min(x["cagr"] for x in v.values())

    shown: dict[str, int] = {}
    for (cadence, name, fee), v in sorted(rows.items(), key=lambda kv: (kv[0][2] != "min5", -worst(kv[1]))):
        shown[fee] = shown.get(fee, 0) + 1
        if shown[fee] > FRESH_ROWS and name not in picked:
            continue
        cells = [f"{v[y]['cagr']:+.1f}% / {v[y]['max_dd']:.0f}%" if y in v else "" for y in years]
        out.append(f"| {CADENCE_NAMES[cadence]} | {short(name)} | {'最低 5 元' if fee == 'min5' else '免五'} | " + " | ".join(cells) + " |")
    out += ["| | 准入池等权 | | " + " | ".join(f"{pool[y]:+.1f}%" for y in years) + " |", ""]
    return out


def fresh_warning(fresh: dict, pick: dict, capital: float) -> str | None:
    """推荐规则在最低 5 元下、换个年份重新起步有亏的，结论里提醒。"""
    bad = {int(k.split("|")[4]): v["cagr"] for k, v in fresh.items()
           if k.split("|")[:4] == [str(int(capital)), pick["cadence"], pick["name"], "min5"] and v["cagr"] < 0}
    if not bad:
        return None
    worst = min(bad, key=bad.get)
    return (f"- **注意：推荐规则换个年份重新起步会亏**（最低 5 元佣金下，{'、'.join(str(y) for y in sorted(bad))} 年起步年化为负，"
            f"最差 {worst} 年起 {bad[worst]:+.1f}%）。它的好成绩靠 2015 年起步时账户先涨大、摊薄了最低佣金。见\"换个年份重新起步\"一节。")


def render(sel: dict, hist: dict, fresh: dict | None = None) -> str:
    fresh = fresh or {}
    cads = sel["cadences"]
    head = [f"# 持股规则搜索（本金 {money(sel['capital'])}）", "",
            f"{len(cads)} 种重训频率（{'、'.join(CADENCE_NAMES[c] for c in cads)}）× {sel['n_configs'] // len(cads)} 条规则 = "
            f"{sel['n_configs']} 组；计分 {sel['first']} ~ {sel['last']}。印花税按现行 0.05%（real），佣金万五最低 5 元，一手取整。"
            "超额 = 每天（组合 − 准入池）收益的均值 × 242，区间按月重抽；信息比率 = 超额均值 / 超额波动（年化）。"
            "规则网格和选法见 PREREG.md 补充登记一至三。", ""]
    if set(cads) != set(CADENCES):
        head += [f"**注意：只有{'、'.join(CADENCE_NAMES[c] for c in cads)}重训的结果，其余频率还没跑完，结论是初步的。**", ""]
    check = sel.get("cross_check", {})
    tail = ["## 核对和没修正的", ""]
    if check.get("n"):
        tail.append(f"- 网格里和十年模拟（decade_sim）同名的 {check['n']} 组，real 档数字最大差 {check['max_abs_diff']}（应为 0）。")
    else:
        tail.append("- 十年模拟（decade_sim）只算了 20 万本金，这个本金没有可核对的组。")
    tail += ["- 没修正的：每年换规则的调仓成本（滚动选优里没扣）；净值涨了以后最低佣金的占比变小，期末金额大的规则后期成本偏低估；"
             "没算冲击成本，本金大了以后集中持仓买卖会推动价格。", ""]
    concl = conclusion(sel)
    warn = fresh_warning(fresh, sel["recommended"][sel["best_picker"]], sel["capital"])
    if warn:
        concl.insert(2, warn)
    picked = {r["name"] for r in sel["recommended"].values()}
    body = [*concl, *fresh_section(fresh, sel["capital"], picked), *pickers(sel), "## 前几名（按信息比率）", "",
            *table(sel["table"]["ir"]), "",
            "## 前几名（按平滑后超额）", "", *table(sel["table"]["smooth"]), "",
            *cadences(sel), *heatmap(sel), *finalists(sel, hist), *detail(sel)]
    return "\n".join([*head, *body, *tail])


def load_hist(capital: float) -> dict:
    out = {}
    for c in CADENCES:
        path = out_dir(capital) / f"hist_{c}.json"
        if path.is_file():
            out.update({(c, n): v for n, v in json.loads(path.read_text(encoding="utf-8")).items()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=200_000.0)
    args = ap.parse_args()
    d = out_dir(args.capital)
    sel = json.loads((d / "select.json").read_text(encoding="utf-8"))
    fresh = {}
    for path in sorted(d.parent.parent.glob("fresh_*.json")):  # 本目录（DATA）下 fresh_start.py 的输出
        fresh.update(json.loads(path.read_text(encoding="utf-8")))
    (d / "REPORT.md").write_text(render(sel, load_hist(args.capital), fresh), encoding="utf-8")
    print(f"-> {d / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
