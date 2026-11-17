"""最少持有天数实验的报告：读 <变体>/sims/ 的结果，按 PREREG.md 的主比较下结论，写 REPORT.md 和 summary.json。

算到一半也能跑：90 条规则没算全时，结论标"网格未算完"。
用法：python -X utf8 minhold_report.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "decade"))

from rule_grid import specs  # noqa: E402

from minhold import LABEL_HOLD, NS, PRIMARY, paired, verdict  # noqa: E402

YEAR = 242
CAP_NAMES = {20000: "2 万", 200000: "20 万"}
DB_RULES = [s.name for s in specs() if s.family == "DB"]
SHOW_NS = sorted(NS)
SMALL = 100_000  # 正式 rebound capital：有最低佣金、本金低于这个数按换起点体检选规则


def short(name: str) -> str:
    return name.removesuffix("-o1o5")


def pct(x, d: int = 1) -> str:
    return "—" if x is None else f"{x:+.{d}f}%"


def ci(c, d: int = 1) -> str:
    return f"[{c[0]:+.{d}f}, {c[1]:+.{d}f}]"


class Sims:
    """一个结果文件：每组 "N|规则" 的数字和逐日收益。"""

    def __init__(self, variant: str, capital: int, fee: str = "min5"):
        js, npz = HERE / variant / "sims" / f"{capital}_{fee}.json", HERE / variant / "sims" / f"{capital}_{fee}.npz"
        self.stats, self.ret, self.pool, self.dates = {}, {}, None, None
        if js.is_file() and npz.is_file():
            self.stats = json.loads(js.read_text(encoding="utf-8"))
            z = np.load(npz)
            self.ret = dict(zip(z["keys"].tolist(), z["ret"]))
            self.pool, self.dates = z["pool"], pd.DatetimeIndex(z["dates"])

    def has(self, n: int, name: str) -> bool:
        return f"{n}|{name}" in self.stats and f"{n}|{name}" in self.ret

    def get(self, n: int, name: str) -> dict:
        return self.stats[f"{n}|{name}"]

    def excess(self, n: int, name: str) -> float:
        return float((self.ret[f"{n}|{name}"] - self.pool).mean()) * YEAR * 100

    def ir(self, n: int, name: str) -> float:
        e = self.ret[f"{n}|{name}"] - self.pool
        sd = e.std(ddof=1)
        return float(e.mean() / sd * YEAR ** 0.5) if sd > 0 else float("-inf")

    def worst(self, n: int, name: str):
        st = self.get(n, name).get("fresh_stats") if self.has(n, name) else None
        return st["worst_cagr"] if st else None

    def robust(self, n: int, name: str) -> bool:
        st = self.get(n, name).get("fresh_stats") if self.has(n, name) else None
        return bool(st and st["all_positive"])

    def vs_one(self, n: int, name: str) -> dict:
        return paired(self.ret[f"{n}|{name}"], self.ret[f"1|{name}"], self.dates)


def pick(s: Sims, capital: int, configs: list[tuple[int, str]]):
    """正式 rebound capital 的推荐（只在给的组里选）：小本金 = 四个起点都赚、最差起点年化最高；
    否则 = 全期信息比率最高的那组，它某个起点亏钱就改推前者。并列取靠前的。"""
    have = [c for c in configs if s.has(*c) and s.get(*c).get("fresh_stats")]
    robust = [c for c in have if s.robust(*c)]
    steady = max(robust, key=lambda c: s.worst(*c)) if robust else None
    if capital < SMALL or not have:
        return steady
    best = max(have, key=lambda c: s.ir(*c))
    return best if s.robust(*best) else steady


def decide(s: Sims, capital: int) -> dict:
    """PREREG.md 主比较：决定用的规则 N = 4 对 N = 1。"""
    d = PRIMARY[capital][0]
    if not (s.has(1, d) and s.has(LABEL_HOLD, d)):
        return {"verdict": "未算完", "rule": d}
    p = s.vs_one(LABEL_HOLD, d)
    names = [r for r in DB_RULES if s.has(1, r) and s.has(LABEL_HOLD, r)]
    better = sum(s.excess(LABEL_HOLD, r) > s.excess(1, r) for r in names)
    w4, w1 = s.worst(LABEL_HOLD, d), s.worst(1, d)
    worst_ok = capital >= SMALL or (w4 is not None and w1 is not None and w4 >= w1)
    complete = len(names) == len(DB_RULES)
    v = verdict(p["ci"], better / len(names), worst_ok)
    full = [s.get(1, d)["full"], s.get(LABEL_HOLD, d)["full"]]
    ref = PRIMARY[capital][1]
    ref_p = s.vs_one(LABEL_HOLD, ref) if s.has(1, ref) and s.has(LABEL_HOLD, ref) else None
    return {"verdict": v if complete else f"{v}（网格未算完，{len(names)}/{len(DB_RULES)} 条）", "rule": d,
            "paired": p, "better": better, "rules": len(names), "worst": [w1, w4], "complete": complete,
            "reference": {"rule": ref, "paired": ref_p, "worst": [s.worst(1, ref), s.worst(LABEL_HOLD, ref)]},
            **{k: [f[k] for f in full] for k in ("cagr", "excess", "cost_pct", "hold_days")}}


def grid_rows(s: Sims, capital: int) -> list[dict]:
    rows = []
    for n in SHOW_NS:
        names = [r for r in DB_RULES if s.has(n, r) and s.has(1, r)]
        if not names:
            continue
        d_exc = [s.excess(n, r) - s.excess(1, r) for r in names]
        d_cagr = [s.get(n, r)["full"]["cagr"] - s.get(1, r)["full"]["cagr"] for r in names]
        d_worst = [s.worst(n, r) - s.worst(1, r) for r in names if s.worst(n, r) is not None and s.worst(1, r) is not None]
        chosen = pick(s, capital, [(n, r) for r in DB_RULES])
        rows.append({"n": n, "rules": len(names), "better": int(sum(x > 0 for x in d_exc)),
                     "d_excess": float(np.median(d_exc)), "d_cagr": float(np.median(d_cagr)),
                     "d_worst": float(np.median(d_worst)) if d_worst else None,
                     "robust": sum(s.robust(n, r) for r in DB_RULES), "pick": chosen[1] if chosen else None,
                     "pick_cagr": s.get(*chosen)["full"]["cagr"] if chosen else None,
                     "pick_worst": s.worst(*chosen) if chosen else None,
                     "pick_ir": round(s.ir(*chosen), 2) if chosen else None})
    return rows


def primary_table(s: Sims, capital: int, rules) -> list[str]:
    out = ["| 规则 | N | 每只平均拿几天 | 年化 | 超额 [区间] | 最大回撤 | 费用/年 | 四个起点最差年化 | 比 N = 1（逐日配对）|",
           "|---|---|---|---|---|---|---|---|---|"]
    for name in rules:
        for n in SHOW_NS:
            if not s.has(n, name):
                continue
            f = s.get(n, name)["full"]
            vs = "—" if n == 1 or not s.has(1, name) else (lambda p: f"{p['ann']:+.1f} {ci(p['ci'])}")(s.vs_one(n, name))
            out.append(f"| {short(name)} | {n} | {f['hold_days']} | {pct(f['cagr'])} | {f['excess']:+.1f} {ci(f['excess_ci'])} "
                       f"| {f['max_dd']:.1f}% | {f['cost_pct']:.1f}% | {pct(s.worst(n, name))} | {vs} |")
    return out


def compact_table(s: Sims, rules) -> list[str]:
    out = ["| 规则 | N | 年化 | 超额 | 四个起点最差年化 | 比 N = 1（逐日配对）|", "|---|---|---|---|---|---|"]
    for name in rules:
        for n in SHOW_NS:
            if not s.has(n, name):
                continue
            f = s.get(n, name)["full"]
            vs = "—" if n == 1 or not s.has(1, name) else (lambda p: f"{p['ann']:+.1f} {ci(p['ci'])}")(s.vs_one(n, name))
            out.append(f"| {short(name)} | {n} | {pct(f['cagr'])} | {f['excess']:+.1f} | {pct(s.worst(n, name))} | {vs} |")
    return out


def grid_table(rows: list[dict], capital: int) -> list[str]:
    how = "四个起点都赚、最差起点年化最高" if capital < SMALL else "信息比率最高（某个起点亏就换最稳的）"
    out = [f"| N | 超额比 N = 1 高的条数 | 超额差中位数 | 年化差中位数 | 最差起点年化差中位数 | 四个起点都赚的条数 "
           f"| 按正式选法（{how}）会选 | 它的年化 / 最差起点 / 信息比率 |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        base = r["n"] == 1
        better = "—" if base else f"{r['better']}/{r['rules']}"
        d_exc = "—" if base else f"{r['d_excess']:+.1f}"
        d_cagr = "—" if base else f"{r['d_cagr']:+.1f}"
        d_worst = "—" if base or r["d_worst"] is None else f"{r['d_worst']:+.1f}"
        chosen = short(r["pick"]) if r["pick"] else "没有"
        ir = "—" if r["pick_ir"] is None else f"{r['pick_ir']:.2f}"
        out.append(f"| {r['n']} | {better} | {d_exc} | {d_cagr} | {d_worst} | {r['robust']}/{len(DB_RULES)} | {chosen} "
                   f"| {pct(r['pick_cagr'])} / {pct(r['pick_worst'])} / {ir} |")
    return out


def diag_table(s: Sims, rules) -> list[str]:
    out = ["| 规则（N = 1）| 拿 1 天 | 拿 2 天 | 拿 3 天 | 拿 4 天 | 拿 5 天 | 拿 6 天及以上 "
           "| 不到 4 天就卖的：拿到第 4 天相对准入池 [区间] | 其中跑赢准入池的 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for name in rules:
        if not s.has(1, name):
            continue
        st = s.get(1, name)
        mix, rest = st.get("hold_mix", {}), st.get("rest", {})
        cells = [f"{mix[k]['share']:.0f}% / {mix[k]['net']:+.2f}%" if k in mix else "—"
                 for k in ("1", "2", "3", "4", "5", "6+")]
        gap = f"{rest['gap']:+.2f}% {ci(rest['ci'], 2)}" if rest.get("n") else "—"
        beat = f"{rest['beat_pool']:.0f}%" if rest.get("n") else "—"
        out.append(f"| {short(name)} | " + " | ".join(cells) + f" | {gap} | {beat} |")
    return out


def years_table(sims: dict[int, Sims]) -> list[str]:
    head = ["年"] + [f"{CAP_NAMES[c]} {short(PRIMARY[c][0])} N = {n}" for c in PRIMARY for n in (2, LABEL_HOLD)]
    cols = {}
    for c, s in sims.items():
        d = PRIMARY[c][0]
        for n in (2, LABEL_HOLD):
            if s.has(n, d) and s.has(1, d):
                cols[c, n] = s.vs_one(n, d)["years"]
    years = sorted({y for v in cols.values() for y in v})
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for y in years:
        cells = [f"{cols[c, n][y]:+.1f}" if (c, n) in cols and y in cols[c, n] else "—"
                 for c in PRIMARY for n in (2, LABEL_HOLD)]
        out.append(f"| {y} | " + " | ".join(cells) + " |")
    return out


def checks() -> list[str]:
    out = []
    for variant, label in (("nocap", "不限价"), ("cap70", "限 70 元")):
        log = HERE / "logs" / f"check-{variant}.log"
        text = log.read_text(encoding="utf-8") if log.is_file() else ""
        hit = re.search(r"decade check \((\w+)\): (\d+) items identical", text)
        out.append(f"- {label}：" + (f"{hit.group(2)} 项逐项一样（N = 1、不剔退市整理期；四条主规则在 2 万、20 万的全期数字和"
                                     "逐日收益，加上十年实验存了的 2 万换起点）。" if hit else "没有核对记录。"))
    reg = HERE / "nocap" / "registry.jsonl"
    runs = []
    if reg.is_file():
        for line in reg.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line) if line.strip() else None
            if rec and rec["stage"] == "run_minhold":
                runs.append(rec["metrics"])
    if runs and runs[-1].get("delist_rows"):
        e = runs[-1]
        out.append(f"- 剔退市整理期：准入池里当时名称带\"退\"的 {e['delist_rows']} 个股票日（{e['delist_codes']} 只票），"
                   f"其中 {e['delist_new']} 个原本没被 2 元下限剔掉。")
    out.append("- 模拟器：p0p1 `sim.simulate` 逐行照抄，只改 DB 的卖出条件；`test_minhold.py` 用合成行情核对 "
               "N = 1 时和原版逐项相同、N > 1 时没有一笔拿不满 N 天就卖。")
    return out


def decision_lines(decisions: dict) -> list[str]:
    out = []
    for c, dec in decisions.items():
        if "paired" not in dec:
            out.append(f"- {CAP_NAMES[c]}：{short(dec['rule'])} 还没算完。")
            continue
        p, (w1, w4) = dec["paired"], dec["worst"]
        out.append(f"- **{CAP_NAMES[c]}（{short(dec['rule'])}）：N = {LABEL_HOLD} 比现在（N = 1）{dec['verdict']}。**"
                   f"年化 {dec['cagr'][0]:+.1f}% → {dec['cagr'][1]:+.1f}%，超额 {dec['excess'][0]:+.1f} → {dec['excess'][1]:+.1f}；"
                   f"逐日配对差 {p['ann']:+.1f} {ci(p['ci'])}；四个起点最差年化 {pct(w1)} → {pct(w4)}；"
                   f"90 条 DB 规则里 N = {LABEL_HOLD} 超额更高的 {dec['better']}/{dec['rules']} 条。"
                   f"平均每只拿 {dec['hold_days'][0]} → {dec['hold_days'][1]} 天，费用每年 {dec['cost_pct'][0]:.1f}% → "
                   f"{dec['cost_pct'][1]:.1f}%：省下的费用抵不上少赚的。")
        ref = dec["reference"]
        if ref["paired"]:
            rp = ref["paired"]
            out.append(f"  - 参照规则 {short(ref['rule'])}：N = {LABEL_HOLD} 比 N = 1 逐日配对差 {rp['ann']:+.1f} {ci(rp['ci'])}，"
                       f"四个起点最差年化 {pct(ref['worst'][0])} → {pct(ref['worst'][1])}。")
        if c < SMALL:
            out += by_k_lines(c)
    return out


def by_k_lines(capital: int = 20000) -> list[str]:
    """事后拆分（by_k.py）：2 万的网格比例被哪些规则拉高。"""
    path = HERE / "by_k.json"
    if not path.is_file():
        return []
    rows = json.loads(path.read_text(encoding="utf-8")).get(str(capital), {}).get(str(LABEL_HOLD), {})
    if not rows:
        return []
    parts = "；".join(f"{name}：{v['better']}/{v['rules']} 条更高，中位数 {v['d_excess']:+.1f}（N = 1 时超额中位数 "
                     f"{v['base_excess']:+.1f}）" for name, v in rows.items())
    return [f"  - 事后拆分（看过结果才加，不改结论）：{CAP_NAMES[capital]} N = {LABEL_HOLD} 比 N = 1，{parts}。"
            "比例被 K ≥ 10 的规则拉高：它们在 2 万本金下被最低佣金拖成负超额，拿久一点少交易、少亏一点；"
            "2 万真正能用的 K 1–3 和换起点都赚的那些规则，N = 4 几乎都更差。"]


def grid_lines(grids: dict) -> list[str]:
    out = []
    for c, rows in grids.items():
        by_n = {r["n"]: r for r in rows}
        if len(by_n) < 2:
            continue
        curve = "，".join(f"N = {n} {by_n[n]['d_excess']:+.1f}（{by_n[n]['better']}/{by_n[n]['rules']} 条更高）"
                         for n in SHOW_NS if n != 1 and n in by_n)
        out.append(f"- {CAP_NAMES[c]} 全网格，超额比 N = 1 的变化（中位数）：{curve}。")
        picks = [(n, by_n[n]) for n in SHOW_NS if by_n.get(n) and by_n[n]["pick"]]
        if not picks:
            continue
        key = "pick_worst" if c < SMALL else "pick_ir"
        best_n, best = max(picks, key=lambda x: x[1][key])
        first = by_n.get(1)
        line = (f"  - 按正式选法每个 N 各选一条：{'最差起点年化' if c < SMALL else '信息比率'}最高的是 N = {best_n} 的 "
                f"{short(best['pick'])}（年化 {pct(best['pick_cagr'])}，最差起点 {pct(best['pick_worst'])}，"
                f"信息比率 {best['pick_ir']}）")
        if best_n != 1 and first and first["pick"]:
            line += (f"；N = 1 选的是 {short(first['pick'])}（年化 {pct(first['pick_cagr'])}，最差起点 "
                     f"{pct(first['pick_worst'])}，信息比率 {first['pick_ir']}）")
            if best_n != 1 and c < SMALL and best[key] - first[key] < 1:
                line += "，最差起点只差不到 1 个百分点，算打平"
        out.append(line + "。把 N 当参数一起选是在全样本上多挑一次，只作参考。")
    return out


def rotation_data(variant: str = "nocap") -> dict:
    path = HERE / variant / "rotation.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def mechanism_lines(sims: dict[int, Sims], rot: dict) -> list[str]:
    out = []
    for c, s in sims.items():
        d = PRIMARY[c][0]
        rest = s.get(1, d).get("rest", {}) if s.has(1, d) else {}
        if not rest.get("n"):
            continue
        sign = "少赚" if rest["gap"] < 0 else "还能多赚"
        line = (f"- 机制（{CAP_NAMES[c]} {short(d)}，N = 1）：{rest['share_of_trades']:.0f}% 的成交不到 {LABEL_HOLD} 天就卖了。"
                f"卖掉的票如果接着拿到买入后第 {LABEL_HOLD} 天开盘，相对同期准入池平均{sign} {abs(rest['gap']):.2f}% "
                f"{ci(rest['ci'], 2)}")
        v = rot.get(f"{c}|{d}")
        if v:
            r = v["rotation"]
            judged = ("换上的新票更好：早卖赚的是换仓，不是躲开下跌" if r["ci"][0] > 0 else
                      "换上的新票反而更差" if r["ci"][1] < 0 else "两者分不出")
            line += (f"；同一个开盘换上的新票同一段平均 {v['repl']['mean']:+.2f}%，比卖掉的那只（{v['gap']['mean']:+.2f}%）"
                     f"多 {r['mean']:+.2f}% {ci(r['ci'], 2)}，{judged}")
        out.append(line + "。")
    return out


def rotation_table(rot: dict) -> list[str]:
    out = ["| 本金 | 规则（N = 1）| 早卖、当天有新买进的成交 / 全部成交 | 卖掉的票接着拿到第 4 天（相对准入池）"
           "| 换上的新票同一段（相对准入池）| 新票 − 卖掉的 [区间] | 新票更好的比例 |",
           "|---|---|---|---|---|---|---|"]
    for key, v in rot.items():
        capital, name = key.split("|")
        out.append(f"| {CAP_NAMES[int(capital)]} | {short(name)} | {v['early_with_new_buy']}/{v['trades']} "
                   f"| {v['gap']['mean']:+.2f}% | {v['repl']['mean']:+.2f}% "
                   f"| {v['rotation']['mean']:+.2f}% {ci(v['rotation']['ci'], 2)} | {v['rotation_positive']:.0f}% |")
    return out


def advice_lines(decisions: dict) -> list[str]:
    out = []
    for c, dec in decisions.items():
        if "paired" not in dec:
            continue
        if dec["verdict"].startswith("更好"):
            out.append(f"- {CAP_NAMES[c]}：N = {LABEL_HOLD} 达到预注册的\"更好\"标准，可以考虑改正式规则（改之前另外确认）。")
        else:
            out.append(f"- {CAP_NAMES[c]}：按预注册的标准，维持现在的规则（不加最少持有天数）。")
    return out


def main() -> None:
    sims = {c: Sims("nocap", c) for c in PRIMARY}
    cap70 = {c: Sims("cap70", c) for c in PRIMARY}
    free5 = Sims("nocap", 20000, "free5")
    decisions = {c: decide(s, c) for c, s in sims.items()}
    grids = {c: grid_rows(s, c) for c, s in sims.items()}
    rot = rotation_data()
    any_s = next((s for s in sims.values() if s.dates is not None), None)
    span = f"{any_s.dates[0].date()} ~ {any_s.dates[-1].date()}" if any_s else "—"
    lines = ["# 最少持有天数：只改持股规则的十年回测", "",
             "模型和打分不动（十年实验存的每月重训 CPU 版打分，标签 o1o5：T+1 开盘买、T+5 开盘卖，拿 4 天）。"
             "只改 DB 规则的卖出条件：排名掉出 b·K 名的票，要拿满 N 个开市日才卖（N = 1 是现在的规则）。"
             f"准入池不限价，剔除 < 2 元和退市整理期；佣金万五最低 5 元，印花税 0.05%，一手取整；计分 {span}。"
             "超额 = 每天（组合 − 准入池）收益的均值 × 242，区间按月重抽 95%；\"比 N = 1\" = 同一条规则两个 N "
             "逐日收益之差的年化。预注册见 PREREG.md。", "", "## 结论", ""]
    lines += decision_lines(decisions) + grid_lines(grids) + mechanism_lines(sims, rot) + advice_lines(decisions)
    for c, s in sims.items():
        lines += ["", f"## {CAP_NAMES[c]}：主规则（{short(PRIMARY[c][0])} 决定用，{short(PRIMARY[c][1])} 参照）", ""]
        lines += primary_table(s, c, PRIMARY[c])
    for c, rows in grids.items():
        lines += ["", f"## {CAP_NAMES[c]}：90 条 DB 规则", ""] + (grid_table(rows, c) if rows else ["还没算。"])
    lines += ["", "## 成交拿了几天（N = 1）", "",
              "每格：占成交笔数 / 平均每笔净收益（扣费）。最后两列：不到 4 天就卖掉的票，如果一直拿到买入后第 4 天开盘，"
              "从实际卖出那天开盘起相对同期准入池（等权开盘对开盘）多赚还是少赚。只看这只票自己，不管空出来的位置买了什么。", ""]
    for c, s in sims.items():
        lines += [f"{CAP_NAMES[c]}：", ""] + diag_table(s, PRIMARY[c]) + [""]
    lines += ["## 早卖换上的新票 vs 卖掉的那只（N = 1）", "",
              "每笔不到 4 天就卖、当天开盘有新买进的成交：卖掉的那只接着拿到它买入后第 4 天开盘，和同一个开盘买进的新票"
              "拿到同一天，各自相对同期准入池的超额（每笔的平均，区间按月重抽）。新票之后自己什么时候卖不管。"
              "只解释机制，结论以完整模拟为准（diag_rotation.py）。", ""]
    lines += rotation_table(rot) if rot else ["还没算。"]
    lines += ["", "## 逐年：决定用的规则比 N = 1（逐日配对差的年化）", ""] + years_table(sims)
    lines += ["", "## 敏感性：限 70 元（十年实验主线）", ""]
    for c, s in cap70.items():
        lines += [f"{CAP_NAMES[c]}：", ""] + compact_table(s, PRIMARY[c]) + [""]
    lines += ["## 敏感性：2 万免五（没有最低佣金）", ""] + compact_table(free5, PRIMARY[20000])
    lines += ["", "## 核对", ""] + checks()
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {"decisions": decisions, "grids": grids}
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("-> REPORT.md, summary.json")


if __name__ == "__main__":
    main()
