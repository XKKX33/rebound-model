"""补充登记八的报告：win/window_eval_y.json、win/window_eval_m.json（没有就是每月版还没跑）、两版同频率的全网格
→ win/REPORT.md 和 win/window_compare.json。判定按补充登记八机械地算。
用法：python -X utf8 window_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rule_grid import CADENCE_NAMES
from theme_eval import CASES, PRIMARY, TERCILES
from theme_report import ci, grid
from window_eval import OUT, VARIANTS

HERE = Path(__file__).resolve().parent
CAPITALS = (200000, 20000)
KINDS = {"Y vs V0-M": "每年重训的窗口版 − 每月重训的现在模型", "Y": "每年重训：窗口版 − 现在的模型",
         "M": "每月重训：窗口版 − 现在的模型"}


def comp(ev: dict, case: str, window: str, kind: str) -> dict | None:
    return next((c for c in ev["comparisons"] if c["case"] == case and c["window"] == window and c["kind"] == kind), None)


def share(grids: dict, window: str, cad: str) -> float:
    """同频率全网格（两个本金合起来）里窗口版账户收益更高的比例（%）。"""
    got = [g["by_cadence"][cad] for (w, _), g in grids.items() if w == window and g and cad in g["by_cadence"]]
    return 100 * sum(v["better"] for v in got) / sum(v["n"] for v in got) if got else float("nan")


def verdict(c: dict | None, grid_share: float) -> str:
    if c is None:
        return "还没跑"
    p = c["2017起"]
    if p["ci"][0] > 0:
        return "确有改进"
    if p["diff"] > 0 and grid_share >= 60:
        return "可能有改进"
    return "没有改进"


def user_verdict(c: dict | None) -> str:
    if c is None:
        return "还没跑"
    lo, hi = c["2017起"]["ci"]
    return "更好" if lo > 0 else ("更差" if hi < 0 else "分不出")


def ir_pick(base: Path, capital: int) -> dict | None:
    """一版的滚动选法，固定用信息比率（和现在的推荐同一种选法）。"""
    path = base / "rules" / str(capital) / "select.json"
    if not path.is_file():
        return None
    s = json.loads(path.read_text(encoding="utf-8"))
    w = s["walk_forward"]["ir"]
    return {"cadences": s["cadences"], **{k: w[k] for k in ("excess", "ci", "cagr", "max_dd")}}


def table(ev: dict, kind: str) -> list[str]:
    out = ["| 规则 | 窗口 | 2017 起 | 全期 | 2026 | " + " | ".join(TERCILES) + " |", "|---" * (5 + len(TERCILES)) + "|"]
    for case in CASES:
        for w in VARIANTS:
            c = comp(ev, case, w, kind)
            if c is None:
                continue
            out.append(f"| {case} | {w} | {c['2017起']['diff']:+.1f} {ci(c['2017起']['ci'])} | {c['全期']['diff']:+.1f} {ci(c['全期']['ci'])} | "
                       f"{c['2026']['diff']:+.1f} | " + " | ".join(f"{c['regime_gap_diff'][t]:+.2f}" for t in TERCILES) + " |")
    return out


def years(ev: dict) -> list[str]:
    rows = [(kind, w, comp(ev, PRIMARY, w, kind)) for kind in KINDS for w in VARIANTS]
    rows = [(kind, w, c) for kind, w, c in rows if c is not None]
    ys = sorted(rows[0][2]["years"]) if rows else []
    out = ["| 比什么 | 窗口 | " + " | ".join(ys) + " |", "|---" * (2 + len(ys)) + "|"]
    return out + [f"| {KINDS[kind]} | {w} | " + " | ".join(f"{c['years'][y]:+.1f}" for y in ys) + " |" for kind, w, c in rows]


def books(ev: dict) -> list[str]:
    out = ["| 规则 | 版本 | 全期年化 | 2017 起年化 | 2017 起超额 | 最大回撤 |", "|---" * 6 + "|"]
    for case in CASES:
        for k, v in sorted(ev["jobs"].items()):
            src, cad, name = k.split("|")
            if name == case:
                a, s = v["all"], v["since2017"]
                out.append(f"| {case} | {src}（{CADENCE_NAMES[cad]}重训） | {a['cagr']:+.1f}% | {s['cagr']:+.1f}% | {s['excess']:+.1f} | {a['max_dd']:.1f}% |")
    return out


def ic_lines(ev: dict) -> list[str]:
    out = ["| 窗口 | 重训 | 秩相关 现在 → 窗口版（×100） | 差 (t) | 前 30 名平均名次 现在 → 窗口版 |", "|---" * 5 + "|"]
    for k, t in ev["ic"].items():
        w, cad = k.split("|")
        mn = t["mean"]
        out.append(f"| {w} | {CADENCE_NAMES[cad]} | {mn['ic_V0']:.2f} → {mn['ic_V1']:.2f} | {t['ic_diff']['mean']:+.2f} ({t['ic_diff']['t']}) | "
                   f"{mn['top30_V0']:.1f} → {mn['top30_V1']:.1f} |")
    return out


def grid_lines(grids: dict, picks: dict) -> list[str]:
    out = []
    for (w, capital), g in grids.items():
        if not g:
            continue
        by = "，".join(f"{CADENCE_NAMES[c]} {v['better']}/{v['n']}（中位 {v['median_gap']:+.2f}）" for c, v in g["by_cadence"].items())
        out.append(f"- {w}，{capital // 10000} 万：同一规则窗口版账户收益更高的 {by}")
    for (w, capital), (old, new) in picks.items():
        if old and new:
            out.append(f"- {w}，{capital // 10000} 万，滚动选法（信息比率，{'/'.join(new['cadences'])}）：窗口版 {new['excess']:+.1f} {ci(new['ci'])}，"
                       f"年化 {new['cagr']:+.1f}%，回撤 {new['max_dd']:.1f}%；现在的模型（{'/'.join(old['cadences'])}）{old['excess']:+.1f} {ci(old['ci'])}，"
                       f"年化 {old['cagr']:+.1f}%，回撤 {old['max_dd']:.1f}%")
    return out


def recommend(m: dict | None, verdicts: dict) -> str:
    """补充登记八里用户定的：只有每月重训那一档判成"确有改进"才建议换窗口，否则保留全历史。"""
    if m is None:
        return "每月重训还没跑完，先不下结论"
    good = [w for w in VARIANTS if verdicts[w]["M"] == "确有改进"]
    return f"建议改用 {'、'.join(good)} 的窗口（每月重训）" if good else "保留全历史（每月重训判不出改进，按事先说好的不换）"


def render(y: dict, m: dict | None, grids: dict, picks: dict) -> tuple[str, dict]:
    ev = m or y
    verdicts = {}
    for w in VARIANTS:
        verdicts[w] = {"Y": verdict(comp(ev, PRIMARY, w, "Y"), share(grids, w, "Y")),
                       "M": verdict(comp(ev, PRIMARY, w, "M"), share(grids, w, "M")) if m else "还没跑",
                       "user": user_verdict(comp(ev, PRIMARY, w, "Y vs V0-M")),
                       "share_Y": share(grids, w, "Y"), "share_M": share(grids, w, "M")}
    rec = recommend(m, verdicts)
    lines = ["# 滚动窗口（补充登记八）", "",
             f"版本：{'每月重训终版' if m else '每年重训的中间结果（每月重训还没跑完）'}；真实账户、现行印花税。"
             "W2 / W3 = 每次重训只用之前 2 / 3 年的数据；V0 = 现在的模型（2013 年起全部历史）。差都是年化百分点，主指标从 2017-01 起算。", "",
             "## 结论", "", f"- **{rec}。**"]
    for w in VARIANTS:
        v = verdicts[w]
        cu, cy, cm = (comp(ev, PRIMARY, w, k) for k in ("Y vs V0-M", "Y", "M"))
        if cu:
            lines.append(f"- **{w}（只用近 {w[1:]} 年）**：")
            lines.append(f"  - 你的设计（每年重训）vs 现在的推荐（每月重训、全历史）：{v['user']}，主规则 2017 起 "
                         f"{cu['2017起']['diff']:+.1f} {ci(cu['2017起']['ci'])}。")
        if cy:
            lines.append(f"  - 每年重训时窗口的作用：{v['Y']}，{cy['2017起']['diff']:+.1f} {ci(cy['2017起']['ci'])}，"
                         f"同频率全网格里更好的 {v['share_Y']:.0f}%。" if not np.isnan(v["share_Y"]) else
                         f"  - 每年重训时窗口的作用：{v['Y']}，{cy['2017起']['diff']:+.1f} {ci(cy['2017起']['ci'])}（全网格还没跑）。")
        if cm:
            lines.append(f"  - 每月重训时窗口的作用：{v['M']}，{cm['2017起']['diff']:+.1f} {ci(cm['2017起']['ci'])}，"
                         f"同频率全网格里更好的 {v['share_M']:.0f}%。" if not np.isnan(v["share_M"]) else
                         f"  - 每月重训时窗口的作用：{v['M']}，{cm['2017起']['diff']:+.1f} {ci(cm['2017起']['ci'])}（全网格还没跑）。")
    lines += [""]
    for kind, title in KINDS.items():
        rows = table(ev, kind)
        if len(rows) > 2:
            lines += [f"## {title}", "", "区间是按月整块重抽的 95% 区间；后三列是每月比准入池多赚多少的变化（百分点）。", "", *rows, ""]
    lines += ["## 主规则（20 万 DB-K30-b8）逐年", "", *years(ev), "", "## 打分（2017 起，同一批行）", "", *ic_lines(ev), "",
              "## 各版本的数字", "", *books(ev), "", "## 全网格（2 个本金 × 150 条规则，同频率比）", "", *grid_lines(grids, picks), "",
              "## 要说明的", "",
              "- 准入池 2013 年 8 月才有数据，2016 年以前 W2、2017 年以前 W3 的训练行和全历史几乎一样，所以主指标从 2017 起算。",
              "- 只改了训练窗口：特征、参数、种子、标签、剔除、准入池都和现在的模型一样。没测换种子会差多少，一两个点的差可能来自训练本身的随机性。",
              "- 三套固定规则是在现在模型上挑出来的，对现在的模型略有利；全网格和滚动选法是为了补这一点。",
              "- 两种窗口、两种重训频率、三套规则一起看，有多重比较：单个区间刚过 0 的要打折扣。", ""]
    return "\n".join(lines), {"recommend": rec, **verdicts}


def main() -> None:
    y = json.loads((OUT / "window_eval_y.json").read_text(encoding="utf-8"))
    path = OUT / "window_eval_m.json"
    m = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    grids = {(w, c): grid(c, HERE, d) for w, d in VARIANTS.items() for c in CAPITALS}
    picks = {(w, c): (ir_pick(HERE, c), ir_pick(d, c)) for w, d in VARIANTS.items() for c in CAPITALS}
    text, verdicts = render(y, m, grids, picks)
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")
    (OUT / "window_compare.json").write_text(json.dumps({"verdicts": verdicts, "grids": {f"{w}|{c}": g for (w, c), g in grids.items()},
                                                         "picks": {f"{w}|{c}": p for (w, c), p in picks.items()}},
                                                        ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(verdicts, ensure_ascii=False), "->", OUT / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
