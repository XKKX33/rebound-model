"""补充登记七的报告：ind/theme_eval_q.json、ind/theme_eval_m.json（没有就是止损了）、两版的全网格（rules/<本金>/real_*.npz）
→ ind/REPORT.md 和 ind/theme_compare.json。判定按补充登记七机械地算。
用法：python -X utf8 theme_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from compare_cap import cagr, load
from rule_grid import CADENCE_NAMES, CADENCES
from theme_eval import CASES, PRIMARY, QS, TERCILES, WINDOWS

HERE = Path(__file__).resolve().parent
IND = HERE / "ind"
CAPITALS = (200000, 20000)


def ci(c: list[float]) -> str:
    return f"[{c[0]:+.1f}, {c[1]:+.1f}]"


def comp(ev: dict, case: str, kind: str) -> dict:
    return next(c for c in ev["comparisons"] if c["case"] == case and c["kind"] == kind)


def grid(capital: int, old: Path = HERE, new: Path = IND) -> dict | None:
    """V1 全网格：同一（频率, 规则）两版的账户年化差（两版准入池一样，所以也是超额的差）；两版滚动选法的结果。"""
    (a, _, dates_a, sel_a), (b, _, dates_b, sel_b) = load(old, capital), load(new, capital)
    if not a or not b:
        return None
    if not dates_a.equals(dates_b):
        raise SystemExit("两版的计分日不一样")
    by, gaps = {}, []
    for c in CADENCES:
        g = np.array([cagr(b[k]) - cagr(a[k]) for k in a if k in b and k[0] == c])
        if len(g):
            gaps.append(g)
            by[c] = {"n": len(g), "better": int((g > 0).sum()), "median_gap": round(float(np.median(g)), 2)}
    picks = {}
    for tag, sel in (("V0", sel_a), ("V1", sel_b)):
        if sel:
            best = sel["best_picker"]
            w = sel["walk_forward"][best]
            picks[tag] = {"picker": best, "rule": [sel["recommended"][best]["cadence"], sel["recommended"][best]["name"]],
                          **{k: w[k] for k in ("excess", "ci", "cagr", "max_dd")}}
    allg = np.concatenate(gaps)
    return {"by_cadence": by, "n": len(allg), "better": int((allg > 0).sum()), "median_gap": round(float(np.median(allg)), 2),
            "picks": picks}


def v1_verdict(q: dict, m: dict | None, grids: dict) -> tuple[str, float | None]:
    if m is None:
        return ("没有改进（每季度版主规则配对差的区间上限 < 0，按预先登记没训练每月版）" if q["stop"]
                else "还没有判定（每月版没跑完，下面是每季度版的中间结果）"), None
    c = comp(m, PRIMARY, "V1")["全期"]
    done = [g for g in grids.values() if g]
    share = 100 * sum(g["better"] for g in done) / sum(g["n"] for g in done) if done else float("nan")
    if c["ci"][0] > 0:
        return "确有改进", share
    if c["diff"] > 0 and share >= 60:
        return "可能有改进，值得模拟盘对照", share
    return "没有改进", share


def f_verdict(ev: dict) -> tuple[str, bool]:
    a, b = (comp(ev, case, "F20")["2015-2025"] for case in ("20万推荐", "2万推荐"))
    verdict = ("确有改进" if a["ci"][0] > 0 else "可能有改进") if a["diff"] > 0 and b["diff"] > 0 else "没有改进"
    signs = {np.sign(comp(ev, PRIMARY, f)["2015-2025"]["diff"]) for f in QS}
    return verdict, len(signs) > 1


def v1_rows(ev: dict) -> list[str]:
    out = ["| 规则 | 频率 | 全期 | 2015–2025 | 2026 | " + " | ".join(TERCILES) + " | " + " | ".join(WINDOWS) + " |",
           "|---" * (5 + len(TERCILES) + len(WINDOWS)) + "|"]
    for case in CASES:
        c = comp(ev, case, "V1")
        cells = [f"{c[p]['diff']:+.1f} {ci(c[p]['ci'])}" for p in ("全期", "2015-2025")] + [f"{c['2026']['diff']:+.1f}"]
        cells += [f"{c['regime_gap_diff'][t]:+.2f}" for t in TERCILES] + [f"{c['windows_diff'][w]:+.1f}" for w in WINDOWS]
        out.append(f"| {case} | {CADENCE_NAMES[c['cadence']]} | " + " | ".join(cells) + " |")
    return out


def f_rows(ev: dict) -> list[str]:
    jobs, bind = ev["jobs"], ev["binding"]
    out = ["| 规则 | q | 2015–2025 | 2026 | 最大回撤 原 → 新 | 2026-07 原 → 新 | 不买名单占前 30 名（每月打分） |", "|---" * 7 + "|"]
    for case, (_, cad, _) in CASES.items():
        base = jobs[f"V0|{cad}|{case}|-"]
        for f in QS:
            c, new = comp(ev, case, f), jobs[f"V0|{cad}|{case}|{f}"]
            out.append(f"| {case} | {f[1:]}% | {c['2015-2025']['diff']:+.1f} {ci(c['2015-2025']['ci'])} | {c['2026']['diff']:+.1f} | "
                       f"{base['all']['max_dd']:.1f} → {new['all']['max_dd']:.1f} | "
                       f"{base['windows']['2026-07']['book']:+.1f} → {new['windows']['2026-07']['book']:+.1f} | "
                       f"{bind.get(f, {}).get('top30%', float('nan')):.1f}% |")
    return out


def book_rows(ev: dict) -> list[str]:
    out = ["| 规则 | 版本 | 年化 | 超额 | 最大回撤 | " + " | ".join(f"{t}比池子" for t in TERCILES) + " |", "|---" * 8 + "|"]
    for case, (_, cad, _) in CASES.items():
        for k, v in ev["jobs"].items():
            src, c, name, f = k.split("|")
            if name != case:
                continue
            label = f"{src}{'' if f == '-' else ' + ' + f}（{CADENCE_NAMES[c]}）"
            a = v["all"]
            out.append(f"| {case} | {label} | {a['cagr']:+.1f}% | {a['excess']:+.1f} | {a['max_dd']:.1f}% | "
                       + " | ".join(f"{v['regime_gap'].get(t, float('nan')):+.2f}" for t in TERCILES) + " |")
    return out


def ic_rows(ev: dict) -> list[str]:
    out = ["| 打分频率 | 秩相关 V0 → V1（×100） | 差 (t) | 前 30 名平均名次 V0 → V1 | 差 (t) | 强势股月的秩相关 V0 → V1 |",
           "|---" * 6 + "|"]
    for cad, t in ev["ic"].items():
        mn, bt = t["mean"], t["by_tercile"]
        out.append(f"| {CADENCE_NAMES[cad]} | {mn['ic_V0']:.2f} → {mn['ic_V1']:.2f} | {t['ic_diff']['mean']:+.2f} ({t['ic_diff']['t']}) | "
                   f"{mn['top30_V0']:.1f} → {mn['top30_V1']:.1f} | {t['top30_diff']['mean']:+.2f} ({t['top30_diff']['t']}) | "
                   f"{bt['ic_V0'].get('强势股月', float('nan')):.2f} → {bt['ic_V1'].get('强势股月', float('nan')):.2f} |")
    return out


def grid_rows(grids: dict) -> list[str]:
    out = []
    for capital, g in grids.items():
        if not g:
            out.append(f"- {capital // 10000} 万：没跑")
            continue
        out.append(f"- {capital // 10000} 万：V1 账户收益更高的 {g['better']}/{g['n']}（{g['better'] / g['n'] * 100:.0f}%），"
                   f"年化差中位 {g['median_gap']:+.2f}。按频率："
                   + "，".join(f"{CADENCE_NAMES[c]} {v['better']}/{v['n']}（中位 {v['median_gap']:+.2f}）" for c, v in g["by_cadence"].items()))
        for tag, p in g["picks"].items():
            out.append(f"  - {tag} 滚动选法（{p['picker']}）：滚动样本外超额 {p['excess']:+.1f} {ci(p['ci'])}，年化 {p['cagr']:+.1f}%，"
                       f"回撤 {p['max_dd']:.1f}%；用全部数据挑的是 {CADENCE_NAMES[p['rule'][0]]}重训 {p['rule'][1]}")
    return out


def render(q: dict, m: dict | None, grids: dict) -> tuple[str, dict]:
    ev = m or q
    v1, share = v1_verdict(q, m, grids)
    version = "每月重训终版" if m else ("每季度版（止损，没训练每月版）" if q["stop"] else "每季度版（中间结果，每月版没跑完）")
    fv, unstable = f_verdict(ev)
    c, f = comp(ev, PRIMARY, "V1"), comp(ev, PRIMARY, "F20")
    shown = share is not None and not np.isnan(share)
    lines = ["# 行业特征和\"热门股刚跌不买\"（补充登记七）", "",
             f"版本：{version}；真实账户、现行印花税、2015-01 起计分。"
             "V0 = 现在的模型；V1 = 加 9 个行业特征重训；F = 热门股刚开始跌就不买（只改买入）。差都是年化百分点。", "",
             "## 结论", "",
             f"- **行业特征（V1）：{v1}。** 主规则（20 万{CADENCE_NAMES[c['cadence']]}重训 DB-K30-b8）V1 − V0：全期 {c['全期']['diff']:+.1f} "
             f"{ci(c['全期']['ci'])}，2015–2025 {c['2015-2025']['diff']:+.1f}，2026 {c['2026']['diff']:+.1f}。"
             + (f"全网格里 V1 账户收益更高的占 {share:.0f}%。" if shown else ""),
             f"  每月比准入池多赚的变化：强势股月 {c['regime_gap_diff']['强势股月']:+.2f}，中间 {c['regime_gap_diff']['中间']:+.2f}，"
             f"反转月 {c['regime_gap_diff']['反转月']:+.2f}（百分点）。",
             f"- **热门股刚跌不买（F，q = 20%）：{fv}{'；q 换成 10% / 30% 符号不一致，不稳' if unstable else ''}。** "
             f"主规则 2015–2025 {f['2015-2025']['diff']:+.1f} {ci(f['2015-2025']['ci'])}，2026 {f['2026']['diff']:+.1f}（2026 不算样本外）。", "",
             "## 行业特征：V1 − V0 逐日配对差", "",
             "区间是按月整块重抽的 95% 区间。三段之后是每月比准入池多赚多少的变化（百分点），最后是几段行情里组合收益的变化（百分点）。", "",
             *v1_rows(ev), ""]
    if m is not None:
        lines += ["每季度版（中间检查，只用来止损）：", "", *v1_rows(q), ""]
    lines += ["## 打分（同一批行，逐日秩相关按月平均）", "", *ic_rows(ev), "",
              "## 热门股刚跌不买：F − 不加（现在的模型；判定只看 2015–2025）", "", *f_rows(ev), ""]
    if m is not None:
        for kind in ("V1+F20 vs V1", "V1+F20 vs V0"):
            lines.append(f"- {kind}：" + "；".join(
                f"{case} 2015–2025 {comp(m, case, kind)['2015-2025']['diff']:+.1f} {ci(comp(m, case, kind)['2015-2025']['ci'])}，"
                f"全期 {comp(m, case, kind)['全期']['diff']:+.1f}" for case in CASES))
        lines.append("")
    lines += ["## 各版本的数字", "", *book_rows(ev), "", "## 全网格（2 个本金 × 4 种频率 × 150 条规则）", "", *grid_rows(grids), "",
              "## 要说明的", "",
              "- 行业用 stock_basic 现在的行业，套在整段历史上，有轻微前视；正式策略的行业上限也是这个口径。",
              "- F 是看过 2026 年 7 月以后想出来的，2026 不算样本外，判定只看 2015–2025。",
              "- 每季度版只用来止损，判定用每月版。V1 只加了行业特征，准入池、标签、参数、种子、剔除都和 V0 一样。",
              "- 三套规则、几个 q、两种改动一起看，有多重比较：单个区间刚过 0 的要打折扣。", ""]
    return "\n".join(lines), {"V1": v1, "V1_grid_share": share, "F20": fv, "F_unstable": unstable}


def main() -> None:
    q = json.loads((IND / "theme_eval_q.json").read_text(encoding="utf-8"))
    path = IND / "theme_eval_m.json"
    m = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    grids = {c: grid(c) if m is not None else None for c in CAPITALS}
    text, verdicts = render(q, m, grids)
    (IND / "REPORT.md").write_text(text, encoding="utf-8")
    (IND / "theme_compare.json").write_text(json.dumps({"verdicts": verdicts, "grids": grids}, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    print(json.dumps(verdicts, ensure_ascii=False), "->", IND / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
