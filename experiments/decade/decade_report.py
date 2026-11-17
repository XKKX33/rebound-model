"""sim10_results.json → 中文报告 REPORT.md。用法：python -X utf8 decade_report.py"""
from __future__ import annotations

import json

from common import DATA

CAD = {"Y": "按年", "H": "按半年", "Q": "按季", "M": "按月"}
DESC = {"FH-K6-next-o1o5": "每天买前 6 只、分 4 批持有（原方案）", "FH-K2-next-o1o5-small": "每天买前 2 只、分 4 批持有"}


def f1(v) -> str:
    return "—" if v is None else f"{v:+.1f}"


def ci(c) -> str:
    return f"[{c[0]:+.1f}, {c[1]:+.1f}]"


def cadences(out: dict) -> list[str]:
    return [c for c in CAD if c in out["results"]]


def context_table(out: dict) -> list[str]:
    ctx, names = out["context"], list(out["context"]["index"])
    lines = ["| 年份 | 准入池等权 | " + " | ".join(names) + " | 准入池每天平均只数 |", "|" + "---|" * (len(names) + 3)]
    for y, p in ctx["pool"]["years"].items():
        cells = [f1(p["total"]), *(f1(ctx["index"][n].get(y)) for n in names), f"{ctx['pool_size'].get(y, 0):.0f}"]
        lines.append(f"| {y} | " + " | ".join(cells) + " |")
    lines.append("| 全期年化 | " + " | ".join([f1(ctx["pool"]["all"]["cagr"]), *(f1(ctx["index"][n]["all"]) for n in names)])
                 + " | |")
    return lines


def year_table(out: dict, name: str) -> list[str]:
    cads, pool = cadences(out), out["context"]["pool"]
    lines = ["| 年份 | 准入池 | " + " | ".join(f"{CAD[c]}：收益 / 超额" for c in cads) + " |", "|" + "---|" * (len(cads) + 2)]
    for y in pool["years"]:
        cells = [f1(pool["years"][y]["total"])]
        for c in cads:
            r = out["results"][c][name]["hist"]["years"][y]
            cells.append(f"{r['total']:+.1f} / {r['excess']:+.1f}")
        lines.append(f"| {y} | " + " | ".join(cells) + " |")
    alls = [f"{out['results'][c][name]['hist']['all']['cagr']:+.1f} / {out['results'][c][name]['hist']['all']['excess']:+.1f}"
            for c in cads]
    lines.append(f"| 全期年化 | {f1(pool['all']['cagr'])} | " + " | ".join(alls) + " |")
    return lines


def summary_table(out: dict, name: str) -> list[str]:
    lines = ["| 频率 | 全期年化 | 年化超额 [区间] | β | α | 最大回撤 | 20 万变成 | 闲置现金 | 现行印花税：年化 / 超额 | 资金不受限：超额 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cadences(out):
        row = out["results"][c][name]
        h, r = row["hist"], row["real"]
        prop = row["prop"]["all"]["excess"] if "prop" in row else None
        lines.append(f"| {CAD[c]} | {h['all']['cagr']:+.1f}% | {h['all']['excess']:+.1f} {ci(h['all']['excess_ci'])} | "
                     f"{h['all']['beta']:.2f} | {h['all']['alpha']:+.1f} | {h['all']['max_dd']:.1f}% | {h['final']:,.0f} | "
                     f"{h.get('idle_cash_pct', 0):.1f}% | {r['all']['cagr']:+.1f}% / {r['all']['excess']:+.1f} | {f1(prop)} |")
    return lines


def beta_table(out: dict, name: str, cadence: str = "Y") -> list[str]:
    pool, res = out["context"]["pool"]["years"], out["results"][cadence][name]["hist"]["years"]
    lines = ["| 年份 | 准入池 | 组合收益 | 超额 | β | α |", "|---|---|---|---|---|---|"]
    for y in pool:
        r = res[y]
        lines.append(f"| {y} | {f1(pool[y]['total'])} | {r['total']:+.1f} | {r['excess']:+.1f} | {r['beta']:.2f} | {r['alpha']:+.1f} |")
    return lines

def regime_table(out: dict) -> list[str]:
    """按准入池当年涨跌分组（只看整年），各组的平均超额和 α。"""
    pool, last = out["context"]["pool"]["years"], out["context"]["last"][:4]
    years = [y for y in pool if y != last and pool[y]["days"] >= 200]
    groups = {"准入池上涨的年份": [y for y in years if pool[y]["total"] > 0],
              "准入池下跌的年份": [y for y in years if pool[y]["total"] <= 0]}
    lines = ["| 分组 | 年份 | " + " | ".join(f"{n.split('-')[1]} 平均超额 / α" for n in out["main"]) + " |",
             "|" + "---|" * (len(out["main"]) + 2)]
    for label, ys in groups.items():
        cells = []
        for name in out["main"]:
            res = out["results"]["Y"][name]["hist"]["years"]
            cells.append("—" if not ys else f"{sum(res[y]['excess'] for y in ys) / len(ys):+.1f} / "
                                             f"{sum(res[y]['alpha'] for y in ys) / len(ys):+.1f}")
        lines.append(f"| {label} | {'、'.join(ys) or '—'} | " + " | ".join(cells) + " |")
    return lines


def cadence_table(out: dict) -> tuple[list[str], list[str]]:
    lines = ["| 频率 | " + " | ".join(f"{n.split('-')[1]}：差 [区间]，好过按年的年数" for n in out["main"])
             + " | 全部规则里差为正 / 区间下限 > 0 |", "|" + "---|" * (len(out["main"]) + 2)]
    verdict = []
    for c in cadences(out)[1:]:
        cells, wins = [], []
        for name in out["main"]:
            d = out["cadence_vs_yearly"][f"{c}|{name}|hist"]
            cells.append(f"{d['ann']:+.1f} {ci(d['ci'])}，{d['better_years']}/{d['n_years']}")
            wins.append(d["ci"][0] > 0)
        every = [v for k, v in out["cadence_vs_yearly"].items() if k.startswith(f"{c}|") and k.endswith("|hist")]
        cells.append(f"{sum(v['ann'] > 0 for v in every)} / {sum(v['ci'][0] > 0 for v in every)}")
        lines.append(f"| {CAD[c]} | " + " | ".join(cells) + " |")
        verdict.append(f"{CAD[c]}：{'比按年好（两条主规则的区间下限都 > 0）' if all(wins) else '和按年分不出'}")
    return lines, verdict


def appendix(out: dict) -> list[str]:
    cads = cadences(out)
    lines = ["| 规则 | " + " | ".join(f"{CAD[c]}：年化 / 超额" for c in cads) + " |", "|" + "---|" * (len(cads) + 1)]
    for name in out["results"][cads[0]]:
        cells = [f"{out['results'][c][name]['hist']['all']['cagr']:+.1f} / {out['results'][c][name]['hist']['all']['excess']:+.1f}"
                 for c in cads]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def render(out: dict) -> str:
    ctx, cads = out["context"], cadences(out)
    lines = [f"# 反弹模型十年回测：逐年收益和更新频率（本金 {out['capital']:,.0f} 元）", "",
             f"计分区间 {ctx['first']} ~ {ctx['last']}（{ctx['first'][:4]} 年从 {ctx['first'][5:]} 起，{ctx['last'][:4]} 年只到 {ctx['last'][5:]}）。"
             f"成本：佣金万五（每笔最低 5 元）、过户费十万分之一，印花税按当时的税率（2023-08-28 之前 0.1%，之后 0.05%）；"
             f"买入按一手取整；不计冲击成本。已算的频率：{'、'.join(CAD[c] for c in cads)}。", "",
             "## 行情背景（逐年涨跌，%）", ""]
    lines += context_table(out)
    for name in out["main"]:
        lines += ["", f"## {name}：{DESC[name]}", "", "逐年：绝对收益 / 相对准入池等权的超额（%）。", ""]
        lines += year_table(out, name)
        lines += ["", "全期："]
        lines += ["", *summary_table(out, name)]
    lines += ["", "## 超额里有多少是\"跟着行情放大\"（按年重训，逐年 β 和 α）", "",
              "β 是组合日收益对准入池日收益的回归系数；α 是扣掉 β × 准入池之后剩下的年化收益。"
              "超额 = α + (β − 1) × 准入池的年化平均日收益：β > 1 时，行情好的年份超额被放大，行情差的年份被压低。", ""]
    for name in out["main"]:
        lines += [f"{name}：", "", *beta_table(out, name), ""]
    lines += ["按准入池当年涨跌分组（只看整年）：", "", *regime_table(out)]
    if len(cads) > 1:
        table, verdict = cadence_table(out)
        lines += ["", "## 更新频率：和按年比", "",
                  "差 = 逐日收益之差的年化（百分点），区间是按月重抽的 95% 区间。", "", *table, "",
                  "按事先登记的判定（两条主规则的区间下限都 > 0 才算更好）：" + "；".join(verdict) + "。"]
    p = {n: out["results"]["Y"][n]["real"]["p0p1"] for n in out["main"]}
    lines += ["", "## 和 p0p1 的对照", "",
              "同样从 2022-09 起、按年重训、现行印花税：" + "；".join(f"{n} 年化超额 {v['excess']:+.1f}" for n, v in p.items())
              + "。p0p1 第四轮（20 万）是 FH-K6 +16.2、FH-K2-small +28.6。"
              "两边不完全一样：这里训练数据从 2013 年起（p0p1 从 2021 年起），ST 按当时的名称判断。",
              "", f"## 附表：{len(out['results'][cads[0]])} 组持股规则（全期，印花税按当时的税率）", "", *appendix(out),
              "", f"用时 {out.get('seconds', 0)} 秒。"]
    return "\n".join(lines) + "\n"


def main() -> None:
    out = json.loads((DATA / "sim10_results.json").read_text(encoding="utf-8"))
    (DATA / "REPORT.md").write_text(render(out), encoding="utf-8")
    print("-> REPORT.md")


if __name__ == "__main__":
    main()
