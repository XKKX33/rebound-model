"""读各模型的模拟结果，按 PREREG.md 的选法写 REPORT.md；--finalists 另写 finalists.json（入围规则，补免五和按当时印花税）。
缺的模型（还没训完）在报告里标"未跑"，不报错。
用法：python -X utf8 report.py [--finalists]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from horizon import CAPITAL, FRESH_YEARS, HERE, MODELS, VARIANT_NAMES, VARIANTS, steadiest, verdict  # noqa: I001
from engine import month_ci
from rule_grid import parse, specs
from sim import YEAR

MAIN, TWO, FIVE, FIVE_GPU = "nocap", "o1o2-cuda", "o1o5-cpu", "o1o5-cuda"
OUT, FINALISTS, IC_CACHE = HERE / "REPORT.md", HERE / "finalists.json", HERE / "ic.json"
ORDER = [s.name for s in specs()]


def load(variant: str, model: str, fee: str = "min5", tier: str = "real") -> dict | None:
    js = HERE / variant / "sims" / f"{model}_{fee}_{tier}.json"
    return json.loads(js.read_text(encoding="utf-8")) if js.is_file() else None


def rows_of(res: dict | None) -> list[tuple[str, dict]] | None:
    """按网格顺序（并列取靠前的，和正式选法一样）。150 条没跑齐就当没有。"""
    if res is None or any(n not in res for n in ORDER):
        return None
    return [(n, {**res[n]["fresh_stats"], "full": res[n]["full"], "fresh": res[n]["fresh"]}) for n in ORDER]


def best_full(rows: list[tuple[str, dict]]) -> str:
    return max(rows, key=lambda item: item[1]["full"]["cagr"])[0]


def compare(a: list[tuple[str, dict]], b: list[tuple[str, dict]]) -> dict:
    """同一条规则两个模型：最差起点年化、全期年化之差（a − b）的中位数，a 更好的比例。"""
    da = np.array([x["worst_cagr"] - y["worst_cagr"] for (_, x), (_, y) in zip(a, b)])
    df = np.array([x["full"]["cagr"] - y["full"]["cagr"] for (_, x), (_, y) in zip(a, b)])
    return {"worst_median": float(np.median(da)), "worst_share": float((da > 0).mean()),
            "full_median": float(np.median(df)), "full_share": float((df > 0).mean()), "abs_worst_median": float(
                np.median(np.abs(da)))}


def paired(variant: str, m1: str, r1: str, m2: str, r2: str) -> dict | None:
    """两条（模型, 规则）的全期逐日收益之差：年化均值 [按月重抽区间]。"""
    z = [np.load(HERE / variant / "sims" / f"{m}_min5_real.npz") for m in (m1, m2)]
    if any(not np.array_equal(z[0][k], z[1][k]) for k in ("days", "dates")):
        raise SystemExit("两个模型的计分日不一样")
    names = [list(zz["names"]) for zz in z]
    diff = z[0]["ret"][names[0].index(r1)] - z[1]["ret"][names[1].index(r2)]
    lo, hi = month_ci(pd.Series(diff, index=pd.DatetimeIndex(z[0]["dates"])))
    return {"mean": float(diff.mean()) * YEAR * 100, "ci": [lo * YEAR * 100, hi * YEAR * 100]}


def day_corr(d: np.ndarray, a: np.ndarray, b: np.ndarray, n_days: int) -> np.ndarray:
    """每天 a、b 的相关系数（行数不到 30 的日子记 NaN）。"""
    cnt = np.bincount(d, minlength=n_days)
    ca = a - (np.bincount(d, a, n_days) / np.maximum(cnt, 1))[d]
    cb = b - (np.bincount(d, b, n_days) / np.maximum(cnt, 1))[d]
    den = np.sqrt(np.bincount(d, ca * ca, n_days) * np.bincount(d, cb * cb, n_days))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((cnt >= 30) & (den > 0), np.bincount(d, ca * cb, n_days) / den, np.nan)


def label_table(variant: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """变体目录里的各个标签（o1o2 / o1o3 / o1o5 的当天百分位），按 di·10000 + ji 排好。"""
    keys, ys = None, {}
    for f in sorted((HERE / variant).glob("labels10_o1o*_*.parquet")):
        lab = pd.read_parquet(f, columns=["di", "ji", "y"])
        key = lab.di.to_numpy().astype(np.int64) * 10_000 + lab.ji.to_numpy()
        order = np.argsort(key, kind="stable")
        if keys is None:
            keys = key[order]
        elif not np.array_equal(keys, key[order]):
            raise SystemExit(f"{f.name} 的行和别的标签对不上")
        ys[f.name.split("_")[1]] = lab.y.to_numpy()[order]
    return keys, ys


def ic(variant: str) -> dict[str, dict]:
    """每个模型的打分和各个标签在同一天里的秩相关（Spearman × 100）：逐年平均和全期平均。缓存在 ic.json。"""
    cache = json.loads(IC_CACHE.read_text(encoding="utf-8")) if IC_CACHE.is_file() else {}
    dates = pd.DatetimeIndex(json.loads((HERE / variant / "axes.json").read_text(encoding="utf-8"))["dates"])
    keys = ys = None
    labels_now = sorted(p.name for p in (HERE / variant).glob("labels10_o1o*_*.parquet"))
    for name, model in MODELS.items():
        f = HERE / variant / model.scores
        tag = f"{variant}|{name}"
        seen = cache.get(tag, {})
        if not f.is_file() or (seen.get("mtime") == f.stat().st_mtime and seen.get("labels") == labels_now):
            continue
        if keys is None:
            keys, ys = label_table(variant)
        s = pd.read_parquet(f)
        key = s.di.to_numpy().astype(np.int64) * 10_000 + s.ji.to_numpy()
        pos = np.minimum(np.searchsorted(keys, key), len(keys) - 1)
        if not (keys[pos] == key).all():
            raise SystemExit(f"{f.name} 有标签里没有的行")
        out = {}
        for label, y in ys.items():
            yy, sc, d = y[pos], s.score.to_numpy(np.float64), s.di.to_numpy()
            ok = np.isfinite(yy) & np.isfinite(sc)
            g = pd.DataFrame({"d": d[ok], "s": sc[ok], "y": yy[ok]})
            r = day_corr(g.d.to_numpy(), g.groupby("d").s.rank().to_numpy(), g.groupby("d").y.rank().to_numpy(),
                         len(dates))
            fin = np.isfinite(r)
            by = pd.Series(r[fin], index=dates.year[fin]).groupby(level=0).mean() * 100
            out[label] = {"all": float(np.nanmean(r) * 100), "years": {str(k): float(v) for k, v in by.items()}}
        cache[tag] = {"mtime": f.stat().st_mtime, "labels": labels_now, "ic": out}
        IC_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return {k.split("|")[1]: v["ic"] for k, v in cache.items() if k.startswith(f"{variant}|")}


def decide(rows: dict[str, list | None]) -> dict:
    """预注册第 1、2 条：每个模型最稳的规则；两日 vs 五日，差距要大过重训噪声（GPU 五日 vs CPU 五日的推荐之差）。"""
    pick = {m: steadiest(r) if r else None for m, r in rows.items()}
    worst = {m: dict(rows[m])[p]["worst_cagr"] if p else None for m, p in pick.items() if rows.get(m)}
    noise = (abs(worst[FIVE_GPU] - worst[FIVE]) if worst.get(FIVE_GPU) is not None and worst.get(FIVE) is not None
             else None)
    ready = rows.get(TWO) and rows.get(FIVE) and noise is not None
    call = verdict(worst.get(TWO), worst.get(FIVE), noise) if ready else "还没跑齐（缺两日或 GPU 五日）"
    winner = TWO if call.startswith("两日更好") else FIVE
    return {"pick": pick, "worst": worst, "noise": noise, "verdict": call, "winner": winner,
            "rule": pick.get(winner)}


def finalists(rows: dict[str, list | None]) -> list[dict]:
    """补免五和按当时印花税的规则：两日、五日各自最稳的、全期最高的；两日对口（FH-h1）里最稳的。"""
    out = []
    for m in (TWO, FIVE):
        r = rows.get(m)
        if not r:
            continue
        names = [steadiest(r), best_full(r)]
        if m == TWO:
            h1 = [(n, x) for n, x in r if n.startswith("FH-") and parse(n).p == 1]
            names.append(max(h1, key=lambda item: item[1]["worst_cagr"])[0])
        out.append({"variant": MAIN, "model": m, "rules": list(dict.fromkeys(n for n in names if n))})
    return out


def sg(v, digits: int = 1) -> str:
    return "—" if v is None else f"{v:+.{digits}f}"


def table(head: list[str], body: list[list]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in body])


def short(name: str) -> str:
    s = parse(name)
    return f"FH-K{s.k}-h{s.p:g}" if s.family == "FH" else f"DB-K{s.k}-b{s.p:g}"


def plain(name: str) -> str:
    s = parse(name)
    if s.family == "FH":
        return f"每天买前 {s.k} 名，拿 {s.p:g} 天卖" + ("（T+1 开盘买、T+2 开盘卖）" if s.p == 1 else "")
    return f"最多 {s.k} 只，掉出前 {s.p * s.k:g} 名才卖"


def rule_cells(name: str, r: dict) -> list:
    """一条规则：四个起点的年化、最差起点、全期（2015 起）年化、期末金额、最大回撤、费用（占资金每年）。"""
    f = r["full"]
    return [short(name), plain(name), *(sg(r["fresh"][str(y)]["cagr"]) for y in FRESH_YEARS), sg(r["worst_cagr"]),
            sg(f["cagr"]), f"{f['final']:,.0f}", f"{f['max_dd']:.1f}", f"{f['cost_pct']:.1f}%"]


RULE_HEAD = ["规则", "做法", *(f"{y} 起" for y in FRESH_YEARS), "最差起点", "全期年化", "全期期末（元）", "全期回撤", "费用/年"]


def conclusion(rows: dict, dec: dict) -> str:
    lines = ["## 结论", "",
             "**现在的模型是不是\"五天收益最大\"：基本是。** 标签是 o1o5：T+1 开盘买、T+5 开盘卖的收益，换成当天准入池里的百分位，"
             "xgboost 回归去拟合它。所以它学的是\"五天后这只票在当天的池子里排第几\"，不是绝对收益最大。"
             "两日模型只换了标签：T+1 开盘买、T+2 开盘卖（A 股 T+1 下最短的一笔），其余全同。", ""]
    pick, worst = dec["pick"], dec["worst"]
    body = []
    for m in (TWO, FIVE, FIVE_GPU):
        if rows.get(m) and pick.get(m):
            body.append([MODELS[m].label, *rule_cells(pick[m], dict(rows[m])[pick[m]])])
        elif rows.get(m):
            body.append([MODELS[m].label, "没有四个起点都赚的规则", *[""] * (len(RULE_HEAD) - 1)])
        else:
            body.append([MODELS[m].label, "还没跑完", *[""] * (len(RULE_HEAD) - 1)])
    lines += ["各模型最稳的规则（2 万、最低 5 元，不限价准入池；四个起点都赚、最差起点年化最高）：", "",
              table(["模型", *RULE_HEAD], body), ""]
    noise = dec["noise"]
    lines.append(f"- 判定（预注册第 2 条）：两日 {sg(worst.get(TWO))} vs 五日 {sg(worst.get(FIVE))}（最差起点年化），"
                 f"重训噪声 {f'{noise:.1f}' if noise is not None else '—'}（同一个五日标签在 GPU 上重训一遍，推荐规则的最差起点年化差了这么多）"
                 f" → **{dec['verdict']}**。")
    if dec["rule"]:
        r = dict(rows[dec["winner"]])[dec["rule"]]
        lines.append(f"- 推荐：**{MODELS[dec['winner']].label}模型 + {short(dec['rule'])}**（{plain(dec['rule'])}，每月重训）。"
                     f"2 万今天开户，按四个起点的经验年化 {sg(r['worst_cagr'])}% ~ "
                     f"{sg(max(v['cagr'] for v in r['fresh'].values()))}%；2015 年起步一路做到现在 2 万 → "
                     f"{r['full']['final']:,.0f} 元（年化 {sg(r['full']['cagr'])}%，最大回撤 {r['full']['max_dd']:.1f}%）。")
    best = [(m, best_full(rows[m])) for m in (TWO, FIVE) if rows.get(m)]
    if best:
        m, n = max(best, key=lambda item: dict(rows[item[0]])[item[1]]["full"]["cagr"])
        f = dict(rows[m])[n]["full"]
        lines.append(f"- 只看 2015 年起步的全期，最高的是{MODELS[m].label} + {short(n)}：2 万 → {f['final']:,.0f} 元（年化 {sg(f['cagr'])}%，"
                     f"回撤 {f['max_dd']:.1f}%），最差起点 {sg(dict(rows[m])[n]['worst_cagr'])}%。这个只作参考："
                     "2015 年起步的账户先涨大了，5 元最低佣金被摊薄，今天新开的 2 万账户做不到。")
    return "\n".join(lines)


def setup_section() -> str:
    return "\n".join([
        "## 怎么比的", "",
        "- 模型：五日（现行，十年实验存的每月重训打分）；两日（本实验训，T+1 开盘买、T+2 开盘卖）；"
        "五日 GPU 重训（和两日同一份代码、同一设备，用来量重训噪声）；三日（T+3 开盘卖，敏感性）。"
        "28 个特征、参数、3 个种子、每月重训的扩张窗口都和十年实验一样。",
        "- 准入池：不限价（正式系统的日常默认），剔除 < 2 元和退市整理期（当时名称带\"退\"；不限价池里多剔 12 个股票日、4 只票）。"
        "不剔退市整理期时，这里的标签和模拟和十年实验逐项一样（`logs/labels-*.log`、`logs/decade-*.log`）。",
        "- 账户：本金 2 万，佣金万五、每笔最低 5 元，过户费十万分之一，一手取整，印花税按现行 0.05%。",
        "- 持股规则 150 条。FH = 每天买前 K 名，每只拿 h 天、到期开盘卖（h = 1 就是两日模型学的那一笔）；"
        "DB = 最多拿 K 只，每天开盘把排名掉出前 b·K 名的卖掉，空位按排名补。每个行业最多 2 只，买不进顺延下一名。",
        "- 选法（预注册）：2017 / 2019 / 2021 / 2023 年第一个开市日各用 2 万重新起步，四个起点都赚、最差起点年化最高的那条。"
        "这是正式 `rebound capital` 给不到 10 万、有最低佣金的账户选规则的标准：2015 年起步的全期数字被早年涨大的账户摊薄了最低佣金。",
    ])


def ic_section(ics: dict | None) -> str:
    if not ics:
        return "## 打分和标签的秩相关\n\n还没算（打分没出齐）。"
    labels = sorted({lab for v in ics.values() for lab in v})
    models = [m for m in MODELS if m in ics]
    body = [[MODELS[m].label, *(sg(ics[m][lab]["all"], 2) if lab in ics[m] else "—" for lab in labels)] for m in models]
    years = sorted({y for m in models for lab in ics[m] for y in ics[m][lab]["years"]})
    pairs = [(m, lab) for m in models for lab in ("o1o2", "o1o5")]
    yearly = [[y, *(sg(ics[m].get(lab, {}).get("years", {}).get(y)) for m, lab in pairs)] for y in years]
    return "\n".join(["## 打分和标签的秩相关（每天在准入池里算 Spearman × 100，再平均）", "",
                      "对 o1o2 标签 = 打分高的票两天后在池子里排得靠不靠前；对 o1o5 = 五天后。", "",
                      table(["模型", *(f"对 {lab}" for lab in labels)], body), "", "逐年：", "",
                      table(["年", *(f"{MODELS[m].label}·{lab}" for m, lab in pairs)], yearly)])


def top_section(rows: dict) -> str:
    out = ["## 每个模型最稳的 10 条（按最差起点年化）"]
    for m in MODELS:
        r = rows.get(m)
        if not r:
            out += ["", f"### {MODELS[m].label}", "", "还没跑完。"]
            continue
        robust = sum(x["all_positive"] for _, x in r)
        best = sorted(r, key=lambda item: -item[1]["worst_cagr"])[:10]
        fh = sum(x["all_positive"] for n, x in r if n.startswith("FH-"))
        out += ["", f"### {MODELS[m].label}（四个起点都赚的 {robust} 条，其中固定持有 {fh} 条）", "",
                table(RULE_HEAD, [rule_cells(n, x) for n, x in best])]
    return "\n".join(out)


def head_to_head(rows: dict) -> str:
    pairs = [(TWO, FIVE, "两日 − 五日"), (FIVE_GPU, FIVE, "五日 GPU 重训 − 五日（重训噪声）"),
             ("o1o3-cuda", FIVE, "三日 − 五日")]
    body = []
    for a, b, title in pairs:
        if rows.get(a) and rows.get(b):
            c = compare(rows[a], rows[b])
            body.append([title, sg(c["worst_median"]), f"{c['worst_share']:.0%}", f"{c['abs_worst_median']:.1f}",
                         sg(c["full_median"]), f"{c['full_share']:.0%}"])
    if not body:
        return "## 同一条规则换模型\n\n还没跑齐。"
    return "\n".join(["## 同一条规则换模型（150 条逐条比）", "",
                      table(["比较", "最差起点年化差（中位数）", "前者更好的比例", "差的绝对值（中位数）",
                             "全期年化差（中位数）", "前者更好的比例"], body), "",
                      "第二行是噪声的量尺：标签没换，只是重训一遍，同一条规则的数字就会动这么多。"])


def h1_section(rows: dict) -> str:
    if not (rows.get(TWO) and rows.get(FIVE)):
        return "## 两日模型的对口规则（FH-h1）\n\n还没跑齐。"
    two, five = dict(rows[TWO]), dict(rows[FIVE])
    names = [n for n in ORDER if n.startswith("FH-") and parse(n).p == 1]
    body = [[short(n), sg(two[n]["worst_cagr"]), sg(two[n]["full"]["cagr"]), f"{two[n]['full']['cost_pct']:.1f}%",
             sg(five[n]["worst_cagr"]), sg(five[n]["full"]["cagr"])] for n in names]
    return "\n".join(["## 两日模型的对口规则（FH-h1：每天买前 K 名，T+1 开盘买、T+2 开盘卖）", "",
                      table(["规则", "两日·最差起点", "两日·全期年化", "两日·费用/年", "五日·最差起点", "五日·全期年化"], body),
                      "", "每天全仓换一遍：2 万下最低 5 元佣金加印花税，一年的费用就是上面那一列。"])


def detail_section(rows: dict, dec: dict) -> str:
    picks = [(m, dec["pick"].get(m)) for m in (TWO, FIVE) if rows.get(m) and dec["pick"].get(m)]
    if not picks:
        return "## 推荐规则的细节\n\n还没跑齐。"
    out = ["## 两个模型最稳那条的细节（2015 年起步的全期，2 万）"]
    years = sorted(dict(rows[picks[0][0]])[picks[0][1]]["full"]["years"])
    head = ["", *(f"{MODELS[m].label} {short(n)}" for m, n in picks)]
    full = [dict(rows[m])[n]["full"] for m, n in picks]
    body = [["期末（元）", *(f"{f['final']:,.0f}" for f in full)], ["年化", *(sg(f["cagr"]) for f in full)],
            ["超额（年化）[区间]", *(f"{sg(f['excess'])} [{sg(f['excess_ci'][0])}, {sg(f['excess_ci'][1])}]" for f in full)],
            ["最大回撤", *(f"{f['max_dd']:.1f}" for f in full)], ["β", *(f"{f['beta']:.2f}" for f in full)],
            ["每天买几只", *(f"{f['buys_per_day']:.2f}" for f in full)], ["平均拿几天", *(str(f["hold_days"]) for f in full)],
            ["换手（倍/年）", *(str(f["turnover"]) for f in full)], ["费用（占资金/年）", *(f"{f['cost_pct']:.1f}%" for f in full)]]
    body += [[f"{y} 收益（超额）", *(f"{sg(f['years'][y]['total'])}（{sg(f['years'][y]['excess'])}）" if y in f["years"]
                                     else "—" for f in full)] for y in years]
    out += ["", table(head, body), "", "换起点（2 万重新起步到现在）：", ""]
    fresh = [dict(rows[m])[n]["fresh"] for m, n in picks]
    out.append(table(["起点", *head[1:], "准入池同期年化"],
                     [[y, *(f"{sg(fr[str(y)]['cagr'])}%（回撤 {fr[str(y)]['max_dd']:.0f}%，×{fr[str(y)]['final_x']}）"
                            for fr in fresh), sg(fresh[0][str(y)]["pool_cagr"])] for y in FRESH_YEARS]))
    if len(picks) == 2:
        p = paired(MAIN, picks[0][0], picks[0][1], picks[1][0], picks[1][1])
        out += ["", f"两条的全期逐日收益之差（前者 − 后者）：年化 {sg(p['mean'])} [{sg(p['ci'][0])}, {sg(p['ci'][1])}]（按月重抽）。"]
    return "\n".join(out)


def sensitivity_section(extra: list[dict] | None = None) -> str:
    """入围规则（finalists.json）+ extra（四版都稳的那条，事后加的）在三档费用下的最差起点和全期。"""
    if not FINALISTS.is_file():
        return "## 免五、按当时印花税\n\n还没跑（全部模型跑完后补）。"
    items = json.loads(FINALISTS.read_text(encoding="utf-8")) + (extra or [])
    tiers = (("min5", "real"), ("free5", "real"), ("min5", "hist"))
    body, seen = [], set()
    for item in items:
        v, m = item["variant"], item["model"]
        res = {k: load(v, m, *k) for k in tiers}
        for n in item["rules"]:
            if (m, n) in seen:
                continue
            seen.add((m, n))
            got = {k: r[n] for k, r in res.items() if r and n in r}

            def cell(k, part):
                g = got.get(k)
                return "—" if g is None else sg(g["fresh_stats"]["worst_cagr"] if part == "worst" else g["full"]["cagr"])

            body.append([MODELS[m].label, short(n), *(cell(k, part) for k in tiers for part in ("worst", "full"))])
    return "\n".join(["## 免五、按当时印花税（入围规则）", "",
                      table(["模型", "规则", "最低 5 元·最差起点", "最低 5 元·全期", "免五·最差起点", "免五·全期",
                             "按当时印花税·最差起点", "按当时印花税·全期"], body), "",
                      "按当时印花税：2023-08-28 之前卖出收 0.1%（现行 0.05%），全期和 2017 / 2019 / 2021 年起步的都受影响。"
                      "\"—\"是那一档没跑。"])


def robust_pick(variants: dict) -> tuple[str | None, list[str]]:
    """事后的稳健性检查：五日模型的四版（两种准入池 × CPU / GPU 重训）里每个起点都赚的规则，
    按四版中最差的那个起点年化排（并列取网格靠前的）。四版没跑齐返回 (None, [])。"""
    runs = [variants[v].get(m) for v in VARIANTS for m in (FIVE, FIVE_GPU)]
    if not all(runs):
        return None, []
    tables = [dict(r) for r in runs]
    ok = [n for n in ORDER if all(t[n]["all_positive"] for t in tables)]
    ranked = sorted(ok, key=lambda n: -min(t[n]["worst_cagr"] for t in tables))
    return (ranked[0] if ranked else None), ranked


RUN_NAMES = [(v, m, f"{VARIANT_NAMES[v]}·{'CPU' if m == FIVE else 'GPU 重训'}") for v in VARIANTS for m in (FIVE, FIVE_GPU)]


def robust_section(variants: dict, dec: dict) -> str:
    best, ranked = robust_pick(variants)
    if best is None:
        return "## 事后检查：换准入池、重训一遍还稳不稳\n\n" + ("四版没跑齐。" if not ranked else "四版里没有处处都赚的规则。")
    tables = {label: dict(variants[v][m]) for v, m, label in RUN_NAMES}
    names = list(dict.fromkeys([*ranked[:6], *(n for n in (dec["pick"].get(FIVE),) if n)]))
    body = [[short(n), plain(n), *(sg(t[n]["worst_cagr"]) for t in tables.values()),
             sg(min(t[n]["worst_cagr"] for t in tables.values())), sum(t[n]["all_positive"] for t in tables.values())]
            for n in names]
    return "\n".join([
        "## 事后检查：换准入池、重训一遍还稳不稳（看完结果才加的，不在预注册里）", "",
        f"五日模型跑了四版（两种准入池 × CPU / GPU 重训）。四版里每个起点都赚的规则有 {len(ranked)} 条；"
        "按四版中最差的那个起点年化排：", "",
        table(["规则", "做法", *(f"{label}·最差起点" for label in tables), "四版最差", "四版都赚的版数"], body), "",
        "预注册选法只看一版，最稳的那条在别的版本里可能掉得很多：集中持仓（1~2 只）对打分的细微变化很敏感。"])


def robust_line(variants: dict, dec: dict) -> str | None:
    """结论里的一句：四版都稳的那条和预注册挑的不一样时，说明并改推它（理由写明）。"""
    best, _ = robust_pick(variants)
    pre = dec["pick"].get(FIVE)
    if best is None or pre is None or best == pre:
        return None
    worst = {label: dict(variants[v][m]) for v, m, label in RUN_NAMES}
    lo_pre = min(t[pre]["worst_cagr"] for t in worst.values())
    lo_best = min(t[best]["worst_cagr"] for t in worst.values())
    r = dict(variants[MAIN][FIVE])[best]
    return (f"- **事后检查后改推 {short(best)}**（{plain(best)}）：预注册挑的 {short(pre)} 换准入池、重训一遍后，"
            f"四版中最差的起点年化是 {sg(lo_pre)}%；{short(best)} 四版每个起点都赚，最差 {sg(lo_best)}%，是 150 条里最高的。"
            f"它在主版本的四个起点 {'、'.join(sg(r['fresh'][str(y)]['cagr']) for y in FRESH_YEARS)}%，"
            f"2015 年起步 2 万 → {r['full']['final']:,.0f} 元（年化 {sg(r['full']['cagr'])}%，最大回撤 {r['full']['max_dd']:.1f}%，"
            f"费用每年 {r['full']['cost_pct']:.1f}%）。这一步是看了结果才定的，不在预注册里。")


def variant_section(rows_cap: dict) -> str:
    dec = decide(rows_cap)
    body = []
    for m in (TWO, FIVE, FIVE_GPU):
        r, p = rows_cap.get(m), dec["pick"].get(m)
        if not r:
            body.append([MODELS[m].label, "还没跑完", "", "", "", ""])
            continue
        robust = sum(x["all_positive"] for _, x in r)
        x = dict(r)[p] if p else None
        body.append([MODELS[m].label, short(p) if p else "无", sg(x["worst_cagr"]) if x else "—",
                     sg(x["full"]["cagr"]) if x else "—", robust, short(best_full(r))])
    return "\n".join(["## 敏感性：限 70 元的准入池（十年实验的主线）", "",
                      table(["模型", "最稳的规则", "最差起点", "全期年化", "四个起点都赚的条数", "全期最高的规则"], body),
                      "", f"按同一判定：{dec['verdict']}（重训噪声 {'—' if dec['noise'] is None else f"{dec['noise']:.1f}"}）。只作参考，不改推荐。"])


def caveats() -> str:
    return "\n".join([
        "## 要说明的", "",
        "- 推荐规则是 150 条里按同一标准挑出来的，排第一的数字会偏高；两个模型都这样挑，比较是公平的，但绝对值别全信。",
        "- 换起点只有四个，都做到 2026-10；2026 年对集中持仓不友好（7 月），四个起点都包含这一年。",
        "- 没算冲击成本（2 万的单子很小，影响不大）；最低 5 元佣金是假设，用户没确认过，免五的数字在上面。",
        "- 两日、三日、GPU 五日是在 GPU 上训的：同一算法，但不逐位等于 CPU；噪声那一行就是在量这个。",
        "- 文件：PREREG.md（预注册）、train_horizon.py、sim_horizon.py、report.py、run_all.py；"
        "结果 nocap/、cap70/ 下的 sims/、scores_*.parquet；日志 run_all.log、logs/。",
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finalists", action="store_true")
    args = ap.parse_args()
    rows = {v: {m: rows_of(load(v, m)) for m in MODELS} for v in VARIANTS}
    dec = decide(rows[MAIN])
    if args.finalists:  # 先写入围名单，报告里的免五一节才不会读到上一次的
        FINALISTS.write_text(json.dumps(finalists(rows[MAIN]), ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        ics = ic(MAIN)
    except (OSError, ValueError, MemoryError) as e:  # 秩相关只是诊断：算不出来照样出报告
        print(f"rank IC skipped: {e!r}", flush=True)
        ics = None
    parts = [f"# 两日反弹模型 vs 五日反弹模型：十年、2 万本金\n\n生成于 {time.strftime('%Y-%m-%d %H:%M')}；"
             f"本金 {CAPITAL:,.0f} 元；预注册见 PREREG.md。",
             "\n".join(filter(None, [conclusion(rows[MAIN], dec), robust_line(rows, dec)])),
             setup_section(), top_section(rows[MAIN]), head_to_head(rows[MAIN]), h1_section(rows[MAIN]),
             detail_section(rows[MAIN], dec), robust_section(rows, dec), ic_section(ics),
             sensitivity_section([{"variant": MAIN, "model": FIVE, "rules": [n]} for n in [robust_pick(rows)[0]] if n]),
             variant_section(rows["cap70"]), caveats()]
    OUT.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"-> {OUT.name}: {dec['verdict']}; pick {dec['pick']}", flush=True)


if __name__ == "__main__":
    main()
