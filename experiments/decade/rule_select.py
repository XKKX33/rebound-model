"""持股规则选优：读 rule_search 的 real 档结果，做全期排名、网格平滑、逐年滚动选优（样本外）、
过拟合概率 PBO（CSCV）、White 现实检验、更新频率的影响，选出入围的规则。输出 rules/<本金>/select.json。

选法（PREREG 补充登记一、二）：smooth = 平滑后的超额；raw = 不平滑的超额；ir = 信息比率；
growth = 平滑后的逐日对数收益均值（复利增长，看到首批结果后补登）。
用法：python -X utf8 rule_select.py [--capital 200000]
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from common import DATA, register
from rule_grid import CADENCES, REFERENCE, Spec, axes_of, neighbors, out_dir, parse
from engine import month_ci
from sim import YEAR

TEST_YEARS = tuple(range(2017, 2027))
PICKERS = ("smooth", "raw", "ir", "growth")
N_TOP = 4     # 入围：平滑超额、信息比率各取前几名，再加四种选法的推荐、事后最好、原方案
N_TABLE = 12
TIE = 0.5     # 滚动样本外超额差不到这么多（百分点）算打平，打平时比区间下限（补充登记六）


def best_of(walked: dict) -> str:
    """滚动样本外超额最高的选法；差不到 TIE 的算打平，打平时取区间下限高的。"""
    top = max(w["excess"] for w in walked.values())
    tied = [p for p in walked if walked[p]["excess"] >= top - TIE]
    return max(tied, key=lambda p: walked[p]["ci"][0])


def load(capital: float, tier: str = "real"):
    """各频率的结果拼成一张表：configs[(频率, Spec)]、逐日收益 book[组, 日]、准入池 pool[日]、计分日、全期数字。"""
    configs, rets, stats, pool, dates = [], [], {}, None, None
    for cadence in CADENCES:
        npz = out_dir(capital) / f"{tier}_{cadence}.npz"
        if not npz.is_file():
            continue
        z = np.load(npz)
        if pool is None:
            pool, dates = z["pool"], pd.DatetimeIndex(z["dates"])
        elif not (np.array_equal(z["pool"], pool) and np.array_equal(pd.DatetimeIndex(z["dates"]), dates)):
            raise SystemExit(f"{npz.name} 的计分日或准入池和别的频率不一致")
        js = json.loads(npz.with_suffix(".json").read_text(encoding="utf-8"))
        for name, r in zip(z["names"], z["ret"]):
            configs.append((cadence, parse(str(name))))
            rets.append(r)
            stats[(cadence, str(name))] = js[str(name)]
    if not configs:
        raise SystemExit(f"{out_dir(capital)} 里没有 {tier} 档的结果：先跑 rule_search.py")
    return configs, np.vstack(rets), pool, dates, stats


def annual(x: np.ndarray, axis: int = -1):
    return x.mean(axis=axis) * YEAR * 100


def neighbor_index(configs: list[tuple[str, Spec]]) -> list[np.ndarray]:
    """每组的网格邻居（同一频率、同一类，含自己）在 configs 里的下标。"""
    where = {c: i for i, c in enumerate(configs)}
    by_cadence: dict[str, set[Spec]] = {}
    for cadence, spec in configs:
        by_cadence.setdefault(cadence, set()).add(spec)
    return [np.array([where[(cadence, s)] for s in neighbors(spec, by_cadence[cadence])]) for cadence, spec in configs]


def smooth(score: np.ndarray, near: list[np.ndarray]) -> np.ndarray:
    return np.array([score[ix].mean() for ix in near])


def scores(book: np.ndarray, excess: np.ndarray, near: list[np.ndarray]) -> dict[str, np.ndarray]:
    """四种选法各自的打分（越大越好）。book、excess 是 [组, 日] 的逐日收益和超额。"""
    mean = excess.mean(axis=1)
    sd = excess.std(axis=1, ddof=1)
    return {"smooth": smooth(mean, near), "raw": mean, "ir": mean / np.where(sd > 0, sd, np.inf),
            "growth": smooth(np.log1p(book).mean(axis=1), near)}


def walk_forward(book, excess, dates: pd.DatetimeIndex, near, picker: str, years=TEST_YEARS) -> dict:
    """每年年初只用之前的数据选一组，记它这一年的逐日收益，拼成一条样本外序列。"""
    picks, b_oos, e_oos, when = {}, [], [], []
    for y in years:
        past, now = dates.year < y, dates.year == y
        if not now.any() or not past.any():
            continue
        i = int(np.argmax(scores(book[:, past], excess[:, past], near)[picker]))
        picks[str(y)] = i
        b_oos.append(book[i, now])
        e_oos.append(excess[i, now])
        when.append(dates[now])
    return {"picks": picks, "book": np.concatenate(b_oos), "excess": np.concatenate(e_oos),
            "dates": when[0].append(when[1:]) if len(when) > 1 else when[0]}


def series_stats(book: np.ndarray, excess: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    """一条逐日序列：年化超额 [按月重抽区间]、年化收益、最大回撤、逐年超额。"""
    lo, hi = month_ci(pd.Series(excess, index=dates))
    nav = np.cumprod(1 + book)
    peak = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
    return {"excess": round(float(annual(excess)), 2), "ci": [round(lo * YEAR * 100, 2), round(hi * YEAR * 100, 2)],
            "cagr": round((float(nav[-1]) ** (YEAR / len(nav)) - 1) * 100, 2),
            "max_dd": round(float((nav / peak - 1).min()) * 100, 2),
            "years": {str(y): round(float(annual(excess[dates.year == y])), 2) for y in np.unique(dates.year)}}


def pbo(excess: np.ndarray, blocks: int = 16, criterion: str = "mean") -> dict:
    """CSCV：计分日切 blocks 段，所有一半对一半的分法里，看样本内第一的那组在样本外排第几。
    criterion：mean 按平均超额排，ir 按信息比率排（样本内外用同一个标准）。
    PBO = 排到中位数以下（logit < 0）的比例；另报这组在样本内、样本外的平均年化超额。"""
    edges = np.linspace(0, excess.shape[1], blocks + 1).astype(int)
    parts = list(zip(edges[:-1], edges[1:]))
    sums = np.stack([excess[:, a:b].sum(axis=1) for a, b in parts], axis=1)  # [组, 段]
    squares = np.stack([(excess[:, a:b] ** 2).sum(axis=1) for a, b in parts], axis=1)
    sizes = np.diff(edges).astype(float)
    combos = np.array(list(itertools.combinations(range(blocks), blocks // 2)))
    m = np.zeros((len(combos), blocks))
    m[np.arange(len(combos))[:, None], combos] = 1.0
    cols = np.arange(len(combos))

    def side(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = mask @ sizes
        mean = (sums @ mask.T) / n                          # [组, 分法]
        if criterion == "mean":
            return mean, mean
        sd = np.sqrt(np.maximum((squares @ mask.T) / n - mean ** 2, 1e-18))
        return mean, mean / sd

    is_mean, is_score = side(m)
    os_mean, os_score = side(1 - m)
    best = is_score.argmax(axis=0)
    os_best = os_score[best, cols]
    worse = (os_score < os_best).sum(axis=0) + 0.5 * ((os_score == os_best).sum(axis=0) - 1)
    omega = (worse + 1) / (excess.shape[0] + 1)
    logit = np.log(omega / (1 - omega))
    return {"pbo": round(float((logit < 0).mean()), 3), "n_splits": len(combos), "criterion": criterion,
            "is_best": round(float(is_mean[best, cols].mean()) * YEAR * 100, 2),
            "oos_of_is_best": round(float(os_mean[best, cols].mean()) * YEAR * 100, 2),
            "oos_median_all": round(float(np.median(os_mean, axis=0).mean()) * YEAR * 100, 2)}


def reality_check(excess: np.ndarray, dates: pd.DatetimeIndex, base: np.ndarray | None = None,
                  n_boot: int = 2000, seed: int = 20261030) -> dict:
    """White 现实检验（按月整块重抽）：零假设"没有一组的期望超额 > 0"（给 base 时是"没有一组跑赢 base"）。
    统计量是各组均值的最大值；p 值 = 重抽后（各组先减去自己的均值）的最大值不小于它的比例。
    n_above = 均值超过这个零分布 95% 分位的组数（单步多重比较校正后仍显著的组）。"""
    d = excess - base if base is not None else excess
    frame = pd.DataFrame(d.T, index=dates.to_period("M"))
    sums = frame.groupby(level=0, sort=True).sum().to_numpy().T      # [组, 月]
    counts = frame.groupby(level=0, sort=True).size().to_numpy().astype(float)
    mean = d.mean(axis=1)
    pick = np.random.default_rng(seed).integers(0, len(counts), size=(n_boot, len(counts)))
    times = np.zeros((n_boot, len(counts)))
    np.add.at(times, (np.repeat(np.arange(n_boot), len(counts)), pick.ravel()), 1.0)
    boot = ((times @ sums.T) / (times @ counts)[:, None] - mean).max(axis=1)
    stat, crit = mean.max(), np.quantile(boot, 0.95)
    return {"stat": round(float(stat) * YEAR * 100, 2), "p": round(float((boot >= stat).mean()), 4),
            "crit": round(float(crit) * YEAR * 100, 2), "n_above": int((mean > crit).sum()), "n": int(len(mean))}


def cadence_effect(configs, excess: np.ndarray) -> dict:
    """每个网格点上，H/Q/M 相对 Y 的年化超额差（同一条规则）：中位数、四分位、为正的比例。"""
    where = {c: i for i, c in enumerate(configs)}
    ann = annual(excess, axis=1)
    out = {}
    for cadence in CADENCES[1:]:
        for family in ("FH", "DB", "all"):
            diffs = np.array([ann[i] - ann[where[("Y", spec)]] for i, (c, spec) in enumerate(configs)
                              if c == cadence and ("Y", spec) in where and family in ("all", spec.family)])
            if len(diffs):
                q = np.percentile(diffs, [25, 50, 75])
                out[f"{cadence}|{family}"] = {"n": len(diffs), "q25": round(float(q[0]), 2),
                                              "median": round(float(q[1]), 2), "q75": round(float(q[2]), 2),
                                              "positive": round(float((diffs > 0).mean()), 3)}
    return out


def paired(a: np.ndarray, b: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    lo, hi = month_ci(pd.Series(a - b, index=dates))
    return {"ann": round(float(annual(a - b)), 2), "ci": [round(lo * YEAR * 100, 2), round(hi * YEAR * 100, 2)]}


def heatmaps(configs, excess: np.ndarray) -> dict:
    """每个频率、每类规则：K × h/b 的全期年化超额和信息比率（网格外的格子是 None）。"""
    values = {"excess": annual(excess, axis=1), "ir": excess.mean(axis=1) / excess.std(axis=1, ddof=1) * YEAR ** 0.5}
    out = {}
    for cadence in CADENCES:
        for family in ("FH", "DB"):
            ks, ps = axes_of(family)
            cell = {(cadence, Spec(family, k, p)): (k, p) for k in ks for p in ps}
            if not any(c in cell for c in configs):
                continue
            where = {c: i for i, c in enumerate(configs) if c in cell}
            grids = {key: [[round(float(v[where[(cadence, Spec(family, k, p))]]), 2)
                            if (cadence, Spec(family, k, p)) in where else None for p in ps] for k in ks]
                     for key, v in values.items()}
            out[f"{cadence}|{family}"] = {"k": list(ks), "p": list(ps), **grids}
    return out


def label(config) -> list[str]:
    return [config[0], config[1].name]


def select(configs, book: np.ndarray, pool: np.ndarray, dates: pd.DatetimeIndex, stats: dict) -> dict:
    """纯计算：全期打分、四种选法的推荐、滚动选优、对照、PBO、现实检验、频率影响、入围名单、主表。"""
    excess = book - pool
    if not np.isfinite(excess).all():
        raise SystemExit("逐日收益里有 NaN：rule_search 的结果不完整")
    near = neighbor_index(configs)
    full = scores(book, excess, near)
    walk = {p: walk_forward(book, excess, dates, near, p) for p in PICKERS}
    test = dates.isin(walk["smooth"]["dates"])
    hind = int(np.argmax(full["raw"]))
    ref = configs.index((REFERENCE[0], parse(REFERENCE[1]))) if (REFERENCE[0], parse(REFERENCE[1])) in configs else None
    bench = {"hindsight": series_stats(book[hind, test], excess[hind, test], dates[test]),
             "average": series_stats(book[:, test].mean(axis=0), excess[:, test].mean(axis=0), dates[test])}
    if ref is not None:
        bench["reference"] = series_stats(book[ref, test], excess[ref, test], dates[test])
    rec = {p: int(np.argmax(full[p])) for p in PICKERS}
    order = {p: list(np.argsort(-full[p])) for p in ("smooth", "ir")}
    finalists = list(dict.fromkeys([*order["smooth"][:N_TOP], *order["ir"][:N_TOP], *rec.values(), hind,
                                    *([ref] if ref is not None else [])]))
    whole = [str(y) for y in np.unique(dates.year) if (dates.year == y).sum() >= 200]

    def row(i: int) -> dict:
        cadence, spec = configs[i]
        s = stats[(cadence, spec.name)]
        yearly = [s["years"][y]["excess"] for y in whole if y in s["years"]]
        return {"cadence": cadence, "name": spec.name, "plain": spec.plain, "family": spec.family, "k": spec.k,
                "smooth": round(float(full["smooth"][i]) * YEAR * 100, 2),
                "ir": round(float(full["ir"][i]) * YEAR ** 0.5, 2),
                "worst_year": min(yearly), "years_positive": f"{sum(v > 0 for v in yearly)}/{len(yearly)}",
                **{k: s[k] for k in ("excess", "excess_ci", "cagr", "max_dd", "beta", "alpha", "final", "buys_per_day",
                                     "hold_days", "turnover", "cost_pct", "idle_cash_pct", "avg_holdings")}}

    def detail(i: int) -> dict:
        cadence, spec = configs[i]
        out = {"years": stats[(cadence, spec.name)]["years"]}
        twin = ("Y", spec)
        if cadence != "Y" and twin in configs:
            out["vs_yearly"] = paired(excess[i], excess[configs.index(twin)], dates)
        if ref is not None and i != ref:
            out["vs_reference"] = paired(excess[i], excess[ref], dates)
        return out

    def walk_row(w: dict) -> dict:
        out = {**series_stats(w["book"], w["excess"], w["dates"]),
               "picks": {y: label(configs[i]) for y, i in w["picks"].items()}}
        if ref is not None:
            out["vs_reference"] = paired(w["excess"], excess[ref, test], w["dates"])
        return out

    walked = {p: walk_row(w) for p, w in walk.items()}
    by_cadence = {p: {c: row(max((i for i, (cc, _) in enumerate(configs) if cc == c), key=lambda i: full[p][i]))
                      for c in CADENCES if any(cc == c for cc, _ in configs)} for p in ("smooth", "ir")}
    return {"n_configs": len(configs), "cadences": sorted({c for c, _ in configs}, key=CADENCES.index),
            "first": str(dates[0].date()), "last": str(dates[-1].date()), "whole_years": whole,
            "recommended": {p: row(i) for p, i in rec.items()}, "hindsight": row(hind),
            "best_picker": best_of(walked),
            "reference": row(ref) if ref is not None else None, "detail": {p: detail(i) for p, i in rec.items()},
            "reference_years": stats[(configs[ref][0], configs[ref][1].name)]["years"] if ref is not None else None,
            "walk_forward": walked, "benchmarks": bench,
            "pbo": {"mean": pbo(excess), "ir": pbo(excess, criterion="ir")},
            "reality": {"zero": reality_check(excess, dates),
                        **({"reference": reality_check(excess, dates, base=excess[ref])} if ref is not None else {})},
            "cadence_effect": cadence_effect(configs, excess), "heatmaps": heatmaps(configs, excess),
            "by_cadence": by_cadence, "table": {p: [row(i) for i in order[p][:N_TABLE]] for p in order},
            "finalists": [label(configs[i]) for i in finalists]}


def cross_check(capital: float, stats: dict) -> dict:
    """网格里和 decade_sim 同名的规则，real 档数字必须一模一样。"""
    path = DATA / "sim10_results.json"
    if not path.is_file():
        return {"n": 0}
    sim = json.loads(path.read_text(encoding="utf-8"))
    if sim.get("capital") != capital:
        return {"n": 0}
    worst, n = 0.0, 0
    for cadence, rows in sim["results"].items():
        for name, row in rows.items():
            mine = stats.get((cadence, name.removesuffix("-small")))
            if mine is None or "real" not in row:
                continue
            a = row["real"]["all"]
            worst = max([worst, abs(row["real"]["final"] - mine["final"])]
                        + [abs(a[k] - mine[k]) for k in ("excess", "cagr", "max_dd", "beta", "alpha", "total")])
            n += 1
    return {"n": n, "max_abs_diff": worst}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=200_000.0)
    args = ap.parse_args()
    configs, book, pool, dates, stats = load(args.capital)
    out = {"capital": args.capital, **select(configs, book, pool, dates, stats),
           "cross_check": cross_check(args.capital, stats)}
    path = out_dir(args.capital) / "select.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    register("rule_select", {"capital": args.capital, "n_configs": out["n_configs"], "cadences": out["cadences"]},
             {p: out["walk_forward"][p]["excess"] for p in PICKERS})
    for p in PICKERS:
        r, w = out["recommended"][p], out["walk_forward"][p]
        print(f"{p:7s} 全期推荐 {r['cadence']} {r['name']:22s} 超额 {r['excess']:+6.1f}  滚动样本外 {w['excess']:+6.1f} "
              f"{w['ci']}", flush=True)
    print(f"PBO 按超额 {out['pbo']['mean']['pbo']} 按信息比率 {out['pbo']['ir']['pbo']}  "
          f"现实检验 p {out['reality']['zero']['p']}  核对 {out['cross_check']} -> {path}", flush=True)


if __name__ == "__main__":
    main()
