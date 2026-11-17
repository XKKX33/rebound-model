"""前 30 名去掉风格后的超额为什么从 2019 年起变薄（只读诊断，接 model_or_market.py）。

"去掉风格后的超额"：每个信号日，T+1 开 → T+5 开收益（carry，百分点）减当天全池平均，对 7 个风格名次回归取残差；
按每月重训的打分排序，前 K 名的残差平均。和 model_or_market.json 的 top30_other 同一算法。
1. 池子变大：前 30 名（固定只数）vs 按准入池比例取前 1% / 2.5% / 5% / 10% / 20%，另看后 2.5% / 10%。
2. 衰减集中在哪：按价格（当天池内三分位，未复权收盘）、成交额（近 20 个开市日的日均，三分位）、
   板块（沪主板 600/601、沪主板 603/605、深主板 000/001/003、原中小板 002）、行业分组。
   前 2.5% 的平均残差 = Σ组 [该组占前 2.5% 的比例 × 组平均残差]（组整体那部分）
                     + Σ组 [该组入选票比本组平均多出来的 ÷ 前 2.5% 只数]（组内挑票那部分），两段时期的变化按组拆开。
   另报每组单独排名时组内前 5% 比本组平均多多少（组内的挑票能力，不受这个组占前 K 名多少的影响）。
3. 隔夜：信号日收盘 → T+1 开盘，同样减全池平均、去掉风格，前 K 名的平均。只在 T+1 开盘买得进的那批行上算。
时段 2015–18、2019–25（2019 这个界是上一轮看了逐年结果才划的），2026 另报；两段的差用月度均值做 Welch t。
DECADE_DATA 指到 nocap 就是不限价版。输出 DATA/top30_decay.json。用法：python -X utf8 top30_decay.py
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
import psutil

from common import DATA, axes
from model_or_market import STYLES, pool_rows

FRACS = (0.01, 0.025, 0.05, 0.10, 0.20)
PICK = 0.025  # 分组拆账用的前 K 名比例：固定比例，免得池子大小掺进来
WITHIN = 0.05  # 组内单独排名时取前 5%
PERIODS = {"2015-2018": (2015, 2018), "2019-2025": (2019, 2025), "2026": (2026, 2026)}
EARLY, LATE = "2015-2018", "2019-2025"
BOARDS = (("沪主板600/601", ("600", "601")), ("沪主板603/605", ("603", "605")),
          ("深主板000/001/003", ("000", "001", "003")), ("原中小板002", ("002",)))
TERCILES = ("低", "中", "高")
AMOUNT_DAYS = 20


def board_of(codes: np.ndarray) -> np.ndarray:
    """代码前三位 → BOARDS 的下标；都不是的记 len(BOARDS)（"其他"，例如误入主板池的创业板新代码 302132）。"""
    out = np.full(len(codes), len(BOARDS), dtype=np.int64)
    head = np.array([c[:3] for c in codes])
    for k, (_, heads) in enumerate(BOARDS):
        out[np.isin(head, heads)] = k
    return out

def trailing_mean_at(x: np.ndarray, n: int, di: np.ndarray, ji: np.ndarray) -> np.ndarray:
    """日 × 股矩阵 x 在 (di, ji) 这些格子上：往前 n 个开市日（含当天）里有值的那些天的平均。"""
    ok = np.isfinite(x)
    s = np.cumsum(np.where(ok, x, 0.0), axis=0)
    c = np.cumsum(ok, axis=0, dtype=np.float64)
    lo = di - n  # 窗口开始的前一天

    def window(cum: np.ndarray) -> np.ndarray:
        return cum[di, ji] - np.where(lo >= 0, cum[np.maximum(lo, 0), ji], 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        return window(s) / window(c)


def load() -> tuple[pd.DataFrame, int]:
    """分析用的行（按 di, ji 排好）：7 个风格名次、收益和隔夜（百分点）、每月重训的打分、
    价格 / 成交额三分位（0 = 低）、板块、行业。另返回行业个数。"""
    p, m, money = pool_rows((*STYLES, "raw_c"), amount=True)
    di, ji = p.di.to_numpy(), p.ji.to_numpy()
    p["amt20"] = trailing_mean_at(money, AMOUNT_DAYS, di, ji)
    del money
    p["night"] = (m.price["open"][di + 1, ji] / m.price["close"][di, ji] - 1) * 100
    p["ret"] *= 100
    p["board"] = board_of(m.codes)[ji]
    p["ind"] = m.industry[ji].astype(np.int64)
    n_ind = int(m.industry.max()) + 1
    del m
    p = p.merge(pd.read_parquet(DATA / "scores10_M.parquet"), on=["di", "ji"])
    p = p.sort_values(["di", "ji"], kind="mergesort").reset_index(drop=True)
    g = p.groupby("di")
    rk = {f: g[f].rank(pct=True).fillna(0.5) for f in (*STYLES, "raw_c", "amt20")}
    for f in STYLES:
        p[f] = rk[f]
    for col, name in (("raw_c", "price_t"), ("amt20", "amount_t")):
        p[name] = np.clip(np.ceil(rk[col].to_numpy() * 3) - 1, 0, 2).astype(np.int64)
    return p, n_ind


def resid(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """v 减均值后对 x（已减均值）回归的残差。"""
    v = v - v.mean()
    return v - x @ np.linalg.lstsq(x, v, rcond=None)[0]


def size(q: float, n: int) -> int:
    return max(1, int(round(q * n)))


def lab(q: float) -> str:
    return f"{q * 100:g}%"


def group_stats(order: np.ndarray, k: int, grp: np.ndarray, e: np.ndarray, sc: np.ndarray, n_grp: int) -> np.ndarray:
    """一天、一种分组：每组 [池里只数, 残差和, 前 k 名里的只数, 前 k 名的残差和, 组内前 5% 的只数, 组内前 5% 的残差和]。"""
    out = np.zeros((n_grp, 6))
    out[:, 0] = np.bincount(grp, minlength=n_grp)
    out[:, 1] = np.bincount(grp, weights=e, minlength=n_grp)
    top = order[:k]
    out[:, 2] = np.bincount(grp[top], minlength=n_grp)
    out[:, 3] = np.bincount(grp[top], weights=e[top], minlength=n_grp)
    by = np.lexsort((-sc, grp))  # 先按组，组内按分数从高到低
    pos = np.arange(len(by)) - np.searchsorted(grp[by], np.arange(n_grp))[grp[by]]
    keep = by[pos < np.maximum(1, np.round(WITHIN * out[grp[by], 0]))]
    out[:, 4] = np.bincount(grp[keep], minlength=n_grp)
    out[:, 5] = np.bincount(grp[keep], weights=e[keep], minlength=n_grp)
    return out

def per_day(p: pd.DataFrame, n_ind: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """每个信号日：各种前 K / 后 K 名的平均残差（百分点），隔夜的同样几项；每种分组的 group_stats（日 × 组 × 6）。"""
    di = p.di.to_numpy()
    days, first = np.unique(di, return_index=True)
    ends = np.append(first[1:], len(di))
    x_all = p[list(STYLES)].to_numpy(np.float64)
    ret, night, sc = p.ret.to_numpy(np.float64), p.night.to_numpy(np.float64), p.score.to_numpy(np.float64)
    grps = {"price_t": (p.price_t.to_numpy(), 3), "amount_t": (p.amount_t.to_numpy(), 3),
            "board": (p.board.to_numpy(), len(BOARDS) + 1), "ind": (p.ind.to_numpy(), n_ind)}
    cubes = {name: np.zeros((len(days), n, 6)) for name, (_, n) in grps.items()}
    rows = []
    for t, (a, b) in enumerate(zip(first, ends)):
        x = x_all[a:b] - x_all[a:b].mean(0)
        e, en = resid(x, ret[a:b]), resid(x, night[a:b])
        raw, nraw = ret[a:b] - ret[a:b].mean(), night[a:b] - night[a:b].mean()
        order, n = np.argsort(-sc[a:b], kind="stable"), b - a
        k = size(PICK, n)
        row = {"n": n, "top30": e[order[:30]].mean(), "raw_top30": raw[order[:30]].mean(),
               "night_top30": en[order[:30]].mean(), "night_raw_top30": nraw[order[:30]].mean(),
               "night_pick": en[order[:k]].mean(), "night_raw_pick": nraw[order[:k]].mean()}
        for q in FRACS:
            row[f"top{lab(q)}"] = e[order[:size(q, n)]].mean()
        for q in (0.025, 0.10):
            row[f"bottom{lab(q)}"] = e[order[n - size(q, n):]].mean()
        rows.append(row)
        for name, (g, ng) in grps.items():
            cubes[name][t] = group_stats(order, k, g[a:b], e, sc[a:b], ng)
    return pd.DataFrame(rows, index=days), cubes


def masks(years: np.ndarray) -> dict[str, np.ndarray]:
    return {name: (years >= a) & (years <= b) for name, (a, b) in PERIODS.items()}


def welch(v: pd.Series) -> dict:
    """日度序列（日期索引）→ 月均值；2019–25 减 2015–18 的差和 Welch t。"""
    mon = v.groupby(v.index.to_period("M")).mean().dropna()
    yr = mon.index.year
    a = mon[(yr >= PERIODS[EARLY][0]) & (yr <= PERIODS[EARLY][1])]
    b = mon[(yr >= PERIODS[LATE][0]) & (yr <= PERIODS[LATE][1])]
    return {"diff": b.mean() - a.mean(), "t": (b.mean() - a.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))}


def curves(d: pd.DataFrame) -> dict:
    """每列：各时段均值、逐年均值、两段的差（月度 Welch t）。d 是日期索引。"""
    m = masks(d.index.year.to_numpy())
    return {c: {"periods": {k: d[c][v].mean() for k, v in m.items()},
                "years": d[c].groupby(d.index.year).mean().to_dict(), "change": welch(d[c])} for c in d.columns}

def decompose(cube: np.ndarray, when: pd.DatetimeIndex, names: tuple[str, ...] | None) -> dict:
    """一种分组的拆账（names=None 只报合计，用于行业）。每组：池里占比、前 2.5% 里占比、组平均残差、
    组整体那部分 seg = 前 2.5% 占比 × 组平均残差、组内挑票那部分 sel = 入选票比本组平均多出的 ÷ 前 2.5% 只数、
    入选票平均比本组多多少、组内前 5% 比本组平均多多少；两段的变化（月度 Welch t）。合计里 seg + sel = 前 2.5% 平均残差。"""
    n, s, k, ks, kw, kws = (cube[..., i] for i in range(6))
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = s / n
        mu0 = np.where(n > 0, mu, 0.0)
        share, weight = n / n.sum(1, keepdims=True), k / k.sum(1, keepdims=True)
        seg, sel = weight * mu0, (ks - k * mu0) / k.sum(1, keepdims=True)
    m = masks(when.year.to_numpy())

    def stat(cols: slice | list[int]) -> dict:
        def ratio(num: np.ndarray, den: np.ndarray, rows: np.ndarray) -> float:
            d = den[rows][:, cols].sum()
            return float(num[rows][:, cols].sum() / d) if d else float("nan")
        per = {}
        for label, rows in m.items():
            per[label] = {"pool_share": share[rows][:, cols].sum(1).mean(), "pick_share": weight[rows][:, cols].sum(1).mean(),
                          "mu": np.nanmean(mu[rows][:, cols]) if not isinstance(cols, slice) else float("nan"),
                          "seg": seg[rows][:, cols].sum(1).mean(), "sel": sel[rows][:, cols].sum(1).mean(),
                          "pick_edge": ratio(ks - k * mu0, k, rows), "within5_edge": ratio(kws - kw * mu0, kw, rows)}
        change = {f: welch(pd.Series(v[:, cols].sum(1), index=when)) for f, v in (("seg", seg), ("sel", sel))}
        return {"periods": per, "change": change}

    out = {"合计": stat(slice(None))}
    for g, name in enumerate(names or ()):
        out[name] = stat([g])
        yr = when.year.to_numpy()
        out[name]["years"] = {int(y): {"pool_share": share[yr == y, g].mean(), "pick_share": weight[yr == y, g].mean(),
                                       "mu": np.nanmean(mu[yr == y, g]), "sel": sel[yr == y, g].mean()}
                              for y in np.unique(yr)}
    return out


def tercile_values(p: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    """价格 / 成交额三分位在各时段的中位数（元、亿元），给"低 / 中 / 高"一个实际的量级。"""
    m = masks(dates[p.di.to_numpy()].year.to_numpy())
    return {col: {label: {TERCILES[t]: float(np.median(p[val].to_numpy()[rows & (p[col].to_numpy() == t)]) * scale)
                          for t in range(3)} for label, rows in m.items()}
            for col, val, scale in (("price_t", "raw_c", 1.0), ("amount_t", "amt20", 1e-8))}


def checks(d: pd.DataFrame, cubes: dict[str, np.ndarray]) -> dict:
    """拆账恒等式（每种分组 Σ seg + sel = 前 2.5% 平均残差，逐日最大误差）；前 30 名和 model_or_market.json 的逐月对照。"""
    out = {}
    for name, cube in cubes.items():
        n, s, k, ks = (cube[..., i] for i in range(4))
        with np.errstate(invalid="ignore", divide="ignore"):
            mu0 = np.where(n > 0, s / n, 0.0)
        total = (k * mu0).sum(1) / k.sum(1) + (ks - k * mu0).sum(1) / k.sum(1)
        out[f"identity_{name}"] = float(np.abs(total - d[f"top{lab(PICK)}"].to_numpy()).max())
    ref = DATA / "model_or_market.json"
    if ref.exists():
        months = json.loads(ref.read_text(encoding="utf-8"))["months"]
        mon = d.groupby(d.index.to_period("M")).mean()
        for mine, theirs in (("top30", "top30_other"), ("raw_top30", "top30")):
            out[f"vs_model_or_market_{mine}"] = max(abs(mon[mine][pd.Period(k)] - v) for k, v in months[theirs].items())
    return out


def plain(x):
    """numpy / pandas 的数和键 → JSON 能写的（浮点留 4 位）。"""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), 4)
    return x


def main() -> None:
    psutil.Process().nice(getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10))  # 别的会话在训练，让它们先用 CPU
    warnings.filterwarnings("ignore", "Mean of empty slice")  # 某些时段某组（如"其他"）一只都没有
    start = time.time()
    _, dates = axes()
    p, n_ind = load()
    if not (np.isfinite(p.night).all() and np.isfinite(p.amt20).all()):
        raise SystemExit("隔夜收益或近 20 日成交额有缺值：先查行情矩阵")
    d, cubes = per_day(p, n_ind)
    d.index = dates[d.index.to_numpy()]
    names = {"price_t": tuple(f"价格{t}" for t in TERCILES), "amount_t": tuple(f"成交额{t}" for t in TERCILES),
             "board": (*(b for b, _ in BOARDS), "其他"), "ind": None}
    out = {"data": str(DATA), "rows": len(p), "pick": PICK, "within": WITHIN,
           "pool": {"n": d.n.groupby(d.index.year).mean().to_dict(),
                    "k_pick": d.n.map(lambda n: size(PICK, n)).groupby(d.index.year).mean().to_dict()},
           "curves": curves(d.drop(columns="n")),
           "groups": {name: decompose(cubes[name], d.index, names[name]) for name in cubes},
           "terciles": tercile_values(p, dates), "checks": checks(d, cubes)}
    out["seconds"] = round(time.time() - start)
    path = DATA / "top30_decay.json"
    path.write_text(json.dumps(plain(out), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {path} ({out['seconds']}s)")


if __name__ == "__main__":
    main()
