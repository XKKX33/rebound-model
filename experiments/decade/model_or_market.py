"""2026 年反弹模型表现差，是模型的问题还是市场的问题（只读诊断，不改任何结果）。

1. 账户收益 = 准入池（市场给的）+ 超额（选股给的）。五套推荐规则逐年拆开，看 2026 在十二年里排第几；同期指数。
2. 选股能力：每个信号日打分和 T+1 开 → T+5 开收益（carry，训练标签的口径）的秩相关（IC），
   前 6 / 前 30 名减当天全池平均（百分点），全池收益的标准差（当天机会多少），前 30 名超额 ÷ 标准差。
3. 风格还是模型：7 个简单风格（近 5/20/60/120 日涨幅、离 60 日高点、20 日波动、20 日涨停次数）各自的 IC、
   打分和它们的相关；打分名次每天对 7 个风格名次回归，拆成"风格那部分"和"剩下的"，各自的 IC；前 30 名超额也这样拆。
4. 是不是模型旧了：半年 / 季 / 月重训比每年重训，2026 年多出来的是不是比平时多；只用前 2 年数据的每年重训（W2，限价版才有）。
每项按月汇总，2026 对 2015–2025：均值、Welch t、十二年里排第几（1 = 最小）、按 2015–2025 的趋势外推的 2026。
前 K 名不按行业限 2 只，量的是排序本身，不是账户。DECADE_DATA 指到 nocap 就是不限价版。输出 DATA/model_or_market.json。
用法：python -X utf8 model_or_market.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

from common import DATA, PANEL, axes, load_market10, row_index, signal_range
from engine import BASELINE, Market, trades
from industry_check import day_corr

HERE = Path(__file__).resolve().parent
BOOKS = {"20万推荐（限价，每月 DB-K30-b8）": (HERE, "200000", "M", "DB-K30-b8-o1o5"),
         "20万推荐（不限价，每季 DB-K30-b4）": (HERE / "nocap", "200000", "Q", "DB-K30-b4-o1o5"),
         "2万推荐（限价，每月 DB-K3-b16）": (HERE, "20000", "M", "DB-K3-b16-o1o5"),
         "2万正式系统推荐（不限价，每月 DB-K2-b6）": (HERE / "nocap", "20000", "M", "DB-K2-b6-o1o5"),
         "原方案（限价，每年 FH-K6）": (HERE, "200000", "Y", "FH-K6-next-o1o5")}
INDEX = {"000300.SH": "沪深300", "000905.SH": "中证500", "000852.SH": "中证1000"}
STYLES = ("r5", "r20", "r60", "r120", "from_high60", "vol20", "limit_up20")
CADENCES = ("Y", "H", "Q", "M")
DEV = (2015, 2025)
YEAR = 242
NW_LAG = 6  # 月度序列的 Newey–West 滞后
#: 存下全部月份的列（给 2026 各月在 142 个月里排名用）
MONTH_COLS = ("ic", "top30", "ic_style", "ic_other", "top30_style", "top30_other", "ic_r5", "ic_r20", "ic_r60")


def compound(x: np.ndarray) -> float:
    return float(np.prod(1.0 + x) - 1.0)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    """两个已经减过均值的向量的相关。"""
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))

def books() -> tuple[dict, pd.DatetimeIndex]:
    """五套推荐规则（真实账户、现行印花税）逐年：账户、准入池、两者之差（复利，%），日均超额年化；2026 排第几。"""
    out, when = {}, None
    for label, (root, cap, cad, rule) in BOOKS.items():
        z = np.load(root / "rules" / cap / f"real_{cad}.npz", allow_pickle=True)
        names = list(z["names"])
        ret, pool, when = np.asarray(z["ret"], float)[names.index(rule)], np.asarray(z["pool"], float), pd.DatetimeIndex(z["dates"])
        years = {}
        for y in np.unique(when.year):
            k = when.year == y
            years[str(y)] = {"book": round(compound(ret[k]) * 100, 1), "pool": round(compound(pool[k]) * 100, 1),
                             "gap": round((compound(ret[k]) - compound(pool[k])) * 100, 1),
                             "excess_ann": round(float((ret[k] - pool[k]).mean()) * YEAR * 100, 1)}
        ann = {y: v["excess_ann"] for y, v in years.items()}
        dev = [v for y, v in ann.items() if DEV[0] <= int(y) <= DEV[1]]
        out[label] = {"years": years, "rank_2026_of_12": 1 + sum(v < ann["2026"] for v in dev),
                      "dev_median_excess_ann": round(float(np.median(dev)), 1), "dev_min_excess_ann": min(dev)}
    return out, when


def indices(when: pd.DatetimeIndex) -> dict:
    """同样的计分日上各指数逐年收益（价格指数，%）。"""
    x = pd.read_parquet(HERE / "index_daily.parquet")
    x["trade_date"] = pd.to_datetime(x.trade_date.astype(str))
    out = {}
    for code, name in INDEX.items():
        r = x[x.ts_code == code].set_index("trade_date").pct_chg.div(100).reindex(when)
        out[name] = {str(y): round(compound(r[r.index.year == y].fillna(0).to_numpy()) * 100, 1) for y in np.unique(when.year)}
    return out


def pool_rows(columns: tuple[str, ...] = STYLES, amount: bool = False) -> tuple[pd.DataFrame, Market, np.ndarray | None]:
    """2015 起准入池（剔除 < 2 元）里 T+1 开 → T+5 开收益（carry）算得出的行：面板的 columns 和 ret（小数）；
    连同行情和全天成交额矩阵（amount=True 时，元）一起返回。"""
    codes, dates = axes()
    d0, _ = signal_range(dates)
    p = pq.read_table(PANEL, columns=["ts_code", "trade_date", *columns], read_dictionary=["ts_code"],
                      filters=[("eligible", "==", True), ("below2", "==", False)], use_threads=False).to_pandas()
    di, ji = row_index(p.pop("ts_code"), p.pop("trade_date"), codes, dates)
    p.insert(0, "di", di.astype(np.int32))
    p.insert(1, "ji", ji.astype(np.int32))
    p = p[p.di >= d0].reset_index(drop=True)
    m, _, money = load_market10(raw=False, amount=amount)
    p["ret"] = trades(m, p.di.to_numpy(), p.ji.to_numpy(), BASELINE, "carry")["ret"]
    return p[np.isfinite(p.ret)].reset_index(drop=True), m, money


def load_rows() -> pd.DataFrame:
    """pool_rows 的行（7 个风格、收益）加上各频率的打分。"""
    p, _, _ = pool_rows()
    d0 = int(p.di.min())
    w2 = HERE / "win" / "w2" / "scores10_Y.parquet"
    sources = {c: DATA / f"scores10_{c}.parquet" for c in CADENCES}
    if DATA.resolve() == HERE and w2.exists():
        sources["W2"] = w2
    for tag, path in sources.items():
        s = pd.read_parquet(path).rename(columns={"score": tag})
        p = p.merge(s[s.di >= d0], on=["di", "ji"], how="left")
    return p.sort_values(["di", "ji"], kind="mergesort").reset_index(drop=True)


def ranks(p: pd.DataFrame, cads: list[str]) -> pd.DataFrame:
    """每天的百分位名次（0~1）。风格缺值放中间 0.5；打分缺值留 NaN（那一行不进这个频率的统计）。"""
    g = p.groupby("di")
    rk = pd.DataFrame({c: g[c].rank(pct=True) for c in (*STYLES, "ret", *cads)})
    rk[list(STYLES)] = rk[list(STYLES)].fillna(0.5)
    return rk

def daily(p: pd.DataFrame, rk: pd.DataFrame, cad: str) -> pd.DataFrame:
    """每个信号日：IC（×100）、前 6 / 前 30 名超额（百分点）、全池收益标准差（百分点）、前 30 名超额 ÷ 标准差。"""
    ok = rk[cad].notna().to_numpy()
    di, ret = p.di.to_numpy()[ok], p.ret.to_numpy()[ok]
    out = {"ic": day_corr(di, rk[cad].to_numpy()[ok], rk["ret"].to_numpy()[ok]) * 100}
    ex = (ret - pd.Series(ret).groupby(di).transform("mean").to_numpy()) * 100
    order = pd.Series(-p[cad].to_numpy()[ok]).groupby(di).rank(method="first").to_numpy()
    for k in (6, 30):
        top = order <= k
        out[f"top{k}"] = pd.Series(ex[top]).groupby(di[top]).mean()
    out["disp"] = pd.Series(ret * 100).groupby(di).std()
    d = pd.DataFrame(out)
    d["top30_sd"] = d.top30 / d.disp
    return d


def style_split(p: pd.DataFrame, rk: pd.DataFrame, cad: str = "M") -> pd.DataFrame:
    """打分名次每天对 7 个风格名次回归：拟合部分 / 剩下的部分各自的 IC（×100），R²；
    前 30 名超额（百分点）= 风格那部分 + 剩下的（收益每天对 7 个风格名次回归，前 30 名的拟合均值 / 残差均值）。"""
    ok = rk[cad].notna().to_numpy()
    di, s, y = p.di.to_numpy()[ok], rk[cad].to_numpy()[ok], rk["ret"].to_numpy()[ok]
    x, ret = rk[list(STYLES)].to_numpy()[ok], p.ret.to_numpy()[ok] * 100
    days, first = np.unique(di, return_index=True)
    rows = {}
    for d, a, b in zip(days, first, np.append(first[1:], len(di))):
        xd = x[a:b] - x[a:b].mean(0)
        sd, yd, rd = s[a:b] - s[a:b].mean(), y[a:b] - y[a:b].mean(), ret[a:b] - ret[a:b].mean()
        fit = xd @ np.linalg.lstsq(xd, sd, rcond=None)[0]
        rfit = xd @ np.linalg.lstsq(xd, rd, rcond=None)[0]
        top = np.argsort(-s[a:b], kind="stable")[:30]
        rows[int(d)] = {"ic_style": corr(fit, yd) * 100, "ic_other": corr(sd - fit, yd) * 100,
                        "r2_style": float(fit @ fit / (sd @ sd)),
                        "top30_style": float(rfit[top].mean()), "top30_other": float((rd - rfit)[top].mean())}
    return pd.DataFrame.from_dict(rows, orient="index")


def style_view(p: pd.DataFrame, rk: pd.DataFrame, cad: str = "M") -> pd.DataFrame:
    """每天：各风格自己的 IC（风格名次 vs 收益名次，×100），打分和各风格的秩相关（×100，打分往哪个风格偏）。"""
    ok = rk[cad].notna().to_numpy()
    di, y, s = p.di.to_numpy()[ok], rk["ret"].to_numpy()[ok], rk[cad].to_numpy()[ok]
    out = {}
    for f in STYLES:
        x = rk[f].to_numpy()[ok]
        out[f"ic_{f}"] = day_corr(di, x, y) * 100
        out[f"tilt_{f}"] = day_corr(di, s, x) * 100
    return pd.DataFrame(out)


def monthly(d: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return d.groupby(dates[d.index.to_numpy()].to_period("M")).mean()

def nw_slope(t: np.ndarray, v: np.ndarray, lag: int = NW_LAG) -> tuple[float, float, float]:
    """v 对 t 的最小二乘：截距、斜率、斜率的 Newey–West t。"""
    x = np.column_stack([np.ones_like(t), t])
    beta = np.linalg.lstsq(x, v, rcond=None)[0]
    xe = x * (v - x @ beta)[:, None]
    s = xe.T @ xe
    for k in range(1, lag + 1):
        g = xe[k:].T @ xe[:-k]
        s += (1 - k / (lag + 1)) * (g + g.T)
    inv = np.linalg.inv(x.T @ x)
    return float(beta[0]), float(beta[1]), float(beta[1] / np.sqrt((inv @ s @ inv)[1, 1]))


def versus(mon: pd.DataFrame) -> dict:
    """每列月度序列：2015–2025 均值、2026 均值、差、Welch t；按年平均后 2026 在十二年里排第几（1 = 最小）；
    2015–2025 的线性趋势（每年变化、Newey–West t）和外推到 2026 的值。"""
    year = mon.index.year.to_numpy()
    t = year + (mon.index.month.to_numpy() - 0.5) / 12
    dev, now = (year >= DEV[0]) & (year <= DEV[1]), year == 2026
    out = {}
    for c in mon.columns:
        v = mon[c].to_numpy(float)
        k = dev & np.isfinite(v)
        a, b = v[k], v[now & np.isfinite(v)]
        by_year = pd.Series(v).groupby(year).mean()
        icpt, slope, t_slope = nw_slope(t[k], a)
        out[c] = {"dev": round(float(a.mean()), 3), "y2026": round(float(b.mean()), 3),
                  "diff": round(float(b.mean() - a.mean()), 3),
                  "welch_t": round(float((b.mean() - a.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))), 2),
                  "rank_2026_of_12": int(1 + (by_year.drop(2026) < by_year[2026]).sum()),
                  "trend_per_year": round(slope, 3), "trend_t": round(t_slope, 2),
                  "trend_2026": round(icpt + slope * float(t[now].mean()), 3),
                  "by_year": {str(y): round(float(x), 3) for y, x in by_year.items()}}
    return out


def main() -> None:
    psutil.Process().nice(getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10))  # 别的会话在训练，让它们先用 CPU
    start = time.time()
    _, dates = axes()
    out = {"data": str(DATA)}
    out["books"], when = books()
    out["index"] = indices(when)
    p = load_rows()
    cads = [c for c in (*CADENCES, "W2") if c in p.columns]
    rk = ranks(p, cads)
    mon = {c: monthly(daily(p, rk, c), dates) for c in cads}
    split, view = monthly(style_split(p, rk), dates), monthly(style_view(p, rk), dates)
    out["skill"], out["style_split"], out["style_view"] = versus(mon["M"]), versus(split), versus(view)
    cols = ["ic", "top6", "top30"]
    out["cadence_minus_Y"] = {c: versus(mon[c][cols] - mon["Y"][cols]) for c in cads if c != "Y"}
    out["cadence_level"] = {c: versus(mon[c][cols]) for c in cads}
    tab = pd.concat([mon["M"], split, view], axis=1)
    out["months_2026"] = {str(k): {c: round(float(v), 2) for c, v in r.items()}
                          for k, r in tab[tab.index.year == 2026].iterrows()}
    out["months"] = {c: {str(k): round(float(v), 3) for k, v in tab[c].items()} for c in MONTH_COLS}
    out["rows"], out["seconds"] = int(len(p)), round(time.time() - start)
    path = DATA / "model_or_market.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {path} ({out['seconds']}s)")


if __name__ == "__main__":
    main()
