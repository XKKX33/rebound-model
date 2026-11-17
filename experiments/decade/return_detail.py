"""两条不限价推荐规则的逐年、逐月收益明细。
20 万：每季度重训 + DB-K30-b4；2 万：每月重训 + DB-K3-b16。
前者的日收益直接取 nocap/rules/200000/real_Q.npz；后者取 nocap/rules/20000/real_M.npz。
2 万另按 2017/2019/2021/2023 年重新起步模拟，给出各起点逐年收益（2015 年起步的账户前两年涨大后摊薄了最低佣金）。
输出 nocap/return-detail.md。用法：DECADE_DATA=nocap python -X utf8 return_detail.py [--no-fresh]
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from common import DATA, load_market10, signal_range
from engine import candidates
from rule_grid import parse
from rule_search import cost_model
from sim import simulate
from train_cadence import no_power_throttling

CASES = [
    {"title": "20 万，不限价：每季度重训 + DB-K30-b4", "path": "rules/200000/real_Q.npz",
     "name": "DB-K30-b4-o1o5", "capital": 200000.0, "cadence": "Q"},
    {"title": "2 万，不限价：每月重训 + DB-K3-b16", "path": "rules/20000/real_M.npz",
     "name": "DB-K3-b16-o1o5", "capital": 20000.0, "cadence": "M"},
]
STARTS = (2017, 2019, 2021, 2023)
PCT = 100.0


def load_series(rel: str, name: str) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    z = np.load(DATA / rel, allow_pickle=True)
    i = list(z["names"]).index(name)
    return z["ret"][i].astype(float), z["pool"].astype(float), pd.DatetimeIndex(z["dates"])


def year_table(ret: np.ndarray, pool: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    frames = []
    for y in np.unique(dates.year):
        pick = dates.year == y
        r, p = ret[pick], pool[pick]
        nav = np.cumprod(1 + r)
        frames.append({"年份": int(y), "年度收益": nav[-1] - 1, "准入池": np.prod(1 + p) - 1,
                       "超额": (nav[-1] - 1) - (np.prod(1 + p) - 1),
                       "年内回撤": float((nav / np.maximum.accumulate(nav) - 1).min()),
                       "期末净值": float((1 + ret[dates.year <= y]).prod())})
    return pd.DataFrame(frames)


def month_table(ret: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    s = pd.Series(ret, index=dates)
    m = s.groupby([dates.year, dates.month]).apply(lambda r: np.prod(1 + r) - 1).unstack()
    m.columns = [f"{c}月" for c in m.columns]
    m["全年"] = s.groupby(dates.year).apply(lambda r: np.prod(1 + r) - 1)
    return m


def pct(v: float, digits: int = 1) -> str:
    return "—" if not np.isfinite(v) else f"{v * PCT:+.{digits}f}%"


def pct_plain(v: float, digits: int = 1) -> str:
    """分数转百分数，不带正负号。"""
    return "—" if not np.isfinite(v) else f"{v * PCT:.{digits}f}%"


def yearly_md(t: pd.DataFrame, capital: float) -> str:
    lines = ["| 年份 | 年度收益 | 期末净值 | 期末资产 | 准入池 | 超额 | 年内最大回撤 |",
             "|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {int(r['年份'])} | {pct(r['年度收益'])} | {r['期末净值']:.2f}× | "
                     f"{r['期末净值'] * capital:,.0f} | {pct(r['准入池'])} | "
                     f"{pct(r['超额'])} | {pct(r['年内回撤'])} |")
    return "\n".join(lines)


def month_md(m: pd.DataFrame, half: str) -> str:
    if half == "下半年":
        cols = [c for c in m.columns if c == "全年" or int(c[:-1]) >= 7]
    else:
        cols = [c for c in m.columns if c != "全年" and int(c[:-1]) <= 6]
    lines = ["| 年份 | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for y, row in m.iterrows():
        lines.append(f"| {y} | " + " | ".join(pct(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def fresh_tables(case: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2 万按不同年份重新起步：年度收益矩阵和期末资产。"""
    no_power_throttling()
    m, low, _ = load_market10(raw=True)
    dates = m.dates
    d_first, d_last = signal_range(dates)
    mf = replace(m, eligible=m.eligible & ~low)
    scores = pd.read_parquet(DATA / f"scores10_{case['cadence']}.parquet")
    cands, rank = candidates(m, scores[~low[scores.di.to_numpy(), scores.ji.to_numpy()]])
    rule = parse(case["name"]).rule
    cost = cost_model(case["capital"], "real")
    years = range(STARTS[0], dates[-1].year + 1)
    matrix = pd.DataFrame(index=years, columns=[f"{y} 起" for y in STARTS], dtype=float)
    finals = {}
    for y0 in STARTS:
        d0 = max(int(dates.searchsorted(pd.Timestamp(y0, 1, 1))), d_first)
        nav = simulate(mf, cands, rank, rule, cost, d0, d_last).nav
        for y in years:
            if y < y0:
                continue
            e = min(int(np.flatnonzero(dates.year == y)[-1]), d_last + 1)
            prev = nav[d0] if y == y0 else nav[min(int(np.flatnonzero(dates.year == y - 1)[-1]), d_last + 1)]
            matrix.loc[y, f"{y0} 起"] = nav[e] / prev - 1
        finals[f"{y0} 起"] = float(nav[min(d_last + 1, len(nav) - 1)])
    return matrix, pd.DataFrame([finals], index=["期末资产"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fresh", action="store_true")
    args = ap.parse_args()
    out = ["# 两条不限价推荐规则的逐年、逐月收益", "",
           "口径：真实账户逐日模拟，佣金万五 + 每笔最低 5 元、过户费、一手取整、现行印花税 0.05%。",
           "年度收益按日历年内逐日复利；月度同理。2015 年 1 月从 1 月 13 日开始，2026 年只到 10 月 19 日，这两个月是残月。", ""]
    for case in CASES:
        ret, pool, dates = load_series(case["path"], case["name"])
        nav = float(np.prod(1 + ret))
        years = len(ret) / 242
        dd = float((np.cumprod(1 + ret) / np.maximum.accumulate(np.cumprod(1 + ret)) - 1).min())
        out += [f"## {case['title']}", "",
                f"- 起点 {dates[0].date()}，终点 {dates[-1].date()}，{len(ret)} 个交易日",
                f"- 累计 {pct_plain(nav - 1)}，年化 {pct_plain(nav ** (1 / years) - 1)}，"
                f"最大回撤 {pct_plain(dd)}，{case['capital']:,.0f} 元 → {case['capital'] * nav:,.0f} 元", ""]
        t = year_table(ret, pool, dates)
        out += ["### 逐年", "", yearly_md(t, case["capital"]), ""]
        m = month_table(ret, dates)
        out += ["### 逐月（上半年）", "", month_md(m, "上半年"), "",
                "### 逐月（下半年）", "", month_md(m, "下半年"), ""]
        if case["capital"] == 20000.0 and not args.no_fresh:
            fm, ff = fresh_tables(case)
            out += ["### 换个年份重新起步（2 万 + 最低 5 元，这是更接近新开户的路径）", "",
                    "| 年份 | " + " | ".join(fm.columns) + " |", "|---" * (len(fm.columns) + 1) + "|"]
            for y, row in fm.iterrows():
                out.append(f"| {y} | " + " | ".join(pct(row[c]) for c in fm.columns) + " |")
            out += ["| **期末资产** | " + " | ".join(f"{ff.loc['期末资产', c]:,.0f}" for c in fm.columns) + " |", ""]
    text = "\n".join(out)
    (DATA / "return-detail.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
