"""把一条规则的逐年、逐月收益和资金占用摊开：收益从 real_*.npz 直接算，持仓/现金要重跑一次模拟。

用法：python -X utf8 detail_rule.py [--capital 200000] [--cadence M] [--name DB-K30-b8-o1o5]
输出：rules/<本金>/detail_<规则>.md
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import duckdb

from common import DATA, DB, load_market10, signal_range
from decade_sim import scoring_days
from engine import candidates
from rule_grid import out_dir, parse
from rule_search import cost_model
from sim import YEAR, simulate

HERE = Path(__file__).resolve().parent


def load_series(capital: float, cadence: str, name: str) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, float]:
    """逐日收益、准入池、日期，以及全期期末净值（npz 的最终值，含计分前 5 天的建仓期收益）。"""
    d = out_dir(capital)
    z = np.load(d / f"real_{cadence}.npz")
    names = [str(n) for n in z["names"]]
    if name not in names:
        raise SystemExit(f"{cadence} 里没有 {name}")
    final = float(json.loads((d / f"real_{cadence}.json").read_text(encoding="utf-8"))[name]["final"])
    return z["ret"][names.index(name)].astype(float), z["pool"].astype(float), pd.DatetimeIndex(z["dates"]), final


INDEXES = {"000001.SH": "上证指数", "000300.SH": "沪深300", "000905.SH": "中证500", "000852.SH": "中证1000"}


def index_returns(dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """同期大盘的逐日收益，和计分日对齐（用指数自己的日历，首日的前收也在）。"""
    con = duckdb.connect(str(DB), read_only=True)
    try:
        raw = con.execute("SELECT ts_code, CAST(trade_date AS TIMESTAMP) AS trade_date, close FROM index_daily").df()
    finally:
        con.close()
    out = {}
    for code, name in INDEXES.items():
        c = raw[raw.ts_code == code].set_index("trade_date").close.sort_index()
        c = c[~c.index.duplicated()]
        sub = c.reindex(dates).ffill().to_numpy()
        before = c[c.index < dates[0]]
        prev = float(before.iloc[-1]) if len(before) else float(sub[0])
        r = np.empty(len(sub))
        r[0] = sub[0] / prev - 1
        r[1:] = sub[1:] / sub[:-1] - 1
        out[name] = r
    return out


def nav_of(ret: np.ndarray, final: float) -> np.ndarray:
    """以期末净值为锚往回推：nav[k] = final / ∏(1+ret[k+1:])，这样期末和统计里的 final 对得上。"""
    tail = np.concatenate([np.cumprod((1 + ret)[::-1])[::-1][1:], [1.0]])
    return final / tail


def by_month(dates: pd.DatetimeIndex) -> pd.PeriodIndex:
    return dates.to_period("M")


def table(ret: np.ndarray, pool: np.ndarray, dates: pd.DatetimeIndex, nav: np.ndarray, freq: str,
          extra: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """按年或按月汇总：组合收益、准入池收益、超额（逐日差之和，百分点）、期末净值、组内最大回撤，外加同期指数。"""
    key = dates.year if freq == "Y" else by_month(dates)
    rows = []
    for k in pd.unique(key):
        sel = np.flatnonzero(key == k)
        r, p, n = ret[sel], pool[sel], nav[sel]
        seg = np.concatenate([[nav[sel[0] - 1] if sel[0] > 0 else n[0] / (1 + r[0])], n])
        dd = float((seg / np.maximum.accumulate(seg) - 1).min()) * 100
        row = {"key": k, "ret": (np.prod(1 + r) - 1) * 100, "pool": (np.prod(1 + p) - 1) * 100,
               "excess": float((r - p).sum()) * 100, "nav": float(n[-1]), "dd": dd, "days": len(sel)}
        for name, ir in (extra or {}).items():
            row[name] = (np.prod(1 + ir[sel]) - 1) * 100
        rows.append(row)
    return pd.DataFrame(rows)


def money(x: float) -> str:
    return f"{x / 10000:,.1f} 万" if x >= 10000 else f"{x:,.0f}"


def run_sim(capital: float, cadence: str, name: str, dates_ref: pd.DatetimeIndex):
    """重跑这一条规则，拿到每天的持仓只数、现金、净值；只留计分日上的。"""
    m, low, _ = load_market10(raw=True)
    dates = m.dates
    d_first, d_last = signal_range(dates)
    mf = replace(m, eligible=m.eligible & ~low)
    scores = pd.read_parquet(DATA / f"scores10_{cadence}.parquet")
    cands, rank = candidates(m, scores[~low[scores.di.to_numpy(), scores.ji.to_numpy()]])
    del scores
    res = simulate(mf, cands, rank, parse(name).rule, cost_model(capital, "real"), d_first, d_last)
    idx = scoring_days(dates, d_first, d_last)
    hold = pd.DataFrame({"date": dates[idx], "nav": res.nav[idx], "cash": res.cash[idx],
                         "holdings": res.holdings[idx]})
    hold = hold[hold.date.isin(dates_ref)].reset_index(drop=True)  # 只留 npz 里的计分日，按日期对齐
    if len(hold) != len(dates_ref):
        raise SystemExit(f"重跑的计分日只有 {len(hold)} 天，npz 里有 {len(dates_ref)} 天")
    return hold


def extra_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("key", "ret", "pool", "excess", "nav", "dd", "days")]


def render(capital: float, cadence: str, name: str, yearly: pd.DataFrame, monthly: pd.DataFrame,
           hold: pd.DataFrame | None) -> str:
    spec = parse(name)
    cols = extra_cols(yearly)
    out = [f"# {cadence} {name}（本金 {capital / 10000:g} 万）", "",
           f"规则：{spec.plain}。收益已扣成本（佣金万五最低 5 元、过户费十万分之一、印花税 0.05%），一手取整，T+1 开盘成交。",
           "超额 = 每天（组合 − 准入池）收益之和，单位是百分点；年化超额 = 均值 × 242。指数是收盘价、不含分红。", ""]
    head = "| 年份 | 组合收益 | " + "".join(f"{c} | " for c in cols) + "准入池 | 超额 | 年内最大回撤 | 年末净值 | 年末持仓 | 年末现金占比 |"
    out += ["## 逐年", "", head, "|" + "---|" * (head.count("|") - 1)]
    for _, r in yearly.iterrows():
        y = int(r["key"])
        line = f"| {y} | {r['ret']:+.1f}% | " + "".join(f"{r[c]:+.1f}% | " for c in cols) +                f"{r['pool']:+.1f}% | {r['excess']:+.1f} | {r['dd']:.1f}% | {money(r['nav'])} |"
        if hold is not None:
            h = hold[hold.date.dt.year == y].iloc[-1]
            line += f" {int(h.holdings)} 只 | {h.cash / h.nav * 100:.1f}% |"
        else:
            line += " — | — |"
        out.append(line)
    out.append("")
    if hold is not None:
        out += ["## 资金占用（按年）", "",
                "| 年份 | 平均持仓只数 | 平均股票仓位 | 平均现金 | 最少/最多持仓 |", "|---|---|---|---|---|"]
        for y, g in hold.groupby(hold.date.dt.year):
            out.append(f"| {y} | {g.holdings.mean():.1f} 只 | {(1 - g.cash / g.nav).mean() * 100:.1f}% | "
                       f"{(g.cash / g.nav).mean() * 100:.1f}% | {int(g.holdings.min())}/{int(g.holdings.max())} |")
        last = hold.iloc[-1]
        out += ["", f"**末期（{last.date:%Y-%m-%d}）**：净值 {money(last.nav)}，持仓 {int(last.holdings)} 只、"
                    f"市值 {money(last.nav - last.cash)}（{((last.nav - last.cash) / last.nav) * 100:.1f}%），"
                    f"现金 {money(last.cash)}（{last.cash / last.nav * 100:.1f}%），每只平均 {money((last.nav - last.cash) / max(last.holdings, 1))}。", ""]
    head = "| 月份 | 组合收益 | " + "".join(f"{c} | " for c in cols) + "准入池 | 超额 | 月末净值 | 月末持仓 | 月末现金占比 |"
    out += ["## 逐月", "", head, "|" + "---|" * (head.count("|") - 1)]
    for _, r in monthly.iterrows():
        k = r["key"]
        line = f"| {k.year}-{k.month:02d} | {r['ret']:+.1f}% | " + "".join(f"{r[c]:+.1f}% | " for c in cols) +                f"{r['pool']:+.1f}% | {r['excess']:+.1f} | {money(r['nav'])} |"
        if hold is not None:
            h = hold[hold.date.dt.to_period("M") == k].iloc[-1]
            line += f" {int(h.holdings)} 只 | {h.cash / h.nav * 100:.1f}% |"
        else:
            line += " — | — |"
        out.append(line)
    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=200_000.0)
    ap.add_argument("--cadence", default="M")
    ap.add_argument("--name", default="DB-K30-b8-o1o5")
    ap.add_argument("--no-sim", action="store_true")
    args = ap.parse_args()
    ret, pool, dates, final = load_series(args.capital, args.cadence, args.name)
    nav = nav_of(ret, final)
    print(f"{len(ret)} 天 {dates[0].date()} ~ {dates[-1].date()}，净值 {money(nav[-1])}", flush=True)
    extra = index_returns(dates)
    yearly = table(ret, pool, dates, nav, "Y", extra)
    monthly = table(ret, pool, dates, nav, "M", extra)
    print(yearly.to_string(index=False, float_format=lambda x: f"{x:,.1f}"), flush=True)
    hold = None if args.no_sim else run_sim(args.capital, args.cadence, args.name, dates)
    text = render(args.capital, args.cadence, args.name, yearly, monthly, hold)
    path = out_dir(args.capital) / f"detail_{args.name}.md"
    path.write_text(text, encoding="utf-8")
    if hold is not None:
        print(hold.groupby(hold.date.dt.year)[["holdings"]].mean().to_string(), flush=True)
        print(f"末期 {hold.iloc[-1].date.date()}: 持仓 {int(hold.iloc[-1].holdings)} 只，现金 {money(float(hold.iloc[-1].cash))}", flush=True)
    print(f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
