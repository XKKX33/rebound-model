"""2026 年 7 月三套规则比准入池多亏 7~12 个点，原因在哪（诊断，只读，不改任何结果）。

1. 历史对照：三套规则 2015–2026 每月"组合 − 准入池"，7 月排第几；准入池下跌的月份平时差多少。
2. 7 月拆账：按规则搜索同一个账户（2015 年起）重新模拟，把每笔持仓每天的盈亏加起来，和净值逐日核对；
   按天、按票、按行业、按买入时间排。另报组合的 β（前一年回归）和 7 月"按 β 该亏多少"。
3. 数据核对：7 月复权收盘的日收益 vs Tushare 的 pct_chg；每天有收盘价的只数 vs 库里的行数；复权因子变动的票。
4. 打分排名：每个信号日打分前 30 名 / 31–100 / 101–300 名的 T+1 开 → T+5 开收益减全池平均（训练标签的口径），逐月；
   2026 年各月前 30 名的特征在当天池子里的百分位（近 5/20/60/120 日涨幅、离 60 日高点、波动、涨停次数、价格、成交额）。
5. 7 月准入池按 6 月底的近 60 日涨幅分五组，各组 7 月的收益，以及三套规则 7 月持仓落在哪几组。
输出 july_drop.json。用法：python -X utf8 july_drop.py
"""
from __future__ import annotations

import json
from dataclasses import replace

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import DATA, DB, PANEL, load_market10, row_index, signal_range
from engine import BASELINE, candidates, pool_daily_return, trades
from rule_grid import parse
from rule_search import cost_model
from sim import simulate
from train_cadence import no_power_throttling

JULY = (pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-31"))
BOOKS = {"20万推荐": (200000, "M", "DB-K30-b8-o1o5"), "2万推荐": (20000, "M", "DB-K3-b16-o1o5"),
         "原方案": (200000, "Y", "FH-K6-next-o1o5")}
FEATURES = ("r5", "r20", "r60", "r120", "from_high60", "vol20", "limit_up20")
BUCKETS = ((1, 30), (31, 100), (101, 300))


def month_gaps() -> dict:
    """三套规则每月"组合 − 准入池"（百分点）：7 月排第几、最差的几个月和下一个月、准入池跌超 3% 的月份的中位数。"""
    out = {}
    for label, (capital, cadence, name) in BOOKS.items():
        z = np.load(DATA / "rules" / str(capital) / f"real_{cadence}.npz")
        s = pd.DataFrame({"b": z["ret"][list(z["names"]).index(name)], "p": z["pool"]}, index=pd.DatetimeIndex(z["dates"]))
        g = s.groupby(s.index.to_period("M"))
        mon = pd.DataFrame({"b": g.b.apply(lambda x: np.prod(1 + x) - 1), "p": g.p.apply(lambda x: np.prod(1 + x) - 1)}) * 100
        gap = mon.b - mon.p
        jul = pd.Period("2026-07", "M")
        worst = gap.sort_values().head(8)
        out[label] = {"july_gap": round(float(gap[jul]), 1), "july_rank": int((gap <= gap[jul]).sum()), "months": len(gap),
                      "worst": {str(k): [round(float(v), 1), round(float(mon.p[k]), 1), round(float(gap.get(k + 1, np.nan)), 1)]
                                for k, v in worst.items()},
                      "pool_down_gap_median": round(float(gap[mon.p < -3].median()), 1), "pool_down_months": int((mon.p < -3).sum())}
    return out


def buy_value(cost: float, cm) -> float:
    """买入金额（不含费）：成本 = 金额 + max(最低佣金, 金额 × 佣金率) + 金额 × 过户费率，反解金额。"""
    hi = cost / (1 + cm.commission + cm.transfer)
    return hi if hi * cm.commission >= cm.min_commission else (cost - cm.min_commission) / (1 + cm.transfer)


def position_pnl(m, log: pd.DataFrame, cm, d0: int, d1: int) -> pd.DataFrame:
    """每笔持仓在 [d0, d1] 每天的盈亏（元），和模拟器记账一致：买入那天 = 股数 × 收盘 − 成本；
    中间 = 股数 × Δ收盘；卖出那天 = 卖出所得 − 股数 × 前一天收盘。停牌按最后一个收盘价计值。"""
    c = m.price["close"]
    rows = []
    for r in log[(log.entry_day <= d1) & (log.exit_day >= d0)].itertuples(index=False):
        j, e, x = int(r.j), int(r.entry_day), int(r.exit_day)
        shares = buy_value(r.cost_basis, cm) / r.entry_px
        val = shares * pd.Series(c[e:x + 1, j]).ffill().fillna(r.entry_px).to_numpy()
        pnl = np.diff(val, prepend=r.cost_basis)
        pnl[-1] = (1 + r.net) * r.cost_basis - val[-2]
        rows += [(j, e, x, int(r.signal_day), e + k, float(v)) for k, v in enumerate(pnl) if d0 <= e + k <= d1]
    return pd.DataFrame(rows, columns=["j", "entry_day", "exit_day", "signal_day", "day", "pnl"])


def part2(m, books: dict, pool: np.ndarray, names: np.ndarray, ind_names: np.ndarray, d0: int, d1: int) -> dict:
    """7 月逐日、逐只拆账（贡献 = 盈亏 / 6 月底净值），和净值逐日核对；β 用 2025-07 ~ 2026-06 的逐日收益回归。"""
    dates, c, out = m.dates, m.price["close"], {}
    for label, (res, cm, rank) in books.items():
        nav = res.nav
        p = position_pnl(m, res.trades, cm, d0, d1)
        days = range(d0, d1 + 1)
        daily = p.groupby("day").pnl.sum().reindex(days, fill_value=0.0).to_numpy()
        base = nav[d0 - 1]
        r_book = nav[1:] / nav[:-1] - 1
        fit = (dates[1:] >= pd.Timestamp("2025-07-01")) & (dates[1:] < JULY[0])
        beta = float(np.polyfit(pool[1:][fit], r_book[fit], 1)[0])
        by_j = p.groupby("j").agg(pnl=("pnl", "sum"), trades=("entry_day", "nunique"), first=("entry_day", "min"),
                                  sig=("signal_day", "min"))
        last = lambda j, d: pd.Series(c[max(d - 40, 0):d + 1, j]).ffill().iloc[-1]
        stock = [{"code": str(m.codes[j]), "name": names[j], "industry": ind_names[m.industry[j]],
                  "contrib": round(v.pnl / base * 100, 2), "trades": int(v.trades), "first_entry": str(dates[int(v["first"])].date()),
                  "rank_at_signal": float(rank[int(v.sig), j]),
                  "stock_july": round((last(j, d1) / last(j, d0 - 1) - 1) * 100, 1)}
                 for j, v in by_j.sort_values("pnl").iterrows()]
        held_june = p[p.entry_day < d0]
        out[label] = {"july_book": round((nav[d1] / base - 1) * 100, 1), "july_pool": round((np.prod(1 + pool[d0:d1 + 1]) - 1) * 100, 1),
                      "beta": round(beta, 2), "beta_pred": round((np.prod(1 + beta * pool[d0:d1 + 1]) - 1) * 100, 1),
                      "reconcile_max_abs": round(float(np.max(np.abs(daily - np.diff(nav[d0 - 1:d1 + 1])))), 2),
                      "daily": [[str(dates[t].date()), round(float(nav[t] / nav[t - 1] - 1) * 100, 2), round(float(pool[t]) * 100, 2)]
                                for t in days],
                      "held_from_june": round(held_june.pnl.sum() / base * 100, 1), "entered_july": round(p[p.entry_day >= d0].pnl.sum() / base * 100, 1),
                      "n_stocks": len(stock), "worst": stock[:12], "best": stock[-5:],
                      "by_industry": p.assign(g=ind_names[m.industry[p.j.to_numpy()]]).groupby("g").pnl.sum()
                      .div(base / 100).round(2).sort_values().head(8).to_dict()}
        out[label]["_positions"] = p
    return out


def part3(m, held: set[int], d0: int, d1: int) -> dict:
    """复权收盘的日收益 vs 库里的 pct_chg（池子里的票和持过的票）；每天库里的行数 vs 面板有收盘价的只数；复权因子变动。"""
    con = duckdb.connect(str(DB), read_only=True)
    try:
        lo, hi = m.dates[d0 - 1].date(), m.dates[d1].date()
        db = con.execute(f"SELECT ts_code, CAST(trade_date AS TIMESTAMP) AS t, pct_chg FROM daily "
                         f"WHERE trade_date BETWEEN DATE '{lo}' AND DATE '{hi}'").df()
        adj = con.execute(f"""SELECT ts_code, CAST(trade_date AS TIMESTAMP) AS t FROM (
            SELECT ts_code, trade_date, adj_factor, lag(adj_factor) OVER (PARTITION BY ts_code ORDER BY trade_date) AS prev
            FROM adj_factor WHERE trade_date BETWEEN DATE '{lo}' AND DATE '{hi}') WHERE prev IS NOT NULL AND adj_factor <> prev""").df()
    finally:
        con.close()
    db = db[db.ts_code.isin(m.codes)]
    di = m.dates.get_indexer(pd.DatetimeIndex(db.t).astype("datetime64[ns]"))
    ji = m.code_index(db.ts_code)
    c = m.price["close"]
    mine = (di >= d0) & (di <= d1)
    di, ji, pct = di[mine], ji[mine], db.pct_chg.to_numpy()[mine]
    want = m.eligible[di - 1, ji] | np.isin(ji, list(held))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = c[di, ji] / c[di - 1, ji] - 1
    ok = want & np.isfinite(r) & np.isfinite(pct)
    diff = np.abs(r[ok] * 100 - pct[ok])
    bad = np.flatnonzero(diff > 0.5)
    rows_db = pd.Series(di).value_counts()
    rows_panel = pd.Series({t: int(np.isfinite(c[t]).sum()) for t in range(d0, d1 + 1)})
    adj = adj[adj.ts_code.isin(m.codes)]
    aj = set(m.code_index(adj.ts_code))
    return {"compared": int(ok.sum()), "max_abs_pp": round(float(diff.max()), 3) if len(diff) else None,
            "over_0.5pp": int(len(bad)),
            "examples": [[str(m.codes[ji[ok][k]]), str(m.dates[di[ok][k]].date()), round(float(r[ok][k] * 100), 2),
                          round(float(pct[ok][k]), 2)] for k in bad[:8]],
            "rows_db_minus_panel_max": int((rows_db.reindex(rows_panel.index, fill_value=0) - rows_panel).abs().max()),
            "adj_changes": int(len(adj)), "adj_changes_held": len(aj & held)}


def part4(m, low: np.ndarray, d_first: int) -> dict:
    """每月的信号日：打分前 30 / 31–100 / 101–300 名的 T+1 开 → T+5 开收益，减当天全池平均（百分点，逐笔平均）。"""
    s = pd.read_parquet(DATA / "scores10_M.parquet")
    s = s[(s.di >= d_first) & ~low[s.di.to_numpy(), s.ji.to_numpy()]]
    s["r"] = trades(m, s.di.to_numpy(), s.ji.to_numpy(), BASELINE, "carry")["ret"]
    s = s[np.isfinite(s.r)]
    s["rank"] = s.groupby("di").score.rank(ascending=False, method="first")
    s["ex"] = (s.r - s.groupby("di").r.transform("mean")) * 100
    s["month"] = m.dates[s.di.to_numpy()].to_period("M").astype(str).to_numpy()
    out = {}
    for lo, hi in BUCKETS:
        b = s[(s["rank"] >= lo) & (s["rank"] <= hi)].groupby(["month", "di"]).ex.mean().groupby("month").mean()
        out[f"top{lo}-{hi}"] = b.round(2).to_dict()
    top = pd.Series(out["top1-30"])
    out["july_top30_rank"] = int((top <= top["2026-07"]).sum())
    out["june_top30_rank"] = int((top <= top["2026-06"]).sum())
    out["months"] = len(top)
    return out


def features_2026(m, low: np.ndarray) -> pd.DataFrame:
    """2026 年面板里准入池（去掉 < 2 元）的特征，加未复权收盘、成交额，和每月打分。"""
    cols = ["ts_code", "trade_date", *FEATURES, "raw_c", "amount", "eligible"]
    p = pq.read_table(PANEL, columns=cols, read_dictionary=["ts_code"],
                      filters=[("trade_date", ">=", pd.Timestamp("2026-01-01").to_pydatetime())]).to_pandas()
    di, ji = row_index(p.pop("ts_code"), p.pop("trade_date"), m.codes, m.dates)
    p["di"], p["ji"] = di.astype(np.int32), ji.astype(np.int32)
    p = p[p.eligible.fillna(False) & ~low[di, ji]].drop(columns="eligible")
    s = pd.read_parquet(DATA / "scores10_M.parquet")
    return p.merge(s[s.di >= p.di.min()], on=["di", "ji"])


def top30_profile(f: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    """每月：打分前 30 名的各特征在当天池子里的平均百分位（50 = 和池子一样）。"""
    keys = [*FEATURES, "raw_c", "amount"]
    pct = f.groupby("di")[keys].rank(pct=True) * 100
    top = f.groupby("di").score.rank(ascending=False, method="first") <= 30
    month = pd.Series(dates[f.di.to_numpy()].to_period("M").astype(str), index=f.index)
    return pct[top].groupby(month[top]).mean().round(0).to_dict(orient="index")


def part5(m, f: pd.DataFrame, positions: dict, d0: int, d1: int) -> dict:
    """6 月底池子按近 60 日涨幅分五组（1 = 跌得最多，5 = 涨得最多），各组 7 月的等权收益；
    三套规则 7 月的盈亏落在哪几组（按 6 月底的近 60 日涨幅，持仓不在池里的也按同一组界）。"""
    c = m.price["close"]
    june = f[f.di == d0 - 1].set_index("ji")
    edges = np.quantile(june.r60, [0.2, 0.4, 0.6, 0.8])
    last = lambda d: pd.DataFrame(c[d - 40:d + 1]).ffill().to_numpy()[-1]
    july = last(d1) / last(d0 - 1) - 1
    q_pool = np.digitize(june.r60.to_numpy(), edges) + 1
    out = {"pool": {int(q): round(float(np.nanmean(july[june.index.to_numpy()][q_pool == q])) * 100, 1) for q in range(1, 6)},
           "edges_r60": [round(float(e), 1) for e in edges]}
    r60_all = np.full(len(m.codes), np.nan)
    allj = pd.read_parquet(PANEL, columns=["ts_code", "trade_date", "r60"],
                           filters=[("trade_date", "==", m.dates[d0 - 1].to_pydatetime())])
    _, ja = row_index(pd.Categorical(allj.ts_code.astype(str)), allj.trade_date, m.codes, m.dates)
    r60_all[ja] = allj.r60.to_numpy()
    for label, (p, base) in positions.items():
        q = np.digitize(r60_all[p.j.to_numpy()], edges) + 1
        q = np.where(np.isfinite(r60_all[p.j.to_numpy()]), q, 0)
        out[label] = p.assign(q=q).groupby("q").pnl.sum().div(base / 100).round(2).to_dict()
    return out


def main() -> None:
    print(f"power throttling off: {no_power_throttling()}", flush=True)
    m, low, _ = load_market10(raw=True)
    dates = m.dates
    d_first, d_last = signal_range(dates)
    d0, d1 = int(dates.searchsorted(JULY[0])), int(dates.searchsorted(JULY[1], "right")) - 1
    mf = replace(m, eligible=m.eligible & ~low)
    pool = pool_daily_return(mf)
    con = duckdb.connect(str(DB), read_only=True)
    basic = con.execute("SELECT ts_code, name, industry FROM stock_basic").df()
    con.close()
    names = basic.set_index("ts_code").name.reindex(m.codes).fillna("").to_numpy()
    ind_names = np.array(sorted({dict(zip(basic.ts_code, basic.industry.fillna(""))).get(c, "") for c in m.codes}))
    out = {"part1": month_gaps()}
    print("part1", {k: (v["july_gap"], v["july_rank"], v["months"]) for k, v in out["part1"].items()}, flush=True)
    books, ranks = {}, {}
    for label, (capital, cadence, name) in BOOKS.items():
        if cadence not in ranks:
            s = pd.read_parquet(DATA / f"scores10_{cadence}.parquet")
            ranks[cadence] = candidates(m, s[~low[s.di.to_numpy(), s.ji.to_numpy()]])
            del s
        cands, rank = ranks[cadence]
        cm = cost_model(capital, "real")
        books[label] = (simulate(mf, cands, rank, parse(name).rule, cm, d_first, d_last), cm, rank)
    out["part2"] = part2(mf, books, pool, names, ind_names, d0, d1)
    positions = {k: (v.pop("_positions"), books[k][0].nav[d0 - 1]) for k, v in out["part2"].items()}
    del books, ranks
    held = set(int(j) for p, _ in positions.values() for j in p.j.unique())
    out["part3"] = part3(mf, held, d0, d1)
    out["part4"] = part4(mf, low, d_first)
    f = features_2026(mf, low)
    out["top30_profile"] = top30_profile(f, dates)
    out["part5"] = part5(mf, f, positions, d0, d1)
    (DATA / "july_drop.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("-> july_drop.json", flush=True)


if __name__ == "__main__":
    main()
