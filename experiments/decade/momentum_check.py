"""强势股行情里模型是不是系统性落后（只读诊断：不训练，不改任何结果文件）。

A. 2015-01 ~ 2026-10 每个月：上月最后一个开市日的准入池按近 60 / 20 日涨幅分五档，这个月
   "最强一档 − 最弱一档"的等权收益差（WML，百分点）、涨幅和这个月收益的秩相关（IC，×100）、
   行业能解释多少截面收益（R²，%）；再看三套规则当月"组合 − 准入池"和这些量的关系。
B. 几段行情：全 A 涨幅前 100 名在哪些板块、几只在准入池里；准入池里涨得最多的 100 只，
   模型在行情开始前给的名次；打分前 30 名在当期最热的 5 个行业里的占比 vs 准入池。
"""
from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import DATA, DB, PANEL, axes
from train_cadence import no_power_throttling, panel_key

BOOKS = {"20万推荐": (200000, "M", "DB-K30-b8-o1o5"), "2万推荐": (20000, "M", "DB-K3-b16-o1o5"),
         "原方案": (200000, "Y", "FH-K6-next-o1o5")}
WINDOWS = {"2020-07~2021-01": ("2020-07-01", "2021-01-31"), "2021-05~11 新能源": ("2021-05-01", "2021-11-30"),
           "2026-04~06 科技": ("2026-04-01", "2026-06-30")}
INDEX = {"000300.SH": "沪深300", "000905.SH": "中证500", "000852.SH": "中证1000"}
ST = r"ST|\*"
SRC = f"read_parquet('{PANEL.as_posix()}')"


def month_panel(con) -> pd.DataFrame:
    """每只股每个月：上月最后一个开市日在准入池里时的 r20 / r60，和这个月的复利收益（整月停牌算 0）。"""
    return con.execute(f"""
        WITH p AS (SELECT ts_code, trade_date, date_trunc('month', trade_date) AS m, pct_chg, r20, r60,
                          eligible AND NOT below2 AS pool FROM {SRC}),
        last_day AS (SELECT m, max(trade_date) AS d FROM p GROUP BY m),
        form AS (SELECT p.ts_code, p.m + INTERVAL 1 MONTH AS m, p.r20, p.r60 FROM p JOIN last_day l ON p.trade_date = l.d
                 WHERE p.pool AND p.r60 IS NOT NULL AND p.r20 IS NOT NULL),
        ret AS (SELECT ts_code, m, exp(sum(ln(1 + pct_chg / 100))) - 1 AS ret FROM p
                WHERE pct_chg IS NOT NULL GROUP BY ts_code, m)
        SELECT f.ts_code, f.m, f.r20, f.r60, coalesce(r.ret, 0) AS ret FROM form f LEFT JOIN ret r USING (ts_code, m)
        WHERE f.m >= TIMESTAMP '2015-01-01' AND f.m <= (SELECT max(m) FROM p) ORDER BY f.m""").df()


def regime(mp: pd.DataFrame, ind_of: dict) -> pd.DataFrame:
    mp = mp.assign(g=mp.ts_code.map(ind_of).fillna(""))
    rows = {}
    for m, x in mp.groupby("m"):
        row = {"n": len(x), "pool_eq": x.ret.mean() * 100}
        for f in ("r60", "r20"):
            q = pd.qcut(x[f].rank(method="first"), 5, labels=False)
            row["wml" + f[1:]] = (x.ret[q == 4].mean() - x.ret[q == 0].mean()) * 100
            row["ic" + f[1:]] = x[f].rank().corr(x.ret.rank()) * 100
        dev = x.ret - x.ret.mean()
        row["ind_r2"] = float(((x.groupby("g").ret.transform("mean") - x.ret.mean()) ** 2).sum() / (dev ** 2).sum() * 100)
        rows[pd.Timestamp(m).to_period("M")] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def book_gaps() -> pd.DataFrame:
    """三套规则每月"组合 − 准入池"（百分点），和准入池当月收益（%）。"""
    cols = {}
    for label, (capital, cadence, name) in BOOKS.items():
        z = np.load(DATA / "rules" / str(capital) / f"real_{cadence}.npz")
        s = pd.DataFrame({"b": z["ret"][list(z["names"]).index(name)], "p": z["pool"]}, index=pd.DatetimeIndex(z["dates"]))
        mon = (1 + s).groupby(s.index.to_period("M")).prod() - 1
        cols[label] = (mon.b - mon.p) * 100
        cols.setdefault("pool", mon.p * 100)
    return pd.DataFrame(cols)


def ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([np.ones(len(y)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    se = np.sqrt(np.diag(e @ e / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)))
    return beta, beta / se


def part_a(reg: pd.DataFrame, gaps: pd.DataFrame) -> dict:
    df = reg.join(gaps, how="inner")
    books = list(BOOKS)
    cut = pd.qcut(df.wml60, 3, labels=["反转月", "中间", "强势股月"])
    out = {"months": len(df), "wml60_cuts": [round(float(v), 1) for v in df.wml60.quantile([1 / 3, 2 / 3])],
           "mean": {c: round(float(df[c].mean()), 2) for c in ("wml60", "wml20", "ic60", "ic20", "ind_r2")},
           "wml60_pos%": round(float((df.wml60 > 0).mean() * 100)), "corr": {}, "by_wml60": {}, "ols": {}}
    for b in books:
        out["corr"][b] = {k: round(float(df[b].corr(df[k])), 2) for k in ("wml60", "wml20", "ic60", "ic20", "ind_r2", "pool")}
        out["by_wml60"][b] = {str(k): {"mean": round(float(v.mean()), 2), "neg%": round(float((v < 0).mean() * 100)),
                                        "sum": round(float(v.sum()), 1)} for k, v in df.groupby(cut, observed=True)[b]}
        beta, t = ols(df[b].to_numpy(), df[["wml60", "pool"]].to_numpy())
        out["ols"][b] = {n: [round(float(c), 3), round(float(s), 1)] for n, c, s in zip(("const", "wml60", "pool"), beta, t)}
    show = ["wml60", "ic60", "wml20", "ind_r2", "pool", *books]

    def fmt(x: pd.DataFrame) -> dict:
        return {str(k): {c: round(float(v[c]), 1) for c in show} for k, v in x.iterrows()}

    out["top_wml60"] = fmt(df.sort_values("wml60", ascending=False).head(15))
    out["m2021"] = fmt(df[df.index.year == 2021])
    out["m2026"] = fmt(df[df.index.year == 2026])
    out["year"] = {str(y): {**{c: round(float(g[c].mean()), 1) for c in ("wml60", "ic60", "ind_r2")},
                            **{c: round(float(g[c].sum()), 1) for c in books}} for y, g in df.groupby(df.index.year)}
    return out


def board(code: str, px: float) -> str:
    if code.endswith(".BJ"):
        return "北交所"
    if code[:3] in ("300", "301", "302"):
        return "创业板"
    if code[:3] in ("688", "689"):
        return "科创板"
    return "主板≥70元" if px >= 70 else "主板<70元"


def leaders(con, a: pd.Timestamp, b: pd.Timestamp, n: int = 100) -> pd.DataFrame:
    """全 A（开始那天不是 ST、上市满 120 天、窗口里交易满八成）窗口复利收益前 n 名。"""
    return con.execute(f"""
        WITH d AS (SELECT ts_code, CAST(trade_date AS TIMESTAMP) AS trade_date, pct_chg, pre_close FROM daily
                   WHERE trade_date BETWEEN DATE '{a.date()}' AND DATE '{b.date()}' AND pct_chg IS NOT NULL),
        days AS (SELECT count(DISTINCT trade_date) AS n FROM d),
        w AS (SELECT ts_code, exp(sum(ln(1 + pct_chg / 100))) - 1 AS ret, count(*) AS k,
                     arg_min(pre_close, trade_date) AS px0 FROM d GROUP BY ts_code),
        names AS (SELECT ts_code, name, CAST(start_date AS TIMESTAMP) AS s FROM namechange WHERE start_date IS NOT NULL),
        snap AS (SELECT ts_code, TIMESTAMP '{a.date()}' AS t FROM w),
        nm AS (SELECT snap.ts_code, n.name AS name_then FROM snap ASOF LEFT JOIN names n ON snap.ts_code = n.ts_code AND snap.t >= n.s)
        SELECT w.ts_code, b.name AS nm, coalesce(b.industry, '') AS industry, w.ret, w.px0 FROM w
        JOIN stock_basic b USING (ts_code) LEFT JOIN nm USING (ts_code)
        WHERE w.k >= 0.8 * (SELECT n FROM days) AND CAST(b.list_date AS TIMESTAMP) <= TIMESTAMP '{a.date()}' - INTERVAL 120 DAY
          AND NOT regexp_matches(coalesce(nm.name_then, b.name, ''), '{ST}')
        ORDER BY w.ret DESC LIMIT {n}""").df()


def counts(s: pd.Series) -> dict:
    return {str(k): int(v) for k, v in s.items()}


def score_view(lab: pd.DataFrame, cad: str, f0: int, d0: int, d1: int, lead: np.ndarray, g: np.ndarray, hot) -> dict:
    """模型怎么看准入池里这段涨得最多的 100 只：开始前一天的名次（百分位，100 = 最看好），期间进前 30 的天数；
    以及打分前 30 名里落在最热 5 个行业的占比。"""
    sc = pq.read_table(DATA / f"scores10_{cad}.parquet", filters=[("di", ">=", f0), ("di", "<=", d1)]).to_pandas()
    s = lab.merge(sc, on=["di", "ji"])
    s["pct"] = s.groupby("di").score.rank(pct=True) * 100
    s["rk"] = s.groupby("di").score.rank(ascending=False, method="first")
    s0 = s[s.di == f0].set_index("ji")
    win = s[s.di >= d0]
    lw = win[win.ji.isin(lead)]
    top30 = win[win.rk <= 30]
    n_pool = float(win.groupby("di").size().mean())
    return {"leaders_pct_before": round(float(s0.pct.reindex(lead).mean()), 1),
            "leaders_top240_before%": round(float((s0.rk.reindex(lead) <= 240).mean() * 100), 1),
            "base_top240%": round(240 / len(s0) * 100, 1),
            "leaders_pct_during": round(float(lw.pct.mean()), 1),
            "leaders_top30_days%": round(float((lw.rk <= 30).mean() * 100), 2),
            "base_top30_days%": round(30 / n_pool * 100, 2),
            "top30_hot%": round(float(np.isin(g[top30.ji.to_numpy()], hot).mean() * 100), 1),
            "pool_hot%": round(float(np.isin(g[win.ji.to_numpy()], hot).mean() * 100), 1)}


def window_view(con, codes: np.ndarray, dates: pd.DatetimeIndex, ind_of: dict, a: pd.Timestamp, b: pd.Timestamp) -> dict:
    d0, d1 = int(dates.searchsorted(a)), int(dates.searchsorted(b, "right")) - 1
    f0 = d0 - 1  # 行情开始前最后一个开市日
    lab = pq.read_table(DATA / f"labels10_o1o5_{panel_key()}.parquet", columns=["di", "ji"],
                        filters=[("di", ">=", f0), ("di", "<=", d1)]).to_pandas()
    w = con.execute(f"""SELECT ts_code, exp(sum(ln(1 + pct_chg / 100))) - 1 AS ret FROM {SRC}
        WHERE trade_date BETWEEN TIMESTAMP '{dates[d0].date()}' AND TIMESTAMP '{dates[d1].date()}' AND pct_chg IS NOT NULL
        GROUP BY ts_code""").df()
    ret = pd.Series(w.ret.to_numpy(), index=pd.Index(codes).get_indexer(w.ts_code))
    r60 = con.execute(f"SELECT ts_code, r60 FROM {SRC} WHERE trade_date = TIMESTAMP '{dates[f0].date()}'").df()
    r60 = pd.Series(r60.r60.to_numpy(), index=pd.Index(codes).get_indexer(r60.ts_code))
    g = np.array([ind_of.get(c, "") for c in codes])
    p0 = lab[lab.di == f0].ji.to_numpy()
    pool0 = pd.DataFrame({"ji": p0, "ret": ret.reindex(p0).fillna(0.0).to_numpy(), "g": g[p0],
                          "r60": r60.reindex(p0).to_numpy()})  # 整段停牌算 0
    by_ind = pool0.groupby("g").ret.agg(["mean", "size"])
    hot = by_ind[by_ind["size"] >= 10].sort_values("mean", ascending=False).head(5)
    top_pool = pool0.nlargest(100, "ret")
    q = pd.qcut(pool0.r60.rank(method="first"), 5, labels=False)
    out = {"days": d1 - d0 + 1, "pool_n": len(p0), "pool_bh": round(float(pool0.ret.mean() * 100), 1),
           "pool_by_r60_quintile": [round(float(pool0.ret[q == k].mean() * 100), 1) for k in range(5)],
           "hot": {k: [round(float(v["mean"] * 100), 1), int(v["size"])] for k, v in hot.iterrows()},
           "pool_top100_median": round(float(top_pool.ret.median() * 100), 1),
           "pool_top100_industry": counts(top_pool.g.value_counts().head(8)),
           "score": {cad: score_view(lab, cad, f0, d0, d1, top_pool.ji.to_numpy(), g, hot.index.to_numpy())
                     for cad in ("M", "Y")}}
    lead = leaders(con, dates[d0], dates[d1])
    lead["board"] = [board(c, px) for c, px in zip(lead.ts_code, lead.px0)]
    j = pd.Index(codes).get_indexer(lead.ts_code)
    days_in = lab[lab.di >= d0].groupby("ji").size()
    out["all_a_top100"] = {
        "median_ret": round(float(lead.ret.median() * 100), 1),
        "board": counts(lead.board.value_counts()),
        "board_median_ret": {k: round(float(v * 100), 1) for k, v in lead.groupby("board").ret.median().items()},
        "in_pool_at_start": int(np.isin(j, p0).sum()),
        "mean_pool_days%": round(float(np.mean([days_in.get(x, 0) for x in j]) / (d1 - d0 + 1) * 100), 1),
        "industry": counts(lead.industry.value_counts().head(8)),
        "examples": [f"{r.nm}({r.board},{r.ret * 100:.0f}%)" for r in lead.head(15).itertuples()]}
    return out


def book_window(a: pd.Timestamp, b: pd.Timestamp) -> dict:
    out = {}
    for label, (capital, cadence, name) in BOOKS.items():
        z = np.load(DATA / "rules" / str(capital) / f"real_{cadence}.npz")
        when = pd.DatetimeIndex(z["dates"])
        k = (when >= a) & (when <= b)
        out[label] = round((float(np.prod(1 + z["ret"][list(z["names"]).index(name)][k])) - 1) * 100, 1)
        out.setdefault("准入池逐日等权", round((float(np.prod(1 + z["pool"][k])) - 1) * 100, 1))
    return out


def index_window(con, a: pd.Timestamp, b: pd.Timestamp) -> dict:
    idx = con.execute(f"SELECT ts_code, CAST(trade_date AS TIMESTAMP) AS d, close FROM index_daily "
                      f"WHERE trade_date BETWEEN DATE '{(a - pd.Timedelta(days=15)).date()}' AND DATE '{b.date()}'").df()
    out = {}
    for code, label in INDEX.items():
        c = idx[idx.ts_code == code].set_index("d").close.sort_index()
        out[label] = round((float(c[c.index <= b].iloc[-1] / c[c.index < a].iloc[-1]) - 1) * 100, 1)
    return out


def main() -> None:
    print(f"power throttling off: {no_power_throttling()}", flush=True)
    codes, dates = axes()
    con = duckdb.connect(str(DB), read_only=True)
    con.execute("SET memory_limit = '1500MB'")
    con.execute("SET threads = 4")
    basic = con.execute("SELECT ts_code, industry FROM stock_basic").df()
    ind_of = dict(zip(basic.ts_code, basic.industry.fillna("")))
    out = {"part_a": part_a(regime(month_panel(con), ind_of), book_gaps()), "part_b": {}}
    pa = out["part_a"]
    print(json.dumps({k: pa[k] for k in ("months", "wml60_cuts", "mean", "wml60_pos%", "corr", "by_wml60", "ols")},
                     ensure_ascii=False), flush=True)
    for label, (a, b) in WINDOWS.items():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        w = window_view(con, codes, dates, ind_of, a, b)
        w["books"], w["index"] = book_window(a, b), index_window(con, a, b)
        out["part_b"][label] = w
        print(label, json.dumps(w, ensure_ascii=False), flush=True)
    con.close()
    (DATA / "momentum_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("-> momentum_check.json", flush=True)


if __name__ == "__main__":
    main()
