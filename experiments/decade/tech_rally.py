"""2026 年 4–6 月科技股上涨时，模型为什么没跟上（诊断，只读，不改任何结果）。

1. 逐月：指数、各板块等权（主板 < 70 元 / 主板 ≥ 70 元 / 创业板 / 科创板，各分科技、非科技）、准入池、几条推荐规则。
2. 4–6 月拆账：推荐规则的持仓按行业和准入池比，分成行业配置和行业内选股（按只数等权近似）；
   另跑"每个行业不限 2 只"作对照。
3. 打分：每天打分和 T+1 开 → T+5 开收益排名的相关（IC），以及打分和几个量价特征、市值的相关（风格暴露），按月平均；
   打分前 30 名里科技股的占比 vs 准入池。
科技 = Tushare 行业里的半导体、元器件、软件服务、通信设备、IT设备、互联网。板块等权按 pct_chg，剔除当时的 ST 和上市不满 60 天。
输出 tech_rally.json。用法：python -X utf8 tech_rally.py
"""
from __future__ import annotations

import json
from dataclasses import replace

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import DATA, DB, PANEL, axes, load_market10, row_index, signal_range
from engine import candidates
from rule_grid import parse
from rule_search import cost_model
from sim import simulate
from train_cadence import no_power_throttling, panel_key

TECH = ("半导体", "元器件", "软件服务", "通信设备", "IT设备", "互联网")
FROM, WIN = pd.Timestamp("2025-07-01"), (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-06-30"))
ST = r"ST|\*"
BOOKS = {"20万推荐": (200000, "M", "DB-K30-b8-o1o5"), "2万推荐": (20000, "M", "DB-K3-b16-o1o5"),
         "原方案": (200000, "Y", "FH-K6-next-o1o5")}
FEATURES = ("r5", "r20", "r60", "vol20", "from_high60", "dev_ma20", "limit_up20")


def monthly(ret: pd.Series) -> dict:
    """逐日收益 → 逐月复利（%）。"""
    return {str(k): round((float(np.prod(1 + v.fillna(0))) - 1) * 100, 1)
            for k, v in ret.groupby(ret.index.to_period("M"))}


def segments(con) -> pd.DataFrame:
    """各板块（主板分 < 70 元 / ≥ 70 元）× 科技 / 非科技 的逐日等权收益和只数。"""
    tech = ", ".join(f"'{t}'" for t in TECH)
    return con.execute(f"""
        WITH names AS (SELECT ts_code, name, start_date FROM namechange WHERE start_date IS NOT NULL),
        d AS (SELECT d.ts_code, CAST(d.trade_date AS TIMESTAMP) AS trade_date, d.pct_chg, d.pre_close
              FROM daily d WHERE d.trade_date >= DATE '{FROM.date()}'),
        r AS (SELECT d.*, n.name AS name_then FROM d ASOF LEFT JOIN names n
              ON d.ts_code = n.ts_code AND d.trade_date >= CAST(n.start_date AS TIMESTAMP))
        SELECT trade_date,
               CASE WHEN r.ts_code LIKE '300%' OR r.ts_code LIKE '301%' OR r.ts_code LIKE '302%' THEN '创业板'
                    WHEN r.ts_code LIKE '688%' OR r.ts_code LIKE '689%' THEN '科创板'
                    WHEN r.ts_code LIKE '%.BJ' THEN '北交所'
                    WHEN r.pre_close >= 70 THEN '主板≥70元' ELSE '主板<70元' END AS board,
               coalesce(b.industry IN ({tech}), false) AS tech,
               avg(r.pct_chg) / 100 AS ret, count(*) AS n
        FROM r JOIN stock_basic b ON r.ts_code = b.ts_code
        WHERE NOT regexp_matches(coalesce(r.name_then, b.name, ''), '{ST}')
          AND r.trade_date >= CAST(b.list_date AS TIMESTAMP) + INTERVAL 60 DAY AND r.pct_chg IS NOT NULL
        GROUP BY 1, 2, 3 ORDER BY 1, 2, 3""").df()


def part1(con, dates: pd.DatetimeIndex) -> dict:
    seg = segments(con)
    out = {"index": {}, "board": {}, "board_tech": {}, "pool": None, "books": {}}
    idx = con.execute("SELECT ts_code, CAST(trade_date AS TIMESTAMP) AS d, close FROM index_daily "
                      f"WHERE trade_date >= DATE '{(FROM - pd.Timedelta(days=10)).date()}'").df()
    for code, name in {"000300.SH": "沪深300", "000905.SH": "中证500", "000852.SH": "中证1000"}.items():
        c = idx[idx.ts_code == code].set_index("d").close.sort_index()
        out["index"][name] = monthly(c.pct_change()[c.index >= FROM])
    for board, g in seg.groupby("board"):
        both = g.groupby("trade_date").apply(lambda x: np.average(x.ret, weights=x.n))
        out["board"][board] = monthly(both)
        tech = g[g.tech].set_index("trade_date").ret
        if len(tech):
            out["board_tech"][board] = monthly(tech)
    for label, (capital, cadence, name) in BOOKS.items():
        z = np.load(DATA / "rules" / str(capital) / f"real_{cadence}.npz")
        when = pd.DatetimeIndex(z["dates"])
        row = list(z["names"]).index(name)
        keep = when >= FROM
        out["books"][label] = monthly(pd.Series(z["ret"][row][keep], index=when[keep]))
        if out["pool"] is None:
            out["pool"] = monthly(pd.Series(z["pool"][keep], index=when[keep]))
    return out


def legs(m, trades: pd.DataFrame, t: int) -> list[tuple[int, float]]:
    """第 t 天持有的每只票当天的收益：买入那天开 → 收，中间收 → 收，卖出那天昨收 → 开。"""
    o, c = m.price["open"], m.price["close"]
    out = []
    for j, e, x in trades.loc[(trades.entry_day <= t) & (trades.exit_day >= t), ["j", "entry_day", "exit_day"]].itertuples(index=False):
        if e == t == x:
            continue
        a, b = (o[t, j], c[t, j]) if t == e else ((c[t - 1, j], o[t, j]) if t == x else (c[t - 1, j], c[t, j]))
        out.append((int(j), float(b / a - 1) if np.isfinite(a) and np.isfinite(b) and a > 0 else 0.0))
    return out


def attribution(m, trades: pd.DataFrame, names: np.ndarray) -> dict:
    """4–6 月逐日：持仓（按只数等权）和准入池按行业比，拆成行业配置 + 行业内选股，逐日相加。"""
    c, pool_ok = m.price["close"], m.eligible
    days = np.flatnonzero((m.dates >= WIN[0]) & (m.dates <= WIN[1]))
    alloc, select, weight_b, weight_p, gap = {}, {}, {}, {}, 0.0
    for t in days:
        held = legs(m, trades, t)
        if not held:
            continue
        ok = pool_ok[t - 1] & np.isfinite(c[t]) & np.isfinite(c[t - 1])
        r_pool = c[t, ok] / c[t - 1, ok] - 1
        g_pool = m.industry[ok]
        rp = float(r_pool.mean())
        book = pd.DataFrame(held, columns=["j", "r"]).assign(g=lambda d: m.industry[d.j.to_numpy()])
        gap += float(book.r.mean()) - rp
        pool = pd.DataFrame({"g": g_pool, "r": r_pool}).groupby("g").r.agg(["mean", "size"])
        wb, rb = book.groupby("g").size() / len(book), book.groupby("g").r.mean()
        wp = pool["size"] / pool["size"].sum()
        for g in set(wb.index) | set(wp.index):
            w_b, w_p = float(wb.get(g, 0.0)), float(wp.get(g, 0.0))
            r_p = float(pool["mean"].get(g, rp))
            name = names[g]
            alloc[name] = alloc.get(name, 0.0) + (w_b - w_p) * (r_p - rp)
            select[name] = select.get(name, 0.0) + w_b * (float(rb.get(g, r_p)) - r_p)
            weight_b[name] = weight_b.get(name, 0.0) + w_b / len(days)
            weight_p[name] = weight_p.get(name, 0.0) + w_p / len(days)
    rows = sorted(({"industry": k, "tech": k in TECH, "book_w": round(weight_b.get(k, 0) * 100, 1),
                    "pool_w": round(weight_p.get(k, 0) * 100, 1), "alloc": round(alloc[k] * 100, 2),
                    "select": round(select.get(k, 0) * 100, 2)} for k in alloc), key=lambda r: r["alloc"] + r["select"])
    tech = [r for r in rows if r["tech"]]
    return {"gap_sum": round(gap * 100, 1), "alloc": round(sum(alloc.values()) * 100, 1),
            "select": round(sum(select.values()) * 100, 1),
            "tech": {"book_w": round(sum(r["book_w"] for r in tech), 1), "pool_w": round(sum(r["pool_w"] for r in tech), 1),
                     "alloc": round(sum(r["alloc"] for r in tech), 1), "select": round(sum(r["select"] for r in tech), 1)},
            "worst": rows[:8], "best": rows[-5:]}


def window_return(nav: np.ndarray, dates: pd.DatetimeIndex) -> float:
    a = int(dates.searchsorted(WIN[0]))
    b = int(dates.searchsorted(WIN[1], "right")) - 1
    return round((float(nav[b] / nav[a - 1]) - 1) * 100, 1)


def daily_rank_corr(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    """每天的截面秩相关（Spearman）。"""
    d = df[["di", a, b]].dropna()
    ra, rb = d.groupby("di")[a].rank(), d.groupby("di")[b].rank()
    ra, rb = ra - ra.groupby(d.di).transform("mean"), rb - rb.groupby(d.di).transform("mean")
    num = (ra * rb).groupby(d.di).sum()
    den = np.sqrt((ra ** 2).groupby(d.di).sum() * (rb ** 2).groupby(d.di).sum())
    return num / den


def part3(con, codes: np.ndarray, dates: pd.DatetimeIndex, names: np.ndarray, industry: np.ndarray) -> dict:
    """打分的 IC、各特征的 IC、打分对特征的暴露（都按月平均），以及打分前 30 名里科技股的占比。"""
    d0 = int(dates.searchsorted(FROM))
    lab = pd.read_parquet(DATA / f"labels10_o1o5_{panel_key()}.parquet")
    lab = lab[lab.di >= d0]
    sc = pd.read_parquet(DATA / "scores10_M.parquet")
    df = lab.merge(sc[sc.di >= d0], on=["di", "ji"])
    p = pq.read_table(PANEL, columns=["ts_code", "trade_date", *FEATURES], read_dictionary=["ts_code"],
                      filters=[("trade_date", ">=", FROM.to_pydatetime())]).to_pandas()
    di, ji = row_index(p.pop("ts_code"), p.pop("trade_date"), codes, dates)
    df = df.merge(p.assign(di=di.astype(np.int32), ji=ji.astype(np.int32)), on=["di", "ji"], how="left")
    mv = con.execute(f"SELECT ts_code, CAST(trade_date AS TIMESTAMP) AS trade_date, circ_mv FROM daily_basic "
                     f"WHERE trade_date >= DATE '{FROM.date()}'").df()
    mv = mv[mv.ts_code.isin(codes) & mv.trade_date.isin(dates) & (mv.circ_mv > 0)]  # 面板只有主板
    di, ji = row_index(pd.Categorical(mv.ts_code), mv.trade_date, codes, dates)
    df = df.merge(pd.DataFrame({"di": di.astype(np.int32), "ji": ji.astype(np.int32),
                                "log_mv": np.log(mv.circ_mv.to_numpy())}), on=["di", "ji"], how="left")
    df["tech"] = np.isin(names[industry[df.ji.to_numpy()]], TECH)
    month = pd.Series(dates.to_period("M").astype(str), index=range(len(dates)))
    out = {"ic_score": (daily_rank_corr(df, "score", "y").groupby(month).mean() * 100).round(1).to_dict()}
    for f in (*FEATURES, "log_mv"):
        out[f"ic_{f}"] = (daily_rank_corr(df, f, "y").groupby(month).mean() * 100).round(1).to_dict()
        out[f"expo_{f}"] = (daily_rank_corr(df, "score", f).groupby(month).mean() * 100).round(1).to_dict()
    top = df.sort_values(["di", "score"], ascending=[True, False]).groupby("di").head(30)
    out["tech_share_top30"] = (top.groupby("di").tech.mean().groupby(month).mean() * 100).round(1).to_dict()
    out["tech_share_pool"] = (df.groupby("di").tech.mean().groupby(month).mean() * 100).round(1).to_dict()
    return out


def main() -> None:
    print(f"power throttling off: {no_power_throttling()}", flush=True)
    con = duckdb.connect(str(DB), read_only=True)
    m, low, _ = load_market10(raw=True)
    dates, codes = m.dates, m.codes
    _, d_last = signal_range(dates)
    mf = replace(m, eligible=m.eligible & ~low)
    basic = con.execute("SELECT ts_code, industry FROM stock_basic").df()
    names = np.array(sorted({dict(zip(basic.ts_code, basic.industry.fillna(""))).get(c, "") for c in codes}))
    out = {"part1": part1(con, dates), "part2": {}}
    d0 = int(dates.searchsorted(FROM))
    scores = {}
    for label, (capital, cadence, name) in BOOKS.items():
        if cadence not in scores:
            s = pd.read_parquet(DATA / f"scores10_{cadence}.parquet")
            scores[cadence] = candidates(m, s[~low[s.di.to_numpy(), s.ji.to_numpy()]])
        cands, rank = scores[cadence]
        rule = parse(name).rule
        row = {}
        for cap in (2, 99):
            res = simulate(mf, cands, rank, replace(rule, cap=cap), cost_model(capital, "real"), d0, d_last)
            row[f"cap{cap}_window"] = window_return(res.nav, dates)
            if cap == 2:
                row["attribution"] = attribution(mf, res.trades, names)
        out["part2"][label] = row
        print(label, {k: v for k, v in row.items() if k != "attribution"},
              {k: row["attribution"][k] for k in ("gap_sum", "alloc", "select", "tech")}, flush=True)
    del scores
    out["part3"] = part3(con, codes, dates, names, m.industry)
    con.close()
    (DATA / "tech_rally.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    p1 = out["part1"]
    for k in ("2026-03", "2026-04", "2026-05", "2026-06", "2026-07"):
        print(k, "pool", p1["pool"].get(k), {b: v.get(k) for b, v in p1["board"].items()},
              {b: v.get(k) for b, v in p1["books"].items()}, flush=True)
    print("-> tech_rally.json", flush=True)


if __name__ == "__main__":
    main()
