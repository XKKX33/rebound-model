"""行业层面的动量 vs 个股层面的反转（只读诊断，接 momentum_check.py）。

1. 月度（同样的成形日和准入池，2015-01 起）：行业动量 = 上月末按行业平均近 60 / 20 日涨幅排序，最强 5 个行业
   − 最弱 5 个行业这个月的等权收益差（百分点）和行业间秩相关；行业内个股动量 = 先减掉行业均值再分五档。
2. 逐日（T+5 名次标签，2015 起）：模型打分（每月重训）的 IC 拆成行业内 / 行业间；行业近 5 / 20 / 60 日平均涨幅
   对行业未来 T+5 平均名次的 IC；打分在行业间和行业动量的秩相关。
"""
from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import DATA, DB, PANEL, axes, row_index, signal_range
from momentum_check import month_panel, regime
from train_cadence import no_power_throttling, panel_key

MIN_MONTH, MIN_DAY = 10, 5  # 行业在当天准入池里至少几只才算


def stats(s: pd.Series) -> dict:
    s = s.dropna()
    return {"mean": round(float(s.mean()), 2), "t": round(float(s.mean() / s.std() * np.sqrt(len(s))), 1),
            "pos%": round(float((s > 0).mean() * 100)), "n": len(s)}


def monthly(mp: pd.DataFrame, ind_of: dict) -> pd.DataFrame:
    mp = mp.assign(g=mp.ts_code.map(ind_of).fillna(""))
    mp = mp[mp.g != ""]
    rows = {}
    for m, x in mp.groupby("m"):
        ind = x.groupby("g").agg(r60=("r60", "mean"), r20=("r20", "mean"), ret=("ret", "mean"), n=("ret", "size"))
        ind = ind[ind.n >= MIN_MONTH]
        x = x[x.g.isin(ind.index)]
        row = {"n_ind": len(ind)}
        for f in ("r60", "r20"):
            s = ind.sort_values(f)
            row["ind_wml" + f[1:]] = (s.ret.tail(5).mean() - s.ret.head(5).mean()) * 100
            row["ind_ic" + f[1:]] = ind[f].rank().corr(ind.ret.rank()) * 100
            dm_f = x[f] - x.groupby("g")[f].transform("mean")
            dm_r = x.ret - x.groupby("g").ret.transform("mean")
            q = pd.qcut(dm_f.rank(method="first"), 5, labels=False)
            row["within_wml" + f[1:]] = (dm_r[q == 4].mean() - dm_r[q == 0].mean()) * 100
        rows[pd.Timestamp(m).to_period("M")] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def demean(v: np.ndarray, key: np.ndarray) -> np.ndarray:
    """v 减去同组均值（key 是从 0 起的组号）。用 bincount：pandas 分类分组的 transform 在全量上出过一次全 0。"""
    return v - (np.bincount(key, weights=v) / np.bincount(key))[key]


def day_corr(di: np.ndarray, a: np.ndarray, b: np.ndarray) -> pd.Series:
    """每天 a、b 的 Pearson 相关（传进来的是名次或去过组均值的名次）。"""
    k, days = pd.factorize(di, sort=True)
    a, b = demean(a.astype(np.float64), k), demean(b.astype(np.float64), k)
    r = np.bincount(k, weights=a * b) / np.sqrt(np.bincount(k, weights=a * a) * np.bincount(k, weights=b * b))
    return pd.Series(r, index=days)


def daily(codes: np.ndarray, dates: pd.DatetimeIndex, ind_of: dict) -> pd.DataFrame:
    d0, _ = signal_range(dates)
    lab = pq.read_table(DATA / f"labels10_o1o5_{panel_key()}.parquet", columns=["di", "ji", "y"],
                        filters=[("di", ">=", d0)]).to_pandas()
    lab = lab[lab.y.notna()]  # 约 2.6 万行没有 T+5 出场（停牌到底等），训练里也不用
    df = lab.merge(pq.read_table(DATA / "scores10_M.parquet", filters=[("di", ">=", d0)]).to_pandas(), on=["di", "ji"])
    del lab
    p = pq.read_table(PANEL, columns=["ts_code", "trade_date", "eligible", "below2", "r5", "r20", "r60"],
                      read_dictionary=["ts_code"], filters=[("eligible", "==", True), ("below2", "==", False),
                                                            ("trade_date", ">=", dates[d0].to_pydatetime())]).to_pandas()
    di, ji = row_index(p.pop("ts_code"), p.pop("trade_date"), codes, dates)
    p = p.drop(columns=["eligible", "below2"]).assign(di=di.astype(np.int32), ji=ji.astype(np.int32))
    df = df.merge(p, on=["di", "ji"], how="left")
    del p
    gname = np.array([ind_of.get(c, "") for c in codes])[df.ji.to_numpy()]
    df = df[gname != ""].reset_index(drop=True)
    gname = gname[gname != ""]
    di = df.di.to_numpy()
    cell = pd.factorize(di.astype(np.int64) * 1000 + pd.factorize(gname, sort=True)[0])[0]  # 日 × 行业
    rk = {c: df.groupby("di")[c].rank(pct=True).to_numpy() for c in ("score", "y", "r20")}
    wi = {c: demean(v, cell) for c, v in rk.items()}
    out = pd.DataFrame({"ic_total": day_corr(di, rk["score"], rk["y"]), "ic_within": day_corr(di, wi["score"], wi["y"]),
                        "mom20_total": day_corr(di, rk["r20"], rk["y"]), "mom20_within": day_corr(di, wi["r20"], wi["y"])})
    ind = pd.DataFrame({"di": di, "cell": cell, "s": rk["score"], "y": rk["y"], "r5": df.r5.to_numpy(),
                        "r20": df.r20.to_numpy(), "r60": df.r60.to_numpy()}).groupby("cell").agg(
        di=("di", "first"), s=("s", "mean"), y=("y", "mean"), r5=("r5", "mean"), r20=("r20", "mean"),
        r60=("r60", "mean"), n=("y", "size"))
    ind = ind[ind.n >= MIN_DAY].dropna()
    for c in ("s", "y", "r5", "r20", "r60"):
        ind[c] = ind.groupby("di")[c].rank()
    d = ind.di.to_numpy()
    out["ic_between"] = day_corr(d, ind.s.to_numpy(), ind.y.to_numpy())
    for c in ("r5", "r20", "r60"):
        out[f"ind_mom{c[1:]}"] = day_corr(d, ind[c].to_numpy(), ind.y.to_numpy())
    out["score_vs_ind_r20"] = day_corr(d, ind.s.to_numpy(), ind.r20.to_numpy())
    if not (out.std() > 0).all():
        raise SystemExit(f"有一列逐日 IC 没有变化，算错了：{out.std().to_dict()}")
    return out * 100


def main() -> None:
    print(f"power throttling off: {no_power_throttling()}", flush=True)
    codes, dates = axes()
    con = duckdb.connect(str(DB), read_only=True)
    con.execute("SET memory_limit = '1500MB'")
    con.execute("SET threads = 4")
    basic = con.execute("SELECT ts_code, industry FROM stock_basic").df()
    ind_of = dict(zip(basic.ts_code, basic.industry.fillna("")))
    mp = month_panel(con)
    con.close()
    reg = regime(mp, ind_of)
    mon = monthly(mp, ind_of).join(reg[["wml60", "ic60"]])
    del mp
    cut = pd.qcut(mon.wml60, 3, labels=["反转月", "中间", "强势股月"])
    out = {"monthly": {c: stats(mon[c]) for c in mon.columns if c != "n_ind"},
           "monthly_by_wml60": {c: {str(k): round(float(v.mean()), 2) for k, v in mon.groupby(cut, observed=True)[c]}
                                for c in ("ind_wml60", "ind_wml20", "within_wml60", "within_wml20")},
           "monthly_2021_2026": {str(k): {c: round(float(v[c]), 1) for c in ("wml60", "ind_wml60", "ind_wml20", "within_wml60")}
                                 for k, v in mon[mon.index.year.isin([2021, 2026])].iterrows()}}
    print(json.dumps(out, ensure_ascii=False), flush=True)
    day = daily(codes, dates, ind_of)
    month = pd.Index(dates[day.index].to_period("M"))
    dm = day.groupby(month).mean()
    dcut = pd.Series(cut, index=mon.index).reindex(dm.index)
    out["daily"] = {c: stats(dm[c]) for c in dm.columns}
    out["daily_by_wml60"] = {c: {str(k): round(float(v.mean()), 2) for k, v in dm[c].groupby(dcut, observed=True)}
                             for c in dm.columns}
    out["daily_2021_2026"] = {str(k): {c: round(float(v[c]), 1) for c in dm.columns}
                              for k, v in dm[dm.index.year.isin([2021, 2026])].iterrows()}
    (DATA / "industry_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("daily", "daily_by_wml60")}, ensure_ascii=False), flush=True)
    print("-> industry_check.json", flush=True)


if __name__ == "__main__":
    main()
