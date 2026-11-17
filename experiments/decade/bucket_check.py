"""2025–2026 的行情结构：准入池、被 70 元上限挡掉的票、以及各行业每月涨跌（只读面板，不改结果）。

回答"五六月科技牛为什么没抓住"：看涨的是不是池子以外的票、哪些行业在涨、池子里有多少这类票。
用法：python -X utf8 bucket_check.py
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from cap70_check import load
from common import DB, axes


def monthly(ret: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """逐日等权收益按月复利。ret[0] 是空的，按日期对齐。"""
    s = pd.Series(ret[1:], index=dates[1:])
    g = s.groupby(s.index.to_period("M"))
    return pd.DataFrame({"ret": g.apply(lambda x: (np.prod(1 + np.nan_to_num(x)) - 1) * 100), "n": g.size()})


def industries(codes: np.ndarray) -> np.ndarray:
    con = duckdb.connect(str(DB), read_only=True)
    try:
        raw = con.execute("SELECT ts_code, industry FROM stock_basic").df()
    finally:
        con.close()
    m = dict(zip(raw.ts_code, raw.industry))
    return np.array([m.get(c, "未知") or "未知" for c in codes])


def industry_returns(r: np.ndarray, member: np.ndarray, dates: pd.DatetimeIndex, ind: np.ndarray,
                     start: str, end: str) -> pd.DataFrame:
    """一段区间里，每个行业的等权累计收益（昨天在组里、今天有价的票）。"""
    sel = (dates[1:] >= pd.Timestamp(start)) & (dates[1:] <= pd.Timestamp(end))
    rows = []
    for group in np.unique(ind):
        cols = np.flatnonzero(ind == group)
        ok = member[:-1][:, cols] & np.isfinite(r[:, cols])
        n = ok.sum(axis=1)
        daily = np.where(n > 0, np.where(ok, r[:, cols], 0).sum(axis=1) / np.maximum(n, 1), np.nan)
        seg = daily[sel]
        rows.append({"行业": group, "股票数": int(n[sel].mean()), "区间收益": (np.prod(1 + np.nan_to_num(seg)) - 1) * 100})
    return pd.DataFrame(rows).sort_values("区间收益", ascending=False)


def main() -> None:
    dates, close, money, pool, capped = load()
    with np.errstate(invalid="ignore", divide="ignore"):
        r = close[1:] / close[:-1] - 1
    fin = np.isfinite(r)
    del close
    from cap70_check import daily as group_daily
    rets = {k: group_daily(r, fin, m)[0] for k, m in (("pool", pool), ("capped", capped), ("nocap", pool | capped))}
    print("== 逐月（等权，%）")
    tab = pd.DataFrame({k: monthly(v, dates)["ret"] for k, v in rets.items()})
    tab["n_pool"] = monthly(rets["pool"], dates)["n"]
    tab["n_capped"] = monthly(rets["capped"], dates)["n"]
    print(tab.loc["2025-01":].round(1).to_string())
    codes, _ = axes()
    ind = industries(codes)
    for span in (("2026-05-01", "2026-06-30"), ("2026-05-01", "2026-05-31"), ("2026-06-01", "2026-06-30")):
        t = industry_returns(r, pool | capped, dates, ind, *span)
        print(f"\n== {span[0]}~{span[1]} 行业等权收益（不设上限的准入范围），前 12 / 后 5")
        print(t.head(12).round(1).to_string(index=False))
        print(t.tail(5).round(1).to_string(index=False))
        inpool = pd.DataFrame({"行业": ind, "池内": pool[-1], "被挡": capped[-1]}).groupby("行业")[["池内", "被挡"]].sum()
        top = t.head(12)["行业"]
        print("上面这些行业当时在池内/被挡的只数：")
        print(inpool.loc[top].to_string())


if __name__ == "__main__":
    main()
