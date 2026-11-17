"""2026 年 5–6 月：模型当天排在前 30 的都是什么行业、这些票后来 5 天涨了多少（只读，不改结果）。

用来回答"科技牛为什么没抓住"：是模型没选，还是行业上限 2 只卡住了。
用法：python -X utf8 top30_check.py
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import DATA, DB, PANEL, axes

COLS = ["ts_code", "trade_date", "c", "eligible", "below2"]
DAYS = ["2026-05-15", "2026-06-15", "2026-06-30"]


def industry_of(codes: np.ndarray) -> dict[str, str]:
    con = duckdb.connect(str(DB), read_only=True)
    try:
        raw = con.execute("SELECT ts_code, industry FROM stock_basic").df()
    finally:
        con.close()
    return dict(zip(raw.ts_code, raw.industry))


def main() -> None:
    codes, dates = axes()
    pos = {c: i for i, c in enumerate(codes)}
    ind = industry_of(codes)
    start, end = pd.Timestamp("2026-05-01"), pd.Timestamp("2026-07-10")
    p = pq.read_table(PANEL, columns=COLS, filters=[("trade_date", ">=", start), ("trade_date", "<=", end)]).to_pandas()
    p["trade_date"] = pd.to_datetime(p.trade_date)
    sc = pd.read_parquet(DATA / "scores10_M.parquet")
    score = pd.Series(sc.score.to_numpy(), index=pd.MultiIndex.from_arrays([sc.di.to_numpy(), sc.ji.to_numpy()]))
    close = p.pivot_table(index="trade_date", columns="ts_code", values="c", aggfunc="last").sort_index()
    for day in DAYS:
        t = pd.Timestamp(day)
        snap = p[p.trade_date == t].copy()
        snap = snap[snap.eligible.fillna(False) & ~snap.below2.fillna(False)]
        di = dates.get_loc(t)
        snap["score"] = [score.get((di, pos.get(c, -1)), np.nan) for c in snap.ts_code]
        snap = snap.dropna(subset=["score"]).sort_values("score", ascending=False)
        top = snap.head(30)
        mix = top.ts_code.map(ind).value_counts()
        # 前 30 名里各行业有几只、行业上限 2 只会砍掉几只
        hot = mix[mix > 2]
        fwd = []
        for c in top.ts_code.head(30):
            if c in close.columns:
                s = close[c].dropna()
                if t in s.index:
                    i = s.index.get_loc(t)
                    if i + 5 < len(s):
                        fwd.append(s.iloc[i + 5] / s.iloc[i] - 1)
        win = close.loc[t:].iloc[:6]
        pool_all = win.iloc[-1] / win.iloc[0] - 1
        pool_all = float(np.nanmean(pool_all)) if len(win) > 1 else np.nan
        print(f"\n== {day} 模型前 30：行业分布（只列 >1 只的）")
        print(mix[mix > 1].to_string())
        print("被行业上限（每行业 2 只）砍掉的：", {k: int(v) - 2 for k, v in hot.items()})
        print(f"前 30 之后 5 个交易日的等权收益：{np.mean(fwd) * 100:+.1f}%（{len(fwd)} 只）；"
              f"准入池同期 {pool_all * 100:+.1f}%")


if __name__ == "__main__":
    main()
