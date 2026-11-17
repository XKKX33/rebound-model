"""V1（行业特征版）2 万书 2026-07 的逐票拆账：和 july_drop.part2 同一口径，只跑 DB-K3-b16。

输出 ind/july_v1.json：7 月组合/池子收益、持仓在 6 月底 r60 五组的分布、最差个股贡献。
用法：DECADE_DATA=ind python -X utf8 ind/july_v1.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import DATA, DB, PANEL, load_market10, row_index, signal_range
from engine import candidates, pool_daily_return
from july_drop import JULY, position_pnl
from rule_grid import parse
from rule_search import cost_model
from sim import simulate
from train_cadence import no_power_throttling

HERE = Path(__file__).resolve().parent
NAME = "DB-K3-b16-o1o5"


def main() -> None:
    print(f"power throttling off: {no_power_throttling()}", flush=True)
    m, low, _ = load_market10(raw=True)
    dates = m.dates
    d_first, d_last = signal_range(dates)
    d0, d1 = int(dates.searchsorted(JULY[0])), int(dates.searchsorted(JULY[1], "right")) - 1
    mf = replace(m, eligible=m.eligible & ~low)
    pool = pool_daily_return(mf)
    s = pd.read_parquet(DATA / "scores10_M.parquet")
    cands, rank = candidates(m, s[~low[s.di.to_numpy(), s.ji.to_numpy()]])
    cm = cost_model(20_000, "real")
    res = simulate(mf, cands, rank, parse(NAME).rule, cm, d_first, d_last)
    p = position_pnl(mf, res.trades, cm, d0, d1)
    base = res.nav[d0 - 1]

    con = duckdb.connect(str(DB), read_only=True)
    basic = con.execute("SELECT ts_code, name, industry FROM stock_basic").df()
    con.close()
    info = basic.set_index("ts_code")
    c = mf.price["close"]
    last = lambda j, d: pd.Series(c[max(d - 40, 0):d + 1, j]).ffill().iloc[-1]

    by = p.groupby("j").agg(pnl=("pnl", "sum"), trades=("entry_day", "nunique"), first=("entry_day", "min"),
                            sig=("signal_day", "min"))
    stock = [{"code": str(m.codes[j]), "name": str(info.name.get(m.codes[j], "")),
              "industry": str(info.industry.get(m.codes[j], "")),
              "contrib": round(v.pnl / base * 100, 2), "trades": int(v.trades),
              "first_entry": str(dates[int(v["first"])].date()), "rank_at_signal": float(rank[int(v.sig), j]),
              "stock_july": round((last(j, d1) / last(j, d0 - 1) - 1) * 100, 1)}
             for j, v in by.sort_values("pnl").iterrows()]

    # 6 月底 r60 五组（和 july_drop.part5 同一组界）
    allj = pq.read_table(PANEL, columns=["ts_code", "trade_date", "r60"],
                         filters=[("trade_date", "==", dates[d0 - 1].to_pydatetime())]).to_pandas()
    _, ja = row_index(pd.Categorical(allj.ts_code.astype(str)), allj.trade_date, m.codes, m.dates)
    r60_all = np.full(len(m.codes), np.nan)
    r60_all[ja] = allj.r60.to_numpy()
    june = r60_all[np.isfinite(r60_all)]
    edges = np.quantile(june, [0.2, 0.4, 0.6, 0.8])
    q = np.digitize(r60_all[p.j.to_numpy()], edges) + 1
    q = np.where(np.isfinite(r60_all[p.j.to_numpy()]), q, 0)
    quint = p.assign(q=q).groupby("q").pnl.sum().div(base / 100).round(2).to_dict()

    held = p[p.entry_day < d0]
    out = {"book": round((res.nav[d1] / base - 1) * 100, 2),
           "pool": round((np.prod(1 + pool[d0:d1 + 1]) - 1) * 100, 2),
           "n_stocks": len(stock), "held_from_june": round(held.pnl.sum() / base * 100, 2),
           "entered_july": round(p[p.entry_day >= d0].pnl.sum() / base * 100, 2),
           "quintile_pnl": {int(k): float(v) for k, v in quint.items()},
           "worst": stock[:10], "best": stock[-5:]}
    (HERE / "july_v1.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
