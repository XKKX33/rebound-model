"""强势股行情能不能提前看出来：月度动量收益差（个股 / 行业）的自相关，以及上个月是强势股月时，这个月三套规则
比准入池多赚多少（只读诊断，接 momentum_check.py / industry_check.py）。"""
from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd

from common import DATA, DB
from industry_check import monthly
from momentum_check import BOOKS, book_gaps, month_panel, regime
from train_cadence import no_power_throttling

COLS = ("wml60", "wml20", "ic60", "ind_wml60", "ind_wml20", "within_wml60")


def main() -> None:
    print(f"power throttling off: {no_power_throttling()}", flush=True)
    con = duckdb.connect(str(DB), read_only=True)
    con.execute("SET memory_limit = '1500MB'")
    con.execute("SET threads = 4")
    basic = con.execute("SELECT ts_code, industry FROM stock_basic").df()
    ind_of = dict(zip(basic.ts_code, basic.industry.fillna("")))
    mp = month_panel(con)
    con.close()
    df = regime(mp, ind_of).join(monthly(mp, ind_of)).join(book_gaps(), how="inner")
    n = len(df)
    out = {"months": n, "ar1": {}, "next_by_prev_wml60": {}, "pos_after_pos": {}}
    for c in COLS:
        rho = float(df[c].autocorr(1))
        out["ar1"][c] = {"rho": round(rho, 2), "t": round(rho * np.sqrt(n), 1),
                         "rho_lag2": round(float(df[c].autocorr(2)), 2), "rho_lag3": round(float(df[c].autocorr(3)), 2)}
        pos = df[c] > 0
        out["pos_after_pos"][c] = {"base%": round(float(pos.mean() * 100)),
                                   "after_pos%": round(float(pos[pos.shift(1, fill_value=False)].mean() * 100)),
                                   "after_neg%": round(float(pos[~pos.shift(1, fill_value=True)].mean() * 100))}
    prev = pd.qcut(df.wml60.shift(1), 3, labels=["上月反转", "上月中间", "上月强势股"])
    for col in (*BOOKS, "wml60", "ind_wml60"):
        out["next_by_prev_wml60"][col] = {str(k): round(float(v.mean()), 2) for k, v in df[col].groupby(prev, observed=True)}
    (DATA / "regime_persist.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
