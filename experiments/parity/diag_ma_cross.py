"""诊断：ma20_gt_ma60 在哪些格子上研究版和正式版不一样，那几格的 ma20、ma60 是不是相等。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE.parent / "shared"), str(ROOT)]

import engine as rs  # noqa: E402

from rebound.features import feature_panel  # noqa: E402
from rebound.store import Store  # noqa: E402

days = [pd.Timestamp(d) for d in json.loads((HERE / "phase1_check.json").read_text(encoding="utf-8"))["days"]]
research = pd.read_parquet(rs.PANEL, columns=["ts_code", "trade_date", "ma20_gt_ma60", "c"],
                           filters=[("trade_date", "in", days)])
research["ts_code"] = research.ts_code.astype(str)
with Store(ROOT / "data/research.duckdb") as store:
    formal = pd.concat([feature_panel(store, d.date(), d.date()) for d in days], ignore_index=True)
formal["ts_code"] = formal.ts_code.astype(str)
both = research.merge(formal[["ts_code", "trade_date", "ma20_gt_ma60"]], on=["ts_code", "trade_date"],
                      suffixes=("_r", "_f"))
diff = both[both.ma20_gt_ma60_r.fillna(-1) != both.ma20_gt_ma60_f.fillna(-1)]
print(f"不一样的格子 {len(diff)} / {len(both)}")
con = duckdb.connect(str(ROOT / "data/research.duckdb"), read_only=True)
for row in diff.itertuples():
    got = con.execute("""
        SELECT avg(c20), avg(c60) FROM (
          SELECT d.close * a.adj_factor AS c60,
                 CASE WHEN row_number() OVER (ORDER BY d.trade_date DESC) <= 20 THEN d.close * a.adj_factor END AS c20
          FROM daily d JOIN adj_factor a USING (ts_code, trade_date)
          WHERE d.ts_code = ? AND d.trade_date <= ? ORDER BY d.trade_date DESC LIMIT 60)""",
        [row.ts_code, row.trade_date.date()]).fetchone()
    exact = con.execute("""
        SELECT count(DISTINCT round(d.close * a.adj_factor, 9)) FROM (
          SELECT * FROM daily WHERE ts_code = ? AND trade_date <= ? ORDER BY trade_date DESC LIMIT 60) d
        JOIN adj_factor a USING (ts_code, trade_date)""", [row.ts_code, row.trade_date.date()]).fetchone()[0]
    print(f"  {row.ts_code} {row.trade_date.date()}  研究 {row.ma20_gt_ma60_r} 正式 {row.ma20_gt_ma60_f}  "
          f"ma20 {got[0]!r} ma60 {got[1]!r}  差 {got[0] - got[1]:.3e}  60 根里不同价格 {exact} 个")
con.close()
