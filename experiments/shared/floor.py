"""价格下限：信号日收盘价（未复权）低于 FLOOR 元的票。供 PREREG 补充登记的敏感性对照使用。"""
from __future__ import annotations

import duckdb
import numpy as np

from engine import ROOT, Market

FLOOR = 2.0


def last_price_day(m: Market) -> np.ndarray:
    """每只股票最后一个有收盘价的开市日下标（之后再没有行情的，视为退市或长期停牌到数据末尾）。"""
    has = np.isfinite(m.price["close"])
    last = len(m.dates) - 1 - np.argmax(has[::-1], axis=0)
    return np.where(has.any(axis=0), last, -1)


def below_floor(m: Market, floor: float = FLOOR) -> np.ndarray:
    """日 × 股：当天收盘价 < floor（没有行情的格子为 False）。"""
    con = duckdb.connect(str(ROOT / "data/research.duckdb"), read_only=True)
    try:
        rows = con.execute("SELECT ts_code, CAST(trade_date AS TIMESTAMP) AS trade_date FROM daily WHERE close < ?",
                           [floor]).df()
    finally:
        con.close()
    out = np.zeros((len(m.dates), len(m.codes)), dtype=bool)
    di, ji = m.date_index(rows.trade_date.astype("datetime64[ns]")), m.code_index(rows.ts_code)
    ok = (di >= 0) & (ji >= 0)
    out[di[ok], ji[ok]] = True
    return out
