"""阶段策略回测的数据部分：主板日线 → 截至当天的量价特征、准入池标记、T+1 开盘买的后续收益。

只读 research.duckdb（READ_ONLY）。特征只用当天收盘及以前，窗口按这只票自己的交易日数，停牌日不补。
准入池近似正式口径：主板、名称不含 ST、收盘 < 70、成交额 >= 0.8 亿、至少 150 根日线。ST 用当前名称，有轻微前视。
收益仿正式 compute_returns：T+1 开盘买（一字涨停记缺失），T+k 开盘卖（一字跌停记缺失），复权价。
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NON_MAIN = r"^(300|301|688|689|8|4|9)"  # 与 rebound/universe.py 的 NON_MAIN_PREFIXES 相同
TOL = 1e-6
EXITS = {"t2": 2, "t5": 5, "t10": 10}
FEATURES = ["pct_chg", "r1", "r5", "r10", "r20", "r60", "r120", "from_low60", "from_high60", "dev_ma5",
            "dev_ma20", "dev_ma60", "ma20_gt_ma60", "ma20_slope5", "ma60_slope10", "dd20", "vol5", "vol20",
            "amp20", "launch10", "up_days10", "max_up20", "limit_up_today", "limit_up20", "dif_ratio",
            "hist_ratio", "rank_r20", "rank_r60"]


def _win(lo: int, hi: str = "CURRENT ROW") -> str:
    return f"(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN {lo} PRECEDING AND {hi})"


SQL = f"""
WITH b AS (
  SELECT d.ts_code, d.trade_date, d.open, d.high, d.low, d.close, d.pct_chg, d.amount,
         d.close * a.adj_factor AS c, d.open * a.adj_factor AS o, l.up_limit, l.down_limit,
         CASE WHEN abs(d.close - l.up_limit) <= {TOL} THEN 1.0 ELSE 0.0 END AS lu
  FROM daily d JOIN adj_factor a USING (ts_code, trade_date) LEFT JOIN daily_limit l USING (ts_code, trade_date)
  WHERE NOT regexp_matches(d.ts_code, '{NON_MAIN}')
), w AS (
  SELECT *, (row_number() OVER s) - 1 AS bar_no,
    c / lag(c, 1) OVER s - 1 AS r1, c / lag(c, 5) OVER s - 1 AS r5, c / lag(c, 10) OVER s - 1 AS r10,
    c / lag(c, 20) OVER s - 1 AS r20, c / lag(c, 60) OVER s - 1 AS r60, c / lag(c, 120) OVER s - 1 AS r120,
    avg(c) OVER w5 AS ma5, avg(c) OVER w20 AS ma20, avg(c) OVER w60 AS ma60, max(c) OVER w20 AS hi20,
    min(c) OVER w60 AS lo60, max(c) OVER w60 AS hi60, avg((high - low) / close) OVER w20 AS amp,
    max(pct_chg) OVER w20 AS maxup, sum(lu) OVER w20 AS lu20, avg(amount) OVER p5 AS amt5,
    avg(amount) OVER p20 AS amt20, sum(CASE WHEN pct_chg >= 5 OR lu = 1 THEN 1 ELSE 0 END) OVER p10 AS ln10,
    sum(CASE WHEN pct_chg > 0 THEN 1 ELSE 0 END) OVER p10 AS up10
  FROM b
  WINDOW s AS (PARTITION BY ts_code ORDER BY trade_date), w5 AS {_win(4)}, w20 AS {_win(19)},
    w60 AS {_win(59)}, p5 AS {_win(5, '1 PRECEDING')}, p10 AS {_win(10, '1 PRECEDING')},
    p20 AS {_win(20, '1 PRECEDING')}
)
SELECT ts_code, trade_date, open, high, low, close, amount, c, o, up_limit, down_limit, bar_no,
  pct_chg, r1, r5, r10, r20, r60, r120,
  CASE WHEN bar_no >= 59 THEN c / lo60 - 1 END AS from_low60,
  CASE WHEN bar_no >= 59 THEN c / hi60 - 1 END AS from_high60,
  CASE WHEN bar_no >= 4 THEN c / ma5 - 1 END AS dev_ma5,
  CASE WHEN bar_no >= 19 THEN c / ma20 - 1 END AS dev_ma20,
  CASE WHEN bar_no >= 59 THEN c / ma60 - 1 END AS dev_ma60,
  CASE WHEN bar_no >= 59 THEN CAST(ma20 > ma60 AS DOUBLE) END AS ma20_gt_ma60,
  CASE WHEN bar_no >= 24 THEN ma20 / lag(ma20, 5) OVER s - 1 END AS ma20_slope5,
  CASE WHEN bar_no >= 69 THEN ma60 / lag(ma60, 10) OVER s - 1 END AS ma60_slope10,
  CASE WHEN bar_no >= 19 THEN c / hi20 - 1 END AS dd20,
  CASE WHEN bar_no >= 5 THEN amount / amt5 END AS vol5,
  CASE WHEN bar_no >= 20 THEN amount / amt20 END AS vol20,
  CASE WHEN bar_no >= 19 THEN amp END AS amp20,
  CASE WHEN bar_no >= 10 THEN ln10 END AS launch10,
  CASE WHEN bar_no >= 10 THEN up10 END AS up_days10,
  CASE WHEN bar_no >= 19 THEN maxup END AS max_up20,
  lu AS limit_up_today,
  CASE WHEN bar_no >= 19 THEN lu20 END AS limit_up20
FROM w WINDOW s AS (PARTITION BY ts_code ORDER BY trade_date)
ORDER BY ts_code, trade_date
"""


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _locked(bars: pd.DataFrame, limit: str) -> np.ndarray:
    hits = [(bars[col] - bars[limit]).abs() <= TOL for col in ("open", "high", "low", "close")]
    return (bars[limit].notna() & np.logical_and.reduce(hits)).to_numpy()


def load_bars() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    con = duckdb.connect(str(ROOT / "data/research.duckdb"), read_only=True)
    try:
        bars = con.execute(SQL).df()
        names = con.execute("SELECT ts_code, name FROM stock_basic").df()
        cal = pd.DatetimeIndex(pd.to_datetime([d for (d,) in con.execute(
            "SELECT cal_date FROM trade_cal WHERE exchange='SSE' AND is_open=1 ORDER BY cal_date").fetchall()]))
    finally:
        con.close()
    bars["trade_date"] = pd.to_datetime(bars.trade_date).astype("datetime64[ns]")
    by_code = bars.groupby("ts_code", sort=False)["c"]
    dif = by_code.transform(_ema, 12) - by_code.transform(_ema, 26)
    dea = dif.groupby(bars.ts_code, sort=False).transform(_ema, 9)
    warm = bars.bar_no < 33
    bars["dif_ratio"] = (dif / bars.c).mask(warm)
    bars["hist_ratio"] = (2 * (dif - dea) / bars.c).mask(warm)
    for col in ("r20", "r60"):
        bars[f"rank_{col}"] = bars.groupby("trade_date")[col].rank(pct=True)
    bars["up_locked"], bars["down_locked"] = _locked(bars, "up_limit"), _locked(bars, "down_limit")
    st = names.name.isna() | names.name.astype(str).str.contains(r"ST|\*", regex=True)
    bars["eligible"] = (bars.ts_code.isin(names.loc[~st, "ts_code"]) & (bars.close < 70)
                        & (bars.amount >= 80_000) & (bars.bar_no >= 149))
    bars[FEATURES] = bars[FEATURES].replace([np.inf, -np.inf], np.nan).astype("float32")
    keep = ["ts_code", "trade_date", "bar_no", "o", "up_locked", "down_locked", "eligible", *FEATURES]
    return bars[keep], cal


def forward_returns(bars: pd.DataFrame, rows: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """rows 每行（某票某天）T+1 开盘买、T+k 开盘卖的收益；买不进、卖不出、缺行情记 NaN。"""
    key = bars.set_index(["ts_code", "trade_date"])[["o", "up_locked", "down_locked"]]
    pos = cal.get_indexer(rows.trade_date)

    def at(k: int) -> pd.DataFrame:
        idx = pos + k
        dates = np.where((pos >= 0) & (idx < len(cal)), cal.values[np.clip(idx, 0, len(cal) - 1)],
                         np.datetime64("NaT"))
        return key.reindex(pd.MultiIndex.from_arrays([rows.ts_code.to_numpy(), dates]))

    entry = at(1)
    entry_open = entry.o.to_numpy()
    entry_locked = entry.up_locked.astype("boolean").fillna(False).to_numpy(dtype=bool)
    out = rows.copy()
    out["entry_locked"] = entry_locked
    with np.errstate(invalid="ignore", divide="ignore"):
        for name, k in EXITS.items():
            exit_ = at(k)
            exit_open = exit_.o.to_numpy()
            sell_locked = exit_.down_locked.astype("boolean").fillna(False).to_numpy(dtype=bool)
            ok = ~np.isnan(entry_open) & ~entry_locked & ~np.isnan(exit_open) & ~sell_locked
            out[name] = np.where(ok, exit_open / entry_open - 1, np.nan)
    return out
