"""任务 1.6：LSTM 的序列数据。

每只主板股票、每个交易日 11 个通道（T 日收盘后就知道的原始日线量），存成张量 `[股票, 交易日, 通道]`。
LSTM 的一个样本 = 主面板的一行 (股票 j, 信号日 T)，输入是 X[j, T−59 : T+1]（60 天 × 11 通道），训练时现切现用。

通道（全部来自 daily、adj_factor、daily_limit、daily_basic、moneyflow，单位见 1.1）：
- ret    ln(复权收盘 / 前一交易日复权收盘)
- ovn    ln(复权开盘 / 前一交易日复权收盘)
- intra  ln(收盘 / 开盘)
- hl     ln(最高 / 最低)
- clv    (2C − H − L) / (H − L)，一字板记 0
- ln_amt ln(成交额)
- ln_turn ln(自由流通换手率)
- mf     净流入额 / 成交额（统一成万元）
- big    (大单 + 特大单买入 − 卖出) / 成交额
- limit  收盘涨停 +0.5，跌停 −0.5，其余 0
- traded 当天有交易 1；停牌、未上市、已退市 0

标准化：前 9 个通道每天在全主板截面排名，映射到 (−0.5, 0.5)（`ml/preprocess.cross_section_rank`，含自检）；
空值（停牌、没有资金流）填 0。每天独立计算、不需要拟合，所以没有跨期泄漏。
“前一交易日”指这只股票自己的上一个有行情的交易日（停牌后复牌的第一天，收益跨过停牌期）。

构建与核对：python -X utf8 -m ml.p1_6_build_seq [--rebuild]
产物：data/ml/seq/X.npy（float16）、codes.npy、dates.npy、panel_j.npy（主面板每行的股票下标）、meta.json
"""

from __future__ import annotations

import json
import time
from datetime import date

import numpy as np
import pandas as pd

from ml.common import connect, data_dir
from ml.features_extra import TOL, _MAIN
from ml.preprocess import cross_section_rank

CHANNELS: tuple[str, ...] = ("ret", "ovn", "intra", "hl", "clv", "ln_amt", "ln_turn", "mf", "big", "limit", "traded")
RANKED: tuple[str, ...] = CHANNELS[:9]
WINDOW = 60

SQL = f"""
WITH base AS (
  SELECT d.ts_code, d.trade_date, d.open, d.high, d.low, d.close, d.amount,
         d.close * a.adj_factor AS c, d.open * a.adj_factor AS o,
         l.up_limit, l.down_limit, b.turnover_rate_f AS tr,
         m.net_mf_amount AS mf_raw,
         m.buy_lg_amount + m.buy_elg_amount - m.sell_lg_amount - m.sell_elg_amount AS big_raw,
         CASE WHEN m.ts_code IS NOT NULL AND d.amount > 0 THEN d.amount / 10.0 END AS amt_wan   -- 千元 → 万元
  FROM daily d
  JOIN adj_factor a USING (ts_code, trade_date)
  LEFT JOIN daily_limit l USING (ts_code, trade_date)
  LEFT JOIN daily_basic b USING (ts_code, trade_date)
  LEFT JOIN moneyflow m USING (ts_code, trade_date)
  WHERE d.trade_date <= $end AND {_MAIN}
),
r AS (
  SELECT ts_code, trade_date,
    CASE WHEN c > 0 AND lag(c) OVER s > 0 THEN ln(c / lag(c) OVER s) END AS ret,
    CASE WHEN o > 0 AND lag(c) OVER s > 0 THEN ln(o / lag(c) OVER s) END AS ovn,
    CASE WHEN open > 0 AND close > 0 THEN ln(close / open) END AS intra,
    CASE WHEN low > 0 THEN ln(high / low) END AS hl,
    CASE WHEN high > low THEN (2 * close - high - low) / (high - low) ELSE 0 END AS clv,
    CASE WHEN amount > 0 THEN ln(amount) END AS ln_amt,
    CASE WHEN tr > 0 THEN ln(tr) END AS ln_turn,
    mf_raw / amt_wan AS mf,
    big_raw / amt_wan AS big,
    CASE WHEN up_limit IS NOT NULL AND close >= up_limit - {TOL} THEN 0.5
         WHEN down_limit IS NOT NULL AND close <= down_limit + {TOL} THEN -0.5
         ELSE 0.0 END AS "limit"
  FROM base WINDOW s AS (PARTITION BY ts_code ORDER BY trade_date)
)
SELECT * FROM r WHERE trade_date BETWEEN $start AND $end ORDER BY trade_date, ts_code
"""


def raw_range(con, start: date, end: date) -> pd.DataFrame:
    """[start, end] 内每只主板股票每天的原始通道值（未排名）。只读到 end 为止的数据（防前视）。"""
    df = con.execute(SQL, {"start": start, "end": end}).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for c in RANKED + ("limit",):
        df[c] = df[c].astype(np.float64)
    return df


def ranked(df: pd.DataFrame) -> pd.DataFrame:
    """前 9 个通道每天截面排名到 (−0.5, 0.5)，空值填 0；加 traded = 1。"""
    day = df["trade_date"].to_numpy()
    r = cross_section_rank(df[list(RANKED)], day).fillna(0.0)
    r["limit"] = df["limit"].to_numpy(np.float32)
    r["traded"] = np.float32(1.0)
    return r[list(CHANNELS)]


def calendar(con) -> pd.DatetimeIndex:
    """全部交易日（与主面板的 di 一一对应：di = 在这个序列里的位置）。"""
    return pd.DatetimeIndex(pd.to_datetime(con.execute("SELECT DISTINCT trade_date FROM daily ORDER BY 1").df()["trade_date"]))


def main_codes(con) -> np.ndarray:
    return con.execute(f"SELECT DISTINCT d.ts_code FROM daily d JOIN adj_factor a USING (ts_code, trade_date) "
                       f"WHERE {_MAIN} ORDER BY 1").df()["ts_code"].to_numpy().astype(str)


def build(log=print) -> dict:
    """按年分批算，填进 [股票, 交易日, 11] 的 float16 张量。"""
    t0 = time.time()
    con = connect()
    dates, codes = calendar(con), main_codes(con)
    X = np.zeros((len(codes), len(dates), len(CHANNELS)), dtype=np.float16)
    jpos = pd.Index(codes)
    for y in range(dates[0].year, dates[-1].year + 1):
        t1 = time.time()
        raw = raw_range(con, date(y, 1, 1), date(y, 12, 31))
        r = ranked(raw)
        j = jpos.get_indexer(raw["ts_code"])
        d = dates.get_indexer(raw["trade_date"])
        if (j < 0).any() or (d < 0).any():
            raise RuntimeError(f"{y}：有行找不到股票或交易日下标")
        X[j, d, :] = r.to_numpy(np.float32).astype(np.float16)
        log(f"{y}：{len(raw):,} 行，{time.time() - t1:.1f}s")
    con.close()
    return {"X": X, "codes": codes, "dates": dates, "seconds": round(time.time() - t0, 1)}


def panel_index(codes: np.ndarray, panel_codes: pd.Series) -> np.ndarray:
    """主面板每行对应张量里的股票下标。"""
    j = pd.Index(codes).get_indexer(panel_codes.astype(str))
    if (j < 0).any():
        raise RuntimeError(f"主面板有 {(j < 0).sum()} 行在序列张量里找不到股票")
    return j.astype(np.int32)


def window(X: np.ndarray, j: np.ndarray, d: np.ndarray, length: int = WINDOW) -> np.ndarray:
    """取样本 (j, d) 的输入序列 X[j, d−length+1 : d+1]，形状 [样本, length, 通道]。给检查和测试用；
    训练时在显卡上用同样的下标切（`ml/lstm_train.py`）。"""
    offs = np.arange(-length + 1, 1)
    dd = d[:, None] + offs[None, :]
    if (dd < 0).any():
        raise ValueError("窗口越过了第一天")
    return X[j[:, None], dd]


def seq_dir():
    p = data_dir() / "seq"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save(out: dict, panel_j: np.ndarray) -> None:
    p = seq_dir()
    np.save(p / "X.npy", out["X"])
    np.save(p / "codes.npy", out["codes"])
    np.save(p / "dates.npy", out["dates"].values.astype("datetime64[D]"))
    np.save(p / "panel_j.npy", panel_j)
    (p / "meta.json").write_text(json.dumps({"channels": CHANNELS, "window": WINDOW, "shape": list(out["X"].shape),
                                             "seconds": out["seconds"]}, ensure_ascii=False, indent=1), encoding="utf-8")


def load_X(mmap: bool = True) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    p = seq_dir()
    X = np.load(p / "X.npy", mmap_mode="r" if mmap else None)
    return X, np.load(p / "codes.npy"), pd.DatetimeIndex(np.load(p / "dates.npy")), np.load(p / "panel_j.npy")
