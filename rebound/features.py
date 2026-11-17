"""反弹模型的 28 个量价特征：研究和正式共用这一份实现。

口径和研究版（`data/research/rank-readiness-20261028/stage_data.py` 的 SQL + 后处理）
逐条相同，研究面板和这里的输出逐格核对过（见 `data/research/model-parity-20261031/`）：

- 只取主板、而且有复权因子的行（和研究版一样内连接 `adj_factor`）；
- 收益、均线、高低点用复权价（价 × 复权因子），涨跌停判定、准入池用不复权价；
- 窗口按这只票自己的交易日数，停牌日不补；`bar_no` 从这只票在库里的第一根算起；
- 窗口不满的特征为空（比如 60 日低点要 60 根）；MACD 两个比值前 33 根为空；
- 截面排名在当天全部主板行里算（含 ST，和研究版一样）。

为什么多读 `LOOKBACK_BARS` 根：MACD 的指数均线理论上要从第一根算起。往前多读 300 根，
截断误差（≈ (25/27)^300 ≈ 1e-10）在 float32 的末位以下。其余窗口最长 120 根，300 根足够。
每天打分、按年训练、全量重算的特征**不是逐位相同**：区间起点不同，DuckDB 窗口求和的顺序、
指数均线的起点就不同，差在 float32 的末位；`ma20_gt_ma60` 在两条均线数学上相等时会翻转。
和十年实验（全历史一次算完）的核对：28 个特征里 21 个逐位相同，其余只差末位
（`data/research/model-parity-20261031/RESULTS.md`）。

准入池（`eligible`）的 ST 和退市整理按**当时的名称**判（`namechange`）。式子和十年实验 `panel10.write_final` 相同，
只多排了退市整理期的"退市XX""XX退"（2026-10-31 补上；十年实验没排，2015 年起每个版本差 12 个池日）。
库里没有 `namechange` 就拒绝：按现在的名称判会把后来才戴帽的票提前剔掉（前视）。

核对结果（2026-10-31，30 个信号日、94,994 行）：27 个特征和研究版差 ≤ 7.5e-9；
`ma20_gt_ma60` 有 4 格不同，都是 20 日均线和 60 日均线数学上相等、只差 1e-14 的求和顺序，
谁大谁小是浮点噪声。准入池、标签完全一致。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd

from rebound.config import Universe
from rebound.data.errors import DataError
from rebound.store import Store
from rebound.universe import NON_MAIN_PREFIXES, ST_PATTERN, universe_mask

#: 顺序即模型输入的列顺序：改顺序等于换了一个模型。
FEATURES: tuple[str, ...] = (
    "pct_chg", "r1", "r5", "r10", "r20", "r60", "r120", "from_low60", "from_high60", "dev_ma5",
    "dev_ma20", "dev_ma60", "ma20_gt_ma60", "ma20_slope5", "ma60_slope10", "dd20", "vol5", "vol20",
    "amp20", "launch10", "up_days10", "max_up20", "limit_up_today", "limit_up20", "dif_ratio",
    "hist_ratio", "rank_r20", "rank_r60",
)
#: 每只票在区间起点之前多读的根数（理由见模块说明）。
LOOKBACK_BARS = 300
#: 价格"贴着"涨跌停价的容差（元），与研究版相同。
TOL = 1e-6
#: MACD 的预热根数：前 33 根的两个比值为空。
MACD_WARM_BARS = 33

_PANDAS_FEATURES = ("dif_ratio", "hist_ratio", "rank_r20", "rank_r60")
_SQL_FEATURES = tuple(f for f in FEATURES if f not in _PANDAS_FEATURES)
_NON_MAIN = "^(" + "|".join(NON_MAIN_PREFIXES) + ")"


def _win(lo: int, hi: str = "CURRENT ROW") -> str:
    return f"(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN {lo} PRECEDING AND {hi})"


def _touch(limit: str, cols: tuple[str, ...]) -> str:
    """这几个价都贴着涨（跌）停价；没有涨跌停价或价格为空时为假（和研究版 pandas 的写法同义）。"""
    hits = " AND ".join(f"coalesce(abs({col} - {limit}) <= {TOL}, false)" for col in cols)
    return f"({limit} IS NOT NULL AND {hits})"


_FOUR = ("open", "high", "low", "close")
_FLOAT = ", ".join(
    f"CAST(CASE WHEN isfinite(CAST({x} AS DOUBLE)) THEN CAST({x} AS DOUBLE) END AS FLOAT) AS {x}"
    for x in _SQL_FEATURES
)

#: 研究版的 SQL 原样搬来，只多两件事：`bar_no` 按全历史算（研究版是全历史一次算完），
#: 每只票只留区间起点前 `LOOKBACK_BARS` 根以来的行。参数：`$start`、`$end`。
_SQL = f"""
WITH j AS (
  SELECT d.ts_code, d.trade_date, d.open, d.high, d.low, d.close, d.pct_chg, d.amount,
         d.close * a.adj_factor AS c, d.open * a.adj_factor AS o, l.up_limit, l.down_limit
  FROM daily d JOIN adj_factor a USING (ts_code, trade_date) LEFT JOIN daily_limit l USING (ts_code, trade_date)
  WHERE d.trade_date <= $end AND NOT regexp_matches(d.ts_code, '{_NON_MAIN}')
), n AS (
  SELECT *, row_number() OVER (PARTITION BY ts_code ORDER BY trade_date) - 1 AS bar_no,
         count(*) FILTER (WHERE trade_date < $start) OVER (PARTITION BY ts_code) AS n_before
  FROM j
), b AS (
  SELECT ts_code, trade_date, open, high, low, close, pct_chg, amount, c, o, up_limit, down_limit, bar_no,
         CASE WHEN abs(close - up_limit) <= {TOL} THEN 1.0 ELSE 0.0 END AS lu,
         {_touch("up_limit", _FOUR)} AS up_open, {_touch("up_limit", ("close",))} AS up_close,
         {_touch("down_limit", _FOUR)} AS dn_open, {_touch("down_limit", ("close",))} AS dn_close
  FROM n WHERE bar_no >= n_before - {LOOKBACK_BARS}
), w AS (
  SELECT *,
    c / lag(c, 1) OVER s - 1 AS r1, c / lag(c, 5) OVER s - 1 AS r5, c / lag(c, 10) OVER s - 1 AS r10,
    c / lag(c, 20) OVER s - 1 AS r20, c / lag(c, 60) OVER s - 1 AS r60, c / lag(c, 120) OVER s - 1 AS r120,
    avg(c) OVER w5 AS ma5, avg(c) OVER w20 AS ma20, avg(c) OVER w60 AS ma60, max(c) OVER w20 AS hi20,
    min(c) OVER w60 AS lo60, max(c) OVER w60 AS hi60, avg((high - low) / close) OVER w20 AS amp,
    max(pct_chg) OVER w20 AS maxup, sum(lu) OVER w20 AS lu20, avg(amount) OVER p5 AS amt5,
    avg(amount) OVER p20 AS amt20, sum(CASE WHEN pct_chg >= 5 OR lu = 1 THEN 1 ELSE 0 END) OVER p10 AS ln10,
    sum(CASE WHEN pct_chg > 0 THEN 1 ELSE 0 END) OVER p10 AS up10
  FROM b
  WINDOW s AS (PARTITION BY ts_code ORDER BY trade_date), w5 AS {_win(4)}, w20 AS {_win(19)},
    w60 AS {_win(59)}, p5 AS {_win(5, "1 PRECEDING")}, p10 AS {_win(10, "1 PRECEDING")},
    p20 AS {_win(20, "1 PRECEDING")}
), f AS (
  SELECT ts_code, trade_date, bar_no, open, close, amount, o, c, up_open, up_close, dn_open, dn_close,
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
)
SELECT ts_code, trade_date, CAST(bar_no AS INTEGER) AS bar_no, open, close, amount, o, c,
       up_open, up_close, dn_open, dn_close, r20 AS r20_raw, r60 AS r60_raw, {_FLOAT}
FROM f ORDER BY ts_code, trade_date
"""

#: 当时的名称：`namechange` 里生效日不晚于当天的最后一个名字（同一生效日取公告日最晚的），
#: 没有记录才用 `stock_basic` 的现在的名称（和十年实验 `panel10.write_final` 同一个式子）。
#: 结果只留一个布尔：`name_ok` = 当时的名称有、而且不是 ST。
_NAMES_SQL = f"""
WITH names AS (
  SELECT ts_code, name, start_date FROM (
    SELECT *, row_number() OVER (PARTITION BY ts_code, start_date ORDER BY ann_date DESC NULLS LAST, name) AS k
    FROM namechange) WHERE k = 1
)
SELECT f.*, coalesce(NOT regexp_matches(coalesce(n.name, b.name), '{ST_PATTERN}'), false) AS name_ok
FROM ({{inner}}) f
ASOF LEFT JOIN names n ON f.ts_code = n.ts_code AND f.trade_date >= n.start_date
LEFT JOIN stock_basic b ON f.ts_code = b.ts_code
ORDER BY f.ts_code, f.trade_date
"""
#: 输出的列：键、根数、不复权开收盘与成交额（准入池、真实账户用）、复权开收盘、涨跌停受阻、
#: 现在的名称与行业（展示、行业去重用）、当时的名称是否可入池、28 个特征。
COLUMNS: tuple[str, ...] = (
    "ts_code", "trade_date", "bar_no", "open", "close", "amount", "o", "c",
    "up_open", "up_close", "dn_open", "dn_close", "name", "industry", "name_ok", *FEATURES,
)


def feature_panel(store: Store, start: date, end: date) -> pd.DataFrame:
    """[start, end] 内每只主板票每个交易日一行：`COLUMNS` 那些列，按 (ts_code, trade_date) 升序。

    训练按段调、每天打分只调当天，走的是同一段代码。`end` 超出可见窗口时抛
    `LookaheadBlocked`（拒绝，而不是悄悄截到能看的那天）。
    """
    store.ensure_visible(end)
    _require_names(store)
    inner = _SQL.strip().removesuffix("ORDER BY ts_code, trade_date")
    raw = store.connection.execute(_NAMES_SQL.format(inner=inner), {"start": start, "end": end}).df()
    raw["ts_code"] = raw["ts_code"].astype("category")
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    if raw.empty:
        return _finish(raw, store)

    # MACD：按这只票自己的交易日算指数均线，含区间之前多读的那几百根
    by_code = raw.groupby("ts_code", sort=False, observed=True)["c"]
    dif = by_code.transform(_ema, 12) - by_code.transform(_ema, 26)
    dea = dif.groupby(raw["ts_code"], sort=False, observed=True).transform(_ema, 9)
    warm = raw["bar_no"] < MACD_WARM_BARS
    raw["dif_ratio"] = (dif / raw["c"]).mask(warm)
    raw["hist_ratio"] = (2 * (dif - dea) / raw["c"]).mask(warm)
    del dif, dea, by_code

    out = raw[raw["trade_date"] >= pd.Timestamp(start)].copy()
    del raw
    for col in ("r20", "r60"):  # 用 DOUBLE 排名（研究版也是先排名再转 float32）
        out[f"rank_{col}"] = out.groupby("trade_date")[f"{col}_raw"].rank(pct=True)
    for col in _PANDAS_FEATURES:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return _finish(out, store)


def _require_names(store: Store) -> None:
    empty = not store.has_table("namechange") or store.connection.execute(
        "SELECT count(*) FROM namechange").fetchone()[0] == 0
    if empty:
        raise DataError("库里没有历史名称（namechange）：准入池的 ST 要按当时的名称判，先跑 rebound sync")


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _finish(frame: pd.DataFrame, store: Store) -> pd.DataFrame:
    """补上名称、行业，空表也补齐 28 个特征列，按 `COLUMNS` 排好。"""
    for col in _PANDAS_FEATURES:
        if col not in frame:
            frame[col] = pd.Series(dtype=np.float32)
    basic = store.connection.execute("SELECT ts_code, name, industry FROM stock_basic").df()
    basic = basic.drop_duplicates("ts_code").set_index("ts_code")
    codes = frame["ts_code"].cat
    for col in ("name", "industry"):
        values = basic[col].reindex(codes.categories)
        labels, uniques = pd.factorize(values)
        frame[col] = pd.Categorical.from_codes(np.where(codes.codes >= 0, labels[codes.codes], -1), uniques)
    return frame.loc[:, list(COLUMNS)].reset_index(drop=True)


def fingerprint() -> str:
    """特征口径的指纹：特征名、SQL、当时名称的判法、多读根数、MACD 预热、容差。改了任何一项，旧模型就不能再用。"""
    blob = json.dumps([FEATURES, _SQL, _NAMES_SQL, LOOKBACK_BARS, MACD_WARM_BARS, TOL]).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def eligible(panel: pd.DataFrame, universe: Universe, *, min_bars: int) -> np.ndarray:
    """准入池：和规则策略同一套 universe 判据，再加"库里至少 `min_bars` 根日线"。

    ST 按当时的名称（`name_ok`）判，不按现在的：用现在的名称会把后来才戴帽的票提前剔掉（前视）。
    """
    mask = universe_mask(panel, replace(universe, exclude_st=False))
    if universe.exclude_st:
        mask &= panel["name_ok"].to_numpy(dtype=bool)
    return np.asarray(mask, dtype=bool) & (panel["bar_no"].to_numpy() >= min_bars - 1)
