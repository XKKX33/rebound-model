"""十年实验的公用部分：日 × 股的坐标轴、行情矩阵、试验登记。模拟器和成本模型直接用 p0p1 的 engine / sim。"""
from __future__ import annotations

import hashlib
import os
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P0P1 = HERE.parent / "shared"
sys.path.insert(0, str(P0P1))
sys.path.insert(0, str(HERE.parent / "shared"))

from engine import Market  # noqa: E402

#: 面板、标签、打分、结果都写在 DATA（默认本目录；冒烟测试用环境变量 DECADE_DATA 指到临时目录）
DATA = Path(os.environ.get("DECADE_DATA", HERE))
#: 十年库（2013-01 ~ 2026-10，表和 `rebound sync` 建的库相同）；面板、行业、指数都只读它。
#: 当时是另外拼的一份研究库；在本仓库里用 `rebound sync --days 3500` 建好 data/market.duckdb，再设 DECADE_DB 指向它
DB = Path(os.environ.get("DECADE_DB", ROOT / "data/research-2013-2026.duckdb"))
PANEL, AXES = DATA / "panel10.parquet", DATA / "axes.json"
SIGNAL_FROM = pd.Timestamp("2015-01-01")
TAIL = 5  # 最后 5 个开市日算不出 T+5


def axes() -> tuple[np.ndarray, pd.DatetimeIndex]:
    a = json.loads(AXES.read_text(encoding="utf-8"))
    return np.array(a["codes"]), pd.DatetimeIndex(pd.to_datetime(a["dates"])).astype("datetime64[ns]")


def row_index(ts_code: pd.Categorical, trade_date, codes: np.ndarray, dates: pd.DatetimeIndex):
    """面板的行 → (日下标, 股下标)。ts_code 是 Categorical，按类别映射一次再广播，省内存。"""
    cat = pd.Categorical(ts_code)
    per_cat = pd.Index(codes).get_indexer(cat.categories)
    if (per_cat < 0).any():
        raise SystemExit("面板里有 axes.json 没有的股票：先重跑 panel10.py")
    ji = per_cat[cat.codes]
    di = dates.get_indexer(pd.DatetimeIndex(trade_date).astype("datetime64[ns]"))
    if (di < 0).any():
        raise SystemExit("面板里有 axes.json 没有的日子：先重跑 panel10.py")
    return di.astype(np.int64), ji.astype(np.int64)


def signal_range(dates: pd.DatetimeIndex) -> tuple[int, int]:
    """信号日下标 [first, last]：2015 年第一个开市日起，到 T+5 还在数据里的最后一天。"""
    return int(dates.searchsorted(SIGNAL_FROM)), len(dates) - 1 - TAIL


def industries(codes: np.ndarray) -> np.ndarray:
    con = duckdb.connect(str(DB), read_only=True)
    try:
        basic = con.execute("SELECT ts_code, industry FROM stock_basic").df()
    finally:
        con.close()
    ind_of = dict(zip(basic.ts_code, basic.industry.fillna("")))
    names = {name: i for i, name in enumerate(sorted({ind_of.get(c, "") for c in codes}))}
    return np.array([names[ind_of.get(c, "")] for c in codes], dtype=np.int32)


def load_market10(raw: bool = True, amount: bool = False) -> tuple[Market, np.ndarray, np.ndarray | None]:
    """十年行情 → (Market（复权价、受阻、准入池，raw 是未复权开收盘）, 收盘 < 2 元, 全天成交额（元）)。"""
    codes, dates = axes()
    cols = ["ts_code", "trade_date", "o", "c", "up_open", "up_close", "dn_open", "dn_close", "eligible", "below2"]
    cols += (["raw_o", "raw_c"] if raw else []) + (["amount"] if amount else [])
    p = pq.read_table(PANEL, columns=cols, read_dictionary=["ts_code"]).to_pandas()
    di, ji = row_index(p.pop("ts_code"), p.pop("trade_date"), codes, dates)
    shape = (len(dates), len(codes))

    def mat(col: str, fill, dtype, scale: float = 1.0) -> np.ndarray:
        """取一列填进 日 × 股 矩阵，并从 p 里删掉这一列（边建边放内存）。"""
        out = np.full(shape, fill, dtype=dtype)
        values = p.pop(col).to_numpy()
        out[di, ji] = values * scale if scale != 1.0 else values
        return out

    m = Market(dates=dates, codes=codes,
               price={"open": mat("o", np.nan, np.float64), "close": mat("c", np.nan, np.float64)},
               buy_blocked={"open": mat("up_open", False, bool), "close": mat("up_close", False, bool)},
               sell_blocked={"open": mat("dn_open", False, bool), "close": mat("dn_close", False, bool)},
               eligible=mat("eligible", False, bool), industry=industries(codes),
               raw={"open": mat("raw_o", np.nan, np.float64), "close": mat("raw_c", np.nan, np.float64)} if raw else None)
    low = mat("below2", False, bool)
    money = mat("amount", np.nan, np.float64, scale=1000.0) if amount else None  # Tushare 成交额单位是千元
    return m, low, money


def register(stage: str, config: dict, metrics: dict) -> None:
    """试验登记（本目录的 registry.jsonl），供以后按试验次数打折扣。"""
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    rec = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage,
           "fingerprint": hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12], "config": config, "metrics": metrics}
    with (DATA / "registry.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
