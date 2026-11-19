"""5.1 回测适配器：分数 → 行情矩阵下标的对齐，以及交给仓库 `market.candidates` 后的排名（6.6 要求的“适配器的下标对齐”）。"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from ml.p5_backtest import to_market_index
from rebound.backtest.market import candidates


@dataclass
class FakeMarket:
    """只有适配器和 candidates 用到的几个字段：日期轴、代码轴。"""
    dates: pd.DatetimeIndex
    codes: np.ndarray

    def code_index(self, ts_code):
        return pd.Index(self.codes).get_indexer(np.asarray(ts_code))


M = FakeMarket(pd.DatetimeIndex(["2025-01-02", "2025-01-03", "2025-01-06"]), np.array(["000001.SZ", "000002.SZ", "600000.SH"]))


def test_index_alignment_and_ranks():
    # 顺序故意打乱：适配器按日期、代码找下标，不依赖行的顺序
    sc = pd.DataFrame({"trade_date": pd.to_datetime(["2025-01-03", "2025-01-02", "2025-01-03", "2025-01-02", "2025-01-03"]),
                       "ts_code": ["600000.SH", "000002.SZ", "000001.SZ", "600000.SH", "000002.SZ"],
                       "score": [0.9, 0.1, 0.5, 0.7, 0.2]})
    t = to_market_index(M, sc)
    assert t[["di", "ji"]].values.tolist() == [[1, 2], [0, 1], [1, 0], [0, 2], [1, 1]]
    assert t["score"].tolist() == sc["score"].tolist()
    cands, rank = candidates(M, t)
    assert cands[1].tolist() == [2, 0, 1]                       # 1 月 3 日：600000（0.9）> 000001（0.5）> 000002（0.2）
    assert cands[0].tolist() == [2, 1]                          # 1 月 2 日：000001 没有分数，不进候选
    assert rank[1].tolist() == [2, 3, 1] and np.isinf(rank[0, 0]) and np.isinf(rank[2]).all()


def test_unknown_or_duplicate_rows_raise():
    bad_code = pd.DataFrame({"trade_date": pd.to_datetime(["2025-01-02"]), "ts_code": ["300001.SZ"], "score": [0.5]})
    with pytest.raises(ValueError):
        to_market_index(M, bad_code)                            # 代码不在行情矩阵里（比如创业板）
    bad_day = pd.DataFrame({"trade_date": pd.to_datetime(["2025-01-04"]), "ts_code": ["000001.SZ"], "score": [0.5]})
    with pytest.raises(ValueError):
        to_market_index(M, bad_day)                             # 不是交易日
    dup = pd.DataFrame({"trade_date": pd.to_datetime(["2025-01-02"] * 2), "ts_code": ["000001.SZ"] * 2, "score": [0.5, 0.6]})
    with pytest.raises(ValueError):
        to_market_index(M, dup)                                 # 同一天同一只票两个分数，会被当成两只候选


def test_seed_columns():
    sc = pd.DataFrame({"trade_date": pd.to_datetime(["2025-01-02"]), "ts_code": ["000001.SZ"], "score": [0.5], "s1": [0.8]})
    assert to_market_index(M, sc, col="s1")["score"].tolist() == [0.8]
