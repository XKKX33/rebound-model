"""股票池（`rebound/universe.py`）：价格下限、ST 和退市整理按名称排除，以及两个版本的口径。"""

from __future__ import annotations

import pandas as pd

from rebound import config
from rebound import params as model_params
from rebound.universe import apply_universe


def _snap(*closes: float) -> pd.DataFrame:
    rows = [(f"60000{i}.SH", f"票{i}", "A", c, 1.0, 9e4) for i, c in enumerate(closes)]
    return pd.DataFrame(rows, columns=["ts_code", "name", "industry", "close", "pct_chg", "amount"])


def test_both_variants_exclude_prices_below_two_yuan() -> None:
    """面值退市风险：两个版本都排除收盘价 < 2 元的票。"""
    assert {v.price_min for v in model_params.VARIANTS.values()} == {2.0}
    assert model_params.VARIANTS["nocap"].price_max is None
    assert model_params.VARIANTS["cap70"].price_max == 70.0


def test_floor_keeps_exactly_two_and_drops_anything_below() -> None:
    universe = config.Universe(price_min=2.0)
    kept = apply_universe(_snap(1.99, 2.0, 2.01, 69.99, 70.0), universe)
    assert list(kept["close"]) == [2.0, 2.01, 69.99]


def test_no_floor_by_default() -> None:
    assert config.Universe().price_min is None
    assert len(apply_universe(_snap(0.5, 1.99, 10.0), config.Universe())) == 3


def test_st_and_delisting_period_names_are_excluded() -> None:
    """名称里有 ST、*、退（退市整理期的"退市XX""XX退"）都不进池。"""
    names = ["浦发银行", "ST甲", "*ST乙", "退市国化", "国恒退"]
    snap = pd.DataFrame([(f"60000{i}.SH", n, "A", 10.0, 1.0, 9e4) for i, n in enumerate(names)],
                        columns=["ts_code", "name", "industry", "close", "pct_chg", "amount"])
    universe = config.Universe(board="mainboard", exclude_st=True, price_max=70.0, min_amount_yi=0.8, price_min=2.0)
    assert apply_universe(snap, universe)["name"].tolist() == ["浦发银行"]
