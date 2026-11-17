from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from common import row_index
from decade_report import render
from decade_sim import paired_diff, period
from train_cadence import CADENCES, block_ends, fit_days, score_end

DATES = pd.bdate_range("2014-06-02", "2026-10-23")


def test_fit_days_are_month_starts_inside_the_signal_range_and_nested():
    d_first, d_last = int(DATES.searchsorted(pd.Timestamp("2015-01-01"))), len(DATES) - 6
    days = {c: fit_days(DATES, months, d_first, d_last) for c, months in CADENCES.items()}
    assert days["Y"][0] == d_first and DATES[days["Y"][0]] == pd.Timestamp("2015-01-01")
    assert len(days["Y"]) == 12 and len(days["Q"]) == 4 * 11 + 4 and len(days["M"]) == 12 * 11 + 10
    assert set(days["Y"]) <= set(days["H"]) <= set(days["Q"]) <= set(days["M"])
    assert all(DATES[d].day <= 3 for d in days["M"]) and all(d_first <= d <= d_last for d in days["M"])


def test_each_model_scores_only_until_the_next_refit_and_never_beyond_one_year():
    d_first, d_last = int(DATES.searchsorted(pd.Timestamp("2015-01-01"))), len(DATES) - 6
    for months in CADENCES.values():
        days = fit_days(DATES, months, d_first, d_last)
        ends = block_ends(days, len(DATES))
        assert ends[:-1] == days[1:] and ends[-1] == len(DATES)
        assert all(stop <= score_end(DATES, f) for f, stop in zip(days, ends))


def test_period_recovers_beta_alpha_and_total():
    dates = pd.bdate_range("2023-01-02", periods=300)
    rng = np.random.default_rng(1)
    pool = np.r_[np.nan, rng.normal(0.0005, 0.01, 299)]
    nav = np.r_[1.0, np.cumprod(1 + 2 * pool[1:])]  # 每天 = 2 × 准入池：β = 2，α = −(准入池年化平均日收益)
    idx = np.arange(1, 300)
    p = period(nav, pool, dates, idx)
    assert p["beta"] == pytest.approx(2.0, abs=1e-3)
    assert p["alpha"] == pytest.approx(0.0, abs=1e-6) and p["excess"] == pytest.approx(np.mean(pool[1:]) * 242 * 100, abs=0.01)
    assert p["total"] == pytest.approx((nav[-1] - 1) * 100, abs=0.01) and p["days"] == 299


def test_paired_diff_of_a_constant_daily_edge():
    dates = pd.bdate_range("2023-01-02", periods=300)
    base = np.r_[1.0, np.cumprod(np.full(299, 1.002))]
    better = np.r_[1.0, np.cumprod(np.full(299, 1.003))]
    d = paired_diff(better, base, dates, np.arange(1, 300))
    assert d["ann"] == pytest.approx(24.2, abs=1e-6) and d["ci"] == pytest.approx([24.2, 24.2], abs=1e-6)
    assert d["better_years"] == d["n_years"] == 2


def test_row_index_maps_and_refuses_unknown_codes():
    codes, dates = np.array(["A", "B"]), pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    di, ji = row_index(pd.Categorical(["B", "A"]), pd.to_datetime(["2024-01-03", "2024-01-02"]), codes, dates)
    assert di.tolist() == [1, 0] and ji.tolist() == [1, 0]
    with pytest.raises(SystemExit):
        row_index(pd.Categorical(["C"]), pd.to_datetime(["2024-01-02"]), codes, dates)


def _p(total: float, excess: float, beta: float = 1.2, alpha: float = 1.0) -> dict:
    return {"days": 242, "total": total, "cagr": total, "excess": excess, "excess_ci": [excess - 5, excess + 5],
            "beta": beta, "alpha": alpha, "max_dd": -20.0}


def _rule(bump: float) -> dict:
    years = {y: _p(t + bump, t + bump - 1) for y, t in (("2015", 5.0), ("2016", -3.0))}
    tier = {"all": _p(10 + bump, 4 + bump), "years": years, "p0p1": _p(9, 3), "final": 300_000.0, "idle_cash_pct": 2.0}
    return {"hist": tier, "real": tier, "prop": tier}


def test_render_report_tables_and_verdict():
    main = ("FH-K6-next-o1o5", "FH-K2-next-o1o5-small")
    out = {"capital": 200_000.0, "main": main,
           "context": {"pool": {"all": _p(1.0, 0.0), "years": {"2015": _p(5.0, 0.0), "2016": _p(-3.0, 0.0)}},
                       "index": {"沪深300": {"2015": 6.0, "2016": -10.0, "all": -2.0}}, "pool_size": {"2015": 900, "2016": 1100},
                       "first": "2015-01-13", "last": "2017-10-19"},
           "results": {"Y": {n: _rule(1.0) for n in main}, "Q": {n: _rule(2.0) for n in main}},
           "cadence_vs_yearly": {f"Q|{n}|hist": {"ann": 1.5, "ci": [0.2, 2.8], "better_years": 2, "n_years": 2}
                                 for n in main}}
    text = render(out)
    assert "| 2015 | +5.0 | +6.0 | 900 |" in text and "| 2015 | +5.0 | +6.0 / +5.0 | +7.0 / +6.0 |" in text
    assert "| 按季 | +12.0% | +6.0 [+1.0, +11.0] | 1.20 | +1.0 | -20.0% | 300,000 | 2.0% |" in text
    assert "| 准入池上涨的年份 | 2015 | +5.0 / +1.0 | +5.0 / +1.0 |" in text
    assert "| 准入池下跌的年份 | 2016 | -3.0 / +1.0 | -3.0 / +1.0 |" in text
    assert "按季：比按年好（两条主规则的区间下限都 > 0）" in text and "2 / 2" in text
