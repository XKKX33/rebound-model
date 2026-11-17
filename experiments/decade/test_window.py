from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import train_cadence as tc
from window_eval import VARIANTS, comparisons
from window_report import recommend, user_verdict, verdict


def test_window_start_goes_back_n_years_but_not_before_the_start(monkeypatch):
    dates = pd.bdate_range("2013-01-01", "2020-12-31")
    di = np.repeat(np.arange(len(dates)), 2)  # 每天两行，按日期排好
    f = int(dates.searchsorted(pd.Timestamp("2019-01-02")))
    assert tc.window_start(f, dates, di, 0) == 0  # 不给窗口：从头扩张
    monkeypatch.setattr(tc, "WINDOW_YEARS", 2)
    first = int(dates.searchsorted(pd.Timestamp("2017-01-02")))
    assert tc.window_start(f, dates, di, 0) == 2 * first
    assert tc.window_start(f, dates, di, 2 * first + 10) == 2 * first + 10
    early = int(dates.searchsorted(pd.Timestamp("2014-06-02")))
    assert tc.window_start(early, dates, di, 0) == 0  # 前面不够 2 年：用全部


def test_fit_name_tags_the_window_and_leaves_the_default_unchanged(monkeypatch):
    day = pd.Timestamp("2020-01-02")
    legacy = hashlib.sha256(json.dumps(["k", tc.PARAMS, tc.FEATURES, 10, 5, "batches"]).encode()).hexdigest()[:8]
    assert tc.fit_name(day, 10, 5, "k") == f"o1o5_20200102_{legacy}"
    monkeypatch.setattr(tc, "WINDOW_YEARS", 2)
    two = tc.fit_name(day, 10, 5, "k")
    monkeypatch.setattr(tc, "WINDOW_YEARS", 3)
    assert len({two, tc.fit_name(day, 10, 5, "k"), f"o1o5_20200102_{legacy}"}) == 3


def test_window_years_refuses_to_write_into_the_original_directory(monkeypatch):
    monkeypatch.setattr(tc, "DATA", Path(tc.__file__).resolve().parent)
    monkeypatch.setattr(sys, "argv", ["train_cadence.py", "--window-years", "2"])
    with pytest.raises(SystemExit, match="DECADE_DATA"):
        tc.main()
    assert tc.WINDOW_YEARS is None


def test_comparisons_cover_both_cadences_and_the_users_design_against_the_monthly_model():
    y, m = comparisons("y"), comparisons("m")
    assert {c["kind"] for c in y} == {"Y", "Y vs V0-M"} and {c["kind"] for c in m} == {"Y", "Y vs V0-M", "M"}
    for c in m:
        assert c["b"][0] in VARIANTS and c["a"][0] == "V0" and c["a"][2] == c["b"][2]
        if c["kind"] == "Y vs V0-M":
            assert c["a"][1] == "M" and c["b"][1] == "Y"
        else:
            assert c["a"][1] == c["b"][1] == c["kind"]
    assert not [c for c in m if c["case"] == "原方案" and c["kind"] != "Y"]


def test_verdicts_follow_the_registered_rule():
    def c(diff, lo, hi):
        return {"2017起": {"diff": diff, "ci": [lo, hi]}}
    assert verdict(c(2.0, 0.1, 4.0), 0) == "确有改进"
    assert verdict(c(1.0, -1.0, 3.0), 60) == "可能有改进" and verdict(c(1.0, -1.0, 3.0), 59) == "没有改进"
    assert verdict(c(-1.0, -3.0, 1.0), 90) == "没有改进" and verdict(None, 90) == "还没跑"
    assert user_verdict(c(2.0, 0.5, 4.0)) == "更好" and user_verdict(c(-2.0, -4.0, -0.5)) == "更差"
    assert user_verdict(c(1.0, -1.0, 3.0)) == "分不出"


def test_only_a_confirmed_monthly_improvement_switches_the_window():
    maybe = {w: {"M": "可能有改进"} for w in VARIANTS}
    assert recommend(None, maybe).startswith("每月重训还没跑完")
    assert recommend({}, maybe).startswith("保留全历史")
    assert "W3" in recommend({}, {**maybe, "W3": {"M": "确有改进"}})


def test_report_from_the_self_check_keeps_full_history():
    import window_report as wr
    path = wr.OUT / "window_eval_y_selfcheck.json"
    if not path.is_file():
        pytest.skip("要先跑 window_eval.py --stage y --self-check")
    y = json.loads(path.read_text(encoding="utf-8"))
    text, v = wr.render(y, None, {}, {})
    assert v["W2"]["Y"] == "没有改进" and v["recommend"].startswith("每月重训还没跑完")
    assert "+0.0 [+0.0, +0.0]" in text
