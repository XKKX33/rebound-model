from __future__ import annotations

import hashlib
import json

import duckdb
import numpy as np
import pandas as pd
import pytest

import train_cadence as tc
from ind_features import COLS, query
from sim import YEAR
from theme_eval import TERCILES, breaking, comparisons, drop, jobs, paired, regime_gap


def test_fit_name_is_unchanged_without_extra_features_and_differs_with_them(monkeypatch):
    day = pd.Timestamp("2020-01-02")
    legacy = hashlib.sha256(json.dumps(["k", tc.PARAMS, tc.FEATURES, 10, 5, "batches"]).encode()).hexdigest()[:8]
    assert tc.fit_name(day, 10, 5, "k") == f"o1o5_20200102_{legacy}"
    monkeypatch.setattr(tc, "EXTRA", ("ind_r5",))
    assert tc.fit_name(day, 10, 5, "k") != f"o1o5_20200102_{legacy}"


def test_breaking_flags_earlier_winners_that_are_now_falling():
    di = np.array([0] * 5 + [1] * 5)
    r60 = np.array([1.0, 0.5, 0.2, 0.0, -0.1, 0.3, 0.3, 0.3, 0.3, np.nan])
    r20 = np.array([-0.1, 0.1, -0.05, -0.2, 0.0, -0.1, 0.1, 0.0, -0.1, -0.1])
    # 第 0 天前一段涨幅依次 1.22、0.36、0.26、0.25、−0.10；第 1 天 0.44、0.18、0.30、0.44、缺值（并列第一，百分位 0.375）
    assert breaking(di, r20, r60, 0.2).tolist() == [True] + [False] * 9
    assert breaking(di, r20, r60, 0.4).tolist() == [True, False, False, False, False, True, False, False, True, False]


def test_drop_only_removes_flagged_names_on_that_day_and_keeps_the_order():
    flag = np.zeros((5, 10), dtype=bool)
    flag[3, 9] = flag[2, 5] = True
    assert drop({3: np.array([5, 2, 9, 1])}, flag)[3].tolist() == [5, 2, 1]


def test_paired_is_zero_for_identical_books_and_signed_new_minus_old():
    when = pd.bdate_range("2025-06-02", periods=300)
    a = np.random.default_rng(0).normal(0, 0.01, 300)
    same = paired(a, a, when)
    assert same["max_abs_daily"] == 0 and same["全期"]["diff"] == 0
    up = paired(a, a + 0.001, when)
    assert up["全期"]["diff"] == pytest.approx(0.001 * YEAR * 100, abs=0.01) and up["2026"]["ci"][0] > 0


def test_regime_gap_compounds_by_month_and_averages_by_tercile():
    when = pd.bdate_range("2025-01-01", "2025-03-31")
    ret = np.where(when.month == 1, 0.01, 0.0)
    cut = pd.Series(pd.Categorical(["强势股月", "反转月", "反转月"], categories=list(TERCILES)),
                    index=pd.period_range("2025-01", periods=3, freq="M"))
    out = regime_gap(ret, np.zeros(len(when)), when, cut)
    assert out["强势股月"] == pytest.approx((1.01 ** 23 - 1) * 100, abs=0.01) and out["反转月"] == 0


def test_every_comparison_is_simulated_and_stage_q_screens_monthly_rules_on_quarterly_scores():
    for stage in ("q", "m"):
        done = set(jobs(stage))
        assert all(a in done and b in done for *_, a, b in comparisons(stage))
    cad = {(case, kind): c for case, kind, c, _, _ in comparisons("q")}
    assert cad[("20万推荐", "V1")] == "Q" and cad[("原方案", "V1")] == "Y" and cad[("20万推荐", "F20")] == "M"
    assert all(src == "V0" for src, _, _, f in jobs("q") if f)


def test_industry_features_use_the_name_at_the_time_and_skip_thin_industries(tmp_path):
    db = tmp_path / "t.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE namechange (ts_code VARCHAR, name VARCHAR, start_date DATE, ann_date DATE)")
    con.execute("INSERT INTO namechange VALUES ('A6.SZ', 'ST甲', '2024-01-01', '2024-01-01'), "
                "('A5.SZ', '乙', '2020-01-01', '2020-01-01'), ('A5.SZ', '*ST乙', '2025-06-01', '2025-06-01')")
    con.execute("CREATE TABLE stock_basic (ts_code VARCHAR, name VARCHAR, industry VARCHAR)")
    basic = [*(f"('A{i}.SZ', 'a{i}', 'A')" for i in range(1, 5)), "('A5.SZ', '*ST乙', 'A')", "('A6.SZ', 'ST甲', 'A')",
             *(f"('B{i}.SZ', 'b{i}', 'B')" for i in range(1, 5)), *(f"('C{i}.SZ', 'c{i}', 'C')" for i in range(1, 8)),
             "('D1.SZ', 'd1', NULL)"]
    con.execute("INSERT INTO stock_basic VALUES " + ", ".join(basic))
    con.close()
    rows = []
    for i in range(1, 7):  # A：A6 是 ST；A5 现在是 *ST，当时不是
        rows.append((f"A{i}.SZ", 200, False, i <= 5, 0.01 * i if i <= 5 else 0.9, 0.1 * i if i <= 5 else 9.0,
                     0.1 if i <= 5 else 9.0, 0.1 if i <= 2 else -0.1, 1.0 if i in (1, 5) else 0.0))
    rows += [(f"B{i}.SZ", 200, False, i == 1, 0.0, 0.5, 0.5, 0.1, 0.0) for i in range(1, 5)]  # B 只有 4 只
    rows += [(f"C{i}.SZ", 200, False, True, 0.0, 0.0, 0.0, 0.1, 0.0) for i in range(1, 6)]
    rows += [("C6.SZ", 30, False, False, 0.0, 5.0, None, 0.1, 0.0), ("C7.SZ", 200, True, False, 0.0, 5.0, 5.0, 0.1, 0.0)]
    rows.append(("D1.SZ", 200, False, True, 0.0, 0.0, 0.0, 0.1, 0.0))
    panel = pd.DataFrame(rows, columns=["ts_code", "bar_no", "below2", "eligible", "r5", "r20", "r60", "dev_ma20", "limit_up20"])
    panel.insert(1, "trade_date", pd.to_datetime(["2025-01-02"] * len(panel)).astype("datetime64[us]"))
    src = tmp_path / "panel.parquet"
    panel.to_parquet(src, index=False)
    con = duckdb.connect()
    con.execute(f"ATTACH '{db.as_posix()}' AS db (READ_ONLY)")
    out = con.execute(query(src)).df().set_index("ts_code")
    con.close()
    assert list(out.index) == [*(f"A{i}.SZ" for i in range(1, 6)), "B1.SZ", *(f"C{i}.SZ" for i in range(1, 6)), "D1.SZ"]
    a = out.loc["A1.SZ"]
    assert a.ind_r20 == pytest.approx(0.3, abs=1e-6) and a.rel_r20 == pytest.approx(-0.2, abs=1e-6)
    assert a.ind_breadth == pytest.approx(0.4, abs=1e-6) and a.ind_heat == pytest.approx(0.4, abs=1e-6)
    assert a.ind_rank20 == 1.0 and out.loc["C1.SZ", "ind_rank20"] == 0.0
    assert out.loc["C1.SZ", "ind_r20"] == pytest.approx(0.0, abs=1e-6)  # C6（日线不足 60）、C7（< 2 元）不算
    assert out.loc[["B1.SZ", "D1.SZ"], list(COLS)].isna().all().all()


def test_report_from_the_self_check_shows_no_change_and_an_identical_grid_shows_no_gain():
    import theme_report as tr
    path = tr.IND / "theme_eval_q_selfcheck.json"
    if not path.is_file() or not (tr.HERE / "rules" / "200000" / "real_M.npz").is_file():
        pytest.skip("要先跑 theme_eval.py --stage q --self-check 和原版的规则搜索")
    q = json.loads(path.read_text(encoding="utf-8"))
    text, verdict = tr.render(q, None, {200000: None, 20000: None})
    assert verdict["V1"].startswith("还没有判定") and "全期 +0.0 [+0.0, +0.0]" in text
    g = tr.grid(200000, tr.HERE, tr.HERE)
    assert g["better"] == 0 and g["median_gap"] == 0 and g["picks"]["V0"] == g["picks"]["V1"]
