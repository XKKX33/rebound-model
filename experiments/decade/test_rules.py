from __future__ import annotations

import numpy as np
import pandas as pd

from rule_grid import DB_B, DB_K, MAX_BOOK, REFERENCE, Spec, neighbors, parse, specs
from rule_report import render, runs, short
from rule_select import neighbor_index, pbo, reality_check, select, series_stats, smooth, walk_forward

ORIGINAL = [*(f"FH-K{k}-next-o1o5" for k in (1, 2, 3, 6, 10, 20)),
            *(f"DB-K{k}-b{b}-o1o5" for k in (2, 3, 4, 6, 10, 20) for b in (2, 4))]


def test_grid_has_150_unique_rules_including_the_original_18_and_names_round_trip():
    grid = specs()
    names = [s.name for s in grid]
    assert len(grid) == 150 and len(set(names)) == 150
    assert set(ORIGINAL) <= set(names)
    assert all(parse(s.name) == s for s in grid)
    assert all(s.k * s.p <= MAX_BOOK for s in grid if s.family == "FH")
    assert Spec("FH", 6, 4).rule.cell.period == 4 and Spec("FH", 6, 4).name == REFERENCE[1]
    assert short("FH-K6-next-o1o5") == "FH-K6-h4" and short("DB-K10-b1.5-o1o5") == "DB-K10-b1.5"


def test_neighbors_stay_on_the_grid_and_respect_the_book_cap():
    pool = set(specs())
    assert len(neighbors(Spec("DB", 1, 1), pool)) == 4
    assert len(neighbors(Spec("DB", 6, 4), pool)) == 9
    near = neighbors(Spec("FH", 15, 6), pool)
    assert Spec("FH", 20, 6) not in near and Spec("FH", 15, 9) not in near and Spec("FH", 20, 4) in near


def test_smoothing_prefers_a_plateau_over_an_isolated_spike():
    configs = [("Y", Spec("DB", k, b)) for k in DB_K for b in DB_B]
    score = np.zeros(len(configs))
    where = {c: i for i, c in enumerate(configs)}
    for k in (4, 6, 8):
        for b in (3, 4, 6):
            score[where[("Y", Spec("DB", k, b))]] = 1.0
    score[where[("Y", Spec("DB", 30, 16))]] = 3.0
    smoothed = smooth(score, neighbor_index(configs))
    assert configs[int(np.argmax(score))][1] == Spec("DB", 30, 16)
    assert configs[int(np.argmax(smoothed))][1] == Spec("DB", 6, 4)


def test_walk_forward_only_uses_the_past_and_splices_the_picked_years():
    dates = pd.bdate_range("2015-01-01", "2019-12-31")
    early = dates.year < 2017
    book = np.zeros((2, len(dates)))
    book[0] = np.where(early, 0.01, -0.03)
    book[1] = 0.001
    near = [np.array([0]), np.array([1])]
    w = walk_forward(book, book.copy(), dates, near, "raw")
    assert w["picks"] == {"2017": 0, "2018": 1, "2019": 1}
    want = np.concatenate([book[0][dates.year == 2017], book[1][dates.year >= 2018]])
    assert np.array_equal(w["excess"], want) and len(w["dates"]) == len(want)


def test_pbo_is_near_half_for_noise_and_zero_for_a_real_edge():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, size=(50, 3200))
    assert 0.25 <= pbo(noise)["pbo"] <= 0.75
    edge = noise.copy()
    edge[7] += 0.003
    assert pbo(edge)["pbo"] == 0.0
    assert 0.25 <= pbo(noise, criterion="ir")["pbo"] <= 0.75 and pbo(edge, criterion="ir")["pbo"] == 0.0


def test_best_picker_breaks_near_ties_by_the_interval_lower_bound():
    from rule_select import best_of
    walked = {"raw": {"excess": 26.14, "ci": [7.0, 47.1]}, "ir": {"excess": 26.10, "ci": [16.9, 35.7]},
              "smooth": {"excess": 20.0, "ci": [18.0, 22.0]}}
    assert best_of(walked) == "ir"
    walked["raw"]["excess"] = 27.0
    assert best_of(walked) == "raw"


def test_runs_merges_consecutive_years_with_the_same_pick():
    picks = {"2017": ["H", "DB-K30-b8-o1o5"], "2018": ["H", "DB-K30-b8-o1o5"], "2019": ["Y", "FH-K6-next-o1o5"]}
    assert runs(picks) == "2017–2018：每半年 DB-K30-b8；2019：每年 FH-K6-h4"


def test_reality_check_rejects_only_with_a_real_edge():
    dates = pd.bdate_range("2015-01-01", periods=2000)
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.01, size=(30, len(dates)))
    assert reality_check(noise, dates)["p"] > 0.05
    edge = noise.copy()
    edge[3] += 0.002
    r = reality_check(edge, dates)
    assert r["p"] < 0.01 and r["n_above"] == 1


def test_series_stats_of_a_constant_daily_return():
    dates = pd.bdate_range("2020-01-01", periods=242)
    s = series_stats(np.full(242, 0.001), np.full(242, 0.001), dates)
    assert s["excess"] == 24.2 and s["max_dd"] == 0.0 and abs(s["cagr"] - (1.001 ** 242 - 1) * 100) < 0.01


def test_compare_cap_pairs_the_same_rule_across_versions(tmp_path, monkeypatch):
    import compare_cap
    dates = pd.bdate_range("2020-01-01", periods=300)
    names = np.array([REFERENCE[1], "DB-K30-b8-o1o5"])
    for version, bump in (("a", 0.0), ("b", 0.001)):
        d = tmp_path / version / "rules" / "200000"
        d.mkdir(parents=True)
        np.savez(d / "real_Y.npz", names=names, ret=np.full((2, len(dates)), 0.0005 + bump), pool=np.zeros(len(dates)),
                 days=np.arange(len(dates)), dates=np.array(dates, dtype="datetime64[D]"))
    monkeypatch.setattr(compare_cap, "VERSIONS", {"capped": tmp_path / "a", "nocap": tmp_path / "b"})
    cmp = compare_cap.compare(200000)
    assert cmp["by_cadence"]["Y"]["nocap_better"] == 1.0 and cmp["by_cadence"]["Y"]["n"] == 2
    assert [p["name"] for p in cmp["pairs"]] == [REFERENCE[1]] and cmp["pairs"][0]["diff"] == 24.2
    assert "## 同一组规则" in compare_cap.render(cmp)


def _stats(configs, book, pool, dates) -> dict:
    out = {}
    for (c, spec), r in zip(configs, book):
        ex = r - pool
        years = {str(y): {"total": 1.0, "excess": float(ex[dates.year == y].mean() * 24200)} for y in np.unique(dates.year)}
        out[(c, spec.name)] = {"excess": float(ex.mean() * 24200), "excess_ci": [0.0, 1.0], "cagr": 1.0, "max_dd": -10.0,
                               "beta": 1.0, "alpha": 0.0, "final": 2e5, "buys_per_day": 1.0, "hold_days": 4.0,
                               "turnover": 50.0, "cost_pct": 1.0, "idle_cash_pct": 0.0, "avg_holdings": 6.0,
                               "years": years}
    return out


def test_select_and_render_end_to_end_on_synthetic_returns():
    dates = pd.bdate_range("2015-01-05", "2019-12-31")
    grid = [Spec("FH", 6, 4), Spec("FH", 4, 4), Spec("DB", 6, 4), Spec("DB", 6, 3), Spec("DB", 4, 4)]
    configs = [(c, s) for c in ("Y", "H") for s in grid]
    rng = np.random.default_rng(3)
    pool = rng.normal(0, 0.01, len(dates))
    book = pool + rng.normal(0.0004, 0.004, size=(len(configs), len(dates)))
    book[configs.index(("H", Spec("DB", 6, 4)))] += 0.0006
    sel = {"capital": 200000.0, **select(configs, book, pool, dates, _stats(configs, book, pool, dates))}
    assert (sel["recommended"]["raw"]["cadence"], sel["recommended"]["raw"]["name"]) == ("H", "DB-K6-b4-o1o5")
    assert set(sel["walk_forward"]["smooth"]["picks"]) == {"2017", "2018", "2019"}
    assert ["Y", REFERENCE[1]] in sel["finalists"] and sel["best_picker"] in sel["walk_forward"]
    assert len(sel["reference_years"]) == 5 and set(sel["pbo"]) == {"mean", "ir"}
    text = render(sel, {})
    for part in ("## 结论", "## 四种选法的对比", "## 前几名（按信息比率）", "## 重训频率", "## 网格全貌", "## 入围规则",
                 "滚动检验", "PBO", "原方案"):
        assert part in text
    assert "换个年份重新起步" not in text
    pick = sel["recommended"][sel["best_picker"]]
    fresh = {f"200000|{pick['cadence']}|{pick['name']}|min5|{y}": {"cagr": c, "max_dd": -50.0, "pool_cagr": 1.0}
             for y, c in ((2017, -12.0), (2019, 5.0))}
    fresh["20000|Y|FH-K6-next-o1o5|min5|2017"] = {"cagr": -30.0, "max_dd": -90.0, "pool_cagr": 1.0}
    text = render(sel, {}, fresh)
    assert "推荐规则换个年份重新起步会亏" in text and "2017 年起步年化为负" in text and "## 换个年份重新起步" in text
    assert "-30.0%" not in text  # 别的本金的不列
