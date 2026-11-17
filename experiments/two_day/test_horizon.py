from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

import horizon
from horizon import cell_of, delisting_rows, fresh_stats, link_inputs, steadiest, verdict
from engine import Market
from train_horizon import fit_name, label_frame, same_frames


def test_cell_names_round_trip_and_bad_names_are_refused():
    assert cell_of("o1o2").name == "o1o2" and cell_of("o1o2").period == 1 and cell_of("o1o5").period == 4
    for bad in ("o2o5", "x1o5", "o1o", "o1oX"):
        with pytest.raises(ValueError):
            cell_of(bad)


def test_steadiest_takes_the_best_worst_start_among_rules_positive_everywhere_and_the_first_on_ties():
    rows = [("A", fresh_stats({"2017": {"cagr": 30.0}, "2019": {"cagr": -1.0}})),
            ("B", fresh_stats({"2017": {"cagr": 12.0}, "2019": {"cagr": 15.0}})),
            ("C", fresh_stats({"2017": {"cagr": 12.0}, "2019": {"cagr": 40.0}})),
            ("D", fresh_stats({"2017": {"cagr": 5.0}, "2019": {"cagr": 6.0}}))]
    assert steadiest(rows) == "B"
    assert steadiest(rows[:1]) is None


def test_verdict_needs_a_gap_larger_than_the_retrain_noise():
    assert verdict(20.0, 18.0, 3.0) == "打平"
    assert verdict(25.0, 18.0, 3.0) == "两日更好" and verdict(10.0, 18.0, 3.0) == "五日更好"
    assert verdict(None, 18.0, 3.0).startswith("五日更好") and verdict(None, None, 3.0).startswith("两个模型")


def test_fit_cache_names_differ_by_label_and_device_but_are_stable():
    day = pd.Timestamp("2020-03-02")
    names = {fit_name(c, d, day, 100, 10, "k") for c in ("o1o2", "o1o5") for d in ("cpu", "cuda")}
    assert len(names) == 4 and fit_name("o1o2", "cuda", day, 100, 10, "k") in names


def test_two_day_labels_sell_at_t2_open_carry_past_a_locked_open_and_rank_within_the_day():
    dates = pd.bdate_range("2020-01-01", periods=8)
    price = np.array([[10, 10], [10, 20], [11, 18], [12, 18], [12, 18], [12, 18], [12, 18], [12, 18]], dtype=float)
    blocked = np.zeros_like(price, dtype=bool)
    blocked[2, 1] = True  # 第二只票 T+2 开盘跌停卖不出，顺延到 T+3
    m = Market(dates=dates, codes=np.array(["a", "b"]), price={"open": price, "close": price},
               buy_blocked={"open": np.zeros_like(blocked), "close": np.zeros_like(blocked)},
               sell_blocked={"open": blocked, "close": np.zeros_like(blocked)},
               eligible=np.ones_like(blocked), industry=np.array([0, 1]))
    out = label_frame(m, np.array([0, 0]), np.array([0, 1]), cell_of("o1o2"))
    assert out.exit_day.tolist() == [2, 3]
    assert out.y.tolist() == [1.0, 0.5]  # a：10 → 11（+10%）；b：20 → 18（−10%）


def test_same_frames_treats_nan_as_equal_only_in_float_columns():
    a = pd.DataFrame({"di": np.array([1, 2], dtype=np.int32), "y": [np.nan, 0.5]})
    assert same_frames(a, a.copy()) and not same_frames(a, a.assign(y=[0.1, 0.5]))


def test_delisting_rows_use_the_name_at_the_time_and_fall_back_to_the_current_name(tmp_path):
    db = tmp_path / "t.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE namechange (ts_code VARCHAR, name VARCHAR, start_date DATE, ann_date DATE)")
    con.execute("INSERT INTO namechange VALUES ('A', '甲股份', '2019-01-01', '2019-01-01'), "
                "('A', '退市甲', '2020-01-03', '2020-01-02'), ('A', '甲退', '2020-01-03', '2020-01-03')")
    con.execute("CREATE TABLE stock_basic (ts_code VARCHAR, name VARCHAR)")
    con.execute("INSERT INTO stock_basic VALUES ('A', '退市甲'), ('B', '乙退'), ('C', '丙')")
    con.close()
    days = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    panel = pd.DataFrame({"ts_code": np.repeat(["A", "B", "C"], 3), "trade_date": np.tile(days, 3),
                          "eligible": [True, True, False, True, True, True, True, True, True]})
    panel.to_parquet(tmp_path / "p.parquet", index=False)
    got = delisting_rows(tmp_path / "p.parquet", db)
    assert list(zip(got.ts_code, got.trade_date.dt.strftime("%m-%d"))) == [
        ("A", "01-03"), ("B", "01-02"), ("B", "01-03"), ("B", "01-06")]


def test_link_inputs_needs_exactly_one_label_file_and_links_the_rest(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    for name in horizon.LINKED:
        (src / name).write_text(name, encoding="utf-8")
    monkeypatch.setitem(horizon.VARIANTS, "nocap", src)
    with pytest.raises(SystemExit):
        link_inputs("nocap", tmp_path / "out")
    (src / "labels10_o1o5_abc.parquet").write_text("x", encoding="utf-8")
    d = link_inputs("nocap", tmp_path / "out")
    assert sorted(p.name for p in d.iterdir()) == sorted([*horizon.LINKED, "labels10_o1o5_abc.parquet"])


def _fake_sims(d, model, rng, shift=0.0):
    """一个模型 150 条规则的假结果（键和 sim_horizon 写的一样），写成 sims/<模型>_min5_real.json/.npz。"""
    import json

    from rule_grid import specs
    names = [s.name for s in specs()]
    days = np.arange(10, 40)
    dates = pd.bdate_range("2015-01-05", periods=len(days)).to_numpy().astype("datetime64[D]")
    out = {}
    for n in names:
        fresh = {str(y): {"first_year": 1.0, "cagr": float(rng.normal(5 + shift, 10)), "max_dd": -30.0, "final_x": 2.0,
                          "pool_cagr": -3.0} for y in horizon.FRESH_YEARS}
        full = {"cagr": float(rng.normal(20, 5)), "final": 123456.0, "max_dd": -40.0, "cost_pct": 3.2, "excess": 10.0,
                "excess_ci": [1.0, 20.0], "beta": 0.8, "buys_per_day": 0.9, "hold_days": 3.5, "turnover": 50.0,
                "years": {"2015": {"total": 50.0, "excess": 30.0}, "2016": {"total": -5.0, "excess": 9.0}}}
        out[n] = {"full": full, "fresh": fresh, "fresh_stats": fresh_stats(fresh)}
    (d / "sims").mkdir(parents=True, exist_ok=True)
    (d / "sims" / f"{model}_min5_real.json").write_text(json.dumps(out), encoding="utf-8")
    np.savez(d / "sims" / f"{model}_min5_real.npz", names=np.array(names), ret=rng.normal(0, 0.01, (150, len(days))),
             pool=np.zeros(len(days)), days=days, dates=dates)


def _load_report():
    """按路径载入：p0p1 目录里也有一个 report.py，排在 sys.path 前面（run_all.py 是按文件路径跑的，不受影响）。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("horizon_report", horizon.HERE / "report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    return report


def _rows(worst: dict[str, float]) -> list:
    """150 条规则的假行：worst 里给了的规则四个起点都是这个年化，其余都是 −1（亏）。"""
    from rule_grid import specs
    out = []
    for s in specs():
        w = worst.get(s.name, -1.0)
        fresh = {str(y): {"cagr": w, "max_dd": -30.0, "final_x": 2.0, "pool_cagr": -3.0} for y in horizon.FRESH_YEARS}
        out.append((s.name, {**fresh_stats(fresh), "fresh": fresh,
                             "full": {"final": 1.0, "cagr": 10.0, "max_dd": -50.0, "cost_pct": 9.0}}))
    return out


def test_robust_pick_prefers_the_rule_that_holds_in_every_version_over_the_single_version_best():
    report = _load_report()
    a, b = "DB-K1-b6-o1o5", "DB-K2-b6-o1o5"
    variants = {"nocap": {report.FIVE: _rows({a: 30.0, b: 20.0}), report.FIVE_GPU: _rows({a: 25.0, b: 22.0})},
                "cap70": {report.FIVE: _rows({a: 12.0, b: 19.0}), report.FIVE_GPU: _rows({a: -3.0, b: 15.0})}}
    assert report.robust_pick(variants) == (b, [b])
    line = report.robust_line(variants, {"pick": {report.FIVE: a}})
    assert "DB-K2-b6" in line and "-3.0" in line and "+15.0" in line
    assert report.robust_line(variants, {"pick": {report.FIVE: b}}) is None
    assert report.robust_pick({"nocap": {report.FIVE: None}, "cap70": {}}) == (None, [])


def test_report_renders_with_complete_main_results_and_missing_extras(tmp_path, monkeypatch):
    import sys

    report = _load_report()
    rng = np.random.default_rng(0)
    for model, shift in (("o1o2-cuda", 3.0), ("o1o5-cpu", 0.0), ("o1o5-cuda", 0.5)):
        _fake_sims(tmp_path / "nocap", model, rng, shift)
    for name, value in (("HERE", tmp_path), ("OUT", tmp_path / "REPORT.md"), ("FINALISTS", tmp_path / "finalists.json"),
                        ("IC_CACHE", tmp_path / "ic.json")):
        monkeypatch.setattr(report, name, value)
    monkeypatch.setattr(sys, "argv", ["report.py", "--finalists"])
    report.main()
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "## 结论" in text and "重训噪声" in text and "还没跑完" in text and "FH-K1-h1" in text
    picks = report.decide({m: report.rows_of(report.load("nocap", m)) for m in report.MODELS})
    assert picks["noise"] is not None and picks["verdict"] in ("打平", "两日更好", "五日更好")
    items = __import__("json").loads((tmp_path / "finalists.json").read_text(encoding="utf-8"))
    assert [i["model"] for i in items] == ["o1o2-cuda", "o1o5-cpu"] and all(i["rules"] for i in items)


def test_data_dir_refuses_paths_outside_this_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("DECADE_DATA", str(tmp_path / "nocap"))
    with pytest.raises(SystemExit):
        horizon.data_dir()
    monkeypatch.setenv("DECADE_DATA", str(horizon.HERE / "nocap"))
    assert horizon.data_dir() == (horizon.HERE / "nocap").resolve()
