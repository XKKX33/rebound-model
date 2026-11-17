"""本金测算的编排（`rebound/backtest/capital.py`）、明天的清单（`plan.py`）和命令行。

用 conftest 的小世界（训好小模型、补好打分），规则网格只取 6 条（正式是 150 条），几秒算完。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import TEST_FIRST_FIT

from rebound.cli import main
from rebound.data.db import WriteStore
from rebound.backtest import capital, stats
from rebound.backtest.market import Market, candidates, load_markets, model_scores, pool_daily_return
from rebound.backtest.plan import holdings, next_orders
from rebound.backtest.rules import parse
from rebound.backtest.sim import simulate
from rebound.model.artifacts import ModelError, models_root
from rebound.model.score import fill_scores

SUBSET = [parse(n) for n in ("FH-K1-h4", "FH-K2-h4", "DB-K1-b2", "DB-K2-b2", "DB-K2-b4", "DB-K3-b4")]
REQUEST = capital.CapitalRequest(capital=200_000.0)


@pytest.fixture(scope="module")
def world(trained_world_pristine: Path, tmp_path_factory: pytest.TempPathFactory):
    """补好打分的小世界，按 6 条规则算过一次（结果在缓存里）。"""
    path = tmp_path_factory.mktemp("capital-world") / "market.duckdb"
    shutil.copy2(trained_world_pristine, path)
    shutil.copytree(models_root(trained_world_pristine), models_root(path))
    fill_scores(path, first_fit=TEST_FIRST_FIT)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(capital, "specs", lambda: list(SUBSET))
        patch.setattr(capital, "SIGNAL_FROM", pd.Timestamp(TEST_FIRST_FIT))
        out = capital.run(path, REQUEST)
    return path, out


@pytest.fixture
def subset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(capital, "specs", lambda: list(SUBSET))
    monkeypatch.setattr(capital, "SIGNAL_FROM", pd.Timestamp(TEST_FIRST_FIT))


def test_every_rule_and_cadence_is_simulated_and_cached(world) -> None:
    path, out = world
    rows = out["rows"]
    assert [(r["cadence"], r["name"]) for r in rows] == [(c, s.short) for c in capital.CADENCES for s in SUBSET]
    assert all((r["fresh"] is None) == (r["cadence"] != "M") or r is rows[out["picker_choice"]] for r in rows)
    assert out["best_picker"] in stats.PICKERS and set(out["walk_forward"]) == set(stats.PICKERS)
    assert out["today"] == "2023-05-31" and out["first"] > TEST_FIRST_FIT.isoformat()
    assert out["recommended"] is None or rows[out["recommended"]]["fresh"]["all_positive"]
    # 推荐按十年实验定的信息比率：选法挑的是全期信息比率最高的那条（不是这一次滚动超额最高的选法）
    assert out["picker"] == "ir" and out["basis"] in {"picker", "fresh"} and out["small_capital"] is False
    assert rows[out["picker_choice"]]["ir"] == max(r["ir"] for r in rows if r["ir"] is not None)
    assert capital.cached(path, REQUEST) == json.loads(json.dumps(out, default=str))
    assert capital.cached(path, replace(REQUEST, capital=100_000.0)) is None
    # 算完同时写一份摘要：本金测算页的"本金"下拉读它，不用每次读整份结果
    meta = json.loads((capital.cache_root(path) / f"{out['key']}.meta.json").read_text(encoding="utf-8"))
    assert meta["request"] == out["request"] and meta["today"] == out["today"] and meta["version"] == capital.VERSION
    assert out["key"] in [r["key"] for r in capital.runs(path)["runs"]]


def test_the_table_is_the_simulator_run_directly(world) -> None:
    """全期表的数字 = 直接调模拟器 + summary（编排没有漏掉或改动什么）。"""
    path, out = world
    m = load_markets(path, end=date(2023, 5, 31))["nocap"]
    cands, rank = candidates(m, model_scores(path, m, "nocap", "Q"))
    d_first, today = int(m.dates.searchsorted(pd.Timestamp(TEST_FIRST_FIT))), len(m.dates) - 1
    idx = stats.scoring_days(d_first, today - 1)
    pool = pool_daily_return(m)
    for spec in SUBSET:
        res = simulate(m, cands, rank, spec.rule, REQUEST.cost, d_first, today - 1)
        row = next(r for r in out["rows"] if (r["cadence"], r["name"]) == ("Q", spec.short))
        want = json.loads(json.dumps(stats.summary(res, pool, m.dates, idx)))
        assert {k: row[k] for k in want} == want


def _truncate(m: Market, n: int) -> Market:
    cut = lambda d: {k: v[:n] for k, v in d.items()}  # noqa: E731
    return replace(m, dates=m.dates[:n], price=cut(m.price), raw=cut(m.raw), buy_blocked=cut(m.buy_blocked),
                   sell_blocked=cut(m.sell_blocked), eligible=m.eligible[:n])


def test_tomorrows_list_is_what_the_simulator_does_the_next_day(world) -> None:
    """照清单做 = 模拟器第二天做的：卖哪些、买哪些（小世界里没有涨跌停开盘、钱也够，估价的差别不影响谁进谁出）。"""
    path, _ = world
    m = load_markets(path, end=date(2023, 5, 31))["nocap"]
    cands, rank = candidates(m, model_scores(path, m, "nocap", "M"))
    d_first = int(m.dates.searchsorted(pd.Timestamp(TEST_FIRST_FIT)))
    names = {c: (c, "") for c in m.codes}
    checked = 0
    for spec in SUBSET:
        for today in (len(m.dates) - 30, len(m.dates) - 12):
            past = simulate(_truncate(m, today + 1), cands, rank, spec.rule, REQUEST.cost, d_first, today - 1)
            plan = next_orders(m, names, cands, rank, spec.rule, REQUEST.cost, past.positions, past.tranche_cash, today)
            nxt = simulate(_truncate(m, today + 2), cands, rank, spec.rule, REQUEST.cost, d_first, today)
            sold = {str(m.codes[j]) for j in nxt.trades.j[nxt.trades.exit_day == today + 1]}
            bought = {str(m.codes[p.j]) for p in nxt.positions if p.entry_day == today + 1}
            assert {s["ts_code"] for s in plan["sells"]} == sold, (spec.short, today)
            assert {b["ts_code"] for b in plan["buys"]} == bought, (spec.short, today)
            for row in holdings(m, names, past.positions, today):
                assert row["shares"] % 100 == 0 and row["value"] > 0
            checked += len(sold) + len(bought)
    assert checked > 0


def test_a_new_account_buys_at_most_k_names_with_whole_lots(world) -> None:
    _, out = world
    for row in out["rows"]:
        buys = row["new"]["buys"]
        assert len(buys) <= row["k"] and not row["new"]["sells"]
        assert all(b["shares"] % 100 == 0 and b["shares"] > 0 for b in buys)
        assert sum(b["amount"] for b in buys) <= REQUEST.capital
        assert len({b["ts_code"] for b in buys + row["new"]["alternates"]}) == len(buys) + len(row["new"]["alternates"])


def test_bad_inputs_and_missing_scores_are_refused(model_world: Path) -> None:
    for bad, message in ((replace(REQUEST, capital=500), "1000 元"), (replace(REQUEST, variant="all"), "未知"),
                         (replace(REQUEST, commission=0.5), "commission")):
        with pytest.raises(capital.CapitalError, match=message):
            capital.run(model_world, bad)
    with pytest.raises(capital.CapitalError, match="还没打分.*rebound model score"):
        capital.run(model_world, REQUEST)
    with pytest.raises(capital.CapitalError, match="不合法"):
        capital.load(model_world, "../etc")


def test_cli_prints_the_recommendation_and_tomorrows_buys(world, subset, capsys: pytest.CaptureFixture) -> None:
    path, out = world
    assert main(["capital", "--db", str(path), "--capital", "200000"]) == 0
    text = capsys.readouterr().out
    assert "本金测算：200,000 元，不限价" in text and "开盘买" in text
    assert "滚动样本外" in text and f"编号 {out['key']}" in text
    assert main(["capital", "--db", str(path), "--capital", "200000", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["key"] == out["key"]
    assert main(["capital", "--db", str(path), "--capital", "500"]) == 4


def _row(cagrs: list[float] | None) -> dict:
    if cagrs is None:
        return {"fresh": None}
    starts = {str(y): {"cagr": c} for y, c in zip(capital.FRESH_YEARS, cagrs)}
    return {"fresh": {"starts": starts, "worst_cagr": min(cagrs), "all_positive": min(cagrs) > 0}}


def test_a_result_of_an_older_version_is_not_loaded(tmp_path: Path) -> None:
    """旧版本的结果是另一套规则或时点算的（比如 2026-11-01 之前的开盘卖）：按编号也不给，提示重新运行。"""
    db = tmp_path / "market.duckdb"
    root = capital.cache_root(db)
    root.mkdir()
    (root / "aaaa.json").write_text(json.dumps({"version": capital.VERSION - 1, "key": "aaaa"}), encoding="utf-8")
    (root / "bbbb.json").write_text(json.dumps({"version": capital.VERSION, "key": "bbbb"}), encoding="utf-8")
    with pytest.raises(capital.CapitalError, match="旧版本"):
        capital.load(db, "aaaa")
    assert capital.load(db, "bbbb")["key"] == "bbbb"


def test_the_recommendation_follows_the_decade_standard_for_the_account_size() -> None:
    """十年实验：20 万这一类按滚动选法（它换起点亏钱才改推最稳的）；有最低佣金的小账户以换起点体检为准。"""
    rows = [_row([31.3, 21.9, 27.5, 10.3]), _row([28.6, 25.0, 26.8, 26.3]), _row([-21.4, 5.0, 9.0, 3.0]), _row(None)]
    assert capital.recommend(rows, 0, small=False) == (0, 1, "picker")
    assert capital.recommend(rows, 0, small=True) == (1, 1, "fresh")
    assert capital.recommend(rows, 2, small=False) == (1, 1, "fresh")  # 选法挑的换起点会亏
    assert capital.recommend(rows, 3, small=False) == (1, 1, "fresh")  # 选法挑的没做换起点体检
    assert capital.recommend([_row([-1.0, 2.0, 3.0, 4.0])], 0, small=True) == (None, None, "fresh")


def test_small_accounts_are_those_with_a_minimum_commission_under_100k() -> None:
    assert capital.CapitalRequest(capital=20_000).small
    assert not capital.CapitalRequest(capital=200_000).small
    assert not capital.CapitalRequest(capital=20_000, min_commission=0).small


def test_changed_industries_or_names_invalidate_the_cached_result(world, tmp_path: Path) -> None:
    """同样的本金和分数，行业改了（行业上限）或者补了历史名称（准入池基准）：不能再读旧结果。"""
    path, out = world
    copy = tmp_path / "market.duckdb"
    shutil.copy2(path, copy)
    shutil.copytree(capital.cache_root(path), capital.cache_root(copy))
    assert capital.cached(copy, REQUEST) is not None
    with WriteStore(copy) as store:
        store.connection.execute("UPDATE stock_basic SET industry = '换了行业' WHERE ts_code = '600000.SH'")
    assert capital.cached(copy, REQUEST) is None
    shutil.copy2(path, copy)
    with WriteStore(copy) as store:
        store.connection.execute("INSERT INTO namechange VALUES ('600001.SH', '新名称', DATE '2023-05-30', NULL, "
                                 "DATE '2023-05-29', '其他')")
    assert capital.cached(copy, REQUEST) is None


def test_duplicated_scores_are_refused_before_one_stock_is_bought_twice(world, tmp_path: Path) -> None:
    """model_score 没有主键：同一天同一只票有两行分，模拟器会把它当成两只候选、买两次。直接拒绝。"""
    path, _ = world
    copy = tmp_path / "market.duckdb"
    shutil.copy2(path, copy)
    with WriteStore(copy) as store:
        store.connection.execute("INSERT INTO model_score SELECT * FROM model_score WHERE variant = 'nocap' "
                                 "AND cadence = 'M' ORDER BY trade_date, rank LIMIT 2")
    m = load_markets(copy, end=date(2023, 5, 31))["nocap"]
    with pytest.raises(ModelError, match="2 行是同一天同一只票的重复分"):
        model_scores(copy, m, "nocap", "M")
