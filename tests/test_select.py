"""每日选股（`rebound/select.py`）、选股前补模型（`rebound/model/refresh.py`）和 `rebound pick` 命令。

用 conftest 的小世界：训好小模型、补好打分之后选股。
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from conftest import MODEL_SESSIONS, ROOT, TEST_FIRST_FIT, TINY_PARAMS

from rebound import params as model_params
from rebound.cli import main
from rebound.data.db import WriteStore
from rebound.model.artifacts import ModelError, list_blocks, models_root
from rebound.model.score import fill_scores
from rebound.select import SelectError, dedup_top, pick_day

DAY = date(2023, 3, 15)
GOLDEN = ROOT / "tests" / "golden" / "expected_return_model_world.json"


def _copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    shutil.copytree(models_root(src), models_root(dst))
    return dst


@pytest.fixture(scope="module")
def scored_pristine(trained_world_pristine: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = _copy(trained_world_pristine, tmp_path_factory.mktemp("scored-world") / "market.duckdb")
    fill_scores(path, first_fit=TEST_FIRST_FIT)
    return path


@pytest.fixture
def scored(scored_pristine: Path, tmp_path: Path) -> Path:
    return _copy(scored_pristine, tmp_path / "scored" / "market.duckdb")


def _stored(path: Path, day: date, variant: str = "nocap", cadence: str = "M") -> pd.DataFrame:
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute("SELECT * FROM model_score WHERE trade_date = ? AND variant = ? AND cadence = ? "
                           "ORDER BY rank", [day, variant, cadence]).df()
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# 行业去重
# --------------------------------------------------------------------------- #


def test_dedup_keeps_the_order_caps_each_industry_and_stops_at_top_n() -> None:
    industries = ["银行", "银行", "银行", "钢铁", "", "", "", "电力", "电力"]
    assert dedup_top(industries, max_per_industry=2, top_n=6) == [0, 1, 3, 4, 5, 7]
    assert dedup_top(industries, max_per_industry=1, top_n=10) == [0, 3, 4, 7]
    assert dedup_top([], max_per_industry=2, top_n=6) == []


# --------------------------------------------------------------------------- #
# 选股
# --------------------------------------------------------------------------- #


def test_the_pool_is_the_stored_scores_and_picks_are_the_top_n_after_dedup(scored: Path) -> None:
    result = pick_day(scored, as_of=DAY)
    stored = _stored(scored, DAY)
    assert result.stored and result.pool.ts_code.tolist() == stored.ts_code.tolist()
    assert result.pool["rank"].tolist() == list(range(1, len(stored) + 1))
    pct = result.pool.pct.tolist()
    assert pct == sorted(pct, reverse=True) and pct[0] == 1.0 and all(0 < p <= 1 for p in pct)
    want, seen = [], {}
    for code, industry in zip(result.pool.ts_code, result.pool.industry):
        if seen.get(industry, 0) < 2:
            want.append(code)
            seen[industry] = seen.get(industry, 0) + 1
    assert result.picks.ts_code.tolist() == want[:6] and len(result.picks) == 6


def test_the_version_and_cadence_pick_their_own_scores_and_model(scored: Path) -> None:
    capped = pick_day(scored, as_of=DAY, variant="cap70")
    assert capped.pool.ts_code.tolist() == _stored(scored, DAY, "cap70").ts_code.tolist()
    assert {"600030.SH", "600032.SH"}.isdisjoint(set(capped.pool.ts_code))  # 70 元以上
    yearly = pick_day(scored, as_of=DAY, cadence="Y")
    monthly = pick_day(scored, as_of=DAY)
    assert yearly.pool.ts_code.tolist() == _stored(scored, DAY, cadence="Y").ts_code.tolist()
    assert (yearly.fit_date, monthly.fit_date) == (date(2023, 1, 2), date(2023, 3, 1))
    assert yearly.model_id != monthly.model_id


def test_a_day_without_stored_scores_is_scored_live_with_the_same_numbers(scored: Path) -> None:
    day = MODEL_SESSIONS[-1]
    with WriteStore(scored) as store:
        store.connection.execute("DELETE FROM model_score WHERE trade_date = ?", [day])
    live = pick_day(scored, as_of=day)
    assert not live.stored
    assert fill_scores(scored, first_fit=TEST_FIRST_FIT).days == 1  # 只补这一天：和当场算的是同一段特征
    kept = pick_day(scored, as_of=day)
    assert kept.stored and kept.model_id == live.model_id
    assert kept.pool[["ts_code", "score"]].equals(live.pool[["ts_code", "score"]])


def test_without_a_day_the_latest_trading_day_is_used(scored: Path) -> None:
    assert pick_day(scored).day == MODEL_SESSIONS[-1]
    with pytest.raises(SelectError, match="本地没有日线行情"):
        pick_day(scored, as_of=date(2023, 3, 18))  # 星期六


def test_stored_scores_win_over_a_pool_recomputed_later(scored: Path) -> None:
    """当时打分的全池是记录：今天重算的股票池多了一只（比如后来补了改名记录），仍按当时的。"""
    victim = _stored(scored, DAY).ts_code.iloc[3]
    with WriteStore(scored) as store:
        store.connection.execute("DELETE FROM model_score WHERE trade_date = ? AND ts_code = ?", [DAY, victim])
    day = pick_day(scored, as_of=DAY)
    assert day.differ == 1 and victim not in set(day.pool.ts_code)


def test_scores_from_another_model_are_refused(scored: Path) -> None:
    with WriteStore(scored) as store:
        store.connection.execute("UPDATE model_score SET model_id = 'deadbeef0000' WHERE trade_date = ?", [DAY])
    with pytest.raises(SelectError, match="deadbeef0000.*模型被换过"):
        pick_day(scored, as_of=DAY)


def test_an_untrained_fit_or_a_day_before_the_first_fit_is_refused(
    scored: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutil.rmtree(models_root(scored) / "nocap" / "2023-03-01")
    with pytest.raises(ModelError, match="2023-03-01 的模型还没训练.*rebound model train"):
        pick_day(scored, as_of=DAY)
    monkeypatch.setattr(model_params, "FIRST_FIT", TEST_FIRST_FIT)
    with pytest.raises(ModelError, match="早于反弹模型的第一个重训日"):
        pick_day(scored, as_of=date(2022, 7, 15))


def test_scores_written_twice_for_one_day_are_refused_with_a_way_out(scored: Path) -> None:
    """两个写库任务同时补过分，同一天同一只票有两行：说清怎么修，而不是抛 pandas 的合并错误。"""
    with WriteStore(scored) as store:
        store.connection.execute("INSERT INTO model_score SELECT * FROM model_score WHERE trade_date = ? "
                                 "AND variant = 'nocap' AND cadence = 'M' ORDER BY rank LIMIT 3", [DAY])
    with pytest.raises(SelectError, match="3 只票的分写了不止一行.*rebound model score"):
        pick_day(scored, as_of=DAY)


def _versions() -> dict[str, str]:
    import numpy
    import xgboost

    return {"duckdb": duckdb.__version__, "numpy": numpy.__version__, "xgboost": xgboost.__version__}


def test_the_golden_day(scored: Path) -> None:
    """小世界 2023-03-15 的全池分数和前 6：特征、标签、训练、打分哪一处变了都会对不上。
    分数的最后几位跟着 xgboost / duckdb / numpy 的版本走，所以文件里记着版本，版本不同就明说。
    要更新：删掉 tests/golden/expected_return_model_world.json 再跑一次这个测试（它会写出新的并报失败）。"""
    result = pick_day(scored, as_of=DAY, top_n=6, max_per_industry=2)
    score = _stored(scored, DAY).set_index("ts_code").score
    got = {"versions": _versions(), "day": DAY.isoformat(),
           "pool": [[code, float(score[code])] for code in result.pool.ts_code], "picks": result.picks.ts_code.tolist()}
    if not GOLDEN.exists():
        GOLDEN.write_text(json.dumps(got, ensure_ascii=False, indent=1), encoding="utf-8")
        pytest.fail(f"没有 {GOLDEN.name}：已按现在的结果写出，检查后再跑一次")
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if want.get("versions") != got["versions"]:
        pytest.fail(f"{GOLDEN.name} 是在 {want.get('versions')} 下写的，现在是 {got['versions']}："
                    "分数的最后几位会变。确认结果合理后删掉这个文件重写")
    assert (got["day"], got["picks"]) == (want["day"], want["picks"])
    assert [code for code, _ in got["pool"]] == [code for code, _ in want["pool"]]
    assert [s for _, s in got["pool"]] == pytest.approx([s for _, s in want["pool"]], rel=0, abs=1e-6)


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #


def test_pick_prints_the_list_and_json_has_the_same_picks(scored: Path, capsys) -> None:
    assert main(["pick", "--db", str(scored), "--as-of", DAY.isoformat()]) == 0
    text = capsys.readouterr().out
    want = pick_day(scored, as_of=DAY).picks.ts_code.tolist()
    assert "2023-03-15 收盘后的名单" in text and all(code in text for code in want)
    assert main(["pick", "--db", str(scored), "--as-of", "20230315", "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert [row["ts_code"] for row in body["picks"]] == want and body["stored"] is True


def test_pick_on_a_database_without_models_says_to_train_first(model_world: Path, capsys) -> None:
    """一次都没训过：拒绝，提示先训练（退出码 4，stdout 不出半截 JSON）。"""
    assert main(["pick", "--db", str(model_world), "--as-of", DAY.isoformat(), "--json"]) == 4
    out = capsys.readouterr()
    assert out.out == "" and "rebound model train" in out.err


# --------------------------------------------------------------------------- #
# 选股前补模型
# --------------------------------------------------------------------------- #


def _add_session(path: Path, new: date) -> None:
    with WriteStore(path) as store:
        con = store.connection
        con.execute("INSERT INTO trade_cal VALUES ('SSE', ?, 1)", [new])
        for table in ("daily", "daily_limit", "adj_factor"):
            con.execute(f"INSERT INTO {table} SELECT * REPLACE (CAST(? AS DATE) AS trade_date) FROM {table} "
                        "WHERE trade_date = ?", [new, MODEL_SESSIONS[-1]])


def test_pick_with_refresh_trains_the_due_fit_and_scores_the_day_first(
    scored: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    monkeypatch.setattr(model_params, "FIRST_FIT", TEST_FIRST_FIT)
    monkeypatch.setattr(model_params, "PARAMS", TINY_PARAMS)
    new = date(2023, 6, 1)
    _add_session(scored, new)
    assert main(["pick", "--db", str(scored), "--as-of", new.isoformat(), "--refresh", "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["day"] == new.isoformat() and len(body["picks"]) == 6 and body["stored"] is True
    for variant in model_params.VARIANTS:
        assert list_blocks(scored, variant)[-1].fit_date == new
        assert len(_stored(scored, new, variant)) > 0
    ids = {b.fit_date: b.model_id for b in list_blocks(scored, "nocap")}
    assert set(_stored(scored, new).model_id) == {ids[new]}


def test_a_look_only_pick_writes_no_model_and_refuses_an_untrained_fit(
    scored: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_params, "FIRST_FIT", TEST_FIRST_FIT)
    new = date(2023, 6, 1)
    _add_session(scored, new)
    with pytest.raises(ModelError, match="2023-06-01 的模型还没训练"):
        pick_day(scored, as_of=new)
    assert list_blocks(scored, "nocap")[-1].fit_date == date(2023, 5, 1)


def test_refresh_refuses_a_long_backlog(model_world: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """新库一次都没训过：全量训练要很久，不藏在选股里。"""
    from rebound.model.refresh import refresh

    monkeypatch.setattr(model_params, "FIRST_FIT", TEST_FIRST_FIT)
    with pytest.raises(ModelError, match="缺 18 次重训.*最早 2022-09-01.*rebound model train"):
        refresh(model_world)


def test_refresh_does_not_call_training_when_no_fit_is_due(scored: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """平常的日子没有到期的重训日：不调训练（训练会逐个核对已训目录，升级 xgboost 不该让每天的选股停掉）。"""
    from rebound.model import refresh as refresh_module

    monkeypatch.setattr(model_params, "FIRST_FIT", TEST_FIRST_FIT)

    def no_training(*_args, **_kwargs):
        raise AssertionError("没有到期的重训日，不该调训练")

    monkeypatch.setattr(refresh_module, "train", no_training)
    report = refresh_module.refresh(scored)
    assert report.trained.trained == () and report.scored.rows == 0


def test_refresh_refuses_a_long_score_backlog(scored: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rebound.model.refresh import refresh

    monkeypatch.setattr(model_params, "FIRST_FIT", TEST_FIRST_FIT)
    with WriteStore(scored) as store:
        store.connection.execute("DELETE FROM model_score WHERE variant = 'cap70' AND cadence = 'Q' "
                                 "AND trade_date > DATE '2023-03-01'")
    with pytest.raises(ModelError, match="个交易日还没打分（2023-03-02 ~ 2023-05-31）.*rebound model score"):
        refresh(scored)


@pytest.mark.parametrize(("table", "keep", "message"), [
    ("daily", 20, "主板日线 20 行，前一个交易日 33 行"),
    ("adj_factor", 30, "有复权因子的只有 30 行"),
])
def test_refresh_refuses_to_freeze_a_day_with_incomplete_market_data(
    scored: Path, monkeypatch: pytest.MonkeyPatch, table: str, keep: int, message: str,
) -> None:
    """分一旦写下就不再改：当天日线只同步到一截、或者有票没复权因子，就不打分、不训练、不选股。"""
    from rebound.model.refresh import refresh

    monkeypatch.setattr(model_params, "FIRST_FIT", TEST_FIRST_FIT)
    new = date(2023, 6, 1)
    _add_session(scored, new)
    with WriteStore(scored) as store:
        store.connection.execute(f"DELETE FROM {table} WHERE trade_date = ? AND ts_code NOT IN "
                                 f"(SELECT ts_code FROM {table} WHERE trade_date = ? ORDER BY ts_code LIMIT {keep})",
                                 [new, new])
    with pytest.raises(ModelError, match=f"{message}.*这次不打分也不选股"):
        refresh(scored)
    assert _stored(scored, new).empty
    assert list_blocks(scored, "nocap")[-1].fit_date == date(2023, 5, 1)  # 没训
