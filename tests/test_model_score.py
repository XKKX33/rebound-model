"""反弹模型的逐日打分（`rebound/model/score.py`）：两个版本 × 三种重训频率，补齐、之后每天追加、旧的不改。"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from conftest import MODEL_SESSIONS, TEST_FIRST_FIT, TINY_PARAMS

from rebound.cli import main
from rebound.data.db import WriteStore
from rebound import params as model_params
from rebound.features import FEATURES, eligible, feature_panel
from rebound.store import Store
from rebound.model.artifacts import ModelError, list_blocks, load_models, models_root
from rebound.model.score import fill_scores
from rebound.model.train import active_fit, fit_days, train


def _scores(path: Path) -> pd.DataFrame:
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute("SELECT * FROM model_score ORDER BY variant, cadence, trade_date, rank").df()
    finally:
        con.close()


def _fits(cadence: str) -> list[date]:
    return fit_days(list(MODEL_SESSIONS), cadence, first_fit=TEST_FIRST_FIT)


def test_backfill_scores_every_pool_row_with_the_fit_of_its_cadence(trained_world: Path) -> None:
    report = fill_scores(trained_world, first_fit=TEST_FIRST_FIT)
    got = _scores(trained_world)
    got["trade_date"] = pd.to_datetime(got["trade_date"]).dt.date
    got["fit_date"] = pd.to_datetime(got["fit_date"]).dt.date
    assert report.rows == len(got) and report.days == sum(d >= TEST_FIRST_FIT for d in MODEL_SESSIONS)
    with Store(trained_world) as store:
        panel = feature_panel(store, TEST_FIRST_FIT, MODEL_SESSIONS[-1])
    for variant, universe in model_params.VARIANTS.items():
        pool = panel[eligible(panel, universe, min_bars=model_params.MIN_BARS)]
        ids = {b.fit_date: b.model_id for b in list_blocks(trained_world, variant)}
        for cadence in model_params.CADENCES:
            fits = _fits(cadence)
            mine = got[(got.variant == variant) & (got.cadence == cadence)]
            want = pool[pool.trade_date >= pd.Timestamp(fits[0])]
            assert len(mine) == len(want) > 0, (variant, cadence)
            expect = mine.trade_date.map(lambda d: active_fit(d, fits))
            assert (mine.fit_date == expect).all()
            assert (mine.model_id == mine.fit_date.map(ids)).all()
    assert "600029.SH" not in set(got[got.trade_date >= date(2022, 11, 1)].ts_code)  # 戴帽之后不在池里
    assert got[got.variant == "nocap"].ts_code.nunique() > got[got.variant == "cap70"].ts_code.nunique()


def test_ranks_run_from_one_by_score_then_code(trained_world: Path) -> None:
    fill_scores(trained_world, first_fit=TEST_FIRST_FIT)
    for _, day in _scores(trained_world).groupby(["variant", "cadence", "trade_date"]):
        assert day["rank"].tolist() == list(range(1, len(day) + 1))
        ordered = day.sort_values(["score", "ts_code"], ascending=[False, True], kind="mergesort")
        assert ordered["rank"].tolist() == day["rank"].tolist()


def test_cadences_share_a_fit_and_part_after_it(trained_world: Path) -> None:
    """2023-01-02 是三种频率共同的重训日：1 月三条序列的分相同；2 月按月换了模型，按年、按季不换。"""
    fill_scores(trained_world, first_fit=TEST_FIRST_FIT)
    got = _scores(trained_world)
    got = got[got.variant == "nocap"].assign(trade_date=pd.to_datetime(got.trade_date))
    wide = got.pivot_table(index=["trade_date", "ts_code"], columns="cadence", values="score")
    jan = wide.loc[(wide.index.get_level_values(0) >= "2023-01-02") & (wide.index.get_level_values(0) < "2023-02-01")]
    feb = wide.loc[(wide.index.get_level_values(0) >= "2023-02-01") & (wide.index.get_level_values(0) < "2023-03-01")]
    assert len(jan) > 0 and (jan["Y"] == jan["M"]).all() and (jan["Q"] == jan["M"]).all()
    assert (feb["Y"] == feb["Q"]).all() and (feb["Y"] != feb["M"]).any()


def test_a_score_is_the_float32_mean_of_the_three_seeds(trained_world: Path) -> None:
    fill_scores(trained_world, first_fit=TEST_FIRST_FIT)
    day = date(2023, 3, 15)
    with Store(trained_world) as store:  # 和打分同一段（2023 年）算特征：逐位相同，才能逐位比分数
        panel = feature_panel(store, date(2023, 1, 2), MODEL_SESSIONS[-1])
    panel = panel[panel.trade_date == pd.Timestamp(day)]
    pool = panel[eligible(panel, model_params.VARIANTS["cap70"], min_bars=model_params.MIN_BARS)]
    block = {b.fit_date: b for b in list_blocks(trained_world, "cap70")}[date(2023, 3, 1)]
    x = pool.loc[:, list(FEATURES)].to_numpy(np.float32)
    want = np.mean([b.inplace_predict(x).astype(np.float32) for b in load_models(block)], axis=0)
    got = _scores(trained_world)
    got = got[(got.variant == "cap70") & (got.cadence == "M") & (pd.to_datetime(got.trade_date) == pd.Timestamp(day))]
    stored = got.set_index("ts_code").loc[pool.ts_code.astype(str), "score"].to_numpy()
    assert len(stored) > 0 and np.array_equal(stored, want.astype(np.float64))


def test_rerunning_writes_nothing_and_never_rewrites_a_stored_day(trained_world: Path) -> None:
    fill_scores(trained_world, first_fit=TEST_FIRST_FIT)
    with WriteStore(trained_world) as store:
        store.connection.execute("UPDATE model_score SET score = 9.0 WHERE trade_date = DATE '2022-10-10'")
    assert fill_scores(trained_world, first_fit=TEST_FIRST_FIT).rows == 0
    con = duckdb.connect(str(trained_world), read_only=True)
    try:
        assert {s for (s,) in con.execute("SELECT score FROM model_score WHERE trade_date = DATE '2022-10-10'").fetchall()} == {9.0}
    finally:
        con.close()


def _add_session(path: Path, new: date) -> None:
    with WriteStore(path) as store:
        con = store.connection
        con.execute("INSERT INTO trade_cal VALUES ('SSE', ?, 1)", [new])
        for table in ("daily", "daily_limit", "adj_factor"):
            con.execute(f"INSERT INTO {table} SELECT * REPLACE (CAST(? AS DATE) AS trade_date) FROM {table} "
                        "WHERE trade_date = ?", [new, MODEL_SESSIONS[-1]])


def test_a_new_month_waits_for_its_fit_then_scores_only_the_new_day(trained_world: Path) -> None:
    """6 月的第一个开市日是新的按月重训日：没训就拒绝（一行不写），训好之后只补这一天。"""
    fill_scores(trained_world, first_fit=TEST_FIRST_FIT)
    before = len(_scores(trained_world))
    new = date(2023, 6, 1)
    _add_session(trained_world, new)
    with pytest.raises(ModelError, match="2023-06-01.*rebound model train"):
        fill_scores(trained_world, first_fit=TEST_FIRST_FIT)
    assert len(_scores(trained_world)) == before
    report = train(trained_world, params=TINY_PARAMS, first_fit=TEST_FIRST_FIT)
    assert [(v, d) for v, d, _ in report.trained] == [("nocap", new), ("cap70", new)]
    filled = fill_scores(trained_world, first_fit=TEST_FIRST_FIT)
    rows = _scores(trained_world)
    assert filled.days == 1 and len(rows) - before == filled.rows > 0
    today = rows[pd.to_datetime(rows.trade_date) == pd.Timestamp(new)]
    for variant in model_params.VARIANTS:
        ids = {b.fit_date: b.model_id for b in list_blocks(trained_world, variant)}
        mine = today[today.variant == variant]
        assert set(mine[mine.cadence == "M"].model_id) == {ids[new]}
        assert set(mine[mine.cadence == "Q"].model_id) == {ids[date(2023, 4, 3)]}
        assert set(mine[mine.cadence == "Y"].model_id) == {ids[date(2023, 1, 2)]}


def test_a_day_whose_fit_is_not_trained_refuses_and_writes_nothing(trained_world: Path) -> None:
    shutil.rmtree(models_root(trained_world) / "nocap" / "2023-03-01")
    with pytest.raises(ModelError, match="2023-03-01.*rebound model train"):
        fill_scores(trained_world, first_fit=TEST_FIRST_FIT)
    assert _scores(trained_world).empty


def test_scoring_before_any_training_refuses(model_world: Path) -> None:
    with pytest.raises(ModelError, match="rebound model train"):
        fill_scores(model_world, first_fit=TEST_FIRST_FIT)


def test_cli_model_score_and_train(
    trained_world: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_params, "FIRST_FIT", TEST_FIRST_FIT)
    assert main(["model", "score", "--db", str(trained_world), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["rows"] > 0
    # 小世界是按 8 棵树的小参数训的：拿正式参数去训，要先删掉旧的重训日，不会悄悄覆盖
    assert main(["model", "train", "--db", str(trained_world)]) == 4
    assert "要重训就先删掉这些目录" in capsys.readouterr().err
    monkeypatch.setattr(model_params, "PARAMS", TINY_PARAMS)
    assert main(["model", "train", "--db", str(trained_world), "--json"]) == 0  # 都训过了：什么都不做
    assert json.loads(capsys.readouterr().out)["trained"] == []


def test_scores_from_a_replaced_model_block_all_further_scoring(trained_world: Path) -> None:
    """重训过（模型换了），旧分不改写：再补分就会混着两个模型，本金测算悄悄用两套口径。拒绝，一行都不写。"""
    fill_scores(trained_world, first_fit=TEST_FIRST_FIT)
    with WriteStore(trained_world) as store:
        store.connection.execute("UPDATE model_score SET model_id = 'deadbeef0000' WHERE variant = 'nocap' "
                                 "AND fit_date = DATE '2023-03-01'")
        store.connection.execute("DELETE FROM model_score WHERE trade_date = ?", [MODEL_SESSIONS[-1]])
    before = len(_scores(trained_world))
    with pytest.raises(ModelError, match="换掉的模型打的.*重训日 2023-03-01.*deadbeef0000.*rebound model score"):
        fill_scores(trained_world, first_fit=TEST_FIRST_FIT)
    assert len(_scores(trained_world)) == before


def test_a_bad_block_in_a_later_year_is_caught_before_anything_is_written(trained_world: Path) -> None:
    """口径不对的模型在后面的年份：一次查完，不会先把前面几年写进去。"""
    block = next(b for b in list_blocks(trained_world, "cap70") if b.fit_date == date(2023, 5, 1))
    manifest = json.loads((block.path / "manifest.json").read_text(encoding="utf-8"))
    manifest["min_bars"] = 999
    (block.path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelError, match="2023-05-01.*min_bars 不同"):
        fill_scores(trained_world, first_fit=TEST_FIRST_FIT)
    assert _scores(trained_world).empty
