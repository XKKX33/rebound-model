"""反弹模型的按月重训（`rebound/model/train.py`、`rebound/model/artifacts.py`）。

用 conftest 的小世界（33 只票、2020-06 ~ 2023-05、第一个重训日 2022-09-01）和 8 棵浅树的小模型：
口径和正式训练一样，只是树少、起点晚。
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import MODEL_SESSIONS, TEST_FIRST_FIT, TINY_PARAMS

from rebound.data.db import WriteStore
from rebound.data.errors import DataError
from rebound import params as model_params
from rebound.labels import percentile
from rebound.model import train as train_module
from rebound.model.artifacts import ModelError, list_blocks, load_models, models_root
from rebound.model.train import TrainingRows, active_fit, fit_days, load_rows, train, training_set

MONTHLY = fit_days(list(MODEL_SESSIONS), "M", first_fit=TEST_FIRST_FIT)


def test_fit_days_are_the_first_session_of_the_chosen_months() -> None:
    sessions = list(MODEL_SESSIONS)
    assert fit_days(sessions, "Y", first_fit=date(2021, 1, 1)) == [date(2021, 1, 1), date(2022, 1, 3), date(2023, 1, 2)]
    assert fit_days(sessions, "Q", first_fit=TEST_FIRST_FIT) == [date(2022, 10, 3), date(2023, 1, 2), date(2023, 4, 3)]
    assert MONTHLY == [date(2022, 9, 1), date(2022, 10, 3), date(2022, 11, 1), date(2022, 12, 1), date(2023, 1, 2),
                       date(2023, 2, 1), date(2023, 3, 1), date(2023, 4, 3), date(2023, 5, 1)]
    assert fit_days(sessions, "M") == fit_days(sessions, "M", first_fit=model_params.FIRST_FIT)
    # 每年、每季度的重训日都是按月重训日里的几个：模型文件三种频率共用
    for cadence in ("Y", "Q"):
        assert set(fit_days(sessions, cadence, first_fit=TEST_FIRST_FIT)) <= set(MONTHLY)


def test_a_day_uses_the_last_fit_not_after_it() -> None:
    fits = [date(2023, 1, 2), date(2023, 4, 3)]
    assert active_fit(date(2022, 12, 30), fits) is None
    assert active_fit(date(2023, 1, 2), fits) == date(2023, 1, 2)
    assert active_fit(date(2023, 4, 2), fits) == date(2023, 1, 2)
    assert active_fit(date(2023, 5, 31), fits) == date(2023, 4, 3)


@pytest.fixture(scope="module")
def rows(model_world_pristine: Path) -> TrainingRows:
    return load_rows(model_world_pristine)


def test_rows_are_sorted_by_day_then_code(rows: TrainingRows) -> None:
    assert (np.lexsort((rows.ji, rows.di)) == np.arange(len(rows.di))).all()
    assert list(rows.codes) == sorted(rows.codes) and len(rows.x) == len(rows.di)


def test_the_capped_pool_is_the_uncapped_one_minus_rows_at_70_or_more(rows: TrainingRows) -> None:
    cap, nocap = rows.member["cap70"], rows.member["nocap"]
    assert not (cap & ~nocap).any() and cap.sum() < nocap.sum()
    only_nocap = {rows.codes[j] for j in rows.ji[nocap & ~cap]}
    assert {"600030.SH", "600032.SH"} <= only_nocap  # 一直在 70 元以上的两只


def test_st_is_judged_by_the_name_at_the_time(rows: TrainingRows) -> None:
    """600029 在 2022-11-01 戴帽、600000 在 2022-03-01 摘帽：当时是 ST 的日子不在池里。"""
    def days_of(code: str) -> pd.DatetimeIndex:
        j = int(np.flatnonzero(rows.codes == code)[0])
        return rows.dates[rows.di[(rows.ji == j) & rows.member["nocap"]]]

    capped = days_of("600029.SH")
    assert len(capped) > 0 and capped.max() < pd.Timestamp("2022-11-01")
    lifted = days_of("600000.SH")
    assert len(lifted) > 0 and lifted.min() >= pd.Timestamp("2022-03-01")


def test_a_fit_trains_only_on_rows_sold_before_it(rows: TrainingRows) -> None:
    fit = int(rows.dates.searchsorted(pd.Timestamp("2023-01-02")))
    data = training_set(rows, "nocap", fit)
    used = data.rows
    assert len(used) > 0 and rows.member["nocap"][used].all()
    assert (rows.exit_day[used] < fit).all() and (rows.exit_day[used] >= 0).all()
    assert (np.diff(rows.di[data.prefix]) >= 0).all() and (rows.di[data.prefix] < fit).all()
    want = percentile(rows.ret[used], rows.di[used])
    assert np.allclose(data.y[data.keep], want) and ((0 < want) & (want <= 1)).all()
    assert np.isnan(data.y[~data.keep]).all()


def _tiny_rows() -> TrainingRows:
    """3 天 × 2 只：第 1 天那只的第 2 笔要到重训日之后才卖出。"""
    return TrainingRows(
        dates=pd.DatetimeIndex(pd.date_range("2024-01-01", periods=12, freq="B")), codes=np.array(["a", "b"]),
        di=np.array([0, 0, 1, 1, 2, 2]), ji=np.array([0, 1, 0, 1, 0, 1]), x=np.zeros((6, 28), np.float32),
        member={"nocap": np.ones(6, bool), "cap70": np.ones(6, bool)},
        ret=np.array([0.1, 0.9, 0.2, 0.3, 0.5, 0.4]), exit_day=np.array([5, 9, 6, 6, 7, 7]))


def test_unsold_rows_do_not_shape_the_labels_unless_asked_for_the_research_ones() -> None:
    """正式口径：重训日（下标 8）之后才卖出的那笔不参加训练，也不影响同一天别的行的百分位。
    十年实验的口径让它参与排名（有一点前视）：只给逐位核对用。"""
    rows = _tiny_rows()
    live, research = training_set(rows, "nocap", 8), training_set(rows, "nocap", 8, research_labels=True)
    assert (live.keep == research.keep).all() and live.keep.tolist() == [True, False, True, True, True, True]
    assert live.y[0] == 1.0 and research.y[0] == 0.5  # 第 0 天：正式只和自己比，研究版和 0.9 那笔比
    assert np.allclose(live.y[2:], research.y[2:])


def test_batches_cut_the_prefix_like_the_decade_experiment(rows: TrainingRows, monkeypatch: pytest.MonkeyPatch) -> None:
    """十年实验 `train_cadence._Batches`：在前缀上每 CHUNK 行一个窗口，送窗口里参加训练的行。"""
    monkeypatch.setattr(train_module, "CHUNK", 1000)
    data = training_set(rows, "cap70", int(rows.dates.searchsorted(pd.Timestamp("2023-03-01"))))
    got: list[tuple[np.ndarray, np.ndarray]] = []
    it = train_module._Batches(rows.x, data)
    while it.next(lambda data, label: got.append((data, label))):
        pass
    x, y, keep = rows.x[data.prefix], data.y, data.keep
    want = []
    for at in range(0, len(keep), 1000):
        picked = at + np.flatnonzero(keep[at: at + 1000])
        if len(picked):
            want.append((x[picked], y[picked]))
    assert len(got) == len(want) > 2
    for (gx, gy), (wx, wy) in zip(got, want):
        assert np.array_equal(gx, wx) and np.array_equal(gy, wy)


def test_training_writes_a_block_per_variant_and_monthly_fit(trained_world: Path) -> None:
    for variant, universe in model_params.VARIANTS.items():
        blocks = list_blocks(trained_world, variant)
        assert [b.fit_date for b in blocks] == MONTHLY
        for block in blocks:
            m = block.manifest
            assert m["variant"] == variant and m["params"] == TINY_PARAMS and m["seeds"] == [0, 1, 2]
            assert m["train"]["rows"] > 0 and date.fromisoformat(m["train"]["last_exit"]) < block.fit_date
            assert m["features"]["names"][0] == "pct_chg" and len(m["features"]["names"]) == 28
            assert m["universe"]["price_max"] == universe.price_max and m["universe"]["price_min"] == 2.0
            assert m["min_bars"] == 150 and m["chunk"] == 250_000 and m["label"]["x"] == 5
            assert set(m["files"]) == {f"seed{s}.ubj" for s in (0, 1, 2)}
            assert len(load_models(block)) == 3
        sizes = [b.manifest["train"]["rows"] for b in blocks]
        assert sizes == sorted(sizes)  # 扩张窗口：越往后训练行越多
    assert list_blocks(trained_world, "cap70")[0].manifest["universe"]["price_max"] == 70.0
    assert list_blocks(trained_world, "nocap")[0].manifest["universe"]["price_max"] is None


def test_training_twice_gives_byte_identical_models(trained_world: Path, model_world: Path) -> None:
    report = train(model_world, params=TINY_PARAMS, first_fit=TEST_FIRST_FIT)
    assert len(report.trained) == 2 * len(MONTHLY) and report.skipped == ()
    for variant in model_params.VARIANTS:
        again = {b.fit_date: b for b in list_blocks(model_world, variant)}
        for block in list_blocks(trained_world, variant):
            assert again[block.fit_date].model_id == block.model_id
            assert again[block.fit_date].manifest["files"] == block.manifest["files"]


def test_rerunning_train_keeps_existing_blocks(trained_world: Path) -> None:
    block = list_blocks(trained_world, "nocap")[0]
    stamp = (block.path / "seed0.ubj").stat().st_mtime_ns
    report = train(trained_world, params=TINY_PARAMS, first_fit=TEST_FIRST_FIT)
    assert report.trained == () and len(report.skipped) == 2 * len(MONTHLY)
    assert (block.path / "seed0.ubj").stat().st_mtime_ns == stamp


def test_other_params_do_not_silently_replace_a_trained_fit(trained_world: Path) -> None:
    with pytest.raises(ModelError, match=r"18 个已训的重训日是按别的口径训的，要重训就先删掉这些目录.*params 不同.*共 18 个"):
        train(trained_world, params={**TINY_PARAMS, "max_depth": 3}, first_fit=TEST_FIRST_FIT)


def test_a_reworded_label_note_or_another_xgboost_does_not_force_a_retrain(
    trained_world: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """说明文字、xgboost 版本、线程数不比：升级 xgboost 不该让 284 个重训日全部作废。准入池、特征变了才拒绝。"""
    import rebound.model.train as train_module

    real = train_module.spec

    def drifted(variant, fit_date, params):
        out = real(variant, fit_date, params)
        return {**out, "label": {**out["label"], "target": "换了个说法"}, "xgboost": "9.9.9", "threads": 3,
                "window": "换了个说法"}

    monkeypatch.setattr(train_module, "spec", drifted)
    report = train(trained_world, params=TINY_PARAMS, first_fit=TEST_FIRST_FIT)
    assert report.trained == () and len(report.skipped) == 2 * len(MONTHLY)
    for block in list_blocks(trained_world, "nocap"):
        train_module.check_current(block)  # 打分、选股也照样能用

    def other_pool(variant, fit_date, params):
        return {**real(variant, fit_date, params), "min_bars": 999}

    monkeypatch.setattr(train_module, "spec", other_pool)
    with pytest.raises(ModelError, match="min_bars 不同"):
        train(trained_world, params=TINY_PARAMS, first_fit=TEST_FIRST_FIT)
    with pytest.raises(ModelError, match="min_bars 不同"):
        train_module.check_current(list_blocks(trained_world, "nocap")[0])


def test_a_block_copied_under_another_fit_date_is_refused(trained_world: Path) -> None:
    import shutil

    import rebound.model.train as train_module

    source = list_blocks(trained_world, "nocap")[0]
    target = source.path.with_name("2024-01-02")
    shutil.copytree(source.path, target)
    copied = next(b for b in list_blocks(trained_world, "nocap") if b.path == target)
    with pytest.raises(ModelError, match="fit_date 不同"):
        train_module.check_current(copied)


def test_a_manifest_naming_a_file_outside_its_directory_is_refused(trained_world: Path) -> None:
    from rebound.model.artifacts import model_id

    block = list_blocks(trained_world, "nocap")[0]
    manifest = json.loads((block.path / "manifest.json").read_text(encoding="utf-8"))
    digest = manifest["files"].pop("seed0.ubj")
    manifest["files"]["../seed0.ubj"] = digest
    manifest["model_id"] = model_id(manifest)
    (block.path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelError, match="不是 seedN.ubj"):
        load_models(list_blocks(trained_world, "nocap")[0])


def test_a_truncated_manifest_is_a_model_error_not_a_traceback(trained_world: Path) -> None:
    block = list_blocks(trained_world, "nocap")[0]
    (block.path / "manifest.json").write_text('{"variant": "noc', encoding="utf-8")
    with pytest.raises(ModelError, match="读不出来"):
        list_blocks(trained_world, "nocap")


def test_a_changed_model_file_or_manifest_is_refused(trained_world: Path) -> None:
    block = list_blocks(trained_world, "nocap")[0]
    path = block.path / "seed1.ubj"
    path.write_bytes(path.read_bytes()[:-1] + b"\x00")
    with pytest.raises(ModelError, match="seed1.ubj"):
        load_models(block)
    other = list_blocks(trained_world, "cap70")[0]
    manifest = json.loads((other.path / "manifest.json").read_text(encoding="utf-8"))
    manifest["params"]["max_depth"] = 9
    (other.path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelError, match="model_id"):
        load_models(list_blocks(trained_world, "cap70")[0])


def test_a_half_written_block_directory_is_ignored(trained_world: Path) -> None:
    root = models_root(trained_world) / "nocap"
    shutil.copytree(root / "2023-01-02", root / "2023-01-02.tmp-123")
    assert [b.fit_date for b in list_blocks(trained_world, "nocap")] == MONTHLY


def test_training_without_enough_history_says_so(model_world: Path) -> None:
    with WriteStore(model_world) as store:
        store.connection.execute("DELETE FROM daily WHERE trade_date >= ?", [date(2022, 8, 1)])
    with pytest.raises(DataError, match="2022-09-01"):
        train(model_world, params=TINY_PARAMS, first_fit=TEST_FIRST_FIT)


def test_a_fit_with_no_sold_pool_rows_before_it_says_so(model_world: Path) -> None:
    """第一个重训日太早（还没有一行满 150 根日线）：训练集是空的，报错而不是训一个空模型。"""
    with pytest.raises(DataError, match="2020-07-01.*没有已卖出"):
        train(model_world, params=TINY_PARAMS, first_fit=date(2020, 7, 1))


def test_training_refuses_a_database_without_name_history(model_world: Path) -> None:
    with WriteStore(model_world) as store:
        store.connection.execute("DELETE FROM namechange")
    with pytest.raises(DataError, match="namechange"):
        train(model_world, params=TINY_PARAMS, first_fit=TEST_FIRST_FIT)
