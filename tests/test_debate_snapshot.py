"""候选快照（`rebound/debate/snapshot.py`）测试：单位换算、前复权、flags、行业统计与真实侧车联调。"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import MODEL_SESSIONS, TEST_FIRST_FIT

from rebound import config as app_config
from rebound.data.db import WriteStore
from rebound.debate import sidecar, snapshot
from rebound.features import FEATURES
from rebound.model.artifacts import models_root
from rebound.model.score import fill_scores
from rebound.select import pick_day

DAY = date(2023, 3, 15)


def _copy_world(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    shutil.copytree(models_root(src), models_root(dst))
    return dst


@pytest.fixture(scope="module")
def scored_pristine(trained_world_pristine: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = _copy_world(trained_world_pristine, tmp_path_factory.mktemp("debate-snap") / "market.duckdb")
    fill_scores(path, first_fit=TEST_FIRST_FIT)
    return path


@pytest.fixture
def scored(scored_pristine: Path, tmp_path: Path) -> Path:
    return _copy_world(scored_pristine, tmp_path / "market.duckdb")


def test_snapshot_order_units_pct_and_empty_industry(scored: Path, settings) -> None:
    # 给排名第 1 的票设置空行业，检查归为「（无行业）」
    day_pick = pick_day(scored, as_of=DAY)
    first_code = str(day_pick.pool.iloc[0]["ts_code"])
    with WriteStore(scored) as store:
        store.connection.execute("UPDATE stock_basic SET industry = '' WHERE ts_code = ?", [first_code])

    snap = snapshot.build_request(scored, settings=settings, as_of=DAY)
    req = snap.request
    assert req.as_of == DAY.isoformat()
    assert len(req.candidates) == settings.debate.candidates
    assert [c.ts_code for c in req.candidates] == list(day_pick.pool["ts_code"].head(settings.debate.candidates))
    assert req.candidates[0].industry == "（无行业）"
    assert req.candidates[0].model.rank == 1
    assert 0.0 < req.candidates[0].model.pct <= 100.0
    assert len(req.candidates[0].bars) == settings.debate.bars
    assert req.candidates[0].bars[-1].date == DAY.isoformat()

    # feature_pct 只含 26 个连续特征，且全部在 0~100 之间
    for c in req.candidates:
        assert "ma20_gt_ma60" not in c.feature_pct and "limit_up_today" not in c.feature_pct
        assert all(0.0 <= v <= 100.0 for v in c.feature_pct.values())
        assert set(c.features).issubset(set(FEATURES))

    # 单位换算：r5 原始小数 0.1 -> 显示 10.0（对照特征面板）
    from rebound.features import feature_panel
    from rebound.model.train import open_store

    with open_store(scored) as store:
        panel = feature_panel(store, DAY, DAY)
    row = panel[panel["ts_code"] == first_code].iloc[0]
    assert req.candidates[0].features["r5"] == round(float(row["r5"]) * 100.0, 2)
    assert req.candidates[0].features["vol5"] == round(float(row["vol5"]), 2)
    assert req.candidates[0].features["rank_r20"] == round(float(row["rank_r20"]) * 100.0, 1)


def test_ex_dividend_bars_forward_adjustment(scored: Path, settings) -> None:
    day_pick = pick_day(scored, as_of=DAY)
    code = str(day_pick.pool.iloc[0]["ts_code"])
    prev_day = MODEL_SESSIONS[MODEL_SESSIONS.index(DAY) - 1]
    with WriteStore(scored) as store:
        # 构造信号日除权（因子翻倍，价格减半）：前一日价格应按 factor_prev / factor_D 折半
        store.connection.execute(
            "UPDATE adj_factor SET adj_factor = 2.0 WHERE ts_code = ? AND trade_date >= ?",
            [code, DAY],
        )
        orig_close = float(
            store.connection.execute(
                "SELECT close FROM daily WHERE ts_code = ? AND trade_date = ?", [code, prev_day]
            ).fetchone()[0]
        )

    snap = snapshot.build_request(scored, settings=settings, as_of=DAY)
    cand = snap.request.candidates[0]
    assert cand.bars[-2].date == prev_day.isoformat()
    assert cand.bars[-2].close == round(orig_close * 0.5, 2)


def test_flags_cover_limit_up_one_word_streak_down_and_missing_features(scored: Path, settings) -> None:
    day_pick = pick_day(scored, as_of=DAY)
    c_up = str(day_pick.pool.iloc[0]["ts_code"])
    c_dn = str(day_pick.pool.iloc[1]["ts_code"])
    c_short = str(day_pick.pool.iloc[2]["ts_code"])
    idx_d = MODEL_SESSIONS.index(DAY)
    prev1 = MODEL_SESSIONS[idx_d - 1]
    prev2 = MODEL_SESSIONS[idx_d - 2]

    with WriteStore(scored) as store:
        # c_up: 连板 2 天，且信号日为一字涨停
        for d in (prev1, DAY):
            up = float(
                store.connection.execute(
                    "SELECT up_limit FROM daily_limit WHERE ts_code = ? AND trade_date = ?", [c_up, d]
                ).fetchone()[0]
            )
            if d == DAY:
                store.connection.execute(
                    "UPDATE daily SET open=?, high=?, low=?, close=? WHERE ts_code=? AND trade_date=?",
                    [up, up, up, up, c_up, d],
                )
            else:
                store.connection.execute(
                    "UPDATE daily SET close=? WHERE ts_code=? AND trade_date=?",
                    [up, c_up, d],
                )
        # c_dn: 最近 5 天有一天收盘跌停
        dn = float(
            store.connection.execute(
                "SELECT down_limit FROM daily_limit WHERE ts_code = ? AND trade_date = ?", [c_dn, prev2]
            ).fetchone()[0]
        )
        store.connection.execute(
            "UPDATE daily SET close=? WHERE ts_code=? AND trade_date=?", [dn, c_dn, prev2]
        )
        # c_short: 删掉早期的行制造 bar_no < 120 使得 r120 为 NaN（通过直接删除该票早期日线并保留已有分数）
        store.connection.execute(
            "DELETE FROM daily WHERE ts_code = ? AND trade_date < ?",
            [c_short, MODEL_SESSIONS[idx_d - 80]],
        )

    snap = snapshot.build_request(scored, settings=settings, as_of=DAY)
    by_code = {c.ts_code: c for c in snap.request.candidates}
    codes_up = {f.code for f in by_code[c_up].flags}
    assert {"limit_up_today", "one_word_limit_up", "limit_up_streak", "industry_crowded"} <= codes_up
    assert "连续 2 天收盘涨停" in [f.text for f in by_code[c_up].flags]

    assert "limit_down_recent" in {f.code for f in by_code[c_dn].flags}

    short_cand = by_code[c_short]
    assert "r120" not in short_cand.features and "r120" not in short_cand.feature_pct
    missing_flags = [f for f in short_cand.flags if f.code == "missing_features"]
    assert missing_flags and "r120" in missing_flags[0].text


def test_model_pick_outside_debate_candidates_marks_debated_false(
    scored: Path, tmp_path: Path, settings,
) -> None:
    # 送辩只数设为 6，同时让第 1~8 名挤在同一行业，使得第 6 只模型精选落在第 7 名以后（在送辩范围外）
    day_pick = pick_day(scored, as_of=DAY)
    top_codes = [str(c) for c in day_pick.pool["ts_code"].head(7)]
    with WriteStore(scored) as store:
        for code in top_codes:
            store.connection.execute("UPDATE stock_basic SET industry = '同一行业' WHERE ts_code = ?", [code])

    cfg_path = tmp_path / "config" / "settings.yaml"
    cfg_path.parent.mkdir()
    base = (Path(app_config.ROOT) / "config" / "settings.yaml").read_text(encoding="utf-8")
    cfg_path.write_text(
        base + "\ndebate:\n  candidates: 6\n  min_debated: 6\n",
        encoding="utf-8",
    )
    custom = app_config.load_settings(cfg_path)
    snap = snapshot.build_request(scored, settings=custom, as_of=DAY)
    outside = [item for item in snap.request.model_picks if not item.debated]
    assert len(outside) >= 1 and all(item.rank > 6 for item in outside)


@pytest.mark.skipif(
    shutil.which("node") is None or not (sidecar.SIDECAR_DIR / "node_modules").is_dir(),
    reason="需要本机 Node 与 debate_agent/node_modules",
)
def test_snapshot_request_passes_real_sidecar_validate(scored: Path, settings) -> None:
    snap = snapshot.build_request(scored, settings=settings, as_of=DAY)
    receipt = sidecar.Sidecar(settings).validate(snap.request)
    assert receipt["protocol_version"] == 4
    assert receipt["candidates"] == settings.debate.candidates
    assert receipt["final"] == settings.pick.top_n
