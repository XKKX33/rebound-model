"""三模型（XGBoost F69 / 随机森林 F69 / LSTM S+F69）接入辩论工作流、对话工具与 Web UI 集成测试。"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from rebound import config as app_config
from rebound.debate import chat, ml_bridge, runs, snapshot, web
from rebound.debate.sidecar import Sidecar


def _http_get_json(url: str) -> tuple[int, dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_series_name_keeps_legacy_and_isolates_ml_models(tmp_path: Path) -> None:
    db = tmp_path / "research-2013-2026.duckdb"
    assert runs.series_name("nocap", "M") == "nocap-M"
    assert runs.series_name("nocap", "M", "return_model") == "nocap-M"
    assert runs.series_name("nocap", "M", "xgb_F69") == "nocap-M-xgb_F69"
    assert runs.series_name("nocap", "M", "rf_F69") == "nocap-M-rf_F69"
    assert runs.series_name("nocap", "M", "lstm_SF69") == "nocap-M-lstm_SF69"

    d_legacy = runs.run_dir(db, variant="nocap", cadence="M", as_of="2025-03-14", input_hash="a" * 64)
    d_xgb = runs.run_dir(
        db, variant="nocap", cadence="M", as_of="2025-03-14", input_hash="a" * 64, model_source="xgb_F69"
    )
    assert d_legacy.parent.parent.name == "nocap-M"
    assert d_xgb.parent.parent.name == "nocap-M-xgb_F69"
    assert d_legacy.parent.parent.parent == d_xgb.parent.parent.parent


@pytest.mark.skipif(not ml_bridge.is_ml_available(), reason="未检测到 data/ml/api_export 数据包")
def test_ml_bridge_and_chat_tools_and_web_api() -> None:
    settings = app_config.load_settings()
    db_path = settings.root / "data" / "research-2013-2026.duckdb"

    # 1. ml_bridge 排行榜 / 净值 / 换起点年化 / 同日共识 / 归因 / 特征目录
    lb = ml_bridge.get_leaderboard()
    assert lb["available"] is True
    assert lb["main_rule"] == "DB-K30-b8"
    assert lb["rules"] == ["DB-K30-b8"]
    assert len(lb["leaderboard"]) == 3
    assert lb["starts_summary"]["available"] is True
    assert lb["starts_summary"]["years"] == list(range(2015, 2025))
    assert lb["starts_summary"]["summary"]["ex_2015"]["rf_F69"]["best_count"] == 0
    assert any("不显著" in note for note in lb["discipline_notes"])

    nav_test = ml_bridge.get_nav_compare("DB-K30-b8", "test")
    assert set(ml_bridge.ML_MODEL_IDS).issubset(nav_test["series"].keys())
    assert "pool_nav" in nav_test["series"]
    assert nav_test["count"] > 100
    assert len(nav_test["series"]["xgb_F69"]["cagr_values"]) == nav_test["count"]
    assert nav_test["series"]["xgb_F69"]["cagr_values"][0] is None
    assert nav_test["series"]["xgb_F69"]["cagr_values"][-1] == nav_test["series"]["xgb_F69"]["stats"]["cagr"]
    assert len(nav_test["yearly_breakdown"]) >= 4

    nav_long = ml_bridge.get_nav_compare("DB-K30-b8", "long")
    assert {"hs300", "zz500", "zz1000"}.issubset(nav_long["series"].keys())

    nav_2016 = ml_bridge.get_nav_compare("DB-K30-b8", start_year=2016)
    assert nav_2016["start_year"] == 2016
    assert nav_2016["date_start"] == "2016-01-04"
    assert nav_2016["date_end"] == "2026-10-23"
    assert abs(nav_2016["series"]["xgb_F69"]["stats"]["cagr"] - 20.93) < 0.05
    assert abs(nav_2016["series"]["rf_F69"]["stats"]["cagr"] - 17.63) < 0.05
    assert abs(nav_2016["series"]["lstm_SF69"]["stats"]["cagr"] - 20.24) < 0.05
    assert {"hs300", "zz500", "zz1000"}.issubset(nav_2016["series"].keys())
    assert nav_2016["series"]["hs300"]["values"][0] == 1.0
    assert len(nav_2016["yearly_breakdown"]) == 11
    assert abs(nav_2016["yearly_breakdown"][0]["models"]["xgb_F69"]["return_pct"] - 30.98) < 0.1

    cons = ml_bridge.get_day_consensus("2025-03-14", top_n=6, with_outcome=True)
    assert cons["as_of"] == "2025-03-14"
    assert len(cons["models"]["xgb_F69"]["picks"]) == 6
    assert len(cons["consensus_rows"]) >= 6
    sample_code = cons["models"]["xgb_F69"]["picks"][0]["ts_code"]

    exp = ml_bridge.get_stock_explain_all("2025-03-14", sample_code, top=8)
    for mid in ml_bridge.ML_MODEL_IDS:
        assert exp["models"][mid]["explain_available"] is True
        assert len(exp["models"][mid]["top_factors"]) == 8

    cat = ml_bridge.get_feature_catalog()
    assert cat["available"] is True
    assert len(cat["items"]) == 80  # 69 特征 + 11 LSTM 序列通道

    # 2. 对话 3 个新只读工具
    toolbox = chat.ChatToolbox(
        db_path, variant="nocap", cadence="M", as_of="2025-03-14", model_source="xgb_F69", run_detail={}
    )
    ok1, _s1, d1 = toolbox.execute("get_model_explain", {"ts_code": sample_code, "as_of": "2025-03-14"})
    assert ok1 is True and d1["models"]["xgb_F69"]["explain_available"] is True

    ok2, _s2, d2 = toolbox.execute("compare_models_on_day", {"as_of": "2025-03-14", "top_n": 6})
    assert ok2 is True and d2["as_of"] == "2025-03-14"

    ok3, _s3, d3 = toolbox.execute("get_ml_leaderboard_and_nav", {"rule": "DB-K30-b8", "period": "long"})
    assert ok3 is True and "xgb_F69" in d3["nav_summary"]

    # 3. Web HTTP API
    server, _hub = web.create_server(settings, default_db=db_path, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        st, meta = _http_get_json(f"{base}/api/meta?model_source=xgb_F69")
        assert st == 200
        assert meta["ml_available"] is True
        assert meta["model_source"] == "xgb_F69"
        assert len(meta["model_sources"]) == 3
        assert [m["id"] for m in meta["model_sources"]] == ["xgb_F69", "rf_F69", "lstm_SF69"]

        st, ov = _http_get_json(f"{base}/api/ml/overview")
        assert st == 200 and len(ov["leaderboard"]) == 3 and ov["feature_catalog"]["available"] is True
        assert ov["starts_summary"]["available"] is True and len(ov["starts_summary"]["rows"]) == 10

        st, nav_api = _http_get_json(f"{base}/api/ml/nav?rule=DB-K30-b8&period=test")
        assert st == 200 and "xgb_F69" in nav_api["series"]

        st, nav_2019 = _http_get_json(f"{base}/api/ml/nav?rule=DB-K30-b8&start_year=2019")
        assert st == 200 and nav_2019["start_year"] == 2019
        assert abs(nav_2019["series"]["lstm_SF69"]["stats"]["cagr"] - 20.27) < 0.05
        assert len(nav_2019["yearly_breakdown"]) == 8

        st_bad, _ = _http_get_json(f"{base}/api/ml/nav?rule=DB-K30-b8&start_year=2010")
        assert st_bad == 400

        st, cons_api = _http_get_json(f"{base}/api/ml/consensus?as_of=2025-03-14&with_outcome=1")
        assert st == 200 and cons_api["with_outcome"] is True

        st, exp_api = _http_get_json(
            f"{base}/api/ml/explain?as_of=2025-03-14&ts_code={urllib.parse.quote(sample_code)}"
        )
        assert st == 200 and exp_api["models"]["lstm_SF69"]["explain_available"] is True
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(
    not (ml_bridge.is_ml_available() and (app_config.ROOT / "data" / "research-2013-2026.duckdb").is_file()),
    reason="需要本地研究库与 data/ml/api_export 数据包",
)
def test_f69_snapshot_with_extra_features_explain_and_sidecar_validate() -> None:
    """端到端验收：组装 xgb_F69 快照（含 41 新特征 + TreeSHAP + 同行模型百分位）并通过真实 Node 侧车 --validate。"""
    settings = app_config.load_settings()
    db_path = settings.root / "data" / "research-2013-2026.duckdb"
    snap = snapshot.build_request(
        db_path,
        settings=settings,
        as_of=date(2025, 3, 14),
        model_source="xgb_F69",
    )
    req = snap.request
    assert req.context.model_source == "xgb_F69"
    assert req.context.feature_set == "F69"
    assert len(req.candidates) == settings.debate.candidates

    first = req.candidates[0]
    assert first.extra_features is not None and len(first.extra_features) >= 35
    assert first.extra_feature_pct is not None and len(first.extra_feature_pct) >= 30
    assert first.explain is not None and len(first.explain["top_factors"]) == 8
    assert first.peer_models is not None
    assert set(first.peer_models.keys()) == {"rf_F69", "lstm_SF69"}
    # 确认跨模型对照不透传原始 score，只给 rank 和 pct
    assert "score" not in first.peer_models["rf_F69"]

    receipt = Sidecar(settings).validate(req)
    assert receipt["type"] == "valid"
    assert receipt["run_id"] == req.run_id
