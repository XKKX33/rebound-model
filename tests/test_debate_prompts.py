"""提示词单一事实源一致性、预置/系统提示词检查接口与单股 User JSON 构造测试。"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from rebound import config as app_config
from rebound.debate import contracts, methodology, prompts, web
from rebound.debate.contracts import AGENT_ROLES, Methodology


def _http_get_json(url: str) -> tuple[int, dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _sample_request(meth: Methodology) -> dict[str, Any]:
    cand_a = {
        "ts_code": "000001.SZ",
        "name": "平安银行",
        "industry": "银行",
        "model": {"rank": 1, "pct": 99.5, "score": 0.91, "in_model_picks": True},
        "features": {"r5": -3.2, "r20": -8.5, "from_low60": 5.1, "dev_ma20": -4.0, "vol5": 1.2},
        "feature_pct": {"r5": 20.0},
        "bars": [{"trade_date": "2024-03-15", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1}],
        "industry_stats": {"pool_count": 20, "top_count": 2, "rank_in_industry": 1, "median_r5": -1.0, "median_r20": -2.0},
        "flags": [{"code": "industry_crowded", "text": "同行业较多"}],
    }
    cand_b = {
        "ts_code": "000002.SZ",
        "name": "万科A",
        "industry": "房地产",
        "model": {"rank": 2, "pct": 98.1, "score": 0.87, "in_model_picks": True},
        "features": {"r5": 1.1, "r20": -4.2, "from_low60": 12.0, "dev_ma20": 0.5, "vol5": 0.9},
        "feature_pct": {"r5": 60.0},
        "bars": [{"trade_date": "2024-03-15", "open": 9.0, "high": 9.2, "low": 8.9, "close": 9.1}],
        "industry_stats": {"pool_count": 15, "top_count": 1, "rank_in_industry": 1, "median_r5": 0.2, "median_r20": -1.5},
        "flags": [],
    }
    return {
        "protocol_version": 4,
        "run_id": "2024-03-15_test",
        "as_of": "2024-03-15",
        "limits": {"candidates": 2, "final": 2, "max_per_industry": 2, "concurrency": 2, "call_timeout_seconds": 60},
        "methodology": meth.payload(),
        "context": {
            "as_of": "2024-03-15",
            "variant": "nocap",
            "cadence": "M",
            "fit_date": "2024-03-01",
            "model_id": "test",
            "pool_size": 2000,
            "candidates": 2,
            "final": 2,
            "max_per_industry": 2,
        },
        "model_picks": [
            {"ts_code": "000001.SZ", "name": "平安银行", "industry": "银行", "rank": 1, "debated": True},
            {"ts_code": "000002.SZ", "name": "万科A", "industry": "房地产", "rank": 2, "debated": True},
        ],
        "candidates": [cand_a, cand_b],
    }


def _sample_stocks() -> dict[str, dict[str, Any]]:
    return {
        "000001.SZ": {
            "ts_code": "000001.SZ",
            "analysts": [
                {"ts_code": "000001.SZ", "role": "model_analyst", "stance": "bull", "conclusion": "模型靠前", "risks": [], "citations": ["features.r5"]},
                {"ts_code": "000001.SZ", "role": "chart_analyst", "stance": "neutral", "conclusion": "缩量企稳", "risks": [], "citations": ["bars[-1].close"]},
            ],
            "debate": [
                {"ts_code": "000001.SZ", "round": "bull", "argument": "多方立论", "citations": ["features.r20"]},
                {"ts_code": "000001.SZ", "round": "bear", "argument": "空方反驳", "citations": ["features.dev_ma20"]},
                {"ts_code": "000001.SZ", "round": "bull_counter", "argument": "多方回应", "citations": ["features.vol5"]},
            ],
            "verdict": {
                "ts_code": "000001.SZ",
                "verdict": "看多",
                "score": 72,
                "thesis": "具备弹性",
                "risks": ["行业拥挤"],
                "action": "高开超3%放弃",
                "citations": ["features.r5"],
            },
        },
        "000002.SZ": {
            "ts_code": "000002.SZ",
            "analysts": [],
            "debate": [],
            "verdict": {
                "ts_code": "000002.SZ",
                "verdict": "看多",
                "score": 84,
                "thesis": "量价更好",
                "risks": [],
                "action": "正常竞价参与",
                "citations": ["features.vol5"],
            },
        },
    }


def test_python_and_typescript_system_prompts_match_byte_for_byte() -> None:
    """方案 A 核心不变量：Python 与 Node 侧车从同一份 prompt_rules.json 拼出的 7 个 system prompt 逐字完全一致。"""
    node_bin = shutil.which("node")
    if not node_bin:
        pytest.skip("未找到 node 可执行文件")

    meth = methodology.load_methodology()
    ts_script = (
        "import { buildSystemPrompt } from './debate_agent/src/prompts.ts';"
        "import { AGENT_ROLES } from './debate_agent/src/contracts.ts';"
        "const chunks = [];"
        "for await (const c of process.stdin) chunks.push(c);"
        "const meth = JSON.parse(Buffer.concat(chunks).toString('utf8'));"
        "const out = Object.fromEntries(AGENT_ROLES.map((r) => [r, buildSystemPrompt(r, meth)]));"
        "process.stdout.write(JSON.stringify(out));"
    )
    proc = subprocess.run(
        [node_bin, "--input-type=module", "-e", ts_script],
        input=json.dumps(meth.payload(), ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        cwd=str(app_config.ROOT),
        check=True,
        timeout=15,
    )
    ts_prompts = json.loads(proc.stdout.decode("utf-8"))
    assert set(ts_prompts.keys()) == set(AGENT_ROLES)

    for role in AGENT_ROLES:
        py_prompt = prompts.build_debate_system_prompt(role, meth)
        assert py_prompt == ts_prompts[role], f"角色 {role} 的 Python 与 TypeScript system prompt 不一致"


def test_inspect_prompts_detects_frozen_diff_and_switches_source() -> None:
    settings = app_config.load_settings()
    ws_meth = methodology.load_methodology(root=settings.root)
    frozen_briefs = dict(ws_meth.role_briefs)
    frozen_briefs["model_analyst"] = "【冻结历史版本】自定义模型解读员职责"
    frozen_req = {"methodology": {"text": ws_meth.text, "role_briefs": frozen_briefs}}

    auto_out = prompts.inspect_prompts(settings=settings, frozen_request=frozen_req, source_preference="auto")
    assert auto_out["source"] == "frozen"
    assert auto_out["has_frozen"] is True
    assert auto_out["differs_from_workspace"] is True
    assert auto_out["methodology_changed"] is False
    assert auto_out["changed_roles"] == ["model_analyst"]
    ma_frozen = next(r for r in auto_out["debate_roles"] if r["id"] == "model_analyst")
    assert ma_frozen["role_brief"] == "【冻结历史版本】自定义模型解读员职责"
    assert ma_frozen["changed_from_workspace"] is True
    assert len(auto_out["debate_roles"]) == 7
    assert len(auto_out["chat_roles"]) == 8
    assert len(auto_out["chat_tools"]) == 9

    ws_out = prompts.inspect_prompts(settings=settings, frozen_request=frozen_req, source_preference="workspace")
    assert ws_out["source"] == "workspace"
    assert ws_out["has_frozen"] is True
    assert ws_out["differs_from_workspace"] is True
    ma_ws = next(r for r in ws_out["debate_roles"] if r["id"] == "model_analyst")
    assert ma_ws["role_brief"] == ws_meth.role_briefs["model_analyst"].strip()


def test_build_role_user_payload_matches_workflow_visibility_and_order() -> None:
    meth = methodology.load_methodology()
    req = _sample_request(meth)
    stocks = _sample_stocks()

    # 1. 两位分析师只看 context + candidate
    for role in ("model_analyst", "chart_analyst"):
        out = prompts.build_role_user_payload(role, req, ts_code="000001.SZ", all_stocks=stocks)
        assert out["ready"] is True
        assert list(out["payload"].keys()) == ["context", "candidate"]
        assert out["payload"]["candidate"]["ts_code"] == "000001.SZ"

    # 2. bull 拿到两位分析师
    bull_out = prompts.build_role_user_payload("bull", req, ts_code="000001.SZ", all_stocks=stocks)
    assert bull_out["ready"] is True
    assert [a["role"] for a in bull_out["payload"]["analyst_views"]] == ["model_analyst", "chart_analyst"]
    assert "debate" not in bull_out["payload"]

    # 3. bear 只拿到 bull 发言；bull_counter 拿到 bull + bear；risk_chair 拿到 3 轮 + roster
    bear_out = prompts.build_role_user_payload("bear", req, ts_code="000001.SZ", all_stocks=stocks)
    assert [d["round"] for d in bear_out["payload"]["debate"]] == ["bull"]

    counter_out = prompts.build_role_user_payload("bull_counter", req, ts_code="000001.SZ", all_stocks=stocks)
    assert [d["round"] for d in counter_out["payload"]["debate"]] == ["bull", "bear"]

    risk_out = prompts.build_role_user_payload("risk_chair", req, ts_code="000001.SZ", all_stocks=stocks)
    assert risk_out["ready"] is True
    assert [d["round"] for d in risk_out["payload"]["debate"]] == ["bull", "bear", "bull_counter"]
    assert len(risk_out["payload"]["roster"]) == 2
    assert risk_out["payload"]["roster"][0]["flags"] == ["industry_crowded"]

    # 4. final_pick 按 score 降序排列（000002.SZ 84 分排在 000001.SZ 72 分前面）
    final_out = prompts.build_role_user_payload("final_pick", req, all_stocks=stocks)
    assert final_out["ready"] is True
    assert final_out["payload"]["final_count"] == 2
    assert [row["ts_code"] for row in final_out["payload"]["candidates"]] == ["000002.SZ", "000001.SZ"]

    # 5. 前序未完成时优雅降级并报告 missing_deps
    partial_out = prompts.build_role_user_payload("bear", req, ts_code="000002.SZ", all_stocks=stocks)
    assert partial_out["ready"] is False
    assert "model_analyst" in partial_out["missing_deps"]
    assert "bull" in partial_out["missing_deps"]


def test_web_prompts_and_payload_http_endpoints(tmp_path: Path) -> None:
    settings = app_config.load_settings()
    db_path = tmp_path / "market.duckdb"
    db_path.write_bytes(b"")

    meth = methodology.load_methodology(root=settings.root)
    req = _sample_request(meth)
    stocks = _sample_stocks()

    as_of = "2024-03-15"
    hash_prefix = "abcdef123456"
    rdir = tmp_path / "market.debate" / f"{settings.model.variant}-{settings.model.cadence}" / as_of / hash_prefix
    (rdir / "stocks").mkdir(parents=True, exist_ok=True)
    (rdir / "request.json").write_text(json.dumps(req, ensure_ascii=False), encoding="utf-8")
    for code, st in stocks.items():
        (rdir / "stocks" / f"{code}.json").write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")

    server, _hub = web.create_server(settings, default_db=db_path, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        qs = urllib.parse.urlencode({"db": str(db_path), "as_of": as_of, "hash": hash_prefix})

        status, body = _http_get_json(f"{base}/api/prompts?{qs}")
        assert status == 200
        assert body["source"] == "frozen"
        assert len(body["debate_roles"]) == 7
        assert len(body["chat_roles"]) == 8

        p_qs = urllib.parse.urlencode(
            {"db": str(db_path), "as_of": as_of, "hash": hash_prefix, "role": "risk_chair", "ts_code": "000001.SZ"}
        )
        status, payload_body = _http_get_json(f"{base}/api/prompts/payload?{p_qs}")
        assert status == 200
        assert payload_body["ready"] is True
        assert payload_body["ts_code"] == "000001.SZ"
        assert "roster" in payload_body["payload"]

        c_qs = urllib.parse.urlencode(
            {"db": str(db_path), "as_of": as_of, "hash": hash_prefix, "role": "chat:research_copilot", "ts_code": "000001.SZ"}
        )
        status, chat_body = _http_get_json(f"{base}/api/prompts/payload?{c_qs}")
        assert status == 200
        assert chat_body["ready"] is True
        assert "## 当前页面常驻运行上下文（JSON）" in chat_body["full_system_prompt"]

        status, err_body = _http_get_json(f"{base}/api/prompts/payload?{qs}")
        assert status == 400
        assert "role" in err_body["error"]
    finally:
        server.shutdown()
        server.server_close()
