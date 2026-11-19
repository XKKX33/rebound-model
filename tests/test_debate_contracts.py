"""辩论协议契约与方法论加载测试（含与 TS 侧车常量的交叉核对）。"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from rebound.debate import contracts
from rebound.debate.methodology import load_methodology

ROOT = Path(__file__).resolve().parents[1]


def _ts_const(source: str, name: str) -> object:
    match = re.search(rf"export const {name}\b[^=]*=\s*(\[[^\]]*\]|\d+|\"[^\"]*\")", source, re.DOTALL)
    assert match is not None, f"contracts.ts 找不到常量 {name}"
    raw = re.sub(r"\bas const\b", "", match.group(1)).strip()
    raw = re.sub(r",(\s*\])", r"\1", raw)
    return json.loads(raw)


def make_candidate(
    code: str,
    rank: int,
    industry: str = "半导体",
    *,
    as_of: str = "2026-10-14",
) -> contracts.Candidate:
    return contracts.Candidate(
        ts_code=code,
        name=f"票{rank}",
        industry=industry,
        model=contracts.CandidateModel(rank=rank, pct=round(100.0 - rank, 1), score=0.8 - rank * 0.01,
                                       in_model_picks=rank <= 2),
        features={"pct_chg": 1.2, "r5": -2.5, "r20": -8.0, "vol5": 1.4},
        feature_pct={"pct_chg": 60.0, "r5": 25.0, "r20": 15.0, "vol5": 70.0},
        bars=(
            contracts.CandidateBar(
                date="2026-10-13", open=10.0, high=10.3, low=9.8, close=10.1,
                pct_chg=1.0, amount_yi=1.2, limit=None, vol5=1.1,
            ),
            contracts.CandidateBar(
                date=as_of, open=10.1, high=10.4, low=10.0, close=10.25,
                pct_chg=1.49, amount_yi=1.5, limit=None, vol5=1.4, dif_ratio=-0.3, hist_ratio=0.1,
            ),
        ),
        industry_stats=contracts.IndustryStats(
            pool_count=10, top_count=2, rank_in_industry=1, median_r5=-1.5, median_r20=-4.2,
        ),
        flags=(),
    )


def make_request(
    candidates: tuple[contracts.Candidate, ...] | None = None,
    *,
    final: int = 2,
    max_per_industry: int = 2,
    methodology: contracts.Methodology | None = None,
) -> contracts.RunRequest:
    cands = candidates or (
        make_candidate("600001.SH", 1, "银行"),
        make_candidate("600002.SH", 2, "银行"),
        make_candidate("600003.SH", 3, "电力"),
    )
    meth = methodology or load_methodology()
    limits = contracts.RunLimits(
        candidates=len(cands), final=min(final, len(cands)),
        max_per_industry=max_per_industry, concurrency=2, call_timeout_seconds=120,
    )
    llm = contracts.LlmSettings(
        id="step-5-preview", base_url="https://example.invalid/v1",
        reasoning_effort="high", max_tokens=16384, temperature=0.2,
    )
    context = contracts.RunContext(
        as_of="2026-10-14", model_label="反弹模型", variant="nocap", cadence="M",
        fit_date="2026-10-09", model_id="m" * 12, scores_stored=True,
        pool_size=100, candidates=len(cands), final=limits.final,
        max_per_industry=max_per_industry,
        pool=contracts.PoolSummary(median_pct_chg=0.2, median_r5=-1.1, median_r20=-3.4, up_ratio=52.0),
    )
    model_picks = tuple(
        contracts.ModelPickItem(ts_code=c.ts_code, name=c.name, industry=c.industry, rank=c.model.rank, debated=True)
        for c in cands[: limits.final]
    )
    cand_hash = contracts.fingerprint([c.payload() for c in cands])
    inp_hash = contracts.input_fingerprint(
        as_of="2026-10-14", limits=limits, methodology=meth, llm=llm,
        context=context, model_picks=model_picks, candidates=cands,
    )
    return contracts.RunRequest(
        run_id=f"2026-10-14-nocap-M-{inp_hash[:12]}",
        as_of="2026-10-14",
        candidate_hash=cand_hash,
        input_hash=inp_hash,
        limits=limits,
        methodology=meth,
        llm=llm,
        context=context,
        model_picks=model_picks,
        candidates=cands,
    )


def _debated_payload(request: contracts.RunRequest, *, fail_last: bool = False) -> dict:
    active = request.codes[:-1] if fail_last else request.codes
    failed = [{"ts_code": request.codes[-1], "role": "bear", "error": "超时"}] if fail_last else []
    return {
        "type": "debated",
        "run_id": request.run_id,
        "candidate_hash": request.candidate_hash,
        "input_hash": request.input_hash,
        "candidate_codes": list(request.codes),
        "analysts": [
            {
                "ts_code": code, "role": role, "stance": "bull",
                "conclusion": f"{role} 结论", "risks": ["风险"], "citations": ["features.r5"],
            }
            for code in active for role in contracts.ANALYST_ROLES
        ],
        "debate": [
            {"ts_code": code, "round": rnd, "argument": f"{rnd} 论点", "citations": ["bars[-1].close"]}
            for code in active for rnd in contracts.DEBATE_ENTRY_ROUNDS
        ],
        "verdicts": [
            {
                "ts_code": code, "verdict": "看多", "score": 85 - idx * 5,
                "thesis": "核心判断", "risks": ["风险"], "action": "开盘可看", "citations": ["features.r20"],
            }
            for idx, code in enumerate(active)
        ],
        "failed_stocks": failed,
        "usage": {"calls": len(active) * 6, "failed_calls": len(failed), "input_tokens": 100, "output_tokens": 50},
    }


def test_python_and_typescript_constants_match() -> None:
    ts = (ROOT / "debate_agent" / "src" / "contracts.ts").read_text(encoding="utf-8")
    assert _ts_const(ts, "PROTOCOL_VERSION") == contracts.PROTOCOL_VERSION == 4
    assert tuple(_ts_const(ts, "ANALYST_ROLES")) == contracts.ANALYST_ROLES
    assert tuple(_ts_const(ts, "DEBATE_ROUNDS")) == contracts.DEBATE_ROUNDS
    assert _ts_const(ts, "FINAL_PICK_ROLE") == contracts.FINAL_PICK_ROLE
    assert tuple(_ts_const(ts, "FLAG_CODES")) == contracts.FLAG_CODES
    assert len(contracts.AGENT_ROLES) == 7


def test_methodology_loads_all_7_roles_and_rejects_duplicates_or_missing(tmp_path: Path) -> None:
    meth = load_methodology()
    assert set(meth.role_briefs) == set(contracts.AGENT_ROLES)
    assert "反弹模型" in meth.text
    assert all(v.strip() for v in meth.role_briefs.values())

    doc = (ROOT / "docs" / "debate-methodology.md").read_text(encoding="utf-8")
    dup = tmp_path / "dup.md"
    dup.write_text(doc + "\n### bull\n```text\n重复一次\n```\n", encoding="utf-8")
    with pytest.raises(contracts.ContractError, match="写了两次"):
        load_methodology(dup)

    missing = tmp_path / "missing.md"
    missing.write_text(doc.replace("### risk_chair", "### removed_role"), encoding="utf-8")
    with pytest.raises(contracts.ContractError, match="risk_chair"):
        load_methodology(missing)


def test_input_fingerprint_changes_when_methodology_or_candidate_changes() -> None:
    req1 = make_request()
    edited_meth = contracts.Methodology(
        text=req1.methodology.text + "\n新增一条。", role_briefs=req1.methodology.role_briefs,
    )
    req2 = make_request(methodology=edited_meth)
    assert req1.input_hash != req2.input_hash
    assert req1.candidate_hash == req2.candidate_hash

    cands3 = (make_candidate("600001.SH", 1, "银行"), make_candidate("600002.SH", 2, "电力"),
              make_candidate("600003.SH", 3, "医药"))
    req3 = make_request(candidates=cands3)
    assert req3.candidate_hash != req1.candidate_hash
    assert req3.input_hash != req1.input_hash


def test_verify_debated_accepts_valid_and_rejects_tampered() -> None:
    req = make_request()
    res = contracts.parse_debated(_debated_payload(req, fail_last=True))
    contracts.verify_debated(res, req)

    bad_hash = contracts.parse_debated({**_debated_payload(req), "input_hash": "0" * 64})
    with pytest.raises(contracts.ContractError, match="input_hash"):
        contracts.verify_debated(bad_hash, req)

    bad_order = contracts.parse_debated({**_debated_payload(req), "candidate_codes": list(reversed(req.codes))})
    with pytest.raises(contracts.ContractError, match="下发顺序"):
        contracts.verify_debated(bad_order, req)

    payload_missing_role = _debated_payload(req)
    payload_missing_role["analysts"] = payload_missing_role["analysts"][:-1]
    with pytest.raises(contracts.ContractError, match="分析师"):
        contracts.verify_debated(contracts.parse_debated(payload_missing_role), req)

    payload_mismatch_set = _debated_payload(req)
    payload_mismatch_set["verdicts"].pop()
    with pytest.raises(contracts.ContractError, match="不等于下发候选"):
        contracts.verify_debated(contracts.parse_debated(payload_mismatch_set), req)


def test_verify_final_checks_count_ranks_foreign_duplicates_and_industry_cap() -> None:
    cands = (
        make_candidate("600001.SH", 1, "银行"),
        make_candidate("600002.SH", 2, "银行"),
        make_candidate("600003.SH", 3, "银行"),
        make_candidate("600004.SH", 4, "电力"),
    )
    req = make_request(candidates=cands, final=3, max_per_industry=2)
    debated = contracts.parse_debated(_debated_payload(req))

    good = contracts.parse_finalized({
        "type": "finalized", "final_count": 3,
        "picks": [
            {"ts_code": "600001.SH", "rank": 1, "reason": "a"},
            {"ts_code": "600002.SH", "rank": 2, "reason": "b"},
            {"ts_code": "600004.SH", "rank": 3, "reason": "c"},
        ],
        "reason": "合规", "usage": {"calls": 1, "failed_calls": 0, "input_tokens": 10, "output_tokens": 5},
    })
    contracts.verify_final(good, debated.verdicts, req)

    # 行业超限
    over_ind = contracts.parse_finalized({
        "type": "finalized", "final_count": 3,
        "picks": [
            {"ts_code": "600001.SH", "rank": 1, "reason": "a"},
            {"ts_code": "600002.SH", "rank": 2, "reason": "b"},
            {"ts_code": "600003.SH", "rank": 3, "reason": "c"},
        ],
        "reason": "超限", "usage": {"calls": 1, "failed_calls": 0, "input_tokens": 10, "output_tokens": 5},
    })
    with pytest.raises(contracts.ContractError, match="超过上限"):
        contracts.verify_final(over_ind, debated.verdicts, req)

    # 名次断档
    bad_rank = contracts.parse_finalized({
        "type": "finalized", "final_count": 3,
        "picks": [
            {"ts_code": "600001.SH", "rank": 1, "reason": "a"},
            {"ts_code": "600002.SH", "rank": 3, "reason": "b"},
            {"ts_code": "600004.SH", "rank": 4, "reason": "c"},
        ],
        "reason": "断档", "usage": {"calls": 1, "failed_calls": 0, "input_tokens": 10, "output_tokens": 5},
    })
    with pytest.raises(contracts.ContractError, match="连续递增"):
        contracts.verify_final(bad_rank, debated.verdicts, req)

    # 重复代码
    dup = contracts.parse_finalized({
        "type": "finalized", "final_count": 3,
        "picks": [
            {"ts_code": "600001.SH", "rank": 1, "reason": "a"},
            {"ts_code": "600001.SH", "rank": 2, "reason": "b"},
            {"ts_code": "600004.SH", "rank": 3, "reason": "c"},
        ],
        "reason": "重复", "usage": {"calls": 1, "failed_calls": 0, "input_tokens": 10, "output_tokens": 5},
    })
    with pytest.raises(contracts.ContractError, match="重复代码"):
        contracts.verify_final(dup, debated.verdicts, req)

    # 池外代码
    foreign = contracts.parse_finalized({
        "type": "finalized", "final_count": 3,
        "picks": [
            {"ts_code": "600001.SH", "rank": 1, "reason": "a"},
            {"ts_code": "600002.SH", "rank": 2, "reason": "b"},
            {"ts_code": "600999.SH", "rank": 3, "reason": "c"},
        ],
        "reason": "池外", "usage": {"calls": 1, "failed_calls": 0, "input_tokens": 10, "output_tokens": 5},
    })
    with pytest.raises(contracts.ContractError, match="候选池外"):
        contracts.verify_final(foreign, debated.verdicts, req)

    # 行业太集中时名额应自动收缩为 2
    verdicts_only_bank = debated.verdicts[:3]
    assert contracts.expected_final_count(3, ["银行", "银行", "银行"], 2) == 2
    wrong_count = contracts.parse_finalized({
        "type": "finalized", "final_count": 3,
        "picks": [
            {"ts_code": "600001.SH", "rank": 1, "reason": "a"},
            {"ts_code": "600002.SH", "rank": 2, "reason": "b"},
            {"ts_code": "600003.SH", "rank": 3, "reason": "c"},
        ],
        "reason": "未收缩", "usage": {"calls": 1, "failed_calls": 0, "input_tokens": 10, "output_tokens": 5},
    })
    with pytest.raises(contracts.ContractError, match="名额应为 2"):
        contracts.verify_final(wrong_count, verdicts_only_bank, req)


def test_stock_checkpoint_and_event_parsing() -> None:
    req = make_request()
    payload = _debated_payload(req)
    cp = contracts.stock_checkpoint({
        "verdict": payload["verdicts"][0],
        "analysts": [a for a in payload["analysts"] if a["ts_code"] == req.codes[0]],
        "debate": [d for d in payload["debate"] if d["ts_code"] == req.codes[0]],
    })
    assert cp["verdict"].ts_code == req.codes[0]

    assert contracts.parse_event("  \n") is None
    with pytest.raises(contracts.ContractError, match="非 JSON"):
        contracts.parse_event("not json")
    unknown = contracts.parse_event('{"type":"future.event"}')
    assert unknown is not None and not unknown.known
