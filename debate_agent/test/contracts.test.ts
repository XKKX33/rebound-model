/**
 * 协议 v4 契约测试：保证 Python 下发的请求形状在真正烧 token 前被严格拦住。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  AGENT_ROLES,
  ContractError,
  PROTOCOL_VERSION,
  parseRunRequest,
  parseVerdicts,
  stableJson,
  type Candidate,
  type RunRequest,
} from "../src/contracts.ts";
import { makeCandidate, makeMethodology, makeRequest } from "./fixtures.ts";

function firstCandidate(request: RunRequest): Candidate {
  const candidate = request.candidates[0];
  assert.ok(candidate !== undefined, "夹具至少要有 1 只候选股");
  return candidate;
}

function expectContractError(input: unknown, needle: string): void {
  assert.throws(
    () => parseRunRequest(input),
    (error: unknown) => {
      assert.ok(error instanceof ContractError, `应抛 ContractError，实际: ${String(error)}`);
      assert.match(error.message, new RegExp(needle));
      return true;
    },
  );
}

test("协议版本为 4 且角色恰好 7 个", () => {
  assert.equal(PROTOCOL_VERSION, 4);
  assert.deepEqual(AGENT_ROLES, [
    "model_analyst",
    "chart_analyst",
    "bull",
    "bear",
    "bull_counter",
    "risk_chair",
    "final_pick",
  ]);
  const parsed = parseRunRequest(makeRequest());
  assert.equal(parsed.protocol_version, 4);
  assert.equal(parsed.candidates.length, 2);
});

test("协议版本不符、非对象或空候选池均拒收", () => {
  expectContractError(makeRequest({ protocol_version: 3 }), "protocol_version");
  expectContractError([], "JSON 对象");
  expectContractError(makeRequest({ candidates: [] }), "candidates");
});

test("方法论缺角色、空简报或多出未知角色都拒收", () => {
  const missing = makeMethodology();
  delete missing.role_briefs.risk_chair;
  expectContractError(makeRequest({ methodology: missing }), "risk_chair");

  const extra = makeMethodology();
  extra.role_briefs.sentiment = "旧角色";
  expectContractError(makeRequest({ methodology: extra }), "sentiment");

  const blank = makeMethodology();
  blank.role_briefs.chart_analyst = "   ";
  expectContractError(makeRequest({ methodology: blank }), "chart_analyst");
});

test("final 不大于 candidates 且候选代码不重复", () => {
  const req = makeRequest();
  expectContractError(
    makeRequest({ limits: { ...req.limits, candidates: 1, final: 2 } }),
    "不能大于",
  );
  const first = firstCandidate(req);
  expectContractError(makeRequest({ candidates: [first, first] }), "重复代码");
});

test("features、feature_pct 形状与范围校验", () => {
  const base = firstCandidate(makeRequest());
  expectContractError(
    makeRequest({ candidates: [{ ...base, features: {} }] }),
    "features",
  );
  expectContractError(
    makeRequest({ candidates: [{ ...base, features: { r5: "坏" } } as unknown as Candidate] }),
    "features.r5",
  );
  expectContractError(
    makeRequest({ candidates: [{ ...base, feature_pct: { r5: 101 } }] }),
    "feature_pct.r5",
  );
});

test("bars 校验：不能为空、日期升序、可选字段与最后一条等于 as_of", () => {
  const base = firstCandidate(makeRequest());
  expectContractError(
    makeRequest({ candidates: [{ ...base, bars: [] }] }),
    "bars",
  );
  expectContractError(
    makeRequest({
      candidates: [
        {
          ...base,
          bars: [
            { ...base.bars[0]!, date: "2026-10-13" },
          ],
        },
      ],
    }),
    "必须等于 as_of",
  );
  expectContractError(
    makeRequest({
      candidates: [
        {
          ...base,
          bars: [
            { ...base.bars[1]!, date: "2026-10-14" },
            { ...base.bars[0]!, date: "2026-10-14" },
          ],
        },
      ],
    }),
    "严格升序",
  );
  expectContractError(
    makeRequest({
      candidates: [
        {
          ...base,
          bars: [
            { ...base.bars[0]!, limit: "flat" as unknown as null },
            base.bars[1]!,
          ],
        },
      ],
    }),
    "limit",
  );

  // 省略可选的 vol5 / dif_ratio / hist_ratio 时也合法
  const minimalBar = {
    date: "2026-10-14",
    open: 10,
    high: 10.2,
    low: 9.9,
    close: 10.1,
    pct_chg: 1.0,
    amount_yi: 0.9,
    limit: null,
  };
  const parsed = parseRunRequest(
    makeRequest({ candidates: [makeCandidate("000001.SZ", 1, { bars: [minimalBar] })] }),
  );
  assert.equal(parsed.candidates[0]?.bars[0]?.vol5, undefined);
});

test("industry_stats 与 flags 形状校验", () => {
  const base = firstCandidate(makeRequest());
  const ok = parseRunRequest(
    makeRequest({
      candidates: [
        {
          ...base,
          industry_stats: { ...base.industry_stats, median_r5: null, median_r20: null },
          flags: [{ code: "limit_up_today", text: "信号日收盘涨停" }],
        },
      ],
    }),
  );
  assert.equal(ok.candidates[0]?.industry_stats.median_r5, null);

  expectContractError(
    makeRequest({
      candidates: [
        {
          ...base,
          industry_stats: { ...base.industry_stats, pool_count: 0 },
        },
      ],
    }),
    "pool_count",
  );
  expectContractError(
    makeRequest({
      candidates: [
        {
          ...base,
          flags: [{ code: "unknown_flag" as "limit_up_today", text: "x" }],
        },
      ],
    }),
    "flags\\[0\\]\\.code",
  );
});

test("llm.max_tokens 越界与 parseVerdicts 非法裁决均拒收", () => {
  const req = makeRequest();
  expectContractError(
    makeRequest({ llm: { ...req.llm, max_tokens: 999_999 } }),
    "max_tokens",
  );
  assert.throws(
    () => parseVerdicts([{ ts_code: "000001.SZ", verdict: "中性", score: 50, thesis: "t", risks: [], action: "a", citations: [] }]),
    ContractError,
  );
});

test("stableJson 键序稳定且不转义非 ASCII", () => {
  assert.equal(stableJson({ b: 1, a: "看多" }), '{"a":"看多","b":1}');
  assert.equal(stableJson({ a: undefined }), '{"a":null}');
});
