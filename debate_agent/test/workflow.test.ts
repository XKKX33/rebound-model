/**
 * 编排层测试：验证 7 角色消息可见性、P1/P5/P6/P9 修正、并发与重试、独立终选。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { ANALYST_ROLES, DEBATE_ROUNDS } from "../src/contracts.ts";
import { LlmError } from "../src/provider.ts";
import { DebateRunner, expectedFinalCount } from "../src/workflow.ts";
import {
  FakeProvider,
  collector,
  happyScript,
  makeCandidate,
  makeRequest,
  makeVerdict,
  roleFromPrompt,
  type Script,
} from "./fixtures.ts";

function build(request = makeRequest(), script: Script = happyScript(1), concurrency?: number) {
  const provider = new FakeProvider(script);
  const events = collector();
  const runner = new DebateRunner(request, {
    provider,
    emit: events.emit,
    sleep: async () => {},
    ...(concurrency === undefined ? {} : { concurrency }),
  });
  return { provider, events, runner };
}

test("个股调用顺序固定：model_analyst → chart_analyst → bull → bear → bull_counter → risk_chair", async () => {
  const request = makeRequest({
    candidates: [makeCandidate("000001.SZ", 1)],
    limits: { candidates: 1, final: 1, max_per_industry: 2, concurrency: 1, call_timeout_seconds: 90 },
  });
  const { runner, provider } = build(request, happyScript(1), 1);
  const debated = await runner.debate();

  assert.deepEqual(
    provider.seen.map((entry) => entry.role),
    [...ANALYST_ROLES, ...DEBATE_ROUNDS],
  );
  assert.equal(debated.analysts.length, 2);
  assert.equal(debated.debate.length, 3);
  assert.equal(debated.verdicts.length, 1);
  assert.equal(debated.usage.calls, 6);
});

test("超时取自请求 limits.call_timeout_seconds", async () => {
  const request = makeRequest({
    candidates: [makeCandidate("000001.SZ", 1)],
    limits: { candidates: 1, final: 1, max_per_industry: 2, concurrency: 1, call_timeout_seconds: 187 },
  });
  const { runner, provider } = build(request, happyScript(1), 1);
  await runner.debate();

  assert.equal(provider.optionsOf("model_analyst")?.timeoutMs, 187_000);
  assert.equal(provider.optionsOf("risk_chair")?.timeoutMs, 187_000);
});

test("两位分析师互相看不见；空方看得到多方；反驳方看得到多空两方；风控主席收到 roster", async () => {
  const request = makeRequest();
  const { runner, provider } = build(request);
  await runner.debate();

  for (const role of ANALYST_ROLES) {
    const payload = provider.payloadOf(role);
    assert.equal(payload.analyst_views, undefined, `${role} 不该看到其他分析师`);
    assert.equal(payload.debate, undefined);
    assert.equal(payload.roster, undefined);
  }

  const bullPayload = provider.payloadOf("bull");
  assert.equal((bullPayload.analyst_views as unknown[]).length, 2);
  assert.equal(bullPayload.debate, undefined);

  // P1：bear 必须看得到 bull 的论点
  const bearDebate = provider.payloadOf("bear").debate as Array<{ round: string }>;
  assert.deepEqual(bearDebate.map((entry) => entry.round), ["bull"]);

  // bull_counter 看到 bull 与 bear
  const counterDebate = provider.payloadOf("bull_counter").debate as Array<{ round: string }>;
  assert.deepEqual(counterDebate.map((entry) => entry.round), ["bull", "bear"]);

  // P5：risk_chair 看到三轮辩论与全候选简表 roster
  const riskPayload = provider.payloadOf("risk_chair");
  const riskDebate = riskPayload.debate as Array<{ round: string }>;
  assert.deepEqual(riskDebate.map((entry) => entry.round), ["bull", "bear", "bull_counter"]);
  const roster = riskPayload.roster as Array<{ ts_code: string; r5: number | null; flags: string[] }>;
  assert.deepEqual(roster.map((row) => row.ts_code), ["000001.SZ", "000002.SZ"]);
  assert.equal(roster[0]?.r5, -3.2);
  assert.deepEqual(roster[0]?.flags, []);
});

test("同一角色对不同候选的 system prompt 逐字相同", async () => {
  const { runner, provider } = build(makeRequest());
  await runner.debate();
  assert.equal(provider.promptOf("model_analyst", 0), provider.promptOf("model_analyst", 1));
  assert.equal(provider.promptOf("risk_chair", 0), provider.promptOf("risk_chair", 1));
});

test("终选 rows 按风控分降序（同分按模型名次升序）并带上 model_picks", async () => {
  const c1 = makeCandidate("000001.SZ", 1, { industry: "半导体" });
  const c2 = makeCandidate("000002.SZ", 2, { industry: "电力" });
  const c3 = makeCandidate("000003.SZ", 3, { industry: "医药" });
  const request = makeRequest({
    candidates: [c1, c2, c3],
    limits: { candidates: 3, final: 2, max_per_industry: 2, concurrency: 2, call_timeout_seconds: 120 },
  });
  const { runner, provider } = build(request, happyScript(2));
  const verdicts = [makeVerdict("000001.SZ", 70), makeVerdict("000002.SZ", 90), makeVerdict("000003.SZ", 70)];

  const finalized = await runner.finalizeWith(verdicts);
  const finalPayload = provider.payloadOf("final_pick");
  const rows = finalPayload.candidates as Array<{ ts_code: string; score: number; model_rank: number }>;
  assert.deepEqual(rows.map((r) => r.ts_code), ["000002.SZ", "000001.SZ", "000003.SZ"]);
  assert.equal(finalPayload.max_per_industry, 2);
  assert.deepEqual(finalPayload.model_picks, request.model_picks);
  assert.deepEqual(finalized.picks.map((p) => p.ts_code), ["000002.SZ", "000001.SZ"]);
});

test("终选超过行业上限时触发重试，重试合规后通过（P6）", async () => {
  const c1 = makeCandidate("000001.SZ", 1, { industry: "银行" });
  const c2 = makeCandidate("000002.SZ", 2, { industry: "银行" });
  const c3 = makeCandidate("000003.SZ", 3, { industry: "银行" });
  const c4 = makeCandidate("000004.SZ", 4, { industry: "医药" });
  const request = makeRequest({
    candidates: [c1, c2, c3, c4],
    limits: { candidates: 4, final: 3, max_per_industry: 2, concurrency: 2, call_timeout_seconds: 120 },
  });
  let attempt = 0;
  const script: Script = (role, payload) => {
    if (role === "final_pick") {
      attempt += 1;
      if (attempt === 1) {
        return JSON.stringify({
          picks: [
            { ts_code: "000001.SZ", rank: 1, reason: "a" },
            { ts_code: "000002.SZ", rank: 2, reason: "b" },
            { ts_code: "000003.SZ", rank: 3, reason: "c" },
          ],
          reason: "全选银行",
        });
      }
      return JSON.stringify({
        picks: [
          { ts_code: "000001.SZ", rank: 1, reason: "a" },
          { ts_code: "000002.SZ", rank: 2, reason: "b" },
          { ts_code: "000004.SZ", rank: 3, reason: "d" },
        ],
        reason: "分散行业",
      });
    }
    return happyScript(3)(role, payload, 0);
  };
  const { runner, provider, events } = build(request, script);
  const finalized = await runner.finalizeWith([
    makeVerdict("000001.SZ", 90),
    makeVerdict("000002.SZ", 85),
    makeVerdict("000003.SZ", 80),
    makeVerdict("000004.SZ", 75),
  ]);

  assert.equal(provider.count("final_pick"), 2);
  assert.equal(finalized.usage.failed_calls, 1);
  assert.match(String(events.of("role.failed")[0]?.error), /超过上限/);
  assert.deepEqual(finalized.picks.map((p) => p.ts_code), ["000001.SZ", "000002.SZ", "000004.SZ"]);
});

test("行业太集中时按可容纳上限收缩终选名额", async () => {
  assert.equal(expectedFinalCount(6, ["银行", "银行", "银行", "银行", "电力"], 2), 3);

  const candidates = [
    makeCandidate("000001.SZ", 1, { industry: "银行" }),
    makeCandidate("000002.SZ", 2, { industry: "银行" }),
    makeCandidate("000003.SZ", 3, { industry: "银行" }),
    makeCandidate("000004.SZ", 4, { industry: "电力" }),
  ];
  const request = makeRequest({
    candidates,
    limits: { candidates: 4, final: 4, max_per_industry: 2, concurrency: 2, call_timeout_seconds: 120 },
  });
  const { runner, provider } = build(request, happyScript(3));
  const finalized = await runner.finalizeWith(candidates.map((c, idx) => makeVerdict(c.ts_code, 90 - idx)));

  assert.equal(provider.payloadOf("final_pick").final_count, 3);
  assert.equal(finalized.final_count, 3);
  assert.equal(finalized.picks.length, 3);
});

test("单只股票全有或全无：截断和 4xx 不重试，且失败票完整作废", async () => {
  const script: Script = (role, payload) => {
    const code = (payload.candidate as { ts_code: string } | undefined)?.ts_code ?? "-";
    if (role === "bull" && code === "000001.SZ") {
      return new LlmError("truncated", "max_tokens reached");
    }
    if (role === "risk_chair" && code === "000002.SZ") {
      return new LlmError("request", "400 Bad Request");
    }
    return happyScript(1)(role, payload, 0);
  };
  const { runner, provider, events } = build(makeRequest(), script, 1);
  const debated = await runner.debate();

  assert.equal(provider.count("bull"), 2);
  assert.equal(provider.count("risk_chair"), 1); // 000001 在 bull 失败后直接整只作废
  assert.equal(debated.verdicts.length, 0);
  assert.equal(debated.analysts.length, 0);
  assert.equal(debated.debate.length, 0);
  assert.deepEqual(
    debated.failed_stocks.map((item) => [item.ts_code, item.role]),
    [
      ["000001.SZ", "bull"],
      ["000002.SZ", "risk_chair"],
    ],
  );
  assert.equal(events.of("role.failed").every((e) => e.retrying === false), true);
  assert.equal(events.of("stock.completed").length, 0);
  await assert.rejects(runner.finalizeWith(debated.verdicts), /没有可终选/);
});

test("并发数不超过设定上限，且过高/过低会被夹到安全区间", async () => {
  let active = 0;
  let peak = 0;
  const candidates = [1, 2, 3, 4, 5].map((n) => makeCandidate(`00000${n}.SZ`, n));
  const request = makeRequest({
    candidates,
    limits: { candidates: 5, final: 2, max_per_industry: 2, concurrency: 2, call_timeout_seconds: 120 },
  });
  const slowScript: Script = async (role, payload) => {
    active += 1;
    peak = Math.max(peak, active);
    await new Promise((resolve) => setTimeout(resolve, 15));
    active -= 1;
    return happyScript(2)(role, payload, 0) as string;
  };
  const { runner } = build(request, slowScript, 2);
  await runner.debate();
  assert.equal(peak <= 2, true, `并发峰值为 ${peak}，超过 2`);

  const high = build(makeRequest(), happyScript(1), 999);
  await high.runner.debate();
  assert.equal(high.events.of("run.started")[0]?.concurrency, 16);

  const low = build(makeRequest(), happyScript(1), 0);
  await low.runner.debate();
  assert.equal(low.events.of("run.started")[0]?.concurrency, 4);
});

test("归一化保留对竖线写法的兼容并拒绝双方向真歧义", async () => {
  const script: Script = (role, payload) => {
    const code = (payload.candidate as { ts_code: string } | undefined)?.ts_code ?? "-";
    if (role === "risk_chair" && code === "000001.SZ") {
      return JSON.stringify({
        verdict: "看多|看空",
        score: 72,
        thesis: "结构较强",
        risks: [],
        action: "观察",
        citations: ["features.r5"],
      });
    }
    if (role === "risk_chair" && code === "000002.SZ") {
      return JSON.stringify({
        verdict: "看多还是看空",
        score: 50,
        thesis: "摇摆",
        risks: [],
        action: "观望",
        citations: [],
      });
    }
    return happyScript(1)(role, payload, 0);
  };
  const { runner, provider } = build(makeRequest(), script);
  const debated = await runner.debate();

  assert.equal(provider.count("risk_chair"), 4); // 1 次成功 + 1 只重试 3 次失败
  assert.equal(debated.verdicts.length, 1);
  assert.equal(debated.verdicts[0]?.verdict, "看多");
  assert.equal(debated.failed_stocks[0]?.ts_code, "000002.SZ");
});

test("续跑传入完整 roster 时，风控主席看到完整横向简表而非仅剩的候选", async () => {
  const fullRoster = [
    {
      ts_code: "000001.SZ",
      name: "股票000001.SZ",
      industry: "行业1",
      rank: 1,
      pct: 100,
      r5: -5,
      r20: -12,
      from_low60: 3,
      dev_ma20: -8,
      vol5: 1.2,
      flags: [],
    },
    {
      ts_code: "000002.SZ",
      name: "股票000002.SZ",
      industry: "行业2",
      rank: 2,
      pct: 99,
      r5: -3,
      r20: -10,
      from_low60: 4,
      dev_ma20: -6,
      vol5: 1.1,
      flags: [],
    },
  ];
  const request = makeRequest({
    candidates: [makeCandidate("000002.SZ", 2)],
    limits: { candidates: 2, final: 1, max_per_industry: 2, concurrency: 1, call_timeout_seconds: 90 },
    roster: fullRoster,
  });
  const { runner, provider } = build(request, happyScript(1), 1);
  await runner.debate();

  const riskPayload = provider.seen.find((entry) => entry.role === "risk_chair")?.payload as {
    roster?: Array<{ ts_code: string }>;
  };
  assert.deepEqual(
    riskPayload?.roster?.map((r) => r.ts_code),
    ["000001.SZ", "000002.SZ"],
  );
});

test("流式思考与正文增量通过 role.delta 事件向外推送", async () => {
  const request = makeRequest({
    candidates: [makeCandidate("000001.SZ", 1)],
    limits: { candidates: 1, final: 1, max_per_industry: 2, concurrency: 1, call_timeout_seconds: 90 },
  });
  const events = collector();
  const baseScript = happyScript(1);
  const streamingProvider = {
    async complete(
      systemPrompt: string,
      userPayload: string,
      options?: { onDelta?: (channel: "thinking" | "text", delta: string) => void },
    ) {
      const role = roleFromPrompt(systemPrompt);
      options?.onDelta?.("thinking", "思考片段A");
      options?.onDelta?.("thinking", "思考片段B");
      const answer = baseScript(role, JSON.parse(userPayload) as Record<string, unknown>, 0);
      if (typeof answer !== "string") throw answer;
      options?.onDelta?.("text", answer.slice(0, 12));
      options?.onDelta?.("text", answer.slice(12));
      return { text: answer, usage: { input: 10, output: 5 }, stopReason: "stop" };
    },
  };
  const runner = new DebateRunner(request, {
    provider: streamingProvider,
    emit: events.emit,
    sleep: async () => {},
    concurrency: 1,
  });
  await runner.debate();

  const deltas = events.events.filter((e) => e.type === "role.delta");
  const thinkingDeltas = deltas.filter((e) => e.role === "model_analyst" && e.channel === "thinking");
  const textDeltas = deltas.filter((e) => e.role === "model_analyst" && e.channel === "text");
  assert.equal(thinkingDeltas.map((e) => e.delta).join(""), "思考片段A思考片段B");
  assert.ok(textDeltas.map((e) => e.delta).join("").includes('"stance"'));
});
