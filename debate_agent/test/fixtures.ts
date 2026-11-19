/**
 * TS 测试夹具：协议 v4 样本构造与假 Provider。
 */

import {
  AGENT_ROLES,
  PROTOCOL_VERSION,
  type Candidate,
  type Methodology,
  type RunRequest,
  type Verdict,
} from "../src/contracts.ts";
import type {
  CompleteOptions,
  CompletionClient,
  CompletionText,
} from "../src/provider.ts";

export function briefOf(role: string): string {
  return `BRIEF[${role}]`;
}

export function roleFromPrompt(systemPrompt: string): string {
  const hit = AGENT_ROLES.find((role) => systemPrompt.includes(briefOf(role)));
  if (hit === undefined) {
    throw new Error(`系统提示词里没有任何角色简报，无法判定角色：${systemPrompt.slice(0, 120)}`);
  }
  return hit;
}

export function makeMethodology(): Methodology {
  return {
    text: "METHODOLOGY-BODY",
    role_briefs: Object.fromEntries(AGENT_ROLES.map((role) => [role, briefOf(role)])),
  };
}

export function makeCandidate(
  tsCode: string,
  rank: number,
  overrides: Partial<Candidate> = {},
): Candidate {
  return {
    ts_code: tsCode,
    name: `票${rank}`,
    industry: rank % 2 === 1 ? "半导体" : "电力",
    model: {
      rank,
      pct: 100 - rank,
      score: Number((0.9 - rank * 0.01).toFixed(4)),
      in_model_picks: rank <= 2,
    },
    features: {
      pct_chg: 1.5,
      r5: -3.2,
      r20: -8.5,
      from_low60: 4.1,
      dev_ma20: -2.4,
      vol5: 1.35,
    },
    feature_pct: {
      pct_chg: 62.0,
      r5: 25.0,
      r20: 18.0,
      from_low60: 30.0,
      dev_ma20: 22.0,
      vol5: 75.0,
    },
    bars: [
      {
        date: "2026-10-13",
        open: 10.0,
        high: 10.3,
        low: 9.8,
        close: 10.1,
        pct_chg: 1.0,
        amount_yi: 1.2,
        vol5: 1.1,
        dif_ratio: -0.5,
        hist_ratio: -0.2,
        limit: null,
      },
      {
        date: "2026-10-14",
        open: 10.1,
        high: 10.4,
        low: 10.0,
        close: 10.25,
        pct_chg: 1.49,
        amount_yi: 1.5,
        vol5: 1.35,
        dif_ratio: -0.3,
        hist_ratio: 0.1,
        limit: null,
      },
    ],
    industry_stats: {
      pool_count: 10,
      top_count: 1,
      rank_in_industry: 1,
      median_r5: -1.5,
      median_r20: -4.2,
    },
    flags: [],
    ...overrides,
  };
}

export function makeRequest(overrides: Partial<RunRequest> = {}): RunRequest {
  const candidates =
    overrides.candidates ?? [makeCandidate("000001.SZ", 1), makeCandidate("000002.SZ", 2)];
  const final = overrides.limits?.final ?? 1;
  const maxPerIndustry = overrides.limits?.max_per_industry ?? 2;
  return {
    protocol_version: PROTOCOL_VERSION,
    run_id: "2026-10-14-nocap-M-abcdef123456",
    as_of: "2026-10-14",
    candidate_hash: "candidate-hash",
    input_hash: "input-hash",
    limits: {
      candidates: candidates.length,
      final,
      max_per_industry: maxPerIndustry,
      concurrency: 2,
      call_timeout_seconds: 120,
      ...overrides.limits,
    },
    methodology: makeMethodology(),
    llm: {
      id: "test-model",
      base_url: "http://127.0.0.1:1",
      reasoning_effort: "low",
      max_tokens: 4096,
      temperature: 0.2,
    },
    context: {
      as_of: "2026-10-14",
      model_label: "反弹模型",
      variant: "nocap",
      cadence: "M",
      fit_date: "2026-10-09",
      model_id: "model-abc",
      scores_stored: true,
      pool_size: 100,
      candidates: candidates.length,
      final,
      max_per_industry: maxPerIndustry,
      pool: {
        median_pct_chg: 0.25,
        median_r5: -1.2,
        median_r20: -3.5,
        up_ratio: 54.0,
      },
    },
    model_picks: candidates.slice(0, final).map((c) => ({
      ts_code: c.ts_code,
      name: c.name,
      industry: c.industry,
      rank: c.model.rank,
      debated: true,
    })),
    candidates,
    ...overrides,
  };
}

export type RoleAnswer = string | Error | Promise<string>;
export type Script = (
  role: string,
  payload: Record<string, unknown>,
  callIndex: number,
  options?: CompleteOptions,
) => RoleAnswer;

export class FakeProvider implements CompletionClient {
  readonly seen: Array<{
    role: string;
    systemPrompt: string;
    payload: Record<string, unknown>;
    options: CompleteOptions | undefined;
  }> = [];
  private readonly script: Script;

  constructor(script: Script) {
    this.script = script;
  }

  count(role: string): number {
    return this.seen.filter((entry) => entry.role === role).length;
  }

  payloadOf(role: string, nth = 0): Record<string, unknown> {
    return this.entryOf(role, nth).payload;
  }

  promptOf(role: string, nth = 0): string {
    return this.entryOf(role, nth).systemPrompt;
  }

  optionsOf(role: string, nth = 0): CompleteOptions | undefined {
    return this.entryOf(role, nth).options;
  }

  private entryOf(role: string, nth: number) {
    const all = this.seen.filter((entry) => entry.role === role);
    const hit = all[nth];
    if (hit === undefined) throw new Error(`${role} 第 ${nth} 次调用不存在（共 ${all.length} 次）`);
    return hit;
  }

  async complete(
    systemPrompt: string,
    userPayload: string,
    options?: CompleteOptions,
  ): Promise<CompletionText> {
    const role = roleFromPrompt(systemPrompt);
    const payload = JSON.parse(userPayload) as Record<string, unknown>;
    const callIndex = this.seen.length;
    this.seen.push({ role, systemPrompt, payload, options });
    const answer = await this.script(role, payload, callIndex, options);
    if (answer instanceof Error) throw answer;
    return { text: answer, usage: { input: 10, output: 5 }, stopReason: "stop" };
  }
}

export function happyScript(finalCount: number): Script {
  return (role, payload) => {
    if (role === "risk_chair") {
      return JSON.stringify({
        verdict: "看多",
        score: 80,
        thesis: "位置与量价匹配",
        risks: ["短线波动"],
        action: "开盘可关注",
        citations: ["features.r20"],
      });
    }
    if (role === "final_pick") {
      const rows = (payload.candidates ?? []) as Array<{ ts_code: string; industry?: string }>;
      const maxPerIndustry = typeof payload.max_per_industry === "number" ? payload.max_per_industry : finalCount;
      const seen = new Map<string, number>();
      const chosen: Array<{ ts_code: string; rank: number; reason: string }> = [];
      for (const row of rows) {
        const ind = row.industry ?? "";
        if ((seen.get(ind) ?? 0) >= maxPerIndustry) continue;
        seen.set(ind, (seen.get(ind) ?? 0) + 1);
        chosen.push({ ts_code: row.ts_code, rank: chosen.length + 1, reason: "分最高" });
        if (chosen.length >= finalCount) break;
      }
      return JSON.stringify({
        picks: chosen,
        reason: "按风控分与行业上限择优",
      });
    }
    if (role === "model_analyst" || role === "chart_analyst") {
      return JSON.stringify({
        stance: "bull",
        conclusion: "结构尚可",
        risks: ["留意高开"],
        citations: ["features.r5"],
      });
    }
    return JSON.stringify({ argument: `${role} 的论点`, citations: ["bars[-1].close"] });
  };
}

export function makeVerdict(tsCode: string, score: number, verdict: "看多" | "看空" = "看多"): Verdict {
  return {
    ts_code: tsCode,
    verdict,
    score,
    thesis: `${tsCode} 判断`,
    risks: ["风险"],
    action: "按计划执行",
    citations: ["features.r20"],
  };
}

export function collector(): {
  events: Array<Record<string, unknown>>;
  emit: (event: Record<string, unknown>) => void;
  of: (type: string) => Array<Record<string, unknown>>;
} {
  const events: Array<Record<string, unknown>> = [];
  return {
    events,
    emit: (event) => {
      events.push(event);
    },
    of: (type) => events.filter((event) => event.type === type),
  };
}
