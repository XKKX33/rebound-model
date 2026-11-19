/**
 * 侧车与 Python 之间的协议契约（协议 v4）。
 *
 * 这个文件是**校验层**，不是文档：Python 下发的每一个请求都必须先过
 * `parseRunRequest`，不合法就整条拒绝，避免跑到一半才发现字段形态不对。
 *
 * 哈希只由 Python 算：侧车把 `candidate_hash` / `input_hash` 当不透明令牌
 * 原样回传，并回传实际辩论过的 `candidate_codes`，避免跨语言浮点序列化差异。
 */

import { createHash } from "node:crypto";

export const PROTOCOL_VERSION = 4;

export const ANALYST_ROLES = ["model_analyst", "chart_analyst"] as const;
export const DEBATE_ROUNDS = ["bull", "bear", "bull_counter", "risk_chair"] as const;
export const FINAL_PICK_ROLE = "final_pick";
export const AGENT_ROLES: readonly string[] = [
  ...ANALYST_ROLES,
  ...DEBATE_ROUNDS,
  FINAL_PICK_ROLE,
];

/** 风控主席只允许两个方向：一份全是"中性"的名单没法比较排序。 */
export const VERDICTS = ["看多", "看空"] as const;
export const STANCES = ["bull", "bear", "neutral"] as const;

export const FLAG_CODES = [
  "limit_up_today",
  "one_word_limit_up",
  "limit_up_streak",
  "limit_down_recent",
  "missing_features",
  "industry_crowded",
] as const;

export type FlagCode = (typeof FLAG_CODES)[number];

export const MAX_TOKENS_LIMIT = 32768;
export const MAX_CANDIDATES = 50;
export const MAX_CALL_TIMEOUT_SECONDS = 10_800;

/** 契约错误：字段路径 + 固定原因，**绝不把模型输出写进消息**（它会被记日志）。 */
export class ContractError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ContractError";
  }
}

export interface Methodology {
  /** 方法论正文（中文）。`docs/debate-methodology.md` 是唯一来源，侧车只负责拼进提示词。 */
  text: string;
  /** 7 个角色各自的职责简报，键必须与 `AGENT_ROLES` 完全一致。 */
  role_briefs: Record<string, string>;
}

export interface LlmSettings {
  id: string;
  base_url: string;
  reasoning_effort: string;
  max_tokens: number;
  temperature: number;
}

export interface RunLimits {
  candidates: number;
  final: number;
  max_per_industry: number;
  concurrency: number;
  call_timeout_seconds: number;
}

export interface PoolSummary {
  median_pct_chg: number;
  median_r5: number;
  median_r20: number;
  up_ratio: number;
}

export interface RunContext {
  as_of: string;
  model_label: string;
  variant: string;
  cadence: string;
  fit_date: string;
  model_id: string;
  scores_stored: boolean;
  pool_size: number;
  candidates: number;
  final: number;
  max_per_industry: number;
  pool: PoolSummary;
}

export interface ModelPickItem {
  ts_code: string;
  name: string;
  industry: string;
  rank: number;
  debated: boolean;
}

export interface CandidateModel {
  rank: number;
  pct: number;
  score: number;
  in_model_picks: boolean;
}

export interface CandidateBar {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  pct_chg: number;
  amount_yi: number;
  vol5?: number;
  dif_ratio?: number;
  hist_ratio?: number;
  limit: "up" | "down" | null;
}

export interface IndustryStats {
  pool_count: number;
  top_count: number;
  rank_in_industry: number;
  median_r5: number | null;
  median_r20: number | null;
}

export interface CandidateFlag {
  code: FlagCode;
  text: string;
}

export interface Candidate {
  ts_code: string;
  name: string;
  industry: string;
  model: CandidateModel;
  features: Record<string, number>;
  feature_pct: Record<string, number>;
  bars: CandidateBar[];
  industry_stats: IndustryStats;
  flags: CandidateFlag[];
}

export interface RosterItem {
  ts_code: string;
  name: string;
  industry: string;
  rank: number;
  pct: number;
  r5: number | null;
  r20: number | null;
  from_low60: number | null;
  dev_ma20: number | null;
  vol5: number | null;
  flags: string[];
}

export interface RunRequest {
  protocol_version: number;
  run_id: string;
  as_of: string;
  candidate_hash: string;
  input_hash: string;
  limits: RunLimits;
  methodology: Methodology;
  llm: LlmSettings;
  context: RunContext;
  model_picks: ModelPickItem[];
  candidates: Candidate[];
  /** 断点续跑时传入全部送辩候选的简表，确保风控主席横向对比口径不变。 */
  roster?: RosterItem[];
}

export interface AnalystView {
  ts_code: string;
  role: string;
  stance: string;
  conclusion: string;
  risks: string[];
  citations: string[];
}

/**
 * 一轮论点。
 *
 * `round` 只会是 `bull` / `bear` / `bull_counter` 三个之一；风控主席那一轮的产物是
 * 结构化裁决 `Verdict`，不在这里重复。
 */
export interface DebateEntry {
  ts_code: string;
  round: string;
  argument: string;
  citations: string[];
}

export interface Verdict {
  ts_code: string;
  verdict: string;
  score: number;
  thesis: string;
  risks: string[];
  action: string;
  citations: string[];
}

export interface FinalPick {
  ts_code: string;
  rank: number;
  reason: string;
}

export interface FinalDecision {
  picks: FinalPick[];
  reason: string;
}

/**
 * 整体用量。
 *
 * `calls` 只数成功的调用，`failed_calls` 单独数失败次数——两者相加才是实际发出去的请求数。
 */
export interface Usage {
  calls: number;
  failed_calls: number;
  input_tokens: number;
  output_tokens: number;
}

export interface FailedStock {
  ts_code: string;
  role: string;
  error: string;
}

export interface DebateResult {
  run_id: string;
  candidate_hash: string;
  input_hash: string;
  candidate_codes: string[];
  analysts: AnalystView[];
  debate: DebateEntry[];
  verdicts: Verdict[];
  failed_stocks: FailedStock[];
  usage: Usage;
}

export interface FinalizedResult {
  final_count: number;
  picks: FinalPick[];
  reason: string;
  usage: Usage;
}

// ---------------------------------------------------------------------------
// 校验工具
// ---------------------------------------------------------------------------

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireText(value: unknown, path: string): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new ContractError(`${path} 不可为空`);
  }
  return value;
}

function requireDateText(value: unknown, path: string): string {
  const text = requireText(value, path);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(text)) {
    throw new ContractError(`${path} 必须是 YYYY-MM-DD，收到 ${text}`);
  }
  return text;
}

function requireBoolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") {
    throw new ContractError(`${path} 必须是布尔值`);
  }
  return value;
}

function requireNumber(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new ContractError(`${path} 必须是有限数字`);
  }
  return value;
}

function requireBoundedNumber(value: unknown, path: string, min: number, max: number): number {
  const parsed = requireNumber(value, path);
  if (parsed < min || parsed > max) {
    throw new ContractError(`${path} 必须在 ${min}~${max} 之间，收到 ${parsed}`);
  }
  return parsed;
}

function requireNullableNumber(value: unknown, path: string): number | null {
  if (value === null) return null;
  return requireNumber(value, path);
}

function requireInt(value: unknown, path: string, min: number, max: number): number {
  const parsed = requireNumber(value, path);
  if (!Number.isInteger(parsed)) throw new ContractError(`${path} 必须是整数`);
  if (parsed < min || parsed > max) {
    throw new ContractError(`${path} 必须在 ${min}~${max} 之间，收到 ${parsed}`);
  }
  return parsed;
}

function requireStringArray(value: unknown, path: string): string[] {
  if (!Array.isArray(value)) throw new ContractError(`${path} 必须是数组`);
  return value.map((item, index) => requireText(item, `${path}[${index}]`));
}

function requireNumberRecord(
  value: unknown,
  path: string,
  bounds?: { min: number; max: number },
): Record<string, number> {
  if (!isRecord(value)) throw new ContractError(`${path} 必须是对象`);
  const out: Record<string, number> = {};
  for (const [key, item] of Object.entries(value)) {
    if (typeof item !== "number" || !Number.isFinite(item)) {
      throw new ContractError(`${path}.${key} 必须是有限数字`);
    }
    if (bounds && (item < bounds.min || item > bounds.max)) {
      throw new ContractError(`${path}.${key} 必须在 ${bounds.min}~${bounds.max} 之间，收到 ${item}`);
    }
    out[key] = item;
  }
  return out;
}

function parseMethodology(raw: unknown): Methodology {
  if (!isRecord(raw)) throw new ContractError("methodology 必须是对象");
  const text = requireText(raw.text, "methodology.text");
  if (!isRecord(raw.role_briefs)) throw new ContractError("methodology.role_briefs 必须是对象");
  const briefs: Record<string, string> = {};
  for (const role of AGENT_ROLES) {
    briefs[role] = requireText(raw.role_briefs[role], `methodology.role_briefs.${role}`);
  }
  for (const key of Object.keys(raw.role_briefs)) {
    if (!AGENT_ROLES.includes(key)) {
      throw new ContractError(`methodology.role_briefs 出现未知角色: ${key}`);
    }
  }
  return { text, role_briefs: briefs };
}

function parseLlm(raw: unknown): LlmSettings {
  if (!isRecord(raw)) throw new ContractError("llm 必须是对象");
  return {
    id: requireText(raw.id, "llm.id"),
    base_url: requireText(raw.base_url, "llm.base_url"),
    reasoning_effort: requireText(raw.reasoning_effort, "llm.reasoning_effort"),
    max_tokens: requireInt(raw.max_tokens, "llm.max_tokens", 1, MAX_TOKENS_LIMIT),
    temperature: requireBoundedNumber(raw.temperature, "llm.temperature", 0, 2),
  };
}

function parseContext(raw: unknown, asOf: string): RunContext {
  if (!isRecord(raw)) throw new ContractError("context 必须是对象");
  const contextAsOf = requireDateText(raw.as_of, "context.as_of");
  if (contextAsOf !== asOf) {
    throw new ContractError(`context.as_of(${contextAsOf}) 与 as_of(${asOf}) 不一致`);
  }
  if (!isRecord(raw.pool)) throw new ContractError("context.pool 必须是对象");
  const candidates = requireInt(raw.candidates, "context.candidates", 1, MAX_CANDIDATES);
  return {
    as_of: contextAsOf,
    model_label: requireText(raw.model_label, "context.model_label"),
    variant: requireText(raw.variant, "context.variant"),
    cadence: requireText(raw.cadence, "context.cadence"),
    fit_date: requireDateText(raw.fit_date, "context.fit_date"),
    model_id: requireText(raw.model_id, "context.model_id"),
    scores_stored: requireBoolean(raw.scores_stored, "context.scores_stored"),
    pool_size: requireInt(raw.pool_size, "context.pool_size", 1, 100_000),
    candidates,
    final: requireInt(raw.final, "context.final", 1, candidates),
    max_per_industry: requireInt(raw.max_per_industry, "context.max_per_industry", 1, MAX_CANDIDATES),
    pool: {
      median_pct_chg: requireNumber(raw.pool.median_pct_chg, "context.pool.median_pct_chg"),
      median_r5: requireNumber(raw.pool.median_r5, "context.pool.median_r5"),
      median_r20: requireNumber(raw.pool.median_r20, "context.pool.median_r20"),
      up_ratio: requireBoundedNumber(raw.pool.up_ratio, "context.pool.up_ratio", 0, 100),
    },
  };
}

function parseModelPicks(raw: unknown): ModelPickItem[] {
  if (!Array.isArray(raw)) throw new ContractError("model_picks 必须是数组");
  return raw.map((entry, index) => {
    const path = `model_picks[${index}]`;
    if (!isRecord(entry)) throw new ContractError(`${path} 必须是对象`);
    return {
      ts_code: requireText(entry.ts_code, `${path}.ts_code`),
      name: requireText(entry.name, `${path}.name`),
      industry: requireText(entry.industry, `${path}.industry`),
      rank: requireInt(entry.rank, `${path}.rank`, 1, 100_000),
      debated: requireBoolean(entry.debated, `${path}.debated`),
    };
  });
}

function parseBar(raw: unknown, path: string): CandidateBar {
  if (!isRecord(raw)) throw new ContractError(`${path} 必须是对象`);
  const limitRaw = raw.limit;
  if (limitRaw !== null && limitRaw !== "up" && limitRaw !== "down") {
    throw new ContractError(`${path}.limit 必须是 "up"、"down" 或 null`);
  }
  const bar: CandidateBar = {
    date: requireDateText(raw.date, `${path}.date`),
    open: requireNumber(raw.open, `${path}.open`),
    high: requireNumber(raw.high, `${path}.high`),
    low: requireNumber(raw.low, `${path}.low`),
    close: requireNumber(raw.close, `${path}.close`),
    pct_chg: requireNumber(raw.pct_chg, `${path}.pct_chg`),
    amount_yi: requireNumber(raw.amount_yi, `${path}.amount_yi`),
    limit: limitRaw,
  };
  if (raw.vol5 !== undefined) bar.vol5 = requireNumber(raw.vol5, `${path}.vol5`);
  if (raw.dif_ratio !== undefined) bar.dif_ratio = requireNumber(raw.dif_ratio, `${path}.dif_ratio`);
  if (raw.hist_ratio !== undefined) bar.hist_ratio = requireNumber(raw.hist_ratio, `${path}.hist_ratio`);
  return bar;
}

function parseIndustryStats(raw: unknown, path: string): IndustryStats {
  if (!isRecord(raw)) throw new ContractError(`${path} 必须是对象`);
  return {
    pool_count: requireInt(raw.pool_count, `${path}.pool_count`, 1, 100_000),
    top_count: requireInt(raw.top_count, `${path}.top_count`, 1, MAX_CANDIDATES),
    rank_in_industry: requireInt(raw.rank_in_industry, `${path}.rank_in_industry`, 1, 100_000),
    median_r5: requireNullableNumber(raw.median_r5, `${path}.median_r5`),
    median_r20: requireNullableNumber(raw.median_r20, `${path}.median_r20`),
  };
}

function parseFlags(raw: unknown, path: string): CandidateFlag[] {
  if (!Array.isArray(raw)) throw new ContractError(`${path} 必须是数组`);
  return raw.map((entry, index) => {
    const itemPath = `${path}[${index}]`;
    if (!isRecord(entry)) throw new ContractError(`${itemPath} 必须是对象`);
    const code = requireText(entry.code, `${itemPath}.code`);
    if (!FLAG_CODES.includes(code as FlagCode)) {
      throw new ContractError(`${itemPath}.code 未知: ${code}`);
    }
    return {
      code: code as FlagCode,
      text: requireText(entry.text, `${itemPath}.text`),
    };
  });
}

function parseCandidate(raw: unknown, index: number, asOf: string): Candidate {
  const path = `candidates[${index}]`;
  if (!isRecord(raw)) throw new ContractError(`${path} 必须是对象`);
  if (!isRecord(raw.model)) throw new ContractError(`${path}.model 必须是对象`);

  const features = requireNumberRecord(raw.features, `${path}.features`);
  if (Object.keys(features).length === 0) {
    throw new ContractError(`${path}.features 不能为空`);
  }
  const featurePct = requireNumberRecord(raw.feature_pct, `${path}.feature_pct`, { min: 0, max: 100 });

  if (!Array.isArray(raw.bars) || raw.bars.length === 0) {
    throw new ContractError(`${path}.bars 必须是非空数组`);
  }
  const bars = raw.bars.map((entry, barIndex) => parseBar(entry, `${path}.bars[${barIndex}]`));
  for (let i = 1; i < bars.length; i += 1) {
    const prev = bars[i - 1];
    const curr = bars[i];
    if (prev !== undefined && curr !== undefined && curr.date <= prev.date) {
      throw new ContractError(`${path}.bars 日期必须严格升序`);
    }
  }
  const lastBar = bars[bars.length - 1];
  if (lastBar === undefined || lastBar.date !== asOf) {
    throw new ContractError(`${path}.bars 最后一根日期(${lastBar?.date ?? "空"})必须等于 as_of(${asOf})`);
  }

  return {
    ts_code: requireText(raw.ts_code, `${path}.ts_code`),
    name: requireText(raw.name, `${path}.name`),
    industry: requireText(raw.industry, `${path}.industry`),
    model: {
      rank: requireInt(raw.model.rank, `${path}.model.rank`, 1, 100_000),
      pct: requireBoundedNumber(raw.model.pct, `${path}.model.pct`, 0, 100),
      score: requireNumber(raw.model.score, `${path}.model.score`),
      in_model_picks: requireBoolean(raw.model.in_model_picks, `${path}.model.in_model_picks`),
    },
    features,
    feature_pct: featurePct,
    bars,
    industry_stats: parseIndustryStats(raw.industry_stats, `${path}.industry_stats`),
    flags: parseFlags(raw.flags, `${path}.flags`),
  };
}

function parseRoster(raw: unknown): RosterItem[] {
  if (!Array.isArray(raw) || raw.length === 0) {
    throw new ContractError("roster 必须是非空数组");
  }
  return raw.map((entry, index) => {
    const path = `roster[${index}]`;
    if (!isRecord(entry)) throw new ContractError(`${path} 必须是对象`);
    return {
      ts_code: requireText(entry.ts_code, `${path}.ts_code`),
      name: requireText(entry.name, `${path}.name`),
      industry: requireText(entry.industry, `${path}.industry`),
      rank: requireInt(entry.rank, `${path}.rank`, 1, 100_000),
      pct: requireBoundedNumber(entry.pct, `${path}.pct`, 0, 100),
      r5: requireNullableNumber(entry.r5 ?? null, `${path}.r5`),
      r20: requireNullableNumber(entry.r20 ?? null, `${path}.r20`),
      from_low60: requireNullableNumber(entry.from_low60 ?? null, `${path}.from_low60`),
      dev_ma20: requireNullableNumber(entry.dev_ma20 ?? null, `${path}.dev_ma20`),
      vol5: requireNullableNumber(entry.vol5 ?? null, `${path}.vol5`),
      flags: requireStringArray(entry.flags, `${path}.flags`),
    };
  });
}

export function parseRunRequest(raw: unknown): RunRequest {
  if (!isRecord(raw)) throw new ContractError("请求体必须是 JSON 对象");
  if (raw.protocol_version !== PROTOCOL_VERSION) {
    throw new ContractError(
      `protocol_version 必须是 ${PROTOCOL_VERSION}，收到 ${String(raw.protocol_version)}`,
    );
  }
  if (!isRecord(raw.limits)) throw new ContractError("limits 必须是对象");
  const limits: RunLimits = {
    candidates: requireInt(raw.limits.candidates, "limits.candidates", 1, MAX_CANDIDATES),
    final: requireInt(raw.limits.final, "limits.final", 1, MAX_CANDIDATES),
    max_per_industry: requireInt(raw.limits.max_per_industry, "limits.max_per_industry", 1, MAX_CANDIDATES),
    concurrency: requireInt(raw.limits.concurrency, "limits.concurrency", 1, MAX_CANDIDATES),
    call_timeout_seconds: requireInt(
      raw.limits.call_timeout_seconds,
      "limits.call_timeout_seconds",
      1,
      MAX_CALL_TIMEOUT_SECONDS,
    ),
  };
  if (limits.final > limits.candidates) {
    throw new ContractError(
      `limits.final(${limits.final}) 不能大于 limits.candidates(${limits.candidates})`,
    );
  }
  const asOf = requireDateText(raw.as_of, "as_of");
  if (!Array.isArray(raw.candidates) || raw.candidates.length === 0) {
    throw new ContractError("candidates 必须是非空数组");
  }
  if (raw.candidates.length > limits.candidates) {
    throw new ContractError(
      `candidates 有 ${raw.candidates.length} 条，超过 limits.candidates=${limits.candidates}`,
    );
  }
  const candidates = raw.candidates.map((item, idx) => parseCandidate(item, idx, asOf));
  const seen = new Set<string>();
  for (const candidate of candidates) {
    if (seen.has(candidate.ts_code)) {
      throw new ContractError(`candidates 出现重复代码: ${candidate.ts_code}`);
    }
    seen.add(candidate.ts_code);
  }

  const roster = raw.roster !== undefined ? parseRoster(raw.roster) : undefined;

  return {
    protocol_version: PROTOCOL_VERSION,
    run_id: requireText(raw.run_id, "run_id"),
    as_of: asOf,
    candidate_hash: requireText(raw.candidate_hash, "candidate_hash"),
    input_hash: requireText(raw.input_hash, "input_hash"),
    limits,
    methodology: parseMethodology(raw.methodology),
    llm: parseLlm(raw.llm),
    context: parseContext(raw.context, asOf),
    model_picks: parseModelPicks(raw.model_picks),
    candidates,
    ...(roster ? { roster } : {}),
  };
}

export function parseVerdicts(raw: unknown): Verdict[] {
  if (!Array.isArray(raw) || raw.length === 0) {
    throw new ContractError("verdicts 必须是非空数组");
  }
  return raw.map((entry, index) => {
    const path = `verdicts[${index}]`;
    if (!isRecord(entry)) throw new ContractError(`${path} 必须是对象`);
    const verdict = requireText(entry.verdict, `${path}.verdict`);
    if (!VERDICTS.includes(verdict as (typeof VERDICTS)[number])) {
      throw new ContractError(`${path}.verdict 必须是看多或看空，收到 ${verdict}`);
    }
    return {
      ts_code: requireText(entry.ts_code, `${path}.ts_code`),
      verdict,
      score: requireBoundedNumber(entry.score, `${path}.score`, 0, 100),
      thesis: requireText(entry.thesis, `${path}.thesis`),
      risks: requireStringArray(entry.risks, `${path}.risks`),
      action: requireText(entry.action, `${path}.action`),
      citations: requireStringArray(entry.citations, `${path}.citations`),
    };
  });
}

// ---------------------------------------------------------------------------
// 规范化 JSON（只用于日志与测试，不参与跨语言浮点比对）
// ---------------------------------------------------------------------------

export function canonicalize(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (isRecord(value)) {
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(value).sort()) out[key] = canonicalize(value[key]);
    return out;
  }
  if (typeof value === "number" && !Number.isFinite(value)) return null;
  if (value === undefined) return null;
  return value;
}

export function stableJson(value: unknown): string {
  return JSON.stringify(canonicalize(value));
}

export function fingerprint(value: unknown): string {
  return createHash("sha256").update(stableJson(value), "utf8").digest("hex");
}
