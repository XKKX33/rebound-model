/**
 * 辩论编排：7 个角色怎么排、失败怎么重试、并发怎么限、两种模式怎么拆。
 *
 * 每只候选的个股辩论（`--debate`）：
 *   model_analyst / chart_analyst（互相看不见）
 *     → bull（看两位分析师）
 *     → bear（看两位分析师 + bull）
 *     → bull_counter（看两位分析师 + bull + bear）
 *     → risk_chair（看以上全部 + 全候选简表 roster）
 *
 * 整批终选（`--finalize`）：
 *   final_pick（收全部辩完的风控结论，按行业上限约束选出期望名额）。
 */

import {
  ANALYST_ROLES,
  DEBATE_ROUNDS,
  FINAL_PICK_ROLE,
  STANCES,
  VERDICTS,
  type AnalystView,
  type Candidate,
  type DebateEntry,
  type DebateResult,
  type FailedStock,
  type FinalDecision,
  type FinalPick,
  type FinalizedResult,
  type RunRequest,
  type Verdict,
} from "./contracts.ts";
import { buildSystemPrompt } from "./prompts.ts";
import {
  RAW_TEXT,
  LlmError,
  maxAttemptsFor,
  parseJsonEnvelope,
  retryDelayMs,
  type CompletionClient,
  type FailureKind,
  type Usage as TokenUsage,
} from "./provider.ts";

export type EventSink = (event: Record<string, unknown>) => void;

export interface DebateDeps {
  provider: CompletionClient;
  emit: EventSink;
  concurrency?: number;
  /** 测试用：把退避睡眠换成立即返回。 */
  sleep?: (ms: number) => Promise<void>;
  signal?: AbortSignal;
}

const DEFAULT_CONCURRENCY = 4;
const MAX_CONCURRENCY = 16;

function resolveConcurrency(raw: number | undefined): number {
  if (raw === undefined || !Number.isFinite(raw) || raw < 1) return DEFAULT_CONCURRENCY;
  return Math.min(Math.floor(raw), MAX_CONCURRENCY);
}

/** 有界并发 map：worker 抢下标，结果按下标回填以保证输出顺序稳定。 */
async function mapWithConcurrency<T, R>(
  items: readonly T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<Array<{ ok: true; value: R } | { ok: false; error: unknown }>> {
  const results = new Array<{ ok: true; value: R } | { ok: false; error: unknown }>(items.length);
  let cursor = 0;
  const worker = async (): Promise<void> => {
    for (;;) {
      const index = cursor++;
      if (index >= items.length) return;
      const item = items[index];
      if (item === undefined) return;
      try {
        results[index] = { ok: true, value: await fn(item, index) };
      } catch (error) {
        results[index] = { ok: false, error };
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, () => worker()));
  return results;
}

function stringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((entry): entry is string => typeof entry === "string" && entry.trim() !== "");
}

function requiredText(value: unknown, what: string): string {
  if (typeof value !== "string" || value.trim() === "") throw new Error(`${what} 为空`);
  return value.trim();
}

/** 把模型输出截短放进错误消息，防止长文本污染日志。 */
function clip(value: string, limit = 40): string {
  const trimmed = value.trim();
  if (trimmed === "") return "(空)";
  return trimmed.length <= limit ? trimmed : `${trimmed.slice(0, limit)}…`;
}

/** 校验失败也按 empty 归类重试，模型偶尔漏字段重采样一次即可恢复。 */
class ValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ValidationError";
  }
}

/** 把模型给的 `stance` 归一成 `bull` / `bear` / `neutral`，歧义返回 `null`。 */
function normalizeStance(raw: unknown): (typeof STANCES)[number] | null {
  if (typeof raw !== "string") return null;
  const text = raw.trim().replace(/^["'「『]|["'」』]$/g, "").trim();
  if (!text) return null;

  const neutral = /neutral|中性|观望|无法判断/i.test(text);
  const bullish = /bull|看多|做多|long/i.test(text);
  const bearish = /bear|看空|做空|short/i.test(text);

  if (neutral) return "neutral";
  if (bullish === bearish) return null;
  return bullish ? "bull" : "bear";
}

function validateAnalyst(parsed: Record<string, unknown>, role: string): AnalystView {
  const raw = typeof parsed.stance === "string" ? parsed.stance.trim() : "";
  const stance = normalizeStance(parsed.stance);
  if (stance === null) {
    throw new ValidationError(`stance 非法: ${clip(raw)}`);
  }
  return {
    ts_code: "",
    role,
    stance,
    conclusion: requiredText(parsed.conclusion, "conclusion"),
    risks: stringList(parsed.risks),
    citations: stringList(parsed.citations),
  };
}

function validateDebater(parsed: Record<string, unknown>, role: string): { argument: string; citations: string[] } {
  const argument = typeof parsed.argument === "string" ? parsed.argument.trim() : "";
  if (argument) return { argument, citations: stringList(parsed.citations) };
  const raw = typeof parsed[RAW_TEXT] === "string" ? (parsed[RAW_TEXT] as string).trim() : "";
  if (raw) return { argument: raw, citations: [] };
  throw new ValidationError(`${role} 没有给出任何论点`);
}

/** 把模型给的 `verdict` 归一成 `看多` / `看空`，方向不唯一时拒绝猜测。 */
function normalizeVerdict(raw: unknown): (typeof VERDICTS)[number] | null {
  if (typeof raw !== "string") return null;
  const text = raw.trim().replace(/^["'「『]|["'」』]$/g, "").trim();
  if (!text) return null;

  const piped = /^(看多|看空)\s*[|｜]\s*(看多|看空)$/.exec(text);
  if (piped) return piped[1] as (typeof VERDICTS)[number];

  const bullish = /看多|做多|bull|long/i.test(text);
  const bearish = /看空|做空|bear|short/i.test(text);
  if (bullish === bearish) return null;
  return bullish ? "看多" : "看空";
}

function validateRiskChair(parsed: Record<string, unknown>): Omit<Verdict, "ts_code"> {
  const verdict = normalizeVerdict(parsed.verdict);
  if (verdict === null) {
    const raw = typeof parsed.verdict === "string" ? parsed.verdict.trim() : "";
    throw new ValidationError(`verdict 必须是看多或看空，收到: ${clip(raw)}`);
  }
  const score = parsed.score;
  if (typeof score !== "number" || !Number.isFinite(score) || score < 0 || score > 100) {
    throw new ValidationError(`score 必须是 0~100 的数字，收到: ${String(score)}`);
  }
  return {
    verdict,
    score,
    thesis: requiredText(parsed.thesis, "thesis"),
    risks: stringList(parsed.risks),
    action: requiredText(parsed.action, "action"),
    citations: stringList(parsed.citations),
  };
}

/** 按行业可容纳数量收缩后的期望终选名额（第 3.6 节）。 */
export function expectedFinalCount(
  targetFinal: number,
  industries: readonly string[],
  maxPerIndustry: number,
): number {
  const counts = new Map<string, number>();
  for (const industry of industries) {
    counts.set(industry, (counts.get(industry) ?? 0) + 1);
  }
  let feasible = 0;
  for (const count of counts.values()) {
    feasible += Math.min(count, maxPerIndustry);
  }
  return Math.min(targetFinal, industries.length, feasible);
}

function validateFinalPick(
  parsed: Record<string, unknown>,
  allowed: Set<string>,
  expected: number,
  industryByCode: ReadonlyMap<string, string>,
  maxPerIndustry: number,
): FinalDecision {
  const raw = parsed.picks;
  if (!Array.isArray(raw)) throw new ValidationError("picks 不是数组");
  if (raw.length !== expected) {
    throw new ValidationError(`picks 必须恰好 ${expected} 条，收到 ${raw.length} 条`);
  }
  const industryCounts = new Map<string, number>();
  const picks: FinalPick[] = raw.map((entry, index) => {
    if (typeof entry !== "object" || entry === null) {
      throw new ValidationError(`picks[${index}] 不是对象`);
    }
    const item = entry as Record<string, unknown>;
    const tsCode = requiredText(item.ts_code, `picks[${index}].ts_code`);
    if (!allowed.has(tsCode)) {
      throw new ValidationError(`picks[${index}].ts_code 不在候选池里: ${tsCode}`);
    }
    const industry = industryByCode.get(tsCode);
    if (industry === undefined) {
      throw new ValidationError(`picks[${index}].ts_code 缺少行业信息: ${tsCode}`);
    }
    const nextCount = (industryCounts.get(industry) ?? 0) + 1;
    if (nextCount > maxPerIndustry) {
      throw new ValidationError(`行业「${industry}」入选超过上限 ${maxPerIndustry} 只`);
    }
    industryCounts.set(industry, nextCount);
    return {
      ts_code: tsCode,
      rank: index + 1,
      reason: requiredText(item.reason, `picks[${index}].reason`),
    };
  });
  const codes = new Set(picks.map((pick) => pick.ts_code));
  if (codes.size !== picks.length) throw new ValidationError("picks 出现重复代码");
  return { picks, reason: requiredText(parsed.reason, "reason") };
}

interface CallOutcome<T> {
  value: T;
  usage: TokenUsage;
}

class RoleFailure extends Error {
  readonly role: string;
  readonly inner: unknown;

  constructor(role: string, inner: unknown) {
    super(inner instanceof Error ? inner.message : String(inner));
    this.name = "RoleFailure";
    this.role = role;
    this.inner = inner;
  }
}

export class DebateRunner {
  private readonly request: RunRequest;
  private readonly provider: CompletionClient;
  private readonly emit: EventSink;
  private readonly sleep: (ms: number) => Promise<void>;
  private readonly concurrency: number;
  private readonly signal: AbortSignal | undefined;

  private readonly systemPrompts = new Map<string, string>();
  private readonly roster: Array<Record<string, unknown>>;
  private usage: TokenUsage = { input: 0, output: 0 };
  private calls = 0;
  private failedCalls = 0;

  constructor(request: RunRequest, deps: DebateDeps) {
    this.request = request;
    this.provider = deps.provider;
    this.emit = deps.emit;
    this.concurrency = resolveConcurrency(deps.concurrency ?? request.limits.concurrency);
    this.sleep = deps.sleep ?? ((ms) => new Promise((resolve) => setTimeout(resolve, ms)));
    this.signal = deps.signal;
    for (const role of [...ANALYST_ROLES, ...DEBATE_ROUNDS, FINAL_PICK_ROLE]) {
      this.systemPrompts.set(role, buildSystemPrompt(role, request.methodology));
    }
    // 全部送辩候选的简表，供风控主席做横向比较（P5）；断点续跑时优先沿用上游传入的完整 roster。
    this.roster =
      request.roster && request.roster.length > 0
        ? request.roster.map((item) => ({ ...item }))
        : request.candidates.map((candidate) => ({
            ts_code: candidate.ts_code,
            name: candidate.name,
            industry: candidate.industry,
            rank: candidate.model.rank,
            pct: candidate.model.pct,
            r5: candidate.features.r5 ?? null,
            r20: candidate.features.r20 ?? null,
            from_low60: candidate.features.from_low60 ?? null,
            dev_ma20: candidate.features.dev_ma20 ?? null,
            vol5: candidate.features.vol5 ?? null,
            flags: candidate.flags.map((flag) => flag.code),
          }));
  }

  private async call<T>(
    tsCode: string,
    role: string,
    payload: Record<string, unknown>,
    validate: (parsed: Record<string, unknown>) => T,
  ): Promise<CallOutcome<T>> {
    const systemPrompt = this.systemPrompts.get(role);
    if (systemPrompt === undefined) throw new Error(`未注册的角色: ${role}`);
    const userPayload = JSON.stringify(payload);
    const timeoutMs = this.request.limits.call_timeout_seconds * 1000;
    let attempt = 1;
    for (;;) {
      this.emit({ type: "role.started", ts_code: tsCode, role, attempt });
      let pendingChannel: "thinking" | "text" | null = null;
      let pendingText = "";
      let flushTimer: ReturnType<typeof setTimeout> | null = null;
      const flushDelta = (): void => {
        if (flushTimer !== null) {
          clearTimeout(flushTimer);
          flushTimer = null;
        }
        if (pendingChannel !== null && pendingText !== "") {
          this.emit({
            type: "role.delta",
            ts_code: tsCode,
            role,
            attempt,
            channel: pendingChannel,
            delta: pendingText,
          });
          pendingText = "";
        }
      };
      const onDelta = (channel: "thinking" | "text", delta: string): void => {
        if (!delta) return;
        if (pendingChannel !== null && pendingChannel !== channel) {
          flushDelta();
        }
        pendingChannel = channel;
        pendingText += delta;
        if (pendingText.length >= 20) {
          flushDelta();
        } else if (flushTimer === null) {
          flushTimer = setTimeout(flushDelta, 45);
        }
      };
      try {
        const completion = await this.provider.complete(systemPrompt, userPayload, {
          timeoutMs,
          ...(this.signal ? { signal: this.signal } : {}),
          onDelta,
        });
        flushDelta();
        this.calls += 1;
        this.usage = {
          input: this.usage.input + completion.usage.input,
          output: this.usage.output + completion.usage.output,
        };
        this.emit({ type: "usage.updated", usage: this.usageSnapshot() });
        const parsed = parseJsonEnvelope(completion.text);
        const value = validate({ ...(parsed ?? {}), [RAW_TEXT]: completion.text });
        this.emit({ type: "role.completed", ts_code: tsCode, role, attempt, data: value });
        return { value, usage: completion.usage };
      } catch (error) {
        flushDelta();
        const classified = classify(error);
        this.failedCalls += 1;
        this.emit({ type: "usage.updated", usage: this.usageSnapshot() });
        const limit = maxAttemptsFor(classified.kind);
        const willRetry = attempt < limit;
        this.emit({
          type: "role.failed",
          ts_code: tsCode,
          role,
          attempt,
          kind: classified.kind,
          error: classified.message,
          retrying: willRetry,
        });
        if (!willRetry) throw new RoleFailure(role, error);
        await this.sleep(retryDelayMs(attempt));
        attempt += 1;
      }
    }
  }

  /** 单只候选股的 6 轮辩论，全有或全无。 */
  private async debateOne(candidate: Candidate, index: number, total: number): Promise<{
    verdict: Verdict;
    analysts: AnalystView[];
    debate: DebateEntry[];
  }> {
    const tsCode = candidate.ts_code;
    this.emit({ type: "stock.started", ts_code: tsCode, index: index + 1, total });

    const analysts: AnalystView[] = [];
    for (const role of ANALYST_ROLES) {
      const outcome = await this.call(
        tsCode,
        role,
        { context: this.request.context, candidate },
        (parsed) => validateAnalyst(parsed, role),
      );
      analysts.push({ ...outcome.value, ts_code: tsCode });
    }

    const debate: DebateEntry[] = [];
    const base = { context: this.request.context, candidate, analyst_views: analysts };

    const bullOutcome = await this.call(tsCode, "bull", base, (parsed) => validateDebater(parsed, "bull"));
    const bullEntry: DebateEntry = { ts_code: tsCode, round: "bull", ...bullOutcome.value };
    debate.push(bullEntry);

    // P1 修复：空方收到 bull 的发言才能逐条回应。
    const bearOutcome = await this.call(
      tsCode,
      "bear",
      { ...base, debate: [bullEntry] },
      (parsed) => validateDebater(parsed, "bear"),
    );
    const bearEntry: DebateEntry = { ts_code: tsCode, round: "bear", ...bearOutcome.value };
    debate.push(bearEntry);

    const counterOutcome = await this.call(
      tsCode,
      "bull_counter",
      { ...base, debate: [bullEntry, bearEntry] },
      (parsed) => validateDebater(parsed, "bull_counter"),
    );
    const counterEntry: DebateEntry = { ts_code: tsCode, round: "bull_counter", ...counterOutcome.value };
    debate.push(counterEntry);

    // P5 修复：风控主席带上 roster 横向对照全部送辩候选。
    const riskOutcome = await this.call(
      tsCode,
      "risk_chair",
      { ...base, debate: [bullEntry, bearEntry, counterEntry], roster: this.roster },
      validateRiskChair,
    );
    const verdict: Verdict = { ts_code: tsCode, ...riskOutcome.value };
    this.emit({
      type: "stock.completed",
      ts_code: tsCode,
      verdict: verdict.verdict,
      score: verdict.score,
      usage: this.usageSnapshot(),
      stock: { verdict, analysts, debate },
    });
    return { verdict, analysts, debate };
  }

  /** `--debate` 模式：跑完下发候选的 6 个个股角色，不做终选。 */
  async debate(): Promise<DebateResult> {
    const candidates = this.request.candidates;
    const failed: FailedStock[] = [];
    this.emit({
      type: "run.started",
      run_id: this.request.run_id,
      candidates: candidates.length,
      concurrency: this.concurrency,
    });

    const outcomes = await mapWithConcurrency(candidates, this.concurrency, async (candidate, index) =>
      this.debateOne(candidate, index, candidates.length),
    );

    const verdicts: Verdict[] = [];
    const analysts: AnalystView[] = [];
    const debate: DebateEntry[] = [];
    for (let index = 0; index < outcomes.length; index += 1) {
      const outcome = outcomes[index];
      const candidate = candidates[index];
      if (outcome === undefined || candidate === undefined) continue;
      if (outcome.ok) {
        verdicts.push(outcome.value.verdict);
        analysts.push(...outcome.value.analysts);
        debate.push(...outcome.value.debate);
      } else {
        const classified = classify(outcome.error);
        failed.push({
          ts_code: candidate.ts_code,
          role: classified.role ?? "unknown",
          error: classified.message,
        });
        this.emit({ type: "stock.failed", ts_code: candidate.ts_code, error: classified.message });
      }
    }

    if (failed.length > 0) {
      this.emit({ type: "run.partial", failed: failed.length, debated: verdicts.length });
    }
    this.emit({ type: "run.completed", usage: this.usageSnapshot() });

    return {
      run_id: this.request.run_id,
      candidate_hash: this.request.candidate_hash,
      input_hash: this.request.input_hash,
      candidate_codes: candidates.map((candidate) => candidate.ts_code),
      analysts,
      debate,
      verdicts,
      failed_stocks: failed,
      usage: this.usageSnapshot(),
    };
  }

  /** `--finalize` 模式：基于全部辩完的风控结论，执行唯一一次终选（含行业上限校验）。 */
  async finalizeWith(verdicts: readonly Verdict[]): Promise<FinalizedResult> {
    if (verdicts.length === 0) throw new Error("没有可终选的股票结论");
    const byCode = new Map(this.request.candidates.map((candidate) => [candidate.ts_code, candidate]));
    const rows = verdicts
      .map((verdict) => {
        const candidate = byCode.get(verdict.ts_code);
        if (candidate === undefined) {
          throw new Error(`终选输入包含请求以外的代码: ${verdict.ts_code}`);
        }
        return {
          ts_code: candidate.ts_code,
          name: candidate.name,
          industry: candidate.industry,
          model_rank: candidate.model.rank,
          model_pct: candidate.model.pct,
          in_model_picks: candidate.model.in_model_picks,
          verdict: verdict.verdict,
          score: verdict.score,
          thesis: verdict.thesis,
          action: verdict.action,
          risks: verdict.risks,
          flags: candidate.flags.map((flag) => flag.code),
        };
      })
      .sort((left, right) => right.score - left.score || left.model_rank - right.model_rank);

    const allowed = new Set(rows.map((row) => row.ts_code));
    const industryByCode = new Map(rows.map((row) => [row.ts_code, row.industry]));
    const maxPerIndustry = this.request.limits.max_per_industry;
    const expected = expectedFinalCount(
      this.request.limits.final,
      rows.map((row) => row.industry),
      maxPerIndustry,
    );

    const outcome = await this.call(
      "-",
      FINAL_PICK_ROLE,
      {
        context: this.request.context,
        final_count: expected,
        max_per_industry: maxPerIndustry,
        model_picks: this.request.model_picks,
        candidates: rows,
      },
      (parsed) => validateFinalPick(parsed, allowed, expected, industryByCode, maxPerIndustry),
    );
    return {
      final_count: expected,
      picks: outcome.value.picks,
      reason: outcome.value.reason,
      usage: this.usageSnapshot(),
    };
  }

  usageSnapshot(): { calls: number; failed_calls: number; input_tokens: number; output_tokens: number } {
    return {
      calls: this.calls,
      failed_calls: this.failedCalls,
      input_tokens: this.usage.input,
      output_tokens: this.usage.output,
    };
  }
}

interface Classified {
  kind: FailureKind;
  message: string;
  role?: string;
}

function classify(error: unknown): Classified {
  if (error instanceof RoleFailure) {
    const inner = classify(error.inner);
    return { ...inner, role: error.role };
  }
  if (error instanceof LlmError) return { kind: error.kind, message: error.message };
  if (error instanceof ValidationError) return { kind: "empty", message: error.message };
  if (error instanceof Error) return { kind: "transport", message: error.message };
  return { kind: "transport", message: String(error) };
}
