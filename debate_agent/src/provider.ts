/**
 * 与模型通信的唯一出口。
 *
 * 只做四件事：把配置拼成 Model、发一次请求、把失败分类、把文本交回去。
 */

import { contentText, type Model } from "@earendil-works/pi-ai";
import { streamSimple } from "@earendil-works/pi-ai/api/openai-completions";

import type { LlmSettings } from "./contracts.ts";

/** 一次请求失败的性质。分类决定重试几次、以及错误信息该长什么样。 */
export type FailureKind =
  /** 超过单次调用上限（连接卡死，不是模型慢）。 */
  | "timeout"
  /** 模型把 max_tokens 用完了，JSON 还没闭环。重试必然同样断，所以不重试。 */
  | "truncated"
  /** 主动取消（Python 侧超时或用户中断杀进程）。 */
  | "aborted"
  /** 网络层失败：断流、429、上游 5xx。最值得重试的一类。 */
  | "transport"
  /** 上游返回了临时错误但连接正常：限流、服务端错误。 */
  | "provider"
  /** 上游明确拒绝请求：角色、参数、余额、模型名。重试不会改变结果。 */
  | "request"
  /** 说完了但一个字都没有，或输出校验不通过。 */
  | "empty";

export class LlmError extends Error {
  readonly kind: FailureKind;

  constructor(kind: FailureKind, message: string) {
    super(message);
    this.name = "LlmError";
    this.kind = kind;
  }
}

/** 每类失败最多试几次（含第一次）。 */
const MAX_ATTEMPTS: Record<FailureKind, number> = {
  transport: 5,
  empty: 3,
  provider: 3,
  request: 1,
  timeout: 2,
  truncated: 1,
  aborted: 1,
};

export function maxAttemptsFor(kind: FailureKind): number {
  return MAX_ATTEMPTS[kind];
}

/** 指数退避：5s → 10s → 20s → 40s（上限 40 秒）。 */
export function retryDelayMs(attempt: number): number {
  const base = 5_000;
  return Math.min(base * 2 ** (attempt - 1), 40_000);
}

export interface Usage {
  input: number;
  output: number;
}

export interface CompletionText {
  text: string;
  usage: Usage;
  /** 上游给的结束原因，原样带出来供排障。 */
  stopReason: string;
}

/**
 * 把配置翻译成 pi-ai 的 Model。
 *
 * 兼容开关：`supportsDeveloperRole: false` 强制用标准 `system` 角色；
 * `maxTokensField: "max_tokens"` 避免网关拒绝 `max_completion_tokens`；
 * `supportsUsageInStreaming: true` 才会带上 token 用量。
 */
export function buildModel(settings: LlmSettings): Model<"openai-completions"> {
  return {
    id: settings.id,
    name: settings.id,
    api: "openai-completions",
    provider: "openai-compatible",
    baseUrl: settings.base_url,
    reasoning: true,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 32_768,
    maxTokens: settings.max_tokens,
    compat: {
      supportsDeveloperRole: false,
      supportsReasoningEffort: true,
      maxTokensField: "max_tokens",
      supportsUsageInStreaming: true,
    },
  };
}

type ReasoningLevel = "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

const REASONING_LEVELS = new Set<ReasoningLevel>([
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
]);

function resolveReasoning(raw: string): ReasoningLevel | undefined {
  const value = raw.trim().toLowerCase();
  if (value === "" || value === "off" || value === "none") return undefined;
  if (!REASONING_LEVELS.has(value as ReasoningLevel)) {
    throw new Error(`reasoning_effort 非法: ${raw}（可用 off、minimal、low、medium、high）`);
  }
  return value as ReasoningLevel;
}

export interface CompleteOptions {
  /** 单次调用上限（毫秒），默认取请求里的 `limits.call_timeout_seconds * 1000`。 */
  timeoutMs?: number;
  signal?: AbortSignal;
  /** 可选流式增量回调，用于 Web UI 实时展示思考过程与打字机输出。 */
  onDelta?: (channel: "thinking" | "text", delta: string) => void;
}

/** 模型没套 JSON 外壳时，把原文挂在这个旁路键下由校验层判断是否可用。 */
export const RAW_TEXT = "_text";

/**
 * 从一段模型输出里挖出一个 JSON 对象：剥 markdown 围栏 → 直接 parse → 首尾 `{}` 切片再 parse。
 */
export function parseJsonEnvelope(text: string): Record<string, unknown> | null {
  const trimmed = text
    .trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/, "")
    .trim();
  const direct = tryParseObject(trimmed);
  if (direct) return direct;
  const start = trimmed.indexOf("{");
  const end = trimmed.lastIndexOf("}");
  if (start < 0 || end <= start) return null;
  return tryParseObject(trimmed.slice(start, end + 1));
}

function tryParseObject(text: string): Record<string, unknown> | null {
  try {
    const parsed: unknown = JSON.parse(text);
    if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return null;
  } catch {
    return null;
  }
}

const DEFAULT_TIMEOUT_MS = 300_000;

/** 编排层依赖的结构接口，便于测试注入假 provider。 */
export interface CompletionClient {
  complete(
    systemPrompt: string,
    userPayload: string,
    options?: CompleteOptions,
  ): Promise<CompletionText>;
}

export class LlmProvider implements CompletionClient {
  private readonly model: Model<"openai-completions">;
  private readonly apiKey: string;
  private readonly reasoning: ReasoningLevel | undefined;
  private readonly temperature: number;
  private readonly timeoutMs: number;

  constructor(settings: LlmSettings, apiKey: string, timeoutMs = DEFAULT_TIMEOUT_MS) {
    if (!apiKey.trim()) {
      throw new Error(
        "缺少 LLM API key（环境变量 DEBATE_LLM_API_KEY）。Python 侧从 .env 读取后通过环境变量传进来，不写进协议。",
      );
    }
    this.model = buildModel(settings);
    this.apiKey = apiKey;
    this.reasoning = resolveReasoning(settings.reasoning_effort);
    this.temperature = settings.temperature;
    this.timeoutMs = timeoutMs;
  }

  async complete(
    systemPrompt: string,
    userPayload: string,
    options: CompleteOptions = {},
  ): Promise<CompletionText> {
    const controller = new AbortController();
    const timeoutMs = options.timeoutMs ?? this.timeoutMs;
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const signal = options.signal
      ? AbortSignal.any([controller.signal, options.signal])
      : controller.signal;
    let timedOut = false;
    const onTimeout = () => {
      timedOut = true;
    };
    controller.signal.addEventListener("abort", onTimeout, { once: true });

    try {
      const stream = streamSimple(
        this.model,
        {
          systemPrompt,
          messages: [{ role: "user", content: userPayload, timestamp: Date.now() }],
        },
        {
          apiKey: this.apiKey,
          signal,
          temperature: this.temperature,
          reasoning: this.reasoning,
          maxRetries: 0,
          timeoutMs,
          samplingParams: { response_format: { type: "json_object" } },
        },
      );
      if (options.onDelta) {
        for await (const ev of stream) {
          if (ev.type === "thinking_delta" && ev.delta) {
            options.onDelta("thinking", ev.delta);
          } else if (ev.type === "text_delta" && ev.delta) {
            options.onDelta("text", ev.delta);
          }
        }
      }
      const message = await stream.result();
      if (timedOut) {
        throw new LlmError("timeout", `单次调用超过 ${Math.round(timeoutMs / 1000)} 秒`);
      }
      this.assertUsable(message);
      const text = contentText(message.content).trim();
      const usage: Usage = {
        input: message.usage?.input ?? 0,
        output: message.usage?.output ?? 0,
      };
      if (!text) throw new LlmError("empty", "模型返回了空内容");
      return { text, usage, stopReason: message.stopReason ?? "unknown" };
    } catch (error) {
      if (error instanceof LlmError) throw error;
      if (timedOut) {
        throw new LlmError("timeout", `单次调用超过 ${Math.round(timeoutMs / 1000)} 秒`);
      }
      if (isAbortError(error)) throw new LlmError("aborted", "调用被取消");
      throw classifyRequestError(error);
    } finally {
      clearTimeout(timer);
      controller.signal.removeEventListener("abort", onTimeout);
    }
  }

  private assertUsable(message: { stopReason?: string; errorMessage?: string }): void {
    const reason = message.stopReason ?? "";
    if (reason === "length") {
      throw new LlmError(
        "truncated",
        `模型把 max_tokens(${this.model.maxTokens}) 用完了，JSON 被截断；按被截断的输出解析会得到残缺或错误的结论，所以判失败`,
      );
    }
    if (reason === "aborted") {
      const detail = message.errorMessage?.trim();
      throw new LlmError("aborted", detail ? `上游报告调用被取消：${detail}` : "上游报告调用被取消");
    }
    if (reason === "error") {
      throw new LlmError("provider", message.errorMessage?.trim() || "上游返回 error");
    }
  }
}

function isAbortError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    (error as { name?: unknown }).name === "AbortError"
  );
}

function describe(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

function classifyRequestError(error: unknown): LlmError {
  const message = describe(error);
  const status = errorStatus(error) ?? statusFromMessage(message);
  if (status !== undefined && status >= 400 && status < 500 && status !== 408 && status !== 429) {
    return new LlmError("request", message);
  }
  if (status !== undefined && status >= 500) {
    return new LlmError("provider", message);
  }
  return new LlmError("transport", message);
}

function errorStatus(error: unknown): number | undefined {
  if (typeof error !== "object" || error === null) return undefined;
  if ("status" in error && typeof (error as { status?: unknown }).status === "number") {
    return (error as { status: number }).status;
  }
  if ("statusCode" in error && typeof (error as { statusCode?: unknown }).statusCode === "number") {
    return (error as { statusCode: number }).statusCode;
  }
  return undefined;
}

function statusFromMessage(message: string): number | undefined {
  const match = /\b(?:Error\s+)?([45]\d{2})\b/.exec(message);
  return match?.[1] ? Number(match[1]) : undefined;
}
