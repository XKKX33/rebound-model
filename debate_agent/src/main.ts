/**
 * 侧车入口：一次运行一个进程，走 stdin/stdout 的 NDJSON。
 *
 * 模式：
 *   --health    确认 Node 版本与入口可执行，不碰网络
 *   --validate  校验一行 RunRequest 协议形状，不碰网络
 *   --debate    对请求里的候选执行 6 个个股角色，末尾输出 type:"debated"
 *   --finalize  读取 {request, verdicts}，执行终选，末尾输出 type:"finalized"
 *
 * stdout 只写协议行，给人看的日志统一走 stderr。
 */

import { createInterface } from "node:readline";

import {
  PROTOCOL_VERSION,
  parseRunRequest,
  parseVerdicts,
  type RunRequest,
} from "./contracts.ts";
import { LlmProvider } from "./provider.ts";
import { DebateRunner } from "./workflow.ts";

/** 密钥只从环境变量读，绝不进协议体。 */
const API_KEY_ENV = "DEBATE_LLM_API_KEY";

function write(event: Record<string, unknown>): void {
  process.stdout.write(`${JSON.stringify(event)}\n`);
}

function log(message: string): void {
  process.stderr.write(`[rebound-debate] ${message}\n`);
}

function describeError(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

function health(): void {
  write({
    type: "health",
    ok: true,
    protocol_version: PROTOCOL_VERSION,
    node: process.version,
    pid: process.pid,
  });
}

async function readRequestLine(): Promise<string> {
  const lines = createInterface({ input: process.stdin, crlfDelay: Number.POSITIVE_INFINITY });
  for await (const line of lines) {
    const trimmed = line.trim();
    if (trimmed === "") continue;
    lines.close();
    return trimmed;
  }
  throw new Error("stdin 关闭前没有收到请求");
}

async function readRequest(): Promise<RunRequest> {
  return parseRunRequest(JSON.parse(await readRequestLine()));
}

async function validate(): Promise<void> {
  const request = await readRequest();
  write({
    type: "valid",
    protocol_version: PROTOCOL_VERSION,
    run_id: request.run_id,
    candidates: request.candidates.length,
    final: request.limits.final,
    max_per_industry: request.limits.max_per_industry,
    concurrency: request.limits.concurrency,
    call_timeout_seconds: request.limits.call_timeout_seconds,
    roles: Object.keys(request.methodology.role_briefs).sort(),
  });
}

function buildRunner(request: RunRequest): DebateRunner {
  const apiKey = process.env[API_KEY_ENV] ?? "";
  const provider = new LlmProvider(request.llm, apiKey, request.limits.call_timeout_seconds * 1000);
  return new DebateRunner(request, {
    provider,
    emit: write,
    concurrency: request.limits.concurrency,
  });
}

async function debate(): Promise<void> {
  const request = await readRequest();
  const runner = buildRunner(request);
  const result = await runner.debate();
  write({ type: "debated", ...result });
}

async function finalize(): Promise<void> {
  const raw: unknown = JSON.parse(await readRequestLine());
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new Error("finalize 输入必须是包含 request 与 verdicts 的 JSON 对象");
  }
  const envelope = raw as Record<string, unknown>;
  const request = parseRunRequest(envelope.request);
  const verdicts = parseVerdicts(envelope.verdicts);
  const runner = buildRunner(request);
  const decision = await runner.finalizeWith(verdicts);
  write({
    type: "finalized",
    final_count: decision.final_count,
    picks: decision.picks,
    reason: decision.reason,
    usage: decision.usage,
  });
}

async function main(): Promise<void> {
  if (process.argv.includes("--health")) {
    health();
    return;
  }
  try {
    if (process.argv.includes("--validate")) {
      await validate();
      return;
    }
    if (process.argv.includes("--debate")) {
      await debate();
      return;
    }
    if (process.argv.includes("--finalize")) {
      await finalize();
      return;
    }
    throw new Error("必须指定模式：--health、--validate、--debate 或 --finalize");
  } catch (error) {
    log(describeError(error));
    write({ type: "error", message: describeError(error) });
    process.exitCode = 1;
  }
}

await main();
