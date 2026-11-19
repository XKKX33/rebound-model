/**
 * provider 兼容层与失败重试策略测试。
 */

import assert from "node:assert/strict";
import http from "node:http";
import { test } from "node:test";

import { buildModel, LlmError, LlmProvider, maxAttemptsFor } from "../src/provider.ts";
import { makeRequest } from "./fixtures.ts";

test("兼容网关强制使用 system 角色并按最大 token 构造", () => {
  const request = makeRequest();
  const model = buildModel(request.llm);
  assert.equal(model.compat?.supportsDeveloperRole, false);
  assert.equal(model.maxTokens, request.llm.max_tokens);
});

test("缺密钥构造 LlmProvider 时立即报错并指向 DEBATE_LLM_API_KEY", () => {
  const request = makeRequest();
  assert.throws(
    () => new LlmProvider(request.llm, "  "),
    /DEBATE_LLM_API_KEY/,
  );
});

test("确定性失败不重试，网络与服务端错误保留重试", () => {
  assert.equal(maxAttemptsFor("request"), 1);
  assert.equal(maxAttemptsFor("truncated"), 1);
  assert.equal(maxAttemptsFor("aborted"), 1);
  assert.equal(maxAttemptsFor("timeout"), 2);
  assert.equal(maxAttemptsFor("empty"), 3);
  assert.equal(maxAttemptsFor("provider"), 3);
  assert.equal(maxAttemptsFor("transport"), 5);
});

test("本地超时中止 SSE 流时归类为 timeout 而非 aborted", async () => {
  const server = http.createServer((req, res) => {
    res.writeHead(200, { "Content-Type": "text/event-stream" });
    res.flushHeaders();
    req.on("close", () => {
      if (!res.writableEnded) res.end();
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("unexpected server address");

  try {
    const provider = new LlmProvider(
      {
        id: "test-model",
        base_url: `http://127.0.0.1:${address.port}/v1`,
        reasoning_effort: "off",
        max_tokens: 100,
        temperature: 0,
      },
      "test-key",
      60,
    );
    await assert.rejects(
      () => provider.complete("sys", "user"),
      (err: unknown) => err instanceof LlmError && err.kind === "timeout",
    );
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});
