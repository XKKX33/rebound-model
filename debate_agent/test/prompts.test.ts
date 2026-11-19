/**
 * 提示词组装测试：结构不变量、无新闻残留、同一角色提示词完全一致。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { AGENT_ROLES } from "../src/contracts.ts";
import { buildSystemPrompt, missingRoleBriefs } from "../src/prompts.ts";
import { makeMethodology, makeRequest } from "./fixtures.ts";

test("每个角色都有输出格式、长度约束与禁令", () => {
  const methodology = makeMethodology();
  for (const role of AGENT_ROLES) {
    const prompt = buildSystemPrompt(role, methodology);
    assert.match(prompt, /## 输出格式/, `${role} 缺输出格式段`);
    assert.match(prompt, /## 长度/, `${role} 缺长度约束段`);
    assert.match(prompt, /## 禁令/, `${role} 缺禁令段`);
  }
});

test("任何角色的 system prompt 都不包含新闻输入说明或新闻字段", () => {
  const methodology = makeMethodology();
  for (const role of AGENT_ROLES) {
    const prompt = buildSystemPrompt(role, methodology);
    assert.doesNotMatch(prompt, /candidate\.news/);
    assert.doesNotMatch(prompt, /context\.news/);
    assert.doesNotMatch(prompt, /akshare/i);
    assert.doesNotMatch(prompt, /sentiment/);
  }
});

test("同一角色对不同候选生成逐字相同的 system prompt，且缺角色直接抛错", () => {
  const request = makeRequest();
  for (const role of AGENT_ROLES) {
    const first = buildSystemPrompt(role, request.methodology);
    const second = buildSystemPrompt(role, request.methodology);
    assert.equal(first, second);
  }
  assert.throws(
    () => buildSystemPrompt("sentiment", makeMethodology()),
    /方法论里缺少角色职责: sentiment/,
  );
  assert.deepEqual(missingRoleBriefs(makeMethodology()), []);
});
