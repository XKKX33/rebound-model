/**
 * 提示词组装：把「方法论正文 + 角色职责 + 格式约束 + 禁令」拼成 system prompt。
 *
 * 方法论正文和 7 个角色职责的唯一来源是 `docs/debate-methodology.md`；
 * 输出 JSON 的形状、硬性立场/名额约束、长度限制和通用禁令统一来自
 * `rebound/debate/prompt_rules.json`，与 Python Web UI 共享同一份规则（零漂移）。
 * 同一运行中同一角色的 system prompt 逐字不变，所有候选数据只走 user 消息，便于上游缓存。
 */

import { readFileSync } from "node:fs";

import { AGENT_ROLES, ANALYST_ROLES, type Methodology } from "./contracts.ts";

interface RoleConstraintRule {
  title: string;
  body: string;
}

interface PromptRulesConfig {
  output_schema: Record<string, string>;
  length_rules: Record<string, string>;
  role_constraints: {
    analyst: RoleConstraintRule;
    risk_chair: RoleConstraintRule;
    final_pick: RoleConstraintRule;
  };
  common_rules: string[];
}

const PROMPT_RULES_URL = new URL("../../rebound/debate/prompt_rules.json", import.meta.url);
const PROMPT_RULES: PromptRulesConfig = JSON.parse(
  readFileSync(PROMPT_RULES_URL, "utf8"),
) as PromptRulesConfig;

const OUTPUT_SCHEMA: Record<string, string> = PROMPT_RULES.output_schema;
const LENGTH_RULES: Record<string, string> = PROMPT_RULES.length_rules;
const ROLE_CONSTRAINTS = PROMPT_RULES.role_constraints;
const COMMON_RULES: readonly string[] = PROMPT_RULES.common_rules;

const ANALYST_SET = new Set<string>(ANALYST_ROLES);

/**
 * 拼出某个角色的 system prompt。
 *
 * 顺序：方法论正文 → 角色职责 → 立场或名额块 → 输出格式 → 长度约束 → 禁令。
 */
export function buildSystemPrompt(role: string, methodology: Methodology): string {
  const brief = methodology.role_briefs[role];
  if (!brief) throw new Error(`方法论里缺少角色职责: ${role}`);
  const schema = OUTPUT_SCHEMA[role];
  if (!schema) throw new Error(`未定义输出格式的角色: ${role}`);

  const parts: string[] = [methodology.text.trim(), `## 你的职责\n${brief.trim()}`];

  if (ANALYST_SET.has(role)) {
    const c = ROLE_CONSTRAINTS.analyst;
    parts.push(`## ${c.title}\n${c.body}`);
  }
  if (role === "risk_chair") {
    const c = ROLE_CONSTRAINTS.risk_chair;
    parts.push(`## ${c.title}\n${c.body}`);
  }
  if (role === "final_pick") {
    const c = ROLE_CONSTRAINTS.final_pick;
    parts.push(`## ${c.title}\n${c.body}`);
  }

  parts.push(`## 输出格式\n${schema}`);
  const lengthRule = LENGTH_RULES[role];
  if (!lengthRule) throw new Error(`未定义长度约束的角色: ${role}`);
  parts.push(`## 长度\n${lengthRule}`);
  parts.push(`## 禁令\n${COMMON_RULES.map((rule) => `- ${rule}`).join("\n")}`);

  return parts.join("\n\n");
}

/** 方法论里必须给 7 个角色都写职责，否则跑起来才发现就晚了。 */
export function missingRoleBriefs(methodology: Methodology): string[] {
  return AGENT_ROLES.filter((role) => !methodology.role_briefs[role]?.trim());
}
