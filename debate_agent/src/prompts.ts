/**
 * 提示词组装：把「方法论正文 + 角色职责 + 格式约束 + 禁令」拼成 system prompt。
 *
 * 方法论正文和 7 个角色职责的唯一来源是 `docs/debate-methodology.md`；
 * 本文件只负责把固定的输出 JSON 形状、硬性立场/名额约束、长度限制和通用禁令拼进去。
 * 同一运行中同一角色的 system prompt 逐字不变，所有候选数据只走 user 消息，便于上游缓存。
 */

import { AGENT_ROLES, ANALYST_ROLES, type Methodology } from "./contracts.ts";

const OUTPUT_SCHEMA: Record<string, string> = {
  model_analyst:
    '{"stance":"bull、bear、neutral 三选一（只填一个词）","conclusion":"一句话结论","risks":["风险1","风险2"],"citations":["引用的字段路径"]}',
  chart_analyst:
    '{"stance":"bull、bear、neutral 三选一（只填一个词）","conclusion":"一句话结论","risks":["风险1","风险2"],"citations":["引用的字段路径"]}',
  bull: '{"argument":"看多理由","citations":["引用的字段路径"]}',
  bear: '{"argument":"先逐条回应多方，再给回避理由","citations":["引用的字段路径"]}',
  bull_counter: '{"argument":"对空方论点的逐条回应","citations":["引用的字段路径"]}',
  risk_chair:
    '{"verdict":"看多 或 看空（只填这两个词之一，不要带引号、不要写竖线）",' +
    '"score":0到100的整数,"thesis":"不超过80字的核心判断","risks":["风险1","风险2"],' +
    '"action":"不超过40字的开盘前操作提示","citations":["引用的字段路径"]}',
  final_pick: '{"picks":[{"ts_code":"代码","rank":1,"reason":"为什么选它"}],"reason":"整体取舍理由"}',
};

/** 长度是防 max_tokens 截断的硬约束，每个角色必须定义一条。 */
const LENGTH_RULES: Record<string, string> = {
  model_analyst: "conclusion 不超过 120 字，risks 最多 3 条、每条不超过 40 字，citations 最多 6 条。不要复述数据。",
  chart_analyst: "conclusion 不超过 120 字，risks 最多 3 条、每条不超过 40 字，citations 最多 6 条。只讲走势、量价和 MACD。",
  bull: "argument 必须少于 300 字。直接给理由，不要复述数据，不要写开场白。",
  bear: "argument 必须少于 350 字。先逐条回应多方，再补充理由，不要写开场白。",
  bull_counter: "argument 必须少于 300 字。只回应空方的具体论点，不要另起炉灶。",
  risk_chair: "thesis 不超过 80 字，action 不超过 40 字，risks 最多 3 条。",
  final_pick: "每只票的 reason 不超过 60 字，整体 reason 不超过 200 字。",
};

const ANALYST_SET = new Set<string>(ANALYST_ROLES);

const COMMON_RULES = [
  "你只能使用用户消息里给出的数字和事实。",
  "绝对不要编造新闻、公告、业绩、资金流、研报或任何输入里没有的信息；输入里没有的不要提，需要说明「无法判断」时，直接写缺什么。",
  "数值的单位和口径以方法论里的「字段说明」为准，不要自己推算新指标再当成事实。",
  "引用证据时写字段路径，例如 features.r20、bars[-1].close。",
  "只输出一个 JSON 对象，不要输出 JSON 以外的任何文字，不要用 markdown 代码块包裹。",
];

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
    parts.push(
      "## 立场\n" +
        "stance 只能填 bull、bear、neutral 三个词之一。选 neutral 时，conclusion 必须写清缺的是什么。",
    );
  }
  if (role === "risk_chair") {
    parts.push(
      "## 立场与打分\n" +
        "verdict 只能填 看多 或 看空，不允许中性，不允许弃权。score 必须是 0 到 100 的整数。",
    );
  }
  if (role === "final_pick") {
    parts.push(
      "## 名额\n" +
        "picks 的条数必须恰好等于 final_count；rank 从 1 开始连续编号；ts_code 必须来自 candidates，不许自创、不许重复；同一个 industry 的票不超过 max_per_industry 只。",
    );
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
