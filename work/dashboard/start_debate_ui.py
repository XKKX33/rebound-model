"""Start the member debate UI against the project database and delivered model package."""
from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path
from shutil import copytree

sys.dont_write_bytecode = True

from paths import PROJECT, DASHBOARD, DATABASE, MODEL_REPO, AGORA_REPO
MEMBER_WEB_UI = MODEL_REPO / "rebound/debate/web_ui"

sys.path[:0] = [
    str(PROJECT / "work/runtime"),
    str(DASHBOARD),
    str(AGORA_REPO),
    str(MODEL_REPO),
]


def prepare_embedded_ui(destination: Path) -> Path:
    ui_dir = destination / "web_ui"
    copytree(MEMBER_WEB_UI, ui_dir)
    style_path = ui_dir / "styles.css"
    styles = style_path.read_text(encoding="utf-8")
    for old, new in (
        ("#1e3a8a", "#8d252e"),
        ("#1e40af", "#a83135"),
        ("#1d4ed8", "#a83135"),
        ("#2563eb", "#cf3f42"),
        ("#3b82f6", "#dd8386"),
        ("#60a5fa", "#e8a5a8"),
        ("#93c5fd", "#ebc4c5"),
        ("#bfdbfe", "#ebc4c5"),
        ("#dbeafe", "#fcf1f1"),
        ("#e0e7ff", "#fcf1f1"),
        ("#eff6ff", "#fcf1f1"),
        ("#f8fbff", "#ffffff"),
        ("#15803d", "#707070"),
        ("#166534", "#555555"),
        ("#86efac", "#d9d9d9"),
        ("#dcfce7", "#f1f1f1"),
        ("#b45309", "#555555"),
        ("#fffbeb", "#f1f1f1"),
        ("#fde68a", "#d9d9d9"),
        ("#5b21b6", "#555555"),
        ("#ddd6fe", "#d9d9d9"),
        ("#f5f3ff", "#f1f1f1"),
        ("#f1f5f9", "#f1f1f1"),
        ("#f8fafc", "#fafafa"),
        ("#e2e8f0", "#e5e5e5"),
        ("#cbd5e1", "#d4d4d4"),
        ("#475569", "#5b5b5b"),
        ("#64748b", "#777777"),
        ("#94a3b8", "#999999"),
        ("#334155", "#454545"),
        ("#1e293b", "#303030"),
        ("#0f172a", "#1c1c1c"),
        ("#eef2f7", "#f2f2f2"),
        ("#e5e7eb", "#e5e5e5"),
        ("#d1d5db", "#d4d4d4"),
        ("#9ca3af", "#929292"),
        ("#6b7280", "#666666"),
        ("#374151", "#454545"),
        ("#111827", "#242424"),
        ("#1d2930", "#242424"),
        ("#41515a", "#4d4d4d"),
        ("#63727b", "#666666"),
    ):
        styles = styles.replace(old, new)
    for old, new in (
        ("rgba(37, 99, 235, 0.25)", "rgba(86, 97, 105, 0.12)"),
        ("rgba(37, 99, 235, 0.12)", "rgba(86, 97, 105, 0.12)"),
        ("rgba(30, 58, 138, 0.28)", "rgba(86, 97, 105, 0.12)"),
        ("rgba(37, 99, 235, 0.08)", "rgba(86, 97, 105, 0.08)"),
        ("rgba(59, 130, 246, 0.1)", "rgba(86, 97, 105, 0.08)"),
    ):
        styles = styles.replace(old, new)
    # Dashboard layout is defined once in assets/debate_layout.css.
    style_path.write_text(styles, encoding="utf-8")
    app_path = ui_dir / "app.js"
    app = app_path.read_text(encoding="utf-8")
    app = app.replace("#2563eb", "#cf3f42")
    charts_path = ui_dir / "charts.js"
    charts = charts_path.read_text(encoding="utf-8")
    for old, new in (
        ("#93c5fd", "#e8a5a8"),
        ("#2563eb", "#cf3f42"),
        ("#60a5fa", "#dd8386"),
        ("#1d4ed8", "#a83135"),
        ("#15803d", "#a83135"),
        ("#86efac", "#ebc4c5"),
        ("#dcfce7", "#fcf1f1"),
        ("#eff6ff", "#fcf1f1"),
        ("#d97706", "#696969"),
        ("#7c3aed", "#737373"),
        ("#0284c7", "#5f5f5f"),
        ("#c81e1e", "#cf3f42"),
        ("#dc2626", "#cf3f42"),
        ("#0d9488", "#128365"),
        ("#99f6e4", "#a4d5c5"),
        ("#94a3b8", "#999999"),
        ("#e2e8f0", "#e5e5e5"),
        ("#cbd5e1", "#d4d4d4"),
        ("#6b7280", "#666666"),
        ("#475569", "#5b5b5b"),
        ("#64748b", "#777777"),
        ("#f8fafc", "#fafafa"),
    ):
        charts = charts.replace(old, new)
    for old, new in (
        ('xgb_F69: { color: "#a83135", width: 2.3, dash: "" }',
         'xgb_F69: { color: "#cf3f42", width: 2.3, dash: "" }'),
        ('rf_F69: { color: "#0f766e", width: 2.0, dash: "" }',
         'rf_F69: { color: "#128365", width: 2.0, dash: "6 2" }'),
        ('lstm_SF69: { color: "#b91c1c", width: 2.0, dash: "" }',
         'lstm_SF69: { color: "#8d252e", width: 2.0, dash: "2 2" }'),
        ('pool_nav: { color: "#777777", width: 1.5, dash: "5 3" }',
         'pool_nav: { color: "#7e7e7e", width: 1.5, dash: "5 3" }'),
    ):
        if old not in charts:
            raise RuntimeError(f"无法定位净值图配色：{old}")
        charts = charts.replace(old, new, 1)

    for old, new in (
        ("XGBoost F69", "XGBoost · 69 项特征"),
        ("随机森林 F69", "随机森林 · 69 项特征"),
        ("LSTM S+F69", "LSTM · 69 项特征 + 60×11 序列"),
        ("LSTM SF69", "LSTM · 69 项特征 + 60×11 序列"),
    ):
        app = app.replace(old, new)

    for old, new in (
        ('{ id: "model_analyst", x: 98, y: 82, w: 150, h: 32, title: "模型解读员", sub: "model_analyst" }',
         '{ id: "model_analyst", x: 98, y: 82, w: 150, h: 32, title: "模型解读员", sub: "排名与特征贡献" }'),
        ('{ id: "chart_analyst", x: 282, y: 82, w: 150, h: 32, title: "走势分析员", sub: "chart_analyst" }',
         '{ id: "chart_analyst", x: 282, y: 82, w: 150, h: 32, title: "走势分析员", sub: "量价走势与成交结构" }'),
        ('sub: "bull · 第 1 轮"', 'sub: "第 1 轮"'),
        ('sub: "bear · 第 2 轮"', 'sub: "第 2 轮"'),
        ('sub: "bull_counter · 第 3 轮"', 'sub: "第 3 轮"'),
        ('sub: "risk_chair · 20股对照"', 'sub: "结合全部候选风险"'),
        ('sub: "final_pick · 行业上限"', 'sub: "行业约束下确定名单"'),
        ('const label = meta?.cn_name || k;', 'const label = meta?.cn_name || "扩展量价特征";'),
        ('desc: `特征 ${f.feature}，数值 ${f.value ?? "—"} · ${candidate.explain.method || ""}`',
         'desc: `模型评分依据，数值 ${f.value ?? "—"} · ${candidate.explain.method || ""}`'),
        ('desc: `模型归因字段 explain.${sub}`', 'desc: "模型归因结果"'),
        ('desc: `模型打分字段 model.${k}`', 'desc: "模型评分结果"'),
        ('desc: `信号日全池环境 context.pool.${k}`', 'desc: "信号日全市场环境"'),
        ('return { value: String(bar[field]), desc: `${bar.date} 日线字段 ${field}` };',
         'return { value: String(bar[field]), desc: `${bar.date} 日线行情` };'),
        ('value: flags.length ? flags.map((f) => f.code).join(", ") : "空（无异常提示）",',
         'value: flags.length ? flags.map((f) => f.text).join("；") : "无风险提示",'),
        ('? { value: hit.code, desc: hit.text }', '? { value: hit.text, desc: hit.text }'),
        ('`当前候选未触发 ${code}`', '"当前候选未触发该风险提示"'),
    ):
        if charts.count(old) < 1:
            raise RuntimeError(f"无法唯一定位可读化字段：{old}")
        charts = charts.replace(old, new)
    charts_path.write_text(charts, encoding="utf-8")

    for old, new in (
        ('xgb_F69: "XGBoost F69"', 'xgb_F69: "XGBoost · 69 项特征"'),
        ('rf_F69: "随机森林 F69"', 'rf_F69: "随机森林 · 69 项特征"'),
        ('lstm_SF69: "LSTM S+F69"', 'lstm_SF69: "LSTM · 69 项特征 + 60×11 序列"'),
    ):
        if old not in charts:
            raise RuntimeError(f"无法定位模型展示名称，未生成整合版工作台：{old}")
        charts = charts.replace(old, new, 1)
    charts = charts.replace('"XGBoost F69"', '"XGBoost · 69 项特征"')
    charts = charts.replace('"随机森林 F69"', '"随机森林 · 69 项特征"')
    charts = charts.replace('"LSTM S+F69"', '"LSTM · 69 项特征 + 60×11 序列"')
    charts = charts.replace('"LSTM · 69 特征 + 60×11 序列"', '"LSTM · 69 项特征 + 60×11 序列"')
    charts_path.write_text(charts, encoding="utf-8")

    for old, new in (
        ('风控主席 (risk_chair)', '风控主席'),
        ('终选人 (final_pick)', '终选官'),
        ('模型解读员 (model_analyst)', '模型解读员'),
        ('走势分析员 (chart_analyst)', '走势分析员'),
        ('多方辩手 (bull)', '多方辩手'),
        ('空方辩手 (bear)', '空方辩手'),
        ('多方反驳手 (bull_counter)', '多方回应'),
        ('偏多 bull', '偏多'),
        ('偏空 bear', '偏空'),
        ('中性 neutral', '中性'),
    ):
        app = app.replace(old, new)

    citation_anchor = "function renderCitationsHtml(citations) {\n"
    citation_bridge = '''function citationLabel(path) {
  const clean = String(path || "").replace(/^candidate\\./, "");
  const factor = /^explain\\.top_factors\\[(\\d+)\\]/.exec(clean);
  if (factor) return `模型评分依据 ${Number(factor[1]) + 1}`;
  if (clean.startsWith("features.")) {
    const label = featureLabel(clean.slice("features.".length));
    return label === clean.slice("features.".length) ? "模型输入指标" : label;
  }
  if (clean.startsWith("feature_pct.")) {
    const label = featureLabel(clean.slice("feature_pct.".length));
    return label === clean.slice("feature_pct.".length) ? "全池分位" : `${label}全池分位`;
  }
  if (clean.startsWith("extra_features.") || clean.startsWith("extra_feature_pct.")) {
    const key = clean.slice(clean.indexOf(".") + 1).split(".")[0];
    const label = state.mlOverview?.feature_catalog?.by_name?.[key]?.cn_name;
    return label || (clean.startsWith("extra_feature_pct.") ? "扩展特征全池分位" : "扩展量价特征");
  }
  if (clean.startsWith("peer_models.")) {
    const key = clean.split(".")[1] || "";
    const names = { xgb_F69: "XGBoost", rf_F69: "随机森林", lstm_SF69: "LSTM" };
    return `${names[key] || "对照模型"}同日排名`;
  }
  if (clean.startsWith("bars[")) return "信号日前行情";
  if (clean.startsWith("flags")) return "风险提示";
  if (clean.startsWith("industry_stats.")) return "行业统计";
  if (clean.startsWith("model.")) return "模型评分结果";
  if (clean.startsWith("context.pool.")) return "信号日市场环境";
  return "输入证据";
}

function featureDisplayName(key) {
  const aliases = {
    from_high60: "距 60 日最高收盘价", from_low60: "距 60 日最低收盘价",
    down_streak: "连续下跌天数", mf_net5: "近 5 日资金净流入占比", big_net5: "近 5 日大单净买占比",
    turn_f: "自由流通换手率", rk_turn20: "20 日换手率分位", rk_ivol60: "60 日特质波动率分位",
    min_ret20: "20 日最小日收益", median_r5: "行业近 5 日涨幅中位数", rank_in_industry: "行业内名次",
    vol5: "5 日量比", vol20: "20 日量比", dif_ratio: "MACD DIF 比率", hist_ratio: "MACD 柱比率",
    dev_ma20: "偏离 20 日均线", dev_ma60: "偏离 60 日均线", pct_chg: "当日涨跌幅",
    r5: "近 5 日涨幅", r20: "近 20 日涨幅",
  };
  const extra = state.mlOverview?.feature_catalog?.by_name?.[key]?.cn_name;
  if (extra) return extra;
  const label = featureLabel(key);
  return label !== key ? label : aliases[key] || "相关量价指标";
}

function localizeResearchText(value) {
  let text = String(value || "");
  text = text.replace(/(?:candidate\\.)?explain\\.top_factors(?:\\[(\\d+)\\])?(?:\\.([A-Za-z0-9_]+))?/g,
    (_match, index, key) => `模型评分依据${index === undefined ? "" : `第 ${Number(index) + 1} 项`}${key ? `中的${featureDisplayName(key)}` : ""}`);
  text = text.replace(/(?:candidate\\.)?peer_models\\.([A-Za-z0-9_]+)\\.(pct|rank|score)/g,
    (_match, model, field) => `${({ xgb_F69: "XGBoost", rf_F69: "随机森林", lstm_SF69: "LSTM" })[model] || "对照模型"}同日${({ pct: "全池分位", rank: "名次", score: "模型评分" })[field]}`);
  text = text.replace(/(?:candidate\\.)?extra_feature_pct\\.([A-Za-z0-9_]+)/g, (_match, key) => `${featureDisplayName(key)}全池分位`);
  text = text.replace(/(?:candidate\\.)?extra_features\\.([A-Za-z0-9_]+)/g, (_match, key) => featureDisplayName(key));
  text = text.replace(/(?:candidate\\.)?feature_pct\\.([A-Za-z0-9_]+)/g, (_match, key) => `${featureDisplayName(key)}全池分位`);
  text = text.replace(/(?:candidate\\.)?features\\.([A-Za-z0-9_]+)/g, (_match, key) => featureDisplayName(key));
  text = text.replace(/(?:candidate\\.)?industry_stats\\.([A-Za-z0-9_]+)/g, (_match, key) => ({
    rank_in_industry: "行业内名次", pool_count: "行业内股票数", median_r5: "行业近 5 日涨幅中位数",
    median_r20: "行业近 20 日涨幅中位数",
  })[key] || "行业统计指标");
  text = text.replace(/(?:candidate\\.)?model\\.(pct|rank|score)/g,
    (_match, field) => ({ pct: "当前模型全池分位", rank: "当前模型名次", score: "当前模型评分" })[field]);
  text = text.replace(/(?:candidate\\.)?bars\\[(-?\\d+)\\]\\.([A-Za-z0-9_]+)/g, (_match, index, field) => {
    const labels = { date: "日期", open: "开盘价", high: "最高价", low: "最低价", close: "收盘价", amt: "成交额", amount: "成交额", limit: "涨停状态", hist: "MACD 柱值", dif: "MACD DIF 值" };
    return `信号日前第 ${Math.abs(Number(index))} 个交易日的${labels[field] || "行情数据"}`;
  });
  text = text.replace(/(?:candidate\\.)?bars\\[(-?\\d+)\\]/g, (_match, index) => `信号日前第 ${Math.abs(Number(index))} 个交易日的行情`);
  text = text.replace(/(?:candidate\\.)?flags\\.([A-Za-z0-9_]+)/g, (_match, key) => ({
    limit_up_today: "当日涨停", one_word_limit_up: "一字涨停", limit_up_streak: "连续涨停",
    limit_down_recent: "近期跌停", missing_features: "特征缺省", industry_crowded: "行业拥挤",
  })[key] || "风险提示");
  text = text.replace(/\\b(?:lstm_SF69|LSTM S\\+F69|S\\+F69)\\b/g, "LSTM");
  text = text.replace(/\\bxgb_F69\\b/g, "XGBoost").replace(/\\brf_F69\\b/g, "随机森林");
  text = text.replace(/\\bxgb\\s*\\/\\s*rf\\b/gi, "XGBoost 与随机森林");
  const aliases = {
    from_high60: "距 60 日最高收盘价", from_low60: "距 60 日最低收盘价", down_streak: "连续下跌天数",
    mf_net5: "近 5 日资金净流入占比", big_net5: "近 5 日大单净买占比", turn_f: "自由流通换手率",
    rk_turn20: "20 日换手率分位", rk_ivol60: "60 日特质波动率分位", min_ret20: "20 日最小日收益",
    median_r5: "行业近 5 日涨幅中位数", rank_in_industry: "行业内名次", vol5: "5 日量比", vol20: "20 日量比",
    dif_ratio: "MACD DIF 比率", hist_ratio: "MACD 柱比率", dev_ma20: "偏离 20 日均线",
    dev_ma60: "偏离 60 日均线", pct_chg: "当日涨跌幅", r5: "近 5 日涨幅", r20: "近 20 日涨幅",
    beta60: "60 日贝塔", dev_ma5: "偏离 5 日均线", final_pick: "终选环节", model_analyst: "模型解读员",
    chart_analyst: "走势分析员", risk_chair: "风控主席", bull_counter: "多方回应",
  };
  const aliasPattern = new RegExp(`\\\\b(?:${Object.keys(aliases).sort((a, b) => b.length - a.length).join("|")})\\\\b`, "g");
  text = text.replace(aliasPattern, (key) => aliases[key]);
  text = text.replace(/\\bbase\\b/gi, "基准评分").replace(/\\bscore\\b/gi, "模型评分");
  return text.replace(/([\\u4e00-\\u9fff（）·% ]+)=([-+]?\\d+(?:\\.\\d+)?)/g, "$1为$2");
}

function chatRoleDisplayName(role) {
  const labels = {
    research_copilot: "投研复盘助手", risk_chair: "风控主席", final_pick: "终选官",
    model_analyst: "模型解读员", chart_analyst: "走势分析员", bull: "多方辩手",
    bear: "空方辩手", bull_counter: "多方回应",
  };
    return labels[role?.id] || String(role?.name || "投研助手").replace(/\\s*\\([^)]*\\)\\s*$/, "").trim();
}

function toolDisplayName(name) {
  const labels = {
    get_stock_debate: "读取个股辩论档案", compare_candidates: "对比候选股票",
    screen_candidates: "筛选候选股票", query_stock_history: "查询历史行情",
    check_post_signal_returns: "核验后续行情", list_debate_runs: "查看历史辩论",
    get_model_explain: "查询模型评分依据", compare_models_on_day: "比较当日模型名单",
    get_ml_leaderboard_and_nav: "查询模型排名与净值",
  };
  return labels[name] || "只读数据查询";
}

function localizeToolArguments(value) {
  const labels = {
    ts_code: "股票", ts_codes: "股票名单", as_of: "信号日期", model_id: "模型",
    top: "显示数量", top_n: "候选数量", limit: "返回数量", bars: "行情根数",
    include_bars: "包含日线", include_features: "包含特征", verdict: "辩论方向",
    industry: "行业", flag: "风险提示", sort_by: "排序依据", descending: "降序排列",
    hold_days: "持有天数", with_outcome: "附带事后表现", rule: "策略",
  };
  if (Array.isArray(value)) return value.map(localizeToolArguments);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [labels[key] || "查询条件", localizeToolArguments(item)]));
  }
  return value;
}

function renderCitationsHtml(citations) {
'''
    if app.count(citation_anchor) != 1:
        raise RuntimeError("无法定位证据标签区，未生成整合版工作台")
    app = app.replace(citation_anchor, citation_bridge, 1)
    readable_content = (
        ('${esc(data.conclusion)}', '${esc(localizeResearchText(data.conclusion))}'),
        ('${esc(data.argument)}', '${esc(localizeResearchText(data.argument))}'),
        ('${esc(verdict.thesis)}', '${esc(localizeResearchText(verdict.thesis))}'),
        ('${esc(verdict.action)}', '${esc(localizeResearchText(verdict.action))}'),
        ('${esc(final.reason)}', '${esc(localizeResearchText(final.reason))}'),
        ('${esc(p.reason)}', '${esc(localizeResearchText(p.reason))}'),
        ('${esc(action)}', '${esc(localizeResearchText(action))}'),
        ('${esc(r)}</li>', '${esc(localizeResearchText(r))}</li>'),
        ('${renderChatMarkdown(msg.content)}', '${renderChatMarkdown(localizeResearchText(msg.content))}'),
        ('${renderChatMarkdown(live.text)}', '${renderChatMarkdown(localizeResearchText(live.text))}'),
        ('${esc(msg.thinking)}', '${esc(localizeResearchText(msg.thinking))}'),
        ('${esc(live.thinking || "正在读取上下文并组织分析…")}', '${esc(localizeResearchText(live.thinking || "正在读取上下文并组织分析…"))}'),
        ('${esc(live.thinking)}', '${esc(localizeResearchText(live.thinking))}'),
        ('${esc(stream.thinking)}', '${esc(localizeResearchText(stream.thinking))}'),
        ('${esc(stream.text)}', '${esc(localizeResearchText(stream.text))}'),
        ('${esc(r.name)}</option>', '${esc(chatRoleDisplayName(r))}</option>'),
        ('${esc(chatRole.name)}', '${esc(chatRoleDisplayName(chatRole))}'),
        ('- 整体理由：${final.reason}', '- 整体理由：${localizeResearchText(final.reason)}'),
        ('| ${p.reason} |', '| ${localizeResearchText(p.reason)} |'),
    )
    for old, new in readable_content:
        if old not in app:
            raise RuntimeError(f"无法定位用户可见内容，未生成整合版工作台：{old}")
        app = app.replace(old, new)
    app = app.replace(
        'title="${esc(r.desc)}"',
        'title="${esc(localizeResearchText(r.desc))}"',
        1,
    )
    app = app.replace(
        '.map((c) => `<span class="citation-chip mono" data-citation="${esc(c)}">${esc(c)}</span>`)',
        '.map((c) => `<span class="citation-chip" data-citation="${esc(c)}">${esc(citationLabel(c))}</span>`)',
        1,
    )
    app = app.replace(
        'return `<span class="citation-chip mono" data-citation="${trimmed}">${trimmed}</span>`;',
        'return `<span class="citation-chip" data-citation="${trimmed}">${citationLabel(trimmed)}</span>`;',
        1,
    )
    app = app.replace(
        '(match) => `<span class="citation-chip mono" data-citation="${match}">${match}</span>`,',
        '(match) => `<span class="citation-chip" data-citation="${match}">${citationLabel(match)}</span>`,',
        1,
    )
    app = app.replace(
        '.map((f) => `<span class="citation-chip mono">${esc(f)}</span>`)',
        '.map((f) => `<span class="citation-chip">${esc(citationLabel(f))}</span>`)',
        1,
    )
    app = app.replace(
        '.map((k) => `<span class="citation-chip mono">${esc(k)}</span>`)',
        '.map((k) => `<span class="citation-chip">${esc(citationLabel(k))}</span>`)',
        1,
    )
    app = app.replace('esc(tc.label || tc.name)', 'esc(tc.label || toolDisplayName(tc.name))')
    app = app.replace('esc(tc.name)', 'esc(toolDisplayName(tc.name))')
    app = app.replace(
        'esc(JSON.stringify(tc.arguments || {}, null, 2))',
        'esc(JSON.stringify(localizeToolArguments(tc.arguments || {}), null, 2))',
    )
    app = app.replace('title="${esc(f.feature)} = ${esc(valStr)}"', 'title="${esc(f.cn_name || "模型特征")}：${esc(valStr)}"')
    app = app.replace('title="${esc(k)}">${esc(featureLabel(k))}', 'title="${esc(featureLabel(k))}">${esc(featureLabel(k))}')
    app = app.replace('group: meta?.group || "新增特征",', 'group: (meta?.group || "新增特征").replace(/^[A-H]\\s+/, ""),')
    app = app.replace('const tip = [item.key, item.formula].filter(Boolean).join(" · ");', 'const tip = [item.cn, item.formula].filter(Boolean).join(" · ");')
    app = app.replace('模型打分归因（Top 8）与三模型百分位对照', '模型评分依据与三模型百分位对照')
    app = app.replace('跨模型仅比百分位 pct，不直接比原始 score', '跨模型比较百分位，不直接比较原始评分')
    app = app.replace('基线分 base=${baseTxt} · 模型分 score=${scoreTxt}', '基准评分 ${baseTxt} · 当前模型评分 ${scoreTxt}')
    app = app.replace('全部 ${totalFeatCount} 项 F69 特征（原 28 项 + 新增 ${Object.keys(extraF).length} 项 · A~H 维度）', '全部 ${totalFeatCount} 项模型特征（28 项基础输入 + 41 项扩展特征）')
    app = app.replace(
        '${esc(state.promptsData?.methodology_path || "docs/debate-methodology.md")}',
        '共享辩论方法论',
    )
    app = app.replace(
        r"/\b(?:features\.[A-Za-z0-9_]+|feature_pct\.[A-Za-z0-9_]+|bars\[-?\d+\](?:\.[A-Za-z0-9_]+)?|flags\.[A-Za-z0-9_]+|industry_stats\.[A-Za-z0-9_]+|model\.[A-Za-z0-9_]+|context\.[A-Za-z0-9_]+)\b/g",
        r"/\b(?:(?:features|feature_pct|extra_features|extra_feature_pct)\.[A-Za-z0-9_]+|explain\.top_factors\[\d+\](?:\.[A-Za-z0-9_]+)?|peer_models\.[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)?|bars\[-?\d+\](?:\.[A-Za-z0-9_]+)?|flags\.[A-Za-z0-9_]+|industry_stats\.[A-Za-z0-9_]+|model\.[A-Za-z0-9_]+|context\.pool\.[A-Za-z0-9_]+)\b/g",
        1,
    )
    app = app.replace(
        r"/^(?:features|feature_pct|flags|industry_stats|model|context)\.[A-Za-z0-9_]+$|^bars\[-?\d+\](?:\.[A-Za-z0-9_]+)?$/",
        r"/^(?:features|feature_pct|extra_features|extra_feature_pct|flags|industry_stats|model|context\.pool)\.[A-Za-z0-9_]+$|^explain\.top_factors\[\d+\](?:\.[A-Za-z0-9_]+)?$|^peer_models\.[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)?$|^bars\[-?\d+\](?:\.[A-Za-z0-9_]+)?$/",
        1,
    )
    app = app.replace('XGBoost F69', 'XGBoost · 69 项特征')
    app = app.replace('随机森林 F69', '随机森林 · 69 项特征')
    app = app.replace('LSTM S+F69', 'LSTM · 69 项特征 + 60×11 序列')
    app = app.replace('LSTM · 69 特征 + 60×11 序列', 'LSTM · 69 项特征 + 60×11 序列')
    app = app.replace('S+F69', '69 项特征 + 60×11 序列')
    feature_set_anchor = 'const ctxFeatureSet = state.run?.request?.context?.feature_set || (hasExtra ? "F69" : "F28");'
    if app.count(feature_set_anchor) != 1:
        raise RuntimeError("无法定位模型特征集显示名称，未生成整合版工作台")
    app = app.replace(
        feature_set_anchor,
        feature_set_anchor + '\n  const ctxFeatureSetLabel = ctxFeatureSet === "F69" ? "69 项模型特征" : ctxFeatureSet;',
        1,
    )
    app = app.replace('${esc(ctxFeatureSet)}</span>', '${esc(ctxFeatureSetLabel)}</span>', 1)
    model_label_anchor = 'const ctxModelLabel = state.run?.request?.context?.model_label || "反弹模型";'
    if app.count(model_label_anchor) != 1:
        raise RuntimeError("无法定位当前模型显示名称，未生成整合版工作台")
    app = app.replace(
        model_label_anchor,
        model_label_anchor + '\n  const ctxModelLabelShort = ctxModelLabel.replace(/ · 69 项特征$/, "");',
        1,
    )
    app = app.replace(
        '${esc(ctxModelLabel)} · ${esc(ctxFeatureSetLabel)}',
        '${esc(ctxModelLabelShort)} · ${esc(ctxFeatureSetLabel)}',
        1,
    )

    if app.count('class="role-code mono"') < 3:
        raise RuntimeError("成员页面角色结构与预期不同，未生成整合版工作台")

    state_anchor = "};\n\nconst dom = {\n"
    state_bridge = '''};

const __dashboardContext = new URLSearchParams(window.location.search);
const __dashboardModels = ["xgb_F69", "rf_F69", "lstm_SF69"];
if (__dashboardModels.includes(__dashboardContext.get("model_source"))) {
  state.selectedModelSource = __dashboardContext.get("model_source");
}
let __dashboardParentOrigin = "";
try {
  const candidate = new URL(__dashboardContext.get("parent_origin") || "");
  if (candidate.protocol === "http:" && ["127.0.0.1", "localhost"].includes(candidate.hostname) && candidate.port) {
    __dashboardParentOrigin = candidate.origin;
  }
} catch {}

const dom = {
'''
    if app.count(state_anchor) != 1:
        raise RuntimeError("无法定位组员页面状态区，未生成整合版工作台")
    app = app.replace(state_anchor, state_bridge, 1)

    app = app.replace('  chatOpen: false,\n  chatWide: false,\n  leftCollapsed: false,',
                      '  chatOpen: true,\n  chatWide: false,\n  leftCollapsed: false,', 1)

    layout_anchor = '''function renderChatLayout() {
  dom.workspace.classList.toggle("chat-open", state.chatOpen);
  dom.workspace.classList.toggle("chat-wide", state.chatOpen && state.chatWide);
  dom.workspace.classList.toggle("left-collapsed", state.leftCollapsed);

  dom.chatDrawer.hidden = !state.chatOpen;
  dom.btnToggleChat.classList.toggle("active", state.chatOpen);
  dom.btnExpandLeft.hidden = !state.leftCollapsed;

  dom.btnChatToggleLeft.textContent = state.leftCollapsed ? "展开左栏" : "收起左栏";
  dom.btnChatToggleLeft.classList.toggle("active", state.leftCollapsed);
  dom.btnChatWide.textContent = state.chatWide ? "标准宽" : "宽屏";
  dom.btnChatWide.classList.toggle("active", state.chatWide);
}'''
    layout_replacement = '''function renderChatLayout() {
  state.chatOpen = true;
  state.chatWide = false;
  state.leftCollapsed = false;
  dom.workspace.classList.remove("chat-open", "chat-wide", "left-collapsed");
  dom.chatDrawer.hidden = false;
  dom.btnToggleChat.classList.remove("active");
  dom.btnExpandLeft.hidden = true;
}'''
    if app.count(layout_anchor) != 1:
        raise RuntimeError("无法定位投研对话布局，未生成整合版工作台")
    app = app.replace(layout_anchor, layout_replacement, 1)

    app = app.replace("✦ 反弹模型投研复盘 Agent（档位 B）", "投研复盘对话", 1)
    app = app.replace(
        "已预置完整方法论 <code>docs/debate-methodology.md</code>，默认掌握终选名单、全部 20 只风控简表与当前聚焦个股的完整 6 轮辩论；问及任意其他股票、跨股筛选或历史事后收益时，会自动调用只读工具精准查证。",
        "可围绕当前候选、角色观点和模型结果追问；需要跨股或历史数据时，对话会调用只读查询。",
        1,
    )
    app = app.replace(
        '''  const quickPrompts = [
    "为什么终选换掉了原模型第 1 名？被换掉的票各自有什么硬伤？",
    "对比当前所有「看多」股票的买点结构与核心风险，列成一张对照表。",
    "帮我筛一下这 20 只送辩股里，近 5 天放量最明显（vol5 最高）的前 5 只是谁？",
    `针对 ${focusName}，给出明日 T+1 开盘前可执行检查清单（什么条件买、什么条件放弃）。`,
  ];''',
        '''  const quickPrompts = [
    "为什么这只股票进入终选？",
    "它的主要风险和证据是什么？",
  ];''',
        1,
    )

    normalize_anchor = '''  const data = await res.json().catch(() => ({}));
  if (!res.ok) {'''
    normalize_bridge = '''  const data = await res.json().catch(() => ({}));
  const normalizeDashboardLabels = (value) => {
    if (typeof value === "string") return value.replaceAll("XGBoost F69", "XGBoost · 69 项特征").replaceAll("随机森林 F69", "随机森林 · 69 项特征").replaceAll("LSTM S+F69", "LSTM · 69 项特征 + 60×11 序列").replaceAll("LSTM SF69", "LSTM · 69 项特征 + 60×11 序列").replaceAll("S+F69", "69 项特征 + 60×11 序列");
    if (Array.isArray(value)) return value.map(normalizeDashboardLabels);
    if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, normalizeDashboardLabels(item)]));
    return value;
  };
  Object.assign(data, normalizeDashboardLabels(data));
  if (!res.ok) {'''
    if app.count(normalize_anchor) != 1:
        raise RuntimeError("无法定位组员页面 API 响应区，未生成整合版工作台")
    app = app.replace(normalize_anchor, normalize_bridge, 1)

    tab_anchor = '''function switchTab(tab) {
  state.activeTab = tab;
  if (tab === "prompts" && !state.promptsData && !state.promptsLoading) {
    refreshPromptsData();
  }
  if (tab === "models") {
    ensureMlViewLoaded();
  }
  renderAll();
}'''
    tab_bridge = '''function switchTab(tab) {
  state.activeTab = tab;
  if (tab === "prompts" && !state.promptsData && !state.promptsLoading) {
    refreshPromptsData();
  }
  if (tab === "models") {
    ensureMlViewLoaded();
  }
  renderAll();
  if (__dashboardParentOrigin) {
    window.parent.postMessage({ type: "debate-tab", tab }, __dashboardParentOrigin);
    requestAnimationFrame(__reportDashboardSize);
  }
}'''
    if app.count(tab_anchor) != 1:
        raise RuntimeError("无法定位组员页面导航区，未生成整合版工作台")
    app = app.replace(tab_anchor, tab_bridge, 1)

    boot_anchor = "\nboot();\n"
    bridge = '''
async function __applyDashboardContext(context) {
  const model = __dashboardModels.includes(context.model_source) ? context.model_source : state.selectedModelSource;
  const asOf = /^\\d{4}-\\d{2}-\\d{2}$/.test(context.as_of || "") ? context.as_of : "";
  const hash = /^[0-9a-f]{12}$/.test(context.hash || "") ? context.hash : "";
  const code = /^[0-9]{6}\\.(?:SH|SZ|BJ)$/.test(context.ts_code || "") ? context.ts_code : "";
  if (model !== state.selectedModelSource) {
    state.selectedModelSource = model;
    state.selectedRunKey = "";
    state.run = null;
    state.stockState = {};
    state.streams = {};
    await refreshMeta({ keepSelection: false });
  }

  const match = (state.meta?.runs || []).find((run) =>
    run.model_source === model && (!asOf || run.as_of === asOf) && (!hash || run.hash_prefix === hash)
  );
  const runKey = match ? `${match.as_of}/${match.hash_prefix}` : "";
  if (match && state.selectedRunKey !== runKey) {
    await loadRunDetail(match.as_of, match.hash_prefix, { modelSource: model });
  } else if (!match) {
    state.run = null;
    state.selectedRunKey = "";
    state.stockState = {};
    state.streams = {};
  }

  if (asOf && state.meta?.trading_days?.includes(asOf)) {
    state.selectedDate = asOf;
    dom.selDate.value = asOf;
  }
  dom.selModelSource.value = model;
  dom.selRun.value = runKey;
  const candidates = state.run?.request?.candidates || [];
  if (code && candidates.some((candidate) => candidate.ts_code === code)) {
    state.selectedCode = code;
    state.chatExcludeSelected = false;
    state.autoFollow = false;
    dom.chkAutofollow.checked = false;
  } else if (!match) {
    state.selectedCode = "";
  }
  const requestedTab = ["stock", "final", "models", "prompts"].includes(context.tab) ? context.tab : "stock";
  state.activeTab = match || requestedTab === "prompts" ? requestedTab : "models";
  if (!match && state.activeTab === "models") ensureMlViewLoaded();
  renderAll();
  if (!match) {
    await Promise.all([
      refreshChatSessions({ preserveActive: false }),
      refreshPromptsData(),
    ]);
  }
}

function __postDashboardMessage(payload) {
  if (__dashboardParentOrigin && window.parent !== window) {
    window.parent.postMessage(payload, __dashboardParentOrigin);
  }
}

let __dashboardApplyQueue = Promise.resolve();
window.addEventListener("message", (event) => {
  if (!__dashboardParentOrigin || event.origin !== __dashboardParentOrigin || event.source !== window.parent) return;
  if (event.data?.type === "dashboard-context") {
    __dashboardApplyQueue = __dashboardApplyQueue
      .then(() => __applyDashboardContext(event.data))
      .catch((error) => showToast(`同步看板状态失败：${error.message}`, "error", 0));
  } else if (event.data?.type === "dashboard-chat-prefill" && typeof event.data.text === "string") {
    dom.chatInput.value = event.data.text.slice(0, 2000);
    dom.chatInput.focus();
    dom.chatInput.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
});

let __dashboardSizePending = false;
let __dashboardLastHeight = 0;
function __reportDashboardSize() {
  if (__dashboardSizePending) return;
  __dashboardSizePending = true;
  requestAnimationFrame(() => {
    __dashboardSizePending = false;
    const shell = document.querySelector(".app-shell");
    if (!shell || !shell.offsetWidth) return;
    // Read content height rather than the iframe viewport, so shorter tabs shrink.
    const height = Math.ceil(Math.max(shell.scrollHeight, shell.getBoundingClientRect().height));
    if (height > 0 && height !== __dashboardLastHeight) {
      __dashboardLastHeight = height;
      __postDashboardMessage({ type: "debate-resize", height });
    }
  });
}

async function __startDashboardBridge() {
  await boot();
  if (!__dashboardParentOrigin) return;
  try {
    await __applyDashboardContext({
      model_source: __dashboardContext.get("model_source"),
      as_of: __dashboardContext.get("as_of"),
      hash: __dashboardContext.get("hash"),
      ts_code: __dashboardContext.get("ts_code"),
      tab: __dashboardContext.get("tab"),
    });
  } catch (error) {
    showToast(`载入看板上下文失败：${error.message}`, "error", 0);
  }
  __postDashboardMessage({ type: "debate-ready" });
  const observer = new ResizeObserver(__reportDashboardSize);
  observer.observe(document.querySelector(".app-shell"));
  const mutations = new MutationObserver(() => requestAnimationFrame(__reportDashboardSize));
  mutations.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["hidden", "class"] });
  window.addEventListener("load", __reportDashboardSize, { once: true });
  requestAnimationFrame(__reportDashboardSize);
}

__startDashboardBridge();
'''
    if app.count(boot_anchor) != 1:
        raise RuntimeError("无法定位组员页面启动区，未生成整合版工作台")
    app = app.replace(boot_anchor, "\n" + bridge, 1)
    app_path.write_text(_dashboard_routes(app), encoding="utf-8")
    charts_path.write_text(_dashboard_routes(charts), encoding="utf-8")

    index_path = ui_dir / "index.html"
    html = index_path.read_text(encoding="utf-8")
    html = html.replace(
        '<link rel="stylesheet" href="/static/styles.css" />',
        '<link rel="stylesheet" href="/static/styles.css" />\n'
        '    <link id="dashboard-debate-layout" rel="stylesheet" href="/assets/debate_layout.css" />',
        1,
    )
    html = html.replace("反弹模型 · AI 多角色辩论复核台", "AI辩论台", 1)
    html = html.replace("反弹模型 · AI 辩论复核台", "AI辩论台", 1)
    html = html.replace(
        '    <script type="module" src="/static/app.js"></script>',
        '''    <script>
      if (window.parent !== window) document.body.classList.add("embedded-workbench");
      const chatDrawer = document.getElementById("chat-drawer");
      const leftPane = document.querySelector(".left-pane");
      const queue = leftPane?.querySelector(".roster-card");
      if (chatDrawer && queue) queue.after(chatDrawer);
    </script>
    <script type="module" src="/static/app.js"></script>''',
        1,
    )
    index_path.write_text(_dashboard_routes(html), encoding="utf-8")
    return ui_dir


def _dashboard_routes(source: str) -> str:
    """Keep API, static resources and SSE under the public dashboard origin."""
    for prefix in ('"', "'", "`"):
        for route in ("/api/", "/static/"):
            source = source.replace(prefix + route, prefix + "/debate" + route)
    return source


class EmbeddedDebate:
    """Run the member backend inside the main service, without a fixed extra port."""

    def __init__(self) -> None:
        import duckdb
        from research_views import model_api

        model_api()
        self._temporary = tempfile.TemporaryDirectory(prefix="rebound-debate-", dir=DASHBOARD)
        self.directory = Path(self._temporary.name).resolve()
        self._real_connect = duckdb.connect
        temp_path = Path(self._temporary.name).as_posix()
        try:
            web_ui_dir = prepare_embedded_ui(Path(self._temporary.name))
        except Exception:
            self._temporary.cleanup()
            raise

        def connect_with_temp_directory(*args, **kwargs):
            connection = self._real_connect(*args, **kwargs)
            database = args[0] if args else kwargs.get("database", ":memory:")
            try:
                resolved = Path(database).resolve()
            except (OSError, TypeError, ValueError):
                return connection
            if resolved == DATABASE.resolve():
                connection.execute(f"SET temp_directory = '{temp_path}'")
            return connection

        try:
            from rebound.config import load_settings
            from rebound.debate import web

            web.WEB_UI_DIR = web_ui_dir
            settings = load_settings(root=MODEL_REPO)
            self.server, self.hub = web.create_server(
                settings, default_db=DATABASE, host="127.0.0.1", port=0,
            )
            duckdb.connect = connect_with_temp_directory
        except Exception:
            self._temporary.cleanup()
            raise
        self.origin = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True, name="debate-web")
        self._thread.start()

    def close(self) -> None:
        import duckdb
        try:
            self.hub.stop_run()
            self.server.shutdown()
            self.server.server_close()
            self._thread.join(timeout=3)
        finally:
            duckdb.connect = self._real_connect
            self._temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit("请运行 server.py；AI 辩论台已合并到主看板 /debate/。")
