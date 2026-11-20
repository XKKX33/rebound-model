import {
  NAV_SERIES_STYLES,
  featureLabel,
  featureUnit,
  renderNavComparisonChart,
  renderStartsAnnualizedChart,
  renderStockKlineChart,
  renderWorkflowGraph,
  resolveCitation,
} from "/static/charts.js";

const STOCK_ROLES = [
  { id: "model_analyst", short: "模", name: "模型解读员", phase: "analyst" },
  { id: "chart_analyst", short: "走", name: "走势分析员", phase: "analyst" },
  { id: "bull", short: "多", name: "多方立论", phase: "debate" },
  { id: "bear", short: "空", name: "空方反驳", phase: "debate" },
  { id: "bull_counter", short: "反", name: "多方二次回应", phase: "debate" },
  { id: "risk_chair", short: "风", name: "风控主席", phase: "verdict" },
];

const FLAG_LABELS = {
  limit_up_today: "当日涨停",
  one_word_limit_up: "一字涨停",
  limit_up_streak: "连续涨停",
  limit_down_recent: "近期跌停",
  missing_features: "特征缺省",
  industry_crowded: "行业拥挤",
};

const state = {
  meta: null,
  selectedDb: "",
  selectedModelSource: "xgb_F69", // "xgb_F69" | "rf_F69" | "lstm_SF69"
  selectedDate: "",
  selectedRunKey: "", // `${as_of}/${hash_prefix}`
  activeTab: "stock", // "stock" | "final" | "models" | "prompts"
  filter: "all",
  sort: "rank",
  autoFollow: true,
  isDryRun: false,

  // 当前展示的运行数据
  run: null, // { as_of, variant, cadence, hash_prefix, completed, official, is_running, request, final, result, usage }
  selectedCode: "",
  // 单股实时进度与结果（合并 stocks 检查点 + events + SSE 增量）
  stockState: {}, // ts_code -> { status, analysts:{}, debate:{}, verdict, failedError, activeRole, attempt, roleErrors:{} }
  // 逐字打字机流缓存：`${ts_code}::${role}` -> { thinking, text, attempt, started_at, completed }
  streams: {},
  // 当前正在运行的全局 job 状态
  activeJob: null,
  // 计时器，用于更新思考耗时
  elapsedTimer: null,
  eventSource: null,
  // 历史模式下的流式路由动画演示状态
  demoFlowStage: -1,
  demoFlowTimer: null,

  // 右侧 AI 投研复盘对话栏（档位 B：工具调用 Agent + 持久化会话）
  chatOpen: false,
  chatWide: false,
  leftCollapsed: false,
  chatRoles: [
    { id: "research_copilot", name: "投研复盘助手", desc: "全局复盘视角：解读终选逻辑、三模型共识与归因对比、排查风险、生成次日开盘清单" },
    { id: "risk_chair", name: "风控主席 (risk_chair)", desc: "以风控拍板视角回应看多/看空、归因稳健性与 0~100 打分依据" },
    { id: "final_pick", name: "终选人 (final_pick)", desc: "以全局终选视角解释行业上限约束与保留/换出取舍" },
    { id: "model_analyst", name: "模型解读员 (model_analyst)", desc: "侧重三套 F69 模型（XGBoost/随机森林/LSTM）排名、69 特征分位、Top-8 归因与跨模型分歧拆解" },
    { id: "chart_analyst", name: "走势分析员 (chart_analyst)", desc: "只看 20 日量价形态、K 线微结构、换手资金与 MACD 动能" },
    { id: "bull", name: "多方辩手 (bull)", desc: "从做多角度深挖 T+1 买进、持有约 5 日的看多依据" },
    { id: "bear", name: "空方辩手 (bear)", desc: "从风控回避角度继续挑刺结构缺陷与买入障碍" },
    { id: "bull_counter", name: "多方反驳手 (bull_counter)", desc: "针对空方质疑做逐条反驳与风险权重评估" },
  ],
  chatTools: [],
  chatSessions: [],
  activeChatSessionId: "",
  chatRole: "research_copilot",
  chatExtraAttachedCodes: [],
  chatExcludeSelected: false,
  chatStreaming: false,
  chatAbortController: null,
  chatLiveTurn: null,
  chatStoreName: "",

  // 提示词与单股实际 User JSON 查看器
  promptsData: null,
  promptsSourcePref: "auto", // "auto" | "frozen" | "workspace"
  promptsLoading: false,
  promptsError: "",
  promptsSelectedId: "model_analyst", // "__methodology__" | debate role id | "chat:<preset>"
  promptsViewMode: "preset", // "preset" | "system" | "payload"
  promptsPayloadCode: "",
  promptsPayloadCompact: false,
  promptsPayloadCache: {}, // `${runKey}|${roleKey}|${tsCode}` -> payload result
  promptsPayloadLoadingKey: "",
  promptModal: {
    open: false,
    roleKey: "model_analyst",
    tsCode: "",
    pane: "preset", // "preset" | "system" | "payload"
  },

  // 多模型对比、净值、共识与单股归因（XGBoost F69 / 随机森林 F69 / LSTM S+F69）
  mlOverview: null,
  mlOverviewLoading: false,
  mlOverviewError: "",
  mlNavCache: {}, // `${rule}|${startKey}` -> nav response
  mlNavRule: "DB-K30-b8", // 固定主规则（20万本金、最多30只）
  mlNavStart: "test", // "test" | "2015" .. "2024"
  mlNavViewMode: "nav", // "nav" | "cagr" | "starts_curve"
  mlNavDetailTab: "starts", // "starts" | "yearly" | "none"
  mlNavLog: false,
  mlNavHiddenKeys: { hs300: true, zz500: true, zz1000: true },
  mlNavLoadingKey: "",
  mlConsensus: null,
  mlConsensusDate: "",
  mlConsensusWithOutcome: false,
  mlConsensusLoading: false,
  mlConsensusError: "",
  mlExplainCache: {}, // `${as_of}|${ts_code}` -> /api/ml/explain
  mlExplainLoadingKey: "",
  mlExplainModelTab: "", // 当前单股归因卡片选中的模型 tab
  featureGroupFilter: "all", // 单股 69 特征分组筛选
  mlCatalogGroup: "f69", // 特征字典筛选："f69" | "all" | "extra41" | 具体分组名
  mlCatalogQuery: "",
};

const dom = {
  workspace: document.getElementById("workspace"),
  modelMetaLine: document.getElementById("model-meta-line"),
  selDb: document.getElementById("sel-db"),
  selModelSource: document.getElementById("sel-model-source"),
  selDate: document.getElementById("sel-date"),
  selRun: document.getElementById("sel-run"),
  selEffort: document.getElementById("sel-effort"),
  selConcurrency: document.getElementById("sel-concurrency"),
  btnDryRun: document.getElementById("btn-dry-run"),
  btnStart: document.getElementById("btn-start"),
  btnStop: document.getElementById("btn-stop"),
  btnToggleChat: document.getElementById("btn-toggle-chat"),
  statusBadge: document.getElementById("status-badge"),
  kpiProgress: document.getElementById("kpi-progress"),
  kpiTokens: document.getElementById("kpi-tokens"),
  toastBar: document.getElementById("toast-bar"),

  workflowGraph: document.getElementById("workflow-graph"),
  workflowSubtitle: document.getElementById("workflow-subtitle"),
  workflowActivePill: document.getElementById("workflow-active-pill"),
  btnDemoFlow: document.getElementById("btn-demo-flow"),
  wfPoolSize: document.getElementById("wf-pool-size"),
  wfLimits: document.getElementById("wf-limits"),
  wfVerdictSplit: document.getElementById("wf-verdict-split"),

  rosterCountLabel: document.getElementById("roster-count-label"),
  rosterProgressFill: document.getElementById("roster-progress-fill"),
  filterPills: document.getElementById("filter-pills"),
  selSort: document.getElementById("sel-sort"),
  chkAutofollow: document.getElementById("chk-autofollow"),
  rosterList: document.getElementById("roster-list"),

  btnExpandLeft: document.getElementById("btn-expand-left"),
  tabStock: document.getElementById("tab-stock"),
  tabFinal: document.getElementById("tab-final"),
  tabModels: document.getElementById("tab-models"),
  tabPrompts: document.getElementById("tab-prompts"),
  tabStockTag: document.getElementById("tab-stock-tag"),
  tabFinalTag: document.getElementById("tab-final-tag"),
  tabModelsTag: document.getElementById("tab-models-tag"),
  tabPromptsTag: document.getElementById("tab-prompts-tag"),
  btnCopyMd: document.getElementById("btn-copy-md"),
  btnDownloadJson: document.getElementById("btn-download-json"),

  viewStock: document.getElementById("view-stock"),
  viewFinal: document.getElementById("view-final"),
  viewModels: document.getElementById("view-models"),
  viewPrompts: document.getElementById("view-prompts"),
  stockEmpty: document.getElementById("stock-empty"),
  stockContent: document.getElementById("stock-content"),
  finalContent: document.getElementById("final-content"),
  modelsContent: document.getElementById("models-content"),
  promptsContent: document.getElementById("prompts-content"),
  citationTooltip: document.getElementById("citation-tooltip"),

  // 快捷提示词与入参弹窗
  promptModal: document.getElementById("prompt-modal"),
  promptModalTitle: document.getElementById("prompt-modal-title"),
  promptModalSubtitle: document.getElementById("prompt-modal-subtitle"),
  btnModalOpenTab: document.getElementById("btn-modal-open-tab"),
  btnModalClose: document.getElementById("btn-modal-close"),
  promptModalTabs: document.getElementById("prompt-modal-tabs"),
  promptModalBody: document.getElementById("prompt-modal-body"),

  // 右侧 AI 投研复盘对话栏
  chatDrawer: document.getElementById("chat-drawer"),
  chatSubtitle: document.getElementById("chat-subtitle"),
  btnChatToggleLeft: document.getElementById("btn-chat-toggle-left"),
  btnChatWide: document.getElementById("btn-chat-wide"),
  btnChatClose: document.getElementById("btn-chat-close"),
  selChatRole: document.getElementById("sel-chat-role"),
  btnChatPromptInfo: document.getElementById("btn-chat-prompt-info"),
  selChatSession: document.getElementById("sel-chat-session"),
  btnChatNew: document.getElementById("btn-chat-new"),
  btnChatDelete: document.getElementById("btn-chat-delete"),
  chatContextPills: document.getElementById("chat-context-pills"),
  selChatAttach: document.getElementById("sel-chat-attach"),
  chatToolsBadge: document.getElementById("chat-tools-badge"),
  chatMessages: document.getElementById("chat-messages"),
  chatForm: document.getElementById("chat-form"),
  chatInput: document.getElementById("chat-input"),
  chatPersistHint: document.getElementById("chat-persist-hint"),
  btnChatStop: document.getElementById("btn-chat-stop"),
  btnChatSend: document.getElementById("btn-chat-send"),
};

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fmtNum(v, digits = 2) {
  if (typeof v !== "number" || !Number.isFinite(v)) return "—";
  return Number.isInteger(v) ? String(v) : v.toFixed(digits);
}

function fmtPct(v) {
  if (typeof v !== "number" || !Number.isFinite(v)) return "—";
  return `${v > 0 ? "+" : ""}${v.toFixed(2)}%`;
}

function fmtTokens(usage) {
  if (!usage) return "0 tok";
  const total = (usage.input_tokens || 0) + (usage.output_tokens || 0);
  if (total >= 10000) return `${(total / 10000).toFixed(1)}w tok`;
  return `${total.toLocaleString()} tok`;
}

function showToast(msg, tone = "info", autoHideMs = 4500) {
  if (!msg) {
    dom.toastBar.hidden = true;
    return;
  }
  dom.toastBar.textContent = msg;
  dom.toastBar.className = `toast-bar${tone === "error" ? " toast-error" : tone === "warn" ? " toast-warn" : ""}`;
  dom.toastBar.hidden = false;
  if (autoHideMs > 0) {
    clearTimeout(showToast._timer);
    showToast._timer = setTimeout(() => {
      dom.toastBar.hidden = true;
    }, autoHideMs);
  }
}

async function apiJson(url, options) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || `请求失败 (${res.status})`);
  }
  return data;
}

// ---------------------------------------------------------------------------
// 数据水合：把 request + stocks + result + events + live_streams 组装成统一视图态
// ---------------------------------------------------------------------------

function resetStockStateFromRequest(request) {
  const map = {};
  for (const c of request?.candidates || []) {
    map[c.ts_code] = {
      status: "pending", // pending | running | completed | failed
      analysts: {},
      debate: {},
      verdict: null,
      failedError: "",
      activeRole: null,
      attempt: 1,
      roleErrors: {},
    };
  }
  state.stockState = map;
}

function hydrateRunDetail(detail) {
  state.isDryRun = false;
  state.run = detail;
  state.streams = {};
  for (const [k, v] of Object.entries(detail.live_streams || {})) {
    state.streams[k] = { ...v };
  }

  resetStockStateFromRequest(detail.request);

  // 1. 先回放 events.ndjson 中已完成的单角色输出（适用于断点中/运行中的票）
  for (const ev of detail.events || []) {
    const code = ev.ts_code;
    if (!code || code === "-" || !state.stockState[code]) continue;
    const st = state.stockState[code];
    if (ev.type === "stock.started" && st.status === "pending") {
      st.status = "running";
    } else if (ev.type === "role.completed" && ev.data) {
      applyRoleCompletedToStock(st, ev.role, ev.data);
    } else if (ev.type === "role.failed") {
      st.roleErrors[ev.role] = ev.error || "调用失败";
    } else if (ev.type === "stock.failed") {
      st.status = "failed";
      st.failedError = ev.error || "未完成";
      st.activeRole = null;
    }
  }

  // 2. 合并已原子落盘的 stocks/*.json 检查点（权威完整）
  for (const [code, bundle] of Object.entries(detail.stocks || {})) {
    const st = state.stockState[code];
    if (!st) continue;
    st.status = "completed";
    st.activeRole = null;
    st.failedError = "";
    for (const a of bundle.analysts || []) st.analysts[a.role] = a;
    for (const d of bundle.debate || []) st.debate[d.round] = d;
    if (bundle.verdict) st.verdict = bundle.verdict;
  }

  // 3. 若已有冻结 result.json，也合并其中的 failed_stocks
  if (detail.result) {
    for (const f of detail.result.failed_stocks || []) {
      const st = state.stockState[f.ts_code];
      if (st && st.status !== "completed") {
        st.status = "failed";
        st.failedError = `${f.role}: ${f.error}`;
        st.activeRole = null;
      }
    }
  }

  // 4. 叠加当前内存里正在活跃推理的角色
  for (const [code, info] of Object.entries(detail.active_roles || {})) {
    if (code === "-") continue;
    const st = state.stockState[code];
    if (st && st.status !== "completed") {
      st.status = "running";
      st.activeRole = info.role;
      st.attempt = info.attempt || 1;
    }
  }

  // 选中默认股票（优先保留原选中，其次选正在跑的，否则选终选第 1 名或模型第 1 名）
  const codes = (detail.request?.candidates || []).map((c) => c.ts_code);
  if (!codes.includes(state.selectedCode)) {
    const runningCode = codes.find((c) => state.stockState[c]?.status === "running");
    const firstPick = detail.final?.picks?.[0]?.ts_code;
    state.selectedCode = runningCode || (codes.includes(firstPick) ? firstPick : codes[0] || "");
  }

  renderAll();
}

function applyRoleCompletedToStock(st, role, data) {
  if (role === "model_analyst" || role === "chart_analyst") {
    st.analysts[role] = { role, ...data };
  } else if (role === "bull" || role === "bear" || role === "bull_counter") {
    st.debate[role] = { round: role, ...data };
  } else if (role === "risk_chair") {
    st.verdict = { ...data };
  }
}

// ---------------------------------------------------------------------------
// 左侧 Magpie 风格工作流图状态计算
// ---------------------------------------------------------------------------

function computeWorkflowGraphState() {
  const candidates = state.run?.request?.candidates || [];
  const hasSnapshot = candidates.length > 0;
  const activeJobRunning = Boolean(state.activeJob?.running && isViewingActiveRun());
  const isFinalized = Boolean(state.run?.final);

  // 汇总当前并发运行中的角色（可能有 2 只股分别在不同节点！）
  const activeByRole = {}; // role -> [stockName, ...]
  const nameByCode = new Map(candidates.map((c) => [c.ts_code, c.name]));

  if (activeJobRunning) {
    for (const [code, st] of Object.entries(state.stockState)) {
      if (st.activeRole) {
        activeByRole[st.activeRole] = activeByRole[st.activeRole] || [];
        activeByRole[st.activeRole].push(nameByCode.get(code) || code);
      }
    }
    if (state.activeJob?.stage === "finalizing" || state.streams["-::final_pick"]?.completed === false) {
      activeByRole.final_pick = ["终选定稿中"];
    }
  } else if (state.demoFlowStage >= 0) {
    const demoStages = [
      ["model_analyst", "chart_analyst"],
      ["bull"],
      ["bear"],
      ["bull_counter"],
      ["risk_chair"],
      ["final_pick"],
    ];
    const activeRoles = demoStages[state.demoFlowStage % demoStages.length] || [];
    const focusName = nameByCode.get(state.selectedCode) || "样本股";
    for (const r of activeRoles) {
      activeByRole[r] = [r === "final_pick" ? "汇总终选前 6" : focusName];
    }
  }

  // 当前选中股票的局部进度（用于历史回放或无并发活跃时）
  const selSt = state.stockState[state.selectedCode];

  const nodeStates = {
    snapshot: hasSnapshot ? "done" : state.activeJob?.running && state.activeJob.stage === "preparing" ? "active" : "idle",
  };
  const activeBadges = {};

  if (hasSnapshot) {
    activeBadges.snapshot = `${state.run.request.context.pool_size} 只 → 前 ${candidates.length} 只`;
  } else if (state.activeJob?.running && state.activeJob.stage === "preparing") {
    activeBadges.snapshot = "正在构建快照…";
  }

  const roleDoneForSelected = (roleId) => {
    if (!selSt) return false;
    if (roleId === "model_analyst" || roleId === "chart_analyst") return Boolean(selSt.analysts[roleId]);
    if (roleId === "bull" || roleId === "bear" || roleId === "bull_counter") return Boolean(selSt.debate[roleId]);
    if (roleId === "risk_chair") return Boolean(selSt.verdict);
    return false;
  };

  for (const r of STOCK_ROLES) {
    if (activeByRole[r.id]?.length) {
      nodeStates[r.id] = "active";
      activeBadges[r.id] = `● ${activeByRole[r.id].join(" / ")}`;
    } else if (roleDoneForSelected(r.id)) {
      nodeStates[r.id] = "done";
    } else if (selSt?.roleErrors?.[r.id] && selSt.status === "failed") {
      nodeStates[r.id] = "error";
    } else {
      nodeStates[r.id] = "idle";
    }
  }

  if (activeByRole.final_pick?.length) {
    nodeStates.final_pick = "active";
    activeBadges.final_pick = "● 正在生成终选前 6…";
  } else if (isFinalized) {
    nodeStates.final_pick = "done";
    activeBadges.final_pick = `已定稿 ${state.run.final.final_count} 只`;
  } else {
    nodeStates.final_pick = "idle";
  }

  // 边状态：目标节点 active → 入口边流光动画；目标节点 done → 入口边高亮实线
  const edgeStatus = (from, to) => {
    if (nodeStates[to] === "active") return "active";
    if (nodeStates[to] === "done" && (nodeStates[from] === "done" || nodeStates[from] === "active")) return "done";
    return "idle";
  };

  const edgeStates = {
    "e-snap-model": edgeStatus("snapshot", "model_analyst"),
    "e-snap-chart": edgeStatus("snapshot", "chart_analyst"),
    "e-model-bull": edgeStatus("model_analyst", "bull"),
    "e-chart-bull": edgeStatus("chart_analyst", "bull"),
    "e-bull-bear": edgeStatus("bull", "bear"),
    "e-bear-counter": edgeStatus("bear", "bull_counter"),
    "e-counter-risk": edgeStatus("bull_counter", "risk_chair"),
    "e-risk-final": edgeStatus("risk_chair", "final_pick"),
  };

  return { nodeStates, edgeStates, activeBadges, activeCount: Object.values(activeByRole).flat().length };
}

function isViewingActiveRun() {
  if (!state.activeJob) return false;
  if (!state.run) return true;
  if (state.activeJob.hash_prefix && state.run.hash_prefix) {
    return state.activeJob.hash_prefix === state.run.hash_prefix;
  }
  return state.activeJob.as_of === state.run.as_of;
}

// ---------------------------------------------------------------------------
// 渲染函数
// ---------------------------------------------------------------------------

function renderAll() {
  renderTopbarStatus();
  renderWorkflowCard();
  renderRosterList();
  renderTabsHeader();
  if (state.activeTab === "stock") {
    dom.viewStock.hidden = false;
    dom.viewFinal.hidden = true;
    dom.viewModels.hidden = true;
    dom.viewPrompts.hidden = true;
    renderStockDetail();
  } else if (state.activeTab === "final") {
    dom.viewStock.hidden = true;
    dom.viewFinal.hidden = false;
    dom.viewModels.hidden = true;
    dom.viewPrompts.hidden = true;
    renderFinalView();
  } else if (state.activeTab === "models") {
    dom.viewStock.hidden = true;
    dom.viewFinal.hidden = true;
    dom.viewModels.hidden = false;
    dom.viewPrompts.hidden = true;
    renderModelsView();
  } else {
    dom.viewStock.hidden = true;
    dom.viewFinal.hidden = true;
    dom.viewModels.hidden = true;
    dom.viewPrompts.hidden = false;
    renderPromptsView();
  }
  renderChatLayout();
  renderChatContextStrip();
  if (state.chatOpen && !state.chatStreaming) {
    const sess = getActiveChatSession();
    if (!sess || !sess.messages?.length) {
      renderChatMessages({ scrollBottom: false });
    }
  }
}

function renderTopbarStatus() {
  const running = Boolean(state.activeJob?.running);
  dom.btnStart.disabled = running;
  dom.btnDryRun.disabled = running;
  dom.btnStop.hidden = !running;
  dom.btnStop.disabled = Boolean(state.activeJob?.stopping);

  const candidates = state.run?.request?.candidates || [];
  const total = candidates.length;
  const doneCount = candidates.filter((c) => state.stockState[c.ts_code]?.status === "completed").length;

  dom.kpiProgress.textContent = `${doneCount}/${total}`;
  dom.kpiTokens.textContent = fmtTokens(state.run?.usage);

  let badgeText = "就绪";
  let badgeClass = "status-idle";
  if (running) {
    const stageMap = {
      preparing: "构建快照",
      debating: "辩论进行中",
      finalizing: "终选定稿中",
    };
    badgeText = state.activeJob.stopping ? "正在停止…" : stageMap[state.activeJob.stage] || "运行中";
    badgeClass = "status-running";
  } else if (state.isDryRun) {
    badgeText = "演练预览";
    badgeClass = "status-idle";
  } else if (state.run?.final) {
    badgeText = "已冻结";
    badgeClass = "status-done";
  } else if (state.activeJob?.stage === "error") {
    badgeText = "运行出错";
    badgeClass = "status-error";
  } else if (doneCount > 0) {
    badgeText = `断点 ${doneCount}/${total}`;
    badgeClass = "status-idle";
  }

  dom.statusBadge.textContent = badgeText;
  dom.statusBadge.className = `status-badge ${badgeClass}`;
}

function renderWorkflowCard() {
  const { nodeStates, edgeStates, activeBadges, activeCount } = computeWorkflowGraphState();
  const reqFeatureSet = state.run?.request?.context?.feature_set;
  const snapshotSub =
    reqFeatureSet === "S+F69"
      ? "69 特征 · 序列 · 归因"
      : "69 特征 · 归因 · 20 日线";
  renderWorkflowGraph(dom.workflowGraph, {
    nodeStates,
    edgeStates,
    activeBadges,
    snapshotSub,
    onNodeClick: (nodeId) => {
      if (nodeId === "final_pick") {
        switchTab("final");
        return;
      }
      switchTab("stock");
      if (nodeId === "snapshot") {
        dom.stockContent.scrollIntoView({ behavior: "smooth", block: "start" });
      } else {
        const el = document.getElementById(`role-card-${nodeId}`);
        if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    },
  });

  const ctx = state.run?.request?.context;
  const candidates = state.run?.request?.candidates || [];
  const bullCount = candidates.filter((c) => state.stockState[c.ts_code]?.verdict?.verdict === "看多").length;
  const bearCount = candidates.filter((c) => state.stockState[c.ts_code]?.verdict?.verdict === "看空").length;

  dom.wfPoolSize.textContent = ctx ? `${ctx.pool_size} 只` : "—";
  dom.wfLimits.textContent = ctx ? `${ctx.candidates} → ${ctx.final} (每业≤${ctx.max_per_industry})` : "—";
  dom.wfVerdictSplit.textContent = bullCount + bearCount > 0 ? `${bullCount} 多 / ${bearCount} 空` : "—";

  if (activeCount > 0) {
    dom.workflowActivePill.textContent = `流式推进 · ${activeCount} 并发`;
  } else if (state.run?.final) {
    dom.workflowActivePill.textContent = "全流程已定稿";
  } else if (state.isDryRun) {
    dom.workflowActivePill.textContent = "仅快照预览";
  } else {
    dom.workflowActivePill.textContent = "待命";
  }

  const cand = candidates.find((c) => c.ts_code === state.selectedCode);
  dom.workflowSubtitle.textContent = cand
    ? `当前聚焦：#${cand.model.rank} ${cand.ts_code} ${cand.name}（点击节点可定位）`
    : "点击任意节点可快速定位右侧内容";
}

function renderRosterList() {
  const candidates = state.run?.request?.candidates || [];
  if (candidates.length === 0) {
    dom.rosterCountLabel.textContent = "0 只";
    dom.rosterProgressFill.style.width = "0%";
    dom.rosterList.innerHTML = `<div class="empty-hint">暂无候选列表，请选择历史运行或点击演练预览</div>`;
    return;
  }

  const doneCount = candidates.filter((c) => state.stockState[c.ts_code]?.status === "completed").length;
  dom.rosterCountLabel.textContent = `${doneCount} / ${candidates.length} 只`;
  dom.rosterProgressFill.style.width = `${Math.round((doneCount / candidates.length) * 100)}%`;

  const aiPickRank = new Map((state.run?.final?.picks || []).map((p) => [p.ts_code, p.rank]));

  let list = candidates.filter((c) => {
    const st = state.stockState[c.ts_code];
    if (state.filter === "picked") return aiPickRank.has(c.ts_code);
    if (state.filter === "bull") return st?.verdict?.verdict === "看多";
    if (state.filter === "bear") return st?.verdict?.verdict === "看空";
    if (state.filter === "active") return st?.status === "running" || Boolean(st?.activeRole);
    return true;
  });

  if (state.sort === "score") {
    list = [...list].sort((a, b) => {
      const sa = state.stockState[a.ts_code]?.verdict?.score ?? -1;
      const sb = state.stockState[b.ts_code]?.verdict?.score ?? -1;
      return sb - sa || a.model.rank - b.model.rank;
    });
  } else {
    list = [...list].sort((a, b) => a.model.rank - b.model.rank);
  }

  if (list.length === 0) {
    dom.rosterList.innerHTML = `<div class="empty-hint">当前筛选条件下没有股票</div>`;
    return;
  }

  dom.rosterList.innerHTML = list
    .map((c) => {
      const st = state.stockState[c.ts_code] || {};
      const selected = c.ts_code === state.selectedCode ? " selected" : "";
      const pickRank = aiPickRank.get(c.ts_code);

      let verdictHtml = `<span class="verdict-chip verdict-pending">待辩</span>`;
      if (st.verdict) {
        const cls = st.verdict.verdict === "看多" ? "verdict-bull" : "verdict-bear";
        verdictHtml = `<span class="verdict-chip ${cls} mono">${esc(st.verdict.verdict)} ${fmtNum(st.verdict.score, 0)}</span>`;
      } else if (st.activeRole || st.status === "running") {
        const roleMeta = STOCK_ROLES.find((r) => r.id === st.activeRole);
        verdictHtml = `<span class="verdict-chip verdict-running">${esc(roleMeta?.short || "…")} 思考中</span>`;
      } else if (st.status === "failed") {
        verdictHtml = `<span class="verdict-chip verdict-failed">失败</span>`;
      }

      const stepsHtml = STOCK_ROLES.map((r) => {
        let cls = "";
        const isDone =
          (r.phase === "analyst" && Boolean(st.analysts?.[r.id])) ||
          (r.phase === "debate" && Boolean(st.debate?.[r.id])) ||
          (r.phase === "verdict" && Boolean(st.verdict));
        if (st.activeRole === r.id) cls = " running";
        else if (isDone) cls = " done";
        else if (st.roleErrors?.[r.id]) cls = " failed";
        return `<span class="step-dot${cls}" title="${esc(r.name)}">${r.short}</span>`;
      }).join("");

      const flagsHtml = (c.flags || [])
        .map((f) => `<span class="flag-mini" title="${esc(f.text)}">${esc(FLAG_LABELS[f.code] || f.code)}</span>`)
        .join("");

      return `
        <div class="roster-row${selected}" data-code="${esc(c.ts_code)}">
          <div class="roster-row-top">
            <div class="roster-id">
              <span class="rank-num mono">#${String(c.model.rank).padStart(2, "0")}</span>
              <span class="stock-name">${esc(c.name)}</span>
              <span class="stock-code mono">${esc(c.ts_code)}</span>
              <span class="stock-ind">${esc(c.industry)}</span>
            </div>
            <div class="roster-badges">
              ${pickRank ? `<span class="ai-pick-tag mono">AI #${pickRank}</span>` : ""}
              ${verdictHtml}
            </div>
          </div>
          <div class="roster-row-bottom">
            <div class="role-steps">${stepsHtml}</div>
            <div class="flag-mini-list">${flagsHtml}</div>
          </div>
        </div>
      `;
    })
    .join("");

  dom.rosterList.querySelectorAll(".roster-row").forEach((row) => {
    row.addEventListener("click", () => {
      state.selectedCode = row.dataset.code;
      state.chatExcludeSelected = false;
      state.autoFollow = false;
      dom.chkAutofollow.checked = false;
      switchTab("stock");
    });
  });
}

function renderTabsHeader() {
  const candidates = state.run?.request?.candidates || [];
  const cand = candidates.find((c) => c.ts_code === state.selectedCode);
  dom.tabStockTag.textContent = cand ? `${cand.ts_code} ${cand.name}` : "—";
  if (state.run?.final) {
    dom.tabFinalTag.textContent = `前 ${state.run.final.final_count} 名`;
  } else if (state.activeJob?.running && state.activeJob.stage === "finalizing") {
    dom.tabFinalTag.textContent = "终选中…";
  } else {
    dom.tabFinalTag.textContent = "待定稿";
  }
  if (state.promptsData?.differs_from_workspace) {
    dom.tabPromptsTag.textContent = "有变更";
  } else if (state.promptsData?.source === "frozen") {
    dom.tabPromptsTag.textContent = "冻结版";
  } else {
    dom.tabPromptsTag.textContent = "7+1";
  }
  if (state.meta?.ml_available) {
    dom.tabModelsTag.textContent = "3 模型";
  } else {
    dom.tabModelsTag.textContent = "未就绪";
  }
  dom.tabStock.classList.toggle("active", state.activeTab === "stock");
  dom.tabFinal.classList.toggle("active", state.activeTab === "final");
  dom.tabModels.classList.toggle("active", state.activeTab === "models");
  dom.tabPrompts.classList.toggle("active", state.activeTab === "prompts");
}

function switchTab(tab) {
  state.activeTab = tab;
  if (tab === "prompts" && !state.promptsData && !state.promptsLoading) {
    refreshPromptsData();
  }
  if (tab === "models") {
    ensureMlViewLoaded();
  }
  renderAll();
}

// ---------------------------------------------------------------------------
// 右侧视图 A：单股研报与多角色实时流式辩论过程
// ---------------------------------------------------------------------------

function renderCitationsHtml(citations) {
  if (!Array.isArray(citations) || citations.length === 0) return "";
  return `
    <div class="citations-row">
      ${citations
        .map((c) => `<span class="citation-chip mono" data-citation="${esc(c)}">${esc(c)}</span>`)
        .join("")}
    </div>
  `;
}

function renderStreamBoxesHtml(code, role, isLive) {
  const stream = state.streams[`${code}::${role}`];
  if (!stream) {
    if (!isLive) return "";
    return `
      <div class="stream-panel" data-stream-box="${esc(code)}::${esc(role)}">
        <div class="stream-label">
          <span>正在等待模型首字输出…</span>
          <span class="mono" data-elapsed="${esc(code)}::${esc(role)}">0.0s</span>
        </div>
        <div class="stream-content"><span class="caret"></span></div>
      </div>
    `;
  }

  if (isLive) {
    const elapsed = stream.started_at ? `${Math.max(0, Date.now() / 1000 - stream.started_at).toFixed(1)}s` : "";
    const hasText = Boolean(stream.text);
    return `
      <div class="stream-panel" data-stream-box="${esc(code)}::${esc(role)}::thinking">
        <div class="stream-label">
          <span>深度思考流 (Thinking · 第 ${stream.attempt || 1} 次尝试)</span>
          <span class="mono" data-elapsed="${esc(code)}::${esc(role)}">${elapsed}</span>
        </div>
        <div class="stream-content" data-stream-text="${esc(code)}::${esc(role)}::thinking">${esc(stream.thinking || "模型正在组织思路…")}${!hasText ? '<span class="caret"></span>' : ""}</div>
      </div>
      ${
        hasText
          ? `
        <div class="stream-panel stream-panel-text" data-stream-box="${esc(code)}::${esc(role)}::text">
          <div class="stream-label">
            <span>结构化回答输出流 (JSON)</span>
            <span class="mono">${stream.text.length} 字</span>
          </div>
          <div class="stream-content" data-stream-text="${esc(code)}::${esc(role)}::text">${esc(stream.text)}<span class="caret"></span></div>
        </div>
      `
          : ""
      }
    `;
  }

  // 已完成的角色：若本次会话保留了流式思考，提供可展开回看
  if (stream.thinking) {
    return `
      <details class="thinking-archive">
        <summary>查看本次实时思考过程（${stream.thinking.length} 字）</summary>
        <pre>${esc(stream.thinking)}</pre>
      </details>
    `;
  }
  return "";
}

function renderAnalystCard(code, roleId, title, data, st) {
  const isLive = st.activeRole === roleId;
  const err = st.roleErrors?.[roleId];
  let stanceHtml = `<span class="stance-badge stance-neutral">${isLive ? "思考中…" : "待执行"}</span>`;
  if (data) {
    const map = { bull: ["偏多 bull", "stance-bull"], bear: ["偏空 bear", "stance-bear"], neutral: ["中性 neutral", "stance-neutral"] };
    const [label, cls] = map[data.stance] || [data.stance, "stance-neutral"];
    stanceHtml = `<span class="stance-badge ${cls}">${esc(label)}</span>`;
  }

  let bodyHtml = "";
  if (isLive) {
    bodyHtml = renderStreamBoxesHtml(code, roleId, true);
  } else if (data) {
    bodyHtml = `
      <div class="role-body">${esc(data.conclusion)}</div>
      ${
        data.risks?.length
          ? `<ul class="risks-list">${data.risks.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>`
          : ""
      }
      ${renderCitationsHtml(data.citations)}
      ${renderStreamBoxesHtml(code, roleId, false)}
    `;
  } else if (err) {
    bodyHtml = `<div class="pending-placeholder" style="color:var(--bull)">失败：${esc(err)}</div>`;
  } else {
    bodyHtml = `<div class="pending-placeholder">等待该角色进入执行…</div>`;
  }

  const candName = (state.run?.request?.candidates || []).find((c) => c.ts_code === code)?.name || code;
  const promptBtnHtml = `<button class="role-prompt-btn" type="button" data-inspect-role="${esc(roleId)}" data-inspect-code="${esc(code)}" title="查看该角色的预置提示词、完整 System Prompt 与本股实际 User JSON">提示词</button>`;
  const askBtnHtml = data
    ? `<button class="role-ask-btn" type="button" data-ask-code="${esc(code)}" data-ask-role="${esc(roleId)}" data-ask-prompt="${esc(`针对 ${candName}(${code}) 的${title}结论，进一步解释你的核心判断依据和最值得警惕的风险点。`)}" title="打开 AI 对话栏以该角色视角追问">✦ 追问</button>`
    : "";

  return `
    <article class="role-card${isLive ? " role-live" : ""}" id="role-card-${roleId}">
      <div class="role-card-head">
        <div class="role-meta">
          <span class="role-name">${esc(title)}</span>
          <span class="role-code mono">${roleId}</span>
        </div>
        <div class="role-head-right">
          ${promptBtnHtml}
          ${askBtnHtml}
          ${stanceHtml}
        </div>
      </div>
      ${bodyHtml}
    </article>
  `;
}

function renderDebateRoundCard(code, roleId, title, badgeText, accentClass, data, st) {
  const isLive = st.activeRole === roleId;
  const err = st.roleErrors?.[roleId];

  let bodyHtml = "";
  if (isLive) {
    bodyHtml = renderStreamBoxesHtml(code, roleId, true);
  } else if (data) {
    bodyHtml = `
      <div class="role-body">${esc(data.argument)}</div>
      ${renderCitationsHtml(data.citations)}
      ${renderStreamBoxesHtml(code, roleId, false)}
    `;
  } else if (err) {
    bodyHtml = `<div class="pending-placeholder" style="color:var(--bull)">失败：${esc(err)}</div>`;
  } else {
    bodyHtml = `<div class="pending-placeholder">等待前序发言结束后开始…</div>`;
  }

  const candName = (state.run?.request?.candidates || []).find((c) => c.ts_code === code)?.name || code;
  const promptBtnHtml = `<button class="role-prompt-btn" type="button" data-inspect-role="${esc(roleId)}" data-inspect-code="${esc(code)}" title="查看该角色的预置提示词、完整 System Prompt 与本股实际 User JSON">提示词</button>`;
  const askBtnHtml = data
    ? `<button class="role-ask-btn" type="button" data-ask-code="${esc(code)}" data-ask-role="${esc(roleId)}" data-ask-prompt="${esc(`围绕 ${candName}(${code}) 在「${title}」中的核心论点，帮我展开检验它有哪些站得住与站不住的地方。`)}" title="打开 AI 对话栏追问该轮攻防">✦ 追问</button>`
    : "";

  return `
    <article class="role-card ${accentClass}${isLive ? " role-live" : ""}" id="role-card-${roleId}">
      <div class="role-card-head">
        <div class="role-meta">
          <span class="role-name">${esc(title)}</span>
          <span class="role-code mono">${roleId}</span>
        </div>
        <div class="role-head-right">
          ${promptBtnHtml}
          ${askBtnHtml}
          <span class="pill pill-soft">${isLive ? "实时推理中…" : esc(badgeText)}</span>
        </div>
      </div>
      ${bodyHtml}
    </article>
  `;
}

function renderRiskChairCard(code, verdict, st) {
  const isLive = st.activeRole === "risk_chair";
  const err = st.roleErrors?.risk_chair;

  let headBadge = `<span class="stance-badge stance-neutral">${isLive ? "横向裁决中…" : "待裁决"}</span>`;
  let bodyHtml = "";

  if (isLive) {
    bodyHtml = renderStreamBoxesHtml(code, "risk_chair", true);
  } else if (verdict) {
    const isBull = verdict.verdict === "看多";
    headBadge = `<span class="stance-badge ${isBull ? "stance-bull" : "stance-bear"} mono">${esc(verdict.verdict)} · ${fmtNum(verdict.score, 0)} 分</span>`;
    bodyHtml = `
      <div class="score-bar-wrap">
        <span class="muted-text">相对吸引力</span>
        <div class="score-bar-track">
          <div class="score-bar-fill ${isBull ? "bull" : "bear"}" style="width:${Math.max(0, Math.min(100, verdict.score))}%"></div>
        </div>
        <span class="mono" style="font-weight:700">${fmtNum(verdict.score, 0)} / 100</span>
      </div>
      <div class="role-body" style="font-weight:600;color:var(--ink-primary)">${esc(verdict.thesis)}</div>
      <div class="action-callout">
        <strong>开盘前操作提示</strong>
        <span>${esc(verdict.action)}</span>
      </div>
      ${
        verdict.risks?.length
          ? `<ul class="risks-list">${verdict.risks.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>`
          : ""
      }
      ${renderCitationsHtml(verdict.citations)}
      ${renderStreamBoxesHtml(code, "risk_chair", false)}
    `;
  } else if (err) {
    bodyHtml = `<div class="pending-placeholder" style="color:var(--bull)">失败：${esc(err)}</div>`;
  } else {
    bodyHtml = `<div class="pending-placeholder">等待三轮辩论完成后由风控主席结合 20 股 roster 拍板…</div>`;
  }

  const candName = (state.run?.request?.candidates || []).find((c) => c.ts_code === code)?.name || code;
  const promptBtnHtml = `<button class="role-prompt-btn" type="button" data-inspect-role="risk_chair" data-inspect-code="${esc(code)}" title="查看风控主席的预置提示词、完整 System Prompt 与本股实际 User JSON（含 roster）">提示词</button>`;
  const askBtnHtml = verdict
    ? `<button class="role-ask-btn" type="button" data-ask-code="${esc(code)}" data-ask-role="risk_chair" data-ask-prompt="${esc(`针对 ${candName}(${code}) 给出的「${verdict.verdict} ${fmtNum(verdict.score, 0)} 分」裁决，请说明它在全部 20 只送辩股中的横向位置，以及 T+1 开盘前具体什么条件下放弃买入。`)}" title="向风控主席追问打分与操作条件">✦ 追问裁决</button>`
    : "";

  return `
    <article class="role-card role-chair-accent${isLive ? " role-live" : ""}" id="role-card-risk_chair">
      <div class="role-card-head">
        <div class="role-meta">
          <span class="role-name">风控主席最终裁决</span>
          <span class="role-code mono">risk_chair · 对照全部送辩 roster</span>
        </div>
        <div class="role-head-right">
          ${promptBtnHtml}
          ${askBtnHtml}
          ${headBadge}
        </div>
      </div>
      ${bodyHtml}
    </article>
  `;
}

function renderStockDetail() {
  const candidates = state.run?.request?.candidates || [];
  const cand = candidates.find((c) => c.ts_code === state.selectedCode);
  if (!cand) {
    dom.stockEmpty.hidden = false;
    dom.stockContent.hidden = true;
    return;
  }

  dom.stockEmpty.hidden = true;
  dom.stockContent.hidden = false;

  const st = state.stockState[cand.ts_code] || { analysts: {}, debate: {}, verdict: null };
  const f = cand.features || {};
  const fp = cand.feature_pct || {};
  const aiPick = (state.run?.final?.picks || []).find((p) => p.ts_code === cand.ts_code);

  let heroVerdictHtml = `<div class="score-pill-lg verdict-pending"><span class="score-dir">${st.activeRole ? "辩论中" : "待裁决"}</span></div>`;
  if (st.verdict) {
    const cls = st.verdict.verdict === "看多" ? "verdict-bull" : "verdict-bear";
    heroVerdictHtml = `
      <div class="score-pill-lg ${cls}">
        <span class="score-dir">${esc(st.verdict.verdict)}</span>
        <span class="score-val mono">${fmtNum(st.verdict.score, 0)}</span>
      </div>
    `;
  }

  const metricClass = (v) => (typeof v === "number" ? (v > 0 ? "val-up" : v < 0 ? "val-down" : "") : "");
  const keyMetrics = [
    ["信号日涨跌", fmtPct(f.pct_chg), fp.pct_chg !== undefined ? `全池 ${fp.pct_chg} 分位` : "", metricClass(f.pct_chg)],
    ["近 5 日", fmtPct(f.r5), fp.r5 !== undefined ? `全池 ${fp.r5} 分位` : "", metricClass(f.r5)],
    ["近 20 日", fmtPct(f.r20), fp.r20 !== undefined ? `全池 ${fp.r20} 分位` : "", metricClass(f.r20)],
    ["距 60 日低点", fmtPct(f.from_low60), `距高 ${fmtPct(f.from_high60)}`, ""],
    ["偏离 MA20", fmtPct(f.dev_ma20), `斜率 ${fmtPct(f.ma20_slope5)}`, metricClass(f.dev_ma20)],
    ["5 日量比", f.vol5 !== undefined ? `${f.vol5} 倍` : "—", `20日量比 ${f.vol20 ?? "—"}`, ""],
  ];

  const extraF = cand.extra_features || null;
  const extraFp = cand.extra_feature_pct || {};
  const hasExtra = Boolean(extraF && Object.keys(extraF).length > 0);
  const totalFeatCount = Object.keys(f).length + (hasExtra ? Object.keys(extraF).length : 0);
  const runAsOf = state.run?.as_of || state.run?.request?.as_of || state.selectedDate || "";

  // 触发懒加载三模型归因（在 2023-01-03 ~ 2026-10-16 测试段内即时展示三模型完整归因与百分位对照）
  if (state.meta?.ml_available && runAsOf) {
    ensureStockExplain(runAsOf, cand.ts_code);
  }
  const attributionHtml = renderStockAttributionCardHtml(cand, runAsOf);
  const featuresDrawerHtml = renderStockFeaturesDrawerHtml(cand, f, fp, extraF, extraFp, totalFeatCount);

  const lastBar = cand.bars?.[cand.bars.length - 1];
  const ctxModelLabel = state.run?.request?.context?.model_label || "反弹模型";
  const ctxFeatureSet = state.run?.request?.context?.feature_set || (hasExtra ? "F69" : "F28");

  dom.stockContent.innerHTML = `
    <!-- 个股研报头卡 -->
    <section class="card stock-hero">
      <div class="stock-hero-top">
        <div>
          <div class="stock-hero-title">
            <h1>${esc(cand.name)}</h1>
            <span class="stock-hero-code mono">${esc(cand.ts_code)}</span>
            <span class="pill pill-soft mono">${esc(ctxModelLabel)} · ${esc(ctxFeatureSet)}</span>
            ${aiPick ? `<span class="ai-pick-tag mono">★ AI 终选第 ${aiPick.rank} 名</span>` : ""}
          </div>
          <div class="hero-tags">
            <span class="tag-neutral">行业：${esc(cand.industry)}（池内 ${cand.industry_stats.pool_count} 只 · 行业第 ${cand.industry_stats.rank_in_industry}）</span>
            <span class="tag-neutral mono">模型第 ${cand.model.rank} 名 · ${cand.model.pct} 分位 · 分数 ${cand.model.score}</span>
            ${cand.model.in_model_picks ? `<span class="tag-pick">在模型原前 6 名单内</span>` : `<span class="tag-neutral">未入模型原前 6</span>`}
          </div>
        </div>
        <div class="hero-verdict-box">${heroVerdictHtml}</div>
      </div>

      ${
        cand.flags?.length
          ? `<div class="flags-banner">${cand.flags
              .map((fl) => `<div class="flag-alert"><strong>${esc(FLAG_LABELS[fl.code] || fl.code)}</strong><span>${esc(fl.text)}</span></div>`)
              .join("")}</div>`
          : ""
      }

      <div class="stock-metrics-grid">
        ${keyMetrics
          .map(
            ([label, val, sub, cls]) => `
          <div class="metric-box">
            <div class="metric-label">${esc(label)}</div>
            <div class="metric-value mono ${cls}">${esc(val)}</div>
            <div class="metric-sub mono">${esc(sub)}</div>
          </div>
        `,
          )
          .join("")}
      </div>

      <div class="chart-panel">
        <div class="chart-header">
          <span>近 ${cand.bars.length} 个交易日前复权日线 · 成交额 · MACD（蓝色底带为信号日）</span>
          <div class="chart-hover-readout mono" id="kline-readout"></div>
        </div>
        <div id="kline-container"></div>
      </div>

      ${attributionHtml}
      ${featuresDrawerHtml}
    </section>

    <!-- 阶段一：独立双视角分析 -->
    <div class="section-header">
      <span class="section-badge">阶段一</span>
      <h3 class="section-title">独立双视角分析</h3>
      <span class="section-note">两位分析师互相不可见，防止观点锚定</span>
    </div>
    <div class="analysts-grid">
      ${renderAnalystCard(cand.ts_code, "model_analyst", "模型解读员", st.analysts?.model_analyst, st)}
      ${renderAnalystCard(cand.ts_code, "chart_analyst", "走势分析员", st.analysts?.chart_analyst, st)}
    </div>

    <!-- 阶段二：三轮公开攻防辩论 -->
    <div class="section-header">
      <span class="section-badge">阶段二</span>
      <h3 class="section-title">公开三轮攻防辩论</h3>
      <span class="section-note">多方立论 → 空方逐条反驳 → 多方二次回应与风险承认</span>
    </div>
    <div class="debate-stack">
      ${renderDebateRoundCard(cand.ts_code, "bull", "第 1 轮 · 多方辩手立论", "多方立论", "role-bull-accent", st.debate?.bull, st)}
      ${renderDebateRoundCard(cand.ts_code, "bear", "第 2 轮 · 空方辩手反驳", "逐条反驳多方", "role-bear-accent", st.debate?.bear, st)}
      ${renderDebateRoundCard(cand.ts_code, "bull_counter", "第 3 轮 · 多方反驳手回应", "逐条回应空方", "role-bull-accent", st.debate?.bull_counter, st)}
    </div>

    <!-- 阶段三：风控主席拍板 -->
    <div class="section-header">
      <span class="section-badge">阶段三</span>
      <h3 class="section-title">风控主席横向裁决</h3>
      <span class="section-note">结合全部 20 只送辩简表 roster 给出看多/看空与 0~100 相对评分</span>
    </div>
    ${renderRiskChairCard(cand.ts_code, st.verdict, st)}
  `;

  // 绘制 SVG K 线图
  const klineContainer = document.getElementById("kline-container");
  const readoutEl = document.getElementById("kline-readout");
  const updateReadout = (bar) => {
    if (!bar || !readoutEl) return;
    const chgCls = bar.pct_chg > 0 ? "val-up" : bar.pct_chg < 0 ? "val-down" : "";
    const lim = bar.limit === "up" ? " [涨停]" : bar.limit === "down" ? " [跌停]" : "";
    readoutEl.innerHTML = `
      <span>${esc(bar.date)}</span>
      <span>开 ${fmtNum(bar.open)} 高 ${fmtNum(bar.high)} 低 ${fmtNum(bar.low)} 收 <strong>${fmtNum(bar.close)}</strong></span>
      <span class="${chgCls}">${fmtPct(bar.pct_chg)}${lim}</span>
      <span>额 ${fmtNum(bar.amount_yi)}亿</span>
      <span>量比 ${bar.vol5 ?? "—"}</span>
      <span>柱 ${bar.hist_ratio ?? "—"}</span>
    `;
  };
  renderStockKlineChart(klineContainer, cand.bars, updateReadout);
  updateReadout(lastBar);
}

// ---------------------------------------------------------------------------
// 右侧视图 B：最终辩论结果总览（AI 前 6 vs 模型前 6 + 20 股评分分布）
// ---------------------------------------------------------------------------

function renderFinalView() {
  const final = state.run?.final;
  const req = state.run?.request;
  const finalStream = state.streams["-::final_pick"];
  const isFinalizing = Boolean(
    state.activeJob?.running &&
      isViewingActiveRun() &&
      (state.activeJob.stage === "finalizing" || (finalStream && !finalStream.completed)),
  );

  if (!final && !isFinalizing) {
    dom.finalContent.innerHTML = `
      <div class="empty-state">
        <div class="empty-title">终选辩论结果尚未生成</div>
        <p class="empty-desc">等待全部候选完成 6 轮辩论并执行终选官 final_pick 后，这里将呈现 AI 重排前 6 名与原模型前 6 名的完整对照。</p>
        <div style="margin-top:12px">
          <button class="role-prompt-btn" type="button" data-inspect-role="final_pick" data-inspect-code="-">查看 final_pick 提示词与入参结构</button>
        </div>
      </div>
    `;
    return;
  }

  let html = "";

  if (isFinalizing) {
    html += `
      <article class="role-card role-chair-accent role-live" style="margin-bottom:14px">
        <div class="role-card-head">
          <div class="role-meta">
            <span class="role-name">终选官正在汇总全局风控结论…</span>
            <span class="role-code mono">final_pick</span>
          </div>
          <div class="role-head-right">
            <button class="role-prompt-btn" type="button" data-inspect-role="final_pick" data-inspect-code="-">提示词</button>
            <span class="pill pill-soft">流式推理中</span>
          </div>
        </div>
        ${renderStreamBoxesHtml("-", "final_pick", true)}
      </article>
    `;
  }

  if (final) {
    const modelCodes = new Set((final.model_picks || []).map((m) => m.ts_code));
    const aiCodes = new Set((final.picks || []).map((p) => p.ts_code));
    const keptCount = (final.picks || []).filter((p) => modelCodes.has(p.ts_code)).length;
    const addedCount = (final.picks || []).length - keptCount;
    const maxPerInd = req?.limits?.max_per_industry ?? 2;
    const indCounts = {};
    for (const p of final.picks || []) indCounts[p.industry] = (indCounts[p.industry] || 0) + 1;
    const maxIndActual = Math.max(0, ...Object.values(indCounts));

    html += `
      <section class="card final-summary-card">
        <div class="stock-hero-top">
          <div>
            <div class="stock-hero-title">
              <h1>${esc(final.as_of)} AI 辩论终选结果</h1>
              <span class="stock-hero-code mono">${esc(final.run_id)}</span>
            </div>
            <div class="hero-tags">
              <span class="tag-neutral">完成时间：${esc(final.completed_at || "—")}</span>
              <span class="tag-neutral">同行业上限：每业 ≤ ${maxPerInd} 只（实选最高 ${maxIndActual} 只）</span>
              <span class="tag-neutral mono">调用 ${final.usage?.calls ?? 0} 次（失败 ${final.usage?.failed_calls ?? 0}）· ${fmtTokens(final.usage)}</span>
            </div>
          </div>
          <div class="role-head-right">
            <button
              class="role-prompt-btn"
              type="button"
              data-inspect-role="final_pick"
              data-inspect-code="-"
              title="查看终选人 final_pick 的预置提示词、完整 System Prompt 与实际终选 User JSON"
            >
              提示词
            </button>
            <button
              class="role-ask-btn"
              type="button"
              data-ask-code=""
              data-ask-role="final_pick"
              data-ask-prompt="${esc(`基于 ${final.as_of} 的终选结果，请详细拆解为什么换掉了未入选的原模型前 6 股票，以及新进替换标的的相对优势与行业上限取舍。`)}"
              title="向终选官追问全局重排理由"
            >
              ✦ 追问终选
            </button>
            <span class="status-badge status-done">已冻结定稿</span>
          </div>
        </div>

        <div class="final-kpi-row">
          <div class="metric-box">
            <div class="metric-label">辩论覆盖</div>
            <div class="metric-value mono">${final.debated_count} / ${final.submitted_count} 只</div>
            <div class="metric-sub">失败 ${final.failed_count} 只</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">终选名额</div>
            <div class="metric-value mono">${final.final_count} 只</div>
            <div class="metric-sub">${esc(final.shortage_reason || "满额选出")}</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">保留模型原名单</div>
            <div class="metric-value mono">${keptCount} 只</div>
            <div class="metric-sub">重合度 ${keptCount}/${(final.model_picks || []).length}</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">AI 新进替换</div>
            <div class="metric-value mono">${addedCount} 只</div>
            <div class="metric-sub">换出 ${(final.model_picks || []).length - keptCount} 只</div>
          </div>
        </div>

        <div class="final-reason-box">
          <strong>终选整体取舍理由：</strong>${esc(final.reason)}
        </div>
        ${renderStreamBoxesHtml("-", "final_pick", false)}
      </section>

      <div class="final-compare-grid" style="margin-top:14px">
        <section class="card" style="padding:14px 16px">
          <div class="card-head" style="padding:0 0 10px">
            <div>
              <h2 class="card-title">AI 重排终选前 ${final.final_count} 名</h2>
              <p class="card-subtitle">点击卡片可直达该股完整辩论过程</p>
            </div>
          </div>
          ${(final.picks || [])
            .map((p) => {
              const isBull = p.verdict === "看多";
              const kept = modelCodes.has(p.ts_code);
              const st = state.stockState[p.ts_code];
              const action = st?.verdict?.action || "";
              return `
                <div class="pick-card" data-jump-code="${esc(p.ts_code)}">
                  <div class="pick-card-top">
                    <div class="roster-id">
                      <span class="pick-rank-badge mono">AI #${p.rank}</span>
                      <span class="stock-name">${esc(p.name)}</span>
                      <span class="stock-code mono">${esc(p.ts_code)}</span>
                      <span class="stock-ind">${esc(p.industry)}</span>
                    </div>
                    <div class="roster-badges">
                      <span class="delta-chip ${kept ? "delta-kept" : "delta-added"} mono">模型 #${p.model_rank} · ${kept ? "保留" : "新进"}</span>
                      <span class="verdict-chip ${isBull ? "verdict-bull" : "verdict-bear"} mono">${esc(p.verdict)} ${fmtNum(p.score, 0)}</span>
                    </div>
                  </div>
                  <div class="role-body" style="font-size:12.5px">${esc(p.reason)}</div>
                  ${action ? `<div class="muted-text">开盘提示：${esc(action)}</div>` : ""}
                </div>
              `;
            })
            .join("")}
        </section>

        <section class="card" style="padding:14px 16px">
          <div class="card-head" style="padding:0 0 10px">
            <div>
              <h2 class="card-title">原模型前 ${(final.model_picks || []).length} 名对照</h2>
              <p class="card-subtitle">直观查看 AI 保留与淘汰了哪些票</p>
            </div>
          </div>
          ${(final.model_picks || [])
            .map((m) => {
              const kept = aiCodes.has(m.ts_code);
              const v = state.stockState[m.ts_code]?.verdict;
              return `
                <div class="model-pick-row" data-jump-code="${esc(m.ts_code)}">
                  <div class="roster-id">
                    <span class="rank-num mono">#${m.rank}</span>
                    <span class="stock-name">${esc(m.name)}</span>
                    <span class="stock-code mono">${esc(m.ts_code)}</span>
                  </div>
                  <div class="roster-badges">
                    ${v ? `<span class="verdict-chip ${v.verdict === "看多" ? "verdict-bull" : "verdict-bear"} mono">${esc(v.verdict)} ${fmtNum(v.score, 0)}</span>` : ""}
                    <span class="delta-chip ${kept ? "delta-kept" : "delta-dropped"}">${kept ? "保留" : "换出"}</span>
                  </div>
                </div>
              `;
            })
            .join("")}
        </section>
      </div>
    `;

    // 20 只评分分布总览
    const allVerdicts = (req?.candidates || [])
      .map((c) => ({ cand: c, verdict: state.stockState[c.ts_code]?.verdict }))
      .filter((x) => x.verdict)
      .sort((a, b) => b.verdict.score - a.verdict.score || a.cand.model.rank - b.cand.model.rank);

    if (allVerdicts.length > 0) {
      html += `
        <section class="card" style="margin-top:14px">
          <div class="card-head">
            <div>
              <h2 class="card-title">全部 ${allVerdicts.length} 只送辩候选风控评分分布</h2>
              <p class="card-subtitle">看多参考区间 55~90，看空参考区间 10~45；点击任意一行可回看辩论原文</p>
            </div>
          </div>
          <div class="dist-list">
            ${allVerdicts
              .map(({ cand, verdict }) => {
                const isBull = verdict.verdict === "看多";
                const picked = aiCodes.has(cand.ts_code);
                return `
                  <div class="dist-row" data-jump-code="${esc(cand.ts_code)}">
                    <div class="dist-label">
                      <span class="mono muted-text">#${String(cand.model.rank).padStart(2, "0")}</span>
                      <strong>${esc(cand.name)}</strong>
                      ${picked ? `<span class="ai-pick-tag" style="font-size:9.5px">选</span>` : ""}
                    </div>
                    <div class="dist-track">
                      <div class="dist-fill ${isBull ? "bull" : "bear"}" style="width:${Math.max(2, Math.min(100, verdict.score))}%"></div>
                    </div>
                    <div class="dist-score mono ${isBull ? "val-up" : "val-down"}">${esc(verdict.verdict)} ${fmtNum(verdict.score, 0)}</div>
                  </div>
                `;
              })
              .join("")}
          </div>
        </section>
      `;
    }
  }

  dom.finalContent.innerHTML = html;
  dom.finalContent.querySelectorAll("[data-jump-code]").forEach((el) => {
    el.addEventListener("click", () => {
      state.selectedCode = el.dataset.jumpCode;
      switchTab("stock");
    });
  });
}

// ---------------------------------------------------------------------------
// 增量打字机 DOM 局部更新（高频 role.delta 不触发整页重绘，零闪烁）
// ---------------------------------------------------------------------------

function patchLiveDeltaDom(code, role, channel) {
  const key = `${code}::${role}`;
  const stream = state.streams[key];
  if (!stream) return;

  const isVisible =
    (state.activeTab === "stock" && code === state.selectedCode) ||
    (state.activeTab === "final" && code === "-" && role === "final_pick");
  if (!isVisible) return;

  const targetEl = document.querySelector(`[data-stream-text="${CSS.escape(`${key}::${channel}`)}"]`);
  if (!targetEl) {
    // 刚从 thinking 切到 text 通道或首个字符到来，局部重绘当前视图即可创建对应容器
    if (state.activeTab === "stock") renderStockDetail();
    else renderFinalView();
    return;
  }

  const rawText = channel === "thinking" ? stream.thinking : stream.text;
  const showCaret = channel === "text" || !stream.text;
  targetEl.innerHTML = `${esc(rawText)}${showCaret ? '<span class="caret"></span>' : ""}`;
  targetEl.scrollTop = targetEl.scrollHeight;
}

function tickElapsedTimers() {
  document.querySelectorAll("[data-elapsed]").forEach((el) => {
    const key = el.dataset.elapsed;
    const stream = state.streams[key];
    if (stream?.started_at && !stream.completed) {
      el.textContent = `${Math.max(0, Date.now() / 1000 - stream.started_at).toFixed(1)}s`;
    }
  });
  if (state.chatStreaming && state.chatLiveTurn?.started_at) {
    const el = document.getElementById("chat-live-elapsed");
    if (el) {
      el.textContent = `${Math.max(0, Date.now() / 1000 - state.chatLiveTurn.started_at).toFixed(1)}s`;
    }
  }
}

// ---------------------------------------------------------------------------
// SSE 实时事件流处理
// ---------------------------------------------------------------------------

function connectSse() {
  if (state.eventSource) {
    state.eventSource.close();
  }
  const es = new EventSource("/api/stream");
  state.eventSource = es;

  const bind = (eventName, handler) => {
    es.addEventListener(eventName, (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        handler(msg.payload || {});
      } catch {
        // 忽略格式异常帧
      }
    });
  };

  bind("status", (payload) => {
    state.activeJob = payload.active || null;
    renderTopbarStatus();
  });

  bind("job.started", (payload) => {
    state.activeJob = payload;
    state.streams = {};
    showToast(`已启动 ${payload.as_of} 辩论（${payload.model} · ${payload.reasoning_effort} · 并发 ${payload.concurrency}）`, "info", 6000);
    renderTopbarStatus();
    renderWorkflowCard();
  });

  bind("job.stopping", (payload) => {
    state.activeJob = payload;
    showToast("正在发送停止信号，已辩完股票的检查点会完整保留…", "warn", 6000);
    renderTopbarStatus();
  });

  bind("run.prepared", async (payload) => {
    if (state.activeJob) {
      state.activeJob.stage = "debating";
      state.activeJob.hash_prefix = payload.hash_prefix;
      state.activeJob.run_id = payload.run_id;
    }
    if (payload.model_source) {
      state.selectedModelSource = payload.model_source;
    }
    await refreshMeta({ keepSelection: true });
    await loadRunDetail(payload.as_of, payload.hash_prefix, { modelSource: payload.model_source });
  });

  bind("stock.started", (payload) => {
    if (!isViewingActiveRun()) return;
    const st = state.stockState[payload.ts_code];
    if (st) st.status = "running";
    if (state.autoFollow && payload.ts_code && !state.stockState[state.selectedCode]?.activeRole) {
      state.selectedCode = payload.ts_code;
    }
    renderRosterList();
    renderWorkflowCard();
  });

  bind("role.started", (payload) => {
    if (!isViewingActiveRun()) return;
    const code = payload.ts_code || "-";
    const role = payload.role;
    const key = `${code}::${role}`;
    state.streams[key] = {
      ts_code: code,
      role,
      attempt: payload.attempt || 1,
      thinking: "",
      text: "",
      started_at: Date.now() / 1000,
      completed: false,
    };

    if (role === "final_pick") {
      if (state.activeJob) state.activeJob.stage = "finalizing";
      if (state.autoFollow) state.activeTab = "final";
    } else if (state.stockState[code]) {
      state.stockState[code].status = "running";
      state.stockState[code].activeRole = role;
      state.stockState[code].attempt = payload.attempt || 1;
      if (state.autoFollow && state.activeTab === "stock") {
        // 如果当前看的票已经完成，自动切到新开跑的票
        const curSt = state.stockState[state.selectedCode];
        if (!curSt || curSt.status === "completed" || curSt.status === "pending") {
          state.selectedCode = code;
        }
      }
    }
    renderAll();
  });

  bind("role.delta", (payload) => {
    if (!isViewingActiveRun()) return;
    const code = payload.ts_code || "-";
    const role = payload.role;
    const channel = payload.channel || "text";
    const key = `${code}::${role}`;
    if (!state.streams[key]) {
      state.streams[key] = {
        ts_code: code,
        role,
        attempt: payload.attempt || 1,
        thinking: "",
        text: "",
        started_at: Date.now() / 1000,
        completed: false,
      };
    }
    if (channel === "thinking") state.streams[key].thinking += payload.delta || "";
    else state.streams[key].text += payload.delta || "";
    patchLiveDeltaDom(code, role, channel);
  });

  bind("role.completed", (payload) => {
    if (!isViewingActiveRun()) return;
    const code = payload.ts_code || "-";
    const role = payload.role;
    const stream = state.streams[`${code}::${role}`];
    if (stream) stream.completed = true;
    const st = state.stockState[code];
    if (st && payload.data) {
      applyRoleCompletedToStock(st, role, payload.data);
      if (st.activeRole === role) st.activeRole = null;
    }
    renderAll();
  });

  bind("role.failed", (payload) => {
    if (!isViewingActiveRun()) return;
    const st = state.stockState[payload.ts_code];
    if (st) {
      st.roleErrors[payload.role] = `${payload.error || "失败"}${payload.retrying ? "（正在重试）" : ""}`;
    }
    showToast(`${payload.ts_code} ${payload.role} 失败：${payload.error}${payload.retrying ? "（自动重试中）" : ""}`, "warn", 5000);
    renderRosterList();
    if (state.selectedCode === payload.ts_code && state.activeTab === "stock") {
      renderStockDetail();
    }
  });

  bind("usage.updated", (payload) => {
    if (!isViewingActiveRun() || !state.run) return;
    if (payload.usage) {
      state.run.usage = payload.usage;
      renderTopbarStatus();
    }
  });

  bind("stock.completed", (payload) => {
    if (!isViewingActiveRun()) return;
    const st = state.stockState[payload.ts_code];
    if (st && payload.stock) {
      st.status = "completed";
      st.activeRole = null;
      for (const a of payload.stock.analysts || []) st.analysts[a.role] = a;
      for (const d of payload.stock.debate || []) st.debate[d.round] = d;
      if (payload.stock.verdict) st.verdict = payload.stock.verdict;
    }
    if (payload.usage && state.run) state.run.usage = payload.usage;
    renderAll();
  });

  bind("stock.failed", (payload) => {
    if (!isViewingActiveRun()) return;
    const st = state.stockState[payload.ts_code];
    if (st) {
      st.status = "failed";
      st.activeRole = null;
      st.failedError = payload.error || "未完成";
    }
    renderAll();
  });

  bind("run.frozen", async (payload) => {
    if (state.run && payload.final) {
      state.run.final = payload.final;
      state.run.completed = true;
      state.run.usage = payload.final.usage || state.run.usage;
    }
    state.activeTab = "final";
    await refreshMeta({ keepSelection: true });
    renderAll();
  });

  bind("job.completed", async (payload) => {
    state.activeJob = payload;
    if (payload.model_source) {
      state.selectedModelSource = payload.model_source;
    }
    showToast(payload.cached ? "命中已冻结运行，已直接载入终选结果。" : "全量辩论与终选已完成并冻结！", "info", 6000);
    await refreshMeta({ keepSelection: true });
    if (payload.as_of && payload.hash_prefix) {
      await loadRunDetail(payload.as_of, payload.hash_prefix, { modelSource: payload.model_source });
      state.activeTab = "final";
      renderAll();
    }
  });

  bind("job.stopped", async (payload) => {
    state.activeJob = payload;
    showToast(payload.error || "已停止辩论，已完成股票检查点已保存，可随时续跑。", "warn", 7000);
    await refreshMeta({ keepSelection: true });
    renderAll();
  });

  bind("job.error", async (payload) => {
    state.activeJob = payload;
    showToast(`辩论中断：${payload.error || "未知错误"}`, "error", 0);
    await refreshMeta({ keepSelection: true });
    renderAll();
  });
}

// ---------------------------------------------------------------------------
// 数据加载与交互动作
// ---------------------------------------------------------------------------

async function refreshMeta({ keepSelection = false } = {}) {
  const qs = new URLSearchParams();
  if (state.selectedDb) qs.set("db", state.selectedDb);
  if (state.selectedModelSource) qs.set("model_source", state.selectedModelSource);
  const meta = await apiJson(`/api/meta?${qs.toString()}`);
  state.meta = meta;
  state.selectedDb = meta.selected_db;
  state.selectedModelSource = meta.model_source || "xgb_F69";
  state.activeJob = meta.active || null;

  // 顶栏模型摘要
  const cfg = meta.config;
  const activeModelMeta = (meta.model_sources || []).find((m) => m.id === state.selectedModelSource);
  const srcTag = activeModelMeta ? `${activeModelMeta.name} (${activeModelMeta.feature_set})` : state.selectedModelSource;
  dom.modelMetaLine.textContent = `${srcTag} · ${cfg.model} · 思考 ${cfg.reasoning_effort} · 并发 ${cfg.concurrency} · ${cfg.api_key_configured ? "密钥已就绪" : "缺少密钥"}`;

  // 填充数据库下拉
  dom.selDb.innerHTML = meta.dbs
    .map((d) => `<option value="${esc(d.path)}">${esc(d.label)}</option>`)
    .join("");
  dom.selDb.value = state.selectedDb;

  // 填充选股模型源下拉
  if (dom.selModelSource) {
    const sources = meta.model_sources || [];
    dom.selModelSource.innerHTML = sources
      .map((m) => {
        const countTag = m.run_count > 0 ? ` (${m.run_count}条历史)` : "";
        return `<option value="${esc(m.id)}">${esc(m.name)}${countTag}</option>`;
      })
      .join("");
    dom.selModelSource.value = state.selectedModelSource;
  }

  // 填充默认参数（首次）
  if (!keepSelection) {
    dom.selEffort.value = cfg.reasoning_effort;
    dom.selConcurrency.value = String(cfg.concurrency);
  }

  // 填充日期下拉
  dom.selDate.innerHTML = meta.trading_days.length
    ? meta.trading_days.map((d) => `<option value="${esc(d)}">${esc(d)}</option>`).join("")
    : `<option value="">无交易日</option>`;

  if (!keepSelection || !meta.trading_days.includes(state.selectedDate)) {
    state.selectedDate = meta.runs[0]?.as_of || meta.trading_days[0] || "";
  }
  dom.selDate.value = state.selectedDate;

  // 填充历史运行下拉
  const runOpts = [
    `<option value="">— 选择历史运行（共 ${meta.runs.length} 条）—</option>`,
    ...meta.runs.map((r) => {
      const tag = r.completed ? (r.official ? "正式" : "已完成") : `续跑 ${r.debated_count}/${r.submitted_count}`;
      const label = `${r.as_of} · ${r.hash_prefix} · ${tag} (${r.model || "?"})`;
      return `<option value="${esc(`${r.as_of}/${r.hash_prefix}`)}">${esc(label)}</option>`;
    }),
  ];
  dom.selRun.innerHTML = runOpts.join("");
  if (state.selectedRunKey) {
    dom.selRun.value = state.selectedRunKey;
  }

  renderTopbarStatus();
}

async function loadRunDetail(asOf, hashPrefix, { modelSource } = {}) {
  const ms = modelSource || state.selectedModelSource || "xgb_F69";
  const qs = new URLSearchParams({
    db: state.selectedDb,
    as_of: asOf,
    hash: hashPrefix,
    model_source: ms,
  });
  const detail = await apiJson(`/api/run?${qs.toString()}`);
  state.selectedModelSource = detail.model_source || ms;
  if (dom.selModelSource) dom.selModelSource.value = state.selectedModelSource;
  state.selectedDate = asOf;
  state.selectedRunKey = `${asOf}/${hashPrefix}`;
  state.chatExtraAttachedCodes = [];
  state.chatExcludeSelected = false;
  dom.selDate.value = asOf;
  dom.selRun.value = state.selectedRunKey;
  state.promptsPayloadCache = {};
  state.promptsSourcePref = "auto";
  hydrateRunDetail(detail);
  await Promise.all([
    refreshChatSessions({ preserveActive: false }),
    refreshPromptsData(),
  ]);
}

async function handleDryRun({ modelSource, asOfOverride } = {}) {
  const asOf = asOfOverride || dom.selDate.value;
  const ms = modelSource || state.selectedModelSource || "xgb_F69";
  if (!asOf) {
    showToast("请先选择一个交易日", "warn");
    return;
  }
  const msLabel = (state.meta?.model_sources || []).find((m) => m.id === ms)?.name || ms;
  showToast(`正在生成 ${asOf}（${msLabel}）演练快照（F69 首次计算约需十余秒，不调大模型）…`, "info", 0);
  try {
    const res = await apiJson("/api/dry-run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        db: state.selectedDb,
        as_of: asOf,
        model_source: ms,
        overrides: {
          reasoning_effort: dom.selEffort.value,
          concurrency: Number(dom.selConcurrency.value),
        },
      }),
    });
    state.isDryRun = true;
    state.selectedModelSource = res.model_source || ms;
    if (dom.selModelSource) dom.selModelSource.value = state.selectedModelSource;
    state.selectedDate = res.as_of;
    if (dom.selDate) dom.selDate.value = res.as_of;
    state.selectedRunKey = "";
    dom.selRun.value = "";
    state.streams = {};
    state.run = {
      db: res.db,
      as_of: res.as_of,
      variant: res.variant,
      cadence: res.cadence,
      model_source: res.model_source || ms,
      hash_prefix: res.hash_prefix,
      completed: false,
      official: false,
      is_running: false,
      request: res.request,
      stocks: {},
      result: null,
      final: null,
      usage: { calls: 0, failed_calls: 0, input_tokens: 0, output_tokens: 0 },
      events: [],
    };
    resetStockStateFromRequest(res.request);
    state.selectedCode = res.request.candidates[0]?.ts_code || "";
    state.chatExtraAttachedCodes = [];
    state.chatExcludeSelected = false;
    state.activeTab = "stock";
    state.promptsPayloadCache = {};
    state.promptsSourcePref = "auto";
    showToast(`演练快照就绪：${res.as_of}（${msLabel}）共 ${res.request.candidates.length} 只送辩候选，含 K 线、特征与打分归因。`, "info", 5500);
    renderAll();
    await Promise.all([
      refreshChatSessions({ preserveActive: false }),
      refreshPromptsData(),
    ]);
  } catch (err) {
    showToast(`演练失败：${err.message}`, "error", 7000);
  }
}

async function handleStartRun({ modelSource, asOfOverride } = {}) {
  const asOf = asOfOverride || dom.selDate.value;
  const ms = modelSource || state.selectedModelSource || "xgb_F69";
  if (!asOf) {
    showToast("请先选择一个交易日", "warn");
    return;
  }
  try {
    await apiJson("/api/runs/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        db: state.selectedDb,
        as_of: asOf,
        model_source: ms,
        overrides: {
          reasoning_effort: dom.selEffort.value,
          concurrency: Number(dom.selConcurrency.value),
        },
      }),
    });
    state.activeTab = "stock";
    renderTopbarStatus();
  } catch (err) {
    showToast(`启动失败：${err.message}`, "error", 7000);
  }
}

async function handleStopRun() {
  try {
    await apiJson("/api/runs/stop", { method: "POST" });
  } catch (err) {
    showToast(`停止失败：${err.message}`, "error", 5000);
  }
}

function buildMarkdownReport() {
  const final = state.run?.final;
  if (!final) return "尚无已冻结终选结果。";
  const lines = [
    `# ${final.as_of} 反弹模型 AI 辩论终选名单`,
    "",
    `- 运行 ID：\`${final.run_id}\``,
    `- 用量：成功 ${final.usage?.calls ?? 0} 次，失败 ${final.usage?.failed_calls ?? 0} 次，输入 ${final.usage?.input_tokens ?? 0} / 输出 ${final.usage?.output_tokens ?? 0} tokens`,
    `- 整体理由：${final.reason}`,
    "",
    "| AI 排名 | 代码 | 名称 | 行业 | 模型名次 | 风控裁决 | 入选理由 |",
    "|---|---|---|---|---|---|---|",
    ...(final.picks || []).map(
      (p) => `| ${p.rank} | ${p.ts_code} | ${p.name} | ${p.industry} | 第 ${p.model_rank} 名 | ${p.verdict} ${fmtNum(p.score, 0)} | ${p.reason} |`,
    ),
  ];
  return lines.join("\n");
}

async function handleCopyMarkdown() {
  const text = buildMarkdownReport();
  try {
    await navigator.clipboard.writeText(text);
    showToast("已复制 Markdown 复盘清单到剪贴板。", "info", 3000);
  } catch {
    showToast("浏览器禁止剪贴板写入，可改用「导出 JSON」。", "warn", 4000);
  }
}

function handleDownloadJson() {
  if (!state.run) return;
  const payload = state.run.final || state.run.request;
  const name = state.run.final
    ? `debate-final-${state.run.as_of}-${state.run.hash_prefix}.json`
    : `debate-preview-${state.run.as_of}.json`;
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// 证据引用悬停气泡
// ---------------------------------------------------------------------------

function bindCitationHover() {
  const tip = dom.citationTooltip;
  document.addEventListener("mouseover", (ev) => {
    const chip = ev.target.closest?.(".citation-chip");
    if (!chip) return;
    const path = chip.dataset.citation;
    const cand = (state.run?.request?.candidates || []).find((c) => c.ts_code === state.selectedCode);
    const resolved = resolveCitation(path, cand, state.run?.request?.context, state.mlOverview?.feature_catalog);
    if (!resolved) {
      tip.innerHTML = `<div class="ct-path">${esc(path)}</div><div class="ct-desc">输入快照引用路径</div>`;
    } else {
      tip.innerHTML = `
        <div class="ct-path">${esc(path)}</div>
        <div class="ct-val">${esc(resolved.value)}</div>
        <div class="ct-desc">${esc(resolved.desc)}</div>
      `;
    }
    tip.hidden = false;
    positionTooltip(ev.clientX, ev.clientY);
  });

  document.addEventListener("mousemove", (ev) => {
    if (!tip.hidden) positionTooltip(ev.clientX, ev.clientY);
  });

  document.addEventListener("mouseout", (ev) => {
    if (ev.target.closest?.(".citation-chip")) {
      tip.hidden = true;
    }
  });
}

function positionTooltip(x, y) {
  const tip = dom.citationTooltip;
  const pad = 14;
  const rect = tip.getBoundingClientRect();
  const left = Math.min(window.innerWidth - rect.width - 10, x + pad);
  const top = Math.min(window.innerHeight - rect.height - 10, y + pad);
  tip.style.left = `${Math.max(10, left)}px`;
  tip.style.top = `${Math.max(10, top)}px`;
}

// ---------------------------------------------------------------------------
// 初始化绑定
// ---------------------------------------------------------------------------

function bindEvents() {
  dom.selDb.addEventListener("change", async () => {
    state.selectedDb = dom.selDb.value;
    state.selectedRunKey = "";
    await refreshMeta({ keepSelection: false });
    if (state.meta.runs.length > 0) {
      await loadRunDetail(state.meta.runs[0].as_of, state.meta.runs[0].hash_prefix);
    } else {
      state.run = null;
      state.promptsPayloadCache = {};
      renderAll();
      await Promise.all([
        refreshChatSessions({ preserveActive: false }),
        refreshPromptsData(),
      ]);
    }
  });

  if (dom.selModelSource) {
    dom.selModelSource.addEventListener("change", async () => {
      state.selectedModelSource = dom.selModelSource.value;
      state.selectedRunKey = "";
      state.isDryRun = false;
      state.promptsPayloadCache = {};
      await refreshMeta({ keepSelection: false });
      if (state.meta.runs.length > 0) {
        await loadRunDetail(state.meta.runs[0].as_of, state.meta.runs[0].hash_prefix);
      } else {
        state.run = null;
        if (state.meta?.ml_available) {
          state.activeTab = "models";
          ensureMlViewLoaded();
        }
        renderAll();
        await Promise.all([
          refreshChatSessions({ preserveActive: false }),
          refreshPromptsData(),
        ]);
      }
    });
  }

  dom.selDate.addEventListener("change", async () => {
    state.selectedDate = dom.selDate.value;
    if (state.activeTab === "models" && state.meta?.ml_available) {
      state.mlConsensusDate = state.selectedDate;
      loadMlConsensus({ asOf: state.selectedDate });
    }
    const match = (state.meta?.runs || []).find((r) => r.as_of === state.selectedDate);
    if (match) {
      await loadRunDetail(match.as_of, match.hash_prefix);
    }
  });

  dom.selRun.addEventListener("change", async () => {
    const val = dom.selRun.value;
    if (!val) return;
    const [asOf, hash] = val.split("/");
    await loadRunDetail(asOf, hash);
  });

  dom.btnDryRun.addEventListener("click", () => handleDryRun());
  dom.btnStart.addEventListener("click", () => handleStartRun());
  dom.btnStop.addEventListener("click", handleStopRun);
  dom.btnCopyMd.addEventListener("click", handleCopyMarkdown);
  dom.btnDownloadJson.addEventListener("click", handleDownloadJson);
  dom.btnDemoFlow.addEventListener("click", toggleDemoFlow);

  // 动态同步顶栏高度，确保吸顶面板在任意窗口宽度下都不会被遮挡
  const topbarEl = document.querySelector(".topbar");
  if (topbarEl && typeof ResizeObserver !== "undefined") {
    const syncTopbar = () => {
      document.documentElement.style.setProperty("--topbar-h", `${Math.ceil(topbarEl.getBoundingClientRect().height)}px`);
    };
    syncTopbar();
    new ResizeObserver(syncTopbar).observe(topbarEl);
  }

  dom.tabStock.addEventListener("click", () => switchTab("stock"));
  dom.tabFinal.addEventListener("click", () => switchTab("final"));
  dom.tabModels.addEventListener("click", () => switchTab("models"));
  dom.tabPrompts.addEventListener("click", () => switchTab("prompts"));

  dom.filterPills.querySelectorAll(".chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      dom.filterPills.querySelectorAll(".chip").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.filter = btn.dataset.filter;
      renderRosterList();
    });
  });

  dom.selSort.addEventListener("change", () => {
    state.sort = dom.selSort.value;
    renderRosterList();
  });

  dom.chkAutofollow.addEventListener("change", () => {
    state.autoFollow = dom.chkAutofollow.checked;
  });

  bindCitationHover();
  bindChatEvents();
  bindPromptEvents();
  bindMlEvents();
}

// ---------------------------------------------------------------------------
// 右侧 AI 投研复盘对话栏（档位 B：9 只读工具 + 逐字思考 + 落盘持久化）
// ---------------------------------------------------------------------------

function chatRunScope() {
  const run = state.run;
  const asOf = run?.as_of || state.selectedDate || "";
  const hash = !state.isDryRun && run?.hash_prefix ? run.hash_prefix : "";
  return {
    db: state.selectedDb || "",
    variant: run?.variant || state.meta?.variant || "",
    cadence: run?.cadence || state.meta?.cadence || "",
    model_source: run?.model_source || state.selectedModelSource || "",
    as_of: asOf,
    hash,
  };
}

function getChatAttachedCodes() {
  const candidates = state.run?.request?.candidates || [];
  const validSet = new Set(candidates.map((c) => c.ts_code));
  const out = [];
  if (state.selectedCode && !state.chatExcludeSelected && validSet.has(state.selectedCode)) {
    out.push(state.selectedCode);
  }
  for (const code of state.chatExtraAttachedCodes) {
    if (validSet.has(code) && !out.includes(code)) {
      out.push(code);
    }
    if (out.length >= 3) break;
  }
  return out.slice(0, 3);
}

function chatRoleLabel(roleId) {
  const found = state.chatRoles.find((r) => r.id === roleId);
  return found ? found.name : roleId || "投研复盘助手";
}

function renderChatLayout() {
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
}

function renderChatControls() {
  dom.selChatRole.innerHTML = state.chatRoles
    .map((r) => `<option value="${esc(r.id)}" title="${esc(r.desc)}">${esc(r.name)}</option>`)
    .join("");
  dom.selChatRole.value = state.chatRole;

  if (state.chatSessions.length === 0) {
    dom.selChatSession.innerHTML = `<option value="">新对话（发送首句自动落盘）</option>`;
    dom.selChatSession.value = "";
  } else {
    const opts = [
      ...(state.activeChatSessionId ? [] : [`<option value="">＋ 新对话（未发送）</option>`]),
      ...state.chatSessions.map((s) => {
        const turns = Math.floor((s.messages?.length || 0) / 2);
        const label = `${s.title || "未命名对话"}${turns > 0 ? ` (${turns}轮)` : ""}`;
        return `<option value="${esc(s.id)}">${esc(label)}</option>`;
      }),
    ];
    dom.selChatSession.innerHTML = opts.join("");
    dom.selChatSession.value = state.activeChatSessionId;
  }

  dom.btnChatDelete.disabled = !state.activeChatSessionId || state.chatStreaming;
  dom.btnChatNew.disabled = state.chatStreaming;
  dom.btnChatSend.disabled = state.chatStreaming;
  dom.btnChatStop.hidden = !state.chatStreaming;
  dom.chatInput.disabled = state.chatStreaming;

  const roleMeta = state.chatRoles.find((r) => r.id === state.chatRole);
  dom.chatSubtitle.textContent = roleMeta ? roleMeta.desc : "预置方法论 · 6 只读工具按需查全池";

  if (state.chatTools.length > 0) {
    dom.chatToolsBadge.textContent = `⚙ ${state.chatTools.length} 工具`;
    dom.chatToolsBadge.title = `只读工具（100% 覆盖本场辩论与 DuckDB）：\n${state.chatTools.map((t) => `• ${t.label} (${t.name})`).join("\n")}`;
  }
  dom.chatPersistHint.textContent = state.chatStreaming
    ? "正在流式思考与调工具…"
    : state.chatStoreName
      ? `已落盘 · ${state.chatStoreName}`
      : "会话自动落盘保存";
}

function renderChatContextStrip() {
  const candidates = state.run?.request?.candidates || [];
  const byCode = new Map(candidates.map((c) => [c.ts_code, c]));
  const attached = getChatAttachedCodes();

  let globalLabel = "仅 DuckDB 数据库";
  if (state.run?.final) {
    globalLabel = `${state.run.as_of} · 终选+${candidates.length}股全览`;
  } else if (state.run?.request) {
    globalLabel = `${state.run.as_of} · ${state.isDryRun ? "演练" : "进行中"} ${candidates.length}股`;
  }

  const pills = [
    `<span class="chat-ctx-pill ctx-global" title="常驻系统提示词：方法论 + 终选结果 + 全部送辩风控简表">● ${esc(globalLabel)}</span>`,
    ...attached.map((code) => {
      const name = byCode.get(code)?.name || "";
      const isAuto = code === state.selectedCode;
      return `
        <span class="chat-ctx-pill ctx-stock" title="${isAuto ? "当前聚焦个股（自动挂载完整 28 特征 + 20 日线 + 6 轮辩论）" : "手动追加挂载个股"}">
          <button type="button" class="chat-ctx-stock-btn mono" data-chat-jump="${esc(code)}">@${esc(code)} ${esc(name)}</button>
          <button type="button" class="chat-ctx-remove" data-chat-detach="${esc(code)}" title="移出常驻上下文">×</button>
        </span>
      `;
    }),
  ];
  dom.chatContextPills.innerHTML = pills.join("");

  const attachable = candidates.filter((c) => !attached.includes(c.ts_code));
  dom.selChatAttach.innerHTML = [
    `<option value="">${attached.length >= 3 ? "已挂满 3 只（可调工具查其余）" : "+ 追加个股明细…"}</option>`,
    ...attachable.map(
      (c) => `<option value="${esc(c.ts_code)}">#${String(c.model.rank).padStart(2, "0")} ${esc(c.ts_code)} ${esc(c.name)}</option>`,
    ),
  ].join("");
  dom.selChatAttach.disabled = attached.length >= 3 || candidates.length === 0;
}

function getActiveChatSession() {
  if (!state.activeChatSessionId) return null;
  return state.chatSessions.find((s) => s.id === state.activeChatSessionId) || null;
}

// --- 安全研报风 Markdown + 字段证据芯片 + 可点击个股代码渲染器 ---

const _CITATION_PATH_RE =
  /\b(?:features\.[A-Za-z0-9_]+|feature_pct\.[A-Za-z0-9_]+|bars\[-?\d+\](?:\.[A-Za-z0-9_]+)?|flags\.[A-Za-z0-9_]+|industry_stats\.[A-Za-z0-9_]+|model\.[A-Za-z0-9_]+|context\.[A-Za-z0-9_]+)\b/g;
const _STOCK_CODE_RE = /\b(\d{6}\.(?:SH|SZ))\b/g;

function enhanceInlineEntities(escapedHtml) {
  const validCodes = new Set((state.run?.request?.candidates || []).map((c) => c.ts_code));
  // 1. 先处理反引号行内代码：若是已知引用路径则直接升级为悬停 citation-chip
  let out = escapedHtml.replace(/`([^`\n]+)`/g, (_m, inner) => {
    const trimmed = inner.trim();
    if (/^(?:features|feature_pct|flags|industry_stats|model|context)\.[A-Za-z0-9_]+$|^bars\[-?\d+\](?:\.[A-Za-z0-9_]+)?$/.test(trimmed)) {
      return `<span class="citation-chip mono" data-citation="${trimmed}">${trimmed}</span>`;
    }
    return `<code>${inner}</code>`;
  });

  // 2. 粗体与斜体
  out = out.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");

  // 3. 在 HTML 标签之外的纯文本段中，升级裸字段路径与候选股票代码
  const parts = out.split(/(<[^>]+>)/g);
  let insideTagDepth = 0;
  return parts
    .map((part) => {
      if (part.startsWith("<")) {
        if (/^<(?:code|button|span)\b/i.test(part)) insideTagDepth += 1;
        else if (/^<\/(?:code|button|span)>/i.test(part)) insideTagDepth = Math.max(0, insideTagDepth - 1);
        return part;
      }
      if (insideTagDepth > 0) return part;
      let seg = part.replace(
        _CITATION_PATH_RE,
        (match) => `<span class="citation-chip mono" data-citation="${match}">${match}</span>`,
      );
      if (validCodes.size > 0) {
        seg = seg.replace(_STOCK_CODE_RE, (code) =>
          validCodes.has(code)
            ? `<button type="button" class="chat-code-link" data-chat-jump="${code}" title="点击切换到 ${code} 的 K 线与辩论详情">${code}</button>`
            : code,
        );
      }
      return seg;
    })
    .join("");
}

function renderChatMarkdown(rawText) {
  const text = String(rawText || "").replace(/\r\n/g, "\n");
  if (!text.trim()) return "";

  // 提取围栏代码块避免被行级规则误拆
  const codeBlocks = [];
  const stripped = text.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (_m, _lang, code) => {
    const idx = codeBlocks.length;
    codeBlocks.push(`<pre><code>${esc(code.replace(/\n$/, ""))}</code></pre>`);
    return `\n@@CODEBLOCK_${idx}@@\n`;
  });

  const lines = stripped.split("\n");
  const htmlBlocks = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      i += 1;
      continue;
    }

    const cbMatch = /^@@CODEBLOCK_(\d+)@@$/.exec(trimmed);
    if (cbMatch) {
      htmlBlocks.push(codeBlocks[Number(cbMatch[1])] || "");
      i += 1;
      continue;
    }

    if (/^---+$|^\*\*\*+$/.test(trimmed)) {
      htmlBlocks.push("<hr />");
      i += 1;
      continue;
    }

    const headMatch = /^(#{1,4})\s+(.+)$/.exec(trimmed);
    if (headMatch) {
      const lvl = Math.min(4, Math.max(2, headMatch[1].length));
      htmlBlocks.push(`<h${lvl}>${enhanceInlineEntities(esc(headMatch[2]))}</h${lvl}>`);
      i += 1;
      continue;
    }

    // Markdown 表格
    if (
      trimmed.includes("|") &&
      i + 1 < lines.length &&
      /^\s*\|?\s*[:\-]+(?:\s*\|\s*[:\-]+)+\s*\|?\s*$/.test(lines[i + 1])
    ) {
      const splitRow = (r) =>
        r
          .trim()
          .replace(/^\|/, "")
          .replace(/\|$/, "")
          .split("|")
          .map((c) => c.trim());
      const headers = splitRow(line);
      i += 2;
      const bodyRows = [];
      while (i < lines.length && lines[i].trim() && lines[i].includes("|")) {
        bodyRows.push(splitRow(lines[i]));
        i += 1;
      }
      htmlBlocks.push(`
        <div class="chat-table-wrap">
          <table>
            <thead><tr>${headers.map((h) => `<th>${enhanceInlineEntities(esc(h))}</th>`).join("")}</tr></thead>
            <tbody>
              ${bodyRows
                .map((r) => `<tr>${headers.map((_h, idx) => `<td>${enhanceInlineEntities(esc(r[idx] ?? ""))}</td>`).join("")}</tr>`)
                .join("")}
            </tbody>
          </table>
        </div>
      `);
      continue;
    }

    // 引用块
    if (trimmed.startsWith(">")) {
      const qLines = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) {
        qLines.push(lines[i].trim().replace(/^>\s?/, ""));
        i += 1;
      }
      htmlBlocks.push(`<blockquote>${qLines.map((l) => enhanceInlineEntities(esc(l))).join("<br />")}</blockquote>`);
      continue;
    }

    // 无序列表
    if (/^[-*•]\s+/.test(trimmed)) {
      const items = [];
      while (i < lines.length && /^[-*•]\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*•]\s+/, ""));
        i += 1;
      }
      htmlBlocks.push(`<ul>${items.map((it) => `<li>${enhanceInlineEntities(esc(it))}</li>`).join("")}</ul>`);
      continue;
    }

    // 有序列表
    if (/^\d+\.\s+/.test(trimmed)) {
      const items = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ""));
        i += 1;
      }
      htmlBlocks.push(`<ol>${items.map((it) => `<li>${enhanceInlineEntities(esc(it))}</li>`).join("")}</ol>`);
      continue;
    }

    // 普通段落（合并连续非空普通行）
    const pLines = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^@@CODEBLOCK_\d+@@$/.test(lines[i].trim()) &&
      !/^(#{1,4})\s+/.test(lines[i].trim()) &&
      !/^[-*•]\s+/.test(lines[i].trim()) &&
      !/^\d+\.\s+/.test(lines[i].trim()) &&
      !lines[i].trim().startsWith(">") &&
      !/^---+$|^\*\*\*+$/.test(lines[i].trim())
    ) {
      if (
        lines[i].includes("|") &&
        i + 1 < lines.length &&
        /^\s*\|?\s*[:\-]+(?:\s*\|\s*[:\-]+)+\s*\|?\s*$/.test(lines[i + 1])
      ) {
        break;
      }
      pLines.push(lines[i].trim());
      i += 1;
    }
    htmlBlocks.push(`<p>${pLines.map((l) => enhanceInlineEntities(esc(l))).join("<br />")}</p>`);
  }

  return htmlBlocks.join("");
}

function summarizeToolArgs(args) {
  if (!args || typeof args !== "object") return "";
  const entries = Object.entries(args).filter(([, v]) => v !== undefined && v !== null && v !== "");
  if (entries.length === 0) return "";
  return entries
    .map(([k, v]) => `${k}=${Array.isArray(v) ? v.join(",") : String(v)}`)
    .join(" · ");
}

function renderToolStepsHtml(toolCalls) {
  if (!Array.isArray(toolCalls) || toolCalls.length === 0) return "";
  return `
    <div class="chat-tool-steps">
      ${toolCalls
        .map((tc) => {
          const isRunning = tc.status === "running";
          const isOk = tc.status === "ok" || (tc.status === undefined && tc.ok !== false);
          const cls = isRunning ? " tool-running" : isOk ? "" : " tool-err";
          const icon = isRunning
            ? `<span class="tool-spinner" aria-hidden="true"></span>`
            : isOk
              ? `<span class="tool-check">✓</span>`
              : `<span class="tool-fail">!</span>`;
          const argPreview = summarizeToolArgs(tc.arguments);
          const timeText = isRunning ? "查询中…" : `${tc.elapsed_ms ?? 0}ms`;
          return `
            <details class="tool-step-card${cls}">
              <summary class="tool-step-summary">
                <div class="tool-step-left">
                  ${icon}
                  <span class="tool-label">${esc(tc.label || tc.name)}</span>
                  ${argPreview ? `<span class="tool-arg-preview mono">${esc(argPreview)}</span>` : ""}
                </div>
                <span class="tool-elapsed mono">${esc(timeText)}</span>
              </summary>
              <div class="tool-step-detail">
                <div><strong>工具：</strong><span class="mono">${esc(tc.name)}</span></div>
                ${tc.summary ? `<div><strong>结果：</strong>${esc(tc.summary)}</div>` : ""}
                <pre>${esc(JSON.stringify(tc.arguments || {}, null, 2))}</pre>
              </div>
            </details>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderChatEmptyState() {
  const candidates = state.run?.request?.candidates || [];
  const focusCand = candidates.find((c) => c.ts_code === state.selectedCode);
  const focusName = focusCand ? `${focusCand.name}(${focusCand.ts_code})` : "当前选中股票";
  const quickPrompts = [
    "为什么终选换掉了原模型第 1 名？被换掉的票各自有什么硬伤？",
    "对比当前所有「看多」股票的买点结构与核心风险，列成一张对照表。",
    "帮我筛一下这 20 只送辩股里，近 5 天放量最明显（vol5 最高）的前 5 只是谁？",
    `针对 ${focusName}，给出明日 T+1 开盘前可执行检查清单（什么条件买、什么条件放弃）。`,
  ];

  return `
    <div class="chat-empty">
      <div class="chat-empty-title">✦ 反弹模型投研复盘 Agent（档位 B）</div>
      <p class="chat-empty-desc">
        已预置完整方法论 <code>docs/debate-methodology.md</code>，默认掌握终选名单、全部 20 只风控简表与当前聚焦个股的完整 6 轮辩论；问及任意其他股票、跨股筛选或历史事后收益时，会自动调用只读工具精准查证。
      </p>
      <div class="chat-empty-caps">
        <span class="chat-cap-tag">100% 辩论账本可读</span>
        <span class="chat-cap-tag">DuckDB 只读查询</span>
        <span class="chat-cap-tag">逐字思考流</span>
        <span class="chat-cap-tag">落盘持久化</span>
      </div>
      <div class="chat-quick-list">
        ${quickPrompts
          .map((q) => `<button type="button" class="chat-quick-btn" data-quick-prompt="${esc(q)}">↗ ${esc(q)}</button>`)
          .join("")}
      </div>
    </div>
  `;
}

function renderPersistedMessageHtml(msg) {
  if (msg.role === "user") {
    const tags = (msg.attached_codes || []).map((c) => `<span class="mono">@${esc(c)}</span>`).join(" ");
    const timeStr = msg.created_at ? msg.created_at.slice(11, 16) : "";
    return `
      <div class="chat-msg chat-msg-user">
        <div class="chat-bubble">${esc(msg.content)}</div>
        <div class="chat-user-meta">${tags} ${timeStr ? `<span class="mono">${esc(timeStr)}</span>` : ""}</div>
      </div>
    `;
  }

  const roleName = chatRoleLabel(msg.role_preset);
  const timeStr = msg.created_at ? msg.created_at.slice(11, 16) : "";
  const tokStr = msg.usage && (msg.usage.input_tokens || msg.usage.output_tokens) ? fmtTokens(msg.usage) : "";
  const thinkingHtml = msg.thinking
    ? `
      <details class="chat-thinking-archive">
        <summary>深度思考过程（${msg.thinking.length} 字）</summary>
        <pre>${esc(msg.thinking)}</pre>
      </details>
    `
    : "";

  return `
    <div class="chat-msg chat-msg-assistant">
      <div class="chat-asst-card">
        <div class="chat-asst-head">
          <span class="chat-asst-role">✦ ${esc(roleName)}</span>
          <span class="mono">${esc([tokStr, timeStr].filter(Boolean).join(" · "))}</span>
        </div>
        ${renderToolStepsHtml(msg.tool_calls)}
        ${thinkingHtml}
        <div class="chat-md">${renderChatMarkdown(msg.content)}</div>
      </div>
    </div>
  `;
}

function renderLiveTurnHtml(live) {
  if (!live) return "";
  const roleName = chatRoleLabel(live.role_preset);
  const elapsed = live.started_at ? `${Math.max(0, Date.now() / 1000 - live.started_at).toFixed(1)}s` : "0.0s";
  const hasText = Boolean(live.text);
  const hasTools = live.tool_calls.length > 0;
  const showThinkingLive = Boolean(live.thinking) || (!hasText && !hasTools && !live.error);

  return `
    <div class="chat-msg chat-msg-assistant" id="chat-live-msg">
      <div class="chat-asst-card is-live">
        <div class="chat-asst-head">
          <span class="chat-asst-role">✦ ${esc(roleName)}</span>
          <span class="mono" id="chat-live-elapsed">${elapsed}</span>
        </div>
        <div id="chat-live-tools">${renderToolStepsHtml(live.tool_calls)}</div>
        ${
          showThinkingLive
            ? `
          <div class="chat-thinking-live" id="chat-live-thinking-wrap">
            <div class="chat-thinking-label">
              <span>深度思考 (Thinking)</span>
              <span class="mono" id="chat-live-thinking-len">${live.thinking.length} 字</span>
            </div>
            <div class="chat-thinking-content" id="chat-live-thinking">${esc(live.thinking || "正在读取上下文并组织分析…")}${!hasText ? '<span class="caret"></span>' : ""}</div>
          </div>
        `
            : ""
        }
        <div class="chat-md" id="chat-live-md">${hasText ? `${renderChatMarkdown(live.text)}<span class="caret"></span>` : ""}</div>
        ${live.error ? `<div class="chat-error-banner">${esc(live.error)}</div>` : ""}
      </div>
    </div>
  `;
}

function scrollChatToBottom(force = false) {
  const el = dom.chatMessages;
  if (!el) return;
  const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  if (force || nearBottom) {
    el.scrollTop = el.scrollHeight;
  }
}

function renderChatMessages({ scrollBottom = false } = {}) {
  const session = getActiveChatSession();
  const persisted = session?.messages || [];

  if (persisted.length === 0 && !state.chatLiveTurn) {
    dom.chatMessages.innerHTML = renderChatEmptyState();
    return;
  }

  const parts = persisted.map((m) => renderPersistedMessageHtml(m));
  if (state.chatLiveTurn) {
    // 若刚才乐观追加的用户消息还未回刷进 session.messages，则先在前端展示
    if (state.chatLiveTurn.optimisticUser && !persisted.some((m) => m.id === state.chatLiveTurn.optimisticUser.id)) {
      parts.push(renderPersistedMessageHtml(state.chatLiveTurn.optimisticUser));
    }
    parts.push(renderLiveTurnHtml(state.chatLiveTurn));
  }
  dom.chatMessages.innerHTML = parts.join("");
  scrollChatToBottom(scrollBottom);
}

function patchLiveChatDom(kind) {
  const live = state.chatLiveTurn;
  if (!live) return;
  if (kind === "thinking") {
    const box = document.getElementById("chat-live-thinking");
    const lenEl = document.getElementById("chat-live-thinking-len");
    if (!box) {
      renderChatMessages({ scrollBottom: true });
      return;
    }
    box.innerHTML = `${esc(live.thinking)}${!live.text ? '<span class="caret"></span>' : ""}`;
    box.scrollTop = box.scrollHeight;
    if (lenEl) lenEl.textContent = `${live.thinking.length} 字`;
    scrollChatToBottom(false);
  } else if (kind === "tools") {
    const wrap = document.getElementById("chat-live-tools");
    if (!wrap) {
      renderChatMessages({ scrollBottom: true });
      return;
    }
    wrap.innerHTML = renderToolStepsHtml(live.tool_calls);
    scrollChatToBottom(false);
  } else if (kind === "text") {
    const mdEl = document.getElementById("chat-live-md");
    if (!mdEl) {
      renderChatMessages({ scrollBottom: true });
      return;
    }
    // 正文开始输出时去掉思考框光标
    const thinkBox = document.getElementById("chat-live-thinking");
    if (thinkBox && thinkBox.querySelector(".caret")) {
      thinkBox.textContent = live.thinking;
    }
    mdEl.innerHTML = `${renderChatMarkdown(live.text)}<span class="caret"></span>`;
    scrollChatToBottom(false);
  }
}

async function refreshChatSessions({ preserveActive = false } = {}) {
  try {
    const scope = chatRunScope();
    const qs = new URLSearchParams();
    if (scope.db) qs.set("db", scope.db);
    if (scope.variant) qs.set("variant", scope.variant);
    if (scope.cadence) qs.set("cadence", scope.cadence);
    if (scope.model_source) qs.set("model_source", scope.model_source);
    if (scope.as_of) qs.set("as_of", scope.as_of);
    if (scope.hash) qs.set("hash", scope.hash);

    const data = await apiJson(`/api/chat/sessions?${qs.toString()}`);
    if (Array.isArray(data.roles) && data.roles.length > 0) state.chatRoles = data.roles;
    if (Array.isArray(data.tools)) state.chatTools = data.tools;
    state.chatSessions = Array.isArray(data.sessions) ? data.sessions : [];
    state.chatStoreName = data.store_name || "";

    const exists = state.chatSessions.some((s) => s.id === state.activeChatSessionId);
    if (!preserveActive || !exists) {
      state.activeChatSessionId = state.chatSessions[0]?.id || "";
    }
    const activeSess = getActiveChatSession();
    if (activeSess?.role_preset) {
      state.chatRole = activeSess.role_preset;
    }
  } catch (err) {
    // 静默降级，避免阻断主界面加载
    state.chatSessions = [];
    state.activeChatSessionId = "";
  }
  renderChatControls();
  renderChatContextStrip();
  renderChatMessages({ scrollBottom: true });
}

async function handleCreateChatSession() {
  if (state.chatStreaming) return;
  try {
    const scope = chatRunScope();
    const res = await apiJson("/api/chat/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...scope,
        role_preset: state.chatRole,
        title: "新对话",
      }),
    });
    if (res.session) {
      state.chatSessions.unshift(res.session);
      state.activeChatSessionId = res.session.id;
      renderChatControls();
      renderChatMessages({ scrollBottom: true });
      dom.chatInput.focus();
    }
  } catch (err) {
    showToast(`创建会话失败：${err.message}`, "error", 5000);
  }
}

async function handleDeleteChatSession() {
  if (!state.activeChatSessionId || state.chatStreaming) return;
  const sid = state.activeChatSessionId;
  try {
    const scope = chatRunScope();
    const qs = new URLSearchParams({ session_id: sid });
    if (scope.db) qs.set("db", scope.db);
    if (scope.variant) qs.set("variant", scope.variant);
    if (scope.cadence) qs.set("cadence", scope.cadence);
    if (scope.model_source) qs.set("model_source", scope.model_source);
    if (scope.as_of) qs.set("as_of", scope.as_of);
    if (scope.hash) qs.set("hash", scope.hash);

    await apiJson(`/api/chat/sessions?${qs.toString()}`, { method: "DELETE" });
    state.chatSessions = state.chatSessions.filter((s) => s.id !== sid);
    state.activeChatSessionId = state.chatSessions[0]?.id || "";
    renderChatControls();
    renderChatMessages({ scrollBottom: false });
  } catch (err) {
    showToast(`删除会话失败：${err.message}`, "error", 5000);
  }
}

function setChatOpen(open) {
  state.chatOpen = Boolean(open);
  renderChatLayout();
  if (state.chatOpen) {
    renderChatControls();
    renderChatContextStrip();
    renderChatMessages({ scrollBottom: true });
    requestAnimationFrame(() => dom.chatInput.focus());
  }
}

function openChatWithPrompt({ code, role, prompt }) {
  if (code) {
    const valid = (state.run?.request?.candidates || []).some((c) => c.ts_code === code);
    if (valid) {
      if (state.selectedCode === code) {
        state.chatExcludeSelected = false;
      } else if (!state.chatExtraAttachedCodes.includes(code)) {
        state.chatExtraAttachedCodes = [code, ...state.chatExtraAttachedCodes].slice(0, 2);
      }
    }
  }
  if (role && state.chatRoles.some((r) => r.id === role)) {
    state.chatRole = role;
  }
  setChatOpen(true);
  if (prompt) {
    dom.chatInput.value = prompt;
    autoResizeChatTextarea();
    dom.chatInput.focus();
    dom.chatInput.setSelectionRange(prompt.length, prompt.length);
  }
}

function autoResizeChatTextarea() {
  const ta = dom.chatInput;
  ta.style.height = "auto";
  ta.style.height = `${Math.min(150, Math.max(46, ta.scrollHeight))}px`;
}

async function handleSendChatMessage(overrideText = null) {
  if (state.chatStreaming) return;
  const text = (overrideText !== null ? overrideText : dom.chatInput.value).trim();
  if (!text) return;

  if (!state.chatOpen) setChatOpen(true);
  dom.chatInput.value = "";
  autoResizeChatTextarea();

  const scope = chatRunScope();
  const attachedCodes = getChatAttachedCodes();
  const controller = new AbortController();

  state.chatStreaming = true;
  state.chatAbortController = controller;
  state.chatLiveTurn = {
    session_id: state.activeChatSessionId,
    message_id: "",
    role_preset: state.chatRole,
    thinking: "",
    text: "",
    tool_calls: [],
    started_at: Date.now() / 1000,
    error: "",
    optimisticUser: {
      id: `local-${Date.now()}`,
      role: "user",
      content: text,
      attached_codes: attachedCodes,
      created_at: new Date().toISOString(),
    },
  };

  renderChatControls();
  renderChatMessages({ scrollBottom: true });

  const handleChatEvent = (evName, payload) => {
    const live = state.chatLiveTurn;
    if (!live) return;
    if (evName === "chat.start") {
      live.session_id = payload.session_id || live.session_id;
      live.message_id = payload.message_id || "";
      if (payload.session_id) state.activeChatSessionId = payload.session_id;
    } else if (evName === "chat.thinking") {
      live.thinking += payload.delta || "";
      patchLiveChatDom("thinking");
    } else if (evName === "chat.tool_start") {
      live.tool_calls.push({
        call_id: payload.call_id,
        name: payload.name,
        label: payload.label || payload.name,
        arguments: payload.arguments || {},
        status: "running",
        elapsed_ms: 0,
        summary: "",
      });
      patchLiveChatDom("tools");
    } else if (evName === "chat.tool_done") {
      const slot = live.tool_calls.find((t) => t.call_id === payload.call_id);
      if (slot) {
        slot.status = payload.ok ? "ok" : "err";
        slot.ok = payload.ok;
        slot.elapsed_ms = payload.elapsed_ms ?? 0;
        slot.summary = payload.summary || "";
      }
      patchLiveChatDom("tools");
    } else if (evName === "chat.delta") {
      live.text += payload.delta || "";
      patchLiveChatDom("text");
    } else if (evName === "chat.error") {
      live.error = payload.error || "对话生成出错";
      renderChatMessages({ scrollBottom: true });
      showToast(`AI 对话出错：${live.error}`, "error", 6000);
    } else if (evName === "chat.done" || evName === "chat.stopped") {
      if (payload.session_id) state.activeChatSessionId = payload.session_id;
    }
  };

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        ...scope,
        session_id: state.activeChatSessionId || null,
        message: text,
        role_preset: state.chatRole,
        attached_codes: attachedCodes,
        reasoning_effort: dom.selEffort.value || undefined,
      }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.error || `请求失败 (${res.status})`);
    }
    if (!res.body) {
      throw new Error("浏览器不支持流式响应读取");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop() || "";

      for (const frame of frames) {
        const dataLines = frame
          .split(/\r?\n/)
          .filter((l) => l.startsWith("data:"))
          .map((l) => l.slice(5).trim());
        if (dataLines.length === 0) continue;
        try {
          const parsed = JSON.parse(dataLines.join("\n"));
          handleChatEvent(parsed.type, parsed.payload || {});
        } catch {
          // 忽略异常帧
        }
      }
    }
  } catch (err) {
    if (err.name === "AbortError") {
      showToast("已中止本次生成。", "warn", 3000);
      await new Promise((r) => setTimeout(r, 150));
    } else {
      if (state.chatLiveTurn) state.chatLiveTurn.error = err.message;
      showToast(`AI 对话失败：${err.message}`, "error", 6000);
    }
  } finally {
    const hadError = Boolean(state.chatLiveTurn?.error);
    state.chatStreaming = false;
    state.chatAbortController = null;
    if (!hadError) {
      state.chatLiveTurn = null;
      await refreshChatSessions({ preserveActive: true });
    } else {
      renderChatControls();
      renderChatMessages({ scrollBottom: true });
      // 下次发送时清掉报错态
      state.chatLiveTurn = null;
    }
  }
}

function handleStopChatGeneration() {
  if (state.chatAbortController) {
    state.chatAbortController.abort();
  }
}

function bindChatEvents() {
  dom.btnToggleChat.addEventListener("click", () => setChatOpen(!state.chatOpen));
  dom.btnChatClose.addEventListener("click", () => setChatOpen(false));

  dom.btnChatWide.addEventListener("click", () => {
    state.chatWide = !state.chatWide;
    renderChatLayout();
  });

  const toggleLeft = () => {
    state.leftCollapsed = !state.leftCollapsed;
    renderChatLayout();
  };
  dom.btnChatToggleLeft.addEventListener("click", toggleLeft);
  dom.btnExpandLeft.addEventListener("click", () => {
    state.leftCollapsed = false;
    renderChatLayout();
  });

  dom.selChatRole.addEventListener("change", () => {
    state.chatRole = dom.selChatRole.value;
    renderChatControls();
  });

  dom.selChatSession.addEventListener("change", () => {
    if (state.chatStreaming) return;
    state.activeChatSessionId = dom.selChatSession.value;
    const sess = getActiveChatSession();
    if (sess?.role_preset) state.chatRole = sess.role_preset;
    renderChatControls();
    renderChatMessages({ scrollBottom: true });
  });

  dom.btnChatNew.addEventListener("click", handleCreateChatSession);
  dom.btnChatDelete.addEventListener("click", handleDeleteChatSession);

  dom.selChatAttach.addEventListener("change", () => {
    const code = dom.selChatAttach.value;
    if (!code) return;
    if (code === state.selectedCode) {
      state.chatExcludeSelected = false;
    } else if (!state.chatExtraAttachedCodes.includes(code)) {
      state.chatExtraAttachedCodes.push(code);
    }
    dom.selChatAttach.value = "";
    renderChatContextStrip();
  });

  dom.chatContextPills.addEventListener("click", (ev) => {
    const detachBtn = ev.target.closest?.("[data-chat-detach]");
    if (detachBtn) {
      const code = detachBtn.dataset.chatDetach;
      if (code === state.selectedCode) state.chatExcludeSelected = true;
      state.chatExtraAttachedCodes = state.chatExtraAttachedCodes.filter((c) => c !== code);
      renderChatContextStrip();
      return;
    }
    const jumpBtn = ev.target.closest?.("[data-chat-jump]");
    if (jumpBtn) {
      state.selectedCode = jumpBtn.dataset.chatJump;
      state.chatExcludeSelected = false;
      switchTab("stock");
    }
  });

  dom.chatMessages.addEventListener("click", (ev) => {
    const quickBtn = ev.target.closest?.("[data-quick-prompt]");
    if (quickBtn) {
      handleSendChatMessage(quickBtn.dataset.quickPrompt);
      return;
    }
    const jumpBtn = ev.target.closest?.("[data-chat-jump]");
    if (jumpBtn) {
      state.selectedCode = jumpBtn.dataset.chatJump;
      state.chatExcludeSelected = false;
      switchTab("stock");
    }
  });

  dom.chatForm.addEventListener("submit", (ev) => {
    ev.preventDefault();
    handleSendChatMessage();
  });

  dom.btnChatStop.addEventListener("click", handleStopChatGeneration);

  dom.chatInput.addEventListener("input", autoResizeChatTextarea);
  dom.chatInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) {
      ev.preventDefault();
      handleSendChatMessage();
    }
  });

  // 辩论卡片上的“✦ 追问”按钮事件委托
  const handleAskClick = (ev) => {
    const btn = ev.target.closest?.("[data-ask-prompt]");
    if (!btn) return;
    ev.stopPropagation();
    openChatWithPrompt({
      code: btn.dataset.askCode || "",
      role: btn.dataset.askRole || "research_copilot",
      prompt: btn.dataset.askPrompt || "",
    });
  };
  dom.stockContent.addEventListener("click", handleAskClick);
  dom.finalContent.addEventListener("click", handleAskClick);

  // 全局快捷键 Ctrl+K / Cmd+K 打开或折叠对话栏
  document.addEventListener("keydown", (ev) => {
    if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "k") {
      ev.preventDefault();
      if (!state.chatOpen) {
        setChatOpen(true);
      } else if (document.activeElement !== dom.chatInput) {
        dom.chatInput.focus();
      } else {
        setChatOpen(false);
      }
    }
  });
}

// ---------------------------------------------------------------------------
// 提示词与角色设定（视图 C）+ 单股实际 User JSON 白盒弹窗
// ---------------------------------------------------------------------------

async function copyTextWithToast(text, label = "内容") {
  if (!text) {
    showToast(`没有可复制的${label}`, "warn");
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
    showToast(`已复制${label}（${text.length.toLocaleString()} 字符）`, "info", 2800);
  } catch {
    showToast(`复制失败，请手动选中复制`, "warn", 3200);
  }
}

function promptPayloadCacheKey(roleKey, tsCode) {
  const scope = chatRunScope();
  const runSig = `${scope.db}|${scope.model_source}|${scope.as_of}|${scope.hash}|${state.isDryRun ? "dry" : "run"}`;
  return `${runSig}|${roleKey}|${tsCode || ""}`;
}

function resolveDefaultPayloadCode(roleKey) {
  if (roleKey === "final_pick") return "-";
  if (roleKey.startsWith("chat:")) return state.selectedCode || "";
  const candidates = state.run?.request?.candidates || [];
  if (state.promptsPayloadCode && candidates.some((c) => c.ts_code === state.promptsPayloadCode)) {
    return state.promptsPayloadCode;
  }
  if (state.selectedCode && candidates.some((c) => c.ts_code === state.selectedCode)) {
    return state.selectedCode;
  }
  return candidates[0]?.ts_code || "";
}

async function refreshPromptsData({ source } = {}) {
  if (source) {
    state.promptsSourcePref = source;
  }
  const scope = chatRunScope();
  const qs = new URLSearchParams();
  if (scope.db) qs.set("db", scope.db);
  if (scope.variant) qs.set("variant", scope.variant);
  if (scope.cadence) qs.set("cadence", scope.cadence);
  if (scope.model_source) qs.set("model_source", scope.model_source);
  if (scope.as_of) qs.set("as_of", scope.as_of);
  if (scope.hash) qs.set("hash", scope.hash);
  if (state.promptsSourcePref) qs.set("source", state.promptsSourcePref);

  state.promptsLoading = true;
  state.promptsError = "";
  if (state.activeTab === "prompts") renderPromptsView();
  if (state.promptModal.open) renderPromptModal();

  try {
    state.promptsData = await apiJson(`/api/prompts?${qs.toString()}`);
  } catch (err) {
    state.promptsError = err.message || String(err);
  } finally {
    state.promptsLoading = false;
    renderTabsHeader();
    if (state.activeTab === "prompts") renderPromptsView();
    if (state.promptModal.open) renderPromptModal();
  }
}

/** 若运行中存在尚未刷盘的实时内存进展，优先用最新内存态构造单股 User JSON 预览。 */
function buildInMemoryLivePayload(roleKey, tsCode) {
  if (!state.activeJob?.running || !isViewingActiveRun()) return null;
  if (!state.run?.request || roleKey.startsWith("chat:") || roleKey === "final_pick") return null;
  const req = state.run.request;
  const cand = (req.candidates || []).find((c) => c.ts_code === tsCode);
  if (!cand) return null;
  const st = state.stockState[tsCode] || { analysts: {}, debate: {}, verdict: null };
  const payload = { context: req.context, candidate: cand };
  const missing = [];

  if (["bull", "bear", "bull_counter", "risk_chair"].includes(roleKey)) {
    const analysts = [];
    for (const r of ["model_analyst", "chart_analyst"]) {
      if (st.analysts?.[r]) analysts.push(st.analysts[r]);
      else missing.push(r);
    }
    payload.analyst_views = analysts;
  }
  const needRounds =
    roleKey === "bear"
      ? ["bull"]
      : roleKey === "bull_counter"
        ? ["bull", "bear"]
        : roleKey === "risk_chair"
          ? ["bull", "bear", "bull_counter"]
          : [];
  if (needRounds.length > 0) {
    const rounds = [];
    for (const r of needRounds) {
      if (st.debate?.[r]) rounds.push(st.debate[r]);
      else missing.push(r);
    }
    payload.debate = rounds;
  }
  if (roleKey === "risk_chair") {
    payload.roster =
      Array.isArray(req.roster) && req.roster.length > 0
        ? req.roster
        : (req.candidates || []).map((c) => ({
            ts_code: c.ts_code,
            name: c.name,
            industry: c.industry,
            rank: c.model?.rank ?? null,
            pct: c.model?.pct ?? null,
            r5: c.features?.r5 ?? null,
            r20: c.features?.r20 ?? null,
            from_low60: c.features?.from_low60 ?? null,
            dev_ma20: c.features?.dev_ma20 ?? null,
            vol5: c.features?.vol5 ?? null,
            flags: (c.flags || []).map((f) => f.code),
          }));
  }

  const wire = JSON.stringify(payload);
  return {
    role: roleKey,
    ts_code: tsCode,
    stock_name: cand.name || tsCode,
    ready: missing.length === 0,
    missing_deps: missing,
    top_keys: Object.keys(payload),
    char_count: wire.length,
    est_tokens: Math.max(1, Math.round(wire.length * 0.32)),
    payload,
  };
}

async function ensureRolePayload(roleKey, tsCode, { force = false } = {}) {
  const effectiveCode = roleKey === "final_pick" ? "-" : tsCode || resolveDefaultPayloadCode(roleKey);
  const liveLocal = buildInMemoryLivePayload(roleKey, effectiveCode);
  if (liveLocal) {
    const key = promptPayloadCacheKey(roleKey, effectiveCode);
    state.promptsPayloadCache[key] = { ok: true, data: liveLocal };
    if (state.activeTab === "prompts") renderPromptsView();
    if (state.promptModal.open) renderPromptModal();
    return;
  }

  const key = promptPayloadCacheKey(roleKey, effectiveCode);
  if (!force && state.promptsPayloadCache[key]) return;
  if (state.promptsPayloadLoadingKey === key) return;

  const scope = chatRunScope();
  const qs = new URLSearchParams();
  if (scope.db) qs.set("db", scope.db);
  if (scope.variant) qs.set("variant", scope.variant);
  if (scope.cadence) qs.set("cadence", scope.cadence);
  if (scope.model_source) qs.set("model_source", scope.model_source);
  if (scope.as_of) qs.set("as_of", scope.as_of);
  if (scope.hash) qs.set("hash", scope.hash);
  qs.set("role", roleKey);
  if (effectiveCode && effectiveCode !== "-") qs.set("ts_code", effectiveCode);

  state.promptsPayloadLoadingKey = key;
  if (state.activeTab === "prompts") renderPromptsView();
  if (state.promptModal.open) renderPromptModal();

  try {
    const data = await apiJson(`/api/prompts/payload?${qs.toString()}`);
    state.promptsPayloadCache[key] = { ok: true, data };
  } catch (err) {
    state.promptsPayloadCache[key] = { ok: false, error: err.message || String(err) };
  } finally {
    if (state.promptsPayloadLoadingKey === key) {
      state.promptsPayloadLoadingKey = "";
    }
    if (state.activeTab === "prompts") renderPromptsView();
    if (state.promptModal.open) renderPromptModal();
  }
}

function getSelectedPromptEntity(roleKey) {
  const data = state.promptsData;
  if (!data) return null;
  if (roleKey === "__methodology__") {
    return { kind: "methodology", item: data.methodology };
  }
  if (roleKey.startsWith("chat:")) {
    const pid = roleKey.slice(5);
    const found = (data.chat_roles || []).find((r) => r.id === pid) || data.chat_roles?.[0];
    return found ? { kind: "chat", item: found, tools: data.chat_tools || [] } : null;
  }
  const found = (data.debate_roles || []).find((r) => r.id === roleKey) || data.debate_roles?.[0];
  return found ? { kind: "debate", item: found } : null;
}

function formatSchemaPretty(rawSchema) {
  try {
    return JSON.stringify(JSON.parse(rawSchema), null, 2);
  } catch {
    return rawSchema;
  }
}

function renderDebatePresetBlocksHtml(roleObj, methodology) {
  const vis = roleObj.input_visibility || {};
  const prettySchema = formatSchemaPretty(roleObj.output_schema);
  return `
    <div class="prompt-blocks">
      <section class="prompt-block prompt-block-primary">
        <div class="prompt-block-head">
          <div>
            <span class="prompt-badge prompt-badge-preset">预置职责</span>
            <strong class="prompt-block-title">## 你的职责（来自 docs/debate-methodology.md）</strong>
          </div>
          <button class="btn btn-ghost btn-sm" type="button" data-copy-slot="role_brief">复制职责</button>
        </div>
        <pre class="prompt-code-block">${esc(roleObj.role_brief)}</pre>
      </section>

      ${
        roleObj.constraint
          ? `
        <section class="prompt-block">
          <div class="prompt-block-head">
            <div>
              <span class="prompt-badge prompt-badge-rule">硬性约束</span>
              <strong class="prompt-block-title">## ${esc(roleObj.constraint.title)}</strong>
            </div>
          </div>
          <div class="prompt-text-body">${esc(roleObj.constraint.body)}</div>
        </section>
      `
          : ""
      }

      <section class="prompt-block">
        <div class="prompt-block-head">
          <div>
            <span class="prompt-badge prompt-badge-rule">输出契约</span>
            <strong class="prompt-block-title">## 输出格式（JSON Schema）</strong>
          </div>
          <button class="btn btn-ghost btn-sm" type="button" data-copy-slot="output_schema">复制 Schema</button>
        </div>
        <pre class="prompt-code-block mono">${esc(prettySchema)}</pre>
      </section>

      <div class="prompt-two-col">
        <section class="prompt-block">
          <div class="prompt-block-head">
            <div>
              <span class="prompt-badge prompt-badge-rule">长度约束</span>
              <strong class="prompt-block-title">## 长度</strong>
            </div>
          </div>
          <div class="prompt-text-body">${esc(roleObj.length_rule)}</div>
        </section>

        <section class="prompt-block">
          <div class="prompt-block-head">
            <div>
              <span class="prompt-badge prompt-badge-vis">输入视野</span>
              <strong class="prompt-block-title">实际可见字段与隔离边界</strong>
            </div>
          </div>
          <div class="prompt-text-body">${esc(vis.summary || "")}</div>
          ${
            Array.isArray(vis.visible_fields) && vis.visible_fields.length
              ? `<div class="prompt-field-pills">${vis.visible_fields
                  .map((f) => `<span class="citation-chip mono">${esc(f)}</span>`)
                  .join("")}</div>`
              : ""
          }
          ${vis.hidden_note ? `<div class="muted-text" style="margin-top:6px">隔离说明：${esc(vis.hidden_note)}</div>` : ""}
        </section>
      </div>

      <section class="prompt-block">
        <div class="prompt-block-head">
          <div>
            <span class="prompt-badge prompt-badge-rule">通用禁令</span>
            <strong class="prompt-block-title">## 禁令（7 角色共用）</strong>
          </div>
        </div>
        <ul class="prompt-rules-list">
          ${(roleObj.common_rules || []).map((r) => `<li>${esc(r)}</li>`).join("")}
        </ul>
      </section>

      <details class="prompt-collapsible">
        <summary>
          <span>共享方法论正文（所有角色 System Prompt 顶部共用 · ${methodology.char_count.toLocaleString()} 字 / 约 ${methodology.est_tokens.toLocaleString()} tok）</span>
        </summary>
        <div class="prompt-collapsible-body">
          <div class="prompt-collapsible-bar">
            <span class="muted-text mono">${esc(state.promptsData?.methodology_path || "docs/debate-methodology.md")}</span>
            <button class="btn btn-ghost btn-sm" type="button" data-copy-slot="methodology">复制方法论</button>
          </div>
          <pre class="prompt-code-block">${esc(methodology.text)}</pre>
        </div>
      </details>
    </div>
  `;
}

function renderChatPresetBlocksHtml(chatRole, tools, methodology) {
  return `
    <div class="prompt-blocks">
      <section class="prompt-block prompt-block-primary">
        <div class="prompt-block-head">
          <div>
            <span class="prompt-badge prompt-badge-preset">对话身份</span>
            <strong class="prompt-block-title">${esc(chatRole.name)}</strong>
            ${
              chatRole.uses_methodology_role_brief
                ? `<span class="tag-neutral" style="margin-left:6px">自动复用 debate-methodology 角色职责</span>`
                : `<span class="tag-neutral" style="margin-left:6px">内置复盘助手身份</span>`
            }
          </div>
          <button class="btn btn-ghost btn-sm" type="button" data-copy-slot="chat_identity">复制身份设定</button>
        </div>
        <pre class="prompt-code-block">${esc(chatRole.preset_identity)}</pre>
      </section>

      <section class="prompt-block">
        <div class="prompt-block-head">
          <div>
            <span class="prompt-badge prompt-badge-vis">只读工具</span>
            <strong class="prompt-block-title">挂载工具清单（共 ${tools.length} 个，绝不改写账本）</strong>
          </div>
        </div>
        <div class="prompt-tools-grid">
          ${tools
            .map(
              (t) => `
            <div class="prompt-tool-card">
              <div class="prompt-tool-head">
                <strong class="mono">${esc(t.name)}</strong>
                <span class="pill pill-soft">${esc(t.label)}</span>
              </div>
              <div class="prompt-tool-desc">${esc(t.description)}</div>
            </div>
          `,
            )
            .join("")}
        </div>
      </section>

      <details class="prompt-collapsible">
        <summary>
          <span>共享方法论正文（对话 System Prompt 底座 · ${methodology.char_count.toLocaleString()} 字 / 约 ${methodology.est_tokens.toLocaleString()} tok）</span>
        </summary>
        <div class="prompt-collapsible-body">
          <div class="prompt-collapsible-bar">
            <span class="muted-text mono">${esc(state.promptsData?.methodology_path || "docs/debate-methodology.md")}</span>
            <button class="btn btn-ghost btn-sm" type="button" data-copy-slot="methodology">复制方法论</button>
          </div>
          <pre class="prompt-code-block">${esc(methodology.text)}</pre>
        </div>
      </details>
    </div>
  `;
}

function renderFullSystemPromptHtml(fullText, { subtitle = "", copySlot = "full_system", note = "" } = {}) {
  const chars = (fullText || "").length;
  const estTok = Math.max(1, Math.round(chars * 0.45));
  return `
    <div class="prompt-blocks">
      <section class="prompt-block">
        <div class="prompt-block-head">
          <div>
            <span class="prompt-badge prompt-badge-system">System Prompt 原文</span>
            <strong class="prompt-block-title">${esc(subtitle || "实际发送给大模型的完整 system 消息")}</strong>
          </div>
          <div class="prompt-block-tools">
            <span class="muted-text mono">${chars.toLocaleString()} 字 · 约 ${estTok.toLocaleString()} tok</span>
            <button class="btn btn-primary btn-sm" type="button" data-copy-slot="${esc(copySlot)}">一键复制全文</button>
          </div>
        </div>
        ${note ? `<div class="prompt-note-banner">${esc(note)}</div>` : ""}
        <pre class="prompt-code-block prompt-code-tall">${esc(fullText)}</pre>
      </section>
    </div>
  `;
}

function renderUserPayloadPanelHtml(roleKey, tsCode, { showStockPicker = true, pickerAttr = "data-payload-stock-select" } = {}) {
  const isChat = roleKey.startsWith("chat:");
  const isFinal = roleKey === "final_pick";
  const candidates = state.run?.request?.candidates || [];
  const effectiveCode = isFinal ? "-" : tsCode || resolveDefaultPayloadCode(roleKey);
  const cacheKey = promptPayloadCacheKey(roleKey, effectiveCode);
  const entry = state.promptsPayloadCache[cacheKey];
  const isLoading = state.promptsPayloadLoadingKey === cacheKey;

  if (!entry && !isLoading) {
    queueMicrotask(() => ensureRolePayload(roleKey, effectiveCode));
  }

  const pickerHtml =
    showStockPicker && !isFinal && candidates.length > 0
      ? `
        <label class="prompt-stock-picker">
          <span class="muted-text">${isChat ? "聚焦挂载个股" : "选择候选股"}</span>
          <select class="ctrl-select ctrl-mini" ${pickerAttr}>
            ${isChat ? `<option value="" ${!effectiveCode ? "selected" : ""}>不额外挂载个股</option>` : ""}
            ${candidates
              .map(
                (c) =>
                  `<option value="${esc(c.ts_code)}" ${c.ts_code === effectiveCode ? "selected" : ""}>#${String(c.model?.rank ?? "").padStart(2, "0")} ${esc(c.name)} (${esc(c.ts_code)})</option>`,
              )
              .join("")}
          </select>
        </label>
      `
      : "";

  if (isLoading && !entry) {
    return `
      <div class="prompt-blocks">
        <div class="prompt-payload-toolbar">${pickerHtml}</div>
        <div class="pending-placeholder">正在组装实际 User JSON…</div>
      </div>
    `;
  }

  if (entry && !entry.ok) {
    return `
      <div class="prompt-blocks">
        <div class="prompt-payload-toolbar">${pickerHtml}</div>
        <div class="prompt-warn-banner">
          <strong>暂无法预览实际 User JSON：</strong>${esc(entry.error)}
          <div class="muted-text" style="margin-top:4px">请先在顶部选择一次历史运行记录，或点击「演练预览」生成当日候选快照。</div>
        </div>
      </div>
    `;
  }

  const data = entry?.data;
  if (!data) {
    return `
      <div class="prompt-blocks">
        <div class="prompt-payload-toolbar">${pickerHtml}</div>
        <div class="pending-placeholder">正在加载…</div>
      </div>
    `;
  }

  const jsonText = state.promptsPayloadCompact
    ? JSON.stringify(data.payload)
    : JSON.stringify(data.payload, null, 2);

  const statusHtml = data.ready
    ? `<span class="status-badge status-done">${isChat ? "实时常驻上下文已就绪" : "完整入参已就绪"}</span>`
    : `<span class="status-badge status-running">前序未完 · 缺 ${(data.missing_deps || []).join(" / ")}</span>`;

  return `
    <div class="prompt-blocks">
      <div class="prompt-payload-toolbar">
        <div class="prompt-payload-toolbar-left">
          ${pickerHtml}
          ${statusHtml}
        </div>
        <div class="prompt-block-tools">
          <span class="muted-text mono">${data.char_count.toLocaleString()} 字符 · 约 ${data.est_tokens.toLocaleString()} tok</span>
          <button class="chip ${!state.promptsPayloadCompact ? "active" : ""}" type="button" data-payload-format="pretty">格式化</button>
          <button class="chip ${state.promptsPayloadCompact ? "active" : ""}" type="button" data-payload-format="compact">单行原文</button>
          <button class="btn btn-primary btn-sm" type="button" data-copy-slot="user_payload" data-copy-role="${esc(roleKey)}" data-copy-code="${esc(effectiveCode)}">复制 JSON</button>
        </div>
      </div>

      ${
        !data.ready && data.missing_deps?.length
          ? `<div class="prompt-note-banner">提示：当前股票的前序角色（<strong>${esc(data.missing_deps.join("、"))}</strong>）尚未执行完成，下方展示的是已就绪字段的结构预览；完整跑完后将自动包含对应前序发言。</div>`
          : ""
      }

      <section class="prompt-block">
        <div class="prompt-block-head">
          <div class="prompt-field-pills" style="margin-top:0">
            <span class="muted-text">顶层字段：</span>
            ${(data.top_keys || []).map((k) => `<span class="citation-chip mono">${esc(k)}</span>`).join("")}
          </div>
          <span class="muted-text mono">${esc(data.stock_name)} ${data.ts_code && data.ts_code !== "-" ? `(${esc(data.ts_code)})` : ""}</span>
        </div>
        <pre class="prompt-code-block prompt-code-tall mono">${esc(jsonText)}</pre>
      </section>
    </div>
  `;
}

function getCopyTextBySlot(slot, roleKey, tsCode) {
  const data = state.promptsData;
  if (!data) return { text: "", label: "内容" };
  if (slot === "methodology") {
    return { text: data.methodology?.text || "", label: "共享方法论正文" };
  }
  if (slot === "user_payload") {
    const effectiveCode = roleKey === "final_pick" ? "-" : tsCode || resolveDefaultPayloadCode(roleKey);
    const entry = state.promptsPayloadCache[promptPayloadCacheKey(roleKey, effectiveCode)];
    if (!entry?.ok) return { text: "", label: "User JSON" };
    const text = state.promptsPayloadCompact
      ? JSON.stringify(entry.data.payload)
      : JSON.stringify(entry.data.payload, null, 2);
    return { text, label: `${roleKey} 实际 User JSON` };
  }
  const ent = getSelectedPromptEntity(roleKey);
  if (!ent) return { text: "", label: "内容" };
  if (ent.kind === "debate") {
    if (slot === "role_brief") return { text: ent.item.role_brief, label: `${ent.item.name} 预置职责` };
    if (slot === "output_schema") return { text: formatSchemaPretty(ent.item.output_schema), label: `${ent.item.name} 输出 Schema` };
    if (slot === "full_system") return { text: ent.item.system_prompt, label: `${ent.item.name} 完整 System Prompt` };
  }
  if (ent.kind === "chat") {
    if (slot === "chat_identity") return { text: ent.item.preset_identity, label: `${ent.item.name} 对话身份设定` };
    if (slot === "full_system") {
      const effectiveCode = tsCode || resolveDefaultPayloadCode(roleKey);
      const entry = state.promptsPayloadCache[promptPayloadCacheKey(roleKey, effectiveCode)];
      const full = entry?.ok && entry.data.full_system_prompt ? entry.data.full_system_prompt : ent.item.system_prompt_base;
      return { text: full, label: `${ent.item.name} 对话 System Prompt` };
    }
  }
  return { text: "", label: "内容" };
}

function renderPromptsView() {
  if (state.promptsLoading && !state.promptsData) {
    dom.promptsContent.innerHTML = `<div class="empty-state"><div class="empty-title">正在加载提示词与角色配置…</div></div>`;
    return;
  }
  if (state.promptsError && !state.promptsData) {
    dom.promptsContent.innerHTML = `<div class="empty-state"><div class="empty-title">加载提示词失败</div><p class="empty-desc">${esc(state.promptsError)}</p></div>`;
    return;
  }
  const data = state.promptsData;
  if (!data) {
    dom.promptsContent.innerHTML = `<div class="empty-state"><div class="empty-title">暂无提示词数据</div></div>`;
    return;
  }

  const selId = state.promptsSelectedId || "model_analyst";
  const ent = getSelectedPromptEntity(selId);
  const debateRoles = data.debate_roles || [];
  const chatRoles = data.chat_roles || [];
  const isFrozen = data.source === "frozen";

  const sourceBannerHtml = `
    <section class="card prompt-source-bar">
      <div class="prompt-source-info">
        <span class="status-badge ${isFrozen ? "status-done" : "status-idle"}">
          ${isFrozen ? `本次运行冻结版 (${esc(data.as_of || "")} · ${esc(data.hash_prefix || "")})` : "当前工作区文档"}
        </span>
        <span class="muted-text mono">方法论源：${esc(data.methodology_path)} · 共享规则：${esc(data.rules_path)}</span>
        ${
          data.differs_from_workspace
            ? `<span class="tag-warn">⚠ 冻结版与当前工作区文档存在差异${data.changed_roles?.length ? `（角色：${esc(data.changed_roles.join("、"))}）` : ""}</span>`
            : ""
        }
      </div>
      ${
        data.has_frozen
          ? `
        <div class="prompt-source-switch">
          <button class="chip ${isFrozen ? "active" : ""}" type="button" data-prompt-source="frozen">本次冻结版</button>
          <button class="chip ${!isFrozen ? "active" : ""}" type="button" data-prompt-source="workspace">当前工作区版</button>
        </div>
      `
          : ""
      }
    </section>
  `;

  const navHtml = `
    <aside class="card prompt-nav">
      <div class="prompt-nav-group">
        <div class="prompt-nav-label">公共底座</div>
        <button class="prompt-nav-item ${selId === "__methodology__" ? "active" : ""}" type="button" data-prompt-select="__methodology__">
          <span class="prompt-nav-title">共享方法论正文</span>
          <span class="prompt-nav-meta mono">
            ${data.methodology_changed ? `<span class="prompt-dot-warn" title="与当前工作区文档不同">●</span>` : ""}
            ${data.methodology.char_count.toLocaleString()}字
          </span>
        </button>
      </div>

      <div class="prompt-nav-group">
        <div class="prompt-nav-label">辩论 Agent（${debateRoles.length} 个）</div>
        ${debateRoles
          .map(
            (r) => `
          <button class="prompt-nav-item ${selId === r.id ? "active" : ""}" type="button" data-prompt-select="${esc(r.id)}">
            <div>
              <div class="prompt-nav-title">${esc(r.name)}</div>
              <div class="prompt-nav-sub mono">${esc(r.id)}</div>
            </div>
            <span class="prompt-nav-meta mono">
              ${r.changed_from_workspace ? `<span class="prompt-dot-warn" title="与当前工作区文档不同">●</span>` : ""}
              ~${r.est_tokens}t
            </span>
          </button>
        `,
          )
          .join("")}
      </div>

      <div class="prompt-nav-group">
        <div class="prompt-nav-label">AI 复盘对话视角（${chatRoles.length} 个）</div>
        ${chatRoles
          .map((c) => {
            const key = `chat:${c.id}`;
            return `
            <button class="prompt-nav-item ${selId === key ? "active" : ""}" type="button" data-prompt-select="${esc(key)}">
              <div>
                <div class="prompt-nav-title">${esc(c.name.replace(/\s*\([^)]+\)$/, ""))}</div>
                <div class="prompt-nav-sub mono">${esc(c.id)}</div>
              </div>
              <span class="prompt-nav-meta mono">~${c.est_tokens}t</span>
            </button>
          `;
          })
          .join("")}
      </div>
    </aside>
  `;

  let detailHtml = "";
  if (ent?.kind === "methodology") {
    detailHtml = `
      <div class="prompt-detail-head">
        <div>
          <h2 class="prompt-detail-title">共享方法论正文</h2>
          <p class="card-subtitle">所有 7 个辩论 Agent 与对话 Agent 的共同底座，位于 ${esc(data.methodology_path)}，改动正文将自动触发新 input_hash。</p>
        </div>
        <div class="prompt-block-tools">
          <span class="muted-text mono">${data.methodology.char_count.toLocaleString()} 字 · 约 ${data.methodology.est_tokens.toLocaleString()} tok</span>
          <button class="btn btn-primary btn-sm" type="button" data-copy-slot="methodology">复制方法论</button>
        </div>
      </div>
      <pre class="prompt-code-block prompt-code-tall">${esc(data.methodology.text)}</pre>
    `;
  } else if (ent?.kind === "debate") {
    const r = ent.item;
    const mode = state.promptsViewMode || "preset";
    const payloadCode = resolveDefaultPayloadCode(r.id);
    let paneHtml = "";
    if (mode === "preset") {
      paneHtml = renderDebatePresetBlocksHtml(r, data.methodology);
    } else if (mode === "system") {
      paneHtml = renderFullSystemPromptHtml(r.system_prompt, {
        subtitle: `${r.name} (${r.id}) · 实际发送给大模型的完整 System Prompt`,
        copySlot: "full_system",
        note: "此全文由「共享方法论正文 + 你的职责 + 硬性约束 + 输出格式 + 长度 + 禁令」拼接而成，与 debate_agent/src/prompts.ts 逐字一致。",
      });
    } else {
      paneHtml = renderUserPayloadPanelHtml(r.id, payloadCode, { showStockPicker: true });
    }

    detailHtml = `
      <div class="prompt-detail-head">
        <div>
          <div class="prompt-detail-title-row">
            <h2 class="prompt-detail-title">${esc(r.name)}</h2>
            <span class="role-code mono">${esc(r.id)}</span>
            <span class="pill pill-soft">${esc(r.phase_label)}</span>
          </div>
          <p class="card-subtitle">System Prompt 共 ${r.char_count.toLocaleString()} 字（约 ${r.est_tokens.toLocaleString()} tok）· ${esc(r.input_visibility?.summary || "")}</p>
        </div>
        <div class="prompt-mode-tabs">
          <button class="chip ${mode === "preset" ? "active" : ""}" type="button" data-prompt-mode="preset">预置提示词（分块）</button>
          <button class="chip ${mode === "system" ? "active" : ""}" type="button" data-prompt-mode="system">完整 System Prompt</button>
          <button class="chip ${mode === "payload" ? "active" : ""}" type="button" data-prompt-mode="payload">${r.id === "final_pick" ? "终选实际 User JSON" : "单股实际 User JSON"}</button>
        </div>
      </div>
      ${paneHtml}
    `;
  } else if (ent?.kind === "chat") {
    const c = ent.item;
    const mode = state.promptsViewMode || "preset";
    const roleKey = `chat:${c.id}`;
    const payloadCode = resolveDefaultPayloadCode(roleKey);
    let paneHtml = "";
    if (mode === "preset") {
      paneHtml = renderChatPresetBlocksHtml(c, ent.tools, data.methodology);
    } else if (mode === "system") {
      const cacheKey = promptPayloadCacheKey(roleKey, payloadCode);
      const cached = state.promptsPayloadCache[cacheKey];
      if (!cached && state.promptsPayloadLoadingKey !== cacheKey) {
        queueMicrotask(() => ensureRolePayload(roleKey, payloadCode));
      }
      const fullText =
        cached?.ok && cached.data.full_system_prompt ? cached.data.full_system_prompt : c.system_prompt_base;
      const note =
        cached?.ok && cached.data.full_system_prompt
          ? "已包含当前页面的实时常驻运行上下文 JSON（与点击发送时实际送入大模型的 system 消息完全一致）。"
          : "下方为不含动态常驻上下文的基础 System Prompt；若已加载运行记录，正在同步实时上下文…";
      paneHtml = renderFullSystemPromptHtml(fullText, {
        subtitle: `AI 复盘对话 · ${c.name}`,
        copySlot: "full_system",
        note,
      });
    } else {
      paneHtml = renderUserPayloadPanelHtml(roleKey, payloadCode, { showStockPicker: true });
    }

    detailHtml = `
      <div class="prompt-detail-head">
        <div>
          <div class="prompt-detail-title-row">
            <h2 class="prompt-detail-title">${esc(c.name)}</h2>
            <span class="role-code mono">${esc(c.id)}</span>
            <span class="pill pill-soft">对话 Agent · 6 只读工具</span>
          </div>
          <p class="card-subtitle">${esc(c.desc)}</p>
        </div>
        <div class="prompt-mode-tabs">
          <button class="chip ${mode === "preset" ? "active" : ""}" type="button" data-prompt-mode="preset">预置身份与工具</button>
          <button class="chip ${mode === "system" ? "active" : ""}" type="button" data-prompt-mode="system">完整 System Prompt</button>
          <button class="chip ${mode === "payload" ? "active" : ""}" type="button" data-prompt-mode="payload">常驻上下文 JSON</button>
        </div>
      </div>
      ${paneHtml}
    `;
  }

  dom.promptsContent.innerHTML = `
    ${sourceBannerHtml}
    <div class="prompt-studio-grid">
      ${navHtml}
      <section class="card prompt-detail-card">${detailHtml}</section>
    </div>
  `;
}

function openPromptModal({ roleKey, tsCode = "", pane = "preset" }) {
  restorePromptModalChrome();
  state.promptModal = {
    open: true,
    roleKey: roleKey || "model_analyst",
    tsCode: roleKey === "final_pick" ? "-" : tsCode || resolveDefaultPayloadCode(roleKey),
    pane: pane || "preset",
  };
  dom.promptModal.hidden = false;
  if (!state.promptsData && !state.promptsLoading) {
    refreshPromptsData();
  }
  if (state.promptModal.pane === "payload" || state.promptModal.roleKey.startsWith("chat:")) {
    ensureRolePayload(state.promptModal.roleKey, state.promptModal.tsCode);
  }
  renderPromptModal();
}

function closePromptModal() {
  if (!state.promptModal.open) return;
  state.promptModal.open = false;
  dom.promptModal.hidden = true;
}

function renderPromptModal() {
  if (!state.promptModal.open) return;
  if (state.promptModal.roleKey === "__ml_explain__") {
    renderMlExplainModalBody();
    return;
  }
  const { roleKey, tsCode, pane } = state.promptModal;
  const data = state.promptsData;

  dom.promptModalTabs.querySelectorAll("[data-modal-pane]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.modalPane === pane);
    if (btn.dataset.modalPane === "payload") {
      btn.textContent = roleKey.startsWith("chat:")
        ? "常驻上下文 JSON"
        : roleKey === "final_pick"
          ? "终选实际 User JSON"
          : "本股实际 User JSON";
    }
  });

  if (state.promptsLoading && !data) {
    dom.promptModalTitle.textContent = "正在加载提示词…";
    dom.promptModalSubtitle.textContent = "";
    dom.promptModalBody.innerHTML = `<div class="pending-placeholder">加载中…</div>`;
    return;
  }
  if (!data) {
    dom.promptModalTitle.textContent = "加载提示词失败";
    dom.promptModalSubtitle.textContent = state.promptsError || "";
    dom.promptModalBody.innerHTML = `<div class="pending-placeholder">${esc(state.promptsError || "暂无数据")}</div>`;
    return;
  }

  const ent = getSelectedPromptEntity(roleKey);
  if (!ent) return;

  const cand = (state.run?.request?.candidates || []).find((c) => c.ts_code === tsCode);
  const stockTag = cand ? ` · ${cand.name} (${cand.ts_code})` : roleKey === "final_pick" ? " · 全批终选" : "";
  const sourceLabel = data.source === "frozen" ? `本次冻结版 (${data.as_of})` : "当前工作区文档";

  if (ent.kind === "debate") {
    const r = ent.item;
    dom.promptModalTitle.textContent = `${r.name} (${r.id})${stockTag}`;
    dom.promptModalSubtitle.textContent = `${sourceLabel} · ${r.phase_label} · System Prompt ${r.char_count.toLocaleString()} 字（约 ${r.est_tokens.toLocaleString()} tok）`;
    if (pane === "preset") {
      dom.promptModalBody.innerHTML = renderDebatePresetBlocksHtml(r, data.methodology);
    } else if (pane === "system") {
      dom.promptModalBody.innerHTML = renderFullSystemPromptHtml(r.system_prompt, {
        subtitle: `${r.name} (${r.id}) 完整 System Prompt`,
        copySlot: "full_system",
      });
    } else {
      dom.promptModalBody.innerHTML = renderUserPayloadPanelHtml(r.id, tsCode, {
        showStockPicker: true,
        pickerAttr: "data-modal-stock-select",
      });
    }
  } else if (ent.kind === "chat") {
    const c = ent.item;
    dom.promptModalTitle.textContent = `AI 复盘对话 · ${c.name}`;
    dom.promptModalSubtitle.textContent = `${sourceLabel} · ${c.desc}`;
    if (pane === "preset") {
      dom.promptModalBody.innerHTML = renderChatPresetBlocksHtml(c, ent.tools, data.methodology);
    } else if (pane === "system") {
      const cacheKey = promptPayloadCacheKey(roleKey, tsCode);
      const cached = state.promptsPayloadCache[cacheKey];
      const fullText =
        cached?.ok && cached.data.full_system_prompt ? cached.data.full_system_prompt : c.system_prompt_base;
      dom.promptModalBody.innerHTML = renderFullSystemPromptHtml(fullText, {
        subtitle: `${c.name} 完整对话 System Prompt`,
        copySlot: "full_system",
        note: cached?.ok && cached.data.full_system_prompt ? "已包含当前运行实时常驻上下文 JSON。" : "基础对话 System Prompt。",
      });
    } else {
      dom.promptModalBody.innerHTML = renderUserPayloadPanelHtml(roleKey, tsCode, {
        showStockPicker: true,
        pickerAttr: "data-modal-stock-select",
      });
    }
  }
}

function bindPromptEvents() {
  // 卡片右上角「提示词」快捷入口（单股辩论 + 终选）
  const handleInspectClick = (ev) => {
    const btn = ev.target.closest?.("[data-inspect-role]");
    if (!btn) return;
    ev.stopPropagation();
    openPromptModal({
      roleKey: btn.dataset.inspectRole || "model_analyst",
      tsCode: btn.dataset.inspectCode || "",
      pane: "preset",
    });
  };
  dom.stockContent.addEventListener("click", handleInspectClick);
  dom.finalContent.addEventListener("click", handleInspectClick);

  // 对话栏顶部「提示词」按钮
  if (dom.btnChatPromptInfo) {
    dom.btnChatPromptInfo.addEventListener("click", () => {
      openPromptModal({
        roleKey: `chat:${state.chatRole || "research_copilot"}`,
        tsCode: state.selectedCode || "",
        pane: "preset",
      });
    });
  }

  // 主视图「提示词与角色」Tab 内交互委托
  dom.promptsContent.addEventListener("click", (ev) => {
    const srcBtn = ev.target.closest?.("[data-prompt-source]");
    if (srcBtn) {
      refreshPromptsData({ source: srcBtn.dataset.promptSource });
      return;
    }
    const navBtn = ev.target.closest?.("[data-prompt-select]");
    if (navBtn) {
      state.promptsSelectedId = navBtn.dataset.promptSelect;
      renderPromptsView();
      return;
    }
    const modeBtn = ev.target.closest?.("[data-prompt-mode]");
    if (modeBtn) {
      state.promptsViewMode = modeBtn.dataset.promptMode;
      if (state.promptsViewMode === "payload") {
        ensureRolePayload(state.promptsSelectedId, resolveDefaultPayloadCode(state.promptsSelectedId));
      }
      renderPromptsView();
      return;
    }
    const fmtBtn = ev.target.closest?.("[data-payload-format]");
    if (fmtBtn) {
      state.promptsPayloadCompact = fmtBtn.dataset.payloadFormat === "compact";
      renderPromptsView();
      return;
    }
    const copyBtn = ev.target.closest?.("[data-copy-slot]");
    if (copyBtn) {
      const roleKey = copyBtn.dataset.copyRole || state.promptsSelectedId;
      const code = copyBtn.dataset.copyCode ?? resolveDefaultPayloadCode(roleKey);
      const { text, label } = getCopyTextBySlot(copyBtn.dataset.copySlot, roleKey, code);
      copyTextWithToast(text, label);
    }
  });

  dom.promptsContent.addEventListener("change", (ev) => {
    const sel = ev.target.closest?.("[data-payload-stock-select]");
    if (!sel) return;
    state.promptsPayloadCode = sel.value;
    ensureRolePayload(state.promptsSelectedId, state.promptsPayloadCode);
    renderPromptsView();
  });

  // 弹窗交互
  dom.btnModalClose.addEventListener("click", closePromptModal);
  dom.promptModal.addEventListener("click", (ev) => {
    if (ev.target === dom.promptModal) closePromptModal();
  });
  dom.btnModalOpenTab.addEventListener("click", () => {
    state.promptsSelectedId = state.promptModal.roleKey;
    state.promptsViewMode = state.promptModal.pane;
    if (state.promptModal.tsCode && state.promptModal.tsCode !== "-") {
      state.promptsPayloadCode = state.promptModal.tsCode;
    }
    closePromptModal();
    switchTab("prompts");
  });
  dom.promptModalTabs.addEventListener("click", (ev) => {
    const btn = ev.target.closest?.("[data-modal-pane]");
    if (!btn) return;
    state.promptModal.pane = btn.dataset.modalPane;
    if (state.promptModal.pane === "payload") {
      ensureRolePayload(state.promptModal.roleKey, state.promptModal.tsCode);
    }
    renderPromptModal();
  });
  dom.promptModalBody.addEventListener("click", (ev) => {
    const fmtBtn = ev.target.closest?.("[data-payload-format]");
    if (fmtBtn) {
      state.promptsPayloadCompact = fmtBtn.dataset.payloadFormat === "compact";
      renderPromptModal();
      return;
    }
    const copyBtn = ev.target.closest?.("[data-copy-slot]");
    if (copyBtn) {
      const roleKey = copyBtn.dataset.copyRole || state.promptModal.roleKey;
      const code = copyBtn.dataset.copyCode ?? state.promptModal.tsCode;
      const { text, label } = getCopyTextBySlot(copyBtn.dataset.copySlot, roleKey, code);
      copyTextWithToast(text, label);
    }
  });
  dom.promptModalBody.addEventListener("change", (ev) => {
    const sel = ev.target.closest?.("[data-modal-stock-select]");
    if (!sel) return;
    state.promptModal.tsCode = sel.value;
    ensureRolePayload(state.promptModal.roleKey, state.promptModal.tsCode);
    renderPromptModal();
  });

  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && state.promptModal.open) {
      closePromptModal();
    }
  });
}

// ---------------------------------------------------------------------------
// 多模型对比与净值（视图 D）+ 单股归因瀑布图 + 69 特征分组面板
// ---------------------------------------------------------------------------

const ML_MODEL_IDS = ["xgb_F69", "rf_F69", "lstm_SF69"];
const ML_MODEL_LABELS = {
  xgb_F69: "XGBoost F69",
  rf_F69: "随机森林 F69",
  lstm_SF69: "LSTM S+F69",
};

async function ensureMlOverview() {
  if (state.mlOverview || state.mlOverviewLoading) return;
  state.mlOverviewLoading = true;
  state.mlOverviewError = "";
  try {
    state.mlOverview = await apiJson("/api/ml/overview");
  } catch (err) {
    state.mlOverviewError = err.message || String(err);
  } finally {
    state.mlOverviewLoading = false;
    if (state.activeTab === "models") renderModelsView();
    if (state.activeTab === "stock") renderStockDetail();
  }
}

async function ensureMlNav(startKey = state.mlNavStart) {
  const rule = state.mlNavRule || "DB-K30-b8";
  const sKey = String(startKey || "test");
  const key = `${rule}|${sKey}`;
  if (state.mlNavCache[key] || state.mlNavLoadingKey === key) return;
  state.mlNavLoadingKey = key;
  if (state.activeTab === "models") renderModelsView();
  try {
    const qs = new URLSearchParams({ rule });
    if (sKey === "test") {
      qs.set("period", "test");
    } else {
      qs.set("start_year", sKey);
    }
    const data = await apiJson(`/api/ml/nav?${qs.toString()}`);
    state.mlNavCache[key] = { ok: true, data };
  } catch (err) {
    state.mlNavCache[key] = { ok: false, error: err.message || String(err) };
  } finally {
    if (state.mlNavLoadingKey === key) state.mlNavLoadingKey = "";
    if (state.activeTab === "models") renderModelsView();
  }
}

async function loadMlConsensus({ asOf, withOutcome } = {}) {
  const targetDate = (asOf ?? state.mlConsensusDate ?? state.selectedDate ?? "").trim();
  const useOutcome = withOutcome !== undefined ? Boolean(withOutcome) : state.mlConsensusWithOutcome;
  state.mlConsensusDate = targetDate;
  state.mlConsensusWithOutcome = useOutcome;
  state.mlConsensusLoading = true;
  state.mlConsensusError = "";
  if (state.activeTab === "models") renderModelsView();

  try {
    const qs = new URLSearchParams({
      top_n: "6",
      with_outcome: useOutcome ? "1" : "0",
    });
    if (targetDate) qs.set("as_of", targetDate);
    const data = await apiJson(`/api/ml/consensus?${qs.toString()}`);
    state.mlConsensus = data;
    state.mlConsensusDate = data.as_of;
  } catch (err) {
    state.mlConsensusError = err.message || String(err);
  } finally {
    state.mlConsensusLoading = false;
    if (state.activeTab === "models") renderModelsView();
  }
}

function ensureMlViewLoaded() {
  if (!state.meta?.ml_available) return;
  ensureMlOverview();
  ensureMlNav(state.mlNavStart);
  if (!state.mlConsensus && !state.mlConsensusLoading) {
    const mlSrc = (state.meta?.model_sources || []).find((m) => m.id === "xgb_F69");
    const defaultDay =
      (state.selectedDate && mlSrc?.date_max && state.selectedDate <= mlSrc.date_max ? state.selectedDate : "") ||
      mlSrc?.date_max ||
      "";
    loadMlConsensus({ asOf: defaultDay, withOutcome: state.mlConsensusWithOutcome });
  }
}

async function ensureStockExplain(asOf, tsCode) {
  if (!asOf || !tsCode || !state.meta?.ml_available) return;
  const key = `${asOf}|${tsCode}`;
  if (state.mlExplainCache[key] || state.mlExplainLoadingKey === key) return;
  state.mlExplainLoadingKey = key;
  try {
    const qs = new URLSearchParams({ as_of: asOf, ts_code: tsCode, top: "8" });
    const data = await apiJson(`/api/ml/explain?${qs.toString()}`);
    state.mlExplainCache[key] = { ok: true, data };
  } catch (err) {
    state.mlExplainCache[key] = { ok: false, error: err.message || String(err) };
  } finally {
    if (state.mlExplainLoadingKey === key) state.mlExplainLoadingKey = "";
    if (state.activeTab === "stock" && state.selectedCode === tsCode) {
      renderStockDetail();
    }
  }
}

function renderExplainBarsHtml(expObj) {
  if (!expObj) return "";
  if (!expObj.explain_available && !Array.isArray(expObj.top_factors)) {
    return `<div class="muted-text" style="padding:8px 0">${esc(expObj.explain_error || "该日期不在测试段（2023-01-03 ~ 2026-10-16），无预计算归因")}</div>`;
  }
  const factors = expObj.top_factors || [];
  if (!factors.length) {
    return `<div class="muted-text" style="padding:8px 0">${esc(expObj.explain_error || "暂无归因数据")}</div>`;
  }
  const maxAbs = Math.max(0.0001, ...factors.map((f) => Math.abs(Number(f.contribution) || 0)));
  const rowsHtml = factors
    .map((f) => {
      const c = Number(f.contribution) || 0;
      const pos = c >= 0;
      const pctW = Math.max(3, Math.min(100, (Math.abs(c) / maxAbs) * 100));
      const signStr = `${pos ? "+" : ""}${c.toFixed(4)}`;
      const valStr = f.value !== null && f.value !== undefined ? Number(f.value).toFixed(4).replace(/\.?0+$/, "") : "—";
      const isSeq = String(f.feature || "").startsWith("seq_");
      return `
        <div class="explain-row">
          <div class="explain-label" title="${esc(f.feature)} = ${esc(valStr)}">
            <span class="explain-cn">${esc(f.cn_name || f.feature)}</span>
            <span class="explain-code mono">${esc(f.feature)}${isSeq ? " · 60日序列" : ""}</span>
          </div>
          <div class="explain-val mono muted-text">${esc(valStr)}</div>
          <div class="explain-bar-cell">
            <div class="explain-bar-half explain-bar-neg">
              ${!pos ? `<div class="explain-bar-fill neg" style="width:${pctW.toFixed(1)}%"></div>` : ""}
            </div>
            <div class="explain-bar-axis"></div>
            <div class="explain-bar-half explain-bar-pos">
              ${pos ? `<div class="explain-bar-fill pos" style="width:${pctW.toFixed(1)}%"></div>` : ""}
            </div>
          </div>
          <div class="explain-contrib mono ${pos ? "val-up" : "val-down"}">${signStr}</div>
        </div>
      `;
    })
    .join("");

  const baseTxt = expObj.base !== null && expObj.base !== undefined ? Number(expObj.base).toFixed(4) : "—";
  const scoreTxt = expObj.score !== null && expObj.score !== undefined ? Number(expObj.score).toFixed(4) : "—";
  return `
    <div class="explain-meta-line">
      <span class="pill pill-soft">${esc(expObj.method || "模型归因")}</span>
      <span class="muted-text mono">基线分 base=${baseTxt} · 模型分 score=${scoreTxt}（红=推高分数，青=拉低分数）</span>
    </div>
    <div class="explain-bars-list">${rowsHtml}</div>
  `;
}

function renderStockAttributionCardHtml(cand, runAsOf) {
  const ctxSource = state.run?.request?.context?.model_source || state.selectedModelSource || "xgb_F69";
  const cacheEntry = state.mlExplainCache[`${runAsOf}|${cand.ts_code}`];
  const liveModels = cacheEntry?.ok ? cacheEntry.data?.models || {} : {};

  // 合并快照自带归因/同行模型 + 实时三模型归因接口
  const modelsMap = {};
  for (const mid of ML_MODEL_IDS) {
    if (liveModels[mid]) {
      modelsMap[mid] = liveModels[mid];
    } else if (mid === ctxSource && cand.explain) {
      modelsMap[mid] = {
        model_id: mid,
        name: ML_MODEL_LABELS[mid],
        score_available: true,
        explain_available: true,
        rank: cand.model?.rank,
        pct: cand.model?.pct,
        in_model_picks: cand.model?.in_model_picks,
        ...cand.explain,
      };
    } else if (cand.peer_models?.[mid]) {
      modelsMap[mid] = {
        model_id: mid,
        name: cand.peer_models[mid].name || ML_MODEL_LABELS[mid],
        score_available: true,
        explain_available: false,
        rank: cand.peer_models[mid].rank,
        pct: cand.peer_models[mid].pct,
        in_model_picks: cand.peer_models[mid].in_model_picks,
        explain_error: "正在加载该模型归因…",
      };
    }
  }

  const availIds = ML_MODEL_IDS.filter((m) => modelsMap[m]);
  if (!availIds.length) {
    if (state.meta?.ml_available && state.mlExplainLoadingKey === `${runAsOf}|${cand.ts_code}`) {
      return `<div class="explain-card"><div class="muted-text">正在加载三模型百分位对照与 Top-8 打分归因…</div></div>`;
    }
    return "";
  }

  const defaultTab = ML_MODEL_IDS.includes(ctxSource) && modelsMap[ctxSource] ? ctxSource : availIds[0];
  const activeMid = modelsMap[state.mlExplainModelTab] ? state.mlExplainModelTab : defaultTab;
  const activeExp = modelsMap[activeMid];

  const peerTabsHtml = availIds
    .map((mid) => {
      const m = modelsMap[mid];
      const isAct = mid === activeMid;
      const rankTxt = m.score_available && m.rank ? `#${m.rank} · ${fmtNum(m.pct, 1)}%` : "无当日分";
      const pickMark = m.in_model_picks ? `<span class="peer-pick-dot" title="入选该模型前6">★</span>` : "";
      return `
        <button class="peer-model-tab ${isAct ? "active" : ""}" type="button" data-explain-model="${esc(mid)}">
          <span class="peer-model-name">${esc(ML_MODEL_LABELS[mid])}${pickMark}</span>
          <span class="peer-model-rank mono">${esc(rankTxt)}</span>
        </button>
      `;
    })
    .join("");

  return `
    <div class="explain-card">
      <div class="explain-card-head">
        <div>
          <strong class="explain-card-title">模型打分归因（Top 8）与三模型百分位对照</strong>
          <span class="muted-text" style="margin-left:8px">跨模型仅比百分位 pct，不直接比原始 score</span>
        </div>
        <div class="peer-model-tabs">${peerTabsHtml}</div>
      </div>
      ${renderExplainBarsHtml(activeExp)}
    </div>
  `;
}

function renderStockFeaturesDrawerHtml(cand, f, fp, extraF, extraFp, totalFeatCount) {
  const catalog = state.mlOverview?.feature_catalog?.by_name || {};
  const hasExtra = Boolean(extraF && Object.keys(extraF).length > 0);
  if (!hasExtra) {
    const allFeatHtml = Object.entries(f)
      .map(([k, v]) => {
        const pct = fp[k] !== undefined ? ` (${fp[k]}%)` : "";
        return `
          <div class="feat-kv">
            <span class="feat-k" title="${esc(k)}">${esc(featureLabel(k))}</span>
            <span class="feat-v mono">${esc(v)}${esc(featureUnit(k))}<span class="muted-text">${pct}</span></span>
          </div>
        `;
      })
      .join("");
    return `
      <details class="features-drawer">
        <summary>展开查看送入模型的全部 ${Object.keys(f).length} 项量价特征与全池分位</summary>
        <div class="features-table-grid">${allFeatHtml}</div>
      </details>
    `;
  }

  // 构建 69 特征分组条目（基础量价 28 项 + A~H 扩展 41 项）
  const entries = [];
  for (const [k, v] of Object.entries(f)) {
    const meta = catalog[k];
    entries.push({
      key: k,
      group: "基础量价（28 项）",
      cn: meta?.cn_name || featureLabel(k),
      formula: meta?.formula || "",
      sign: "",
      val: `${v}${featureUnit(k)}`,
      pct: fp[k],
    });
  }
  for (const [k, v] of Object.entries(extraF)) {
    const meta = catalog[k];
    entries.push({
      key: k,
      group: meta?.group || "新增特征",
      cn: meta?.cn_name || k,
      formula: meta?.formula || "",
      sign: meta?.expected_sign || "",
      val: String(v),
      pct: extraFp[k],
    });
  }

  const groups = ["all", ...Array.from(new Set(entries.map((e) => e.group)))];
  const curGroup = groups.includes(state.featureGroupFilter) ? state.featureGroupFilter : "all";
  const filtered = curGroup === "all" ? entries : entries.filter((e) => e.group === curGroup);

  const pillsHtml = groups
    .map((g) => {
      const cnt = g === "all" ? entries.length : entries.filter((e) => e.group === g).length;
      const lbl = g === "all" ? `全部 (${cnt})` : `${g} (${cnt})`;
      return `<button class="chip ${curGroup === g ? "active" : ""}" type="button" data-feat-group="${esc(g)}">${esc(lbl)}</button>`;
    })
    .join("");

  const gridHtml = filtered
    .map((item) => {
      const pctTxt = item.pct !== undefined ? ` (${item.pct}%)` : "";
      const signBadge =
        item.sign === "+"
          ? `<span class="feat-sign pos" title="预期正向">+</span>`
          : item.sign === "-"
            ? `<span class="feat-sign neg" title="预期负向">−</span>`
            : "";
      const tip = [item.key, item.formula].filter(Boolean).join(" · ");
      return `
        <div class="feat-kv" title="${esc(tip)}">
          <span class="feat-k">${esc(item.cn)}${signBadge}</span>
          <span class="feat-v mono">${esc(item.val)}<span class="muted-text">${pctTxt}</span></span>
        </div>
      `;
    })
    .join("");

  return `
    <details class="features-drawer" ${curGroup !== "all" ? "open" : ""}>
      <summary>展开查看送入模型的全部 ${totalFeatCount} 项 F69 特征（原 28 项 + 新增 ${Object.keys(extraF).length} 项 · A~H 维度）</summary>
      <div class="feat-group-bar">${pillsHtml}</div>
      <div class="features-table-grid">${gridHtml}</div>
    </details>
  `;
}

function fmtPctSigned(v, digits = 1) {
  if (typeof v !== "number" || !Number.isFinite(v)) return "—";
  return `${v > 0 ? "+" : ""}${v.toFixed(digits)}%`;
}

function fmtWan(yuan, digits = 1) {
  if (typeof yuan !== "number" || !Number.isFinite(yuan)) return "—";
  return `${(yuan / 10000).toFixed(digits)}万`;
}

function renderStartsSummaryTableHtml(startsSummary) {
  if (!startsSummary?.available || !Array.isArray(startsSummary.rows) || !startsSummary.rows.length) {
    return `<div class="muted-text" style="padding:10px 0">暂无换起点汇总数据</div>`;
  }
  const activeYr = state.mlNavStart === "test" ? null : Number(state.mlNavStart);
  const rowsHtml = startsSummary.rows
    .map((r) => {
      const isSelected = activeYr === r.start_year;
      const cellFor = (mid) => {
        const m = r.models?.[mid];
        if (!m) return `<td class="mono muted-text">—</td>`;
        const isBest = r.best_model_id === mid;
        const cagrCls = (m.cagr ?? 0) >= 0 ? "val-up" : "val-down";
        return `
          <td class="mono ${isBest ? "ml-cell-best" : ""}">
            <div><strong class="${cagrCls}">${fmtPctSigned(m.cagr, 1)}</strong>${isBest ? " ★" : ""} <span class="muted-text">(${fmtWan(m.end_capital)})</span></div>
            <div class="muted-text" style="font-size:10.5px">超额 ${fmtPctSigned(m.annual_excess, 1)} · 回撤 ${fmtPctSigned(m.max_drawdown, 1)}</div>
          </td>
        `;
      };
      return `
        <tr class="ml-clickable-row ${isSelected ? "ml-row-selected" : ""}" data-select-start-year="${r.start_year}" title="点击将上方曲线切换至 ${r.start_year} 年独立起步">
          <td class="mono">
            <strong>${r.start_year}</strong>
            ${r.is_oos ? `<span class="tag-pick" style="font-size:10px;margin-left:4px">样本外</span>` : ""}
          </td>
          <td class="mono muted-text">${esc(r.start_date)} <span style="font-size:10.5px">(${fmtNum(r.hold_years, 1)}年)</span></td>
          ${cellFor("xgb_F69")}
          ${cellFor("rf_F69")}
          ${cellFor("lstm_SF69")}
          <td class="mono ${(r.pool_cagr ?? 0) >= 0 ? "val-up" : "val-down"}">${fmtPctSigned(r.pool_cagr, 1)}</td>
          <td><span class="pill pill-soft">${esc(r.best_model_name || "—")}</span></td>
        </tr>
      `;
    })
    .join("");

  const sAll = startsSummary.summary?.all_10 || {};
  const sEx = startsSummary.summary?.ex_2015 || {};
  const summaryRowsHtml = `
    <tr class="ml-summary-row">
      <td colspan="2"><strong>10 个起点中位数（2015~2024）</strong></td>
      ${ML_MODEL_IDS.map((mid) => `<td class="mono"><strong>${fmtPctSigned(sAll[mid]?.median_cagr, 1)}</strong> <span class="muted-text">(回撤 ${fmtPctSigned(sAll[mid]?.median_dd, 1)})</span></td>`).join("")}
      <td class="mono">${fmtPctSigned(sAll.pool_nav?.median_cagr, 1)}</td>
      <td class="muted-text mono" style="font-size:11px">最高次数：X ${sAll.xgb_F69?.best_count ?? 0} / R ${sAll.rf_F69?.best_count ?? 0} / L ${sAll.lstm_SF69?.best_count ?? 0}</td>
    </tr>
    <tr class="ml-summary-row">
      <td colspan="2"><strong>2016~2024 中位数（不含 2015 股灾）</strong></td>
      ${ML_MODEL_IDS.map((mid) => `<td class="mono"><strong>${fmtPctSigned(sEx[mid]?.median_cagr, 1)}</strong> <span class="muted-text">(最差 ${fmtPctSigned(sEx[mid]?.min_cagr, 1)})</span></td>`).join("")}
      <td class="mono">${fmtPctSigned(sEx.pool_nav?.median_cagr, 1)}</td>
      <td class="muted-text mono" style="font-size:11px">最高次数：X ${sEx.xgb_F69?.best_count ?? 0} / R ${sEx.rf_F69?.best_count ?? 0} / L ${sEx.lstm_SF69?.best_count ?? 0}</td>
    </tr>
  `;

  return `
    <div class="ml-table-wrap">
      <table class="ml-table">
        <thead>
          <tr>
            <th>起步年</th>
            <th>起步日 (持有年数)</th>
            <th>XGBoost F69 年化 (期末 / 超额 / 回撤)</th>
            <th>随机森林 F69 年化 (期末 / 超额 / 回撤)</th>
            <th>LSTM S+F69 年化 (期末 / 超额 / 回撤)</th>
            <th>准入池年化</th>
            <th>年化最高</th>
          </tr>
        </thead>
        <tbody>
          ${rowsHtml}
          ${summaryRowsHtml}
        </tbody>
      </table>
    </div>
    <div class="muted-text" style="margin-top:6px;font-size:11.5px">${esc(startsSummary.note || "")}</div>
  `;
}

function renderYearlyBreakdownTableHtml(navData) {
  const rows = navData?.yearly_breakdown || [];
  if (!rows.length) {
    return `<div class="muted-text" style="padding:10px 0">暂无逐年明细数据</div>`;
  }
  const rowsHtml = rows
    .map((r) => {
      const cellFor = (mid) => {
        const m = r.models?.[mid];
        if (!m || m.return_pct === null || m.return_pct === undefined) return `<td class="mono muted-text">—</td>`;
        const cls = m.return_pct >= 0 ? "val-up" : "val-down";
        return `
          <td class="mono">
            <div><strong class="${cls}">${fmtPctSigned(m.return_pct, 1)}</strong> <span class="muted-text">(${fmtWan(m.end_capital)})</span></div>
            <div class="muted-text" style="font-size:10.5px">年内回撤 ${fmtPctSigned(m.max_drawdown, 1)}</div>
          </td>
        `;
      };
      const b = r.benchmarks || {};
      const bCell = (k) => {
        const v = b[k];
        if (v === null || v === undefined) return `<td class="mono muted-text">—</td>`;
        return `<td class="mono ${v >= 0 ? "val-up" : "val-down"}">${fmtPctSigned(v, 1)}</td>`;
      };
      return `
        <tr>
          <td class="mono"><strong>${r.year}</strong>${r.is_partial ? `<span class="muted-text" style="font-size:10px"> (至今)</span>` : ""}</td>
          <td class="mono muted-text">${esc(r.end_date)}</td>
          ${cellFor("xgb_F69")}
          ${cellFor("rf_F69")}
          ${cellFor("lstm_SF69")}
          ${bCell("pool_nav")}
          ${bCell("hs300")}
          ${bCell("zz500")}
          ${bCell("zz1000")}
        </tr>
      `;
    })
    .join("");

  return `
    <div class="ml-table-wrap">
      <table class="ml-table">
        <thead>
          <tr>
            <th>年份</th>
            <th>年末日期</th>
            <th>XGBoost F69 (年收益 / 期末 / 回撤)</th>
            <th>随机森林 F69 (年收益 / 期末 / 回撤)</th>
            <th>LSTM S+F69 (年收益 / 期末 / 回撤)</th>
            <th>准入池等权</th>
            <th>沪深 300</th>
            <th>中证 500</th>
            <th>中证 1000</th>
          </tr>
        </thead>
        <tbody>${rowsHtml}</tbody>
      </table>
    </div>
    <div class="muted-text" style="margin-top:6px;font-size:11.5px">
      口径：起始本金 20 万元，年末资金为账户总值（已扣全部费用）；年收益率 = 本年末资金 ÷ 上年末资金 − 1，年内最大回撤按逐日净值计算。
    </div>
  `;
}

function renderModelsView() {
  if (!state.meta?.ml_available) {
    dom.modelsContent.innerHTML = `
      <div class="empty-state">
        <div class="empty-title">未检测到三模型数据包（data/ml/api_export/）</div>
        <p class="empty-desc">请将 api_export.zip 解压至 data/ml/，得到 data/ml/api_export/ 与 data/ml/deploy/ 后刷新页面。</p>
      </div>
    `;
    return;
  }

  if (state.mlOverviewLoading && !state.mlOverview) {
    dom.modelsContent.innerHTML = `<div class="empty-state"><div class="empty-title">正在加载三模型排行榜与净值数据…</div></div>`;
    return;
  }

  const ov = state.mlOverview;
  const lb = ov?.leaderboard || [];
  const startsSummary = ov?.starts_summary || null;
  const startYears = ov?.available_start_years || [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024];
  const curStart = String(state.mlNavStart || "test");
  const viewMode = state.mlNavViewMode || "nav";
  const detailTab = state.mlNavDetailTab || "starts";
  const navKey = `${state.mlNavRule}|${curStart}`;
  const navEntry = state.mlNavCache[navKey];
  const navData = navEntry?.ok ? navEntry.data : null;
  const curStartLabel = curStart === "test" ? "正式测试段" : `${curStart}起`;

  const disciplineHtml = (ov?.discipline_notes || [])
    .map((n) => `<li>${esc(n)}</li>`)
    .join("");

  const lbRowsHtml = lb
    .map((row) => {
      const mid = row.model_id;
      const isDef = mid === "xgb_F69";
      const st = navData?.series?.[mid]?.stats;
      return `
        <tr class="${state.selectedModelSource === mid ? "ml-row-selected" : ""}">
          <td>
            <div class="ml-model-cell">
              <span class="ml-color-dot" style="background:${(NAV_SERIES_STYLES[mid] || {}).color || "#2563eb"}"></span>
              <strong>${esc(row["模型"] || mid)}</strong>
              ${isDef ? `<span class="tag-pick" style="font-size:10px">默认</span>` : ""}
            </div>
            <div class="muted-text mono">${esc(mid)}</div>
          </td>
          <td class="mono"><strong>${fmtNum(row["测试 IC 全段"], 3)}</strong><div class="muted-text">${esc(row["测试 IC 全段 区间"] || "")}</div></td>
          <td class="mono">${fmtNum(row["测试 IC T1"], 3)} / ${fmtNum(row["测试 IC T2"], 3)}</td>
          <td class="mono">${fmtNum(row["测试 ICIR"], 2)} <span class="muted-text">(t=${fmtNum(row["测试 NW t"], 1)})</span></td>
          <td class="mono">${fmtPctSigned((row["测试前30超额"] || 0) * 100, 2)}</td>
          <td class="mono">${fmtPctSigned((row["测试多空"] || 0) * 100, 2)}</td>
          <td class="mono">${fmtNum(row["测试 IC 2023"], 3)} / ${fmtNum(row["测试 IC 2024"], 3)} / ${fmtNum(row["测试 IC 2025"], 3)} / ${fmtNum(row["测试 IC 2026"], 3)}</td>
          <td class="mono">
            ${
              st
                ? `<span class="val-up">${fmtPctSigned(st.cagr, 1)}</span> / <span class="val-down">${fmtPctSigned(st.max_drawdown, 1)}</span>`
                : "—"
            }
          </td>
          <td>
            <button class="role-prompt-btn" type="button" data-switch-model="${esc(mid)}">切换使用</button>
          </td>
        </tr>
      `;
    })
    .join("");

  // 起点切换条（固定主规则 DB-K30-b8）
  const startBarHtml = `
    <div class="ml-start-bar">
      <button class="chip ${curStart === "test" ? "active" : ""}" type="button" data-nav-start="test" title="正式测试段（2023-01-10 ~ 2026-10-19，现行印花税）">
        测试段 2023~2026
      </button>
      <span class="ml-start-sep" aria-hidden="true"></span>
      <span class="muted-text" style="font-size:11.5px;white-space:nowrap">换起点(20万独立起步→至今)：</span>
      <div class="filter-pills">
        ${startYears
          .map((yr) => {
            const yrStr = String(yr);
            const isOos = yr >= 2023;
            return `<button class="chip ${curStart === yrStr ? "active" : ""} ${isOos ? "chip-oos" : ""}" type="button" data-nav-start="${yrStr}" title="${yr} 年首交易日拿 20 万现金起步至 2026-10-23${isOos ? "（样本外）" : "（含2023年前历史，仅供参考）"}">${yr}起</button>`;
          })
          .join("")}
      </div>
    </div>
  `;

  // 净值 / 动态年化 / 10起点年化对比主体
  let navChartSectionHtml = `<div class="pending-placeholder">正在加载主规则 DB-K30-b8 曲线数据…</div>`;
  if (viewMode === "starts_curve") {
    if (!startsSummary?.available) {
      navChartSectionHtml = `<div class="prompt-warn-banner">未检测到换起点汇总数据（ml/results/delivery/long_starts.csv）</div>`;
    } else {
      const sAll = startsSummary.summary?.all_10 || {};
      const startsIds = [
        { id: "xgb_F69", label: "XGBoost F69" },
        { id: "rf_F69", label: "随机森林 F69" },
        { id: "lstm_SF69", label: "LSTM S+F69" },
        { id: "pool_nav", label: "准入池等权（基准）" },
      ];
      const legendHtml = startsIds
        .map((it) => {
          const hidden = Boolean(state.mlNavHiddenKeys[it.id]);
          const col = (NAV_SERIES_STYLES[it.id] || {}).color || "#2563eb";
          const med = sAll[it.id]?.median_cagr;
          const wins = sAll[it.id]?.best_count;
          const winTxt = wins !== undefined ? ` · 最高 ${wins}/10次` : "";
          return `
            <button class="nav-legend-chip ${hidden ? "is-hidden" : ""}" type="button" data-nav-toggle="${esc(it.id)}">
              <span class="ml-color-dot" style="background:${col}"></span>
              <span>${esc(it.label)}</span>
              <span class="mono">中位年化 ${fmtPctSigned(med, 1)}</span>
              <span class="muted-text mono">${esc(winTxt)}</span>
            </button>
          `;
        })
        .join("");
      navChartSectionHtml = `
        <div class="nav-legend-row">${legendHtml}</div>
        <div class="nav-readout mono" id="nav-hover-readout">
          主规则 DB-K30-b8 · 2015 ~ 2024 每年首交易日拿 20 万独立起步 → 至今（${esc(startsSummary.end_date)}）年化收益率 · 移动鼠标查看各年详情，点击任意年份切入逐日曲线
        </div>
        <div class="nav-chart-wrap" id="ml-nav-chart-container"></div>
        <div class="muted-text" style="margin-top:6px;font-size:11.5px">${esc(startsSummary.note || "")}</div>
      `;
    }
  } else if (navEntry && !navEntry.ok) {
    navChartSectionHtml = `<div class="prompt-warn-banner">${esc(navEntry.error)}</div>`;
  } else if (navData) {
    const seriesEntries = Object.values(navData.series || {});
    const legendHtml = seriesEntries
      .map((s) => {
        const hidden = Boolean(state.mlNavHiddenKeys[s.id]);
        const col = (NAV_SERIES_STYLES[s.id] || {}).color || "#2563eb";
        const cagrTxt = s.stats?.cagr !== null && s.stats?.cagr !== undefined ? `年化 ${fmtPctSigned(s.stats.cagr, 1)}` : "—";
        const retTxt = s.stats?.total_return !== null && s.stats?.total_return !== undefined ? `累计 ${fmtPctSigned(s.stats.total_return, 1)}` : "";
        const ddTxt = s.stats?.max_drawdown !== null && s.stats?.max_drawdown !== undefined ? `回撤 ${fmtPctSigned(s.stats.max_drawdown, 1)}` : "";
        const isIdx = s.kind === "index";
        return `
          <button class="nav-legend-chip ${hidden ? "is-hidden" : ""}" type="button" data-nav-toggle="${esc(s.id)}" title="${isIdx ? "点击叠加或隐藏该市场指数基准" : "点击隐藏或显示该曲线"}">
            <span class="ml-color-dot" style="background:${col}"></span>
            <span>${esc(s.label)}</span>
            <span class="mono"><strong>${esc(cagrTxt)}</strong></span>
            ${!isIdx ? `<span class="muted-text mono">${esc(retTxt)} · ${esc(ddTxt)}</span>` : `<span class="muted-text mono">${esc(retTxt)}</span>`}
          </button>
        `;
      })
      .join("");
    const modeHint =
      viewMode === "cagr"
        ? `截至当日年化收益率曲线（满 ${navData.cagr_min_days || 60} 交易日起按 242 日/年折算）`
        : "逐日净值曲线（起点=1 · 对应 20 万本金）";
    navChartSectionHtml = `
      <div class="nav-legend-row">${legendHtml}</div>
      <div class="nav-readout mono" id="nav-hover-readout">
        区间：${esc(navData.date_start)} ~ ${esc(navData.date_end)}（${navData.trading_days ?? navData.count} 个交易日 / ${fmtNum(navData.hold_years, 1)} 年）· ${esc(modeHint)} · 移动鼠标查看逐日详情
      </div>
      <div class="nav-chart-wrap" id="ml-nav-chart-container"></div>
      <div class="muted-text" style="margin-top:6px;font-size:11.5px">${esc(navData.note || "")}</div>
    `;
  }

  // 下方明细表区（换起点年化表 / 当前起点逐年明细）
  let navDetailBodyHtml = "";
  if (detailTab === "starts") {
    navDetailBodyHtml = renderStartsSummaryTableHtml(startsSummary);
  } else if (detailTab === "yearly") {
    navDetailBodyHtml = navData
      ? renderYearlyBreakdownTableHtml(navData)
      : `<div class="muted-text" style="padding:10px 0">正在加载当前起点逐年明细…</div>`;
  }

  const navDetailSectionHtml = `
    <div class="ml-nav-detail-wrap">
      <div class="ml-nav-detail-head">
        <div class="filter-pills">
          <button class="chip ${detailTab === "starts" ? "active" : ""}" type="button" data-nav-detail="starts">换起点年化对比表 (2015~2024)</button>
          <button class="chip ${detailTab === "yearly" ? "active" : ""}" type="button" data-nav-detail="yearly">当前起点逐年收益与回撤 (${esc(curStartLabel)})</button>
          <button class="chip ${detailTab === "none" ? "active" : ""}" type="button" data-nav-detail="none">收起明细表</button>
        </div>
        ${
          detailTab === "starts"
            ? `<span class="muted-text" style="font-size:11.5px">点击任意起步年行，即可把上方曲线切换至该年起步</span>`
            : ""
        }
      </div>
      ${navDetailBodyHtml}
    </div>
  `;

  // 同日三模型选股共识区
  const cons = state.mlConsensus;
  let consensusBodyHtml = "";
  if (state.mlConsensusLoading && !cons) {
    consensusBodyHtml = `<div class="pending-placeholder">正在加载当日三模型选股共识…</div>`;
  } else if (state.mlConsensusError) {
    consensusBodyHtml = `<div class="prompt-warn-banner">${esc(state.mlConsensusError)}</div>`;
  } else if (cons) {
    const ovp = cons.pairwise_overlap || {};
    const colsHtml = ML_MODEL_IDS.map((mid) => {
      const mData = cons.models?.[mid];
      const picks = mData?.picks || [];
      return `
        <div class="consensus-model-col">
          <div class="consensus-col-head">
            <div class="ml-model-cell">
              <span class="ml-color-dot" style="background:${(NAV_SERIES_STYLES[mid] || {}).color}"></span>
              <strong>${esc(ML_MODEL_LABELS[mid])}</strong>
            </div>
            <div style="display:flex;gap:5px">
              <button class="role-prompt-btn" type="button" data-ml-dry-run="${esc(mid)}" title="使用该模型在 ${esc(cons.as_of)} 生成 69 特征 + 归因演练快照">演练</button>
              <button class="role-ask-btn" type="button" data-ml-start-run="${esc(mid)}" title="使用该模型在 ${esc(cons.as_of)} 启动 7 角色 AI 辩论">辩论</button>
            </div>
          </div>
          <div class="consensus-picks-list">
            ${picks
              .map((p) => {
                const cBadge =
                  p.consensus_count === 3
                    ? `<span class="cons-badge cons-3">三模型共识</span>`
                    : p.consensus_count === 2
                      ? `<span class="cons-badge cons-2">双模型</span>`
                      : `<span class="cons-badge cons-1">独选</span>`;
                const outHtml =
                  cons.with_outcome && p.ret_o1o5_pct !== null && p.ret_o1o5_pct !== undefined
                    ? `<span class="mono ${p.ret_o1o5_pct >= 0 ? "val-up" : "val-down"}" title="事后 T+1→T+5 开盘收益（仅复盘）">${fmtPctSigned(p.ret_o1o5_pct, 1)}</span>`
                    : "";
                return `
                  <div class="consensus-pick-item" data-ml-inspect-stock="${esc(p.ts_code)}" data-ml-inspect-model="${esc(mid)}">
                    <div class="consensus-pick-top">
                      <div>
                        <span class="mono muted-text">#${p.rank}</span>
                        <strong>${esc(p.name)}</strong>
                        <span class="mono muted-text">${esc(p.ts_code)}</span>
                      </div>
                      ${cBadge}
                    </div>
                    <div class="consensus-pick-sub">
                      <span>${esc(p.industry)}</span>
                      <span class="mono">pct ${fmtNum(p.pct, 1)}% ${outHtml}</span>
                    </div>
                  </div>
                `;
              })
              .join("")}
          </div>
        </div>
      `;
    }).join("");

    const tableRowsHtml = (cons.consensus_rows || [])
      .map((r) => {
        const cellFor = (mid) => {
          const d = r.models?.[mid];
          if (!d) return `<td class="mono muted-text">—</td>`;
          return `<td class="mono ${d.in_picks ? "ml-cell-picked" : ""}">#${d.rank} (${fmtNum(d.pct, 1)}%)${d.in_picks ? " ★" : ""}</td>`;
        };
        const outCol = cons.with_outcome
          ? `<td class="mono ${r.ret_o1o5_pct >= 0 ? "val-up" : "val-down"}">${fmtPctSigned(r.ret_o1o5_pct, 2)} <span class="muted-text">(超额 ${fmtPctSigned(r.excess_o1o5_pct, 2)})</span></td>`
          : "";
        return `
          <tr class="ml-clickable-row" data-ml-inspect-stock="${esc(r.ts_code)}" data-ml-inspect-model="${esc(r.picked_by?.[0] || "xgb_F69")}">
            <td><strong>${esc(r.name)}</strong> <span class="mono muted-text">${esc(r.ts_code)}</span></td>
            <td>${esc(r.industry)}</td>
            <td><span class="cons-badge cons-${r.consensus_count}">${r.consensus_count}/3 模型入选</span></td>
            ${cellFor("xgb_F69")}
            ${cellFor("rf_F69")}
            ${cellFor("lstm_SF69")}
            <td class="mono"><strong>${fmtNum(r.avg_pct, 1)}%</strong></td>
            ${outCol}
          </tr>
        `;
      })
      .join("");

    consensusBodyHtml = `
      <div class="consensus-overlap-strip">
        <span class="pill pill-soft mono">三模型全共识：${ovp.all_three ?? 0} 只</span>
        <span class="pill pill-soft mono">XGB ∩ RF：${ovp.xgb_rf ?? 0}</span>
        <span class="pill pill-soft mono">XGB ∩ LSTM：${ovp.xgb_lstm ?? 0}</span>
        <span class="pill pill-soft mono">RF ∩ LSTM：${ovp.rf_lstm ?? 0}</span>
        ${
          cons.with_outcome
            ? `<span class="tag-warn">⚠ 复盘模式已开启：右侧 T+1→T+5 收益为事后已知信息，切勿作为信号日预测依据</span>`
            : `<span class="muted-text">点击任意股票可即时弹出三模型 Top-8 打分归因对比</span>`
        }
      </div>
      <div class="consensus-cols-grid">${colsHtml}</div>
      <div class="ml-table-wrap" style="margin-top:12px">
        <table class="ml-table">
          <thead>
            <tr>
              <th>股票</th>
              <th>行业</th>
              <th>共识度</th>
              <th>XGBoost F69</th>
              <th>随机森林 F69</th>
              <th>LSTM S+F69</th>
              <th>三模型均分位</th>
              ${cons.with_outcome ? `<th>事后 T+1→T+5 收益（仅复盘）</th>` : ""}
            </tr>
          </thead>
          <tbody>${tableRowsHtml}</tbody>
        </table>
      </div>
    `;
  }

  dom.modelsContent.innerHTML = `
    <section class="card ml-discipline-banner">
      <div class="ml-discipline-title">统计纪律与跨模型口径说明</div>
      <ul class="prompt-rules-list">${disciplineHtml}</ul>
    </section>

    <section class="card ml-section-card">
      <div class="card-head" style="padding:0 0 10px">
        <div>
          <h2 class="card-title">三模型正式测试段排行榜（2023-01 ~ 2026-10，每年初重训）</h2>
          <p class="card-subtitle">同一套 F69 特征与准入池（LSTM 另含 60×11 日线序列）；三者差异统计上均不显著</p>
        </div>
      </div>
      <div class="ml-table-wrap">
        <table class="ml-table">
          <thead>
            <tr>
              <th>模型</th>
              <th>测试 IC [95% 区间]</th>
              <th>T1 / T2 IC</th>
              <th>ICIR (NW t)</th>
              <th>前 30 超额</th>
              <th>多空收益</th>
              <th>逐年 IC (23/24/25/26)</th>
              <th>主规则(${esc(curStartLabel)}) 年化 / 回撤</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>${lbRowsHtml}</tbody>
        </table>
      </div>
    </section>

    <section class="card ml-section-card">
      <div class="ml-nav-head">
        <div>
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
            <h2 class="card-title">组合净值与换起点年化对比</h2>
            <span class="pill pill-soft mono">主规则 DB-K30-b8 · 20万本金 · 最多30只</span>
          </div>
          <p class="card-subtitle">仓库真实费用与涨跌停模拟器回测 · 切换不同起步年查看至今曲线，点击图例可叠加沪深300等指数或隐藏任意曲线</p>
        </div>
        <div class="ml-nav-controls">
          <div class="filter-pills">
            <button class="chip ${viewMode === "nav" ? "active" : ""}" type="button" data-nav-mode="nav">净值曲线 (起点=1)</button>
            <button class="chip ${viewMode === "cagr" ? "active" : ""}" type="button" data-nav-mode="cagr">动态年化曲线 (%)</button>
            <button class="chip ${viewMode === "starts_curve" ? "active" : ""}" type="button" data-nav-mode="starts_curve">10起点年化对比</button>
          </div>
          ${
            viewMode === "nav"
              ? `<label class="follow-toggle" title="长期净值建议使用对数坐标以便观察回撤比例">
                  <input type="checkbox" id="chk-nav-log" ${state.mlNavLog ? "checked" : ""} />
                  <span>对数坐标</span>
                </label>`
              : ""
          }
        </div>
      </div>
      ${startBarHtml}
      ${navChartSectionHtml}
      ${navDetailSectionHtml}
    </section>

    <section class="card ml-section-card">
      <div class="ml-nav-head">
        <div>
          <h2 class="card-title">同日三模型 Top 6 选股共识与分歧</h2>
          <p class="card-subtitle">支持 2021-01-04 ~ 2026-10-16 任意有分数的交易日 · 可一键发起对应模型的 69 特征演练或 AI 辩论</p>
        </div>
        <div class="ml-nav-controls">
          <label class="prompt-stock-picker">
            <span class="muted-text">对比日期</span>
            <input type="date" id="inp-consensus-date" class="ctrl-select ctrl-mini mono" value="${esc(state.mlConsensusDate || cons?.as_of || "")}" min="2021-01-04" max="2026-10-16" />
          </label>
          <button class="btn btn-secondary btn-sm" type="button" id="btn-consensus-go">加载</button>
          <label class="follow-toggle" title="显示 T+1 开盘买、T+5 开盘卖的事后收益（仅限复盘）">
            <input type="checkbox" id="chk-consensus-outcome" ${state.mlConsensusWithOutcome ? "checked" : ""} />
            <span>复盘事后收益</span>
          </label>
        </div>
      </div>
      ${consensusBodyHtml}
    </section>

    ${renderFeatureCatalogSectionHtml(ov?.feature_catalog)}
  `;

  const chartEl = document.getElementById("ml-nav-chart-container");
  const readoutEl = document.getElementById("nav-hover-readout");
  if (chartEl) {
    if (viewMode === "starts_curve" && startsSummary?.available) {
      const visibleKeys = ["xgb_F69", "rf_F69", "lstm_SF69", "pool_nav"].filter((k) => !state.mlNavHiddenKeys[k]);
      const activeYr = curStart === "test" ? null : Number(curStart);
      renderStartsAnnualizedChart(chartEl, startsSummary, {
        visibleKeys,
        activeStartYear: activeYr,
        onHover: (snap) => {
          if (!readoutEl) return;
          if (!snap) {
            readoutEl.textContent = `主规则 DB-K30-b8 · 2015 ~ 2024 每年首交易日拿 20 万独立起步 → 至今（${startsSummary.end_date}）年化收益率 · 移动鼠标查看各年详情，点击任意年份切入逐日曲线`;
            return;
          }
          const r = snap.row;
          readoutEl.innerHTML =
            `<strong>${r.start_year}年起步</strong>（${esc(r.start_date)} → ${esc(r.end_date)} · ${fmtNum(r.hold_years, 1)}年）· ` +
            snap.items
              .map(
                (it) =>
                  `<span style="color:${it.color};font-weight:600">${esc(it.label)} ${fmtPctSigned(it.cagr, 1)}</span>`,
              )
              .join(" · ") +
            ` <span class="muted-text">（点击切入该年逐日曲线）</span>`;
        },
        onSelectYear: (yr) => {
          state.mlNavStart = String(yr);
          state.mlNavViewMode = "cagr";
          ensureMlNav(state.mlNavStart);
          renderModelsView();
        },
      });
    } else if (navData) {
      const visibleKeys = Object.keys(navData.series || {}).filter((k) => !state.mlNavHiddenKeys[k]);
      const defaultReadout = `区间：${navData.date_start} ~ ${navData.date_end}（${navData.trading_days ?? navData.count} 个交易日 / ${fmtNum(navData.hold_years, 1)} 年）· ${viewMode === "cagr" ? `截至当日年化收益率曲线（满 ${navData.cagr_min_days || 60} 交易日起按 242 日/年折算）` : "逐日净值曲线（起点=1 · 对应 20 万本金）"} · 移动鼠标查看逐日详情`;
      renderNavComparisonChart(chartEl, navData, {
        visibleKeys,
        mode: viewMode === "cagr" ? "cagr" : "nav",
        logScale: state.mlNavLog,
        onHover: (snap) => {
          if (!readoutEl) return;
          if (!snap) {
            readoutEl.textContent = defaultReadout;
            return;
          }
          readoutEl.innerHTML =
            `<strong>${esc(snap.date)}</strong> · ` +
            snap.items
              .map((it) => {
                const cagrPart = it.cagrPct !== null ? `年化 ${fmtPctSigned(it.cagrPct, 1)}` : `前${navData.cagr_min_days || 60}日观察期`;
                const capPart = it.kind === "model" ? `${it.capitalWan.toFixed(1)}万 · ` : "";
                if (viewMode === "cagr") {
                  return `<span style="color:${it.color};font-weight:600">${esc(it.label)} ${cagrPart}</span> <span class="muted-text">(${capPart}${it.nav.toFixed(3)})</span>`;
                }
                return `<span style="color:${it.color};font-weight:600">${esc(it.label)} ${it.nav.toFixed(3)}</span> <span class="muted-text">(${capPart}${fmtPctSigned(it.retPct, 1)} · ${cagrPart})</span>`;
              })
              .join(" · ");
        },
      });
    }
  }
}

function catalogGroupDisplayLabel(group) {
  if (group === "仓库原特征") return "基础量价";
  if (group === "日线序列") return "LSTM 序列通道";
  return group;
}

function getFilteredCatalogItems(catalog) {
  const items = catalog?.items || [];
  const grp = state.mlCatalogGroup || "f69";
  const q = (state.mlCatalogQuery || "").trim().toLowerCase();

  return items.filter((it) => {
    if (grp === "f69" && it.group === "日线序列") return false;
    if (grp === "extra41" && (it.group === "仓库原特征" || it.group === "日线序列")) return false;
    if (grp !== "all" && grp !== "f69" && grp !== "extra41" && it.group !== grp) return false;
    if (!q) return true;
    const hay = `${it.name} ${it.cn_name} ${it.group} ${it.formula} ${it.reason}`.toLowerCase();
    return hay.includes(q);
  });
}

function renderCatalogRowsHtml(items) {
  if (!items.length) {
    return `<tr><td colspan="7" class="muted-text" style="text-align:center;padding:18px">没有匹配的特征项，请调整分组或关键词</td></tr>`;
  }
  return items
    .map((it, idx) => {
      const sign = it.expected_sign || "";
      let signHtml = `<span class="muted-text">—</span>`;
      if (sign === "+") signHtml = `<span class="feat-sign pos" title="预期值越大收益越高">+ 正向</span>`;
      else if (sign === "-") signHtml = `<span class="feat-sign neg" title="预期值越小收益越高">− 负向</span>`;
      else if (sign) signHtml = `<span class="feat-sign neutral">${esc(sign)}</span>`;
      const isSeq = it.group === "日线序列";
      const codeName = isSeq ? `seq_${it.name}` : it.name;
      return `
        <tr>
          <td class="mono muted-text">${idx + 1}</td>
          <td class="mono"><strong>${esc(codeName)}</strong></td>
          <td><strong>${esc(it.cn_name)}</strong></td>
          <td><span class="pill pill-soft">${esc(catalogGroupDisplayLabel(it.group))}</span></td>
          <td class="ml-catalog-wrap-cell mono">${esc(it.formula || "—")}</td>
          <td>${signHtml}</td>
          <td class="ml-catalog-wrap-cell muted-text">${esc(it.reason || (it.group === "仓库原特征" ? "基础量价与均线/MACD 指标" : isSeq ? "LSTM 近 60 个交易日逐日序列输入通道" : "—"))}</td>
        </tr>
      `;
    })
    .join("");
}

function renderFeatureCatalogSectionHtml(catalog) {
  if (!catalog?.available || !Array.isArray(catalog.items) || !catalog.items.length) {
    return "";
  }
  const allItems = catalog.items;
  const f69Count = allItems.filter((it) => it.group !== "日线序列").length;
  const extra41Count = allItems.filter((it) => it.group !== "仓库原特征" && it.group !== "日线序列").length;
  const seqCount = allItems.filter((it) => it.group === "日线序列").length;
  const curGrp = state.mlCatalogGroup || "f69";
  const filtered = getFilteredCatalogItems(catalog);

  const pills = [
    { key: "f69", label: `F69 全部特征 (${f69Count})` },
    { key: "extra41", label: `A~H 新增 (${extra41Count})` },
    ...(catalog.groups || []).map((g) => ({
      key: g,
      label: `${catalogGroupDisplayLabel(g)} (${allItems.filter((it) => it.group === g).length})`,
    })),
    { key: "all", label: `含序列全部 (${allItems.length})` },
  ];

  const pillsHtml = pills
    .map(
      (p) =>
        `<button class="chip ${curGrp === p.key ? "active" : ""}" type="button" data-catalog-group="${esc(p.key)}">${esc(p.label)}</button>`,
    )
    .join("");

  return `
    <section class="card ml-section-card" id="ml-feature-catalog-card">
      <div class="ml-nav-head">
        <div>
          <h2 class="card-title">F69 特征与 LSTM 序列通道字典</h2>
          <p class="card-subtitle">三套模型共用 ${f69Count} 项特征（28 基础量价 + ${extra41Count} 项 A~H 扩展特征），LSTM 另加近 60 日 × ${seqCount} 个日线序列通道</p>
        </div>
        <div class="ml-nav-controls">
          <input
            type="search"
            id="inp-feat-catalog-search"
            class="ctrl-select ctrl-mini ml-catalog-search"
            placeholder="搜索特征名 / 中文名 / 公式…"
            value="${esc(state.mlCatalogQuery || "")}"
          />
          <span class="pill pill-soft mono" id="ml-catalog-count">显示 ${filtered.length} / ${allItems.length}</span>
        </div>
      </div>

      <div class="feat-group-bar" id="ml-catalog-group-bar">${pillsHtml}</div>

      <div class="ml-table-wrap ml-catalog-scroll">
        <table class="ml-table ml-catalog-table">
          <thead>
            <tr>
              <th style="width:42px">#</th>
              <th style="width:140px">特征名</th>
              <th style="width:160px">中文名</th>
              <th style="width:145px">分组</th>
              <th>计算口径</th>
              <th style="width:82px">预期方向</th>
              <th>构造理由 / 经济含义</th>
            </tr>
          </thead>
          <tbody id="ml-feat-catalog-tbody">${renderCatalogRowsHtml(filtered)}</tbody>
        </table>
      </div>
    </section>
  `;
}

function patchFeatureCatalogTable() {
  const catalog = state.mlOverview?.feature_catalog;
  if (!catalog?.available) return;
  const tbody = document.getElementById("ml-feat-catalog-tbody");
  const countEl = document.getElementById("ml-catalog-count");
  const barEl = document.getElementById("ml-catalog-group-bar");
  const filtered = getFilteredCatalogItems(catalog);
  if (tbody) tbody.innerHTML = renderCatalogRowsHtml(filtered);
  if (countEl) countEl.textContent = `显示 ${filtered.length} / ${catalog.items.length}`;
  if (barEl) {
    barEl.querySelectorAll("[data-catalog-group]").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.catalogGroup === (state.mlCatalogGroup || "f69"));
    });
  }
}

async function openMlStockExplainModal(asOf, tsCode, initialModel = "xgb_F69") {
  // 复用快捷弹窗展示三模型归因对照（简洁清晰，不额外堆叠弹窗）
  state.promptModal = {
    open: true,
    roleKey: "__ml_explain__",
    tsCode,
    pane: initialModel,
    asOf,
  };
  dom.promptModal.hidden = false;
  dom.promptModalTitle.textContent = `${tsCode} · 三模型打分归因与百分位对照 (${asOf})`;
  dom.promptModalSubtitle.textContent = "TreeSHAP（XGBoost）· 路径分解 Saabas（随机森林）· 积分梯度（LSTM 含 60×11 序列通道）";
  dom.btnModalOpenTab.hidden = true;
  dom.promptModalTabs.innerHTML = ML_MODEL_IDS.map(
    (mid) => `<button class="chip ${mid === initialModel ? "active" : ""}" data-ml-modal-model="${esc(mid)}" type="button">${esc(ML_MODEL_LABELS[mid])}</button>`,
  ).join("");
  dom.promptModalBody.innerHTML = `<div class="pending-placeholder">正在读取三模型归因…</div>`;

  await ensureStockExplain(asOf, tsCode);
  renderMlExplainModalBody();
}

function renderMlExplainModalBody() {
  if (!state.promptModal.open || state.promptModal.roleKey !== "__ml_explain__") return;
  const { asOf, tsCode, pane } = state.promptModal;
  const entry = state.mlExplainCache[`${asOf}|${tsCode}`];
  dom.promptModalTabs.querySelectorAll("[data-ml-modal-model]").forEach((b) => {
    b.classList.toggle("active", b.dataset.mlModalModel === pane);
  });
  if (!entry) {
    dom.promptModalBody.innerHTML = `<div class="pending-placeholder">加载中…</div>`;
    return;
  }
  if (!entry.ok) {
    dom.promptModalBody.innerHTML = `<div class="prompt-warn-banner">${esc(entry.error)}</div>`;
    return;
  }
  const d = entry.data;
  dom.promptModalTitle.textContent = `${d.name || tsCode} (${tsCode}) · 三模型打分归因 (${asOf})`;
  const models = d.models || {};
  const cur = models[pane] || models.xgb_F69;
  const summaryRow = ML_MODEL_IDS.map((mid) => {
    const m = models[mid];
    if (!m || !m.score_available) return `<span class="tag-neutral">${esc(ML_MODEL_LABELS[mid])}：无当日分</span>`;
    return `<span class="tag-neutral mono">${esc(ML_MODEL_LABELS[mid])}：第 ${m.rank} 名 · ${fmtNum(m.pct, 1)}%${m.in_model_picks ? " ★前6" : ""}</span>`;
  }).join("");
  dom.promptModalBody.innerHTML = `
    <div class="hero-tags" style="margin-bottom:12px">${summaryRow}</div>
    ${renderExplainBarsHtml(cur)}
  `;
}

function restorePromptModalChrome() {
  dom.btnModalOpenTab.hidden = false;
  dom.promptModalTabs.innerHTML = `
    <button class="chip active" data-modal-pane="preset" type="button">预置提示词（分块）</button>
    <button class="chip" data-modal-pane="system" type="button">完整 System Prompt</button>
    <button class="chip" data-modal-pane="payload" type="button">实际 User JSON</button>
  `;
}

function bindMlEvents() {
  // 单股详情页内：切换归因模型 Tab、切换 69 特征分组
  dom.stockContent.addEventListener("click", (ev) => {
    const expBtn = ev.target.closest?.("[data-explain-model]");
    if (expBtn) {
      state.mlExplainModelTab = expBtn.dataset.explainModel;
      renderStockDetail();
      return;
    }
    const grpBtn = ev.target.closest?.("[data-feat-group]");
    if (grpBtn) {
      state.featureGroupFilter = grpBtn.dataset.featGroup;
      renderStockDetail();
    }
  });

  // 「模型对比与净值」Tab 内交互委托
  dom.modelsContent.addEventListener("click", async (ev) => {
    const swBtn = ev.target.closest?.("[data-switch-model]");
    if (swBtn) {
      const mid = swBtn.dataset.switchModel;
      if (dom.selModelSource) {
        dom.selModelSource.value = mid;
        dom.selModelSource.dispatchEvent(new Event("change"));
      }
      return;
    }
    const modeBtn = ev.target.closest?.("[data-nav-mode]");
    if (modeBtn) {
      state.mlNavViewMode = modeBtn.dataset.navMode || "nav";
      if (state.mlNavViewMode !== "starts_curve") {
        ensureMlNav(state.mlNavStart);
      }
      renderModelsView();
      return;
    }
    const startBtn = ev.target.closest?.("[data-nav-start]");
    if (startBtn) {
      state.mlNavStart = startBtn.dataset.navStart || "test";
      if (state.mlNavViewMode === "starts_curve") {
        state.mlNavViewMode = "cagr";
      }
      ensureMlNav(state.mlNavStart);
      renderModelsView();
      return;
    }
    const detailBtn = ev.target.closest?.("[data-nav-detail]");
    if (detailBtn) {
      state.mlNavDetailTab = detailBtn.dataset.navDetail || "starts";
      if (state.mlNavDetailTab === "yearly") {
        ensureMlNav(state.mlNavStart);
      }
      renderModelsView();
      return;
    }
    const startRow = ev.target.closest?.("[data-select-start-year]");
    if (startRow) {
      const yr = startRow.dataset.selectStartYear;
      if (yr) {
        state.mlNavStart = String(yr);
        if (state.mlNavViewMode === "starts_curve") {
          state.mlNavViewMode = "cagr";
        }
        ensureMlNav(state.mlNavStart);
        renderModelsView();
      }
      return;
    }
    const legBtn = ev.target.closest?.("[data-nav-toggle]");
    if (legBtn) {
      const k = legBtn.dataset.navToggle;
      state.mlNavHiddenKeys[k] = !state.mlNavHiddenKeys[k];
      renderModelsView();
      return;
    }
    const catGrpBtn = ev.target.closest?.("[data-catalog-group]");
    if (catGrpBtn) {
      state.mlCatalogGroup = catGrpBtn.dataset.catalogGroup;
      patchFeatureCatalogTable();
      return;
    }
    const goBtn = ev.target.closest?.("#btn-consensus-go");
    if (goBtn) {
      const inp = document.getElementById("inp-consensus-date");
      if (inp?.value) {
        loadMlConsensus({ asOf: inp.value, withOutcome: state.mlConsensusWithOutcome });
      }
      return;
    }
    const dryBtn = ev.target.closest?.("[data-ml-dry-run]");
    if (dryBtn) {
      ev.stopPropagation();
      const mid = dryBtn.dataset.mlDryRun;
      const day = state.mlConsensus?.as_of || state.mlConsensusDate || state.selectedDate;
      state.selectedModelSource = mid;
      await refreshMeta({ keepSelection: true });
      await handleDryRun({ modelSource: mid, asOfOverride: day });
      return;
    }
    const startBtn = ev.target.closest?.("[data-ml-start-run]");
    if (startBtn) {
      ev.stopPropagation();
      const mid = startBtn.dataset.mlStartRun;
      const day = state.mlConsensus?.as_of || state.mlConsensusDate || state.selectedDate;
      state.selectedModelSource = mid;
      await refreshMeta({ keepSelection: true });
      await handleStartRun({ modelSource: mid, asOfOverride: day });
      return;
    }
    const stockEl = ev.target.closest?.("[data-ml-inspect-stock]");
    if (stockEl) {
      const code = stockEl.dataset.mlInspectStock;
      const mid = stockEl.dataset.mlInspectModel || "xgb_F69";
      const day = state.mlConsensus?.as_of || state.mlConsensusDate || state.selectedDate;
      openMlStockExplainModal(day, code, mid);
    }
  });

  dom.modelsContent.addEventListener("change", (ev) => {
    if (ev.target?.id === "chk-nav-log") {
      state.mlNavLog = Boolean(ev.target.checked);
      renderModelsView();
    } else if (ev.target?.id === "chk-consensus-outcome") {
      loadMlConsensus({
        asOf: state.mlConsensusDate || state.mlConsensus?.as_of,
        withOutcome: Boolean(ev.target.checked),
      });
    } else if (ev.target?.id === "inp-consensus-date" && ev.target.value) {
      loadMlConsensus({ asOf: ev.target.value, withOutcome: state.mlConsensusWithOutcome });
    }
  });

  dom.modelsContent.addEventListener("input", (ev) => {
    if (ev.target?.id === "inp-feat-catalog-search") {
      state.mlCatalogQuery = ev.target.value || "";
      patchFeatureCatalogTable();
    }
  });

  // 弹窗中三模型归因 Tab 切换
  dom.promptModalTabs.addEventListener("click", (ev) => {
    const mlBtn = ev.target.closest?.("[data-ml-modal-model]");
    if (!mlBtn) return;
    state.promptModal.pane = mlBtn.dataset.mlModalModel;
    renderMlExplainModalBody();
  });
}

function toggleDemoFlow() {
  if (state.demoFlowTimer !== null) {
    clearInterval(state.demoFlowTimer);
    state.demoFlowTimer = null;
    state.demoFlowStage = -1;
    dom.btnDemoFlow.classList.remove("active");
    dom.btnDemoFlow.textContent = "演示流光";
    renderWorkflowCard();
    return;
  }
  state.demoFlowStage = 0;
  dom.btnDemoFlow.classList.add("active");
  dom.btnDemoFlow.textContent = "停止演示";
  renderWorkflowCard();
  state.demoFlowTimer = setInterval(() => {
    state.demoFlowStage = (state.demoFlowStage + 1) % 6;
    renderWorkflowCard();
  }, 1600);
}

async function boot() {
  bindEvents();
  connectSse();
  state.elapsedTimer = setInterval(tickElapsedTimers, 500);

  try {
    await refreshMeta({ keepSelection: false });
    const active = state.meta?.active;
    if (active?.running && active.model_source && active.model_source !== state.selectedModelSource) {
      state.selectedModelSource = active.model_source;
      await refreshMeta({ keepSelection: false });
    }

    if (state.meta?.ml_available) {
      ensureMlOverview();
    }

    const curActive = state.meta?.active;
    if (curActive?.running && curActive.as_of && curActive.hash_prefix) {
      await loadRunDetail(curActive.as_of, curActive.hash_prefix, { modelSource: curActive.model_source });
    } else if (state.meta?.runs?.length > 0) {
      const latest = state.meta.runs[0];
      await loadRunDetail(latest.as_of, latest.hash_prefix);
    } else {
      if (state.meta?.ml_available) {
        state.activeTab = "models";
        ensureMlViewLoaded();
      }
      renderAll();
      await Promise.all([
        refreshChatSessions({ preserveActive: false }),
        refreshPromptsData(),
      ]);
    }
  } catch (err) {
    showToast(`初始化加载失败：${err.message}`, "error", 0);
  }
}

boot();
