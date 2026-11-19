import {
  featureLabel,
  featureUnit,
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
  selectedDate: "",
  selectedRunKey: "", // `${as_of}/${hash_prefix}`
  activeTab: "stock", // "stock" | "final"
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
    { id: "research_copilot", name: "投研复盘助手", desc: "全局复盘视角：解读终选逻辑、跨股对比、排查风险、生成次日开盘清单" },
    { id: "risk_chair", name: "风控主席 (risk_chair)", desc: "以风控拍板视角回应看多/看空与 0~100 打分依据" },
    { id: "final_pick", name: "终选人 (final_pick)", desc: "以全局终选视角解释行业上限约束与保留/换出取舍" },
    { id: "model_analyst", name: "模型解读员 (model_analyst)", desc: "侧重 XGBoost 排名、特征分位与模型盲区拆解" },
    { id: "chart_analyst", name: "走势分析员 (chart_analyst)", desc: "只看 20 日量价形态、均线支撑压力与 MACD 动能" },
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
};

const dom = {
  workspace: document.getElementById("workspace"),
  modelMetaLine: document.getElementById("model-meta-line"),
  selDb: document.getElementById("sel-db"),
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
  tabStockTag: document.getElementById("tab-stock-tag"),
  tabFinalTag: document.getElementById("tab-final-tag"),
  btnCopyMd: document.getElementById("btn-copy-md"),
  btnDownloadJson: document.getElementById("btn-download-json"),

  viewStock: document.getElementById("view-stock"),
  viewFinal: document.getElementById("view-final"),
  stockEmpty: document.getElementById("stock-empty"),
  stockContent: document.getElementById("stock-content"),
  finalContent: document.getElementById("final-content"),
  citationTooltip: document.getElementById("citation-tooltip"),

  // 右侧 AI 投研复盘对话栏
  chatDrawer: document.getElementById("chat-drawer"),
  chatSubtitle: document.getElementById("chat-subtitle"),
  btnChatToggleLeft: document.getElementById("btn-chat-toggle-left"),
  btnChatWide: document.getElementById("btn-chat-wide"),
  btnChatClose: document.getElementById("btn-chat-close"),
  selChatRole: document.getElementById("sel-chat-role"),
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
    renderStockDetail();
  } else {
    dom.viewStock.hidden = true;
    dom.viewFinal.hidden = false;
    renderFinalView();
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
  renderWorkflowGraph(dom.workflowGraph, {
    nodeStates,
    edgeStates,
    activeBadges,
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
  dom.tabStock.classList.toggle("active", state.activeTab === "stock");
  dom.tabFinal.classList.toggle("active", state.activeTab === "final");
}

function switchTab(tab) {
  state.activeTab = tab;
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

  const lastBar = cand.bars?.[cand.bars.length - 1];

  dom.stockContent.innerHTML = `
    <!-- 个股研报头卡 -->
    <section class="card stock-hero">
      <div class="stock-hero-top">
        <div>
          <div class="stock-hero-title">
            <h1>${esc(cand.name)}</h1>
            <span class="stock-hero-code mono">${esc(cand.ts_code)}</span>
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

      <details class="features-drawer">
        <summary>展开查看送入模型的全部 ${Object.keys(f).length} 项量价特征与全池分位</summary>
        <div class="features-table-grid">${allFeatHtml}</div>
      </details>
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
          <span class="pill pill-soft">流式推理中</span>
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
    await refreshMeta({ keepSelection: true });
    await loadRunDetail(payload.as_of, payload.hash_prefix);
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
    showToast(payload.cached ? "命中已冻结运行，已直接载入终选结果。" : "全量辩论与终选已完成并冻结！", "info", 6000);
    await refreshMeta({ keepSelection: true });
    if (payload.as_of && payload.hash_prefix) {
      await loadRunDetail(payload.as_of, payload.hash_prefix);
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
  const meta = await apiJson(`/api/meta?${qs.toString()}`);
  state.meta = meta;
  state.selectedDb = meta.selected_db;
  state.activeJob = meta.active || null;

  // 顶栏模型摘要
  const cfg = meta.config;
  dom.modelMetaLine.textContent = `${cfg.model} · 思考 ${cfg.reasoning_effort} · 并发 ${cfg.concurrency} · ${cfg.api_key_configured ? "密钥已就绪" : "缺少密钥"}`;

  // 填充数据库下拉
  dom.selDb.innerHTML = meta.dbs
    .map((d) => `<option value="${esc(d.path)}">${esc(d.label)}</option>`)
    .join("");
  dom.selDb.value = state.selectedDb;

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

async function loadRunDetail(asOf, hashPrefix) {
  const qs = new URLSearchParams({
    db: state.selectedDb,
    as_of: asOf,
    hash: hashPrefix,
  });
  const detail = await apiJson(`/api/run?${qs.toString()}`);
  state.selectedDate = asOf;
  state.selectedRunKey = `${asOf}/${hashPrefix}`;
  state.chatExtraAttachedCodes = [];
  state.chatExcludeSelected = false;
  dom.selDate.value = asOf;
  dom.selRun.value = state.selectedRunKey;
  hydrateRunDetail(detail);
  await refreshChatSessions({ preserveActive: false });
}

async function handleDryRun() {
  const asOf = dom.selDate.value;
  if (!asOf) {
    showToast("请先选择一个交易日", "warn");
    return;
  }
  showToast(`正在生成 ${asOf} 演练快照（不调大模型）…`, "info", 0);
  try {
    const res = await apiJson("/api/dry-run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        db: state.selectedDb,
        as_of: asOf,
        overrides: {
          reasoning_effort: dom.selEffort.value,
          concurrency: Number(dom.selConcurrency.value),
        },
      }),
    });
    state.isDryRun = true;
    state.selectedRunKey = "";
    dom.selRun.value = "";
    state.streams = {};
    state.run = {
      db: res.db,
      as_of: res.as_of,
      variant: res.variant,
      cadence: res.cadence,
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
    showToast(`演练快照就绪：${res.as_of} 共 ${res.request.candidates.length} 只送辩候选，可直接浏览 20 日 K 线与特征。`, "info", 5000);
    renderAll();
    await refreshChatSessions({ preserveActive: false });
  } catch (err) {
    showToast(`演练失败：${err.message}`, "error", 7000);
  }
}

async function handleStartRun() {
  const asOf = dom.selDate.value;
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
    const resolved = resolveCitation(path, cand, state.run?.request?.context);
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
      renderAll();
      await refreshChatSessions({ preserveActive: false });
    }
  });

  dom.selDate.addEventListener("change", async () => {
    state.selectedDate = dom.selDate.value;
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

  dom.btnDryRun.addEventListener("click", handleDryRun);
  dom.btnStart.addEventListener("click", handleStartRun);
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
}

// ---------------------------------------------------------------------------
// 右侧 AI 投研复盘对话栏（档位 B：6 只读工具 + 逐字思考 + 落盘持久化）
// ---------------------------------------------------------------------------

function chatRunScope() {
  const run = state.run;
  const asOf = run?.as_of || state.selectedDate || "";
  const hash = !state.isDryRun && run?.hash_prefix ? run.hash_prefix : "";
  return {
    db: state.selectedDb || "",
    variant: run?.variant || state.meta?.variant || "",
    cadence: run?.cadence || state.meta?.cadence || "",
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
    // 若有正在运行的任务，优先定位到它；否则自动打开最新一次历史辩论回放
    const active = state.meta?.active;
    if (active?.running && active.as_of && active.hash_prefix) {
      await loadRunDetail(active.as_of, active.hash_prefix);
    } else if (state.meta?.runs?.length > 0) {
      const latest = state.meta.runs[0];
      await loadRunDetail(latest.as_of, latest.hash_prefix);
    } else {
      renderAll();
      await refreshChatSessions({ preserveActive: false });
    }
  } catch (err) {
    showToast(`初始化加载失败：${err.message}`, "error", 0);
  }
}

boot();
