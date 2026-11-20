/**
 * 可视化组件：
 * 1. Magpie 风格流式路由工作流 SVG 图（平滑贝塞尔路径 + 发光流光粒子 + 并发节点呼吸光晕）
 * 2. 个股近 20 日前复权日 K 线 + 成交额 + MACD 柱组合图（带悬停十字准星）
 * 3. 字段引用值解析器（用于悬停显示原始快照数值与分位）
 */

const SVG_NS = "http://www.w3.org/2000/svg";

/** 工作流节点拓扑坐标（ViewBox 380 x 342）：紧凑、清晰简洁呈现 分叉-汇合-裁决-终选 顺序。 */
const WORKFLOW_NODES = [
  { id: "snapshot", x: 190, y: 22, w: 162, h: 30, title: "① 模型前 20 快照", sub: "69 特征 · 归因 · 20 日线" },
  { id: "model_analyst", x: 98, y: 82, w: 150, h: 32, title: "模型解读员", sub: "model_analyst" },
  { id: "chart_analyst", x: 282, y: 82, w: 150, h: 32, title: "走势分析员", sub: "chart_analyst" },
  { id: "bull", x: 190, y: 138, w: 166, h: 30, title: "③ 多方立论", sub: "bull · 第 1 轮" },
  { id: "bear", x: 190, y: 180, w: 166, h: 30, title: "④ 空方反驳", sub: "bear · 第 2 轮" },
  { id: "bull_counter", x: 190, y: 222, w: 166, h: 30, title: "⑤ 多方二次回应", sub: "bull_counter · 第 3 轮" },
  { id: "risk_chair", x: 190, y: 266, w: 176, h: 30, title: "⑥ 风控主席裁决", sub: "risk_chair · 20股对照" },
  { id: "final_pick", x: 190, y: 312, w: 176, h: 30, title: "⑦ 终选官定稿前 6", sub: "final_pick · 行业上限" },
];

const WORKFLOW_EDGES = [
  {
    id: "e-snap-model",
    from: "snapshot",
    to: "model_analyst",
    d: "M 164 37 C 164 55, 98 48, 98 66",
  },
  {
    id: "e-snap-chart",
    from: "snapshot",
    to: "chart_analyst",
    d: "M 216 37 C 216 55, 282 48, 282 66",
  },
  {
    id: "e-model-bull",
    from: "model_analyst",
    to: "bull",
    d: "M 98 98 C 98 115, 164 108, 164 123",
  },
  {
    id: "e-chart-bull",
    from: "chart_analyst",
    to: "bull",
    d: "M 282 98 C 282 115, 216 108, 216 123",
  },
  { id: "e-bull-bear", from: "bull", to: "bear", d: "M 190 153 L 190 165" },
  { id: "e-bear-counter", from: "bear", to: "bull_counter", d: "M 190 195 L 190 207" },
  { id: "e-counter-risk", from: "bull_counter", to: "risk_chair", d: "M 190 237 L 190 251" },
  { id: "e-risk-final", from: "risk_chair", to: "final_pick", d: "M 190 281 L 190 297" },
];

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, String(v));
  }
  return el;
}

/**
 * 渲染并更新 Magpie 风格流式路由 SVG 工作流图。
 * @param {HTMLElement} container
 * @param {object} state - { nodeStates, edgeStates, activeBadges, onNodeClick }
 */
export function renderWorkflowGraph(container, state) {
  const { nodeStates = {}, edgeStates = {}, activeBadges = {}, onNodeClick, snapshotSub = "" } = state;
  const svg = svgEl("svg", {
    viewBox: "0 0 380 336",
    class: "wf-svg",
    role: "img",
    "aria-label": "AI 多角色辩论路由工作流图",
  });

  // 渐变与柔和发光滤镜
  const defs = svgEl("defs");
  const doneGrad = svgEl("linearGradient", { id: "wf-done-grad", x1: "0%", y1: "0%", x2: "0%", y2: "100%" });
  doneGrad.append(
    svgEl("stop", { offset: "0%", "stop-color": "#93c5fd" }),
    svgEl("stop", { offset: "100%", "stop-color": "#2563eb" }),
  );
  const activeGrad = svgEl("linearGradient", { id: "wf-active-grad", x1: "0%", y1: "0%", x2: "0%", y2: "100%" });
  activeGrad.append(
    svgEl("stop", { offset: "0%", "stop-color": "#60a5fa" }),
    svgEl("stop", { offset: "50%", "stop-color": "#2563eb" }),
    svgEl("stop", { offset: "100%", "stop-color": "#1d4ed8" }),
  );
  const glow = svgEl("filter", { id: "wf-glow", x: "-60%", y: "-60%", width: "220%", height: "220%" });
  glow.append(
    svgEl("feGaussianBlur", { stdDeviation: "2.4", result: "blur" }),
    (() => {
      const merge = svgEl("feMerge");
      merge.append(svgEl("feMergeNode", { in: "blur" }), svgEl("feMergeNode", { in: "SourceGraphic" }));
      return merge;
    })(),
  );
  defs.append(doneGrad, activeGrad, glow);
  svg.append(defs);

  // 左侧独立双视角说明微标
  const splitNote = svgEl("text", { x: 190, y: 58, "text-anchor": "middle", class: "wf-roster-tag" });
  splitNote.textContent = "独立双视角（互不可见）";
  svg.append(splitNote);

  // 边层：基底轨道 + 已完成亮线 + 活跃流动虚线 + 发光粒子沿路径滑行
  const edgeGroup = svgEl("g");
  for (const edge of WORKFLOW_EDGES) {
    const status = edgeStates[edge.id] || "idle";
    edgeGroup.append(svgEl("path", { d: edge.d, class: "wf-edge-base" }));
    if (status === "done") {
      edgeGroup.append(svgEl("path", { d: edge.d, class: "wf-edge-done" }));
    } else if (status === "active") {
      edgeGroup.append(svgEl("path", { d: edge.d, class: "wf-edge-active" }));
      // 两颗错峰流动光点，营造高级路由流光感
      for (const begin of ["0s", "0.75s"]) {
        const dot = svgEl("circle", { r: 3.2, class: "wf-particle" });
        const anim = svgEl("animateMotion", {
          dur: "1.5s",
          repeatCount: "indefinite",
          begin,
          path: edge.d,
        });
        dot.append(anim);
        edgeGroup.append(dot);
      }
    }
  }
  svg.append(edgeGroup);

  // 节点层
  const nodeGroup = svgEl("g");
  for (const node of WORKFLOW_NODES) {
    const status = nodeStates[node.id] || "idle";
    const g = svgEl("g", {
      class: `wf-node wf-node-${status}`,
      "data-node-id": node.id,
    });
    const left = node.x - node.w / 2;
    const top = node.y - node.h / 2;

    // 呼吸光晕环
    g.append(
      svgEl("rect", {
        x: left - 2,
        y: top - 2,
        width: node.w + 4,
        height: node.h + 4,
        rx: 10,
        class: "wf-node-halo",
      }),
    );

    g.append(
      svgEl("rect", {
        x: left,
        y: top,
        width: node.w,
        height: node.h,
        rx: 8,
        class: "wf-node-bg",
      }),
    );

    // 左侧状态圆点 / 对勾
    const dotX = left + 13;
    const dotY = node.y;
    if (status === "done") {
      g.append(svgEl("circle", { cx: dotX, cy: dotY, r: 5.5, fill: "#dcfce7", stroke: "#86efac", "stroke-width": 1 }));
      g.append(
        svgEl("path", {
          d: `M ${dotX - 2.6} ${dotY} L ${dotX - 0.6} ${dotY + 2.1} L ${dotX + 3} ${dotY - 2}`,
          fill: "none",
          stroke: "#15803d",
          "stroke-width": 1.6,
          "stroke-linecap": "round",
          "stroke-linejoin": "round",
        }),
      );
    } else if (status === "active") {
      g.append(svgEl("circle", { cx: dotX, cy: dotY, r: 4.2, fill: "#2563eb" }));
    } else if (status === "error") {
      g.append(svgEl("circle", { cx: dotX, cy: dotY, r: 4.2, fill: "#dc2626" }));
    } else {
      g.append(svgEl("circle", { cx: dotX, cy: dotY, r: 3.4, fill: "#cbd5e1" }));
    }

    const title = svgEl("text", {
      x: left + 25,
      y: node.y - 2,
      class: "wf-node-title",
    });
    title.textContent = node.title;
    g.append(title);

    const defaultSub = node.id === "snapshot" && snapshotSub ? snapshotSub : node.sub;
    const badgeText = activeBadges[node.id] || defaultSub;
    const sub = svgEl("text", {
      x: left + 25,
      y: node.y + 9.5,
      class: activeBadges[node.id] ? "wf-node-badge" : "wf-node-sub",
    });
    sub.textContent = badgeText.length > 24 ? `${badgeText.slice(0, 23)}…` : badgeText;
    g.append(sub);

    if (onNodeClick) {
      g.addEventListener("click", () => onNodeClick(node.id));
    }
    nodeGroup.append(g);
  }
  svg.append(nodeGroup);

  container.innerHTML = "";
  container.append(svg);
}

/**
 * 绘制个股近 20 日前复权 K 线 + 成交额 + MACD 柱组合 SVG 图。
 */
export function renderStockKlineChart(container, bars, onHoverBar) {
  if (!Array.isArray(bars) || bars.length === 0) {
    container.innerHTML = `<div class="empty-hint">暂无日线数据</div>`;
    return;
  }

  const width = 660;
  const height = 216;
  const padL = 44;
  const padR = 12;
  const priceTop = 10;
  const priceH = 112;
  const volTop = 132;
  const volH = 34;
  const macdTop = 174;
  const macdH = 30;
  const plotW = width - padL - padR;

  const lows = bars.map((b) => b.low);
  const highs = bars.map((b) => b.high);
  let minP = Math.min(...lows);
  let maxP = Math.max(...highs);
  const spanP = Math.max(maxP - minP, minP * 0.02, 0.01);
  minP -= spanP * 0.08;
  maxP += spanP * 0.08;

  const maxAmt = Math.max(...bars.map((b) => b.amount_yi || 0), 0.1);
  const maxHist = Math.max(...bars.map((b) => Math.abs(b.hist_ratio || 0)), 0.1);

  const stepX = plotW / bars.length;
  const bodyW = Math.max(5, Math.min(16, stepX * 0.56));
  const yPrice = (v) => priceTop + ((maxP - v) / (maxP - minP)) * priceH;
  const yVol = (v) => volTop + volH - (v / maxAmt) * volH;
  const macdMid = macdTop + macdH / 2;

  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, class: "kline-svg" });

  // 水平网格线 + 价位刻度
  for (let i = 0; i <= 3; i += 1) {
    const val = maxP - ((maxP - minP) * i) / 3;
    const y = yPrice(val);
    svg.append(
      svgEl("line", {
        x1: padL,
        y1: y,
        x2: width - padR,
        y2: y,
        stroke: "#eef2f7",
        "stroke-width": 1,
      }),
    );
    const lbl = svgEl("text", {
      x: padL - 6,
      y: y + 3,
      "text-anchor": "end",
      fill: "#94a3b8",
      "font-size": 9.5,
      class: "mono",
    });
    lbl.textContent = val.toFixed(2);
    svg.append(lbl);
  }

  // 分区小标题
  const volLbl = svgEl("text", { x: padL - 6, y: volTop + 10, "text-anchor": "end", fill: "#94a3b8", "font-size": 9 });
  volLbl.textContent = "额(亿)";
  svg.append(volLbl);

  const macdLbl = svgEl("text", { x: padL - 6, y: macdMid + 3, "text-anchor": "end", fill: "#94a3b8", "font-size": 9 });
  macdLbl.textContent = "MACD";
  svg.append(macdLbl);
  svg.append(
    svgEl("line", {
      x1: padL,
      y1: macdMid,
      x2: width - padR,
      y2: macdMid,
      stroke: "#e2e8f0",
      "stroke-width": 1,
    }),
  );

  // 信号日高亮底带（最后一根）
  const lastX = padL + (bars.length - 1) * stepX;
  svg.append(
    svgEl("rect", {
      x: lastX + 1,
      y: priceTop,
      width: Math.max(stepX - 2, 4),
      height: height - priceTop - 10,
      fill: "#eff6ff",
      rx: 3,
    }),
  );

  // 悬停引导线
  const hoverLine = svgEl("line", {
    x1: 0,
    y1: priceTop,
    x2: 0,
    y2: height - 10,
    stroke: "#2563eb",
    "stroke-width": 1,
    "stroke-dasharray": "3 3",
    opacity: 0,
  });
  svg.append(hoverLine);

  bars.forEach((bar, idx) => {
    const cx = padL + idx * stepX + stepX / 2;
    const isUp = bar.close >= bar.open;
    const color = isUp ? "#c81e1e" : "#0f766e";
    const fill = isUp ? "#fef2f2" : "#0f766e";

    // 上下影线
    svg.append(
      svgEl("line", {
        x1: cx,
        y1: yPrice(bar.high),
        x2: cx,
        y2: yPrice(bar.low),
        stroke: color,
        "stroke-width": 1.2,
      }),
    );

    // K 线实体
    const yOpen = yPrice(bar.open);
    const yClose = yPrice(bar.close);
    const rectY = Math.min(yOpen, yClose);
    const rectH = Math.max(Math.abs(yClose - yOpen), 1.6);
    svg.append(
      svgEl("rect", {
        x: cx - bodyW / 2,
        y: rectY,
        width: bodyW,
        height: rectH,
        fill,
        stroke: color,
        "stroke-width": 1.1,
        rx: 1,
      }),
    );

    // 涨跌停标记
    if (bar.limit === "up" || bar.limit === "down") {
      const mk = svgEl("text", {
        x: cx,
        y: bar.limit === "up" ? Math.max(10, yPrice(bar.high) - 3) : Math.min(priceTop + priceH, yPrice(bar.low) + 9),
        "text-anchor": "middle",
        fill: bar.limit === "up" ? "#b91c1c" : "#0f766e",
        "font-size": 8.5,
        "font-weight": 700,
      });
      mk.textContent = bar.limit === "up" ? "涨停" : "跌停";
      svg.append(mk);
    }

    // 成交额柱
    const vY = yVol(bar.amount_yi || 0);
    svg.append(
      svgEl("rect", {
        x: cx - bodyW / 2,
        y: vY,
        width: bodyW,
        height: Math.max(volTop + volH - vY, 1),
        fill: isUp ? "#fca5a5" : "#99f6e4",
        rx: 1,
      }),
    );

    // MACD hist_ratio 柱
    const hist = typeof bar.hist_ratio === "number" ? bar.hist_ratio : 0;
    const hPx = (Math.abs(hist) / maxHist) * (macdH / 2 - 1);
    svg.append(
      svgEl("rect", {
        x: cx - bodyW / 2,
        y: hist >= 0 ? macdMid - hPx : macdMid,
        width: bodyW,
        height: Math.max(hPx, 1),
        fill: hist >= 0 ? "#dc2626" : "#0d9488",
        opacity: 0.85,
        rx: 1,
      }),
    );

    // 首、中、尾日期标注
    if (idx === 0 || idx === Math.floor(bars.length / 2) || idx === bars.length - 1) {
      const dt = svgEl("text", {
        x: cx,
        y: height - 1,
        "text-anchor": idx === 0 ? "start" : idx === bars.length - 1 ? "end" : "middle",
        fill: idx === bars.length - 1 ? "#1d4ed8" : "#94a3b8",
        "font-size": 9,
        "font-weight": idx === bars.length - 1 ? 700 : 400,
        class: "mono",
      });
      dt.textContent = bar.date.slice(5);
      svg.append(dt);
    }

    // 透明交互热区
    const hit = svgEl("rect", {
      x: padL + idx * stepX,
      y: priceTop,
      width: stepX,
      height: height - priceTop - 10,
      fill: "transparent",
      style: "cursor: crosshair;",
    });
    hit.addEventListener("mouseenter", () => {
      hoverLine.setAttribute("x1", String(cx));
      hoverLine.setAttribute("x2", String(cx));
      hoverLine.setAttribute("opacity", "0.75");
      if (onHoverBar) onHoverBar(bar, idx);
    });
    svg.append(hit);
  });

  svg.addEventListener("mouseleave", () => {
    hoverLine.setAttribute("opacity", "0");
    if (onHoverBar) onHoverBar(bars[bars.length - 1], bars.length - 1);
  });

  container.innerHTML = "";
  container.append(svg);
}

const FEATURE_META = {
  pct_chg: ["当日涨跌幅", "%"],
  r1: ["1 日涨跌幅", "%"],
  r5: ["近 5 日涨跌幅", "%"],
  r10: ["近 10 日涨跌幅", "%"],
  r20: ["近 20 日涨跌幅", "%"],
  r60: ["近 60 日涨跌幅", "%"],
  r120: ["近 120 日涨跌幅", "%"],
  from_low60: ["距 60 日最低收盘", "%"],
  from_high60: ["距 60 日最高收盘", "%"],
  dev_ma5: ["偏离 5 日均线", "%"],
  dev_ma20: ["偏离 20 日均线", "%"],
  dev_ma60: ["偏离 60 日均线", "%"],
  ma20_gt_ma60: ["20 日线在 60 日线上", ""],
  ma20_slope5: ["20 日均线 5 日斜率", "%"],
  ma60_slope10: ["60 日均线 10 日斜率", "%"],
  dd20: ["距 20 日最高回撤", "%"],
  vol5: ["5 日量比", "倍"],
  vol20: ["20 日量比", "倍"],
  amp20: ["近 20 日平均振幅", "%"],
  launch10: ["近 10 日启动天数", "天"],
  up_days10: ["近 10 日上涨天数", "天"],
  max_up20: ["近 20 日最大单日涨幅", "%"],
  limit_up_today: ["当日是否涨停", ""],
  limit_up20: ["近 20 日涨停天数", "天"],
  dif_ratio: ["MACD DIF 比率", "%"],
  hist_ratio: ["MACD 柱比率", "%"],
  rank_r20: ["20 日涨幅主板分位", "分位"],
  rank_r60: ["60 日涨幅主板分位", "分位"],
};

export function featureLabel(key) {
  return FEATURE_META[key]?.[0] || key;
}

export function featureUnit(key) {
  return FEATURE_META[key]?.[1] || "";
}

/**
 * 解析引用字段路径，返回 { value, desc } 供悬停浮窗展示真实快照数值。
 */
export function resolveCitation(path, candidate, context, featureCatalog = null) {
  if (!path || !candidate) return null;
  const clean = path.replace(/^candidate\./, "").trim();

  if (clean.startsWith("features.")) {
    const k = clean.slice("features.".length);
    if (k in (candidate.features || {})) {
      const val = candidate.features[k];
      const pct = candidate.feature_pct?.[k];
      const unit = featureUnit(k);
      const pctText = pct !== undefined ? `（全池 ${pct} 分位）` : "";
      return { value: `${val}${unit}`, desc: `${featureLabel(k)}${pctText}` };
    }
  }
  if (clean.startsWith("feature_pct.")) {
    const k = clean.slice("feature_pct.".length);
    if (k in (candidate.feature_pct || {})) {
      const pct = candidate.feature_pct[k];
      const raw = candidate.features?.[k];
      const rawText = raw !== undefined ? `（原值 ${raw}${featureUnit(k)}）` : "";
      return { value: `${pct} 分位`, desc: `${featureLabel(k)}在全池中的分位${rawText}` };
    }
  }
  if (clean.startsWith("extra_features.")) {
    const k = clean.slice("extra_features.".length);
    if (k in (candidate.extra_features || {})) {
      const val = candidate.extra_features[k];
      const pct = candidate.extra_feature_pct?.[k];
      const meta = featureCatalog?.by_name?.[k];
      const label = meta?.cn_name || k;
      const pctText = pct !== undefined ? `（全池 ${pct} 分位）` : "";
      return { value: String(val), desc: `${label}${pctText}${meta?.formula ? ` · ${meta.formula}` : ""}` };
    }
  }
  if (clean.startsWith("extra_feature_pct.")) {
    const k = clean.slice("extra_feature_pct.".length);
    if (k in (candidate.extra_feature_pct || {})) {
      const pct = candidate.extra_feature_pct[k];
      const raw = candidate.extra_features?.[k];
      const meta = featureCatalog?.by_name?.[k];
      const label = meta?.cn_name || k;
      const rawText = raw !== undefined ? `（原值 ${raw}）` : "";
      return { value: `${pct} 分位`, desc: `${label}全池分位${rawText}` };
    }
  }
  if (clean.startsWith("explain.") && candidate.explain) {
    const m = /^explain\.top_factors\[(\d+)\](?:\.([a-zA-Z0-9_]+))?$/.exec(clean);
    if (m && Array.isArray(candidate.explain.top_factors)) {
      const f = candidate.explain.top_factors[Number(m[1])];
      if (f) {
        const sign = f.contribution > 0 ? "+" : "";
        return {
          value: `${f.cn_name}（${sign}${f.contribution} · ${f.direction}）`,
          desc: `特征 ${f.feature}，数值 ${f.value ?? "—"} · ${candidate.explain.method || ""}`,
        };
      }
    }
    const sub = clean.slice("explain.".length);
    if (sub in candidate.explain) {
      return { value: String(candidate.explain[sub]), desc: `模型归因字段 explain.${sub}` };
    }
  }
  if (clean.startsWith("peer_models.") && candidate.peer_models) {
    const parts = clean.slice("peer_models.".length).split(".");
    const peer = candidate.peer_models[parts[0]];
    if (peer) {
      if (parts[1] && parts[1] in peer) {
        return {
          value: String(peer[parts[1]]),
          desc: `同日对照模型 ${peer.name}（第 ${peer.rank} 名 · ${peer.pct}% 分位）`,
        };
      }
      return {
        value: `第 ${peer.rank} 名 · ${peer.pct}% 分位`,
        desc: `同日对照模型 ${peer.name}${peer.in_model_picks ? "（入选该模型前 6）" : ""}`,
      };
    }
  }
  if (clean.startsWith("model.")) {
    const k = clean.slice("model.".length);
    if (k in (candidate.model || {})) {
      return { value: String(candidate.model[k]), desc: `模型打分字段 model.${k}` };
    }
  }
  if (clean.startsWith("industry_stats.")) {
    const k = clean.slice("industry_stats.".length);
    if (k in (candidate.industry_stats || {})) {
      return { value: String(candidate.industry_stats[k]), desc: `同行业统计（${candidate.industry}）` };
    }
  }
  if (clean === "flags" || clean.startsWith("flags.")) {
    const flags = candidate.flags || [];
    if (clean === "flags") {
      return {
        value: flags.length ? flags.map((f) => f.code).join(", ") : "空（无异常提示）",
        desc: flags.length ? flags.map((f) => f.text).join("；") : "无涨停/跌停/拥挤等风险提示",
      };
    }
    const code = clean.slice("flags.".length);
    const hit = flags.find((f) => f.code === code);
    return hit
      ? { value: hit.code, desc: hit.text }
      : { value: "未触发", desc: `当前候选未触发 ${code}` };
  }
  const barMatch = /^bars\[(-?\d+)\](?:\.([a-zA-Z0-9_]+))?$/.exec(clean);
  if (barMatch && Array.isArray(candidate.bars)) {
    let idx = Number(barMatch[1]);
    if (idx < 0) idx = candidate.bars.length + idx;
    const bar = candidate.bars[idx];
    if (bar) {
      const field = barMatch[2];
      if (!field) {
        return {
          value: `${bar.date} 收 ${bar.close} (${bar.pct_chg > 0 ? "+" : ""}${bar.pct_chg}%)`,
          desc: `成交额 ${bar.amount_yi} 亿 · 量比 ${bar.vol5 ?? "—"} · MACD柱 ${bar.hist_ratio ?? "—"}`,
        };
      }
      if (field in bar) {
        return { value: String(bar[field]), desc: `${bar.date} 日线字段 ${field}` };
      }
    }
  }
  if (clean.startsWith("context.pool.") && context?.pool) {
    const k = clean.slice("context.pool.".length);
    if (k in context.pool) {
      return { value: `${context.pool[k]}%`, desc: `信号日全池环境 context.pool.${k}` };
    }
  }
  return null;
}


export const NAV_SERIES_STYLES = {
  xgb_F69: { color: "#1d4ed8", width: 2.3, dash: "" },
  rf_F69: { color: "#0f766e", width: 2.0, dash: "" },
  lstm_SF69: { color: "#b91c1c", width: 2.0, dash: "" },
  pool_nav: { color: "#64748b", width: 1.5, dash: "5 3" },
  hs300: { color: "#d97706", width: 1.35, dash: "3 3" },
  zz500: { color: "#7c3aed", width: 1.35, dash: "3 3" },
  zz1000: { color: "#0284c7", width: 1.35, dash: "2 2" },
};

/**
 * 渲染多模型净值对比折线图 / 截至当日动态年化收益率曲线（SVG，支持线性/对数坐标与悬停十字准星）。
 */
export function renderNavComparisonChart(
  container,
  navData,
  { visibleKeys, mode = "nav", logScale = false, onHover } = {},
) {
  const dates = navData?.dates || [];
  const allSeries = navData?.series || {};
  const keys = (visibleKeys || Object.keys(allSeries)).filter((k) => allSeries[k]);
  if (!dates.length || !keys.length) {
    container.innerHTML = `<div class="empty-hint">暂无净值曲线数据</div>`;
    return;
  }

  const isCagr = mode === "cagr";
  const useLog = !isCagr && Boolean(logScale);
  const capital = Number(navData?.capital) || 200000;
  const cagrMinDays = Number(navData?.cagr_min_days) || 60;

  const getArray = (k) => (isCagr ? allSeries[k].cagr_values || [] : allSeries[k].values || []);
  const isValidVal = (v) =>
    typeof v === "number" && Number.isFinite(v) && (isCagr ? true : v > 0);

  const width = 820;
  const height = 286;
  const padL = 58;
  const padR = 16;
  const padT = 16;
  const padB = 30;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  let minV = Infinity;
  let maxV = -Infinity;
  for (const k of keys) {
    for (const v of getArray(k)) {
      if (isValidVal(v)) {
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      }
    }
  }
  if (!Number.isFinite(minV) || !Number.isFinite(maxV)) {
    container.innerHTML = `<div class="empty-hint">${isCagr ? "有效交易日不足，暂无年化曲线数据" : "净值数据为空"}</div>`;
    return;
  }
  if (isCagr) {
    // 尽量包含 0% 基准线，且避免极窄波动区间过度放大
    if (minV > 0 && minV < 8) minV = 0;
    if (maxV < 0 && maxV > -8) maxV = 0;
    if (maxV - minV < 4) {
      minV -= 2;
      maxV += 2;
    }
  } else if (maxV <= minV) {
    maxV = minV + 0.1;
  }

  const mapVal = (v) => (useLog ? Math.log(Math.max(v, 1e-4)) : v);
  const lo = mapVal(minV);
  const hi = mapVal(maxV);
  const padSpan = (hi - lo) * 0.07 || (isCagr ? 2 : 0.05);
  const yLo = lo - padSpan;
  const yHi = hi + padSpan;

  const xOf = (idx) => padL + (dates.length <= 1 ? 0 : (idx / (dates.length - 1)) * plotW);
  const yOf = (v) => padT + ((yHi - mapVal(v)) / (yHi - yLo)) * plotH;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "nav-svg",
    role: "img",
    "aria-label": isCagr ? "三模型截至当日累计年化收益率曲线" : "三模型组合净值对比曲线",
  });

  // 动态年化模式：淡灰底纹标出前 60 个交易日观察期
  if (isCagr && dates.length > cagrMinDays) {
    const warmX = xOf(cagrMinDays);
    svg.append(
      svgEl("rect", {
        x: padL,
        y: padT,
        width: Math.max(0, warmX - padL),
        height: plotH,
        fill: "#f8fafc",
      }),
    );
    svg.append(
      svgEl("line", {
        x1: warmX,
        y1: padT,
        x2: warmX,
        y2: padT + plotH,
        stroke: "#cbd5e1",
        "stroke-width": 1,
        "stroke-dasharray": "3 3",
      }),
    );
    if (warmX - padL >= 52) {
      const wLbl = svgEl("text", {
        x: padL + (warmX - padL) / 2,
        y: padT + 14,
        "text-anchor": "middle",
        fill: "#94a3b8",
        "font-size": 9.5,
      });
      wLbl.textContent = `前${cagrMinDays}日观察期`;
      svg.append(wLbl);
    }
  }

  // 水平网格与 Y 轴刻度
  const tickCount = 5;
  for (let i = 0; i <= tickCount; i += 1) {
    const frac = i / tickCount;
    const yVal = yHi - frac * (yHi - yLo);
    const dispVal = useLog ? Math.exp(yVal) : yVal;
    const y = padT + frac * plotH;
    svg.append(
      svgEl("line", {
        x1: padL,
        y1: y,
        x2: width - padR,
        y2: y,
        stroke: "#e5e7eb",
        "stroke-width": 1,
        "stroke-dasharray": i === tickCount ? "" : "2 3",
      }),
    );
    const lbl = svgEl("text", {
      x: padL - 8,
      y: y + 3.5,
      "text-anchor": "end",
      fill: "#6b7280",
      "font-size": 10.5,
      class: "mono",
    });
    if (isCagr) {
      const rounded = Math.abs(dispVal) < 0.05 ? 0 : dispVal;
      lbl.textContent = `${rounded > 0 ? "+" : ""}${rounded.toFixed(0)}%`;
    } else {
      lbl.textContent = dispVal >= 10 ? dispVal.toFixed(1) : dispVal.toFixed(2);
    }
    svg.append(lbl);
  }

  // 基准参考线（净值模式 1.0；年化模式 0%）
  const refBaseline = isCagr ? 0.0 : 1.0;
  if (minV <= refBaseline && maxV >= refBaseline) {
    const yRef = yOf(refBaseline);
    svg.append(
      svgEl("line", {
        x1: padL,
        y1: yRef,
        x2: width - padR,
        y2: yRef,
        stroke: "#94a3b8",
        "stroke-width": 1.1,
        "stroke-dasharray": "4 4",
      }),
    );
  }

  // X 轴日期刻度（最多 7 个均匀分布）
  const xTicks = Math.min(7, dates.length);
  for (let i = 0; i < xTicks; i += 1) {
    const idx = Math.round((i / Math.max(1, xTicks - 1)) * (dates.length - 1));
    const x = xOf(idx);
    const lbl = svgEl("text", {
      x,
      y: height - 8,
      "text-anchor": i === 0 ? "start" : i === xTicks - 1 ? "end" : "middle",
      fill: "#6b7280",
      "font-size": 10.5,
      class: "mono",
    });
    lbl.textContent = dates[idx].slice(0, 7);
    svg.append(lbl);
  }

  // 降采样步长（超过 700 个点时按步长抽样，保住首尾有效点与视觉平滑）
  const step = Math.max(1, Math.floor(dates.length / 650));

  for (const k of keys) {
    const vals = getArray(k);
    const st = NAV_SERIES_STYLES[k] || { color: "#2563eb", width: 1.8, dash: "" };
    const pts = [];
    let firstValid = -1;
    for (let i = 0; i < dates.length; i += 1) {
      if (isValidVal(vals[i])) {
        firstValid = i;
        break;
      }
    }
    if (firstValid >= 0) {
      for (let i = firstValid; i < dates.length; i += step) {
        const v = vals[i];
        if (isValidVal(v)) {
          pts.push(`${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`);
        }
      }
      const lastIdx = dates.length - 1;
      if ((lastIdx - firstValid) % step !== 0 && isValidVal(vals[lastIdx])) {
        pts.push(`${xOf(lastIdx).toFixed(1)},${yOf(vals[lastIdx]).toFixed(1)}`);
      }
    }
    if (pts.length >= 2) {
      const attrs = {
        points: pts.join(" "),
        fill: "none",
        stroke: st.color,
        "stroke-width": st.width,
        "stroke-linejoin": "round",
        "stroke-linecap": "round",
      };
      if (st.dash) attrs["stroke-dasharray"] = st.dash;
      svg.append(svgEl("polyline", attrs));
    }
  }

  // 悬停十字准星与圆点
  const hoverLine = svgEl("line", {
    x1: padL,
    y1: padT,
    x2: padL,
    y2: padT + plotH,
    stroke: "#475569",
    "stroke-width": 1,
    "stroke-dasharray": "3 3",
    opacity: 0,
  });
  svg.append(hoverLine);

  const dots = {};
  for (const k of keys) {
    const st = NAV_SERIES_STYLES[k] || { color: "#2563eb" };
    const dot = svgEl("circle", {
      cx: padL,
      cy: padT,
      r: 3.6,
      fill: st.color,
      stroke: "#ffffff",
      "stroke-width": 1.4,
      opacity: 0,
    });
    dots[k] = dot;
    svg.append(dot);
  }

  const overlay = svgEl("rect", {
    x: padL,
    y: padT,
    width: plotW,
    height: plotH,
    fill: "transparent",
    style: "cursor: crosshair;",
  });

  const emitIdx = (idx) => {
    const clamped = Math.max(0, Math.min(dates.length - 1, idx));
    const cx = xOf(clamped);
    hoverLine.setAttribute("x1", String(cx));
    hoverLine.setAttribute("x2", String(cx));
    hoverLine.setAttribute("opacity", "0.75");
    const snapshot = { date: dates[clamped], index: clamped, mode, items: [] };
    for (const k of keys) {
      const navV = allSeries[k].values?.[clamped];
      const cagrV = allSeries[k].cagr_values?.[clamped];
      const plotV = isCagr ? cagrV : navV;
      if (isValidVal(plotV)) {
        dots[k].setAttribute("cx", String(cx));
        dots[k].setAttribute("cy", String(yOf(plotV)));
        dots[k].setAttribute("opacity", "1");
      } else {
        dots[k].setAttribute("opacity", "0");
      }
      if (typeof navV === "number" && Number.isFinite(navV) && navV > 0) {
        snapshot.items.push({
          id: k,
          label: allSeries[k].label,
          kind: allSeries[k].kind,
          nav: navV,
          capitalWan: (navV * capital) / 10000,
          retPct: (navV - 1) * 100,
          cagrPct: typeof cagrV === "number" && Number.isFinite(cagrV) ? cagrV : null,
          color: (NAV_SERIES_STYLES[k] || {}).color || "#2563eb",
        });
      }
    }
    if (onHover) onHover(snapshot);
  };

  overlay.addEventListener("mousemove", (ev) => {
    const rect = overlay.getBoundingClientRect();
    if (rect.width <= 0) return;
    const ratio = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
    const idx = Math.round(ratio * (dates.length - 1));
    emitIdx(idx);
  });

  overlay.addEventListener("mouseleave", () => {
    hoverLine.setAttribute("opacity", "0");
    for (const k of keys) dots[k].setAttribute("opacity", "0");
    if (onHover) onHover(null);
  });

  svg.append(overlay);
  container.innerHTML = "";
  container.append(svg);
}

/**
 * 渲染「换起点年化对比曲线」（X 轴为 2015 ~ 2024 起步年，Y 轴为对应至今 2026-10 的年化收益率 %）。
 * 点击任意年份列可直接切换到该起点的逐日曲线。
 */
export function renderStartsAnnualizedChart(
  container,
  startsSummary,
  { visibleKeys, activeStartYear = null, onHover, onSelectYear } = {},
) {
  const rows = startsSummary?.rows || [];
  if (!rows.length) {
    container.innerHTML = `<div class="empty-hint">暂无换起点年化对比数据</div>`;
    return;
  }

  const allIds = ["xgb_F69", "rf_F69", "lstm_SF69", "pool_nav"];
  const labels = {
    xgb_F69: "XGBoost F69",
    rf_F69: "随机森林 F69",
    lstm_SF69: "LSTM S+F69",
    pool_nav: "准入池等权（基准）",
  };
  const keys = (visibleKeys || allIds).filter((k) => allIds.includes(k));
  if (!keys.length) {
    container.innerHTML = `<div class="empty-hint">请至少显示一条对比曲线</div>`;
    return;
  }

  const valOf = (row, k) => (k === "pool_nav" ? row.pool_cagr : row.models?.[k]?.cagr);

  let minV = 0;
  let maxV = 0;
  for (const r of rows) {
    for (const k of keys) {
      const v = valOf(r, k);
      if (typeof v === "number" && Number.isFinite(v)) {
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      }
    }
  }
  const padSpan = Math.max(3, (maxV - minV) * 0.12);
  const yLo = Math.floor((minV - padSpan) / 5) * 5;
  const yHi = Math.ceil((maxV + padSpan) / 5) * 5;

  const width = 820;
  const height = 286;
  const padL = 56;
  const padR = 26;
  const padT = 22;
  const padB = 34;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const xOf = (idx) => padL + (rows.length <= 1 ? plotW / 2 : (idx / (rows.length - 1)) * plotW);
  const yOf = (v) => padT + ((yHi - v) / Math.max(1e-6, yHi - yLo)) * plotH;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "nav-svg",
    role: "img",
    "aria-label": "2015至2024换起点年化收益率对比曲线",
  });

  // 标注 2023~2024 样本外起点区间
  const oosIdx = rows.findIndex((r) => r.start_year >= 2023);
  if (oosIdx >= 0) {
    const colW = rows.length > 1 ? plotW / (rows.length - 1) : plotW;
    const bx = Math.max(padL, xOf(oosIdx) - colW * 0.42);
    const bw = width - padR - bx + 10;
    svg.append(
      svgEl("rect", {
        x: bx,
        y: padT,
        width: bw,
        height: plotH,
        rx: 4,
        fill: "#eff6ff",
      }),
    );
    const tag = svgEl("text", {
      x: bx + bw / 2,
      y: padT + 13,
      "text-anchor": "middle",
      fill: "#1d4ed8",
      "font-size": 10,
      "font-weight": 600,
    });
    tag.textContent = "样本外起点 (2023~2024)";
    svg.append(tag);
  }

  // 水平网格与 Y 轴刻度
  const tickCount = 5;
  for (let i = 0; i <= tickCount; i += 1) {
    const frac = i / tickCount;
    const val = yHi - frac * (yHi - yLo);
    const y = padT + frac * plotH;
    svg.append(
      svgEl("line", {
        x1: padL,
        y1: y,
        x2: width - padR,
        y2: y,
        stroke: "#e5e7eb",
        "stroke-width": 1,
        "stroke-dasharray": i === tickCount ? "" : "2 3",
      }),
    );
    const lbl = svgEl("text", {
      x: padL - 8,
      y: y + 3.5,
      "text-anchor": "end",
      fill: "#6b7280",
      "font-size": 10.5,
      class: "mono",
    });
    const rVal = Math.abs(val) < 0.05 ? 0 : val;
    lbl.textContent = `${rVal > 0 ? "+" : ""}${rVal.toFixed(0)}%`;
    svg.append(lbl);
  }

  // 0% 基准线
  if (yLo <= 0 && yHi >= 0) {
    const y0 = yOf(0);
    svg.append(
      svgEl("line", {
        x1: padL,
        y1: y0,
        x2: width - padR,
        y2: y0,
        stroke: "#94a3b8",
        "stroke-width": 1.1,
        "stroke-dasharray": "4 4",
      }),
    );
  }

  // 当前选中起步年的竖向高亮带
  if (activeStartYear !== null && activeStartYear !== undefined) {
    const aIdx = rows.findIndex((r) => r.start_year === Number(activeStartYear));
    if (aIdx >= 0) {
      const cx = xOf(aIdx);
      svg.append(
        svgEl("line", {
          x1: cx,
          y1: padT,
          x2: cx,
          y2: padT + plotH,
          stroke: "#2563eb",
          "stroke-width": 1.4,
          "stroke-dasharray": "3 2",
          opacity: 0.45,
        }),
      );
    }
  }

  // 折线与节点
  for (const k of keys) {
    const st = NAV_SERIES_STYLES[k] || { color: "#2563eb", width: 2, dash: "" };
    const pts = [];
    rows.forEach((r, idx) => {
      const v = valOf(r, k);
      if (typeof v === "number" && Number.isFinite(v)) {
        pts.push(`${xOf(idx).toFixed(1)},${yOf(v).toFixed(1)}`);
      }
    });
    if (pts.length >= 2) {
      const attrs = {
        points: pts.join(" "),
        fill: "none",
        stroke: st.color,
        "stroke-width": st.width,
        "stroke-linejoin": "round",
        "stroke-linecap": "round",
      };
      if (st.dash) attrs["stroke-dasharray"] = st.dash;
      svg.append(svgEl("polyline", attrs));
    }
    rows.forEach((r, idx) => {
      const v = valOf(r, k);
      if (typeof v !== "number" || !Number.isFinite(v)) return;
      const isActive = Number(activeStartYear) === r.start_year;
      svg.append(
        svgEl("circle", {
          cx: xOf(idx),
          cy: yOf(v),
          r: isActive ? 5 : k === "pool_nav" ? 3.2 : 4,
          fill: st.color,
          stroke: "#ffffff",
          "stroke-width": 1.5,
        }),
      );
    });
  }

  // X 轴年份刻度与悬停列
  const hoverColLine = svgEl("line", {
    x1: padL,
    y1: padT,
    x2: padL,
    y2: padT + plotH,
    stroke: "#475569",
    "stroke-width": 1,
    "stroke-dasharray": "3 3",
    opacity: 0,
  });
  svg.append(hoverColLine);

  const stepW = rows.length > 1 ? plotW / (rows.length - 1) : plotW;
  rows.forEach((r, idx) => {
    const cx = xOf(idx);
    const isActive = Number(activeStartYear) === r.start_year;
    const lbl = svgEl("text", {
      x: cx,
      y: height - 10,
      "text-anchor": "middle",
      fill: isActive ? "#1d4ed8" : "#475569",
      "font-size": 11,
      "font-weight": isActive ? 700 : 500,
      class: "mono",
    });
    lbl.textContent = `${r.start_year}`;
    svg.append(lbl);

    const hit = svgEl("rect", {
      x: cx - stepW / 2,
      y: padT,
      width: stepW,
      height: plotH + 22,
      fill: "transparent",
      style: "cursor: pointer;",
    });
    hit.addEventListener("mouseenter", () => {
      hoverColLine.setAttribute("x1", String(cx));
      hoverColLine.setAttribute("x2", String(cx));
      hoverColLine.setAttribute("opacity", "0.75");
      if (onHover) {
        onHover({
          row: r,
          items: keys.map((k) => ({
            id: k,
            label: labels[k] || k,
            cagr: valOf(r, k),
            color: (NAV_SERIES_STYLES[k] || {}).color || "#2563eb",
          })),
        });
      }
    });
    hit.addEventListener("click", () => {
      if (onSelectYear) onSelectYear(r.start_year);
    });
    svg.append(hit);
  });

  svg.addEventListener("mouseleave", () => {
    hoverColLine.setAttribute("opacity", "0");
    if (onHover) onHover(null);
  });

  container.innerHTML = "";
  container.append(svg);
}
