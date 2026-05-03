"use strict";
const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.layout = "LAYOUT_16x9";

const C = {
  bg:        "0D1117",
  panel:     "161B27",
  headerBg:  "1C2B4A",
  codeBg:    "0A1014",
  purple:    "8B5CF6",
  cyan:      "22D3EE",
  amber:     "FBBF24",
  green:     "34D399",
  pink:      "F472B6",
  midBlue:   "1C7293",
  white:     "F0F6FC",
  sub:       "8B949E",
  border:    "30363D",
  formula:   "BAE6FD",
};

const makeShadow = () => ({ type: "outer", blur: 10, offset: 3, angle: 135, color: "000000", opacity: 0.4 });

function addHeader(slide, title, subtitle) {
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.72, fill: { color: C.headerBg }, line: { color: C.headerBg } });
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.12, h: 0.72, fill: { color: C.purple }, line: { color: C.purple } });
  slide.addText(title, { x: 0.28, y: 0, w: 7.2, h: 0.72, fontSize: 20, fontFace: "Calibri", color: C.white, bold: true, valign: "middle", margin: 0 });
  if (subtitle) {
    slide.addText(subtitle, { x: 7.5, y: 0, w: 2.3, h: 0.72, fontSize: 9.5, fontFace: "Calibri", color: C.sub, align: "right", valign: "middle", margin: 0 });
  }
}

// Two-panel layout helper: left context, right formula box
function makeAlgoSlide(title, subtitle, leftItems, rightItems, insightText, insightColor) {
  let s = pres.addSlide();
  s.background = { color: C.bg };
  addHeader(s, title, subtitle);

  // Left panel
  s.addShape(pres.shapes.RECTANGLE, { x: 0.22, y: 0.84, w: 4.28, h: 4.55, fill: { color: C.panel }, line: { color: C.border }, shadow: makeShadow() });

  let ly = 0.96;
  for (const item of leftItems) {
    if (item.type === "heading") {
      s.addText(item.text, { x: 0.32, y: ly, w: 4.08, h: 0.3, fontSize: 11.5, fontFace: "Calibri", color: item.color || C.cyan, bold: true, margin: 0 });
      ly += 0.32;
    } else if (item.type === "bullet") {
      s.addShape(pres.shapes.OVAL, { x: 0.34, y: ly + 0.1, w: 0.08, h: 0.08, fill: { color: item.color || C.purple }, line: { color: item.color || C.purple } });
      s.addText(item.text, { x: 0.48, y: ly, w: 3.92, h: item.h || 0.3, fontSize: 10.5, fontFace: "Calibri", color: C.white, margin: 0 });
      ly += item.h || 0.3;
    } else if (item.type === "code") {
      s.addShape(pres.shapes.RECTANGLE, { x: 0.32, y: ly, w: 4.08, h: item.h || 0.3, fill: { color: C.codeBg }, line: { color: C.border, width: 0.5 } });
      s.addText(item.text, { x: 0.36, y: ly + 0.01, w: 4.0, h: item.h || 0.3, fontSize: 9.5, fontFace: "Consolas", color: C.formula, margin: 0 });
      ly += (item.h || 0.3) + 0.04;
    } else if (item.type === "tag") {
      s.addShape(pres.shapes.RECTANGLE, { x: 0.32, y: ly, w: 1.9, h: 0.26, fill: { color: item.color || C.purple, transparency: 80 }, line: { color: item.color || C.purple, width: 0.5 } });
      s.addText(item.text, { x: 0.32, y: ly, w: 1.9, h: 0.26, fontSize: 9, color: item.color || C.purple, align: "center", valign: "middle", margin: 0 });
      ly += 0.3;
    } else if (item.type === "gap") {
      ly += item.h || 0.1;
    }
  }

  // Right formula panel
  s.addShape(pres.shapes.RECTANGLE, { x: 4.72, y: 0.84, w: 5.07, h: 4.55, fill: { color: C.codeBg }, line: { color: C.purple, width: 1 }, shadow: makeShadow() });

  let ry = 0.97;
  for (const item of rightItems) {
    if (item.type === "heading") {
      s.addText(item.text, { x: 4.84, y: ry, w: 4.82, h: 0.3, fontSize: 11, fontFace: "Calibri", color: item.color || C.purple, bold: true, margin: 0 });
      ry += 0.32;
    } else if (item.type === "formula") {
      s.addText(item.text, { x: 4.84, y: ry, w: 4.82, h: item.h || 0.3, fontSize: item.size || 11, fontFace: "Consolas", color: item.color || C.formula, margin: 0 });
      ry += (item.h || 0.3) + 0.03;
    } else if (item.type === "comment") {
      s.addText(item.text, { x: 4.84, y: ry, w: 4.82, h: item.h || 0.27, fontSize: 9.5, fontFace: "Calibri", color: C.sub, italic: true, margin: 0 });
      ry += (item.h || 0.27) + 0.03;
    } else if (item.type === "highlight") {
      s.addShape(pres.shapes.RECTANGLE, { x: 4.8, y: ry, w: 4.86, h: item.h || 0.36, fill: { color: item.color || C.purple, transparency: 82 }, line: { color: item.color || C.purple, width: 0.5 } });
      s.addText(item.text, { x: 4.84, y: ry + 0.01, w: 4.82, h: item.h || 0.36, fontSize: item.size || 11, fontFace: "Consolas", color: C.formula, margin: 0 });
      ry += (item.h || 0.36) + 0.06;
    } else if (item.type === "divider") {
      s.addShape(pres.shapes.LINE, { x: 4.8, y: ry, w: 4.86, h: 0, line: { color: C.border, width: 0.5 } });
      ry += 0.12;
    } else if (item.type === "gap") {
      ry += item.h || 0.1;
    }
  }

  // Insight footer
  if (insightText) {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.22, y: 5.38, w: 9.57, h: 0.22, fill: { color: insightColor || C.purple, transparency: 85 }, line: { color: insightColor || C.purple, width: 0.5 } });
    s.addText([
      { text: "⚡ ", options: { bold: true, color: insightColor || C.amber } },
      { text: insightText, options: { color: C.white } },
    ], { x: 0.32, y: 5.38, w: 9.38, h: 0.22, fontSize: 9.5, fontFace: "Calibri", valign: "middle", margin: 0 });
  }
}

// ================================================================
// SLIDE 1 — Cover
// ================================================================
{
  let s = pres.addSlide();
  s.background = { color: C.bg };

  for (let i = 1; i < 10; i++) {
    s.addShape(pres.shapes.LINE, { x: i, y: 0, w: 0, h: 5.625, line: { color: C.border, width: 0.3, transparency: 60 } });
  }

  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 0.9, w: 8.4, h: 3.7, fill: { color: C.panel }, line: { color: C.purple, width: 1 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 0.9, w: 0.12, h: 3.7, fill: { color: C.purple }, line: { color: C.purple } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 0.9, w: 8.4, h: 0.1, fill: { color: C.purple }, line: { color: C.purple } });

  s.addText("自監督對比學習", { x: 1.1, y: 1.12, w: 7.8, h: 0.68, fontSize: 38, fontFace: "Arial Black", color: C.white, bold: true, margin: 0 });
  s.addText("演算法數學原理深度剖析", { x: 1.1, y: 1.85, w: 7.8, h: 0.55, fontSize: 26, fontFace: "Arial Black", color: C.purple, bold: true, margin: 0 });

  s.addText("ℒ_NCE = -log  exp(zᵢ·zⱼ/τ) / Σₖ exp(zᵢ·zₖ/τ)      [ InfoNCE 通用形式 ]", {
    x: 1.1, y: 2.56, w: 7.8, h: 0.38, fontSize: 13, fontFace: "Consolas", color: C.formula, margin: 0,
  });

  const tags = [
    { t: "14 Algorithms", c: C.purple },
    { t: "Expert Level", c: C.cyan },
    { t: "Mathematical Focus", c: C.amber },
    { t: "2018 – 2024", c: C.green },
  ];
  tags.forEach((tag, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 1.1 + i * 1.88, y: 3.1, w: 1.7, h: 0.32, fill: { color: tag.c, transparency: 82 }, line: { color: tag.c, width: 1 } });
    s.addText(tag.t, { x: 1.1 + i * 1.88, y: 3.1, w: 1.7, h: 0.32, fontSize: 9.5, color: tag.c, align: "center", valign: "middle", margin: 0 });
  });

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.0, w: 10, h: 0.625, fill: { color: C.headerBg }, line: { color: C.headerBg } });
  const eras = [
    { t: "Era 1: Proxy Tasks (2018-19)", c: C.purple },
    { t: "Era 2: In-Batch Contrastive (2020)", c: C.cyan },
    { t: "Era 3: No-Negative (2020-21)", c: C.green },
    { t: "Era 4: Transformers (2021-24)", c: C.amber },
  ];
  eras.forEach((e, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: i * 2.5, y: 5.0, w: 2.5, h: 0.625, fill: { color: e.c, transparency: 88 }, line: { color: e.c, width: 0.5 } });
    s.addText(e.t, { x: i * 2.5, y: 5.0, w: 2.5, h: 0.625, fontSize: 9, fontFace: "Calibri", color: e.c, align: "center", valign: "middle", margin: 0 });
  });
}

// ================================================================
// SLIDE 2 — Notation & Framework
// ================================================================
{
  let s = pres.addSlide();
  s.background = { color: C.bg };
  addHeader(s, "共用數學框架與符號定義", "Common Mathematical Framework");

  // Left: notation
  s.addShape(pres.shapes.RECTANGLE, { x: 0.22, y: 0.84, w: 4.55, h: 4.55, fill: { color: C.panel }, line: { color: C.border }, shadow: makeShadow() });
  s.addText("符號定義", { x: 0.32, y: 0.92, w: 4.3, h: 0.3, fontSize: 12, fontFace: "Calibri", color: C.cyan, bold: true, margin: 0 });

  const notations = [
    ["x",                "原始圖像"],
    ["xᵢ, xⱼ",          "同一圖像的兩個增強視角"],
    ["f_θ",             "Backbone Encoder（參數 θ）"],
    ["g_φ",             "Projection Head（參數 φ）"],
    ["q_ψ",             "Predictor MLP（參數 ψ，部分方法）"],
    ["hᵢ = f_θ(xᵢ)",   "Backbone 特徵（下游評估用）"],
    ["zᵢ = g_φ(hᵢ)",   "Projection 特徵（Loss 計算用）"],
    ["z̃ᵢ = zᵢ/‖zᵢ‖",  "L2 正規化 projection"],
    ["τ",               "Temperature（控制分布尖銳度）"],
    ["m",               "EMA Momentum 係數"],
    ["K",               "Queue 大小 / Prototype 數量"],
    ["sg(·)",           "Stop-gradient 算子"],
  ];
  notations.forEach(([sym, desc], i) => {
    const y = 1.27 + i * 0.33;
    s.addText(sym,  { x: 0.34, y, w: 1.9, h: 0.28, fontSize: 10.5, fontFace: "Consolas", color: C.formula, margin: 0 });
    s.addText(desc, { x: 2.3,  y, w: 2.35, h: 0.28, fontSize: 10, fontFace: "Calibri", color: C.white, margin: 0 });
  });

  // Right top: problem setup
  s.addShape(pres.shapes.RECTANGLE, { x: 4.98, y: 0.84, w: 4.82, h: 1.1, fill: { color: C.panel }, line: { color: C.border }, shadow: makeShadow() });
  s.addText("問題設定", { x: 5.08, y: 0.9, w: 4.6, h: 0.3, fontSize: 12, fontFace: "Calibri", color: C.cyan, bold: true, margin: 0 });
  s.addText("給定未標記資料集 𝒳，學習 Encoder f_θ 使其特徵可遷移至下游任務（無需標籤）", {
    x: 5.08, y: 1.22, w: 4.6, h: 0.55, fontSize: 10.5, fontFace: "Calibri", color: C.white, margin: 0,
  });

  // Right mid: pipeline
  s.addShape(pres.shapes.RECTANGLE, { x: 4.98, y: 2.04, w: 4.82, h: 0.55, fill: { color: C.panel }, line: { color: C.border } });
  s.addText("x  →  augment  →  xᵢ,xⱼ  →  f_θ  →  hᵢ,hⱼ  →  g_φ  →  zᵢ,zⱼ  →  ℒ", {
    x: 5.08, y: 2.04, w: 4.6, h: 0.55, fontSize: 10, fontFace: "Consolas", color: C.formula, valign: "middle", margin: 0,
  });

  // Right bottom: InfoNCE general form
  s.addShape(pres.shapes.RECTANGLE, { x: 4.98, y: 2.7, w: 4.82, h: 2.7, fill: { color: C.codeBg }, line: { color: C.purple, width: 1 }, shadow: makeShadow() });
  s.addText("InfoNCE 通用形式 (van den Oord 2018)", { x: 5.08, y: 2.78, w: 4.6, h: 0.28, fontSize: 10, fontFace: "Calibri", color: C.purple, margin: 0 });

  const flines = [
    "ℒ = -𝔼[log f(x,c) / Σ_{x'∈X} f(x',c)]",
    "",
    "f(x,c) = exp(sim(z,c) / τ)",
    "sim(u,v) = uᵀv / (‖u‖₂·‖v‖₂)   [cosine]",
    "",
    "Positive pair: (xᵢ,xⱼ) same image",
    "Negatives:  all other samples in X",
    "",
    "‖zᵢ‖₂ = 1  before loss (L2-normalize)",
  ];
  flines.forEach((l, i) => {
    if (!l) return;
    s.addText(l, { x: 5.08, y: 3.12 + i * 0.3, w: 4.6, h: 0.26, fontSize: 10.5, fontFace: "Consolas", color: C.formula, margin: 0 });
  });
}

// ================================================================
// SLIDES 3–13: Algorithm slides
// ================================================================

// SLIDE 3 — Instance Discrimination
makeAlgoSlide(
  "Instance Discrimination",
  "Wu et al., CVPR 2018  ·  Era 1",
  [
    { type: "tag", text: "Memory Bank NCE", color: C.purple },
    { type: "gap", h: 0.08 },
    { type: "heading", text: "架構" },
    { type: "bullet", text: "N 張圖各為一類（非參數 N-way 分類）" },
    { type: "bullet", text: "Memory Bank: 儲存 N 個 L2-norm 特徵向量" },
    { type: "bullet", text: "每步用當前 Encoder 輸出直接替換 Bank（無 EMA）" },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "NCE 正規化常數 Z" },
    { type: "bullet", text: "Z 估計自第一個 mini-batch，之後固定不動" },
    { type: "bullet", text: "重新計算 Z 會使訓練不穩定（重要 gotcha）" },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "Negative 抽樣" },
    { type: "bullet", text: "m ≈ 4096 個 negatives 從 Bank 隨機抽取" },
    { type: "bullet", text: "非 in-batch negatives（解耦 B 與 negative 數量）" },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "注意事項", color: C.amber },
    { type: "bullet", text: "n_views=1；Bank 特徵扮演第二視角" },
    { type: "bullet", text: "使用弱增強（非 SimCLR 強增強）" },
  ],
  [
    { type: "heading", text: "(m+1)-way NCE Loss" },
    { type: "formula", text: "P(i | zᵢ) = exp(zᵢ · vᵢ / τ) / Z" },
    { type: "comment", text: "vᵢ: Bank 中圖像 i 的儲存特徵；Z: 固定正規化常數" },
    { type: "gap" },
    { type: "highlight", text: "ℒ = -log P(i | zᵢ)", color: C.purple },
    { type: "gap" },
    { type: "divider" },
    { type: "heading", text: "Z 估計（首批執行，之後固定）" },
    { type: "formula", text: "Z̃ = mean(exp(logits)) × (m+1)" },
    { type: "comment", text: "logits = [pos_logit, neg_logit_1, ..., neg_logit_m]" },
    { type: "gap" },
    { type: "divider" },
    { type: "heading", text: "Bank 更新規則（非 EMA）" },
    { type: "formula", text: "Bank[idx] ← L2_norm(z_encoder)" },
    { type: "comment", text: "直接替換：特徵隨訓練進展而「過時」(stale)" },
    { type: "gap" },
    { type: "divider" },
    { type: "heading", text: "關鍵超參數" },
    { type: "highlight", text: "m=4096  |  τ=0.07  |  proj_dim=128", color: C.cyan },
  ],
  "首個以每張圖像為獨立類別的方法；固定 Z 是訓練穩定的關鍵工程細節",
  C.purple
);

// SLIDE 4 — Invariant Spread
makeAlgoSlide(
  "Invariant and Spreading Instance Feature",
  "Ye et al., CVPR 2019  ·  Era 1",
  [
    { type: "tag", text: "In-Batch InfoNCE", color: C.cyan },
    { type: "gap", h: 0.08 },
    { type: "heading", text: "架構（SimCLR 直接前身）" },
    { type: "bullet", text: "2 視角共用 Backbone + 2-layer Projector" },
    { type: "bullet", text: "Symmetric InfoNCE，所有 negatives 來自當前 batch" },
    { type: "bullet", text: "無 Memory Bank / Queue / EMA" },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "Batch Size 敏感性" },
    { type: "bullet", text: "有效 negatives = 2(B−1)" },
    { type: "bullet", text: "B < 256 → 表示品質顯著下降" },
    { type: "bullet", text: "SimCLR 用強增強 + LARS optimizer 解決此問題" },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "與 SimCLR v1 的差異", color: C.amber },
    { type: "bullet", text: "弱增強：color jitter s=0.4（SimCLR s=1.0）" },
    { type: "bullet", text: "無 Gaussian Blur" },
    { type: "bullet", text: "Loss 與 SimCLR 完全相同" },
  ],
  [
    { type: "heading", text: "Symmetric In-Batch InfoNCE" },
    { type: "formula", text: "Z = [z₁..zB, z₁'..zB'] ∈ ℝ²ᴮˣᴰ", h: 0.28 },
    { type: "formula", text: "Z̃ = L2_norm(Z, dim=1)" },
    { type: "formula", text: "S = Z̃ @ Z̃ᵀ / τ         ∈ ℝ²ᴮˣ²ᴮ" },
    { type: "formula", text: "S = S.fill_diagonal(-∞)", h: 0.28 },
    { type: "gap" },
    { type: "highlight", text: "ℒ = CrossEntropy(S, labels)", color: C.cyan },
    { type: "comment", text: "labels[i]=i+B, labels[i+B]=i（正樣本在另一半）" },
    { type: "gap" },
    { type: "divider" },
    { type: "heading", text: "展開的 per-sample loss" },
    { type: "formula", text: "ℓ(i,j) = -log exp(z̃ᵢ·z̃ⱼ/τ)", size: 10.5 },
    { type: "formula", text: "              / Σ_{k≠i}²ᴮ exp(z̃ᵢ·z̃ₖ/τ)", size: 10.5 },
    { type: "highlight", text: "ℒ = [ℓ(i,j) + ℓ(j,i)] / 2", color: C.purple },
    { type: "gap" },
    { type: "divider" },
    { type: "formula", text: "Negatives: 2(B-1)  |  τ=0.1  |  dim=128", size: 10 },
  ],
  "Invariant Spread 確立 In-Batch Symmetric InfoNCE 框架，SimCLR 僅升級增強策略與 optimizer",
  C.cyan
);

// SLIDE 5 — MoCo v1/v2
makeAlgoSlide(
  "MoCo v1 / v2",
  "He et al., CVPR 2020  ·  Chen et al., arXiv 2020  ·  Era 2",
  [
    { type: "tag", text: "Momentum Queue", color: C.midBlue },
    { type: "gap", h: 0.08 },
    { type: "heading", text: "架構" },
    { type: "bullet", text: "Online q = g_φ(f_θ(xᵢ))  [梯度流過]", h: 0.28 },
    { type: "bullet", text: "Momentum k = g_φ'(f_θ'(xⱼ))  [no_grad]", h: 0.28 },
    { type: "bullet", text: "FIFO Queue 存 K 個 key 作為 negatives", h: 0.28 },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "v1 vs v2 唯一差異" },
    { type: "code", text: "v1: projector = nn.Linear(feat, 128)", h: 0.28 },
    { type: "code", text: "v2: projector = MLP(feat→2048→128)", h: 0.28 },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "注意事項", color: C.amber },
    { type: "bullet", text: "m=0.9 比 m=0.999 差很多（queue key 一致性）", h: 0.28 },
    { type: "bullet", text: "Queue 更新必須在 Loss 計算之後", h: 0.28 },
    { type: "bullet", text: "Momentum encoder 不加入 optimizer", h: 0.28 },
  ],
  [
    { type: "heading", text: "Asymmetric Queue InfoNCE" },
    { type: "formula", text: "l_pos = (q · k) / τ              [B×1]" },
    { type: "formula", text: "l_neg = q @ queue / τ            [B×K]" },
    { type: "formula", text: "logits = cat([l_pos, l_neg], 1)  [B×(K+1)]", size: 10.5 },
    { type: "gap" },
    { type: "highlight", text: "ℒ = CrossEntropy(logits, label=0)", color: C.cyan },
    { type: "comment", text: "label=0: 第一個 logit 是正樣本" },
    { type: "gap" },
    { type: "divider" },
    { type: "heading", text: "EMA Momentum Update" },
    { type: "formula", text: "θ' ← m·θ' + (1-m)·θ   [Backbone]" },
    { type: "formula", text: "φ' ← m·φ' + (1-m)·φ   [Projector]" },
    { type: "comment", text: "m=0.999（v1/v2），在 on_train_batch_end 執行" },
    { type: "gap" },
    { type: "divider" },
    { type: "heading", text: "FIFO Queue 更新（Loss 計算之後）" },
    { type: "formula", text: "queue ← dequeue → enqueue(k)" },
    { type: "highlight", text: "K=65536  |  τ=0.07  |  m=0.999", color: C.purple },
  ],
  "Momentum Encoder 使 key 特徵跨 batch 一致，解耦 negative 數量與 batch size 的依賴",
  C.midBlue
);

// SLIDE 6 — SimCLR v1/v2
makeAlgoSlide(
  "SimCLR v1 / v2",
  "Chen et al., ICML 2020 / NeurIPS 2020  ·  Era 2",
  [
    { type: "tag", text: "Symmetric NT-Xent", color: C.green },
    { type: "gap", h: 0.08 },
    { type: "heading", text: "架構" },
    { type: "bullet", text: "強增強：color jitter s=1.0 + Gaussian blur" },
    { type: "bullet", text: "共用 Backbone + MLP Projector" },
    { type: "bullet", text: "Loss 用 z（projection），評估用 h（backbone）" },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "v1 vs v2 唯一差異" },
    { type: "code", text: "v1: ProjectionHead(feat→2048→128)    2-layer", h: 0.28 },
    { type: "code", text: "v2: ProjectionHead(feat→2048→2048→128) 3-layer", h: 0.28 },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "注意事項", color: C.amber },
    { type: "bullet", text: "color jitter s=1.0（非 torchvision 預設 0.4）" },
    { type: "bullet", text: "B < 256 → 表示品質急劇下降" },
    { type: "bullet", text: "下游評估用 h，不是 z（loss 用 z）" },
  ],
  [
    { type: "heading", text: "Symmetric NT-Xent（SimCLR 核心）" },
    { type: "formula", text: "Z̃ = L2_norm([z₁..zB, z₁'..zB'])", h: 0.28 },
    { type: "formula", text: "S = Z̃ @ Z̃ᵀ / τ   ∈ ℝ²ᴮˣ²ᴮ" },
    { type: "formula", text: "S.fill_diagonal_(-∞)         [mask self]" },
    { type: "gap" },
    { type: "highlight", text: "ℒ = CrossEntropy(S, labels)", color: C.green },
    { type: "comment", text: "等價於 NT-Xent；使用 F.cross_entropy 數值穩定" },
    { type: "gap" },
    { type: "divider" },
    { type: "heading", text: "NT-Xent 展開（per-sample）" },
    { type: "formula", text: "ℓ(i)= -log exp(s_ii'/τ) / Σ_{k≠i} exp(s_ik/τ)", size: 10.5 },
    { type: "comment", text: "i' = i+B（正樣本索引）；分母有 2B-1 項" },
    { type: "gap" },
    { type: "divider" },
    { type: "heading", text: "關鍵超參數" },
    { type: "formula", text: "Negatives: 2(B-1)  [in-batch only]" },
    { type: "highlight", text: "τ=0.07  |  B≥256  |  LARS optimizer", color: C.cyan },
  ],
  "強增強是效果的關鍵：v2 的 3-layer projector 提升表示品質但 Loss 公式與 v1 完全相同",
  C.green
);

// SLIDE 7 — SwAV
makeAlgoSlide(
  "SwAV",
  "Caron et al., NeurIPS 2020  ·  Era 2: Online Clustering",
  [
    { type: "tag", text: "Sinkhorn-Knopp OT", color: C.amber },
    { type: "gap", h: 0.08 },
    { type: "heading", text: "架構" },
    { type: "bullet", text: "Multi-crop: 2 大（224px）+ N 小（96px）" },
    { type: "bullet", text: "Backbone → 2-layer projector → L2 normalize" },
    { type: "bullet", text: "K 個可學習 Prototype 向量（K≈3000）" },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "Swapped Prediction 邏輯" },
    { type: "bullet", text: "視角 j 的特徵去預測視角 i 的 prototype 分配" },
    { type: "bullet", text: "只對大視角計算 Sinkhorn codes" },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "注意事項", color: C.amber },
    { type: "bullet", text: "前 freeze_epochs epoch 凍結 prototype（預設 1）" },
    { type: "bullet", text: "每步 optimizer 後 L2 normalize prototype 向量" },
    { type: "bullet", text: "ε=0.05 Sinkhorn regularization（不穩定降至 0.03）" },
  ],
  [
    { type: "heading", text: "Prototype Score & Soft Assignment" },
    { type: "formula", text: "s(z,cₖ) = z̃ᵀcₖ / τ      [原型相似度分數]", size: 10.5 },
    { type: "formula", text: "Q = SinkhornKnopp(S/ε)   [軟分配碼矩陣]" },
    { type: "comment", text: "Q 為雙隨機矩陣：所有 prototype 被均勻分配" },
    { type: "gap" },
    { type: "divider" },
    { type: "heading", text: "Sinkhorn-Knopp 最優傳輸" },
    { type: "formula", text: "Q* = argmin_Q <-S,Q> + ε·H(Q)" },
    { type: "formula", text: "s.t.  Q1_K = 1_B/B,  Qᵀ1_B = 1_K/K", size: 10.5 },
    { type: "comment", text: "迭代解法：交替行/列正規化 3 次即收斂" },
    { type: "gap" },
    { type: "divider" },
    { type: "heading", text: "Swapped Prediction Cross-Entropy Loss" },
    { type: "formula", text: "pᵢₖ = softmax(s(zᵢ,cₖ)/τ)" },
    { type: "formula", text: "ℒ(zᵢ,Qⱼ) = -Σₖ qⱼₖ log pᵢₖ" },
    { type: "highlight", text: "ℒ = Σᵢⱼ [ℒ(zᵢ,Qⱼ)+ℒ(zⱼ,Qᵢ)] / 2", color: C.amber },
  ],
  "OT 確保所有 prototype 均勻使用，避免坍塌至少數幾個 prototype",
  C.amber
);

// SLIDE 8 — InfoMin
makeAlgoSlide(
  "InfoMin",
  "Tian et al., NeurIPS 2020  ·  Era 2: View Design",
  [
    { type: "tag", text: "Minimal-MI Views", color: C.pink },
    { type: "gap", h: 0.08 },
    { type: "heading", text: "核心原則" },
    { type: "bullet", text: "良好視角僅共享任務所需的最小 MI" },
    { type: "bullet", text: "多餘 MI（texture、color 偏差）是噪音" },
    { type: "bullet", text: "Loss 與 SimCLR 完全相同，差異只在增強" },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "InfoMin 增強策略（vs SimCLR）" },
    { type: "code", text: "SimCLR: jitter(s=1.0) + blur(p=0.5)", h: 0.28 },
    { type: "code", text: "InfoMin: jitter(s=1.5) + gray(p=0.4)", h: 0.28 },
    { type: "code", text: "        + NO Gaussian blur", h: 0.28 },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "注意事項", color: C.amber },
    { type: "bullet", text: "本庫實作為增強策略示範版本" },
    { type: "bullet", text: "原論文含半監督視角學習框架（未實作）" },
    { type: "bullet", text: "在自然圖像外的 domain 需重新設計增強策略" },
  ],
  [
    { type: "heading", text: "互信息理論基礎" },
    { type: "formula", text: "I(v₁; v₂) = H(v₁) - H(v₁|v₂)" },
    { type: "formula", text: "I(v₁; v₂) ≥ I(v₁; Y)   [Data Processing Ineq.]", size: 10.5 },
    { type: "comment", text: "Y: 下游任務標籤；視角需保留足夠 Y 的資訊" },
    { type: "gap" },
    { type: "divider" },
    { type: "heading", text: "InfoMin 最優視角條件" },
    { type: "formula", text: "v* = argmin I(v₁;v₂)" },
    { type: "formula", text: "      s.t. I(v;Y) ≥ I_min", h: 0.28 },
    { type: "comment", text: "在保留足夠語意信息下最小化視角間 MI" },
    { type: "gap" },
    { type: "divider" },
    { type: "heading", text: "Loss（與 SimCLR 完全相同）" },
    { type: "highlight", text: "ℒ_InfoMin = ℒ_NT-Xent(zᵢ, zⱼ)", color: C.pink },
    { type: "comment", text: "差異只在 build_augmentation() 的策略設計" },
    { type: "gap" },
    { type: "divider" },
    { type: "formula", text: "Negatives: 2(B-1)  |  τ=0.07  |  dim=128" },
  ],
  "揭示增強策略的設計原則：最小 MI 視角使 encoder 學到更通用的語意特徵",
  C.pink
);

// SLIDE 9 — BYOL
makeAlgoSlide(
  "BYOL",
  "Grill et al., NeurIPS 2020  ·  Era 3: No Negatives",
  [
    { type: "tag", text: "EMA Bootstrap", color: C.green },
    { type: "gap", h: 0.08 },
    { type: "heading", text: "Online vs Target 架構" },
    { type: "bullet", text: "Online:  f_θ → g_φ → q_ψ（Predictor 獨有）" },
    { type: "bullet", text: "Target:  f_θ' → g_φ'（無 Predictor）" },
    { type: "bullet", text: "Target 以 cosine-scheduled EMA 更新" },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "防坍塌雙重機制" },
    { type: "bullet", text: "Predictor 不對稱 → 破壞梯度捷徑" },
    { type: "bullet", text: "EMA 慢更新 → Target 特徵穩定" },
    { type: "bullet", text: "缺一都會立即坍塌" },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "注意事項", color: C.amber },
    { type: "bullet", text: "EMA 動量必須 cosine schedule（0.996→1.0）" },
    { type: "bullet", text: "Target 無 Predictor（加了就坍塌）" },
    { type: "bullet", text: "Projector 輸出 L2-norm 後再送 Predictor" },
  ],
  [
    { type: "heading", text: "Symmetric Cosine Loss (Stop-gradient)" },
    { type: "formula", text: "z₁ = norm(g_φ(f_θ(v₁)))   [online proj]" },
    { type: "formula", text: "p₁ = q_ψ(z₁)              [online pred]" },
    { type: "formula", text: "t₂ = sg(norm(g_φ'(f_θ'(v₂)))) [target]", size: 10.5 },
    { type: "gap" },
    { type: "formula", text: "ℓ(p,t) = 2 - 2·cos_sim(p, t.detach())", size: 10.5 },
    { type: "highlight", text: "ℒ = [ℓ(p₁,t₂) + ℓ(p₂,t₁)] / 2", color: C.green },
    { type: "comment", text: "等價於 ‖p̃-t̃‖²₂；sg 使梯度只流經 online 端" },
    { type: "gap" },
    { type: "divider" },
    { type: "heading", text: "Cosine-Scheduled EMA" },
    { type: "formula", text: "m(t) = 1-(1-m₀)·(cos(πt/T)+1)/2" },
    { type: "formula", text: "θ' ← m(t)·θ' + (1-m(t))·θ" },
    { type: "comment", text: "m₀=0.996, m(T)=1.0；T=total training steps" },
    { type: "gap" },
    { type: "divider" },
    { type: "highlight", text: "proj: D→4096→256  |  pred: 256→4096→256", color: C.purple },
  ],
  "理論上等價於 EM 算法兩步交替：predictor 的 E-step，backbone 的 M-step",
  C.green
);

// SLIDE 10 — SimSiam
makeAlgoSlide(
  "SimSiam",
  "Chen & He, CVPR 2021  ·  Era 3: Stop-Gradient Only",
  [
    { type: "tag", text: "Stop-Gradient", color: C.cyan },
    { type: "gap", h: 0.08 },
    { type: "heading", text: "架構（最簡 No-Negative 方法）" },
    { type: "bullet", text: "兩視角共用完全相同的 Backbone + Projector" },
    { type: "bullet", text: "Predictor: bottleneck MLP 2048→512→2048" },
    { type: "bullet", text: "無 EMA、無 Queue、無 Negative、無 Batch Norm 依賴" },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "唯一防坍塌機制" },
    { type: "bullet", text: ".detach() on z（stop-gradient）是唯一機制" },
    { type: "bullet", text: "拿掉 .detach() → 2 epoch 坍塌至 loss=−1.0" },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "注意事項", color: C.amber },
    { type: "bullet", text: "projector 末層：BN 但無 ReLU（加 ReLU 變差）" },
    { type: "bullet", text: "使用 SGD，lr = 0.05 × B / 256" },
    { type: "bullet", text: "崩潰指標：z.std(dim=0).mean() → 0（應≈0.707）" },
  ],
  [
    { type: "heading", text: "SimSiam Loss（Stop-Gradient Cosine）" },
    { type: "formula", text: "z₁ = g_φ(f_θ(v₁))   z₂ = g_φ(f_θ(v₂))" },
    { type: "formula", text: "p₁ = q_ψ(z₁)         p₂ = q_ψ(z₂)" },
    { type: "gap" },
    { type: "highlight", text: "ℒ = -[cos(p₁,sg(z₂)) + cos(p₂,sg(z₁))]/2", color: C.cyan, size: 10.5 },
    { type: "comment", text: "sg=.detach()；梯度只流過 p 端（predictor）" },
    { type: "gap" },
    { type: "divider" },
    { type: "heading", text: "為何 Stop-Gradient 防坍塌？" },
    { type: "formula", text: "無 sg: ∂ℒ/∂θ → 恆等解（所有輸出相同）", size: 10.5 },
    { type: "formula", text: "有 sg: EM 交替優化（Chen & He, App.D）", size: 10.5 },
    { type: "comment", text: "等效於：固定目標 → 優化 predictor → 固定 pred → 優化 enc" },
    { type: "gap" },
    { type: "divider" },
    { type: "heading", text: "崩潰診斷" },
    { type: "formula", text: "std = z.std(dim=0).mean()" },
    { type: "highlight", text: "std → 0.707 (正常)  |  std → 0 (坍塌)", color: C.amber },
  ],
  "Stop-gradient 是 SimSiam 中唯一防坍塌的機制；移除後 loss 在 2 個 epoch 內降至 −1.0",
  C.cyan
);

// SLIDE 11 — Barlow Twins
makeAlgoSlide(
  "Barlow Twins",
  "Zbontar et al., ICML 2021  ·  Era 3: Redundancy Reduction",
  [
    { type: "tag", text: "Cross-Correlation → Identity", color: C.purple },
    { type: "gap", h: 0.08 },
    { type: "heading", text: "架構" },
    { type: "bullet", text: "共用 Backbone + 3-layer Projector（8192-dim）" },
    { type: "bullet", text: "高維 projection 是關鍵（128/256-dim 明顯劣）" },
    { type: "bullet", text: "無 EMA / Predictor / Negative / Queue" },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "Loss 設計直覺" },
    { type: "bullet", text: "對角 Cᵢᵢ → 1：不同視角的同維度應一致" },
    { type: "bullet", text: "非對角 Cᵢⱼ → 0：不同維度應去相關" },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "注意事項", color: C.amber },
    { type: "bullet", text: "C 必須除以 B（batch size）再計算 loss" },
    { type: "bullet", text: "λ 敏感：5e-3 是論文建議值" },
    { type: "bullet", text: "崩潰指標：diag(C).mean() < 0.5" },
  ],
  [
    { type: "heading", text: "Cross-Correlation Matrix" },
    { type: "formula", text: "zₐ = norm(g(f(v₁)))   ∈ ℝᴮˣᴰ" },
    { type: "formula", text: "z_b = norm(g(f(v₂)))   ∈ ℝᴮˣᴰ" },
    { type: "formula", text: "C = zₐᵀ @ z_b / B     ∈ ℝᴰˣᴰ" },
    { type: "comment", text: "C_ij: 視角 v₁ 的第 i 維與 v₂ 的第 j 維的相關係數" },
    { type: "gap" },
    { type: "divider" },
    { type: "heading", text: "Barlow Twins Loss" },
    { type: "formula", text: "ℒ_inv = Σᵢ (1 - Cᵢᵢ)²         [對角項]" },
    { type: "formula", text: "ℒ_red = Σᵢ Σ_{j≠i} Cᵢⱼ²      [非對角項]" },
    { type: "highlight", text: "ℒ = ℒ_inv + λ·ℒ_red    (λ=5e-3)", color: C.purple },
    { type: "comment", text: "C → I（單位矩陣）是 Loss 全域最小值" },
    { type: "gap" },
    { type: "divider" },
    { type: "heading", text: "高效 Off-Diagonal 提取" },
    { type: "formula", text: "off = C.flatten()[:-1].view(D-1,D+1)[:,1:]", size: 10.5 },
    { type: "highlight", text: "proj_dim=8192  |  λ=5e-3  |  BN on proj", color: C.cyan },
  ],
  "受 Barlow (1961) 神經冗餘消除啟發；Loss 本身即是防坍塌機制（無需 EMA/predictor）",
  C.purple
);

// SLIDE 12 — MoCo v3
makeAlgoSlide(
  "MoCo v3",
  "Chen, Xie & He, ICCV 2021  ·  Era 4: Transformer-Based",
  [
    { type: "tag", text: "ViT + Patch Freeze", color: C.cyan },
    { type: "gap", h: 0.08 },
    { type: "heading", text: "vs MoCo v2 的三大修改" },
    { type: "bullet", text: "1. 凍結 patch_embed.proj（ViT 穩定關鍵）" },
    { type: "bullet", text: "2. 捨棄 Queue，改用 in-batch symmetric loss" },
    { type: "bullet", text: "3. 使用 AdamW（ViT 需要 adaptive optimizer）" },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "架構" },
    { type: "bullet", text: "Online:  f_θ → g_φ → q_ψ（Predictor）" },
    { type: "bullet", text: "Momentum: f_θ' → g_φ'（無 Predictor）" },
    { type: "bullet", text: "Symmetric InfoNCE（兩方向各算一次）" },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "注意事項", color: C.amber },
    { type: "bullet", text: "m=0.99（比 v1/v2 的 0.999 更快 EMA 更新）" },
    { type: "bullet", text: "patch_embed 凍結前先 deepcopy → EMA 也凍結" },
    { type: "bullet", text: "gradient_clip_val=1.0 增加穩定性" },
  ],
  [
    { type: "heading", text: "Patch Embedding Freeze（最重要的工程細節）" },
    { type: "formula", text: "backbone.patch_embed.proj.weight", size: 10 },
    { type: "formula", text: "    .requires_grad_(False)  # freezed", size: 10 },
    { type: "comment", text: "防止 Conv2d patch 映射成為 trivial shortcut" },
    { type: "gap" },
    { type: "divider" },
    { type: "heading", text: "Symmetric In-Batch InfoNCE（無 Queue）" },
    { type: "formula", text: "q₁=q_ψ(g(f(v₁)))  k₂=sg(g'(f'(v₂)))" },
    { type: "formula", text: "ℒ_fwd = InfoNCE(q₁, k₂)" },
    { type: "formula", text: "ℒ_bwd = InfoNCE(q₂, k₁)" },
    { type: "highlight", text: "ℒ = (ℒ_fwd + ℒ_bwd) / 2", color: C.cyan },
    { type: "comment", text: "InfoNCE 用 [B×B] similarity matrix，無需 queue" },
    { type: "gap" },
    { type: "divider" },
    { type: "heading", text: "常數 EMA（無 cosine schedule）" },
    { type: "formula", text: "θ' ← m·θ' + (1-m)·θ  [m=0.99, constant]", size: 10.5 },
    { type: "highlight", text: "m=0.99  |  τ=0.2  |  proj: D→4096→256", color: C.purple },
  ],
  "凍結 patch_embed 是 ViT contrastive learning 中最重要的單一工程細節",
  C.cyan
);

// SLIDE 13 — DINO
makeAlgoSlide(
  "DINO / DINOv2",
  "Caron et al., ICCV 2021  ·  Oquab et al., TMLR 2024  ·  Era 4",
  [
    { type: "tag", text: "Centering + Sharpening", color: C.amber },
    { type: "gap", h: 0.08 },
    { type: "heading", text: "Student-Teacher 架構" },
    { type: "bullet", text: "Student: 所有 crops（2 大 + N 小）" },
    { type: "bullet", text: "Teacher（EMA）: 只處理大 crops" },
    { type: "bullet", text: "Prototype 層 65536-dim（維度很重要）" },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "防坍塌雙重機制" },
    { type: "bullet", text: "Centering: 減去 Teacher 輸出的動量均值" },
    { type: "bullet", text: "Sharpening: Teacher 用低溫 τ_t 尖銳化分布" },
    { type: "gap", h: 0.06 },
    { type: "heading", text: "注意事項", color: C.amber },
    { type: "bullet", text: "Center 更新必須在 Loss 之前（順序關鍵！）" },
    { type: "bullet", text: "Teacher 溫度 warmup: 0.07 → 0.04（高→低）" },
    { type: "bullet", text: "gradient_clip_val=3.0 穩定 ViT 訓練" },
  ],
  [
    { type: "heading", text: "Teacher Centering + Sharpening" },
    { type: "formula", text: "t_logits = Prototype_ema(norm(g_ema(f_ema(x))))", size: 9.5 },
    { type: "formula", text: "c ← λ·c + (1-λ)·mean(t_logits)  [動量更新]", size: 10 },
    { type: "formula", text: "t_probs = softmax((t_logits-c) / τ_t)", size: 10.5 },
    { type: "comment", text: "c 中心化防止所有維度飽和；τ_t 低溫使分布尖銳" },
    { type: "gap" },
    { type: "divider" },
    { type: "heading", text: "Cross-Entropy Loss（跳過同視角對）" },
    { type: "formula", text: "s_log_p = log_softmax(Prototype(norm(g(f(crop))))/τ_s)", size: 9.5 },
    { type: "formula", text: "ℒ(i,j) = -Σₖ t_probs_j[k]·s_log_p_i[k]" },
    { type: "highlight", text: "ℒ = Σ_{i≠j} ℒ(student_i, teacher_j) / N_pairs", color: C.amber, size: 10 },
    { type: "comment", text: "i≠j：跳過同視角對；N_pairs=(n_crops−2)×2" },
    { type: "gap" },
    { type: "divider" },
    { type: "formula", text: "n_prototypes=65536  |  τ_s=0.1  |  τ_t=0.04→0.07", size: 10 },
    { type: "comment", text: "DINOv2 加入 iBOT masked image modeling 及 LVD-142M 資料" },
  ],
  "DINOv2 在 LVD-142M 資料上訓練，本庫僅提供 feature extraction demo",
  C.amber
);

// ================================================================
// SLIDE 14 — Collapse Prevention Comparison
// ================================================================
{
  let s = pres.addSlide();
  s.background = { color: C.bg };
  addHeader(s, "坍塌防止機制比較分析", "Collapse Prevention · All 14 Methods");

  const rows_data = [
    ["方法", "防坍塌機制", "需 Negative", "EMA", "Predictor", "崩潰監控指標"],
    ["Instance Disc.", "NCE + Memory Bank", "✓ (m≈4096)", "✗", "✗", "loss ≫ -log(1/N)"],
    ["Invariant Spread", "In-batch negatives", "✓ (2B-2)", "✗", "✗", "loss 不收斂"],
    ["MoCo v1/v2", "Queue negatives + EMA", "✓ (K=65536)", "✓ m=0.999", "✗", "loss 不收斂"],
    ["SimCLR v1/v2", "In-batch negatives", "✓ (2B-2)", "✗", "✗", "loss 不收斂"],
    ["SwAV", "Sinkhorn OT 均勻分配", "✗ (prototypes)", "✗", "✗", "diag(C) → 0"],
    ["InfoMin", "In-batch negatives", "✓ (2B-2)", "✗", "✗", "loss 不收斂"],
    ["BYOL", "Predictor 非對稱 + EMA", "✗", "✓ cosine", "✓ standard", "embed_std → 0"],
    ["SimSiam", "Stop-gradient（唯一）", "✗", "✗", "✓ bottleneck", "std→0, loss→−1"],
    ["Barlow Twins", "Cross-corr → Identity", "✗", "✗", "✗", "diag(C) < 0.5"],
    ["MoCo v3", "In-batch + EMA + patch freeze", "✓ (2B-2)", "✓ m=0.99", "✓ standard", "loss spike"],
    ["DINO", "Centering + Sharpening + EMA", "✗", "✓ cosine", "✗", "teacher entropy ↑"],
  ];

  const tableRows = rows_data.map((row, ri) => {
    if (ri === 0) {
      return row.map(cell => ({ text: cell, options: { bold: true, color: C.white, fill: { color: C.headerBg }, fontSize: 9, align: "center" } }));
    }
    const bg = ri % 2 === 0 ? "161B27" : "1C2335";
    return row.map(cell => ({ text: cell, options: { fontSize: 8.5, color: C.white, fill: { color: bg } } }));
  });

  s.addTable(tableRows, {
    x: 0.2, y: 0.82, w: 9.6,
    border: { pt: 0.5, color: C.border },
    colW: [1.2, 2.3, 1.25, 0.95, 1.15, 1.95],
    rowH: 0.355,
  });

  s.addShape(pres.shapes.RECTANGLE, { x: 0.2, y: 5.22, w: 9.6, h: 0.37, fill: { color: C.purple, transparency: 85 }, line: { color: C.purple, width: 0.5 } });
  s.addText([
    { text: "理論統一視角: ", options: { bold: true, color: C.amber } },
    { text: "所有防坍塌策略均可詮釋為破壞 trivial constant solution 的梯度路徑 — 顯式排斥（負樣本）、隱式排斥（架構不對稱 + EMA）、結構約束（去相關 loss）三種範式", options: { color: C.white } },
  ], { x: 0.32, y: 5.22, w: 9.38, h: 0.37, fontSize: 9.5, fontFace: "Calibri", valign: "middle", margin: 0 });
}

// ================================================================
// SLIDE 15 — Method Comparison Table
// ================================================================
{
  let s = pres.addSlide();
  s.background = { color: C.bg };
  addHeader(s, "14 種方法關鍵超參數對照表", "Key Hyperparameters · Full Comparison");

  const methods = [
    ["Instance Disc.",   "1",  "NCE",        "0.07", "4096 (bank)", "2-layer", "128",  "✗",      "✗"],
    ["Invariant Spread", "2",  "InfoNCE",    "0.1",  "2(B-1)",     "2-layer", "128",  "✗",      "✗"],
    ["MoCo v1",          "2",  "InfoNCE",    "0.07", "65536",       "Linear",  "128",  "✓ 0.999","✗"],
    ["MoCo v2",          "2",  "InfoNCE",    "0.07", "65536",       "2-layer", "128",  "✓ 0.999","✗"],
    ["SimCLR v1",        "2",  "NT-Xent",    "0.07", "2(B-1)",     "2-layer", "128",  "✗",      "✗"],
    ["SimCLR v2",        "2",  "NT-Xent",    "0.07", "2(B-1)",     "3-layer", "128",  "✗",      "✗"],
    ["SwAV",             "8",  "CE (OT)",    "0.1",  "K≈3000",      "2-layer", "128",  "✗",      "✗"],
    ["InfoMin",          "2",  "NT-Xent",    "0.07", "2(B-1)",     "2-layer", "128",  "✗",      "✗"],
    ["BYOL",             "2",  "Cosine",     "—",    "0",           "2-layer", "256",  "✓ cos",  "✓ std"],
    ["SimSiam",          "2",  "Cosine",     "—",    "0",           "3-layer", "2048", "✗",      "✓ btlnk"],
    ["Barlow Twins",     "2",  "Cross-corr", "—",    "0",           "3-layer", "8192", "✗",      "✗"],
    ["MoCo v3",          "2",  "InfoNCE",    "0.2",  "2(B-1)",     "3-layer", "256",  "✓ 0.99", "✓ std"],
    ["DINO",             "8",  "CE (distil)","0.04", "0",           "3-layer", "65536","✓ cos",  "✗"],
    ["DINOv2",           "8",  "iBOT+DINO",  "0.04", "0",           "3-layer", "65536","✓ cos",  "✗"],
  ];

  const header = ["方法", "n_views", "Loss", "τ", "Negatives", "Projector", "dim", "EMA", "Predictor"].map(
    cell => ({ text: cell, options: { bold: true, color: C.white, fill: { color: C.headerBg }, fontSize: 9, align: "center" } })
  );

  const tableRows = [header, ...methods.map((row, ri) => {
    const bg = ri % 2 === 0 ? "161B27" : "1C2335";
    return row.map(cell => ({ text: cell, options: { fontSize: 8.5, color: C.white, fill: { color: bg } } }));
  })];

  s.addTable(tableRows, {
    x: 0.18, y: 0.82, w: 9.64,
    border: { pt: 0.5, color: C.border },
    colW: [1.3, 0.65, 0.95, 0.45, 1.0, 0.85, 0.7, 0.74, 0.9],
    rowH: 0.3,
  });

  s.addShape(pres.shapes.RECTANGLE, { x: 0.18, y: 5.13, w: 9.64, h: 0.46, fill: { color: C.amber, transparency: 88 }, line: { color: C.amber, width: 0.5 } });
  s.addText([
    { text: "趨勢觀察: ", options: { bold: true, color: C.amber } },
    { text: "Era 3/4 拋棄大量負樣本換取架構非對稱性；Projection dim 從 128 增至 65536；Projector 從 2-layer 升至 3-layer；τ 在 no-negative 方法中不再有效（改由架構設計控制分布）", options: { color: C.white } },
  ], { x: 0.3, y: 5.13, w: 9.38, h: 0.46, fontSize: 9.5, fontFace: "Calibri", valign: "middle", margin: 0 });
}

// ================================================================
// SLIDE 16 — Closing
// ================================================================
{
  let s = pres.addSlide();
  s.background = { color: C.bg };

  for (let i = 1; i < 10; i++) {
    s.addShape(pres.shapes.LINE, { x: i, y: 0, w: 0, h: 5.625, line: { color: C.border, width: 0.3, transparency: 65 } });
  }

  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 0.75, w: 8.4, h: 4.15, fill: { color: C.panel }, line: { color: C.purple, width: 1 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 0.75, w: 0.12, h: 4.15, fill: { color: C.purple }, line: { color: C.purple } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 0.75, w: 8.4, h: 0.1,  fill: { color: C.purple }, line: { color: C.purple } });

  s.addText("演算法全景回顧", { x: 1.1, y: 0.95, w: 7.8, h: 0.55, fontSize: 26, fontFace: "Arial Black", color: C.white, bold: true, margin: 0 });

  const takeaways = [
    { icon: "ℒ", c: C.formula, text: "所有方法均衍生自 InfoNCE:  ℒ = -log f(pos) / Σ f(neg)" },
    { icon: "τ", c: C.cyan,    text: "溫度 τ 控制分布尖銳度，直接決定訓練穩定性與表示品質" },
    { icon: "→", c: C.green,   text: "Era 3/4: 架構不對稱（Predictor + EMA）取代顯式負樣本" },
    { icon: "D", c: C.amber,   text: "Projection dim 演進: 128 → 256 → 2048 → 8192 → 65536" },
    { icon: "∇", c: C.pink,    text: "ViT 需要 patch freeze + AdamW + gradient clip；ResNet 習慣不可直接遷移" },
  ];

  takeaways.forEach((t, i) => {
    s.addShape(pres.shapes.OVAL, { x: 1.18, y: 1.65 + i * 0.46, w: 0.28, h: 0.28, fill: { color: C.purple, transparency: 65 }, line: { color: C.purple } });
    s.addText(t.icon, { x: 1.18, y: 1.65 + i * 0.46, w: 0.28, h: 0.28, fontSize: 11, fontFace: "Consolas", color: t.c, align: "center", valign: "middle", margin: 0 });
    s.addText(t.text, { x: 1.56, y: 1.65 + i * 0.46, w: 7.35, h: 0.35, fontSize: 11.5, fontFace: "Calibri", color: C.white, margin: 0 });
  });

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.0, w: 10, h: 0.625, fill: { color: C.headerBg }, line: { color: C.headerBg } });
  s.addText("Self-Supervised Contrastive Learning  ·  Mathematical Deep Dive  ·  Era 1–4  ·  2018–2024", {
    x: 0, y: 5.0, w: 10, h: 0.625, fontSize: 10.5, color: C.sub, align: "center", valign: "middle", margin: 0,
  });
}

// ── Write ──────────────────────────────────────────────────────
pres.writeFile({ fileName: "/Users/yi-tingli/Documents/Projects/ml_topic_contrastive_learning/contrastive_learning_algorithms.pptx" })
  .then(() => console.log("✅ Expert PPT generated!"))
  .catch(err => { console.error("❌", err); process.exit(1); });
