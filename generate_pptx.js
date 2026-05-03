"use strict";
const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Yi-Ting Li";
pres.title = "自監督對比學習方法教學庫";

// ── Color palette ──────────────────────────────────────────────
const C = {
  darkBg:    "021B2E",
  deepBlue:  "0A3557",
  midBlue:   "065A82",
  teal:      "0D9488",
  cyan:      "06B6D4",
  mint:      "02C39A",
  purple:    "7C3AED",
  amber:     "F59E0B",
  lightBg:   "EFF6FF",
  white:     "FFFFFF",
  textDark:  "1E293B",
  textMid:   "475569",
  textLight: "94A3B8",
  codeBg:    "0D1B2A",
  border:    "CBD5E1",
  border2:   "E2E8F0",
};

const makeShadow = () => ({
  type: "outer", blur: 8, offset: 2, angle: 135,
  color: "000000", opacity: 0.12,
});

// ================================================================
// SLIDE 1 — Cover
// ================================================================
{
  let s = pres.addSlide();
  s.background = { color: C.darkBg };

  // Top accent stripe
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.cyan }, line: { color: C.cyan } });

  // Left vertical accent bar
  s.addShape(pres.shapes.RECTANGLE, { x: 0.65, y: 1.45, w: 0.07, h: 2.5, fill: { color: C.mint }, line: { color: C.mint } });

  // Background decorative circles
  s.addShape(pres.shapes.OVAL, { x: 6.8, y: -1.0, w: 5.0, h: 5.0, fill: { color: C.midBlue, transparency: 72 }, line: { color: C.midBlue, transparency: 72 } });
  s.addShape(pres.shapes.OVAL, { x: 8.0, y: 2.8,  w: 3.5, h: 3.5, fill: { color: C.teal,    transparency: 78 }, line: { color: C.teal,    transparency: 78 } });

  // Main title
  s.addText("自監督對比學習\n方法教學庫", {
    x: 0.9, y: 1.35, w: 6.5, h: 2.2,
    fontSize: 44, fontFace: "Arial Black",
    color: C.white, bold: true, margin: 0,
  });

  // Subtitle
  s.addText("14 種方法完整實作　·　從 2018 到 2024", {
    x: 0.9, y: 3.7, w: 7.2, h: 0.5,
    fontSize: 17, fontFace: "Calibri",
    color: C.cyan, margin: 0,
  });

  // Tag pills
  const tags = ["PyTorch Lightning", "ResNet / ViT", "CIFAR-10 / ImageNet"];
  tags.forEach((tag, i) => {
    const tx = 0.9 + i * 2.55;
    s.addShape(pres.shapes.RECTANGLE, { x: tx, y: 4.42, w: 2.3, h: 0.36, fill: { color: C.deepBlue }, line: { color: C.teal, width: 1 } });
    s.addText(tag, { x: tx, y: 4.42, w: 2.3, h: 0.36, fontSize: 10, fontFace: "Calibri", color: C.cyan, align: "center", valign: "middle", margin: 0 });
  });

  // Bottom bar
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.4, w: 10, h: 0.225, fill: { color: C.deepBlue }, line: { color: C.deepBlue } });
  s.addText("ML Tutorial Series  ·  Contrastive Learning", {
    x: 0, y: 5.4, w: 10, h: 0.225, fontSize: 10, color: C.textLight, align: "center", valign: "middle", margin: 0,
  });
}

// ================================================================
// SLIDE 2 — 什麼是自監督對比學習
// ================================================================
{
  let s = pres.addSlide();
  s.background = { color: C.lightBg };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.65, fill: { color: C.deepBlue }, line: { color: C.deepBlue } });
  s.addText("什麼是自監督對比學習？", {
    x: 0.4, y: 0, w: 9.2, h: 0.65, fontSize: 22, fontFace: "Arial Black",
    color: C.white, bold: true, valign: "middle", margin: 0,
  });

  // Three concept cards
  const cards = [
    { icon: "🔒", title: "不需要標籤", body: "從大量未標記資料中學習視覺表示，大幅降低人工標注成本", color: C.midBlue },
    { icon: "📐", title: "對比目標",   body: "拉近同一圖像的不同增強視角，同時推開不同圖像的特徵", color: C.teal },
    { icon: "🚀", title: "強大表示",   body: "預訓練特徵可遷移至下游分類、物件偵測等任務", color: C.cyan },
  ];
  cards.forEach((c, i) => {
    const x = 0.35 + i * 3.15;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 0.85, w: 2.9, h: 3.45, fill: { color: C.white }, line: { color: C.border2 }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y: 0.85, w: 2.9, h: 0.12, fill: { color: c.color }, line: { color: c.color } });
    s.addShape(pres.shapes.OVAL, { x: x + 1.05, y: 1.1, w: 0.8, h: 0.8, fill: { color: c.color, transparency: 15 }, line: { color: c.color } });
    s.addText(c.icon, { x: x + 1.05, y: 1.1, w: 0.8, h: 0.8, fontSize: 24, align: "center", valign: "middle", margin: 0 });
    s.addText(c.title, { x: x + 0.1, y: 2.05, w: 2.7, h: 0.44, fontSize: 16, fontFace: "Arial Black", color: C.textDark, bold: true, align: "center", margin: 0 });
    s.addText(c.body,  { x: x + 0.15, y: 2.6, w: 2.6, h: 1.45, fontSize: 12, fontFace: "Calibri", color: C.textMid, align: "center", margin: 0 });
  });

  // Core idea banner
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 4.5, w: 9.3, h: 0.78, fill: { color: C.deepBlue }, line: { color: C.deepBlue }, shadow: makeShadow() });
  s.addText([
    { text: "核心思想：", options: { bold: true, color: C.cyan } },
    { text: "最大化同一樣本不同視角之間的互信息（Mutual Information），同時最小化不同樣本間的相似度", options: { color: C.white } },
  ], { x: 0.35, y: 4.5, w: 9.3, h: 0.78, fontSize: 13, fontFace: "Calibri", valign: "middle", align: "center", margin: 0 });
}

// ================================================================
// SLIDE 3 — 專案架構
// ================================================================
{
  let s = pres.addSlide();
  s.background = { color: C.lightBg };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.65, fill: { color: C.deepBlue }, line: { color: C.deepBlue } });
  s.addText("專案架構", {
    x: 0.4, y: 0, w: 9.2, h: 0.65, fontSize: 22, fontFace: "Arial Black",
    color: C.white, bold: true, valign: "middle", margin: 0,
  });

  // Directory tree panel
  s.addShape(pres.shapes.RECTANGLE, { x: 0.3, y: 0.85, w: 4.05, h: 4.4, fill: { color: C.codeBg }, line: { color: C.teal, width: 1 }, shadow: makeShadow() });

  const dirLines = [
    { t: "ml_topic_contrastive_learning/", col: C.cyan,  bold: true },
    { t: "├── core/       # 共用基礎設施",  col: C.mint, bold: false },
    { t: "├── methods/    # 14 種 SSL 方法", col: C.mint, bold: false },
    { t: "├── configs/    # 17 個 YAML 設定",col: C.mint, bold: false },
    { t: "├── eval/       # 6 種評估工具",  col: C.mint, bold: false },
    { t: "├── tests/      # 326+ 測試案例", col: C.mint, bold: false },
    { t: "└── train.py    # 統一訓練入口",  col: C.cyan, bold: true  },
  ];
  dirLines.forEach((l, i) => {
    s.addText(l.t, { x: 0.45, y: 1.07 + i * 0.52, w: 3.8, h: 0.44, fontSize: 11, fontFace: "Consolas", color: l.col, bold: l.bold, margin: 0 });
  });

  // Right description cards
  const folders = [
    { name: "core/",    desc: "BaseSSLModule、Config schema、EMA、Projection heads、Losses、Data module", color: C.midBlue },
    { name: "methods/", desc: "每個方法為獨立 sub-package，繼承 BaseSSLModule，只需實作 loss function", color: C.teal },
    { name: "configs/", desc: "YAML + Pydantic v2 嚴格驗證，extra='forbid' 防止 typo", color: C.cyan },
    { name: "eval/",    desc: "kNN / Linear Probe / t-SNE / UMAP / Fine-tuning / CAM 六種評估工具", color: C.mint },
  ];
  folders.forEach((f, i) => {
    const y = 0.85 + i * 1.1;
    s.addShape(pres.shapes.RECTANGLE, { x: 4.6, y, w: 5.1, h: 0.97, fill: { color: C.white }, line: { color: C.border2 }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x: 4.6, y, w: 0.08, h: 0.97, fill: { color: f.color }, line: { color: f.color } });
    s.addText(f.name, { x: 4.8, y: y + 0.08, w: 4.8, h: 0.32, fontSize: 13, fontFace: "Consolas", color: f.color, bold: true, margin: 0 });
    s.addText(f.desc, { x: 4.8, y: y + 0.44, w: 4.8, h: 0.45, fontSize: 10.5, fontFace: "Calibri", color: C.textMid, margin: 0 });
  });
}

// ================================================================
// SLIDE 4 — 四個時代演進
// ================================================================
{
  let s = pres.addSlide();
  s.background = { color: C.lightBg };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.65, fill: { color: C.deepBlue }, line: { color: C.deepBlue } });
  s.addText("四個時代的方法演進", {
    x: 0.4, y: 0, w: 9.2, h: 0.65, fontSize: 22, fontFace: "Arial Black",
    color: C.white, bold: true, valign: "middle", margin: 0,
  });

  const eras = [
    { era: "Era 1", name: "Proxy Tasks",          period: "2018–2019", methods: ["Instance Discrimination", "Invariant Spread"],           key: "Memory Bank；每圖自成一類",          color: C.purple },
    { era: "Era 2", name: "In-Batch Contrastive", period: "2020",      methods: ["MoCo v1/v2", "SimCLR v1/v2", "SwAV", "InfoMin"],         key: "Momentum Encoder；強資料增強；NT-Xent",color: C.midBlue },
    { era: "Era 3", name: "No-Negative",          period: "2020–2021", methods: ["BYOL", "SimSiam", "Barlow Twins"],                        key: "無需負樣本；Predictor 不對稱",        color: C.teal },
    { era: "Era 4", name: "Transformer-Based",    period: "2021–2024", methods: ["MoCo v3", "DINO", "DINOv2"],                             key: "ViT 骨幹；Centering + Sharpening",    color: C.cyan },
  ];

  eras.forEach((e, i) => {
    const x = 0.22 + i * 2.42;
    const cardH = 4.25;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 0.85, w: 2.28, h: cardH, fill: { color: C.white }, line: { color: e.color, width: 1.5 }, shadow: makeShadow() });
    // Header badge
    s.addShape(pres.shapes.RECTANGLE, { x, y: 0.85, w: 2.28, h: 0.62, fill: { color: e.color }, line: { color: e.color } });
    s.addText(e.era,    { x, y: 0.85, w: 2.28, h: 0.34, fontSize: 14, fontFace: "Arial Black", color: C.white, bold: true, align: "center", margin: 0 });
    s.addText(e.period, { x, y: 1.19, w: 2.28, h: 0.27, fontSize: 10, fontFace: "Calibri",     color: C.white, align: "center", margin: 0 });
    // Era name
    s.addText(e.name, { x: x + 0.1, y: 1.57, w: 2.1, h: 0.52, fontSize: 12, fontFace: "Arial Black", color: e.color, bold: true, align: "center", margin: 0 });
    // Method list
    e.methods.forEach((m, j) => {
      s.addShape(pres.shapes.OVAL, { x: x + 0.16, y: 2.21 + j * 0.48, w: 0.12, h: 0.12, fill: { color: e.color }, line: { color: e.color } });
      s.addText(m, { x: x + 0.35, y: 2.17 + j * 0.48, w: 1.87, h: 0.34, fontSize: 10.5, fontFace: "Calibri", color: C.textDark, margin: 0 });
    });
    // Key insight box
    const boxY = 0.85 + cardH - 0.75;
    s.addShape(pres.shapes.RECTANGLE, { x: x + 0.06, y: boxY, w: 2.16, h: 0.63, fill: { color: e.color, transparency: 85 }, line: { color: e.color, width: 0.5 } });
    s.addText(e.key, { x: x + 0.06, y: boxY, w: 2.16, h: 0.63, fontSize: 9, fontFace: "Calibri", color: C.textDark, align: "center", valign: "middle", margin: 0 });
  });
}

// ================================================================
// SLIDE 5 — 14 種方法總覽
// ================================================================
{
  let s = pres.addSlide();
  s.background = { color: C.lightBg };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.65, fill: { color: C.deepBlue }, line: { color: C.deepBlue } });
  s.addText("14 種方法總覽", {
    x: 0.4, y: 0, w: 9.2, h: 0.65, fontSize: 22, fontFace: "Arial Black",
    color: C.white, bold: true, valign: "middle", margin: 0,
  });

  const methods = [
    ["Instance Discrimination", "2018", "Era 1", "Non-param Memory Bank，每張圖像自成一類"],
    ["Invariant Spread",        "2019", "Era 1", "In-batch softmax 對比，SimCLR 直接前身"],
    ["MoCo v1",                 "2020", "Era 2", "Momentum encoder + FIFO queue 大量負樣本"],
    ["MoCo v2",                 "2020", "Era 2", "MoCo + MLP head + cosine LR 改進"],
    ["SimCLR v1",               "2020", "Era 2", "強資料增強 + symmetric NT-Xent loss"],
    ["SimCLR v2",               "2020", "Era 2", "更深 3-layer projection head"],
    ["SwAV",                    "2020", "Era 2", "Sinkhorn-Knopp online clustering + multi-crop"],
    ["InfoMin",                 "2020", "Era 2", "Minimal-MI 增強策略原則"],
    ["BYOL",                    "2020", "Era 3", "EMA + predictor 不對稱，無需任何負樣本"],
    ["SimSiam",                 "2021", "Era 3", "Stop-gradient 防塌陷，無 EMA"],
    ["Barlow Twins",            "2021", "Era 3", "交叉相關矩陣趨近單位矩陣（冗餘消除）"],
    ["MoCo v3",                 "2021", "Era 4", "MoCo for ViT；patch projection freeze 穩定訓練"],
    ["DINO",                    "2021", "Era 4", "Centering + sharpening student-teacher；無對比損失"],
    ["DINOv2",                  "2024", "Era 4", "大規模預訓練；教學庫提供 feature extraction demo"],
  ];

  const header = [
    { text: "方法",     options: { bold: true, color: C.white, fill: { color: C.deepBlue }, fontSize: 11, align: "center" } },
    { text: "年份",     options: { bold: true, color: C.white, fill: { color: C.deepBlue }, fontSize: 11, align: "center" } },
    { text: "時代",     options: { bold: true, color: C.white, fill: { color: C.deepBlue }, fontSize: 11, align: "center" } },
    { text: "主要貢獻", options: { bold: true, color: C.white, fill: { color: C.deepBlue }, fontSize: 11, align: "center" } },
  ];

  const rows = [header, ...methods.map((row, ri) => {
    const bg = ri % 2 === 0 ? "EFF6FF" : C.white;
    return row.map(cell => ({ text: cell, options: { fontSize: 10, color: C.textDark, fill: { color: bg } } }));
  })];

  s.addTable(rows, {
    x: 0.3, y: 0.75, w: 9.4,
    border: { pt: 0.5, color: C.border },
    colW: [2.1, 0.65, 0.75, 5.9],
    rowH: 0.305,
  });
}

// ================================================================
// SLIDE 6 — 共用訓練框架
// ================================================================
{
  let s = pres.addSlide();
  s.background = { color: C.lightBg };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.65, fill: { color: C.deepBlue }, line: { color: C.deepBlue } });
  s.addText("共用訓練框架", {
    x: 0.4, y: 0, w: 9.2, h: 0.65, fontSize: 22, fontFace: "Arial Black",
    color: C.white, bold: true, valign: "middle", margin: 0,
  });

  const stack = [
    { icon: "⚡", label: "PyTorch Lightning", desc: "BaseSSLModule — 統一 training loop、checkpoint 管理、混合精度", color: C.midBlue },
    { icon: "📋", label: "Pydantic v2 Config", desc: "YAML + extra='forbid' 嚴格驗證，防止 copy-paste typo", color: C.teal },
    { icon: "🦴", label: "timm Backbone",      desc: "ResNet-18/50、ViT-Small/Base 等任意 timm 模型名稱", color: C.cyan },
    { icon: "🔧", label: "Optimizer Suite",    desc: "AdamW / SGD / LARS + cosine warmup LR scheduler", color: C.purple },
  ];

  stack.forEach((item, i) => {
    const y = 0.85 + i * 1.13;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.3, y, w: 4.5, h: 0.98, fill: { color: C.white }, line: { color: C.border2 }, shadow: makeShadow() });
    s.addShape(pres.shapes.OVAL,      { x: 0.45, y: y + 0.19, w: 0.58, h: 0.58, fill: { color: item.color, transparency: 15 }, line: { color: item.color } });
    s.addText(item.icon, { x: 0.45, y: y + 0.19, w: 0.58, h: 0.58, fontSize: 20, align: "center", valign: "middle", margin: 0 });
    s.addText(item.label, { x: 1.15, y: y + 0.1,  w: 3.55, h: 0.35, fontSize: 14, fontFace: "Calibri", color: item.color, bold: true, margin: 0 });
    s.addText(item.desc,  { x: 1.15, y: y + 0.49, w: 3.55, h: 0.4,  fontSize: 11, fontFace: "Calibri", color: C.textMid, margin: 0 });
  });

  // YAML code panel
  s.addShape(pres.shapes.RECTANGLE, { x: 5.05, y: 0.85, w: 4.62, h: 4.52, fill: { color: C.codeBg }, line: { color: C.teal, width: 1 }, shadow: makeShadow() });
  s.addText("# configs/simclr_v1_resnet18.yaml", {
    x: 5.18, y: 0.96, w: 4.36, h: 0.3, fontSize: 9, fontFace: "Consolas", color: "64748B", margin: 0,
  });

  const yaml = [
    ["method:",       " simclr_v1"],
    ["backbone:",     " resnet18"],
    ["pretrained:",   " false"],
    ["max_epochs:",   " 200"],
    ["batch_size:",   " 256"],
    ["lr:",           " 3.0e-4"],
    ["optimizer:",    " adamw"],
    ["n_views:",      " 2"],
    ["simclr:",       ""],
    ["  temperature:"," 0.07"],
    ["  proj_dim:",   " 128"],
  ];
  yaml.forEach(([k, v], i) => {
    s.addText([
      { text: k, options: { color: C.cyan } },
      { text: v, options: { color: C.mint } },
    ], { x: 5.2, y: 1.35 + i * 0.36, w: 4.3, h: 0.3, fontSize: 11, fontFace: "Consolas", margin: 0 });
  });
}

// ================================================================
// SLIDE 7 — 6 種評估工具
// ================================================================
{
  let s = pres.addSlide();
  s.background = { color: C.lightBg };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.65, fill: { color: C.deepBlue }, line: { color: C.deepBlue } });
  s.addText("6 種評估工具", {
    x: 0.4, y: 0, w: 9.2, h: 0.65, fontSize: 22, fontFace: "Arial Black",
    color: C.white, bold: true, valign: "middle", margin: 0,
  });

  const tools = [
    { icon: "📊", name: "k-NN Evaluation",  cmd: "KNNCallback (內建)",              desc: "訓練期間每 N epoch 自動評估，無需額外指令", color: C.midBlue },
    { icon: "📈", name: "Linear Probe",      cmd: "eval/linear_probe.py",            desc: "凍結 backbone，訓練 linear head，weight_decay=0", color: C.teal },
    { icon: "🗺️", name: "t-SNE 視覺化",     cmd: "eval/tsne_vis.py",               desc: "Perplexity sweep: 10/30/50，特徵分布可視化", color: C.cyan },
    { icon: "🌐", name: "UMAP 視覺化",       cmd: "eval/umap_vis.py",               desc: "大樣本（>5K）推薦，比 t-SNE 更快速穩定", color: C.purple },
    { icon: "🎯", name: "Fine-tuning",       cmd: "eval/finetune.py",               desc: "backbone LR 1e-4 / head LR 1e-3，分組學習率", color: C.mint },
    { icon: "👁️",  name: "CAM 視覺化",       cmd: "eval/cam_vis.py",                desc: "SSL 用 EigenCAM，含分類器用 GradCAM", color: C.amber },
  ];

  tools.forEach((t, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = 0.28 + col * 3.18;
    const y = 0.82 + row * 2.38;

    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 3.04, h: 2.18, fill: { color: C.white }, line: { color: C.border2 }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 3.04, h: 0.1, fill: { color: t.color }, line: { color: t.color } });

    s.addShape(pres.shapes.OVAL, { x: x + 0.13, y: y + 0.2, w: 0.56, h: 0.56, fill: { color: t.color, transparency: 15 }, line: { color: t.color } });
    s.addText(t.icon, { x: x + 0.13, y: y + 0.2, w: 0.56, h: 0.56, fontSize: 18, align: "center", valign: "middle", margin: 0 });

    s.addText(t.name, { x: x + 0.8, y: y + 0.23, w: 2.12, h: 0.38, fontSize: 13, fontFace: "Calibri", color: C.textDark, bold: true, margin: 0 });

    s.addShape(pres.shapes.RECTANGLE, { x: x + 0.12, y: y + 0.87, w: 2.8, h: 0.3, fill: { color: C.codeBg }, line: { color: t.color, width: 0.5 } });
    s.addText(t.cmd, { x: x + 0.12, y: y + 0.87, w: 2.8, h: 0.3, fontSize: 9, fontFace: "Consolas", color: C.mint, align: "center", valign: "middle", margin: 0 });

    s.addText(t.desc, { x: x + 0.12, y: y + 1.25, w: 2.8, h: 0.76, fontSize: 10.5, fontFace: "Calibri", color: C.textMid, margin: 0 });
  });
}

// ================================================================
// SLIDE 8 — 快速開始
// ================================================================
{
  let s = pres.addSlide();
  s.background = { color: C.lightBg };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.65, fill: { color: C.deepBlue }, line: { color: C.deepBlue } });
  s.addText("快速開始", {
    x: 0.4, y: 0, w: 9.2, h: 0.65, fontSize: 22, fontFace: "Arial Black",
    color: C.white, bold: true, valign: "middle", margin: 0,
  });

  const steps = [
    { num: "1", title: "安裝依賴",          cmd: "pip install -r requirements.txt" },
    { num: "2", title: "準備 CIFAR-10 資料集", cmd: "python convert_cifar10.py  # 一次轉換為 ImageFolder 格式" },
    { num: "3", title: "訓練 SimCLR v1",    cmd: "python train.py --config configs/simclr_v1_resnet18.yaml --data-dir data/cifar10_imagefolder" },
    { num: "4", title: "Linear Probe 評估", cmd: "python eval/linear_probe.py configs/simclr_v1_resnet18.yaml --ckpt checkpoints/last.ckpt" },
    { num: "5", title: "UMAP 特徵可視化",   cmd: "python eval/umap_vis.py configs/simclr_v1_resnet18.yaml --ckpt checkpoints/last.ckpt" },
  ];

  steps.forEach((step, i) => {
    const y = 0.82 + i * 0.96;
    s.addShape(pres.shapes.OVAL, { x: 0.3, y: y + 0.12, w: 0.5, h: 0.5, fill: { color: C.teal }, line: { color: C.teal } });
    s.addText(step.num, { x: 0.3, y: y + 0.12, w: 0.5, h: 0.5, fontSize: 16, color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });
    s.addText(step.title, { x: 0.95, y: y + 0.1, w: 8.75, h: 0.3, fontSize: 13, fontFace: "Calibri", color: C.textDark, bold: true, margin: 0 });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.95, y: y + 0.43, w: 8.75, h: 0.38, fill: { color: C.codeBg }, line: { color: C.teal, width: 0.5 } });
    s.addText("$ " + step.cmd, { x: 0.95, y: y + 0.43, w: 8.75, h: 0.38, fontSize: 10, fontFace: "Consolas", color: C.mint, valign: "middle", margin: [0, 8, 0, 8] });
  });
}

// ================================================================
// SLIDE 9 — 主要特色
// ================================================================
{
  let s = pres.addSlide();
  s.background = { color: C.lightBg };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.65, fill: { color: C.deepBlue }, line: { color: C.deepBlue } });
  s.addText("主要特色", {
    x: 0.4, y: 0, w: 9.2, h: 0.65, fontSize: 22, fontFace: "Arial Black",
    color: C.white, bold: true, valign: "middle", margin: 0,
  });

  const features = [
    {
      icon: "🔬", title: "可重現性", sub: "Reproducibility First", color: C.midBlue,
      points: ["統一 config YAML — 每個實驗完整記錄", "Pydantic 驗證防止設定錯誤", "PyTorch Lightning checkpoint 自動管理", "固定隨機種子支援"],
    },
    {
      icon: "🧩", title: "模組化設計", sub: "Modular Architecture", color: C.teal,
      points: ["每個方法繼承 BaseSSLModule", "只需實作各自的 loss function", "共用 Projection heads / EMA / Losses", "新增一個方法 < 100 行程式碼"],
    },
    {
      icon: "⚖️", title: "橫向比較", sub: "Fair Comparison", color: C.cyan,
      points: ["14 種方法跑在完全相同的 pipeline", "相同 backbone 與增強策略", "6 種評估工具統一量測", "適合學術研究對比實驗"],
    },
  ];

  features.forEach((f, i) => {
    const x = 0.28 + i * 3.18;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 0.82, w: 3.04, h: 4.42, fill: { color: C.white }, line: { color: C.border2 }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y: 0.82, w: 3.04, h: 0.65, fill: { color: f.color }, line: { color: f.color } });
    s.addShape(pres.shapes.OVAL, { x: x + 1.1, y: 0.95, w: 0.52, h: 0.52, fill: { color: C.white, transparency: 20 }, line: { color: C.white } });
    s.addText(f.icon, { x: x + 1.1, y: 0.95, w: 0.52, h: 0.52, fontSize: 18, align: "center", valign: "middle", margin: 0 });
    s.addText(f.title, { x: x + 0.1, y: 1.58, w: 2.84, h: 0.42, fontSize: 17, fontFace: "Arial Black", color: f.color, bold: true, align: "center", margin: 0 });
    s.addText(f.sub,   { x: x + 0.1, y: 2.02, w: 2.84, h: 0.3,  fontSize: 10, fontFace: "Calibri", color: C.textLight, italic: true, align: "center", margin: 0 });
    f.points.forEach((pt, j) => {
      s.addShape(pres.shapes.OVAL, { x: x + 0.18, y: 2.46 + j * 0.5, w: 0.1, h: 0.1, fill: { color: f.color }, line: { color: f.color } });
      s.addText(pt, { x: x + 0.35, y: 2.42 + j * 0.5, w: 2.6, h: 0.38, fontSize: 11, fontFace: "Calibri", color: C.textMid, margin: 0 });
    });
  });
}

// ================================================================
// SLIDE 10 — 結尾
// ================================================================
{
  let s = pres.addSlide();
  s.background = { color: C.darkBg };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.cyan }, line: { color: C.cyan } });

  // Background circles
  s.addShape(pres.shapes.OVAL, { x: -0.8, y: 3.2, w: 4.5, h: 4.5, fill: { color: C.midBlue, transparency: 76 }, line: { color: C.midBlue, transparency: 76 } });
  s.addShape(pres.shapes.OVAL, { x: 6.5,  y: -1.2, w: 6.0, h: 6.0, fill: { color: C.teal,    transparency: 80 }, line: { color: C.teal,    transparency: 80 } });

  // Accent lines
  s.addShape(pres.shapes.RECTANGLE, { x: 3.2, y: 1.32, w: 3.6, h: 0.06, fill: { color: C.cyan }, line: { color: C.cyan } });
  s.addShape(pres.shapes.RECTANGLE, { x: 3.2, y: 3.92, w: 3.6, h: 0.06, fill: { color: C.cyan }, line: { color: C.cyan } });

  s.addText("開始探索", {
    x: 0.5, y: 1.48, w: 9, h: 0.9,
    fontSize: 48, fontFace: "Arial Black", color: C.white, bold: true, align: "center", margin: 0,
  });

  s.addText("14 種對比學習方法，等你來實驗", {
    x: 0.5, y: 2.48, w: 9, h: 0.5,
    fontSize: 18, fontFace: "Calibri", color: C.cyan, align: "center", margin: 0,
  });

  // Requirement pills
  const reqs = ["Python 3.10+", "PyTorch 2.0+", "CPU / CUDA 皆支援", "326+ 測試案例"];
  reqs.forEach((r, i) => {
    const rx = 0.8 + i * 2.15;
    s.addShape(pres.shapes.RECTANGLE, { x: rx, y: 3.18, w: 2.0, h: 0.4, fill: { color: C.deepBlue }, line: { color: C.teal, width: 1 } });
    s.addText(r, { x: rx, y: 3.18, w: 2.0, h: 0.4, fontSize: 10, fontFace: "Calibri", color: C.mint, align: "center", valign: "middle", margin: 0 });
  });

  s.addText("pip install -r requirements.txt", {
    x: 2.0, y: 4.08, w: 6, h: 0.5,
    fontSize: 16, fontFace: "Consolas", color: C.mint, align: "center", margin: 0,
  });

  s.addText("ML Tutorial Series  ·  Contrastive Learning", {
    x: 0, y: 5.1, w: 10, h: 0.35,
    fontSize: 11, fontFace: "Calibri", color: C.textLight, align: "center", margin: 0,
  });

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.4, w: 10, h: 0.225, fill: { color: C.deepBlue }, line: { color: C.deepBlue } });
}

// ── Write file ──────────────────────────────────────────────────
pres.writeFile({ fileName: "/Users/yi-tingli/Documents/Projects/ml_topic_contrastive_learning/contrastive_learning_tutorial.pptx" })
  .then(() => console.log("✅ PPT generated successfully!"))
  .catch(err => { console.error("❌ Error:", err); process.exit(1); });
