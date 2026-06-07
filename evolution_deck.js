// Contrastive Learning — Evolution History deck
// Four-act narrative: "the field spent 5 years removing its crutches"
// Run: node evolution_deck.js  ->  contrastive_learning_evolution.pptx
const pptxgen = require("pptxgenjs");
const fs = require("fs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
const PW = 13.33, PH = 7.5;
pres.author = "Yi-Ting Li";
pres.title = "對比學習的演化史";

// ---- palette ----
const NAVY = "0E1626";   // dark background
const NAVY2 = "182740";  // dark card
const NAVY3 = "23375C";  // dark card 2
const LIGHT = "F4F6FB";  // light background
const CARD = "FFFFFF";
const INK = "162039";    // dark text on light
const MUTE = "5C6B8A";   // muted text
const ICE = "C6D6F2";    // light muted text on dark
const WHITE = "FFFFFF";
const E1 = "F59E0B";     // Era 1 amber
const E2 = "06B6D4";     // Era 2 cyan
const E3 = "F43F5E";     // Era 3 rose (the heretical turn)
const E4 = "8B5CF6";     // Era 4 violet
const ACCENT = "22D3EE"; // bright cyan accent

const FACE = "PingFang TC";
const MONO = "Menlo";

const makeShadow = () => ({ type: "outer", color: "000000", blur: 9, offset: 3, angle: 135, opacity: 0.18 });
const softShadow = () => ({ type: "outer", color: "0E1626", blur: 7, offset: 2, angle: 135, opacity: 0.12 });

// ---------- speaker notes (concise talking-point bullets, one per slide) ----------
// Full verbatim script lives in docs/TALK_SCRIPT.md; these are quick glance bullets.
const B = (...pts) => pts.map((p) => "• " + p).join("\n");
const NOTES = {
  cover: B(
    "開場提問:給你一百萬張照片、一個標籤都沒有,模型怎麼自己分貓狗?",
    "主題是「對比學習」;接下來 20 幾分鐘講它 5 年的演化故事"),
  bigPicture: B(
    "一副眼鏡看全場:這領域 5 年只做一件事——不斷「拿掉拐杖」",
    "四步:2018 拿掉標籤 → 2020 拿掉 memory bank → 2021 拿掉負樣本 → 2021+ 換 Transformer",
    "貫穿口號:每個方法都問「它又拿掉哪根拐杖?」"),
  coreIntuition: B(
    "核心直覺超簡單:同圖兩個增強 = 拉近;不同圖 = 推開",
    "學到「不管怎麼變都不變的東西」= 語意",
    "陷阱:只拉近不推開 → 全塌成一個點 = 坍塌(全場核心反派)"),
  infonce: B(
    "把「拉近/推開」寫成數學 = InfoNCE",
    "看三個位置:分子=正樣本(要大)、分母=負樣本(要小)、τ=溫度",
    "記憶點:分子拉、分母推",
    "之後 14 個方法都在玩這條式子(變形或逃脫)"),
  evoMap: B(
    "一張地圖、四個時代,照著走",
    "箭頭 = 「誰解決誰的瓶頸」,不只是時間順序",
    "一條有因有果的演化鏈,不是零散論文"),
  act1: B(
    "回到 2018:一堆圖、沒標籤,模型要學什麼?",
    "標籤是最大那根拐杖,現在被抽掉",
    "第一幕:看前人怎麼「無中生有」造出任務"),
  instance_discrimination: B(
    "點子:每張圖自成一類(百萬張 = 百萬類)",
    "架構:一個編碼器 + memory bank(存全資料特徵當負樣本)",
    "loss:softmax 算不動 → 用 NCE 抽樣近似",
    "撞牆:bank 特徵會過時(下一步要解決)"),
  invariant_spread: B(
    "不要 bank,直接在 batch 內:同圖拉近(invariant)、異圖推開(spread)",
    "架構:兩條共享權重分支;loss:對稱 InfoNCE",
    "跟前一個差異:丟掉 memory bank,改用 in-batch 負樣本",
    "它是 SimCLR 的直系祖先"),
  act2: B(
    "第二幕 = 最精彩分岔:負樣本到底從哪來?",
    "兩派世紀對決:一派用 queue 存、一派把 batch 開到爆"),
  moco_v1: B(
    "何愷明團隊:用 FIFO queue 存歷史特徵當負樣本",
    "負樣本量與 batch size 脫鉤(小 batch 也有上萬個)",
    "關鍵招:動量編碼器(EMA 慢慢跟上)→ queue 特徵一致、不過時",
    "loss 還是 InfoNCE,只是負樣本來自 queue"),
  moco_v2: B(
    "loss/架構跟 v1 一樣,純工程升級",
    "從 SimCLR 借三樣:MLP 投影頭、高斯模糊、cosine LR",
    "結果:小資源逼近 TPU 大 batch 的 SimCLR"),
  simclr_v1: B(
    "Hinton 團隊,理念跟 MoCo 相反:不要 queue,batch 開超大",
    "架構:兩條共享分支 + MLP 投影頭;loss:NT-Xent(= InfoNCE 加 L2 正規化)",
    "關鍵:強增強(color jitter + 高斯模糊)才是靈魂",
    "痛點:要 TPU 等級的超大 batch"),
  simclr_v2: B(
    "同一條路加強版,loss 沒變、主軸是「規模」",
    "改動:投影頭加深到 3 層 + 更大 backbone",
    "證明:大型 SSL 模型是強半監督學習者(pretrain → 少量標籤微調 → 蒸餾)"),
  swav: B(
    "不再兩兩比較,改線上分群(prototypes 群心)",
    "loss:swapped prediction(交叉熵)——用一視角預測另一視角的分配",
    "Sinkhorn 強迫分配均勻攤平 → 本身就在防坍塌",
    "multi-crop 免費多視角;關鍵:不需成對負樣本了"),
  infomin: B(
    "架構/loss 沿用 SimCLR,只動「視角」",
    "問:什麼是好視角?答:minimal sufficient(留語意、砍捷徑)",
    "用更激進增強破壞顏色/紋理捷徑",
    "收幕:SwAV 已暗示負樣本不是唯一解 → 下一幕乾脆全拿掉"),
  act3: B(
    "最瘋狂一幕:拿掉最後一根拐杖——負樣本",
    "負樣本 = 「推開」的力量;只剩拉近 → 最會塌成一點",
    "當年共識:沒有負樣本一定坍塌(接下來打臉它)"),
  byol: B(
    "不要負樣本,一個都不要",
    "架構:online + target 兩分支;online 多一個 predictor(製造不對稱)",
    "target = online 的 EMA 影分身 + stop-gradient",
    "loss:用 predictor 預測 target 特徵的 MSE(不是推開)",
    "靠「不對稱」而非推開來防坍塌,震撼全場"),
  simsiam: B(
    "再砍:連 EMA 都不要,兩分支共享同一網路",
    "只留 predictor + stop-gradient;loss = 負餘弦相似度",
    "最小化證明:stop-gradient 才是防坍塌的關鍵"),
  barlow_twins: B(
    "換哲學:不玩拉近推開,看兩視角的「互相關矩陣」",
    "目標:矩陣 → 單位矩陣(對角 1 = 不變性、非對角 0 = 去冗餘)",
    "高維投影頭(8192);無負樣本/EMA/predictor,功夫全在 loss",
    "三篇合看的線索:防坍塌的關鍵是「打破對稱」,不是推開"),
  act4: B(
    "拐杖幾乎拿光,但引擎一直是 CNN",
    "同時 ViT 出現;問:把引擎換成 Transformer 會怎樣?",
    "最後一幕:馴服新引擎,一路衝到基礎模型"),
  moco_v3: B(
    "架構 = MoCo 那套搬上 ViT(backbone ResNet → Transformer)",
    "問題:超不穩、loss 亂跳",
    "兇手 = 最前面的 patch embedding;解法:凍結它就穩了",
    "順手丟 queue,loss 改回對稱 in-batch InfoNCE"),
  dino: B(
    "跟 MoCo v3 一樣在 ViT,但更進一步:連負樣本都不要",
    "自蒸餾:student 學 teacher(teacher = student 的 EMA);loss = 交叉熵",
    "防坍塌:teacher 端 centering(拉回中心)+ sharpening(變尖)",
    "最炸:attention map 自己框出物體(免標註浮現分割)"),
  dinov2: B(
    "沒發明新招,做更難的事:把對的方法放大",
    "架構/loss = DINO + iBOT(patch 級遮罩,補特徵不補像素)",
    "餵 LVD-142M(上億張精選)→ 通用視覺基礎模型",
    "特徵免訓練直接接分類/分割/深度;接上 foundation model 浪潮"),
  dinov3: B(
    "2025 續章,跟 DINOv2 同路(架構/loss 幾乎一樣)",
    "差異一:規模再爆(7B 參數 / 17 億張圖)",
    "差異二:Gram anchoring 穩住 dense 特徵、防退化",
    "dense 任務再上一層;右邊是官方權重的 CIFAR-10 UMAP",
    "DINO 線 v1 → v3 = 同一點子不斷放大"),
  collapseTable: B(
    "回扣開場伏筆:最怕的就是坍塌",
    "看表:每代「用不用負樣本 + 靠什麼防坍塌」全列出",
    "機制各不同(負樣本 / Sinkhorn / EMA / stop-grad / 去冗餘 / centering)",
    "punchline:14 個方法都在回答同一句——沒標籤怎麼不坍塌"),
  prog1: B(
    "理論講完看實跑:2×H100、CIFAR-10、200 epoch、每 epoch 拍 UMAP",
    "Era 1–2 四個:epoch 0 一坨(坍塌起點)→ 逐漸分群",
    "SimCLR 分群更俐落乾淨(in-batch 大量正負樣本的威力)",
    "純靠「同圖像、異圖不像」,沒用任何標籤"),
  prog2: B(
    "Era 2–3:InfoMin、BYOL、SimSiam",
    "重點:BYOL/SimSiam 完全沒負樣本、理論上最易塌",
    "卻一樣穩穩分群、沒塌成一點 → 眼見為憑",
    "stop-gradient、predictor+EMA 真的能防坍塌"),
  prog3: B(
    "補完:原本用官方權重的 6 個也從零自訓",
    "MoCo v1/v2、SwAV、Barlow、MoCo v3、DINO",
    "連兩個 ViT(最怕發散)也乾淨分群",
    "到此 13 個自訓方法全部眼見為憑"),
  liveDemo: B(
    "最核心對照:左 = 訓練前一團毛球(坍塌長相)、右 = 訓練後乾淨分群",
    "全程沒用任何標籤",
    "最大體會:分群好不好主要看「有沒有充分訓練」",
    "方法差別不在「能不能分群」,而在「用什麼代價、走哪條路防坍塌」",
    "播 GIF,留幾秒讓大家看點群散開、歸隊"),
  closing: B(
    "一句話帶走:5 年來對比學習一直在「拿掉依賴」",
    "三重點:① 共用 InfoNCE 直覺 ② 真難題是防坍塌、負樣本只是一條路 ③ 趨勢:更少假設、更大規模 → foundation model",
    "開源教學專案:14 方法統一實作、可重現,歡迎 clone",
    "謝謝聆聽,進 Q&A"),
};

// ---------- reusable bits ----------
function timelineRibbon(slide, y) {
  // a thin ribbon with 4 era nodes + years, used as a footer motif
  const eras = [
    { yr: "2018", c: E1, lab: "proxy" },
    { yr: "2020", c: E2, lab: "contrastive" },
    { yr: "2021", c: E3, lab: "no-negative" },
    { yr: "2021+", c: E4, lab: "transformer" },
  ];
  const x0 = 0.9, x1 = PW - 0.9;
  slide.addShape(pres.shapes.LINE, { x: x0, y: y, w: x1 - x0, h: 0, line: { color: "33486E", width: 1.5 } });
  const step = (x1 - x0) / 3;
  eras.forEach((e, i) => {
    const cx = x0 + step * i;
    slide.addShape(pres.shapes.OVAL, { x: cx - 0.09, y: y - 0.09, w: 0.18, h: 0.18, fill: { color: e.c }, line: { color: NAVY, width: 1.5 } });
    slide.addText(e.yr, { x: cx - 0.6, y: y + 0.12, w: 1.2, h: 0.26, align: "center", fontFace: FACE, fontSize: 11, bold: true, color: e.c, margin: 0 });
    slide.addText(e.lab, { x: cx - 0.7, y: y + 0.38, w: 1.4, h: 0.22, align: "center", fontFace: MONO, fontSize: 8.5, color: ICE, margin: 0 });
  });
}

function eraChip(slide, x, y, label, color) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 1.95, h: 0.42, rectRadius: 0.21, fill: { color }, shadow: softShadow() });
  slide.addText(label, { x, y, w: 1.95, h: 0.42, align: "center", valign: "middle", fontFace: FACE, fontSize: 12.5, bold: true, color: WHITE, charSpacing: 1, margin: 0 });
}

function contentHeader(slide, chipLabel, color, title) {
  slide.background = { color: LIGHT };
  eraChip(slide, 0.7, 0.55, chipLabel, color);
  slide.addText(title, { x: 0.68, y: 1.05, w: 12, h: 0.85, fontFace: FACE, fontSize: 33, bold: true, color: INK, margin: 0 });
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.72, y: 1.92, w: 0.55, h: 0.07, fill: { color } });
}

function actDivider(actNo, color, years, question, notes) {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  // big tinted act number block on the left
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.28, h: PH, fill: { color } });
  s.addText("ACT", { x: 0.9, y: 2.0, w: 3, h: 0.5, fontFace: MONO, fontSize: 20, bold: true, color: color, charSpacing: 6, margin: 0 });
  s.addText(actNo, { x: 0.78, y: 2.35, w: 4.2, h: 2.6, fontFace: FACE, fontSize: 150, bold: true, color: WHITE, margin: 0 });
  s.addText(years, { x: 5.0, y: 2.55, w: 7.5, h: 0.5, fontFace: MONO, fontSize: 17, bold: true, color: color, margin: 0 });
  s.addText(question, { x: 5.0, y: 3.05, w: 7.6, h: 1.9, fontFace: FACE, fontSize: 31, bold: true, color: WHITE, margin: 0, lineSpacingMultiple: 1.1 });
  timelineRibbon(s, 6.7);
  if (notes) s.addNotes(notes);
  return s;
}

// ---------- per-paper slide template + architecture diagram ----------
function abox(s, x, y, w, h, text, fill, tcol, bcol) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.05, fill: { color: fill }, line: { color: bcol || fill, width: 1 }, shadow: softShadow() });
  s.addText(text, { x, y, w, h, align: "center", valign: "middle", fontFace: FACE, fontSize: 9.5, bold: true, color: tcol, margin: 0, lineSpacingMultiple: 0.95 });
}
function hArrow(s, x, y, w) {
  s.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color: "90A0C0", width: 1.5, endArrowType: "triangle" } });
}
function diagArrow(s, x, yTop, w, h, up) {
  s.addShape(pres.shapes.LINE, { x, y: yTop, w, h, flipV: !!up, line: { color: "90A0C0", width: 1.25, endArrowType: "triangle" } });
}
const BOXF = "EAF0FA", BOXB = "C3D2EA";

// generic two-branch (siamese) architecture; cfg flags morph it per method
function siamese(s, o, cfg) {
  const rowH = 0.5;
  const midY = o.y + o.h / 2;
  const topY = o.y + 0.04, botY = o.y + o.h - rowH - 0.04;
  const topMid = topY + rowH / 2, botMid = botY + rowH / 2;
  const ix = o.x, iw = 0.62;
  abox(s, ix, midY - 0.3, iw, 0.6, "輸入\nx", NAVY3, WHITE);
  const ex = o.x + 1.45, ew = 1.3;

  if (cfg.single) {
    hArrow(s, ix + iw, midY, ex - (ix + iw));
    abox(s, ex, midY - rowH / 2, ew, rowH, cfg.top, BOXF, INK, BOXB);
    let cx = ex + ew;
    if (cfg.proj) { hArrow(s, cx, midY, 0.5); abox(s, cx + 0.5, midY - rowH / 2, 0.85, rowH, "投影\ng", BOXF, INK, BOXB); cx += 0.5 + 0.85; }
    const lx = cx + 0.55;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: lx, y: midY - 0.32, w: 1.55, h: 0.64, rectRadius: 0.08, fill: { color: NAVY }, shadow: makeShadow() });
    s.addText(cfg.loss, { x: lx, y: midY - 0.32, w: 1.55, h: 0.64, align: "center", valign: "middle", fontFace: FACE, fontSize: 10.5, bold: true, color: ACCENT, margin: 0 });
    hArrow(s, cx, midY, lx - cx);
    if (cfg.extra) {
      const exx = lx + 1.55 + 0.55;
      abox(s, exx, midY - 0.32, 1.7, 0.64, cfg.extra, E2, NAVY);
      s.addShape(pres.shapes.LINE, { x: lx + 1.55, y: midY, w: exx - (lx + 1.55), h: 0, line: { color: "90A0C0", width: 1.5, beginArrowType: "triangle" } });
    }
    return;
  }

  diagArrow(s, ix + iw, topMid, ex - (ix + iw), midY - topMid, true);
  diagArrow(s, ix + iw, midY, ex - (ix + iw), botMid - midY, false);
  abox(s, ex, topY, ew, rowH, cfg.top, BOXF, INK, BOXB);
  abox(s, ex, botY, ew, rowH, cfg.bot, BOXF, INK, BOXB);
  if (cfg.botNote) s.addText(cfg.botNote, { x: ex, y: botY + rowH + 0.01, w: ew, h: 0.22, align: "center", fontFace: MONO, fontSize: 8, bold: true, color: cfg.noteColor || MUTE, margin: 0 });
  let cx = ex + ew;
  if (cfg.proj) {
    hArrow(s, cx, topMid, 0.45); hArrow(s, cx, botMid, 0.45);
    abox(s, cx + 0.45, topY, 0.8, rowH, "投影\ng", BOXF, INK, BOXB);
    abox(s, cx + 0.45, botY, 0.8, rowH, "投影\ng", BOXF, INK, BOXB);
    cx += 0.45 + 0.8;
  }
  let zxTop = cx;
  if (cfg.pred) {
    hArrow(s, cx, topMid, 0.45);
    abox(s, cx + 0.45, topY, 0.95, rowH, "predictor\nq", E3, WHITE);
    zxTop = cx + 0.45 + 0.95;
  }
  const gap0 = (cfg.sgBot && !cfg.pred) ? 1.45 : 0.55;
  const lx = Math.max(zxTop, cx) + gap0;
  const lossTop = topMid - 0.06, lossH = (botMid - topMid) + 0.12;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: lx, y: lossTop, w: 1.55, h: lossH, rectRadius: 0.08, fill: { color: NAVY }, shadow: makeShadow() });
  s.addText(cfg.loss, { x: lx, y: lossTop, w: 1.55, h: lossH, align: "center", valign: "middle", fontFace: FACE, fontSize: 10.5, bold: true, color: ACCENT, margin: 0, lineSpacingMultiple: 1.0 });
  hArrow(s, zxTop, topMid, lx - zxTop);
  hArrow(s, cx, botMid, lx - cx);
  if (cfg.sgBot) s.addText("⊘ stop-grad", { x: cx + 0.1, y: botMid - 0.34, w: (lx - cx) - 0.2, h: 0.24, align: "center", fontFace: MONO, fontSize: 8.5, bold: true, color: E3, margin: 0 });
  if (cfg.extra) {
    const exx = lx + 1.55 + 0.5;
    abox(s, exx, midY - 0.32, 1.75, 0.64, cfg.extra, E2, NAVY);
    s.addShape(pres.shapes.LINE, { x: lx + 1.55, y: midY, w: exx - (lx + 1.55), h: 0, line: { color: "90A0C0", width: 1.5, beginArrowType: "triangle" } });
  }
}

function methodSlide(spec) {
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  eraChip(s, 0.7, 0.5, spec.era.chip, spec.era.color);
  s.addText(spec.name, { x: 0.68, y: 0.97, w: 9.0, h: 0.66, fontFace: FACE, fontSize: 29, bold: true, color: INK, margin: 0 });
  s.addText(spec.venue, { x: 9.5, y: 1.05, w: 3.1, h: 0.5, align: "right", valign: "middle", fontFace: MONO, fontSize: 13, bold: true, color: spec.era.color, margin: 0 });
  s.addText("📄 " + spec.authors, { x: 0.72, y: 1.62, w: 11.85, h: 0.3, fontFace: FACE, fontSize: 10.5, italic: true, color: MUTE, margin: 0 });
  const onEra = (spec.era.color === E1 || spec.era.color === E2) ? NAVY : WHITE;

  // diagram card
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 2.04, w: 11.93, h: 2.02, fill: { color: CARD }, line: { color: "DCE3F0", width: 1 }, shadow: softShadow() });
  s.addText("架構示意", { x: 0.88, y: 2.12, w: 3, h: 0.26, fontFace: FACE, fontSize: 10, bold: true, color: MUTE, charSpacing: 1, margin: 0 });
  spec.diagram(s, { x: 1.05, y: 2.46, w: 11.3, h: 1.5 });

  // --- bottom: three columns (text · loss/contribution · UMAP demo) ---
  // col1: text card
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 4.22, w: 5.35, h: 2.73, fill: { color: CARD }, line: { color: "DCE3F0", width: 1 }, shadow: softShadow() });
  s.addText("核心點子", { x: 0.92, y: 4.34, w: 4.95, h: 0.28, fontFace: FACE, fontSize: 12, bold: true, color: spec.era.color, margin: 0 });
  s.addText(spec.idea, { x: 0.92, y: 4.62, w: 4.95, h: 0.74, fontFace: FACE, fontSize: 11.5, color: INK, margin: 0, lineSpacingMultiple: 1.05 });
  s.addText("關鍵機制", { x: 0.92, y: 5.42, w: 4.95, h: 0.28, fontFace: FACE, fontSize: 12, bold: true, color: spec.era.color, margin: 0 });
  s.addText(spec.mechanism.map((m, i) => ({ text: m, options: { bullet: { code: "2022" }, breakLine: i < spec.mechanism.length - 1, color: MUTE } })),
    { x: 0.92, y: 5.72, w: 4.95, h: 1.15, fontFace: FACE, fontSize: 10.5, paraSpaceAfter: 5, lineSpacingMultiple: 1.0, margin: 0 });

  // col2: loss panel + contribution
  s.addShape(pres.shapes.RECTANGLE, { x: 6.2, y: 4.22, w: 3.05, h: 1.5, fill: { color: NAVY }, shadow: makeShadow() });
  s.addText("LOSS", { x: 6.4, y: 4.31, w: 2.6, h: 0.24, fontFace: MONO, fontSize: 9, bold: true, color: ACCENT, charSpacing: 2, margin: 0 });
  s.addText(spec.loss, { x: 6.26, y: 4.55, w: 2.94, h: 0.74, align: "center", valign: "middle", fontFace: MONO, fontSize: 10, bold: true, color: WHITE, margin: 0, lineSpacingMultiple: 1.05 });
  s.addText(spec.lossNote, { x: 6.28, y: 5.33, w: 2.9, h: 0.32, align: "center", fontFace: FACE, fontSize: 9, italic: true, color: ICE, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: 6.2, y: 5.88, w: 3.05, h: 1.07, fill: { color: spec.era.color } });
  s.addText("貢獻 / 關鍵", { x: 6.4, y: 5.95, w: 2.7, h: 0.26, fontFace: FACE, fontSize: 10, bold: true, color: onEra, charSpacing: 1, margin: 0 });
  s.addText(spec.contribution, { x: 6.4, y: 6.22, w: 2.75, h: 0.68, fontFace: FACE, fontSize: 11, bold: true, color: onEra, margin: 0, lineSpacingMultiple: 1.03 });

  // col3: per-method UMAP demo
  const hasImg = spec.demo && fs.existsSync(spec.demo);
  const TRAINED = ["instance_discrimination", "invariant_spread", "simclr_v1", "simclr_v2", "infomin", "byol", "simsiam", "moco_v1", "moco_v2", "swav", "barlow_twins", "moco_v3", "dino"];
  const dkey = spec.demo ? spec.demo.split("/").pop().replace(/\.(png|gif)$/, "") : "";
  const isTrained = TRAINED.includes(dkey);
  const isGif = /\.gif$/.test(spec.demo || "");
  s.addShape(pres.shapes.RECTANGLE, { x: 9.4, y: 4.22, w: 3.23, h: 2.73, fill: { color: NAVY }, shadow: makeShadow() });
  s.addText(!hasImg ? "LIVE DEMO · 待訓練" : (isTrained ? "LIVE DEMO · 自訓動畫 UMAP" : "LIVE DEMO · 官方權重 UMAP"), { x: 9.5, y: 4.31, w: 3.05, h: 0.26, align: "center", fontFace: MONO, fontSize: 8, bold: true, color: ACCENT, charSpacing: 1, margin: 0 });
  if (hasImg) {
    s.addImage({ path: spec.demo, x: 9.965, y: 4.62, w: 2.1, h: 2.1, sizing: { type: "contain", w: 2.1, h: 2.1 } });
    s.addText(isTrained ? (isGif ? "CIFAR-10 · 自訓 200ep · GIF（PPT 365 播放）" : "CIFAR-10 · 自訓 200 epoch（H100）") : "官方釋出之預訓練權重 · CIFAR-10 特徵", { x: 9.45, y: 6.66, w: 3.15, h: 0.24, align: "center", fontFace: FACE, fontSize: 7.5, italic: true, color: MUTE, margin: 0 });
  } else {
    s.addText("⏳", { x: 9.4, y: 5.02, w: 3.23, h: 0.55, align: "center", fontFace: FACE, fontSize: 28, color: ICE, margin: 0 });
    s.addText("待 GPU 充分訓練後\n補上該方法的 UMAP", { x: 9.5, y: 5.68, w: 3.05, h: 0.7, align: "center", valign: "top", fontFace: FACE, fontSize: 11, color: ICE, margin: 0, lineSpacingMultiple: 1.2 });
  }
  // speaker notes — keyed by the method's demo basename (instance_discrimination … dinov2)
  const _nk = spec.demo ? spec.demo.split("/").pop().replace(/\.(png|gif)$/, "") : "";
  if (NOTES[_nk]) s.addNotes(NOTES[_nk]);
}

// era presets
const A1 = (chip) => ({ chip, color: E1 });
const A2 = (chip) => ({ chip, color: E2 });
const A3 = (chip) => ({ chip, color: E3 });
const A4 = (chip) => ({ chip, color: E4 });

// ============================================================
// 1. COVER
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  // faint corner accent
  s.addShape(pres.shapes.OVAL, { x: 10.4, y: -2.6, w: 5.5, h: 5.5, fill: { color: NAVY3, transparency: 35 } });
  s.addShape(pres.shapes.OVAL, { x: 11.6, y: 4.6, w: 4.2, h: 4.2, fill: { color: E2, transparency: 82 } });

  s.addText("自監督學習 · SELF-SUPERVISED LEARNING", {
    x: 0.95, y: 1.25, w: 11, h: 0.4, fontFace: MONO, fontSize: 14, bold: true, color: ACCENT, charSpacing: 3, margin: 0 });
  s.addText("對比學習的演化史", {
    x: 0.9, y: 1.75, w: 11.5, h: 1.3, fontFace: FACE, fontSize: 60, bold: true, color: WHITE, margin: 0 });
  s.addText("The Evolution of Contrastive Learning", {
    x: 0.95, y: 3.05, w: 11, h: 0.6, fontFace: FACE, fontSize: 24, color: ICE, margin: 0 });
  s.addText([
    { text: "從 2018 到今天 — 一場「", options: {} },
    { text: "不斷拿掉依賴", options: { color: ACCENT, bold: true } },
    { text: "」的旅程", options: {} },
  ], { x: 0.95, y: 3.75, w: 11, h: 0.5, fontFace: FACE, fontSize: 17, color: ICE, margin: 0 });

  timelineRibbon(s, 5.55);
  s.addText("Yi-Ting Li · 2026", { x: 0.95, y: 6.85, w: 6, h: 0.35, fontFace: MONO, fontSize: 11, color: MUTE, margin: 0 });
  s.addNotes(NOTES.cover);
}

// ============================================================
// 2. BIG PICTURE — one sentence
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("一句話看懂整個領域", { x: 0.9, y: 0.6, w: 11.5, h: 0.7, fontFace: FACE, fontSize: 30, bold: true, color: WHITE, margin: 0 });
  s.addText([
    { text: "對比學習花了 5 年，一步步拿掉它對", options: {} },
    { text: "「外部拐杖」", options: { color: ACCENT, bold: true } },
    { text: "的依賴。", options: {} },
  ], { x: 0.9, y: 1.45, w: 11.6, h: 0.9, fontFace: FACE, fontSize: 23, color: ICE, margin: 0 });

  const steps = [
    { yr: "2018", t: "拿掉標籤", s: "proxy task", c: E1 },
    { yr: "2020", t: "拿掉 memory bank", s: "queue / 大 batch", c: E2 },
    { yr: "2021", t: "拿掉負樣本", s: "no-negative", c: E3 },
    { yr: "2021+", t: "換上 Transformer", s: "foundation model", c: E4 },
  ];
  const bx = 0.9, bw = 2.72, gap = 0.28, by = 3.05, bh = 2.5;
  steps.forEach((st, i) => {
    const x = bx + i * (bw + gap);
    s.addShape(pres.shapes.RECTANGLE, { x, y: by, w: bw, h: bh, fill: { color: NAVY2 }, line: { color: st.c, width: 1.5 }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y: by, w: bw, h: 0.12, fill: { color: st.c } });
    s.addText(st.yr, { x: x + 0.25, y: by + 0.32, w: bw - 0.5, h: 0.5, fontFace: MONO, fontSize: 22, bold: true, color: st.c, margin: 0 });
    s.addText(st.t, { x: x + 0.25, y: by + 0.95, w: bw - 0.5, h: 0.85, fontFace: FACE, fontSize: 18.5, bold: true, color: WHITE, margin: 0, valign: "top" });
    s.addText(st.s, { x: x + 0.25, y: by + 1.95, w: bw - 0.5, h: 0.4, fontFace: MONO, fontSize: 11.5, color: ICE, margin: 0 });
    if (i < steps.length - 1) {
      s.addText("→", { x: x + bw - 0.06, y: by + 0.9, w: gap + 0.12, h: 0.6, align: "center", valign: "middle", fontFace: FACE, fontSize: 26, bold: true, color: ACCENT, margin: 0 });
    }
  });
  s.addText("每一步，都是前一代「撞到牆」之後翻牆的結果。", {
    x: 0.9, y: 6.05, w: 11.6, h: 0.5, fontFace: FACE, fontSize: 15, italic: true, color: ACCENT, margin: 0 });
  s.addNotes(NOTES.bigPicture);
}

// ============================================================
// 3. CORE INTUITION (light, two-column with diagram)
// ============================================================
{
  const s = pres.addSlide();
  contentHeader(s, "核心直覺", ACCENT.replace("22D3EE", "0E7490"), "對比學習在做什麼？");
  // left text
  s.addText([
    { text: "沒有標籤，從資料本身造出學習信號", options: { bullet: { code: "2022" }, breakLine: true, bold: true, color: INK } },
    { text: "同一張圖的兩個增強版本 = 正樣本對 → 拉近", options: { bullet: { code: "2022" }, breakLine: true, color: "2A6B4F" } },
    { text: "不同圖的特徵 = 負樣本 → 推開", options: { bullet: { code: "2022" }, breakLine: true, color: "B43050" } },
    { text: "學到「對增強保持不變」的語意特徵", options: { bullet: { code: "2022" }, color: INK } },
  ], { x: 0.7, y: 2.35, w: 6.0, h: 3.6, fontFace: FACE, fontSize: 17.5, color: INK, paraSpaceAfter: 16, lineSpacingMultiple: 1.05 });

  // right diagram panel
  const px = 7.15, py = 2.3, pw2 = 5.45, ph2 = 4.4;
  s.addShape(pres.shapes.RECTANGLE, { x: px, y: py, w: pw2, h: ph2, fill: { color: CARD }, line: { color: "DCE3F0", width: 1 }, shadow: softShadow() });
  const cx = px + pw2 / 2, cy = py + ph2 / 2;
  // pull arrow + positive
  s.addShape(pres.shapes.LINE, { x: cx - 1.55, y: cy, w: 1.35, h: 0, line: { color: "2A8F5F", width: 2.5, endArrowType: "triangle", beginArrowType: "triangle" } });
  s.addShape(pres.shapes.OVAL, { x: px + 0.45, y: cy - 0.38, w: 0.76, h: 0.76, fill: { color: "2A8F5F" }, shadow: softShadow() });
  s.addText("正", { x: px + 0.45, y: cy - 0.38, w: 0.76, h: 0.76, align: "center", valign: "middle", fontFace: FACE, fontSize: 16, bold: true, color: WHITE, margin: 0 });
  s.addText("拉近", { x: cx - 1.7, y: cy - 0.62, w: 1.6, h: 0.3, align: "center", fontFace: FACE, fontSize: 12, bold: true, color: "2A8F5F", margin: 0 });
  // anchor
  s.addShape(pres.shapes.OVAL, { x: cx - 0.5, y: cy - 0.5, w: 1.0, h: 1.0, fill: { color: INK }, line: { color: ACCENT, width: 3 }, shadow: makeShadow() });
  s.addText("錨點", { x: cx - 0.5, y: cy - 0.5, w: 1.0, h: 1.0, align: "center", valign: "middle", fontFace: FACE, fontSize: 15, bold: true, color: WHITE, margin: 0 });
  // push arrows + negatives (two)
  s.addShape(pres.shapes.LINE, { x: cx + 0.62, y: cy - 0.85, w: 1.4, h: 0.7, flipV: true, line: { color: "C0395A", width: 2.5, endArrowType: "triangle" } });
  s.addShape(pres.shapes.LINE, { x: cx + 0.62, y: cy + 0.15, w: 1.4, h: 0.7, line: { color: "C0395A", width: 2.5, endArrowType: "triangle" } });
  s.addShape(pres.shapes.OVAL, { x: cx + 2.0, y: cy - 1.15, w: 0.7, h: 0.7, fill: { color: "C0395A" }, shadow: softShadow() });
  s.addText("負", { x: cx + 2.0, y: cy - 1.15, w: 0.7, h: 0.7, align: "center", valign: "middle", fontFace: FACE, fontSize: 14, bold: true, color: WHITE, margin: 0 });
  s.addShape(pres.shapes.OVAL, { x: cx + 2.0, y: cy + 0.45, w: 0.7, h: 0.7, fill: { color: "C0395A" }, shadow: softShadow() });
  s.addText("負", { x: cx + 2.0, y: cy + 0.45, w: 0.7, h: 0.7, align: "center", valign: "middle", fontFace: FACE, fontSize: 14, bold: true, color: WHITE, margin: 0 });
  s.addText("推開", { x: cx + 0.72, y: cy - 0.15, w: 1.2, h: 0.3, align: "center", valign: "middle", fontFace: FACE, fontSize: 12, bold: true, color: "C0395A", margin: 0 });
  s.addNotes(NOTES.coreIntuition);
}

// ============================================================
// 4. InfoNCE — shared language
// ============================================================
{
  const s = pres.addSlide();
  contentHeader(s, "共同語言", "0E7490", "一條式貫穿所有方法：InfoNCE");
  // formula panel (dark) — rendered as a stacked fraction so nothing wraps
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 2.35, w: 11.9, h: 1.55, fill: { color: NAVY }, shadow: makeShadow() });
  s.addText("ℒ  =  − log", { x: 1.7, y: 2.35, w: 3.1, h: 1.55, align: "right", valign: "middle", fontFace: MONO, fontSize: 24, bold: true, color: WHITE, margin: 0 });
  s.addText("exp( sim(zᵢ, zⱼ) / τ )", { x: 5.1, y: 2.52, w: 6.6, h: 0.55, align: "center", valign: "bottom", fontFace: MONO, fontSize: 19, bold: true, color: "7CF0A0", margin: 0 });
  s.addShape(pres.shapes.LINE, { x: 5.35, y: 3.14, w: 6.1, h: 0, line: { color: WHITE, width: 1.5 } });
  s.addText("Σₖ  exp( sim(zᵢ, zₖ) / τ )", { x: 5.1, y: 3.2, w: 6.6, h: 0.55, align: "center", valign: "top", fontFace: MONO, fontSize: 19, bold: true, color: "FF8FA8", margin: 0 });

  const cards = [
    { t: "分子 = 正樣本", d: "同圖兩個增強的相似度\n→ 希望它「大」", c: "2A8F5F" },
    { t: "分母 = 負樣本", d: "與所有其他樣本的相似度\n→ 希望它「小」", c: "C0395A" },
    { t: "τ = 溫度", d: "控制分布尖銳程度\n→ 影響對難負樣本的關注", c: "0E7490" },
  ];
  const cw = 3.83, cg = 0.2, cy0 = 4.25, ch = 1.55, x0 = 0.7;
  cards.forEach((c, i) => {
    const x = x0 + i * (cw + cg);
    s.addShape(pres.shapes.RECTANGLE, { x, y: cy0, w: cw, h: ch, fill: { color: CARD }, line: { color: "DCE3F0", width: 1 }, shadow: softShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y: cy0, w: 0.1, h: ch, fill: { color: c.c } });
    s.addText(c.t, { x: x + 0.3, y: cy0 + 0.18, w: cw - 0.5, h: 0.4, fontFace: FACE, fontSize: 16, bold: true, color: c.c, margin: 0 });
    s.addText(c.d, { x: x + 0.3, y: cy0 + 0.6, w: cw - 0.5, h: 0.85, fontFace: FACE, fontSize: 12.5, color: MUTE, margin: 0, lineSpacingMultiple: 1.05 });
  });
  s.addText([
    { text: "接下來 14 種方法，全是這條式的 ", options: {} },
    { text: "「變形」", options: { bold: true, color: "0E7490" } },
    { text: " 或對它的 ", options: {} },
    { text: "「逃脫」", options: { bold: true, color: E3 } },
    { text: "。", options: {} },
  ], { x: 0.7, y: 6.1, w: 11.9, h: 0.5, align: "center", fontFace: FACE, fontSize: 15, italic: true, color: INK, margin: 0 });
  s.addNotes(NOTES.infonce);
}

// ============================================================
// 5. EVOLUTION MAP (dark) — the centerpiece
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("演化全景圖", { x: 0.9, y: 0.5, w: 11.5, h: 0.7, fontFace: FACE, fontSize: 30, bold: true, color: WHITE, margin: 0 });
  s.addText("時間 →　每一代都在解決前一代的瓶頸", { x: 0.9, y: 1.18, w: 11.5, h: 0.4, fontFace: FACE, fontSize: 14, color: ICE, margin: 0 });

  const cols = [
    { yr: "2018–19", c: E1, head: "Era 1 · 代理任務", ms: ["Instance\nDiscrimination", "Invariant\nSpread", "memory bank\n+ NCE"] },
    { yr: "2020", c: E2, head: "Era 2 · 對比 / 分群", ms: ["MoCo v1 / v2", "SimCLR v1 / v2", "SwAV · InfoMin"] },
    { yr: "2020–21", c: E3, head: "Era 3 · 無負樣本", ms: ["BYOL", "SimSiam", "Barlow Twins"] },
    { yr: "2021+", c: E4, head: "Era 4 · Transformer", ms: ["MoCo v3", "DINO", "DINOv2"] },
  ];
  const cw = 2.78, cg = 0.32, x0 = 0.9, y0 = 2.0, ch = 4.3;
  cols.forEach((col, i) => {
    const x = x0 + i * (cw + cg);
    s.addShape(pres.shapes.RECTANGLE, { x, y: y0, w: cw, h: 0.6, fill: { color: col.c } });
    s.addText(col.yr, { x: x + 0.15, y: y0 + 0.08, w: cw - 0.3, h: 0.44, fontFace: MONO, fontSize: 15, bold: true, color: NAVY, margin: 0, valign: "middle" });
    s.addShape(pres.shapes.RECTANGLE, { x, y: y0 + 0.6, w: cw, h: ch - 0.6, fill: { color: NAVY2 }, line: { color: col.c, width: 1.5 } });
    s.addText(col.head, { x: x + 0.2, y: y0 + 0.75, w: cw - 0.4, h: 0.55, fontFace: FACE, fontSize: 14.5, bold: true, color: col.c, margin: 0 });
    let yy = y0 + 1.4;
    col.ms.forEach((m) => {
      const lines = m.split("\n").length;
      const mh = lines > 1 ? 0.78 : 0.55;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.2, y: yy, w: cw - 0.4, h: mh, rectRadius: 0.06, fill: { color: NAVY3 } });
      s.addText(m.replace("\n", " "), { x: x + 0.32, y: yy, w: cw - 0.6, h: mh, valign: "middle", fontFace: FACE, fontSize: 13, bold: true, color: WHITE, margin: 0 });
      yy += mh + 0.18;
    });
    if (i < cols.length - 1) {
      s.addText("→", { x: x + cw - 0.04, y: y0 + 1.8, w: cg + 0.08, h: 0.6, align: "center", valign: "middle", fontFace: FACE, fontSize: 24, bold: true, color: ACCENT, margin: 0 });
    }
  });
  s.addText("拿掉標籤 → 拿掉 memory bank → 拿掉負樣本 → 換新架構", {
    x: 0.9, y: 6.55, w: 11.5, h: 0.45, align: "center", fontFace: MONO, fontSize: 13, bold: true, color: ICE, margin: 0 });
  s.addNotes(NOTES.evoMap);
}

// ============================================================
// ACT 1
// ============================================================
actDivider("1", E1, "2018 – 2019", "起點：沒有標籤，\n要怎麼定義「任務」？", NOTES.act1);

// --- Instance Discrimination ---
methodSlide({
  era: A1("ERA 1 · 2018"), name: "Instance Discrimination", demo: "demo_assets/gifs/instance_discrimination.gif", venue: "CVPR 2018",
  authors: "Z. Wu, Y. Xiong, S. X. Yu, D. Lin — Unsupervised Feature Learning via Non-Parametric Instance Discrimination",
  idea: "把每張圖當成「自己的一類」，做 instance-level 區分，完全不需人工標籤。",
  mechanism: ["Memory Bank 儲存每張圖的 L2 特徵向量", "NCE 近似完整 softmax（常數 Z 首批估計後固定）", "弱增強、n_views=1，bank 提供「第二視角」"],
  contribution: "非參數記憶庫先驅；痛點：bank 特徵會過時。",
  diagram: (s, o) => siamese(s, o, { single: true, proj: false, top: "編碼器\nf", loss: "NCE", extra: "Memory\nBank（全資料特徵）" }),
  loss: "P(i|v) = exp(vᵢᵀf/τ) / Σⱼ exp(vⱼᵀf/τ)", lossNote: "非參數 softmax + NCE 近似",
});

// --- Invariant Spread ---
methodSlide({
  era: A1("ERA 1 · 2019"), name: "Invariant and Spreading Instance Feature", demo: "demo_assets/gifs/invariant_spread.gif", venue: "CVPR 2019",
  authors: "M. Ye, X. Zhang, P. C. Yuen, S.-F. Chang — Unsupervised Embedding Learning via Invariant and Spreading Instance Feature",
  idea: "不用記憶庫，直接在 batch 內把同圖兩視角拉近、不同圖推開。",
  mechanism: ["兩個增強視角 + 對稱 InfoNCE", "in-batch 負樣本（無 queue、無 memory bank）", "SimCLR 的直系祖先"],
  contribution: "in-batch softmax 對比的雛形，啟發後續整個 Era 2。",
  diagram: (s, o) => siamese(s, o, { top: "編碼器\nf", bot: "編碼器\nf", botNote: "共享權重", proj: false, loss: "對稱\nInfoNCE" }),
  loss: "−Σ log exp(sim/τ) / Σ exp(sim/τ)", lossNote: "in-batch 對稱 softmax",
});

// ============================================================
// ACT 2
// ============================================================
actDivider("2", E2, "2020", "大爆發：\n負樣本要從哪裡來？", NOTES.act2);

// --- MoCo v1 ---
methodSlide({
  era: A2("ERA 2 · 2020"), name: "MoCo v1", demo: "demo_assets/gifs/moco_v1.gif", venue: "CVPR 2020",
  authors: "K. He, H. Fan, Y. Wu, S. Xie, R. Girshick — Momentum Contrast for Unsupervised Visual Representation Learning",
  idea: "用動量編碼器 + FIFO queue，提供大量「不過時」的負樣本。",
  mechanism: ["query 編碼器（梯度更新）+ key 編碼器（EMA 動量）", "FIFO queue 儲存歷史 key 當負樣本", "負樣本量與 batch size 解耦"],
  contribution: "小 batch 也能有大量、一致的負樣本。",
  diagram: (s, o) => siamese(s, o, { top: "f_q\n(梯度)", bot: "f_k", botNote: "EMA 動量", proj: false, loss: "InfoNCE", extra: "Queue\n(歷史 key 負樣本)" }),
  loss: "−log exp(q·k₊/τ) / Σ_{queue} exp(q·k/τ)", lossNote: "queue 提供大量負樣本",
});

// --- MoCo v2 ---
methodSlide({
  era: A2("ERA 2 · 2020"), name: "MoCo v2", demo: "demo_assets/gifs/moco_v2.gif", venue: "arXiv 2020",
  authors: "X. Chen, H. Fan, R. Girshick, K. He — Improved Baselines with Momentum Contrastive Learning",
  idea: "把 SimCLR 的工程改良搬進 MoCo，用更少資源逼近其表現。",
  mechanism: ["2 層 MLP 投影頭（取代 v1 的線性層）", "加入 Gaussian blur 強增強 + cosine LR", "其餘架構與 v1 完全相同"],
  contribution: "MLP 頭 + 強增強 = 小資源逼近 SimCLR。",
  diagram: (s, o) => siamese(s, o, { top: "f_q\n(梯度)", bot: "f_k", botNote: "EMA 動量", proj: true, loss: "InfoNCE", extra: "Queue\n(歷史 key 負樣本)" }),
  loss: "−log exp(q·k₊/τ) / Σ_{queue} exp(q·k/τ)", lossNote: "同 v1，新增 MLP 投影頭 g",
});

// --- SimCLR v1 ---
methodSlide({
  era: A2("ERA 2 · 2020"), name: "SimCLR v1", demo: "demo_assets/gifs/simclr_v1.gif", venue: "ICML 2020",
  authors: "T. Chen, S. Kornblith, M. Norouzi, G. Hinton — A Simple Framework for Contrastive Learning of Visual Representations",
  idea: "不要 queue/bank，直接用超大 batch，batch 內互當負樣本。",
  mechanism: ["共享 backbone + MLP 投影頭", "對稱 NT-Xent；loss 算在 z、評估用 h", "強增強（color jitter s=1.0 + blur）是關鍵"],
  contribution: "證明「簡單框架 + 強增強」即可；痛點：需大 batch。",
  diagram: (s, o) => siamese(s, o, { top: "編碼器\nf", bot: "編碼器\nf", botNote: "共享權重", proj: true, loss: "NT-Xent" }),
  loss: "−log exp(sim(zᵢ,zⱼ)/τ) / Σ_{k≠i} exp(·)", lossNote: "NT-Xent（in-batch 對稱）",
});

// --- SimCLR v2 ---
methodSlide({
  era: A2("ERA 2 · 2020"), name: "SimCLR v2", demo: "demo_assets/gifs/simclr_v2.gif", venue: "NeurIPS 2020",
  authors: "T. Chen, S. Kornblith, K. Swersky, M. Norouzi, G. Hinton — Big Self-Supervised Models are Strong Semi-Supervised Learners",
  idea: "更深的投影頭 + 更大模型 → 強半監督學習者。",
  mechanism: ["3 層投影頭（僅預訓練階段使用）", "更大、更深的 backbone", "蒸餾到小模型完成半監督"],
  contribution: "大型 SSL 模型是強半監督學習者（pretrain→distill）。",
  diagram: (s, o) => siamese(s, o, { top: "編碼器\nf", bot: "編碼器\nf", botNote: "共享權重", proj: true, loss: "NT-Xent" }),
  loss: "−log exp(sim(zᵢ,zⱼ)/τ) / Σ_{k≠i} exp(·)", lossNote: "同 v1，投影頭加深為 3 層",
});

// --- SwAV ---
methodSlide({
  era: A2("ERA 2 · 2020"), name: "SwAV", demo: "demo_assets/gifs/swav.gif", venue: "NeurIPS 2020",
  authors: "M. Caron, I. Misra, J. Mairal, P. Goyal, P. Bojanowski, A. Joulin — Unsupervised Learning of Visual Features by Contrasting Cluster Assignments",
  idea: "不再兩兩比較，改成線上分群（prototypes）。",
  mechanism: ["multi-crop：2 大 + N 小裁切", "Sinkhorn-Knopp 最佳傳輸求軟分配 code", "swapped prediction：互相預測對方的 code"],
  contribution: "免成對負樣本的對比；多裁切大幅提升效率。",
  diagram: (s, o) => siamese(s, o, { top: "編碼器\nf", bot: "編碼器\nf", botNote: "共享權重", proj: true, loss: "Swapped\nCE", extra: "Prototypes C\n+ Sinkhorn" }),
  loss: "−Σ qₜ·log pₛ − qₛ·log pₜ", lossNote: "code q 由 Sinkhorn-Knopp 求得",
});

// --- InfoMin ---
methodSlide({
  era: A2("ERA 2 · 2020"), name: "InfoMin", demo: "demo_assets/gifs/infomin.gif", venue: "NeurIPS 2020",
  authors: "Y. Tian, C. Sun, B. Poole, D. Krishnan, C. Schmid, P. Isola — What Makes for Good Views for Contrastive Learning?",
  idea: "退一步問：什麼才是「好的」增強視角？",
  mechanism: ["沿用 SimCLR 的 backbone 與 NT-Xent loss", "更激進增強（s=1.5、grayscale 0.4、去 blur）", "去除紋理／顏色等捷徑相關性"],
  contribution: "minimal sufficient 視角原則：共享任務資訊、其餘越少越好。",
  diagram: (s, o) => siamese(s, o, { top: "編碼器\nf", bot: "編碼器\nf", botNote: "共享權重", proj: true, loss: "NT-Xent" }),
  loss: "min I(v₁;v₂)  +  NT-Xent(z₁,z₂)", lossNote: "loss 同 SimCLR，重點在視角設計",
});

// ============================================================
// ACT 3
// ============================================================
actDivider("3", E3, "2020 – 2021", "異端：\n我們真的需要負樣本嗎？", NOTES.act3);

// --- BYOL ---
methodSlide({
  era: A3("ERA 3 · 2020"), name: "BYOL", demo: "demo_assets/gifs/byol.gif", venue: "NeurIPS 2020",
  authors: "J.-B. Grill, F. Strub, F. Altché, et al. (DeepMind) — Bootstrap Your Own Latent",
  idea: "完全不用負樣本也能學 —— 靠 predictor 不對稱 + EMA target。",
  mechanism: ["online 分支多一個 predictor（製造不對稱）", "target 分支 = online 的 EMA（動量 0.996→1）", "預測 target 表徵，target 端 stop-gradient"],
  contribution: "證明 predictor 不對稱即可防坍塌，震撼全場。",
  diagram: (s, o) => siamese(s, o, { top: "online\nf_θ", bot: "target\nf_ξ", botNote: "EMA", proj: true, pred: true, sgBot: true, loss: "MSE" }),
  loss: "‖ q(z₁) − sg(z₂′) ‖²  (對稱化)", lossNote: "predictor 不對稱 + sg(target)",
});

// --- SimSiam ---
methodSlide({
  era: A3("ERA 3 · 2021"), name: "SimSiam", demo: "demo_assets/gifs/simsiam.gif", venue: "CVPR 2021",
  authors: "X. Chen, K. He — Exploring Simple Siamese Representation Learning",
  idea: "連 EMA 都不要，只靠一個 stop-gradient 就不坍塌。",
  mechanism: ["共享 backbone（無動量編碼器、無 queue）", "online 分支 predictor + target 端 stop-gradient", "負 cosine 相似度損失"],
  contribution: "最小化證明：stop-gradient 才是防坍塌的關鍵。",
  diagram: (s, o) => siamese(s, o, { top: "編碼器\nf", bot: "編碼器\nf", botNote: "共享權重", proj: true, pred: true, sgBot: true, loss: "−cos" }),
  loss: "½D(p₁, sg z₂) + ½D(p₂, sg z₁)", lossNote: "stop-grad 是唯一防坍塌機制",
});

// --- Barlow Twins ---
methodSlide({
  era: A3("ERA 3 · 2021"), name: "Barlow Twins", demo: "demo_assets/gifs/barlow_twins.gif", venue: "ICML 2021",
  authors: "J. Zbontar, L. Jing, I. Misra, Y. LeCun, S. Deny — Self-Supervised Learning via Redundancy Reduction",
  idea: "換個哲學：讓兩視角嵌入的互相關矩陣趨近單位矩陣。",
  mechanism: ["計算兩視角嵌入的互相關矩陣 C", "對角→1（不變性）、非對角→0（去冗餘）", "高維投影頭（8192）效果最好"],
  contribution: "無負樣本／EMA／predictor，全靠 loss 作用於 C。",
  diagram: (s, o) => siamese(s, o, { top: "編碼器\nf", bot: "編碼器\nf", botNote: "共享權重", proj: true, loss: "Cross-Corr\nC → I" }),
  loss: "Σᵢ(1−Cᵢᵢ)² + λ Σ_{i≠j} Cᵢⱼ²", lossNote: "互相關矩陣 → 單位矩陣（去冗餘）",
});

// ============================================================
// ACT 4
// ============================================================
actDivider("4", E4, "2021 → 今天", "架構遷移：\n換上 Transformer", NOTES.act4);

// --- MoCo v3 ---
methodSlide({
  era: A4("ERA 4 · 2021"), name: "MoCo v3", demo: "demo_assets/gifs/moco_v3.gif", venue: "ICCV 2021",
  authors: "X. Chen, S. Xie, K. He — An Empirical Study of Training Self-Supervised Vision Transformers",
  idea: "把對比學習搬上 ViT，並找出讓它穩定訓練的配方。",
  mechanism: ["凍結 patch embedding 投影（最關鍵的穩定性修正）", "對稱 in-batch InfoNCE（丟掉 queue）", "AdamW + cosine LR（非 SGD/LARS），m=0.99"],
  contribution: "ViT 對比訓練的穩定配方；大 batch 下毋需 queue。",
  diagram: (s, o) => siamese(s, o, { top: "ViT f_q\n(梯度)", bot: "ViT f_k", botNote: "EMA 動量", proj: true, loss: "對稱\nInfoNCE" }),
  loss: "ctr(q₁,k₂) + ctr(q₂,k₁)", lossNote: "對稱 InfoNCE，無 queue",
});

// --- DINO ---
methodSlide({
  era: A4("ERA 4 · 2021"), name: "DINO", demo: "demo_assets/gifs/dino.gif", venue: "ICCV 2021",
  authors: "M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, A. Joulin — Emerging Properties in Self-Supervised ViTs",
  idea: "student–teacher 自蒸餾，無對比負樣本。",
  mechanism: ["teacher = student 的 EMA（只看 global crop）", "teacher 輸出做 centering + sharpening", "cross-entropy：student 預測 teacher 分布"],
  contribution: "attention map 浮現語意分割；防坍塌靠 centering+sharpening。",
  diagram: (s, o) => siamese(s, o, { top: "student\nf_s", bot: "teacher\nf_ξ", botNote: "EMA", proj: true, sgBot: true, loss: "Cross-\nEntropy", extra: "centering\n+ sharpening" }),
  loss: "−Σ Pₜ log Pₛ , Pₜ=σ((g−C)/τₜ)", lossNote: "σ=softmax；centering + sharpening 防坍塌",
});

// --- DINOv2 ---
methodSlide({
  era: A4("ERA 4 · 2023"), name: "DINOv2", demo: "demo_assets/methods/dinov2.png", venue: "TMLR 2024",
  authors: "M. Oquab, T. Darcet, T. Moutakanni, et al. (Meta AI) — Learning Robust Visual Features without Supervision",
  idea: "把自蒸餾規模化，做出通用視覺「基礎模型」。",
  mechanism: ["DINO（影像級）+ iBOT（patch 級）自蒸餾", "LVD-142M 大規模精選資料 + 訓練技巧", "本教學僅提供特徵抽取／微調 demo"],
  contribution: "免微調即強的通用特徵，foundation model 的前身。",
  diagram: (s, o) => siamese(s, o, { top: "student\nf_s", bot: "teacher\nf_ξ", botNote: "EMA", proj: true, sgBot: true, loss: "DINO\n+ iBOT", extra: "iBOT\n(patch 級遮罩)" }),
  loss: "ℒ_DINO + ℒ_iBOT + 正則項", lossNote: "影像級 + patch 級自蒸餾，規模化",
});

// --- DINOv3 (2025 續章) ---
methodSlide({
  era: A4("ERA 4 · 2025"), name: "DINOv3", demo: "demo_assets/methods/dinov3.png", venue: "arXiv 2025",
  authors: "O. Siméoni, H. V. Vo, et al. (Meta AI) — DINOv3",
  idea: "同一條 DINO 路線再往上推到極限規模，並用 Gram anchoring 穩住 dense 特徵。",
  mechanism: ["沿用 DINO + iBOT 自蒸餾，放大到 7B 參數 / 17 億張影像", "Gram anchoring：訓練後期穩住 patch 級 dense 特徵、防退化", "再從巨模型蒸餾出 ViT-S/B/L 等好用的小模型"],
  contribution: "凍結 backbone 的 dense 特徵達 SOTA；免微調就打贏專用模型。",
  diagram: (s, o) => siamese(s, o, { top: "student\nf_s", bot: "teacher\nf_ξ", botNote: "EMA", proj: true, sgBot: true, loss: "DINO+iBOT\n+Gram", extra: "Gram\nanchoring" }),
  loss: "ℒ_DINO + ℒ_iBOT + ℒ_Gram", lossNote: "自蒸餾 + Gram anchoring 穩住 dense 特徵",
});

// 15. Comparison / collapse payoff table
{
  const s = pres.addSlide();
  contentHeader(s, "收束", "0E7490", "回頭看：每一代怎麼防坍塌？");
  const hdr = (t) => ({ text: t, options: { fill: { color: NAVY }, color: WHITE, bold: true, fontFace: FACE, fontSize: 14, align: "center", valign: "middle" } });
  const rows = [
    [hdr("時代"), hdr("代表方法"), hdr("用負樣本？"), hdr("防坍塌機制")],
    ["Era 1 · 2018", "Instance Discrimination", { text: "是（memory bank）", options: { color: "2A8F5F" } }, "用大量負樣本把表徵推開"],
    ["Era 2 · 2020", "MoCo / SimCLR", { text: "是（queue / batch）", options: { color: "2A8F5F" } }, "in-batch / queue 負樣本"],
    ["Era 2 · 2020", "SwAV", { text: "否（用 prototype）", options: { color: E3, bold: true } }, "線上分群 + Sinkhorn 均勻分配"],
    ["Era 3 · 2021", "BYOL / SimSiam", { text: "否", options: { color: E3, bold: true } }, "EMA / stop-gradient 打破對稱"],
    ["Era 3 · 2021", "Barlow Twins", { text: "否", options: { color: E3, bold: true } }, "冗餘消除（特徵去相關）"],
    ["Era 4 · 2021+", "DINO", { text: "否", options: { color: E3, bold: true } }, "centering + sharpening"],
  ];
  s.addTable(rows, {
    x: 0.7, y: 2.3, w: 11.93, colW: [2.1, 3.5, 2.73, 3.6],
    rowH: [0.5, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52],
    border: { pt: 0.5, color: "D5DCEA" }, align: "left", valign: "middle",
    fontFace: FACE, fontSize: 13, color: INK, margin: [0, 0.12, 0, 0.12],
    fill: { color: CARD },
  });
  s.addText([
    { text: "Punchline：14 個方法，其實都在回答同一個問題 —— ", options: { color: INK } },
    { text: "「沒有標籤，怎麼不讓表徵坍塌？」", options: { bold: true, color: "0E7490" } },
  ], { x: 0.7, y: 6.35, w: 11.9, h: 0.5, align: "center", fontFace: FACE, fontSize: 15, italic: true, margin: 0 });
  s.addNotes(NOTES.collapseTable);
}

// 15b. Epoch progression (appendix) — clusters emerging over training, all 7 self-trained methods
function progSlide(suffix, rows, notes) {
  const s = pres.addSlide();
  contentHeader(s, "訓練過程", "0E7490", "分群如何隨 epoch 浮現？" + suffix);
  s.addText("自訓 SSL（CIFAR-10，ResNet-18 / ViT-S，2×H100，200 epoch）：特徵從 epoch 0 的「一團」逐步分化成清楚的類別群落。", {
    x: 0.7, y: 1.92, w: 11.9, h: 0.4, fontFace: FACE, fontSize: 12.5, color: MUTE, margin: 0 });
  const n = rows.length;
  const top = 2.34, bottom = 6.84, gap = 0.12;
  const sh = (bottom - top - (n - 1) * gap) / n;
  const sw = 9.6, sx = 3.0;
  let yy = top;
  rows.forEach((r) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: yy, w: 2.1, h: sh, fill: { color: NAVY } });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: yy, w: 0.09, h: sh, fill: { color: r.c } });
    s.addText(r.label, { x: 0.95, y: yy + sh / 2 - 0.32, w: 1.95, h: 0.34, fontFace: FACE, fontSize: 14, bold: true, color: WHITE, margin: 0 });
    s.addText(r.sub, { x: 0.95, y: yy + sh / 2 + 0.03, w: 1.95, h: 0.3, fontFace: FACE, fontSize: 10, color: ICE, margin: 0 });
    const p = "demo_assets/progression/" + r.key + ".png";
    if (fs.existsSync(p)) s.addImage({ path: p, x: sx, y: yy, w: sw, h: sh, sizing: { type: "contain", w: sw, h: sh } });
    yy += sh + gap;
  });
  s.addText("← 隨機初始化（epoch 0）　　訓練越久，同類越聚、異類越分　　充分訓練（epoch 200）→", {
    x: 3.0, y: 6.9, w: 9.6, h: 0.3, align: "center", fontFace: FACE, fontSize: 10, italic: true, color: MUTE, margin: 0 });
  if (notes) s.addNotes(notes);
}
progSlide("（1/3 · Era 1–2）", [
  { key: "instance_discrimination", label: "Instance Discrim.", sub: "memory bank", c: E1 },
  { key: "invariant_spread", label: "Invariant Spread", sub: "in-batch softmax", c: E1 },
  { key: "simclr_v1", label: "SimCLR v1", sub: "in-batch 對比", c: E2 },
  { key: "simclr_v2", label: "SimCLR v2", sub: "深 3 層投影頭", c: E2 },
], NOTES.prog1);
progSlide("（2/3 · Era 2–3）", [
  { key: "infomin", label: "InfoMin", sub: "視角設計", c: E2 },
  { key: "byol", label: "BYOL", sub: "predictor + EMA", c: E3 },
  { key: "simsiam", label: "SimSiam", sub: "stop-gradient", c: E3 },
], NOTES.prog2);
progSlide("（3/3 · 官方權重方法自訓補完）", [
  { key: "moco_v1", label: "MoCo v1", sub: "queue + 動量", c: E2 },
  { key: "moco_v2", label: "MoCo v2", sub: "MLP 頭 + 強增強", c: E2 },
  { key: "swav", label: "SwAV", sub: "prototype + Sinkhorn", c: E2 },
  { key: "barlow_twins", label: "Barlow Twins", sub: "去冗餘", c: E3 },
  { key: "moco_v3", label: "MoCo v3", sub: "ViT + 凍結 patch", c: E4 },
  { key: "dino", label: "DINO", sub: "自蒸餾 + centering", c: E4 },
], NOTES.prog3);

// 16. Live demo
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.28, h: PH, fill: { color: ACCENT } });
  s.addText("LIVE DEMO", { x: 0.9, y: 0.5, w: 6, h: 0.45, fontFace: MONO, fontSize: 15, bold: true, color: ACCENT, charSpacing: 4, margin: 0 });
  s.addText("眼見為憑：特徵空間從「一團」變「分群」", { x: 0.9, y: 0.95, w: 11.5, h: 0.65, fontFace: FACE, fontSize: 26, bold: true, color: WHITE, margin: 0 });

  // command strip
  s.addShape(pres.shapes.RECTANGLE, { x: 0.9, y: 1.75, w: 11.5, h: 0.98, fill: { color: "060B16" }, line: { color: NAVY3, width: 1 }, shadow: makeShadow() });
  s.addText([
    { text: "$ python train.py --config configs/simclr_v1_resnet18.yaml --data-dir data/cifar10", options: { breakLine: true, color: "7CF0A0" } },
    { text: "$ python eval/umap_vis.py configs/simclr_v1_resnet18.yaml --ckpt checkpoints/last.ckpt", options: { color: "7CF0A0" } },
  ], { x: 1.15, y: 1.9, w: 11, h: 0.7, fontFace: MONO, fontSize: 12.5, lineSpacingMultiple: 1.3, margin: 0 });

  // before / after UMAP images
  const imgY = 3.25, imgS = 2.65, labY = 2.88;
  const beforeX = 3.0, afterX = 7.68; // arrow sits in the ~0.95" gap between
  s.addText("訓練前 · 隨機初始化", { x: beforeX, y: labY, w: imgS, h: 0.34, align: "center", fontFace: FACE, fontSize: 14, bold: true, color: "FF8FA8", margin: 0 });
  s.addText("訓練後 · 預訓練特徵", { x: afterX, y: labY, w: imgS, h: 0.34, align: "center", fontFace: FACE, fontSize: 14, bold: true, color: "7CF0A0", margin: 0 });
  s.addImage({ path: "demo_assets/umap_before.png", x: beforeX, y: imgY, w: imgS, h: imgS, sizing: { type: "contain", w: imgS, h: imgS } });
  s.addImage({ path: "demo_assets/umap_after.png", x: afterX, y: imgY, w: imgS, h: imgS, sizing: { type: "contain", w: imgS, h: imgS } });
  s.addShape(pres.shapes.RECTANGLE, { x: beforeX, y: imgY, w: imgS, h: imgS, fill: { color: "FFFFFF", transparency: 100 }, line: { color: "C0395A", width: 1.5 } });
  s.addShape(pres.shapes.RECTANGLE, { x: afterX, y: imgY, w: imgS, h: imgS, fill: { color: "FFFFFF", transparency: 100 }, line: { color: "2A8F5F", width: 1.5 } });
  s.addText("→", { x: beforeX + imgS, y: imgY + imgS / 2 - 0.4, w: afterX - beforeX - imgS, h: 0.8, align: "center", valign: "middle", fontFace: FACE, fontSize: 40, bold: true, color: ACCENT, margin: 0 });

  s.addText([
    { text: "※ 教學示意：以「ImageNet 預訓練 backbone」代理訓練後特徵（CIFAR-10）。", options: { breakLine: true } },
    { text: "  換上你自己訓練好的 SSL checkpoint 即可重現真實結果。", options: {} },
  ], { x: 0.9, y: 6.35, w: 11.5, h: 0.8, align: "center", fontFace: FACE, fontSize: 12, italic: true, color: MUTE, margin: 0, lineSpacingMultiple: 1.15 });
  s.addNotes(NOTES.liveDemo);
}

// 17. Closing
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape(pres.shapes.OVAL, { x: -2.5, y: 4.5, w: 6, h: 6, fill: { color: NAVY3, transparency: 45 } });
  s.addText("演化史，一句話總結", { x: 0.9, y: 0.85, w: 11.5, h: 0.6, fontFace: FACE, fontSize: 26, bold: true, color: ACCENT, margin: 0 });
  s.addText([
    { text: "5 年來，對比學習不斷「拿掉依賴」——\n從 memory bank，到負樣本，最後連", options: { color: WHITE } },
    { text: "架構", options: { color: ACCENT, bold: true } },
    { text: "都換新。", options: { color: WHITE } },
  ], { x: 0.9, y: 1.6, w: 11.5, h: 1.5, fontFace: FACE, fontSize: 28, bold: true, lineSpacingMultiple: 1.15, margin: 0 });

  const takeaways = [
    "所有方法共用 InfoNCE 直覺：拉近正樣本、推開負樣本",
    "真正的難題是「防坍塌」，負樣本只是其中一條路",
    "趨勢：更少假設、更大規模 → 走向 foundation model",
  ];
  let yy = 3.5;
  takeaways.forEach((t, i) => {
    s.addShape(pres.shapes.OVAL, { x: 0.95, y: yy + 0.02, w: 0.42, h: 0.42, fill: { color: ACCENT } });
    s.addText(String(i + 1), { x: 0.95, y: yy + 0.02, w: 0.42, h: 0.42, align: "center", valign: "middle", fontFace: FACE, fontSize: 16, bold: true, color: NAVY, margin: 0 });
    s.addText(t, { x: 1.6, y: yy, w: 11, h: 0.46, valign: "middle", fontFace: FACE, fontSize: 16.5, color: ICE, margin: 0 });
    yy += 0.62;
  });

  s.addShape(pres.shapes.LINE, { x: 0.9, y: 5.75, w: 11.5, h: 0, line: { color: NAVY3, width: 1 } });
  s.addText([
    { text: "開源教學專案 · 14 種方法統一實作 · 每個 loss 都能獨立閱讀", options: { color: ICE } },
  ], { x: 0.9, y: 5.95, w: 11.5, h: 0.5, fontFace: FACE, fontSize: 14, margin: 0 });
  s.addText("謝謝聆聽 — 歡迎提問", { x: 0.9, y: 6.5, w: 11.5, h: 0.6, fontFace: FACE, fontSize: 22, bold: true, color: WHITE, margin: 0 });
  s.addNotes(NOTES.closing);
}

pres.writeFile({ fileName: "contrastive_learning_evolution.pptx" }).then((f) => console.log("WROTE", f));
