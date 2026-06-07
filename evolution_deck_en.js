// Contrastive Learning — Evolution History deck (ENGLISH edition)
// English mirror of evolution_deck.js. Same layout/coords/asset paths; only the
// visible text + speaker notes are in English, fonts are cross-platform, and the
// output filename/title differ. Run: node evolution_deck_en.js
const pptxgen = require("pptxgenjs");
const fs = require("fs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
const PW = 13.33, PH = 7.5;
pres.author = "Yi-Ting Li";
pres.title = "The Evolution of Contrastive Learning";

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

const FACE = "Arial";        // cross-platform sans for an all-English deck
const MONO = "Courier New";  // cross-platform monospace

const makeShadow = () => ({ type: "outer", color: "000000", blur: 9, offset: 3, angle: 135, opacity: 0.18 });
const softShadow = () => ({ type: "outer", color: "0E1626", blur: 7, offset: 2, angle: 135, opacity: 0.12 });

// ---------- speaker notes (concise talking-point bullets, one per slide) ----------
// Full verbatim script lives in docs/TALK_SCRIPT.md; these are quick glance bullets.
const B = (...pts) => pts.map((p) => "• " + p).join("\n");
const NOTES = {
  cover: B(
    "Opener: a million photos, not one label — how does a model learn cats vs dogs on its own?",
    "Topic: contrastive learning; next ~20 min = its 5-year evolution story"),
  bigPicture: B(
    "Glasses for the whole talk: 5 years doing one thing — removing crutches",
    "Four steps: 2018 drop labels → 2020 drop the memory bank → 2021 drop negatives → 2021+ swap in Transformer",
    "Refrain: for each method ask 'which crutch did it remove?'"),
  coreIntuition: B(
    "Core intuition: two augments of one image = pull together; different images = push apart",
    "Learn what stays the same under augmentation = semantics",
    "Trap: pull-only with no push → everything collapses to one point = collapse (the villain)"),
  infonce: B(
    "Write pull/push as math = InfoNCE",
    "Three spots: numerator = positives (large), denominator = negatives (small), tau = temperature",
    "Hook: numerator pulls, denominator pushes",
    "All 14 methods are variations on (or escapes from) this one equation"),
  evoMap: B(
    "One map, four eras — we follow it",
    "Arrows = who solved whose bottleneck, not just chronology",
    "A causal evolution chain, not scattered papers"),
  act1: B(
    "Back to 2018: piles of images, no labels — what is the model to learn?",
    "Labels were the biggest crutch, now pulled away",
    "Act 1: how people conjured a task out of thin air"),
  instance_discrimination: B(
    "Idea: every image is its own class (a million images = a million classes)",
    "Architecture: one encoder + a memory bank (stores all features as negatives)",
    "Loss: softmax is uncomputable → NCE approximates it by sampling",
    "Wall: bank features go stale (the next step fixes this)"),
  invariant_spread: B(
    "Drop the bank, stay in-batch: same image pull (invariant), different images push (spread)",
    "Architecture: two shared-weight branches; loss: symmetric InfoNCE",
    "Diff vs previous: drops the memory bank for in-batch negatives",
    "It is the direct ancestor of SimCLR"),
  act2: B(
    "Act 2 = the big fork: where do negatives come from?",
    "Two camps clash: store them in a queue, or blow up the batch"),
  moco_v1: B(
    "Kaiming He's team: a FIFO queue stores past features as negatives",
    "Negatives decoupled from batch size (tiny batch, still tens of thousands)",
    "Key move: momentum encoder (EMA catches up slowly) → queue stays consistent",
    "Loss is still InfoNCE; only negatives now come from the queue"),
  moco_v2: B(
    "Same loss/architecture as v1, pure engineering upgrade",
    "Borrows three things from SimCLR: MLP head, Gaussian blur, cosine LR",
    "Result: near-SimCLR on a small budget"),
  simclr_v1: B(
    "Hinton's team, opposite philosophy: no queue, just a huge batch",
    "Architecture: two shared branches + MLP head; loss: NT-Xent (= InfoNCE with L2-normalized features)",
    "Key: strong augmentation (color jitter + Gaussian blur) is the soul",
    "Pain point: needs a TPU-scale batch"),
  simclr_v2: B(
    "Same path, leveled up — loss unchanged, theme is scale",
    "Changes: 3-layer projection head + bigger backbone",
    "Proves big SSL models are strong semi-supervised learners (pretrain → tiny labels → distill)"),
  swav: B(
    "No more pairwise comparison — online clustering with prototypes",
    "Loss: swapped prediction (cross-entropy) — one view predicts the other's assignment",
    "Sinkhorn forces assignments to spread evenly → itself prevents collapse",
    "Multi-crop = free extra views; key: no pairwise negatives needed"),
  infomin: B(
    "Architecture/loss reuse SimCLR — only the views change",
    "Asks: what makes a good view? Answer: minimal sufficient (keep semantics, cut shortcuts)",
    "More aggressive augmentation destroys color/texture shortcuts",
    "Closes the act: SwAV hinted negatives are not the only answer → next act drops them entirely"),
  act3: B(
    "The wildest act: remove the last crutch — negatives",
    "Negatives = the push force; pull-only is most likely to collapse to a point",
    "Old consensus: no negatives = guaranteed collapse (about to be debunked)"),
  byol: B(
    "No negatives, not even one",
    "Architecture: online + target branches; online adds a predictor (creates asymmetry)",
    "Target = EMA clone of online + stop-gradient",
    "Loss: predictor predicts the target's features via MSE (not pushing apart)",
    "Prevents collapse via asymmetry, not pushing — shocked the field"),
  simsiam: B(
    "Cut further: drop even the EMA, both branches share one network",
    "Keep only predictor + stop-gradient; loss = negative cosine similarity",
    "Minimal proof: stop-gradient is the real key to preventing collapse"),
  barlow_twins: B(
    "New philosophy: not pull/push — look at the cross-correlation matrix of the two views",
    "Goal: drive it to the identity (diagonal 1 = invariance, off-diagonal 0 = decorrelation)",
    "Wide projection head (8192); no negatives/EMA/predictor — all in the loss",
    "Thread across the 3 papers: the key is breaking symmetry, not pushing apart"),
  act4: B(
    "Crutches nearly all gone, but the engine was always a CNN",
    "Meanwhile the ViT arrives; what if we swap the engine to a Transformer?",
    "Final act: taming the new engine, all the way to foundation models"),
  moco_v3: B(
    "Architecture = MoCo on a ViT (backbone ResNet → Transformer)",
    "Problem: very unstable, loss spikes",
    "Culprit = the first patch-embedding layer; fix: freeze it and it stabilizes",
    "Also drops the queue; loss back to symmetric in-batch InfoNCE"),
  dino: B(
    "Same ViT as MoCo v3, but goes further: no negatives at all",
    "Self-distillation: student learns teacher (teacher = EMA of student); loss = cross-entropy",
    "Anti-collapse: centering (recenter) + sharpening on the teacher",
    "Stunner: attention map frames whole objects (emergent segmentation, no labels)"),
  dinov2: B(
    "No new trick — does the harder thing: scale the right method up",
    "Architecture/loss = DINO + iBOT (patch masking, predicts features not pixels)",
    "Feed LVD-142M (100M+ curated) → a general visual foundation model",
    "Features need no training, plug into classification/segmentation/depth"),
  dinov3: B(
    "2025 coda, same line as DINOv2 (architecture/loss almost identical)",
    "Diff 1: scale explodes again (7B params / 1.7B images)",
    "Diff 2: Gram anchoring keeps dense features stable, no degradation",
    "Another leap on dense tasks; the UMAP on the right = official weights on CIFAR-10",
    "DINO line v1 → v3 = one idea scaled up and up"),
  collapseTable: B(
    "Callback to the opening seed: collapse is the real fear",
    "Table: per era, do they use negatives + what prevents collapse",
    "Mechanisms all differ (negatives / Sinkhorn / EMA / stop-grad / decorrelation / centering)",
    "Punchline: all 14 answer one question — without labels, how to avoid collapse"),
  prog1: B(
    "From theory to real runs: 2xH100, CIFAR-10, 200 epochs, a UMAP per epoch",
    "Era 1–2 four: epoch 0 is one blob (collapse start) → gradually clusters",
    "SimCLR clusters faster/cleaner (power of in-batch positives + negatives)",
    "Purely from 'same image alike, different unalike' — zero labels"),
  prog2: B(
    "Era 2–3: InfoMin, BYOL, SimSiam",
    "Point: BYOL/SimSiam use no negatives — in theory the most collapse-prone",
    "Yet they cluster cleanly, never collapse to a point → seeing is believing",
    "stop-gradient and predictor+EMA really do prevent collapse"),
  prog3: B(
    "Completion: the 6 official-weight methods, also trained from scratch",
    "MoCo v1/v2, SwAV, Barlow, MoCo v3, DINO",
    "Even the two ViTs (most prone to diverge) cluster cleanly",
    "All 13 self-trained methods now proven on screen"),
  liveDemo: B(
    "Core comparison: left = pre-training fuzz ball (what collapse looks like), right = clean clusters",
    "No labels used at any point",
    "Biggest takeaway: clustering quality mainly comes down to enough training",
    "Method difference isn't 'can it cluster' but 'at what cost / which path to prevent collapse'",
    "Play the GIF; leave a few seconds to watch points spread and settle"),
  closing: B(
    "One takeaway: for 5 years contrastive learning kept removing dependencies",
    "Three points: (1) shared InfoNCE intuition (2) the real challenge is preventing collapse, negatives are just one path (3) trend: fewer assumptions, larger scale → foundation models",
    "Open-source teaching project: 14 methods, unified + reproducible — clone it",
    "Thanks; into Q&A"),
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
  slide.addText(title, { x: 0.68, y: 1.05, w: 12, h: 0.85, fontFace: FACE, fontSize: 31, bold: true, color: INK, margin: 0 });
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
  s.addText(question, { x: 5.0, y: 3.05, w: 7.6, h: 1.9, fontFace: FACE, fontSize: 29, bold: true, color: WHITE, margin: 0, lineSpacingMultiple: 1.1 });
  timelineRibbon(s, 6.7);
  if (notes) s.addNotes(notes);
  return s;
}

// ---------- per-paper slide template + architecture diagram ----------
function abox(s, x, y, w, h, text, fill, tcol, bcol) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.05, fill: { color: fill }, line: { color: bcol || fill, width: 1 }, shadow: softShadow() });
  s.addText(text, { x, y, w, h, align: "center", valign: "middle", fontFace: FACE, fontSize: 9, bold: true, color: tcol, margin: 0, lineSpacingMultiple: 0.95 });
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
  abox(s, ix, midY - 0.3, iw, 0.6, "Input\nx", NAVY3, WHITE);
  const ex = o.x + 1.45, ew = 1.3;

  if (cfg.single) {
    hArrow(s, ix + iw, midY, ex - (ix + iw));
    abox(s, ex, midY - rowH / 2, ew, rowH, cfg.top, BOXF, INK, BOXB);
    let cx = ex + ew;
    if (cfg.proj) { hArrow(s, cx, midY, 0.5); abox(s, cx + 0.5, midY - rowH / 2, 0.85, rowH, "Proj\ng", BOXF, INK, BOXB); cx += 0.5 + 0.85; }
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
    abox(s, cx + 0.45, topY, 0.8, rowH, "Proj\ng", BOXF, INK, BOXB);
    abox(s, cx + 0.45, botY, 0.8, rowH, "Proj\ng", BOXF, INK, BOXB);
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
  s.addText(spec.name, { x: 0.68, y: 0.97, w: 9.0, h: 0.66, fontFace: FACE, fontSize: 26, bold: true, color: INK, margin: 0 });
  s.addText(spec.venue, { x: 9.5, y: 1.05, w: 3.1, h: 0.5, align: "right", valign: "middle", fontFace: MONO, fontSize: 13, bold: true, color: spec.era.color, margin: 0 });
  s.addText("📄 " + spec.authors, { x: 0.72, y: 1.62, w: 11.85, h: 0.3, fontFace: FACE, fontSize: 9.5, italic: true, color: MUTE, margin: 0 });
  const onEra = (spec.era.color === E1 || spec.era.color === E2) ? NAVY : WHITE;

  // diagram card
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 2.04, w: 11.93, h: 2.02, fill: { color: CARD }, line: { color: "DCE3F0", width: 1 }, shadow: softShadow() });
  s.addText("Architecture", { x: 0.88, y: 2.12, w: 3, h: 0.26, fontFace: FACE, fontSize: 10, bold: true, color: MUTE, charSpacing: 1, margin: 0 });
  spec.diagram(s, { x: 1.05, y: 2.46, w: 11.3, h: 1.5 });

  // --- bottom: three columns (text · loss/contribution · UMAP demo) ---
  // col1: text card
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 4.22, w: 5.35, h: 2.73, fill: { color: CARD }, line: { color: "DCE3F0", width: 1 }, shadow: softShadow() });
  s.addText("Core idea", { x: 0.92, y: 4.34, w: 4.95, h: 0.28, fontFace: FACE, fontSize: 12, bold: true, color: spec.era.color, margin: 0 });
  s.addText(spec.idea, { x: 0.92, y: 4.62, w: 4.95, h: 0.74, fontFace: FACE, fontSize: 10.5, color: INK, margin: 0, lineSpacingMultiple: 1.03 });
  s.addText("Key mechanisms", { x: 0.92, y: 5.42, w: 4.95, h: 0.28, fontFace: FACE, fontSize: 12, bold: true, color: spec.era.color, margin: 0 });
  s.addText(spec.mechanism.map((m, i) => ({ text: m, options: { bullet: { code: "2022" }, breakLine: i < spec.mechanism.length - 1, color: MUTE } })),
    { x: 0.92, y: 5.72, w: 4.95, h: 1.15, fontFace: FACE, fontSize: 9.5, paraSpaceAfter: 4, lineSpacingMultiple: 1.0, margin: 0 });

  // col2: loss panel + contribution
  s.addShape(pres.shapes.RECTANGLE, { x: 6.2, y: 4.22, w: 3.05, h: 1.5, fill: { color: NAVY }, shadow: makeShadow() });
  s.addText("LOSS", { x: 6.4, y: 4.31, w: 2.6, h: 0.24, fontFace: MONO, fontSize: 9, bold: true, color: ACCENT, charSpacing: 2, margin: 0 });
  s.addText(spec.loss, { x: 6.26, y: 4.55, w: 2.94, h: 0.74, align: "center", valign: "middle", fontFace: MONO, fontSize: 9.5, bold: true, color: WHITE, margin: 0, lineSpacingMultiple: 1.05 });
  s.addText(spec.lossNote, { x: 6.28, y: 5.33, w: 2.9, h: 0.32, align: "center", fontFace: FACE, fontSize: 8.5, italic: true, color: ICE, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: 6.2, y: 5.88, w: 3.05, h: 1.07, fill: { color: spec.era.color } });
  s.addText("Contribution / Key", { x: 6.4, y: 5.95, w: 2.7, h: 0.26, fontFace: FACE, fontSize: 10, bold: true, color: onEra, charSpacing: 1, margin: 0 });
  s.addText(spec.contribution, { x: 6.4, y: 6.22, w: 2.75, h: 0.68, fontFace: FACE, fontSize: 10, bold: true, color: onEra, margin: 0, lineSpacingMultiple: 1.03 });

  // col3: per-method UMAP demo
  const hasImg = spec.demo && fs.existsSync(spec.demo);
  const TRAINED = ["instance_discrimination", "invariant_spread", "simclr_v1", "simclr_v2", "infomin", "byol", "simsiam", "moco_v1", "moco_v2", "swav", "barlow_twins", "moco_v3", "dino"];
  const dkey = spec.demo ? spec.demo.split("/").pop().replace(/\.(png|gif)$/, "") : "";
  const isTrained = TRAINED.includes(dkey);
  const isGif = /\.gif$/.test(spec.demo || "");
  s.addShape(pres.shapes.RECTANGLE, { x: 9.4, y: 4.22, w: 3.23, h: 2.73, fill: { color: NAVY }, shadow: makeShadow() });
  s.addText(!hasImg ? "LIVE DEMO · pending" : (isTrained ? "LIVE DEMO · self-trained UMAP (animated)" : "LIVE DEMO · official-weights UMAP"), { x: 9.5, y: 4.31, w: 3.05, h: 0.26, align: "center", fontFace: MONO, fontSize: 7.5, bold: true, color: ACCENT, charSpacing: 1, margin: 0 });
  if (hasImg) {
    s.addImage({ path: spec.demo, x: 9.965, y: 4.62, w: 2.1, h: 2.1, sizing: { type: "contain", w: 2.1, h: 2.1 } });
    s.addText(isTrained ? (isGif ? "CIFAR-10 · self-trained 200ep · GIF (plays in PPT 365)" : "CIFAR-10 · self-trained 200 epochs (H100)") : "Official pretrained weights · CIFAR-10 features", { x: 9.45, y: 6.66, w: 3.15, h: 0.24, align: "center", fontFace: FACE, fontSize: 7, italic: true, color: MUTE, margin: 0 });
  } else {
    s.addText("⏳", { x: 9.4, y: 5.02, w: 3.23, h: 0.55, align: "center", fontFace: FACE, fontSize: 28, color: ICE, margin: 0 });
    s.addText("UMAP to be added after\nfull GPU training", { x: 9.5, y: 5.68, w: 3.05, h: 0.7, align: "center", valign: "top", fontFace: FACE, fontSize: 11, color: ICE, margin: 0, lineSpacingMultiple: 1.2 });
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

  s.addText("SELF-SUPERVISED LEARNING", {
    x: 0.95, y: 1.2, w: 11, h: 0.4, fontFace: MONO, fontSize: 14, bold: true, color: ACCENT, charSpacing: 3, margin: 0 });
  s.addText("The Evolution of\nContrastive Learning", {
    x: 0.9, y: 1.7, w: 11.7, h: 1.7, fontFace: FACE, fontSize: 46, bold: true, color: WHITE, margin: 0, lineSpacingMultiple: 1.0 });
  s.addText([
    { text: "From 2018 to today — a journey of continually ", options: {} },
    { text: "removing dependencies", options: { color: ACCENT, bold: true } },
    { text: ".", options: {} },
  ], { x: 0.95, y: 3.7, w: 11.5, h: 0.5, fontFace: FACE, fontSize: 18, color: ICE, margin: 0 });

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
  s.addText("The whole field in one sentence", { x: 0.9, y: 0.6, w: 11.5, h: 0.7, fontFace: FACE, fontSize: 28, bold: true, color: WHITE, margin: 0 });
  s.addText([
    { text: "Over five years, contrastive learning peeled away its reliance on ", options: {} },
    { text: "external crutches", options: { color: ACCENT, bold: true } },
    { text: ", one at a time.", options: {} },
  ], { x: 0.9, y: 1.45, w: 11.6, h: 0.9, fontFace: FACE, fontSize: 20, color: ICE, margin: 0 });

  const steps = [
    { yr: "2018", t: "Drop labels", s: "proxy task", c: E1 },
    { yr: "2020", t: "Drop the memory bank", s: "queue / big batch", c: E2 },
    { yr: "2021", t: "Drop negatives", s: "no-negative", c: E3 },
    { yr: "2021+", t: "Swap in Transformer", s: "foundation model", c: E4 },
  ];
  const bx = 0.9, bw = 2.72, gap = 0.28, by = 3.05, bh = 2.5;
  steps.forEach((st, i) => {
    const x = bx + i * (bw + gap);
    s.addShape(pres.shapes.RECTANGLE, { x, y: by, w: bw, h: bh, fill: { color: NAVY2 }, line: { color: st.c, width: 1.5 }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y: by, w: bw, h: 0.12, fill: { color: st.c } });
    s.addText(st.yr, { x: x + 0.25, y: by + 0.32, w: bw - 0.5, h: 0.5, fontFace: MONO, fontSize: 22, bold: true, color: st.c, margin: 0 });
    s.addText(st.t, { x: x + 0.25, y: by + 0.95, w: bw - 0.5, h: 0.85, fontFace: FACE, fontSize: 16, bold: true, color: WHITE, margin: 0, valign: "top" });
    s.addText(st.s, { x: x + 0.25, y: by + 1.95, w: bw - 0.5, h: 0.4, fontFace: MONO, fontSize: 11, color: ICE, margin: 0 });
    if (i < steps.length - 1) {
      s.addText("→", { x: x + bw - 0.06, y: by + 0.9, w: gap + 0.12, h: 0.6, align: "center", valign: "middle", fontFace: FACE, fontSize: 26, bold: true, color: ACCENT, margin: 0 });
    }
  });
  s.addText("Each step is the previous generation hitting a wall — and climbing over it.", {
    x: 0.9, y: 6.05, w: 11.6, h: 0.5, fontFace: FACE, fontSize: 15, italic: true, color: ACCENT, margin: 0 });
  s.addNotes(NOTES.bigPicture);
}

// ============================================================
// 3. CORE INTUITION (light, two-column with diagram)
// ============================================================
{
  const s = pres.addSlide();
  contentHeader(s, "Core intuition", "0E7490", "What is contrastive learning doing?");
  // left text
  s.addText([
    { text: "No labels — build the learning signal from the data itself", options: { bullet: { code: "2022" }, breakLine: true, bold: true, color: INK } },
    { text: "Two augmented views of one image = a positive pair → pull together", options: { bullet: { code: "2022" }, breakLine: true, color: "2A6B4F" } },
    { text: "Features of different images = negatives → push apart", options: { bullet: { code: "2022" }, breakLine: true, color: "B43050" } },
    { text: "Learn semantic features invariant to augmentation", options: { bullet: { code: "2022" }, color: INK } },
  ], { x: 0.7, y: 2.35, w: 6.0, h: 3.6, fontFace: FACE, fontSize: 15.5, color: INK, paraSpaceAfter: 16, lineSpacingMultiple: 1.05 });

  // right diagram panel
  const px = 7.15, py = 2.3, pw2 = 5.45, ph2 = 4.4;
  s.addShape(pres.shapes.RECTANGLE, { x: px, y: py, w: pw2, h: ph2, fill: { color: CARD }, line: { color: "DCE3F0", width: 1 }, shadow: softShadow() });
  const cx = px + pw2 / 2, cy = py + ph2 / 2;
  // pull arrow + positive
  s.addShape(pres.shapes.LINE, { x: cx - 1.55, y: cy, w: 1.35, h: 0, line: { color: "2A8F5F", width: 2.5, endArrowType: "triangle", beginArrowType: "triangle" } });
  s.addShape(pres.shapes.OVAL, { x: px + 0.45, y: cy - 0.38, w: 0.76, h: 0.76, fill: { color: "2A8F5F" }, shadow: softShadow() });
  s.addText("+", { x: px + 0.45, y: cy - 0.38, w: 0.76, h: 0.76, align: "center", valign: "middle", fontFace: FACE, fontSize: 24, bold: true, color: WHITE, margin: 0 });
  s.addText("pull", { x: cx - 1.7, y: cy - 0.62, w: 1.6, h: 0.3, align: "center", fontFace: FACE, fontSize: 12, bold: true, color: "2A8F5F", margin: 0 });
  // anchor
  s.addShape(pres.shapes.OVAL, { x: cx - 0.5, y: cy - 0.5, w: 1.0, h: 1.0, fill: { color: INK }, line: { color: ACCENT, width: 3 }, shadow: makeShadow() });
  s.addText("anchor", { x: cx - 0.5, y: cy - 0.5, w: 1.0, h: 1.0, align: "center", valign: "middle", fontFace: FACE, fontSize: 13, bold: true, color: WHITE, margin: 0 });
  // push arrows + negatives (two)
  s.addShape(pres.shapes.LINE, { x: cx + 0.62, y: cy - 0.85, w: 1.4, h: 0.7, flipV: true, line: { color: "C0395A", width: 2.5, endArrowType: "triangle" } });
  s.addShape(pres.shapes.LINE, { x: cx + 0.62, y: cy + 0.15, w: 1.4, h: 0.7, line: { color: "C0395A", width: 2.5, endArrowType: "triangle" } });
  s.addShape(pres.shapes.OVAL, { x: cx + 2.0, y: cy - 1.15, w: 0.7, h: 0.7, fill: { color: "C0395A" }, shadow: softShadow() });
  s.addText("−", { x: cx + 2.0, y: cy - 1.15, w: 0.7, h: 0.7, align: "center", valign: "middle", fontFace: FACE, fontSize: 22, bold: true, color: WHITE, margin: 0 });
  s.addShape(pres.shapes.OVAL, { x: cx + 2.0, y: cy + 0.45, w: 0.7, h: 0.7, fill: { color: "C0395A" }, shadow: softShadow() });
  s.addText("−", { x: cx + 2.0, y: cy + 0.45, w: 0.7, h: 0.7, align: "center", valign: "middle", fontFace: FACE, fontSize: 22, bold: true, color: WHITE, margin: 0 });
  s.addText("push", { x: cx + 0.72, y: cy - 0.15, w: 1.2, h: 0.3, align: "center", valign: "middle", fontFace: FACE, fontSize: 12, bold: true, color: "C0395A", margin: 0 });
  s.addNotes(NOTES.coreIntuition);
}

// ============================================================
// 4. InfoNCE — shared language
// ============================================================
{
  const s = pres.addSlide();
  contentHeader(s, "Shared language", "0E7490", "One equation runs through them all: InfoNCE");
  // formula panel (dark) — rendered as a stacked fraction so nothing wraps
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 2.35, w: 11.9, h: 1.55, fill: { color: NAVY }, shadow: makeShadow() });
  s.addText("ℒ  =  − log", { x: 1.7, y: 2.35, w: 3.1, h: 1.55, align: "right", valign: "middle", fontFace: MONO, fontSize: 24, bold: true, color: WHITE, margin: 0 });
  s.addText("exp( sim(zᵢ, zⱼ) / τ )", { x: 5.1, y: 2.52, w: 6.6, h: 0.55, align: "center", valign: "bottom", fontFace: MONO, fontSize: 19, bold: true, color: "7CF0A0", margin: 0 });
  s.addShape(pres.shapes.LINE, { x: 5.35, y: 3.14, w: 6.1, h: 0, line: { color: WHITE, width: 1.5 } });
  s.addText("Σₖ  exp( sim(zᵢ, zₖ) / τ )", { x: 5.1, y: 3.2, w: 6.6, h: 0.55, align: "center", valign: "top", fontFace: MONO, fontSize: 19, bold: true, color: "FF8FA8", margin: 0 });

  const cards = [
    { t: "Numerator = positives", d: "Similarity of two views of one image\n→ want it LARGE", c: "2A8F5F" },
    { t: "Denominator = negatives", d: "Similarity to all other samples\n→ want it SMALL", c: "C0395A" },
    { t: "τ = temperature", d: "Controls how sharp the distribution is\n→ affects attention to hard negatives", c: "0E7490" },
  ];
  const cw = 3.83, cg = 0.2, cy0 = 4.25, ch = 1.55, x0 = 0.7;
  cards.forEach((c, i) => {
    const x = x0 + i * (cw + cg);
    s.addShape(pres.shapes.RECTANGLE, { x, y: cy0, w: cw, h: ch, fill: { color: CARD }, line: { color: "DCE3F0", width: 1 }, shadow: softShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y: cy0, w: 0.1, h: ch, fill: { color: c.c } });
    s.addText(c.t, { x: x + 0.3, y: cy0 + 0.18, w: cw - 0.5, h: 0.4, fontFace: FACE, fontSize: 15, bold: true, color: c.c, margin: 0 });
    s.addText(c.d, { x: x + 0.3, y: cy0 + 0.6, w: cw - 0.5, h: 0.85, fontFace: FACE, fontSize: 11.5, color: MUTE, margin: 0, lineSpacingMultiple: 1.05 });
  });
  s.addText([
    { text: "The next 14 methods are all either a ", options: {} },
    { text: "variation", options: { bold: true, color: "0E7490" } },
    { text: " on this equation, or an ", options: {} },
    { text: "escape", options: { bold: true, color: E3 } },
    { text: " from it.", options: {} },
  ], { x: 0.7, y: 6.1, w: 11.9, h: 0.5, align: "center", fontFace: FACE, fontSize: 15, italic: true, color: INK, margin: 0 });
  s.addNotes(NOTES.infonce);
}

// ============================================================
// 5. EVOLUTION MAP (dark) — the centerpiece
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("The evolution map", { x: 0.9, y: 0.5, w: 11.5, h: 0.7, fontFace: FACE, fontSize: 28, bold: true, color: WHITE, margin: 0 });
  s.addText("Time →   each generation solves the previous one's bottleneck", { x: 0.9, y: 1.18, w: 11.5, h: 0.4, fontFace: FACE, fontSize: 14, color: ICE, margin: 0 });

  const cols = [
    { yr: "2018–19", c: E1, head: "Era 1 · Proxy task", ms: ["Instance\nDiscrimination", "Invariant\nSpread", "memory bank\n+ NCE"] },
    { yr: "2020", c: E2, head: "Era 2 · Contrastive / Clustering", ms: ["MoCo v1 / v2", "SimCLR v1 / v2", "SwAV · InfoMin"] },
    { yr: "2020–21", c: E3, head: "Era 3 · No negatives", ms: ["BYOL", "SimSiam", "Barlow Twins"] },
    { yr: "2021+", c: E4, head: "Era 4 · Transformer", ms: ["MoCo v3", "DINO", "DINOv2"] },
  ];
  const cw = 2.78, cg = 0.32, x0 = 0.9, y0 = 2.0, ch = 4.3;
  cols.forEach((col, i) => {
    const x = x0 + i * (cw + cg);
    s.addShape(pres.shapes.RECTANGLE, { x, y: y0, w: cw, h: 0.6, fill: { color: col.c } });
    s.addText(col.yr, { x: x + 0.15, y: y0 + 0.08, w: cw - 0.3, h: 0.44, fontFace: MONO, fontSize: 15, bold: true, color: NAVY, margin: 0, valign: "middle" });
    s.addShape(pres.shapes.RECTANGLE, { x, y: y0 + 0.6, w: cw, h: ch - 0.6, fill: { color: NAVY2 }, line: { color: col.c, width: 1.5 } });
    s.addText(col.head, { x: x + 0.2, y: y0 + 0.75, w: cw - 0.4, h: 0.55, fontFace: FACE, fontSize: 13.5, bold: true, color: col.c, margin: 0 });
    let yy = y0 + 1.4;
    col.ms.forEach((m) => {
      const lines = m.split("\n").length;
      const mh = lines > 1 ? 0.78 : 0.55;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.2, y: yy, w: cw - 0.4, h: mh, rectRadius: 0.06, fill: { color: NAVY3 } });
      s.addText(m.replace("\n", " "), { x: x + 0.32, y: yy, w: cw - 0.6, h: mh, valign: "middle", fontFace: FACE, fontSize: 12.5, bold: true, color: WHITE, margin: 0 });
      yy += mh + 0.18;
    });
    if (i < cols.length - 1) {
      s.addText("→", { x: x + cw - 0.04, y: y0 + 1.8, w: cg + 0.08, h: 0.6, align: "center", valign: "middle", fontFace: FACE, fontSize: 24, bold: true, color: ACCENT, margin: 0 });
    }
  });
  s.addText("Drop labels → drop the memory bank → drop negatives → swap the architecture", {
    x: 0.9, y: 6.55, w: 11.5, h: 0.45, align: "center", fontFace: MONO, fontSize: 12.5, bold: true, color: ICE, margin: 0 });
  s.addNotes(NOTES.evoMap);
}

// ============================================================
// ACT 1
// ============================================================
actDivider("1", E1, "2018 – 2019", "The start: with no labels,\nhow do we even define a task?", NOTES.act1);

// --- Instance Discrimination ---
methodSlide({
  era: A1("ERA 1 · 2018"), name: "Instance Discrimination", demo: "demo_assets/gifs/instance_discrimination.gif", venue: "CVPR 2018",
  authors: "Z. Wu, Y. Xiong, S. X. Yu, D. Lin — Unsupervised Feature Learning via Non-Parametric Instance Discrimination",
  idea: "Treat every image as its own class and do instance-level discrimination — no human labels at all.",
  mechanism: ["Memory bank stores an L2 feature vector per image", "NCE approximates the full softmax (constant Z fixed after first estimate)", "Weak aug, n_views=1; the bank provides the 'second view'"],
  contribution: "Pioneered the non-parametric memory bank; pain point: bank features go stale.",
  diagram: (s, o) => siamese(s, o, { single: true, proj: false, top: "Encoder\nf", loss: "NCE", extra: "Memory\nBank (all-data feats)" }),
  loss: "P(i|v) = exp(vᵢᵀf/τ) / Σⱼ exp(vⱼᵀf/τ)", lossNote: "Non-parametric softmax + NCE approximation",
});

// --- Invariant Spread ---
methodSlide({
  era: A1("ERA 1 · 2019"), name: "Invariant and Spreading Instance Feature", demo: "demo_assets/gifs/invariant_spread.gif", venue: "CVPR 2019",
  authors: "M. Ye, X. Zhang, P. C. Yuen, S.-F. Chang — Unsupervised Embedding Learning via Invariant and Spreading Instance Feature",
  idea: "No memory bank — pull two views of the same image together and push different images apart, all in the batch.",
  mechanism: ["Two augmented views + symmetric InfoNCE", "In-batch negatives (no queue, no memory bank)", "The direct ancestor of SimCLR"],
  contribution: "A prototype of in-batch softmax contrast; it set the stage for Era 2.",
  diagram: (s, o) => siamese(s, o, { top: "Encoder\nf", bot: "Encoder\nf", botNote: "shared weights", proj: false, loss: "Symmetric\nInfoNCE" }),
  loss: "−Σ log exp(sim/τ) / Σ exp(sim/τ)", lossNote: "In-batch symmetric softmax",
});

// ============================================================
// ACT 2
// ============================================================
actDivider("2", E2, "2020", "The explosion:\nwhere do negatives come from?", NOTES.act2);

// --- MoCo v1 ---
methodSlide({
  era: A2("ERA 2 · 2020"), name: "MoCo v1", demo: "demo_assets/gifs/moco_v1.gif", venue: "CVPR 2020",
  authors: "K. He, H. Fan, Y. Wu, S. Xie, R. Girshick — Momentum Contrast for Unsupervised Visual Representation Learning",
  idea: "A momentum encoder + FIFO queue supply many negatives that never go stale.",
  mechanism: ["Query encoder (gradient) + key encoder (EMA momentum)", "FIFO queue stores past keys as negatives", "Negative count decoupled from batch size"],
  contribution: "Many consistent negatives even with a small batch.",
  diagram: (s, o) => siamese(s, o, { top: "f_q\n(grad)", bot: "f_k", botNote: "EMA momentum", proj: false, loss: "InfoNCE", extra: "Queue\n(history-key negs)" }),
  loss: "−log exp(q·k₊/τ) / Σ_{queue} exp(q·k/τ)", lossNote: "Queue supplies many negatives",
});

// --- MoCo v2 ---
methodSlide({
  era: A2("ERA 2 · 2020"), name: "MoCo v2", demo: "demo_assets/gifs/moco_v2.gif", venue: "arXiv 2020",
  authors: "X. Chen, H. Fan, R. Girshick, K. He — Improved Baselines with Momentum Contrastive Learning",
  idea: "Port SimCLR's engineering tricks into MoCo to approach its performance with fewer resources.",
  mechanism: ["2-layer MLP projection head (replaces v1's linear)", "Adds Gaussian-blur strong aug + cosine LR", "Otherwise identical to v1"],
  contribution: "MLP head + strong aug = near-SimCLR results on a budget.",
  diagram: (s, o) => siamese(s, o, { top: "f_q\n(grad)", bot: "f_k", botNote: "EMA momentum", proj: true, loss: "InfoNCE", extra: "Queue\n(history-key negs)" }),
  loss: "−log exp(q·k₊/τ) / Σ_{queue} exp(q·k/τ)", lossNote: "Same as v1, plus an MLP projection head g",
});

// --- SimCLR v1 ---
methodSlide({
  era: A2("ERA 2 · 2020"), name: "SimCLR v1", demo: "demo_assets/gifs/simclr_v1.gif", venue: "ICML 2020",
  authors: "T. Chen, S. Kornblith, M. Norouzi, G. Hinton — A Simple Framework for Contrastive Learning of Visual Representations",
  idea: "No queue or bank — just a huge batch where samples serve as each other's negatives.",
  mechanism: ["Shared backbone + MLP projection head", "Symmetric NT-Xent; loss on z, evaluate on h", "Strong aug (color jitter s=1.0 + blur) is the key"],
  contribution: "Proved a simple framework + strong aug is enough; pain point: needs a huge batch.",
  diagram: (s, o) => siamese(s, o, { top: "Encoder\nf", bot: "Encoder\nf", botNote: "shared weights", proj: true, loss: "NT-Xent" }),
  loss: "−log exp(sim(zᵢ,zⱼ)/τ) / Σ_{k≠i} exp(·)", lossNote: "NT-Xent (in-batch, symmetric)",
});

// --- SimCLR v2 ---
methodSlide({
  era: A2("ERA 2 · 2020"), name: "SimCLR v2", demo: "demo_assets/gifs/simclr_v2.gif", venue: "NeurIPS 2020",
  authors: "T. Chen, S. Kornblith, K. Swersky, M. Norouzi, G. Hinton — Big Self-Supervised Models are Strong Semi-Supervised Learners",
  idea: "Deeper projection head + bigger model → a strong semi-supervised learner.",
  mechanism: ["3-layer projection head (used only during pretraining)", "Larger, deeper backbone", "Distill into a small model for semi-supervised learning"],
  contribution: "Big SSL models are strong semi-supervised learners (pretrain → distill).",
  diagram: (s, o) => siamese(s, o, { top: "Encoder\nf", bot: "Encoder\nf", botNote: "shared weights", proj: true, loss: "NT-Xent" }),
  loss: "−log exp(sim(zᵢ,zⱼ)/τ) / Σ_{k≠i} exp(·)", lossNote: "Same as v1, projection head deepened to 3 layers",
});

// --- SwAV ---
methodSlide({
  era: A2("ERA 2 · 2020"), name: "SwAV", demo: "demo_assets/gifs/swav.gif", venue: "NeurIPS 2020",
  authors: "M. Caron, I. Misra, J. Mairal, P. Goyal, P. Bojanowski, A. Joulin — Unsupervised Learning of Visual Features by Contrasting Cluster Assignments",
  idea: "No more pairwise comparison — switch to online clustering with prototypes.",
  mechanism: ["Multi-crop: 2 large + N small crops", "Sinkhorn-Knopp optimal transport for soft-assignment codes", "Swapped prediction: each view predicts the other's code"],
  contribution: "Contrast without pairwise negatives; multi-crop boosts efficiency.",
  diagram: (s, o) => siamese(s, o, { top: "Encoder\nf", bot: "Encoder\nf", botNote: "shared weights", proj: true, loss: "Swapped\nCE", extra: "Prototypes C\n+ Sinkhorn" }),
  loss: "−Σ qₜ·log pₛ − qₛ·log pₜ", lossNote: "Codes q from Sinkhorn-Knopp",
});

// --- InfoMin ---
methodSlide({
  era: A2("ERA 2 · 2020"), name: "InfoMin", demo: "demo_assets/gifs/infomin.gif", venue: "NeurIPS 2020",
  authors: "Y. Tian, C. Sun, B. Poole, D. Krishnan, C. Schmid, P. Isola — What Makes for Good Views for Contrastive Learning?",
  idea: "Step back and ask: what makes a 'good' augmentation view?",
  mechanism: ["Reuses SimCLR's backbone and NT-Xent loss", "More aggressive aug (s=1.5, grayscale 0.4, no blur)", "Removes shortcut cues like texture / color"],
  contribution: "Minimal-sufficient views: keep the shared task information, drop the rest.",
  diagram: (s, o) => siamese(s, o, { top: "Encoder\nf", bot: "Encoder\nf", botNote: "shared weights", proj: true, loss: "NT-Xent" }),
  loss: "min I(v₁;v₂)  +  NT-Xent(z₁,z₂)", lossNote: "Same loss as SimCLR; the point is view design",
});

// ============================================================
// ACT 3
// ============================================================
actDivider("3", E3, "2020 – 2021", "The heresy:\ndo we even need negatives?", NOTES.act3);

// --- BYOL ---
methodSlide({
  era: A3("ERA 3 · 2020"), name: "BYOL", demo: "demo_assets/gifs/byol.gif", venue: "NeurIPS 2020",
  authors: "J.-B. Grill, F. Strub, F. Altché, et al. (DeepMind) — Bootstrap Your Own Latent",
  idea: "Learn with no negatives at all — via predictor asymmetry + an EMA target.",
  mechanism: ["Online branch adds a predictor (creates asymmetry)", "Target branch = EMA of online (momentum 0.996→1)", "Predict the target representation; stop-gradient on target"],
  contribution: "Showed predictor asymmetry alone prevents collapse — a shock to the field.",
  diagram: (s, o) => siamese(s, o, { top: "online\nf_θ", bot: "target\nf_ξ", botNote: "EMA", proj: true, pred: true, sgBot: true, loss: "MSE" }),
  loss: "‖ q(z₁) − sg(z₂′) ‖²  (symmetrized)", lossNote: "Predictor asymmetry + sg(target)",
});

// --- SimSiam ---
methodSlide({
  era: A3("ERA 3 · 2021"), name: "SimSiam", demo: "demo_assets/gifs/simsiam.gif", venue: "CVPR 2021",
  authors: "X. Chen, K. He — Exploring Simple Siamese Representation Learning",
  idea: "Drop even the EMA — a single stop-gradient is enough to avoid collapse.",
  mechanism: ["Shared backbone (no momentum encoder, no queue)", "Predictor on online branch + stop-gradient on target", "Negative cosine similarity loss"],
  contribution: "A minimalist demonstration: stop-gradient is the key to preventing collapse.",
  diagram: (s, o) => siamese(s, o, { top: "Encoder\nf", bot: "Encoder\nf", botNote: "shared weights", proj: true, pred: true, sgBot: true, loss: "−cos" }),
  loss: "½D(p₁, sg z₂) + ½D(p₂, sg z₁)", lossNote: "Stop-grad is the only anti-collapse mechanism",
});

// --- Barlow Twins ---
methodSlide({
  era: A3("ERA 3 · 2021"), name: "Barlow Twins", demo: "demo_assets/gifs/barlow_twins.gif", venue: "ICML 2021",
  authors: "J. Zbontar, L. Jing, I. Misra, Y. LeCun, S. Deny — Self-Supervised Learning via Redundancy Reduction",
  idea: "A different philosophy: drive the two views' cross-correlation matrix toward the identity matrix.",
  mechanism: ["Compute the cross-correlation matrix C of the two views", "Diagonal→1 (invariance), off-diagonal→0 (decorrelation)", "High-dim projection head (8192) works best"],
  contribution: "No negatives / EMA / predictor — it's all in the loss on C.",
  diagram: (s, o) => siamese(s, o, { top: "Encoder\nf", bot: "Encoder\nf", botNote: "shared weights", proj: true, loss: "Cross-Corr\nC → I" }),
  loss: "Σᵢ(1−Cᵢᵢ)² + λ Σ_{i≠j} Cᵢⱼ²", lossNote: "Cross-correlation matrix → identity (decorrelation)",
});

// ============================================================
// ACT 4
// ============================================================
actDivider("4", E4, "2021 → today", "Architecture shift:\nswap in the Transformer", NOTES.act4);

// --- MoCo v3 ---
methodSlide({
  era: A4("ERA 4 · 2021"), name: "MoCo v3", demo: "demo_assets/gifs/moco_v3.gif", venue: "ICCV 2021",
  authors: "X. Chen, S. Xie, K. He — An Empirical Study of Training Self-Supervised Vision Transformers",
  idea: "Bring contrastive learning to the ViT and find the recipe that trains it stably.",
  mechanism: ["Freeze the patch-embedding projection (the key stability fix)", "Symmetric in-batch InfoNCE (drop the queue)", "AdamW + cosine LR (not SGD/LARS), m=0.99"],
  contribution: "Stable recipe for ViT contrastive training; no queue at large batch.",
  diagram: (s, o) => siamese(s, o, { top: "ViT f_q\n(grad)", bot: "ViT f_k", botNote: "EMA momentum", proj: true, loss: "Symmetric\nInfoNCE" }),
  loss: "ctr(q₁,k₂) + ctr(q₂,k₁)", lossNote: "Symmetric InfoNCE, no queue",
});

// --- DINO ---
methodSlide({
  era: A4("ERA 4 · 2021"), name: "DINO", demo: "demo_assets/gifs/dino.gif", venue: "ICCV 2021",
  authors: "M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, A. Joulin — Emerging Properties in Self-Supervised ViTs",
  idea: "Student–teacher self-distillation, with no contrastive negatives.",
  mechanism: ["Teacher = EMA of student (sees only global crops)", "Teacher output gets centering + sharpening", "Cross-entropy: student predicts the teacher's distribution"],
  contribution: "Attention maps show emergent object segmentation; anti-collapse via centering + sharpening.",
  diagram: (s, o) => siamese(s, o, { top: "student\nf_s", bot: "teacher\nf_ξ", botNote: "EMA", proj: true, sgBot: true, loss: "Cross-\nEntropy", extra: "centering\n+ sharpening" }),
  loss: "−Σ Pₜ log Pₛ , Pₜ=σ((g−C)/τₜ)", lossNote: "σ=softmax; centering + sharpening prevent collapse",
});

// --- DINOv2 ---
methodSlide({
  era: A4("ERA 4 · 2023"), name: "DINOv2", demo: "demo_assets/methods/dinov2.png", venue: "TMLR 2024",
  authors: "M. Oquab, T. Darcet, T. Moutakanni, et al. (Meta AI) — Learning Robust Visual Features without Supervision",
  idea: "Scale up self-distillation into a general-purpose visual foundation model.",
  mechanism: ["DINO (image-level) + iBOT (patch-level) self-distillation", "LVD-142M large curated dataset + training tricks", "This tutorial only includes a feature-extraction / fine-tuning demo"],
  contribution: "Strong general features, no fine-tuning needed — a visual foundation model.",
  diagram: (s, o) => siamese(s, o, { top: "student\nf_s", bot: "teacher\nf_ξ", botNote: "EMA", proj: true, sgBot: true, loss: "DINO\n+ iBOT", extra: "iBOT\n(patch masking)" }),
  loss: "ℒ_DINO + ℒ_iBOT + regularizers", lossNote: "Image-level + patch-level self-distillation, scaled up",
});

// --- DINOv3 (2025 coda) ---
methodSlide({
  era: A4("ERA 4 · 2025"), name: "DINOv3", demo: "demo_assets/methods/dinov3.png", venue: "arXiv 2025",
  authors: "O. Siméoni, H. V. Vo, et al. (Meta AI) — DINOv3",
  idea: "Push the same DINO line to extreme scale, with Gram anchoring to keep dense features sharp.",
  mechanism: ["DINO + iBOT self-distillation scaled to 7B params / 1.7B images", "Gram anchoring: stabilizes patch-level dense features late in training", "Distill the giant into usable smaller ViT-S/B/L models"],
  contribution: "SOTA frozen-backbone dense features; beats specialized models with no fine-tuning.",
  diagram: (s, o) => siamese(s, o, { top: "student\nf_s", bot: "teacher\nf_ξ", botNote: "EMA", proj: true, sgBot: true, loss: "DINO+iBOT\n+Gram", extra: "Gram\nanchoring" }),
  loss: "ℒ_DINO + ℒ_iBOT + ℒ_Gram", lossNote: "Self-distillation + Gram anchoring for dense features",
});

// 15. Comparison / collapse payoff table
{
  const s = pres.addSlide();
  contentHeader(s, "Synthesis", "0E7490", "Looking back: how did each era prevent collapse?");
  const hdr = (t) => ({ text: t, options: { fill: { color: NAVY }, color: WHITE, bold: true, fontFace: FACE, fontSize: 13.5, align: "center", valign: "middle" } });
  const rows = [
    [hdr("Era"), hdr("Representative"), hdr("Negatives?"), hdr("Anti-collapse mechanism")],
    ["Era 1 · 2018", "Instance Discrimination", { text: "Yes (memory bank)", options: { color: "2A8F5F" } }, "Push representations apart with many negatives"],
    ["Era 2 · 2020", "MoCo / SimCLR", { text: "Yes (queue / batch)", options: { color: "2A8F5F" } }, "In-batch / queue negatives"],
    ["Era 2 · 2020", "SwAV", { text: "No (uses prototypes)", options: { color: E3, bold: true } }, "Online clustering + Sinkhorn even assignment"],
    ["Era 3 · 2021", "BYOL / SimSiam", { text: "No", options: { color: E3, bold: true } }, "EMA / stop-gradient break symmetry"],
    ["Era 3 · 2021", "Barlow Twins", { text: "No", options: { color: E3, bold: true } }, "Redundancy reduction (feature decorrelation)"],
    ["Era 4 · 2021+", "DINO", { text: "No", options: { color: E3, bold: true } }, "centering + sharpening"],
  ];
  s.addTable(rows, {
    x: 0.7, y: 2.3, w: 11.93, colW: [2.1, 3.5, 2.73, 3.6],
    rowH: [0.5, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52],
    border: { pt: 0.5, color: "D5DCEA" }, align: "left", valign: "middle",
    fontFace: FACE, fontSize: 12.5, color: INK, margin: [0, 0.12, 0, 0.12],
    fill: { color: CARD },
  });
  s.addText([
    { text: "Punchline: all 14 methods are really answering one question — ", options: { color: INK } },
    { text: "'Without labels, how do we keep representations from collapsing?'", options: { bold: true, color: "0E7490" } },
  ], { x: 0.7, y: 6.35, w: 11.9, h: 0.5, align: "center", fontFace: FACE, fontSize: 14.5, italic: true, margin: 0 });
  s.addNotes(NOTES.collapseTable);
}

// 15b. Epoch progression (appendix) — clusters emerging over training, all 7 self-trained methods
function progSlide(suffix, rows, notes) {
  const s = pres.addSlide();
  contentHeader(s, "Training", "0E7490", "How do clusters emerge over epochs?" + suffix);
  s.addText("Self-trained SSL (CIFAR-10, ResNet-18 / ViT-S, 2×H100, 200 epochs): features go from a single blob at epoch 0 to clear class clusters.", {
    x: 0.7, y: 1.92, w: 11.9, h: 0.4, fontFace: FACE, fontSize: 12, color: MUTE, margin: 0 });
  const n = rows.length;
  const top = 2.34, bottom = 6.84, gap = 0.12;
  const sh = (bottom - top - (n - 1) * gap) / n;
  const sw = 9.6, sx = 3.0;
  let yy = top;
  rows.forEach((r) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: yy, w: 2.1, h: sh, fill: { color: NAVY } });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: yy, w: 0.09, h: sh, fill: { color: r.c } });
    s.addText(r.label, { x: 0.95, y: yy + sh / 2 - 0.32, w: 1.95, h: 0.34, fontFace: FACE, fontSize: 13.5, bold: true, color: WHITE, margin: 0 });
    s.addText(r.sub, { x: 0.95, y: yy + sh / 2 + 0.03, w: 1.95, h: 0.3, fontFace: FACE, fontSize: 9.5, color: ICE, margin: 0 });
    const p = "demo_assets/progression/" + r.key + ".png";
    if (fs.existsSync(p)) s.addImage({ path: p, x: sx, y: yy, w: sw, h: sh, sizing: { type: "contain", w: sw, h: sh } });
    yy += sh + gap;
  });
  s.addText("← Random init (epoch 0)      longer training: same-class tighter, different-class wider      Fully trained (epoch 200) →", {
    x: 3.0, y: 6.9, w: 9.6, h: 0.3, align: "center", fontFace: FACE, fontSize: 9.5, italic: true, color: MUTE, margin: 0 });
  if (notes) s.addNotes(notes);
}
progSlide(" (1/3 · Era 1–2)", [
  { key: "instance_discrimination", label: "Instance Discrim.", sub: "memory bank", c: E1 },
  { key: "invariant_spread", label: "Invariant Spread", sub: "in-batch softmax", c: E1 },
  { key: "simclr_v1", label: "SimCLR v1", sub: "in-batch contrast", c: E2 },
  { key: "simclr_v2", label: "SimCLR v2", sub: "deeper 3-layer head", c: E2 },
], NOTES.prog1);
progSlide(" (2/3 · Era 2–3)", [
  { key: "infomin", label: "InfoMin", sub: "view design", c: E2 },
  { key: "byol", label: "BYOL", sub: "predictor + EMA", c: E3 },
  { key: "simsiam", label: "SimSiam", sub: "stop-gradient", c: E3 },
], NOTES.prog2);
progSlide(" (3/3 · official-weight methods, now self-trained)", [
  { key: "moco_v1", label: "MoCo v1", sub: "queue + momentum", c: E2 },
  { key: "moco_v2", label: "MoCo v2", sub: "MLP head + strong aug", c: E2 },
  { key: "swav", label: "SwAV", sub: "prototype + Sinkhorn", c: E2 },
  { key: "barlow_twins", label: "Barlow Twins", sub: "decorrelation", c: E3 },
  { key: "moco_v3", label: "MoCo v3", sub: "ViT + frozen patch", c: E4 },
  { key: "dino", label: "DINO", sub: "self-distillation + centering", c: E4 },
], NOTES.prog3);

// 16. Live demo
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.28, h: PH, fill: { color: ACCENT } });
  s.addText("LIVE DEMO", { x: 0.9, y: 0.5, w: 6, h: 0.45, fontFace: MONO, fontSize: 15, bold: true, color: ACCENT, charSpacing: 4, margin: 0 });
  s.addText("Seeing is believing: feature space goes from one blob to clusters", { x: 0.9, y: 0.95, w: 11.5, h: 0.65, fontFace: FACE, fontSize: 24, bold: true, color: WHITE, margin: 0 });

  // command strip
  s.addShape(pres.shapes.RECTANGLE, { x: 0.9, y: 1.75, w: 11.5, h: 0.98, fill: { color: "060B16" }, line: { color: NAVY3, width: 1 }, shadow: makeShadow() });
  s.addText([
    { text: "$ python train.py --config configs/simclr_v1_resnet18.yaml --data-dir data/cifar10", options: { breakLine: true, color: "7CF0A0" } },
    { text: "$ python eval/umap_vis.py configs/simclr_v1_resnet18.yaml --ckpt checkpoints/last.ckpt", options: { color: "7CF0A0" } },
  ], { x: 1.15, y: 1.9, w: 11, h: 0.7, fontFace: MONO, fontSize: 12, lineSpacingMultiple: 1.3, margin: 0 });

  // before / after UMAP images
  const imgY = 3.25, imgS = 2.65, labY = 2.88;
  const beforeX = 3.0, afterX = 7.68; // arrow sits in the ~0.95" gap between
  s.addText("Before · random init", { x: beforeX, y: labY, w: imgS, h: 0.34, align: "center", fontFace: FACE, fontSize: 14, bold: true, color: "FF8FA8", margin: 0 });
  s.addText("After · pretrained features", { x: afterX, y: labY, w: imgS, h: 0.34, align: "center", fontFace: FACE, fontSize: 14, bold: true, color: "7CF0A0", margin: 0 });
  s.addImage({ path: "demo_assets/umap_before.png", x: beforeX, y: imgY, w: imgS, h: imgS, sizing: { type: "contain", w: imgS, h: imgS } });
  s.addImage({ path: "demo_assets/umap_after.png", x: afterX, y: imgY, w: imgS, h: imgS, sizing: { type: "contain", w: imgS, h: imgS } });
  s.addShape(pres.shapes.RECTANGLE, { x: beforeX, y: imgY, w: imgS, h: imgS, fill: { color: "FFFFFF", transparency: 100 }, line: { color: "C0395A", width: 1.5 } });
  s.addShape(pres.shapes.RECTANGLE, { x: afterX, y: imgY, w: imgS, h: imgS, fill: { color: "FFFFFF", transparency: 100 }, line: { color: "2A8F5F", width: 1.5 } });
  s.addText("→", { x: beforeX + imgS, y: imgY + imgS / 2 - 0.4, w: afterX - beforeX - imgS, h: 0.8, align: "center", valign: "middle", fontFace: FACE, fontSize: 40, bold: true, color: ACCENT, margin: 0 });

  s.addText([
    { text: "* Illustrative: an ImageNet-pretrained backbone stands in for post-training features (CIFAR-10).", options: { breakLine: true } },
    { text: "  Swap in your own trained SSL checkpoint to reproduce the real result.", options: {} },
  ], { x: 0.9, y: 6.35, w: 11.5, h: 0.8, align: "center", fontFace: FACE, fontSize: 12, italic: true, color: MUTE, margin: 0, lineSpacingMultiple: 1.15 });
  s.addNotes(NOTES.liveDemo);
}

// 17. Closing
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape(pres.shapes.OVAL, { x: -2.5, y: 4.5, w: 6, h: 6, fill: { color: NAVY3, transparency: 45 } });
  s.addText("The evolution in one sentence", { x: 0.9, y: 0.85, w: 11.5, h: 0.6, fontFace: FACE, fontSize: 26, bold: true, color: ACCENT, margin: 0 });
  s.addText([
    { text: "For five years, contrastive learning kept removing dependencies —\nfrom the memory bank, to negatives, and finally even the ", options: { color: WHITE } },
    { text: "architecture", options: { color: ACCENT, bold: true } },
    { text: " got replaced.", options: { color: WHITE } },
  ], { x: 0.9, y: 1.6, w: 11.5, h: 1.5, fontFace: FACE, fontSize: 24, bold: true, lineSpacingMultiple: 1.15, margin: 0 });

  const takeaways = [
    "All methods share the InfoNCE intuition: pull positives together, push negatives apart",
    "The real challenge is preventing collapse; negatives are just one path",
    "Trend: fewer assumptions, larger scale → toward foundation models",
  ];
  let yy = 3.5;
  takeaways.forEach((t, i) => {
    s.addShape(pres.shapes.OVAL, { x: 0.95, y: yy + 0.02, w: 0.42, h: 0.42, fill: { color: ACCENT } });
    s.addText(String(i + 1), { x: 0.95, y: yy + 0.02, w: 0.42, h: 0.42, align: "center", valign: "middle", fontFace: FACE, fontSize: 16, bold: true, color: NAVY, margin: 0 });
    s.addText(t, { x: 1.6, y: yy, w: 11, h: 0.46, valign: "middle", fontFace: FACE, fontSize: 15.5, color: ICE, margin: 0 });
    yy += 0.62;
  });

  s.addShape(pres.shapes.LINE, { x: 0.9, y: 5.75, w: 11.5, h: 0, line: { color: NAVY3, width: 1 } });
  s.addText([
    { text: "Open-source teaching project · 14 methods, unified implementation · every loss readable on its own", options: { color: ICE } },
  ], { x: 0.9, y: 5.95, w: 11.5, h: 0.5, fontFace: FACE, fontSize: 13.5, margin: 0 });
  s.addText("Thanks for listening — questions welcome", { x: 0.9, y: 6.5, w: 11.5, h: 0.6, fontFace: FACE, fontSize: 22, bold: true, color: WHITE, margin: 0 });
  s.addNotes(NOTES.closing);
}

pres.writeFile({ fileName: "contrastive_learning_evolution_EN.pptx" }).then((f) => console.log("WROTE", f));
