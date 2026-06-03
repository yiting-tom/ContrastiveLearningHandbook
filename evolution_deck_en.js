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

// ---------- speaker notes (verbatim spoken script + 🎬 stage cue, one per slide) ----------
// English mirror of the zh-TW NOTES in evolution_deck.js, so the presenter sees
// the spoken script in PowerPoint's Notes / Presenter View. Keys map 1:1 to slides.
const N = (spoken, cue) => `${spoken}\n\n🎬 ${cue}`;
const NOTES = {
  cover: N(
    "So, let me start with a question. Suppose I hand you a million photos — cats, dogs, cars, houses — but I give you not a single label. Nobody tells you which one is a cat and which is a dog. So how is a model supposed to learn to tell cats from dogs on its own? Sounds a bit magical, right? Like asking a child who has never seen an animal to sort cats and dogs into two piles. But over the next twenty-some minutes, I want to tell you a story — the story of how this field, over five whole years, step by step taught machines to do exactly that. The topic is called contrastive learning.",
    "Stand center stage, don't rush to advance. Pause a beat on 'not a single label' to let it sink in. After 'tell cats from dogs?' hold about 2 seconds to let the suspense build, then bring up the title."),
  bigPicture: N(
    "Before we get into the details, let me hand you a pair of glasses to wear for the whole talk. For five years, this entire field has really been doing just one thing — removing crutches, one after another. What do I mean? At first the model needs a lot of external crutches just to stand; then researchers pull them away one by one, forcing it to stand on its own. We will see four steps. 2018: the first crutch goes — human labels; you can learn without them. 2020: drop a big memory store called the memory bank and use something smarter. 2021: even bolder — they remove negatives themselves, which back then was almost unthinkable. And after 2021, even the underlying architecture gets swapped, from CNN to Transformer. So keep these glasses on: four steps, four crutches. Whatever method you see next, ask yourself one thing — which crutch did this one remove?",
    "Advance. You can count the four steps on your fingers — 1, 2, 3, 4. Raise your voice and look a little surprised on 'they remove negatives themselves.' Slow down on 'which crutch did it remove?' — make it the refrain for the whole talk."),
  coreIntuition: N(
    "OK, so 'learning without labels' — what is the magic? The core intuition is almost embarrassingly simple. Take the same photo of a cat and make two different versions — crop one, recolor or flip the other. To us it is still the same cat; so we tell the model: pull these two close together in feature space. Conversely, a photo of a dog versus my cat — push them apart. Pull together, push apart, over and over, and the model gradually learns to grab whatever stays the same no matter how you crop or distort — that is semantics. That is the secret to telling cats from dogs without labels. But — here is the trap. If I only tell you 'pull the two versions of the same image together,' what is the laziest solution? Map every image in the world to the exact same point! Then 'pull together' is always a perfect score. That disaster has a name: collapse. Remember that word — collapse — it will haunt the rest of this talk like a ghost.",
    "Advance, point at the pull/push diagram. Use a half-joking tone on 'that's it?'. At the end, lower your voice and slow down on 'collapse' — this is the central tension of the whole talk, so perform it."),
  infonce: N(
    "That 'pull together, push apart' — written as math, it is this equation, called InfoNCE. You do not have to memorize it; just read three spots. The numerator is the positive pair — the two views of the same image — we want it large. The denominator is a pile of negatives — different images — we want it small. And there is a little τ, the temperature, that tunes how tight things are. One memory hook: numerator pulls, denominator pushes. Why show you this equation up front? Because all 14 methods coming up are playing with this one equation — either rewriting it, or trying every trick to escape it.",
    "Advance, point at the numerator then the denominator with your finger or pointer. You can go a bit faster here — method pages, don't drag. Use 'escape it' as the hook into the next page with a tiny pause."),
  evoMap: N(
    "Finally, here is a map we will follow. See — laid out left to right are four eras: 2018's dropping labels, 2020's big batches and memory stores, 2021's farewell to negatives, then the Transformer era. These arrows in between matter — they are not just chronology, they are 'who solved whose bottleneck.' The previous method got stuck, hit a problem, and that is why the next one was born. So this is not a pile of scattered papers — it is one cause-and-effect chain of evolution. OK, the story begins for real — let us go back to 2018, where the first crutch, labels, is about to be removed.",
    "Advance, sweep your hand along the timeline left to right, then tap a few arrows to stress cause and effect. Use the last line 'back to 2018' as the transition into ACT 1 — lift your tone for a sense of takeoff, then pause and hand off to the next page."),
  act1: N(
    "OK, the story begins. Rewind to 2018. Back then everyone was stuck on a very basic but annoying problem: I have a huge pile of images, but absolutely no labels — nobody tells me which is a cat, which is a dog. So what is the model supposed to learn? You have to give it a task, right? But where does the task come from? Labels were our biggest crutch, and now that crutch has been pulled away. In Act 1 we will see how people conjured a task out of thin air.",
    "Walk to center stage, pause a beat before speaking. Spread your hands on 'no labels'; deliberately slow down on 'out of thin air' to build suspense, then cut to the next page."),
  instance_discrimination: N(
    "The first answer is clever to the point of being stubborn: since there are no labels, I will just declare — every single image is its own class. A million images, a million classes. The model's job is to tell each image apart from all the others. But there is an engineering problem: you cannot compare against all million every step. So they used a memory bank, storing each image's feature ahead of time as a ready-made pool of negatives; and NCE to approximate that giant, uncomputable softmax. One memory hook: it treats each image as a class. Its wall is classic too — the features in the bank were computed several rounds ago; the model keeps improving while the bank chews on stale food, so the features go out of date. That staleness is exactly what the next step has to fix.",
    "Make the 'one image = one class' gesture. Point at the bank box in the diagram on 'memory bank.' Emphasize 'chews on stale food' with a touch of humor."),
  invariant_spread: N(
    "The next year, someone said: then let us just drop the bank. Invariant Spread's idea is direct — take two augmentations of one image; these two views should pull together, that is 'invariant'; different images push apart, that is 'spread.' Negatives do not come from any bank — they are the other images in this very batch, computed and used right now, never stale. It uses a symmetric InfoNCE. The memory hook: this is the direct ancestor of SimCLR — the core recipe is already here, they just had not cranked up the scale yet. So look — we just dropped labels, and right away we also tossed the memory-bank crutch.",
    "Use both hands for 'pull together' and 'push apart.' Pause on 'direct ancestor,' look at the audience, plant the seed. End crisply on 'tossed.'"),
  act2: N(
    "OK, into Act 2 — and this act is the most exciting fork in the whole story. We had established one thing: for contrastive learning to work, you need many, fresh negatives to push representations apart, or it collapses into a blob. The question is — where do all these negatives come from? In 2020 two camps gave two completely different answers, and it turned into a clash of the titans. One says 'I will store them in a queue,' the other says 'I will just blow up the batch.' Let us look at the first camp.",
    "Pick up the pace, call out 'the most exciting fork.' On 'two camps,' point left and right with each hand, striking a face-off pose."),
  moco_v1: N(
    "The first camp is Kaiming He's team, MoCo. Their insight: negatives do not have to be in this one batch — I can take a FIFO queue and line up features from the past several batches as negatives. That decouples the number of negatives from batch size entirely — tiny batch, but still tens of thousands of negatives. But wait — does not that bring back Instance Discrimination's 'stale features' problem? That is their prettiest move: the momentum encoder. The encoder on the negatives side is not updated by gradient — it slowly, smoothly catches up to the main encoder, so the features in the queue, though old, are consistent and do not jump around. Memory hook: a small batch can still have a huge, consistent pool of negatives.",
    "Point at the queue box on 'queue.' On 'but wait,' deliberately pause and frown to set up the callback, then reveal the momentum-encoder move."),
  moco_v2: N(
    "MoCo v2 is a quick page. No big new theory — just a very practical engineering upgrade: take the three good things proven over at SimCLR — the MLP projection head, stronger Gaussian-blur augmentation, and the cosine learning-rate schedule — and drop them straight into MoCo's framework. The result: with small resources and a small batch, it closes in on the TPU-big-batch SimCLR. Memory hook: good ideas get borrowed, shamelessly. This also previews the rival we are about to meet.",
    "Light, smiling tone. Be playful on 'borrowed.' Gesture toward the next page."),
  simclr_v1: N(
    "OK, the other camp steps up — Hinton's team, SimCLR, with the exact opposite philosophy. They say: why bother with queues and momentum? Too much hassle. I will just blow the batch up huge — so big that within this one batch there are thousands of negatives to compare against, no need to store history. The framework is dead simple. But they found one key thing — strong augmentation is the real soul, especially fierce color jitter plus Gaussian blur; without it the whole thing will not learn. The memory hook, and its pain point: for this path to work you need a TPU-scale batch, which ordinary labs simply cannot afford. So you see, the MoCo-versus-SimCLR fight is, at heart, the same question — where do negatives come from — answered by two philosophies: one saves memory with a queue, the other brute-forces it with batch size.",
    "Step up with a stronger tone, like introducing a rival. Mimic a dismissive look on 'too much hassle.' At the end, contrast the two camps side by side, one hand per camp, for a summary gesture."),
  simclr_v2: N(
    "SimCLR v2 is the same path, leveled up, and the theme is scale. Two changes: the projection head goes from two layers to three, and the backbone is swapped for a bigger, deeper model. What it really wants to prove is fascinating — big self-supervised models are very strong semi-supervised learners: first pretrain a big model on tons of unlabeled data, then fine-tune with a tiny bit of labels, and finally distill into a small model. Memory hook: the bigger the model, the more it can squeeze out of unlabeled data.",
    "Gesture 'deeper, bigger.' On 'just a tiny bit of labels,' pinch your thumb and finger into a small gap."),
  swav: N(
    "By now some people start to feel 'pairwise comparison' is itself clunky. SwAV jumps in: I will not compare image to image one-on-one anymore. I set up a group of prototypes — think of them as learnable cluster centers; each image computes which clusters it belongs to, getting an 'assignment.' The trick is it uses the Sinkhorn algorithm to force these assignments to spread evenly across the clusters — no piling everyone into one cluster — and that step itself prevents collapse. Add multi-crop — one image cut into two big plus several small crops — and you get a bunch of extra views basically for free. The most important memory hook: it needs no pairwise negatives at all. Wait — no negatives and still no collapse? That line points us straight to the final suspense.",
    "Draw a few cluster centers in the air on 'prototypes.' Slow down and lift your tone on 'no negatives and still no collapse?' — make it a deliberate hook."),
  infomin: N(
    "Before closing this act, InfoMin asks a very philosophical but crucial question. While everyone else was grinding on negatives and architectures, it steps back and asks: we keep pulling two augmented views together — but what actually makes a 'good' view? Its answer is minimal sufficient: between the two views, keep all the shared semantics (sufficient), but cut the unshared stuff — shortcuts like color and texture — as much as possible (minimal). How to cut? Use more aggressive augmentation to destroy those shortcuts, so the model cannot cheat. Memory hook: sometimes the bottleneck is not the loss or the architecture — it is the very 'views' you feed in. OK, that wraps Act 2. We saw negatives played a hundred ways, but SwAV already quietly hinted — negatives are not the only answer. And if we push that thought to the extreme — drop even a single negative? See you in the next act.",
    "Lean back slightly on 'step back and ask,' a thinking pose. Slow down and stress the last three sentences; after 'drop even a single negative?', hold two seconds, then cut to the transition."),
  act3: N(
    "OK, we are at the wildest act of the whole story. We have been removing crutches all along: first human labels, then the giant batch, then that memory queue. But one crutch — from day one until now — nobody dared touch: negatives. What are negatives for? They are the 'push apart' force. The spirit of contrastive learning is one line: pull yourself together, push others apart. So what if I remove 'push others apart' entirely? Think about it — if only 'pull together' is left, with no 'push,' what is the model's smartest cheat? Map everything to the same point — every image looks identical, the loss drops straight to zero, job perfectly done. That is the nightmare we have feared all along: collapse. So back then the consensus was: no negatives, guaranteed collapse. No exceptions. Remember that consensus — because the next few papers are here to smack it down.",
    "Walk to center stage, pause for a ceremonial beat. Slow down and stress 'no negatives, guaranteed collapse,' hold 1 second, then advance. You can make a 'push apart' then 'shrink to a dot' gesture."),
  byol: N(
    "The first to jump out is BYOL. It flatly says: I want no negatives, not even one. The whole field waited for it to collapse. How does it work? Two branches, one called online, one called target. Here is the key: the online side secretly adds a small network called a predictor, making the two sides asymmetric; and the target does not learn on its own — it is a 'slow-motion clone' of online, catching up via momentum, with the gradient cut off on the target side. The result? It just does not collapse, and it works shockingly well. Everyone's first reaction was 'there must be a bug somewhere.' One memory hook: it holds up without collapsing by relying on asymmetry, not on pushing apart. The negative-sample crutch is dropped for the first time.",
    "Turn to the BYOL page, point at the small predictor box. Gesture on 'slow-motion clone.' Stress 'doesn't collapse' with a surprised look to spark the audience's curiosity."),
  simsiam: N(
    "After BYOL everyone breathed easier, figuring 'ah, it is that momentum EMA secretly preventing collapse.' Then Kaiming He's team comes out and says: no, you are all overthinking it. SimSiam cuts the momentum EMA too — the two branches just share the same network, no clone, no queue, nothing. So what keeps it from collapsing? One single move — stop-gradient, the gradient stops on that side. That is it. It is basically a minimalist experiment: strip everything away, leave one thing, and point at it: look, the real key to preventing collapse, from start to finish, is this stop-gradient. Clean, elegant, straight to the point.",
    "Advance. Make repeated 'deleting' gestures on 'strip everything.' On stop-gradient, point at the ⊘ stop symbol in the diagram, hold half a second, stress 'just this one.'"),
  barlow_twins: N(
    "Barlow Twins is even more interesting — it does not play pull-and-push with you anymore, it swaps in a whole new philosophy. It says: I will look at the cross-correlation matrix between the two views' embeddings. You just force this matrix toward the identity — diagonal to 1, meaning the two views of the same image should agree; off-diagonal squeezed to 0, meaning each dimension should not repeat another, should not say the same thing. This is called redundancy reduction. The amazing part: no negatives, no EMA, no predictor — all the work is in the loss. So at this point, putting the three papers together, a common thread appears: the key to preventing collapse was never 'pushing others apart,' it is 'breaking symmetry.' Whether you use a predictor, a stop-gradient, or force the cross-correlation matrix to decorrelate, it is essentially the same thing — do not let the two sides collapse into the exact same thing. That is the line Act 3 leaves us with.",
    "Advance, point at the loss formula C→I. Slow down on the closing line 'breaking symmetry,' look across the room — this is the act's conclusion, let it land."),
  act4: N(
    "OK, by now we have stripped away nearly every crutch: labels, big batch, queue, negatives — one by one. But one thing never changed — the engine underneath has always been a CNN, a ResNet. Meanwhile a major earthquake was hitting vision: the Transformer arrived, the ViT appeared. So the question is: if I swap the entire engine under contrastive learning to a Transformer, what happens? A painless plug-in, or a new kind of collapse blowing up? Our final act is about how this crowd tamed the new engine and rode it all the way to today's foundation models.",
    "Walk back to center stage. Shift to a 'final act' wrap-up tone. Pause on 'swap to a Transformer,' toss out the suspense, then advance."),
  moco_v3: N(
    "The first to brave it is MoCo v3. It puts contrastive learning straight onto a ViT, and right away in training: super unstable, the loss suddenly spikes, and halfway through the whole thing falls apart. The team hunted for a long time and finally caught a deeply counterintuitive culprit — the very first layer, the patch embedding, the layer that chops the image into patches and projects them into vectors. Their fix is almost absurdly simple: just freeze that layer, do not train it. Freeze it, and the whole thing stabilizes. While they were at it, they also tossed the memory queue and switched to in-batch negatives. Memory hook: sometimes the key to stability is not where you frantically tune — it is the most inconspicuous first layer.",
    "Advance, point at the patch embedding at the bottom of the diagram. Make a 'lock it down' gesture on 'freeze that layer.' You can raise an eyebrow, with a knowing tone, on 'the most inconspicuous first layer.'"),
  dino: N(
    "Next is DINO, my personal favorite. It plays 'self-distillation': a student and a teacher — but the teacher is not some hired expert, it is the student's own clone. The student works to predict the teacher's output. To avoid collapse, it adds two small moves on the teacher side: centering, pulling the output back to center so no single class dominates; and sharpening, making the distribution sharper to force a decision. One pulls, one pushes, balanced right in the middle so it does not collapse. The key point: it has no negatives at all. But DINO's most stunning part is not the score — it is this. Look at its attention map: nobody taught it what an object is, no annotations given, yet its attention lights up along the outline of a dog, of a bird, all by itself. It inadvertently learned to 'segment objects.' That is the most enchanting 'aha' moment in self-supervised learning.",
    "Advance. After the mechanism, pause, then switch to the attention-map image and point at the object outlines lighting up. Slow down and sound delighted on 'lights up by itself,' so the audience goes 'wow' with you."),
  dinov2: N(
    "Last stop, DINOv2. It did not invent a brand-new trick — it did something harder: take the right method and scale it up. It takes DINO's image-level self-distillation, adds iBOT's patch-level masked learning, then feeds it a carefully curated dataset of over a hundred million images, LVD-142M — brute force, miracle. The result is a truly general visual foundation model: take its features, no further training, plug straight into classification, segmentation, depth estimation — it does great on all of them. That connects our whole thread to the foundation-model wave everyone talks about today. So looking back over these five years — we went from 'needing a human to label every image' all the way to 'no labels, no negatives, just feed it enough images and the model grows its own understanding of the world.' The crutches, one by one, are all gone.",
    "Advance, the last page of this section. Slow down on the final 'the crutches are all gone,' look across the room for the wrap, with a 'spreading out, setting down' gesture, handing this section cleanly back to the main thread."),
  collapseTable: N(
    "OK, we have gone from 2018 to now, fourteen methods, four eras, each name cooler than the last. But I am going to clear the whole table and leave just one sentence. Remember the seed I planted at the start? I said the thing contrastive learning fears most is collapse — the model cheating by mapping every image to the same point, loss beautifully zero, but the representation worthless. Look at this table. I have listed each era's representative method, whether it uses negatives, and what it relies on to prevent collapse. You will notice something striking: Era 1 used the memory bank's many negatives to push representations apart; Era 2's MoCo and SimCLR switched to queue and big-batch negatives; then SwAV got rebellious and used prototypes plus Sinkhorn to spread points evenly; by Era 3, BYOL and SimSiam just dropped negatives, relying on EMA and stop-gradient to break symmetry; Barlow Twins used decorrelation; DINO used centering plus sharpening. The mechanisms are all different, right? But — they are all answering the same question. That is today's punchline: fourteen methods, all essentially answering this one line — 'without labels, how do I keep the representation from collapsing?' Five years of evolution, in a nutshell, is humanity coming up with fourteen different ways to answer this one question.",
    "Pause a beat before speaking, callback to the opening. Make a 'squeeze everything into one ball' gesture on 'collapse.' Point row by row at the last column (anti-collapse mechanism) so everyone sees they differ. Slow down and stress the final punchline, circling the bottom line 'without labels, how to avoid collapse?' with your pointer."),
  prog1: N(
    "After all that theory, you might think: OK, sounds fancy, but what does 'successfully preventing collapse' actually look like? These I ran myself. On the school's two H100s, with CIFAR-10 and ResNet-18, I trained from scratch for two hundred epochs, then took a UMAP snapshot of the features at several epochs and strung them into this evolution strip. First, these four from Era 1 to Era 2: Instance Discrimination, Invariant Spread, SimCLR v1, SimCLR v2. Look at the far left — epoch 0, random init, just one blob, all ten classes smeared together; that is literally 'the starting point of collapse.' Then look right — as training goes, same-class points gather, different classes get pushed apart, and by the far right, epoch 200, you can see clear clusters. Notice the two SimCLR rows — they cluster noticeably faster and cleaner, with sharper boundaries; that is the power of in-batch contrast with lots of positives and negatives. Without a single label, purely from the signal 'two views of one image should match, different images should not,' it forces the structure out.",
    "Switch to the first evolution strip. With your pointer, sweep from the epoch-0 blob on the left rightward to the epoch-200 clusters, with a 'blob → spread → cluster' gesture. Point specifically at the two SimCLR rows on 'faster and cleaner.' A touch of 'I ran this myself' pride."),
  prog2: N(
    "Now the three from Era 2 to Era 3, and here is the point: InfoMin, BYOL, and SimSiam. The earlier ones still had negatives to lean on, but BYOL and SimSiam — remember? — are the two that use no negatives at all. In theory the most prone to collapse. Yet look at these two strips — from the epoch-0 blob, they steadily differentiate into beautiful clusters too, never collapsing to a point. That is seeing-is-believing: just stop-gradient, just a predictor plus EMA, really can prevent collapse. Their cluster shapes look a little different from SimCLR's contrastive style, but 'they can cluster' — this figure vouches for them directly.",
    "Switch to the second evolution strip. Point at the BYOL and SimSiam rows first, lift your tone for the 'they actually didn't collapse' surprise. You can glance back at the previous strip for half a second to compare. Keep the pace quick — this is a transition page."),
  liveDemo: N(
    "OK, slides alone are not satisfying enough — let us see it live. This is the single most important comparison of the whole demo: on the left, the features before training, randomly initialized — see, just a ball of fuzz, reds and greens all mixed, you cannot tell who is who; that is what collapse looks like. On the right, the feature space after full training — ten colors, cleanly separated into their own clusters. From one blob to clear clusters — that is what self-supervised learning does, with no labels used the entire time. Here I want to clear up a common misconception, which is also my biggest takeaway from running all these experiments: how well it clusters mainly comes down to — whether you trained enough. As long as the epochs are sufficient and training is on track, almost every method can eventually separate the clusters. So where is the difference between methods? In the architecture and how the loss is designed — that is, which trick it uses to prevent collapse, how fast it converges, how big a batch it needs, whether it needs negatives. In other words, the difference between methods is not 'can it cluster,' it is 'at what cost, and via which path, does it prevent collapse.' I have put a GIF here — let me just run it once; watch that ball of points slowly, slowly spread out and fall into line.",
    "Switch to the LIVE DEMO page. Point at the red box on the left, 'before training,' say 'ball of fuzz'; then the green box on the right, 'after training,' say 'clusters'; slide your hand along the arrow in the middle left to right. Stress and pause on 'trained enough.' Then actually play the GIF/UMAP animation, following the moving points with your hand, leaving a few seconds for people to watch — don't talk over it."),
  closing: N(
    "OK, let us wrap up. If you take just one sentence from this talk, I hope it is this: for five years, contrastive learning has been doing one thing — removing dependencies. From needing the memory-bank crutch at first, to dropping negatives later, to finally swapping out even the CNN architecture for a Transformer. Step by step, taking away every external crutch. Three takeaways for you. First, all methods really share the same InfoNCE intuition — pull positives together, push negatives apart; however fancy the names, at the core it is this. Second, and this is what I most want to stress: the real challenge from start to finish is preventing collapse; negatives are just one solution, not the only path. Third, the overall trend: fewer and fewer assumptions, larger and larger scale, all the way to today's foundation models. Finally, a quick plug — all fourteen methods I covered today, I have implemented as an open-source teaching project, one unified codebase, every loss readable on its own and runnable yourself; you can reproduce those H100 results too. Feel free to clone it and play. Thank you all for listening — and now, questions are welcome!",
    "Switch to the closing page. Slow down and weight the line 'removing dependencies' the most — it's the thesis of the whole talk. Point at the on-screen 1, 2, 3 dots for the three takeaways. Point at the bottom line (or show a GitHub QR/link) for the open-source project. On the final 'questions welcome,' face the room, smile, open your arms slightly, and move into Q&A."),
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
  const TRAINED = ["instance_discrimination", "invariant_spread", "simclr_v1", "simclr_v2", "infomin", "byol", "simsiam"];
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
  era: A2("ERA 2 · 2020"), name: "MoCo v1", demo: "demo_assets/methods/moco_v1.png", venue: "CVPR 2020",
  authors: "K. He, H. Fan, Y. Wu, S. Xie, R. Girshick — Momentum Contrast for Unsupervised Visual Representation Learning",
  idea: "A momentum encoder + FIFO queue supply many negatives that never go stale.",
  mechanism: ["Query encoder (gradient) + key encoder (EMA momentum)", "FIFO queue stores past keys as negatives", "Negative count decoupled from batch size"],
  contribution: "Many consistent negatives even with a small batch.",
  diagram: (s, o) => siamese(s, o, { top: "f_q\n(grad)", bot: "f_k", botNote: "EMA momentum", proj: false, loss: "InfoNCE", extra: "Queue\n(history-key negs)" }),
  loss: "−log exp(q·k₊/τ) / Σ_{queue} exp(q·k/τ)", lossNote: "Queue supplies many negatives",
});

// --- MoCo v2 ---
methodSlide({
  era: A2("ERA 2 · 2020"), name: "MoCo v2", demo: "demo_assets/methods/moco_v2.png", venue: "arXiv 2020",
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
  era: A2("ERA 2 · 2020"), name: "SwAV", demo: "demo_assets/methods/swav.png", venue: "NeurIPS 2020",
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
  era: A3("ERA 3 · 2021"), name: "Barlow Twins", demo: "demo_assets/methods/barlow_twins.png", venue: "ICML 2021",
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
  era: A4("ERA 4 · 2021"), name: "MoCo v3", demo: "demo_assets/methods/moco_v3.png", venue: "ICCV 2021",
  authors: "X. Chen, S. Xie, K. He — An Empirical Study of Training Self-Supervised Vision Transformers",
  idea: "Bring contrastive learning to the ViT and find the recipe that trains it stably.",
  mechanism: ["Freeze the patch-embedding projection (the key stability fix)", "Symmetric in-batch InfoNCE (drop the queue)", "AdamW + cosine LR (not SGD/LARS), m=0.99"],
  contribution: "Stable recipe for ViT contrastive training; no queue at large batch.",
  diagram: (s, o) => siamese(s, o, { top: "ViT f_q\n(grad)", bot: "ViT f_k", botNote: "EMA momentum", proj: true, loss: "Symmetric\nInfoNCE" }),
  loss: "ctr(q₁,k₂) + ctr(q₂,k₁)", lossNote: "Symmetric InfoNCE, no queue",
});

// --- DINO ---
methodSlide({
  era: A4("ERA 4 · 2021"), name: "DINO", demo: "demo_assets/methods/dino.png", venue: "ICCV 2021",
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
  s.addText("Self-trained SSL (CIFAR-10, ResNet-18, 2×H100, 200 epochs): features go from a single blob at epoch 0 to clear class clusters.", {
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
progSlide(" (1/2 · Era 1–2)", [
  { key: "instance_discrimination", label: "Instance Discrim.", sub: "memory bank", c: E1 },
  { key: "invariant_spread", label: "Invariant Spread", sub: "in-batch softmax", c: E1 },
  { key: "simclr_v1", label: "SimCLR v1", sub: "in-batch contrast", c: E2 },
  { key: "simclr_v2", label: "SimCLR v2", sub: "deeper 3-layer head", c: E2 },
], NOTES.prog1);
progSlide(" (2/2 · Era 2–3)", [
  { key: "infomin", label: "InfoMin", sub: "view design", c: E2 },
  { key: "byol", label: "BYOL", sub: "predictor + EMA", c: E3 },
  { key: "simsiam", label: "SimSiam", sub: "stop-gradient", c: E3 },
], NOTES.prog2);

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
