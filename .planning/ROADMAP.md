# Roadmap: Contrastive Learning Tutorial Repository

## Overview

This roadmap builds the tutorial repository from shared infrastructure outward through each era of contrastive and self-supervised learning. Phase 1 establishes the foundation every method depends on. Phases 2–8 implement methods chronologically, each era building conceptually on the prior. Phase 9 delivers the evaluation suite that lets users compare methods. Phase 10 produces the documentation and tutorial content that makes the repo educational.

## Phases

- [x] **Phase 1: Foundation** - Shared base class, config schema, timm backbone factory, projection head, augmentation pipeline, data module, dispatcher, EMA updater, and logging wiring (completed 2026-03-31)
- [ ] **Phase 2: Proxy Tasks Era** - Instance Discrimination (memory bank, NCE loss) and Invariant Spread (in-batch softmax baseline)
- [ ] **Phase 3: SimCLR** - SimCLR v1 and v2 with NT-Xent loss, LARS optimizer, and augmentation pipeline
- [ ] **Phase 4: MoCo** - MoCo v1 and v2 with momentum encoder, FIFO queue, and shuffled BN
- [ ] **Phase 5: SwAV and InfoMin** - Prototype assignment, Sinkhorn-Knopp OT, multi-crop dataset, and InfoMin augmentation demo
- [ ] **Phase 6: No-Negative Methods** - BYOL, SimSiam, and Barlow Twins with collapse monitoring
- [ ] **Phase 7: Transformer Era** - MoCo v3, DINO, and DINOv2 feature-extraction tutorial
- [ ] **Phase 8: Supervised Contrastive** - SupCon loss, class-balanced sampler, two-stage training
- [ ] **Phase 9: Evaluation Suite** - k-NN callback, linear probe, t-SNE, UMAP, fine-tuning, and CAM visualization
- [ ] **Phase 10: Documentation and Tutorial** - README, per-method docstrings, walkthrough notebook

## Phase Details

### Phase 1: Foundation
**Goal**: The shared infrastructure that every method subclass can build on is in place and verified
**Depends on**: Nothing (first phase)
**Requirements**: FOUND-01, FOUND-02, FOUND-03, FOUND-04, FOUND-05, FOUND-06, FOUND-07, FOUND-08, FOUND-09, FOUND-10, INFRA-01, INFRA-06
**Success Criteria** (what must be TRUE):
  1. `BaseSSLModule` can be subclassed with a single `training_step` override and the resulting module trains for one epoch without error on a toy ImageFolder dataset
  2. `build_backbone("resnet50")` and `build_backbone("vit_small_patch16_224")` both return `(backbone, feat_dim)` where `feat_dim == backbone.num_features`
  3. A YAML config file loads via `TrainConfig.model_validate(yaml.safe_load(...))` and produces typed per-method sub-configs with Pydantic validation errors on invalid input
  4. `ProjectionHead` with `num_layers=2` and `num_layers=3` produces the correct BN+ReLU pattern on intermediate layers and BN-only on the final layer
  5. `EMAUpdater.step()` updates target parameters and the target parameters never appear in `learnable_params`
  6. `InfoNCELoss` produces a finite loss value in symmetric (SimCLR) and asymmetric (MoCo queue) modes
  7. `SSLDataModule` with `n_views=2` yields batches of shape `[2, B, C, H, W]` and with `n_views=8` yields batches of shape `[8, B, C, H, W]`

**Plans**: 7 plans

Plans:
- [x] 01-01-PLAN.md — Project scaffold (requirements.txt, pyproject.toml, package inits, test fixtures) + Pydantic v2 `TrainConfig` / `EvalConfig` config schema
- [x] 01-02-PLAN.md — `build_backbone()` timm factory + `ProjectionHead` reusable MLP
- [x] 01-03-PLAN.md — `InfoNCELoss` (symmetric + asymmetric modes) + `LARS` optimizer from scratch
- [x] 01-04-PLAN.md — `ContrastiveAugmentation` (strong/weak paths) + `SSLDataModule` (multi-view ImageFolder)
- [x] 01-05-PLAN.md — `EMAUpdater` with cosine-scheduled momentum
- [x] 01-06-PLAN.md — `BaseSSLModule` abstract base class (configure_optimizers, EMA hook, logging) + `core/__init__.py` re-exports
- [x] 01-07-PLAN.md — `method_dispatcher` factory with registry pattern

**UI hint**: no

---

### Phase 2: Proxy Tasks Era
**Goal**: Instance Discrimination and Invariant Spread are fully working methods that train through `BaseSSLModule` and document the memory-bank era's core ideas and limitations
**Depends on**: Phase 1
**Requirements**: ERA1-01, ERA1-02, INFRA-02
**Success Criteria** (what must be TRUE):
  1. `InstanceDiscriminationModule` trains for 5 epochs on CIFAR-10 without loss divergence; the normalization constant Z is fixed after the first mini-batch estimation and does not change thereafter
  2. `MemoryBank` correctly retrieves and updates feature vectors by index; stale-key behavior is documented in the class docstring with a cross-reference to why MoCo's queue solves this
  3. `InvariantSpreadModule` trains for 5 epochs and its loss decreases monotonically in the first 3 epochs; per-method config YAML for both methods is provided in `configs/`
  4. Both methods are registered in `method_dispatcher` and selectable via `method: instance_discrimination` and `method: invariant_spread` in a YAML config

**Plans**: 5 plans

Plans:
- [x] 02-01: Implement `MemoryBank(n_samples, dim)` as `nn.Embedding` with `update(indices, features)` and `get(indices)` interface; initialize with L2-normalized random vectors; write tests for update-by-index correctness
- [x] 02-02: Implement NCE loss function — (m+1)-way NCE with sampled negatives from the bank, temperature τ=0.07, normalization constant Z estimated on first batch and fixed, ε=1e-7 in denominator; write unit test that Z does not change after first call
- [x] 02-03: Implement `InstanceDiscriminationModule(BaseSSLModule)` — encoder, `MemoryBank`, NCE loss, weak augmentation via `ContrastiveAugmentation(strong=False)`; update bank each step with current encoder output (no EMA); register in dispatcher
- [x] 02-04: Implement `InvariantSpreadModule(BaseSSLModule)` — in-batch cross-entropy loss reusing `InfoNCELoss` in symmetric mode; one augmented view per image; document batch-size sensitivity in docstring; register in dispatcher
- [x] 02-05: Write per-method YAML configs (`configs/instance_discrimination_resnet18.yaml`, `configs/invariant_spread_resnet18.yaml`); add method docstrings per DOC-02 minimum content (paper, gotchas, reference implementation link); smoke-test both configs end-to-end

**UI hint**: no

---

### Phase 3: SimCLR
**Goal**: SimCLR v1 and v2 are working with correct NT-Xent loss, verified augmentation pipeline, and LARS optimizer — establishing the canonical "two-view in-batch contrastive" pattern that later methods reference
**Depends on**: Phase 1
**Requirements**: ERA2-03, ERA2-04
**Success Criteria** (what must be TRUE):
  1. `SimCLRv1Module` trains on CIFAR-10 for 5 epochs without loss divergence; color jitter strength is confirmed as `s=1.0` (not torchvision default) via a config assertion or comment in the augmentation code
  2. NT-Xent loss is symmetric: `loss(z1, z2) == loss(z2, z1)` verified in a unit test
  3. `SimCLRv2Module` uses a 3-layer projection head; switching from `num_layers=2` to `num_layers=3` via config changes only the projection depth and nothing else
  4. Both methods are selectable via `method: simclr_v1` and `method: simclr_v2` in YAML; per-method configs exist in `configs/`
  5. LARS optimizer activates when `optimizer: lars` is set in config; AdamW is the default

**Plans**: 3 plans

Plans:
- [x] 03-01-PLAN.md — SimCLRv1Module + SimCLRv2Module implementation with dispatcher registration and comprehensive test suite (NT-Xent symmetry, projection head depth, training, dispatcher)
- [x] 03-02-PLAN.md — Per-method YAML configs (v1 AdamW, v1 LARS, v2) and augmentation visualization script at tools/visualize_augmentations.py
- [x] 03-03-PLAN.md — DOC-02 docstrings for both modules and end-to-end smoke tests from YAML configs

**UI hint**: no

---

### Phase 4: MoCo
**Goal**: MoCo v1 and v2 are working with correct momentum encoder, FIFO queue, and documented shuffled-BN requirement — establishing the queue-based contrastive pattern and its evolution
**Depends on**: Phase 1
**Requirements**: ERA2-01, ERA2-02, INFRA-03
**Success Criteria** (what must be TRUE):
  1. `MomentumQueue` correctly enqueues new keys and dequeues oldest keys FIFO; after filling, the queue size stays at exactly K; verified in unit tests
  2. Momentum encoder parameters do not appear in any optimizer param group; verified by asserting `set(learnable_params ids)` is disjoint from `set(ema_params ids)`
  3. EMA update occurs in `on_train_batch_end` (not `training_step`); verified by mock-inspecting call order in a unit test
  4. `MoCoV1Module` and `MoCoV2Module` both train 5 epochs without loss divergence; MoCo v2 uses the 2-layer MLP projection head while v1 uses a single FC layer
  5. Both methods are selectable via `method: moco_v1` and `method: moco_v2` in YAML

**Plans**: 3 plans

Plans:
- [x] 04-01-PLAN.md — `MomentumQueue(queue_size, dim)` FIFO buffer with register_buffer, L2-normalized init, pointer wrap-around, comprehensive unit tests, core/__init__.py re-export
- [x] 04-02-PLAN.md — `MoCoV1Module` + `MoCoV2Module` implementation with momentum encoder, FIFO queue, InfoNCELoss queue mode, EMA via BaseSSLModule hook, dispatcher registration, and full test suite
- [x] 04-03-PLAN.md — Per-method YAML configs (v1 SGD, v2 SGD), DOC-02 docstrings with shuffled-BN and momentum sensitivity gotchas, end-to-end smoke tests

**UI hint**: no

---

### Phase 5: SwAV and InfoMin
**Goal**: SwAV's online clustering with multi-crop is working; `MultiCropDataset` is a reusable component; InfoMin is presented as an augmentation-policy demonstration on top of the existing SimCLR/MoCo backbone
**Depends on**: Phase 1, Phase 3 (InfoMin reuses SimCLR)
**Requirements**: ERA2-05, ERA2-06, INFRA-04
**Success Criteria** (what must be TRUE):
  1. `MultiCropDataset` with `n_large_crops=2, n_small_crops=6` yields batches where large crops are 224x224 and small crops are 96x96; configurable crop counts via YAML
  2. Sinkhorn-Knopp iteration produces a code matrix `Q` that is doubly stochastic — row sums and column sums are both uniform; verified by assertion in a unit test
  3. Prototype vectors are frozen during the first `freeze_prototypes_epochs` epochs; after that epoch boundary, gradients flow through the prototype layer
  4. `SwAVModule` trains for 5 epochs without loss divergence; prototype vectors remain L2-normalized after each optimizer step
  5. InfoMin augmentation demo is runnable via `method: infomin` config and produces a side-by-side augmentation comparison output

**Plans**: 7 plans

Plans:
- [x] 05-01: Implement `MultiCropDataset` wrapper — accepts `(dataset, n_large_crops, large_size, n_small_crops, small_size)`; applies separate `ContrastiveAugmentation` instances for large and small crops; yields a list of `n_large + n_small` tensors per sample; register in `SSLDataModule` when `n_views > 2`
- [x] 05-02: Implement Sinkhorn-Knopp optimal transport — `sinkhorn_knopp(scores, n_iters=3, epsilon=0.05)` returning doubly-stochastic code matrix `Q`; write unit test asserting row-sum uniformity and column-sum uniformity after convergence
- [x] 05-03: Implement prototype layer — `nn.Linear(feat_dim, n_prototypes, bias=False)` with L2-normalization hook applied after every optimizer step via `on_after_backward` + manual renormalization; `freeze_prototypes_epochs` config parameter that zeroes prototype gradients before the threshold epoch
- [x] 05-04: Implement swapped-prediction loss — given codes `(q1, q2)` and features `(z1, z2)`, compute `cross_entropy(z1 @ C.T / tau, q2) + cross_entropy(z2 @ C.T / tau, q1)` averaged over crops; handle multi-crop by iterating over all small crop views
- [x] 05-05: Implement `SwAVModule(BaseSSLModule)` — `MultiCropDataset` integration, prototype layer, Sinkhorn-Knopp codes, swapped-prediction loss, prototype freeze logic; add `learnable_params` override that includes prototype parameters; register as `swav`
- [x] 05-06: Implement `InfoMinModule` as a thin wrapper around `SimCLRv1Module` or `MoCoV2Module` that substitutes a "minimal-MI" augmentation policy (aggressive color jitter + random grayscale + no blur); include a comparison script that visualizes standard SimCLR augmentation vs. InfoMin augmentation side-by-side; document the full semi-supervised view-learning variant as v2 scope
- [x] 05-07: Write per-method YAML configs for SwAV (2-large + 6-small crop variant) and InfoMin; add DOC-02 docstrings; smoke-test both for 3 epochs; document memory usage warning for 8-crop configuration in config YAML comments

**UI hint**: no

---

### Phase 6: No-Negative Methods
**Goal**: BYOL, SimSiam, and Barlow Twins are implemented with collapse monitoring — demonstrating that strong representations are achievable without explicit negatives and surfacing the common failure modes
**Depends on**: Phase 1
**Requirements**: ERA3-01, ERA3-02, ERA3-03
**Success Criteria** (what must be TRUE):
  1. `BYOLModule` trains for 5 epochs; `z.std(dim=0).mean()` is logged as `train/embedding_std` and stays above 0.1 throughout; predictor is on the online branch only
  2. `SimSiamModule` trains for 5 epochs; removing `.detach()` from `z` in the loss causes immediate collapse to loss=-1.0 within 2 epochs (documented via a comment/test), confirming the stop-gradient is load-bearing
  3. `BarlowTwinsModule` trains for 5 epochs; the cross-correlation matrix `C` has diagonal values > 0.5 by epoch 5 on CIFAR-10 (verified in a diagnostic log)
  4. All three methods are selectable via `method: byol`, `method: simsiam`, `method: barlow_twins` in YAML
  5. EMA momentum schedule (cosine 0.996->1.0) is used in BYOL; a unit test asserts momentum at step 0 and step `total_steps` match expected values

**Plans**: 7 plans

Plans:
- [ ] 06-01: Implement `PredictorHead` — reuse or extend `ProjectionHead` for standard 2-layer predictor (BYOL) and bottleneck 2-layer predictor (SimSiam: 2048->512->2048, BN on all layers including output, no ReLU on output); expose both variants via `predictor_type` config
- [ ] 06-02: Implement `BYOLModule(BaseSSLModule)` — online network (backbone + projector + predictor) and target network (backbone + projector via EMA, no predictor); MSE loss on L2-normalized outputs (`2 - 2*cosine_similarity`); cosine-scheduled EMA momentum via `EMAUpdater`; log `train/embedding_std`; register as `byol`
- [ ] 06-03: Implement stop-gradient validation for `BYOLModule` — unit test that asserts target branch parameters receive zero gradient during `training_step`; add comment marking the stop-gradient site in source
- [ ] 06-04: Implement `SimSiamModule(BaseSSLModule)` — shared encoder for both views; bottleneck predictor; symmetric stop-gradient loss `-(cosim(p1, z2.detach()) + cosim(p2, z1.detach())) / 2`; log `train/embedding_std`; add comment at `.detach()` call with collapse warning; register as `simsiam`
- [ ] 06-05: Implement `BarlowTwinsModule(BaseSSLModule)` — high-dimensional projector (8192-dim output, 3 layers, BN+ReLU on first 2, BN on output); cross-correlation matrix normalized by batch size; loss driving `C` toward identity; lambda=5e-3 as configurable parameter; log diagonal mean of `C`; register as `barlow_twins`
- [ ] 06-06: Add collapse monitoring to all three modules — `z.std(dim=0).mean()` logged as `train/embedding_std`; add docstring note that collapse is indicated when this value approaches 0
- [ ] 06-07: Write per-method YAML configs; add DOC-02 docstrings with collapse gotchas prominently listed; smoke-test all three for 3 epochs

**UI hint**: no

---

### Phase 7: Transformer Era
**Goal**: MoCo v3, DINO, and DINOv2 (feature extraction) are implemented — demonstrating how the contrastive/self-distillation paradigm transfers to Vision Transformers and establishing the patch-projection freeze and centering tricks
**Depends on**: Phase 1, Phase 4 (MoCo v3 extends MoCo), Phase 6 (DINO student-teacher pattern is BYOL-inspired)
**Requirements**: ERA4-01, ERA4-02, ERA4-03, INFRA-05
**Success Criteria** (what must be TRUE):
  1. `MoCoV3Module` trains for 3 epochs with a ViT-Small backbone; patch projection layer is frozen from epoch 0; verified by asserting `requires_grad=False` on `backbone.patch_embed.proj.weight`
  2. `DINOModule` trains for 3 epochs with ViT-Small; centering vector is updated before loss computation each step; teacher receives only global crops; student receives all crops
  3. `DINOv2Tutorial` script loads a pretrained `vit_small_patch14_dinov2` via timm, runs zero-shot k-NN evaluation, and produces a linear probe accuracy number on a small downstream dataset
  4. All three methods are selectable via YAML; MoCo v3 uses AdamW by default; gradient clipping is enabled
  5. `PredictorHead` is shared between BYOL, SimSiam, MoCo v3, and DINO without code duplication

**Plans**: 4 plans

Plans:
- [x] 07-01-PLAN.md — MoCoV3Config, DINOConfig extension, gradient_clip_val on TrainConfig, PredictorHead docstring update (INFRA-05)
- [x] 07-02-PLAN.md — MoCoV3Module implementation with ViT patch freeze, symmetric in-batch loss, prediction MLP, momentum encoder, tests
- [x] 07-03-PLAN.md — DINOModule implementation with student-teacher self-distillation, centering + sharpening, multi-crop, tests
- [x] 07-04-PLAN.md — DINOv2 demo script, YAML configs for moco_v3 and dino, DOC-02 docstrings, smoke tests

**UI hint**: no

---

### Phase 8: Supervised Contrastive
**Goal**: SupCon is implemented with the class-balanced sampler and correct sum-outside loss formulation, demonstrating how the self-supervised contrastive loss extends to a supervised setting
**Depends on**: Phase 1, Phase 3 (SupCon generalizes SimCLR's NT-Xent)
**Requirements**: SUP-01
**Success Criteria** (what must be TRUE):
  1. `SupConLoss(temperature, labels=None)` produces identical output to `InfoNCELoss` (SimCLR mode) when `labels=None` — verified in a unit test
  2. With `labels` provided, positives for anchor `i` include all other samples with the same class label, not just the other augmented view; verified by checking loss value is lower when more same-class images are in the batch
  3. Class-balanced sampler guarantees at least 2 instances per class per batch; verified by asserting `min(class_counts_per_batch) >= 2` in a sampler unit test
  4. Two-stage training works: stage 1 trains SupCon pretraining (no classifier), stage 2 freezes encoder and trains linear head; the two stages are invoked sequentially from a single config or two configs
  5. `method: supcon` is selectable via YAML

**Plans**: 5 plans

Plans:
- [x] 08-01: Implement `SupConLoss(temperature, reduction, labels=None)` — when `labels=None`, degenerate to SimCLR NT-Xent (one positive per anchor); when labels provided, use sum-outside formulation (Eq. 2 in paper): `loss_i = -1/|P(i)| * sum_{p in P(i)} [s_{ip}/tau - log sum_{a != i} exp(s_{ia}/tau)]`; write unit test confirming SimCLR equivalence when labels=None
- [x] 08-02: Implement class-balanced sampler `ClassBalancedSampler(dataset, n_classes_per_batch, n_samples_per_class)` — guarantees at least `n_samples_per_class` per class per batch; integrate into `SSLDataModule` under `sampler: class_balanced` config key
- [x] 08-03: Implement `SupConModule(BaseSSLModule)` — two augmented views, 2-layer projection head, `SupConLoss` with labels from batch; stage-1-only (no classifier during pretraining); `ClassBalancedSampler` wired in; register as `supcon`
- [x] 08-04: Implement stage-2 fine-tuning via `LinearProbeModule` reuse or a dedicated `SupConFinetuneModule` — freeze backbone, train linear head with SGD, weight_decay=0.0; document two-stage workflow in module docstring with explicit command sequence
- [ ] 08-05: Write YAML config for both stages; add DOC-02 docstring with sum-outside gotcha, class-sampler requirement, and two-stage training note; smoke-test stage 1 for 3 epochs

**UI hint**: no

---

### Phase 9: Evaluation Suite
**Goal**: A complete evaluation toolkit exists that can measure and visualize representation quality for any trained method without modifying method code
**Depends on**: Phase 1 (all evaluation tools operate on backbone outputs)
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06, FOUND-08
**Success Criteria** (what must be TRUE):
  1. `KNNCallback` runs every `every_n_epochs` epochs during training and logs `eval/knn_acc` visible in TensorBoard
  2. `eval/linear_probe.py` loads a checkpoint, freezes the backbone, trains a linear head for 100 epochs, and reports top-1 accuracy; weight_decay is 0.0 on the linear head by default
  3. `eval/tsne_vis.py` runs on 2000 samples with perplexity values 10, 30, 50 and saves three PNG files distinguishable by perplexity value in the filename
  4. `eval/umap_vis.py` runs on up to 5000 samples and saves a PNG; for datasets >50K samples the script prints a note suggesting `torchdr` GPU path
  5. `eval/finetune.py` uses separate LR groups (backbone 1e-4, head 1e-3) and the fine-tuned model reaches higher accuracy than the linear probe on the same dataset
  6. `eval/cam_vis.py` runs on 8 reference images using `EigenCAM` by default for SSL (no classifier); switches to `GradCAM` when a classifier is present; saves CAM overlay images

**Plans**: 5 plans

Plans:
- [x] 09-01-PLAN.md — KNNCallback (Lightning Callback) with FAISS/brute-force k-NN, temperature-scaled weighted voting, epoch scheduling, eval dependencies installation
- [x] 09-02-PLAN.md — LinearProbeModule + eval/linear_probe.py with frozen backbone, SGD weight_decay=0.0, MultiStepLR, feature caching to disk
- [x] 09-03-PLAN.md — eval/tsne_vis.py (PCA + t-SNE perplexity sweep) and eval/umap_vis.py (cosine UMAP with torchdr suggestion)
- [x] 09-04-PLAN.md — eval/finetune.py (FinetuneModule with dual LR groups, freeze_bn) and eval/cam_vis.py (EigenCAM/GradCAM with architecture-aware target layers)
- [x] 09-05-PLAN.md — Integration test: synthetic checkpoint + synthetic ImageFolder, full eval pipeline smoke test

**UI hint**: no

---

### Phase 10: Documentation and Tutorial
**Goal**: The repository is ready to publish as a tutorial — every method is documented, the README enables a first-time user to run an experiment end-to-end, and the walkthrough guide explains how to add a new method
**Depends on**: All prior phases
**Requirements**: DOC-01, DOC-02, DOC-03
**Success Criteria** (what must be TRUE):
  1. `README.md` contains: project overview, `pip install -r requirements.txt` installation, a single-command SimCLR training invocation, config system explanation, method table with era/venue/contribution columns, and evaluation instructions
  2. Every `LightningModule` subclass has a docstring containing: paper title, authors, venue, year, arXiv/DOI link, 2-sentence algorithm description, gotcha list, reference implementation URL
  3. The tutorial notebook or guide demonstrates: (a) adding a new method by subclassing `BaseSSLModule`, (b) running a full experiment from config to evaluation, (c) comparing two methods on the same dataset using the evaluation suite
  4. A new user following the README can train SimCLR on CIFAR-10 and run k-NN evaluation in under 5 commands with no undocumented steps
  5. The method table in `README.md` is complete — all 14 v1 methods are listed with correct era, venue, and primary contribution

**Plans**: 6 plans

Plans:
**Wave 1**
- [ ] 10-01-PLAN.md — Create train.py + tests/test_train_script.py + README.md (DOC-01 — overview, install, quickstart, method table for all 14 v1 methods, eval CLI)
- [ ] 10-02-PLAN.md — Audit and complete DOC-02 class docstrings for all 8 non-compliant LightningModule subclasses + create tests/test_docstrings.py (DOC-02 enforcement)

**Wave 2** *(blocked on Wave 1 completion)*
- [ ] 10-03-PLAN.md — Tutorial section (a): How to Add a New Method (DOC-03 — subclass BaseSSLModule, register in dispatcher, write YAML, run train.py)
- [ ] 10-04-PLAN.md — Tutorial section (b): Running an Experiment End-to-End (DOC-03 — config -> train.py -> checkpoint -> eval/linear_probe.py + UMAP + k-NN)
- [ ] 10-05-PLAN.md — Tutorial section (c): Comparing Two Methods (DOC-03 — run eval suite on two checkpoints, build comparison table)

**Wave 3** *(blocked on Wave 2 completion)*
- [ ] 10-06-PLAN.md — Assemble docs/tutorial.md (era narrative + sections a+b+c) + final review pass verifying every config/module/CLI reference resolves

**UI hint**: no

---

## Progress

**Execution Order:** 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 7/7 | Complete   | 2026-03-31 |
| 2. Proxy Tasks Era | 5/5 | Complete | 2026-04-02 |
| 3. SimCLR | 3/3 | Complete | - |
| 4. MoCo | 2/3 | In Progress|  |
| 5. SwAV and InfoMin | 0/7 | Not started | - |
| 6. No-Negative Methods | 0/7 | Not started | - |
| 7. Transformer Era | 0/8 | Not started | - |
| 8. Supervised Contrastive | 0/5 | Not started | - |
| 9. Evaluation Suite | 0/5 | Not started | - |
| 10. Documentation and Tutorial | 0/6 | Not started | - |
