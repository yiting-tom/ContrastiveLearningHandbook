# Requirements: Contrastive Learning Tutorial Repo

**Defined:** 2026-03-29
**Core Value:** Any contrastive learning method can be implemented by conforming to a shared interface and immediately work with the same dataset pipeline, timm backbone, and Lightning training loop.

---

## v1 Requirements

### Foundation

- [ ] **FOUND-01**: `BaseSSLModule(LightningModule)` abstract base class with `build_projector()`, `learnable_params` property, `configure_optimizers()` (AdamW/SGD/LARS dispatch + warmup-cosine scheduler), and `on_train_batch_end()` hook for EMA updates. All method subclasses must extend this class — no standalone training scripts.

- [x] **FOUND-02**: Pydantic v2 config schema (`TrainConfig`) with top-level fields for `method`, `backbone`, `pretrained`, `max_epochs`, `warmup_epochs`, `batch_size`, `lr`, `weight_decay`, `optimizer`, `scheduler`, plus namespaced per-method sub-configs (e.g., `simclr.temperature`). YAML is loaded via `yaml.safe_load` and validated by `TrainConfig.model_validate(raw)`. No Hydra. No OmegaConf.

- [ ] **FOUND-03**: `build_backbone(model_name, pretrained=False)` factory using `timm.create_model(..., num_classes=0)`. Must return `(backbone, feat_dim)` where `feat_dim = backbone.num_features`. Never hard-code feature dimensions or use `backbone.inplanes` (ResNet-only). Supports any timm architecture name; bare names (`resnet50`, `vit_small_patch16_224`) resolve correctly under timm 1.x.

- [ ] **FOUND-04**: `ProjectionHead(input_dim, hidden_dim, output_dim, num_layers, use_bn=True)` reusable MLP. Configurable depth, width, and output normalization. BN + ReLU on intermediate layers; BN only (no ReLU) on final layer. Used by all methods — no duplicated MLP code.

- [x] **FOUND-05**: `ContrastiveAugmentation(size, n_views, strong=True)` shared augmentation pipeline: random crop + resize, color jitter (strength s=1.0 for strong), random grayscale (p=0.2), Gaussian blur (sigma 0.1–2.0, p=0.5), horizontal flip. A `strong=False` path covers era-1 methods (Instance Discrimination, CMC) which use weaker augmentation. Uses `torchvision.transforms.v2` API.

- [x] **FOUND-06**: `SSLDataModule(LightningDataModule)` wrapping either ImageFolder-style data directories or a custom dataset class. Accepts `n_views` to produce the correct number of augmented views per sample. DataModule is shared across all methods; per-method differences are in `n_views` only (2 for most, 8+ for SwAV/DINO multi-crop).

- [ ] **FOUND-07**: `method_dispatcher(cfg: TrainConfig) -> BaseSSLModule` factory. Maps the `method` string in config to the correct `LightningModule` subclass. Raises `ValueError` with available methods listed when an unknown method is specified.

- [x] **FOUND-08**: `EvalConfig` Pydantic sub-schema added to `TrainConfig` under an `eval:` key. Sub-configs: `LinearProbeConfig`, `KNNConfig`, `TSNEConfig`, `UMAPConfig`, `FinetuneConfig`, `CAMConfig`. All evaluation settings live in the same YAML as pretraining — one file per experiment.

- [ ] **FOUND-09**: TensorBoard logging wired by default. At minimum, log `train/loss`, `train/lr`, and `eval/knn_acc` (when k-NN callback is enabled). Logging is handled through `self.log(...)` / `self.log_dict(...)` in the base class — no per-method logging boilerplate.

- [x] **FOUND-10**: `EMAUpdater` utility class with `base_momentum`, `end_momentum`, `total_steps` parameters and a `step(online_params, target_params)` method. Shared by MoCo v1/v2/v3, BYOL, and DINO. EMA update must happen in `on_train_batch_end`, not in `training_step` or `on_before_optimizer_step`. Momentum encoder parameters must never appear in `learnable_params` / optimizer param groups.

---

### Era 1: Proxy Tasks (2018–2019)

- [ ] **ERA1-01**: **Instance Discrimination** (Wu et al., CVPR 2018). Encoder + non-parametric memory bank (one L2-normalized vector per training sample). InfoNCE/NCE loss with NCE sample size of 4096 negatives sampled from the bank. Normalization constant Z estimated on first mini-batch and fixed. Memory bank updated with current encoder outputs each step (no EMA). Temperature τ = 0.07. Numerical stability: add ε = 1e-7 in NCE denominator. **Gotcha:** Z must be fixed after initial estimation; recomputing it each step destabilizes training. **Gotcha:** Bank staleness (keys from earlier encoder snapshots) is a known limitation — document it explicitly as the problem MoCo solves.

- [ ] **ERA1-02**: **Invariant Spread** (Ye et al., CVPR 2019). In-batch softmax loss without memory bank or queue. One augmented view per image. Cross-entropy over in-batch instances. Simple re-use of `InfoNCELoss` with in-batch negatives only. **Note:** Performance is sensitive to batch size (unlike MoCo/SimCLR); document this. Present as the direct ancestor of SimCLR.

---

### Era 2: MoCo and SimCLR (2019–2020)

- [ ] **ERA2-01**: **MoCo v1** (He et al., CVPR 2020). Query encoder + momentum encoder (EMA copy, `deactivate_requires_grad`). FIFO queue of size K=65,536 for negative keys. InfoNCE loss. EMA momentum m=0.999. **Gotcha:** Shuffling BN is required for multi-GPU training — keys must be shuffled across GPUs before encoding to prevent BN leaking positive-pair statistics. **Gotcha:** m=0.9 produces significantly worse results; document the sensitivity. `MomentumQueue` class (FIFO buffer with pointer) is a standalone reusable component.

- [ ] **ERA2-02**: **MoCo v2** (Chen et al., 2020 tech report). MoCo v1 + 2-layer MLP projection head + Gaussian blur augmentation + cosine LR schedule. Queue, momentum encoder, and EMA update unchanged. Implementation is a subclass of MoCo v1 or a 5-line config diff. Present as "SimCLR's architecture improvements applied to MoCo."

- [ ] **ERA2-03**: **SimCLR v1** (Chen et al., ICML 2020). In-batch NT-Xent loss (symmetric). Shared encoder for both views. 2-layer MLP projection head (128-dim output). Loss computed on `z` (projection output), evaluation done on `h` (backbone output). **Dependency:** LARS optimizer required for batch sizes above ~1024. Provide `LARS` implementation or import from `lightly`. **Gotcha:** Color jitter strength must be s=1.0 (stronger than torchvision default); Gaussian blur is required. **Gotcha:** Performance degrades sharply below batch size 256 — document the batch size sensitivity. This is the recommended first implementation in the tutorial sequence.

- [ ] **ERA2-04**: **SimCLR v2** (Chen et al., NeurIPS 2020). SimCLR v1 + 3-layer MLP projection head (depth 3 instead of 2). Full semi-supervised distillation stage is out of v1 scope (deferred to v2 requirements); implement the pretraining stage only. **Gotcha:** Weight norm scales differ from SimCLR v1 due to different weight decay; document this before applying v1 fine-tune hyperparameters.

- [ ] **ERA2-05**: **SwAV** (Caron et al., NeurIPS 2020). Online clustering via Sinkhorn-Knopp optimal transport. Learnable prototype vectors C (K=3000 for ImageNet, configurable) normalized after every update step. Swapped-prediction loss. Multi-crop strategy: 2 large crops (224×224) + 6 small crops (96×96) via `MultiCropDataset` wrapper. **Gotcha:** Prototypes must be frozen for the first epoch; implement a `freeze_prototypes_epochs` config parameter. **Gotcha:** Sinkhorn-Knopp code matrix must be doubly stochastic — the implementation must enforce equal prototype usage. **Gotcha:** Prototype vectors must be L2-normalized at every update step. Memory usage with 8 crops is ~4× SimCLR; document required batch size reduction.

- [ ] **ERA2-06**: **InfoMin** (Tian et al., NeurIPS 2020). Implement as an augmentation-policy demonstration on top of SimCLR or MoCo v2 backbone — present the InfoMin principle (minimal mutual information between views, maximal task-relevant information) rather than the full semi-supervised view-learning framework. **Note:** Full view-learning requires a labeled subset; implement the "strong augmentation" interpretation as the practical tutorial version. Full semi-supervised variant deferred to v2.

---

### Era 3: No-Negative Methods (2020)

- [ ] **ERA3-01**: **BYOL** (Grill et al., NeurIPS 2020). Online network (backbone + projector + predictor) and target network (backbone + projector, no predictor). MSE loss between L2-normalized predictor output and L2-normalized target projection. Target updated via EMA with cosine-scheduled momentum (0.996 → 1.0, not a fixed value). **Gotcha:** The EMA momentum schedule — not a fixed value — is critical; fixed m=0.99 degrades performance measurably. **Gotcha:** Removing the predictor from the online branch causes immediate collapse. **Gotcha:** BN in projector/predictor is load-bearing; do not replace with LayerNorm without validation. **Gotcha:** EMA update must use `on_train_batch_end`, not `on_before_optimizer_step`. Loss must operate on L2-normalized vectors (equivalent to `2 - 2·cos_sim`).

- [ ] **ERA3-02**: **SimSiam** (Chen & He, CVPR 2021). Shared encoder for both views. Projector MLP (3 layers, 2048 output, BN on all layers including output, no ReLU on output layer). Predictor MLP (2-layer bottleneck: 2048→512→2048). Stop-gradient applied to z in the symmetric loss: `loss = -(cos_sim(p1, z2.detach()) + cos_sim(p2, z1.detach())) / 2`. **Gotcha:** Forgetting `.detach()` causes immediate collapse to loss = -1.0 plateau — this is the single most common implementation bug. **Gotcha:** Final projector layer has BN but no ReLU; adding ReLU hurts performance. **Gotcha:** Use SGD not LARS (lr = 0.05 × batch/256). **Mandatory training diagnostic:** log `z.std(dim=0).mean()` — should stay near 0.707; if it approaches 0, model has collapsed.

- [ ] **ERA3-03**: **Barlow Twins** (Zbontar et al., ICML 2021). Shared encoder for both views. High-dimensional projector (3 layers: 2048→8192→8192→8192, BN + ReLU on first 2 layers, BN on output). Cross-correlation matrix C normalized by batch size. Loss drives C toward identity matrix. λ = 5e-3 (redundancy weight). **Gotcha:** C must be normalized by batch size (divide by N) before computing the loss. **Gotcha:** Projector output must be standardized (zero mean, unit variance per dimension across batch) before computing C — BN at the projector output handles this, but the standardization must happen at loss computation time. **Gotcha:** λ is sensitive; document the effect of varying it. Do not reduce projector output dim below 2048 without performance validation. Fully symmetric — no stop-gradient, no EMA, no predictor needed.

---

### Era 4: Transformer Era (2021+)

- [ ] **ERA4-01**: **MoCo v3** (Chen et al., ICCV 2021). Extends MoCo to ViT backbones. Queue removed; large batch (in-batch keys). Symmetric contrastive loss (two query-key pairs from two views). Momentum encoder with m=0.99 (not 0.999 — this is different from v1/v2). Prediction MLP on top of projection MLP. **Gotcha:** Freeze the patch projection layer (first ViT layer) — this is the key stability fix; without it, ViT training silently produces poor representations. **Gotcha:** AdamW for ViT backbones, not SGD/LARS. **Gotcha:** m=0.99 optimal; m=0.999 gives 1–2% worse accuracy. Warmup 40 epochs. Gradient clipping recommended.

- [ ] **ERA4-02**: **DINO** (Caron et al., ICCV 2021). Student-teacher framework. Teacher is EMA of student; both are the same ViT architecture. Teacher receives global crops only; student receives all crops. Loss is cross-entropy between student and teacher softmax outputs. Collapse prevention: centering (running mean subtracted from teacher output, updated before loss computation) + sharpening (low teacher temperature τ_t with warmup from 0.04→0.07). Output dim 65,536 (high-dimensional softmax is important for quality). Multi-crop: 2 global (224) + 6–10 local (96). Gradient clipping max_norm=3.0 is required. **Gotcha:** Centering vector must be updated with teacher outputs before computing the loss. **Gotcha:** Teacher temperature warmup is critical; starting at 0.07 immediately causes early instability. **Gotcha:** Passing local crops to the teacher changes the objective semantics — teacher receives global crops only.

- [ ] **ERA4-03**: **DINOv2** (Oquab et al., 2023). Implement as a **feature extraction and fine-tuning** tutorial, not as a from-scratch training implementation (training requires hundreds of GPU-days and LVD-142M dataset). Load pretrained DINOv2 checkpoints (ViT-S/14, ViT-B/14) via the official `facebookresearch/dinov2` repo or `timm`. Demonstrate: (1) zero-shot k-NN evaluation, (2) linear probing, (3) fine-tuning for a downstream task. **Note:** Register tokens (added Oct 2023) change the model API; use the version that matches the checkpoint. Document that "DINOv3" does not exist — the correct lineage is DINO → DINOv2 → DINOv2 + Registers.

---

### Supervised Contrastive

- [ ] **SUP-01**: **SupCon** (Khosla et al., NeurIPS 2020). Two-stage: (1) supervised contrastive pretraining with SupCon loss + 2-layer MLP projection head; (2) standard cross-entropy fine-tuning with frozen encoder + linear head. SupCon loss extends NT-Xent: positives for anchor i are all other images with the same class label (not just the other view). **Gotcha:** Features must be L2-normalized before the loss. **Gotcha:** Batch sampling must guarantee multiple instances per class per batch — use a class-balanced sampler; without it, SupCon degenerates to SimCLR. **Gotcha:** Use the sum-outside variant (Eq. 2 in paper), not sum-inside. `SupConLoss(temperature, labels=None)` should generalize to SimCLR when labels=None — this design is worth replicating. Two-stage training is important; do not add a classifier during SupCon pretraining.

---

### Shared Infrastructure (Cross-Method)

- [x] **INFRA-01**: `InfoNCELoss(temperature, reduction='mean')` standalone loss module. Covers SimCLR (in-batch symmetric), MoCo v1/v2 (asymmetric with queue), MoCo v3 (symmetric in-batch), Instance Discrimination (NCE approximation), CMC (multi-view), and InfoMin. SupCon variant accepts an optional `labels` parameter.

- [ ] **INFRA-02**: `MemoryBank(n_samples, dim)` implemented as `nn.Embedding` with update-by-index. Shared by Instance Discrimination and CMC.

- [ ] **INFRA-03**: `MomentumQueue(queue_size, dim)` FIFO buffer (torch.zeros initialized, with pointer). Shared by MoCo v1 and MoCo v2. Exposes `get_negatives()` and `update(keys)` interface.

- [ ] **INFRA-04**: `MultiCropDataset` wrapper that applies `n_large_crops` large augmentations and `n_small_crops` small augmentations to each image. Shared by SwAV and DINO.

- [ ] **INFRA-05**: `PredictorHead` (either a re-use of `ProjectionHead` or a separate class for bottleneck architectures). Shared by BYOL, SimSiam, MoCo v3, and DINO.

- [x] **INFRA-06**: `LARS` optimizer implementation. Required by SimCLR v1/v2, SwAV, BYOL, Barlow Twins. Either implement from scratch or import from `lightly`. Pin as an explicit dependency.

---

### Evaluation Suite

- [ ] **EVAL-01**: **k-NN callback** (`KNNCallback(L.Callback)`). Runs every `every_n_epochs` epochs (configurable; 0 = end-of-training only). Uses L2-normalized backbone features. Weighted temperature-scaled voting (DINO / MoCo v3 protocol). Default k=200, τ=0.07. Logs `eval/knn_acc` via `pl_module.log(...)`. For datasets >100K samples, use FAISS `IndexFlatIP` instead of brute-force matrix multiplication. Config: `eval.knn` in YAML.

- [ ] **EVAL-02**: **Linear probe** (`eval/linear_probe.py`, `LinearProbeModule`). Offline script launched after pretraining from a checkpoint. Freezes backbone entirely (`requires_grad_(False)`, kept in `eval()` mode). Trains only the linear head with SGD, weight_decay=0.0 (critical — no regularization on linear head when backbone is frozen), MultiStepLR decay at epochs [60, 80]. For large datasets, pre-extract and cache features once; train the linear head on cached tensors. Reports top-1 accuracy. Config: `eval.linear_probe` in YAML.

- [ ] **EVAL-03**: **t-SNE visualization** (`eval/tsne_vis.py`). Runs on a fixed subset (default 2000 samples). PCA pre-reduction to 50 dims before t-SNE. `init='pca'`, `metric='cosine'`, `learning_rate='auto'` (scikit-learn 1.2+). Sweeps at least 3 perplexity values (10, 30, 50) to check consistency. Run as a one-off script, not a per-epoch callback. Config: `eval.tsne` in YAML.

- [ ] **EVAL-04**: **UMAP visualization** (`eval/umap_vis.py`). Default visualization (preferred over t-SNE as primary). Runs on up to 5000 samples. `metric='cosine'`, `random_state=42`. Returns reducer object for new-sample mapping. For datasets >50K, offer `torchdr` GPU-accelerated UMAP path. Config: `eval.umap` in YAML.

- [ ] **EVAL-05**: **Fine-tuning script** (`eval/finetune.py`, `FinetuneModule`). Unfreezes backbone. Separate LR groups: backbone LR = 1e-4, head LR = 1e-3 (10× lower to prevent catastrophic forgetting). AdamW + warmup-cosine scheduler. `freeze_bn=True` option keeps BN layers in eval mode for small downstream datasets. Config: `eval.finetune` in YAML.

- [ ] **EVAL-06**: **CAM visualization** (`eval/cam_vis.py`). Uses `pytorch-grad-cam` library. Default method: `EigenCAM` (works without a classifier, uses first PC of final layer output — correct default for SSL). `GradCAM` available when a downstream classifier exists. Architecture-aware target layer selection: `backbone.layer4[-1]` for ResNets, `backbone.blocks[-1].norm1` for ViTs (with `vit_reshape_transform` to convert `[B, N_patches+1, D]` to `[B, D, H, W]`). Run as a one-off diagnostic script on 8–16 reference images per class. Config: `eval.cam` in YAML.

---

### Documentation

- [ ] **DOC-01**: `README.md` covering: project overview, installation (`pip install -r requirements.txt`), quick-start (train SimCLR in one command), config system explanation, method list with paper links, evaluation instructions. Include a table mapping each method to its era, publication venue, and primary contribution.

- [ ] **DOC-02**: Per-method docstring in each `LightningModule` subclass. Minimum required content: paper title, authors, venue, year, arXiv/DOI link, 2-sentence algorithm description, list of method-specific gotchas, reference implementation URL.

- [ ] **DOC-03**: Tutorial notebook or guide (`notebooks/walkthrough.ipynb` or `docs/tutorial.md`). Covers: (1) how to add a new method (implement the interface, register in dispatcher); (2) how to run an experiment end-to-end (config → train → evaluate); (3) how to compare two methods on the same dataset using the evaluation suite.

---

## v2 Requirements

These methods are either more complex, less commonly cited in tutorials, or have significant infrastructure overhead that would distract from the core learning objectives.

- [ ] **V2-01**: **CPC v1** (van den Oord et al., 2018). InfoNCE for sequential/spatial data with GRU autoregressive context model. Requires a fundamentally different architecture (autoregressive, patch-grid, causal masking) that cannot reuse the standard `BaseSSLModule`. High complexity (Tier 3); tutorial value is in understanding InfoNCE origins.

- [ ] **V2-02**: **CPC v2** (Hénaff et al., ICML 2020). Scales CPC to ResNets with layer normalization (not batch norm) inside the patch encoder. Can share CPC v1 infrastructure. Layer norm is required — batch norm leaks spatial position information across patches.

- [ ] **V2-03**: **CMC** (Tian et al., ECCV 2020). Multi-view contrastive with separate encoder + memory bank per view. Memory and code complexity scales linearly with view count. Reuses `MemoryBank` from INFRA-02. **Note:** Uses YCbCr not Lab color space for ~0.5–1% improvement.

- [ ] **V2-04**: **Deep Cluster** (Caron et al., ECCV 2018). Fundamentally different training loop: offline k-means clustering over entire dataset before each epoch, then classify using pseudo-labels. Requires PCA (to 256 dims) + whitening before k-means. Uniform cluster sampling required to prevent large-cluster dominance. Empty cluster handling required. Typical training: 500 epochs vs. 200 for contrastive methods. Not compatible with standard `BaseSSLModule.training_step` design.

- [ ] **V2-05**: **SimCLR v2 semi-supervised** (knowledge distillation stage). Requires a labeled subset and a distillation loop. Extend ERA2-04 (SimCLR v2 pretraining) with the distillation stage.

- [ ] **V2-06**: **InfoMin full view-learning** (semi-supervised view optimization). Requires labeled subset for view quality estimation. Extend ERA2-06 (InfoMin as augmentation policy) with the view-learning framework.

- [ ] **V2-07**: Advanced multi-crop strategies beyond SwAV/DINO basics (e.g., focal crops, adaptive crop sizing).

- [ ] **V2-08**: DINOv2 training from scratch (requires LVD-142M-scale curated dataset, FlashAttention, hundreds of GPU-days). Tutorial coverage for v1 is ERA4-03 (feature extraction / fine-tuning only).

---

## Out of Scope

| Feature | Reason |
|---------|--------|
| Hydra / OmegaConf | Over-engineered for a tutorial; hides config structure behind framework magic; plain Pydantic + YAML is clearer for learners |
| Real-time training dashboards | Lightning's built-in TensorBoard logging is sufficient; adding a live dashboard is operational overhead not relevant to SSL concepts |
| Multi-node distributed training | Tutorial targets single-machine GPU setups; multi-node adds NCCL tuning complexity unrelated to SSL |
| Pre-trained weight downloads / model zoo | Users train from scratch or load their own checkpoints; hosting weights adds maintenance burden |
| Hyperparameter sweep framework (Optuna, Ray Tune) | Out of tutorial scope; users can run multiple config files manually |
| GAN-based SSL methods | Different training paradigm (adversarial); not contrastive |
| Self-supervised NLP methods (BERT, GPT) | Separate domain; this repo is vision-only |
| Video or audio SSL (outside CPC v1 context) | Multi-modal methods are a distinct tutorial topic |
| Automatic mixed precision as a user-configurable option | Lightning handles AMP transparently; exposing it adds config surface area without educational value |
| DINO attention map visualization (beyond CAM) | The self-attention map display from the DINO paper is a nice demo but is specific to ViT + DINO and does not generalize to the evaluation suite |

---

## Traceability

| Requirement ID | Description | Source | Phase |
|----------------|-------------|--------|-------|
| FOUND-01 | BaseSSLModule base class | STACK.md Pattern 1 | Phase 1 |
| FOUND-02 | Pydantic v2 config schema | STACK.md Config System | Phase 1 |
| FOUND-03 | timm backbone factory | STACK.md timm Integration | Phase 1 |
| FOUND-04 | ProjectionHead MLP | METHODS.md Shared Pattern 3 | Phase 1 |
| FOUND-05 | ContrastiveAugmentation pipeline | METHODS.md Shared Pattern 1 | Phase 1 |
| FOUND-06 | SSLDataModule | PROJECT.md, EVALUATION.md | Phase 1 |
| FOUND-07 | Method dispatcher | STACK.md Dispatcher pattern | Phase 1 |
| FOUND-08 | EvalConfig schema | EVALUATION.md Integration | Phase 9 |
| FOUND-09 | TensorBoard logging | STACK.md Pattern 1 | Phase 1 |
| FOUND-10 | EMAUpdater utility | STACK.md Pattern 2; METHODS.md Shared Pattern 5 | Phase 1 |
| ERA1-01 | Instance Discrimination | METHODS.md Era 1 | Phase 2 |
| ERA1-02 | Invariant Spread | METHODS.md Era 1 | Phase 2 |
| ERA2-01 | MoCo v1 | METHODS.md Era 2 | Phase 4 |
| ERA2-02 | MoCo v2 | METHODS.md Era 2 | Phase 4 |
| ERA2-03 | SimCLR v1 | METHODS.md Era 2 | Phase 3 |
| ERA2-04 | SimCLR v2 (pretraining stage) | METHODS.md Era 2 | Phase 3 |
| ERA2-05 | SwAV | METHODS.md Era 2 | Phase 5 |
| ERA2-06 | InfoMin (augmentation demo) | METHODS.md Era 2 | Phase 5 |
| ERA3-01 | BYOL | METHODS.md Era 3 | Phase 6 |
| ERA3-02 | SimSiam | METHODS.md Era 3 | Phase 6 |
| ERA3-03 | Barlow Twins | METHODS.md Era 3 | Phase 6 |
| ERA4-01 | MoCo v3 | METHODS.md Era 4 | Phase 7 |
| ERA4-02 | DINO | METHODS.md Era 4 | Phase 7 |
| ERA4-03 | DINOv2 (fine-tuning only) | METHODS.md Era 4 | Phase 7 |
| SUP-01 | SupCon | METHODS.md Supervised | Phase 8 |
| INFRA-01 | InfoNCELoss | METHODS.md Shared Pattern 2 | Phase 1 |
| INFRA-02 | MemoryBank | METHODS.md Shared Pattern 7 | Phase 2 |
| INFRA-03 | MomentumQueue | METHODS.md Shared Pattern 7 | Phase 4 |
| INFRA-04 | MultiCropDataset | METHODS.md Shared Pattern 8 | Phase 5 |
| INFRA-05 | PredictorHead | METHODS.md Shared Pattern 4 | Phase 7 |
| INFRA-06 | LARS optimizer | METHODS.md (SimCLR, BYOL, SwAV) | Phase 1 |
| EVAL-01 | k-NN callback | EVALUATION.md k-NN section | Phase 9 |
| EVAL-02 | Linear probe script | EVALUATION.md Linear Probing | Phase 9 |
| EVAL-03 | t-SNE visualization | EVALUATION.md t-SNE section | Phase 9 |
| EVAL-04 | UMAP visualization | EVALUATION.md UMAP section | Phase 9 |
| EVAL-05 | Fine-tuning script | EVALUATION.md Fine-tuning | Phase 9 |
| EVAL-06 | CAM visualization | EVALUATION.md CAM section | Phase 9 |
| DOC-01 | README | PROJECT.md | Phase 10 |
| DOC-02 | Per-method docstrings | METHODS.md sources | Phase 10 |
| DOC-03 | Tutorial notebook/guide | PROJECT.md | Phase 10 |

---

## Key Implementation Constraints (Cross-Cutting)

The following constraints apply to all method implementations and must be verified during review:

1. **EMA update location:** Always `on_train_batch_end`, never `training_step` or `on_before_optimizer_step`. Violation silently degrades MoCo, BYOL, and DINO.

2. **Momentum encoder exclusion from optimizer:** `backbone_ema` and `projector_ema` must never appear in `learnable_params`. Use `deactivate_requires_grad(...)` immediately after `deepcopy`. Violation corrupts EMA and causes training instability.

3. **Stop-gradient placement:** In SimSiam, `.detach()` must be applied to z in the loss, not to p. In BYOL, stop-gradient is applied to the target branch output. Missing either causes collapse.

4. **Sinkhorn-Knopp for SwAV:** The assignment matrix must be doubly stochastic. Prototype vectors must be L2-normalized after every update. Prototypes must be frozen for `freeze_prototypes_epochs` (default: 1 epoch).

5. **Collapse monitoring for no-negative methods:** BYOL, SimSiam, and Barlow Twins must log `z.std(dim=0).mean()` during training. Value approaching 0 indicates collapse.

6. **LARS optimizer dependency:** Must be pinned in `requirements.txt`. Required for SimCLR v1/v2, SwAV, and BYOL at large batch sizes.

7. **timm feature dim:** Always use `backbone.num_features`, never `backbone.inplanes` or hard-coded integers. Applies to all uses of `build_backbone()`.

8. **SupCon batch sampler:** Must guarantee at least 2 instances per class per batch. A standard random sampler causes SupCon to degenerate to SimCLR.

9. **Linear probe weight decay:** Must be 0.0 when backbone is frozen. Any weight decay on the linear head suppresses accuracy.

10. **DINOv2 scope:** Implement as feature extraction / fine-tuning only. Do not attempt from-scratch training.
