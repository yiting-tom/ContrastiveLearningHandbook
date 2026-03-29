# Method Research
# Contrastive and Self-Supervised Learning — Implementation Reference

**Researched:** 2026-03-29
**Scope:** Algorithm details, hyperparameters, code-sharing opportunities, gotchas, and reference implementations for 18 methods spanning 2018–2023.

---

## Era 1: 2018–2019 Proxy Tasks

### Instance Discrimination (Wu et al., CVPR 2018)

**Core Algorithm:**
Each image instance is treated as its own class. The encoder learns to discriminate every image from all others via a non-parametric softmax over the entire dataset. Because a full softmax is infeasible, Noise Contrastive Estimation (NCE) is used: each update samples `m` random negatives and performs an (m+1)-way classification. Negative keys are stored in a memory bank (one vector per training image), updated with the current encoder.

**Unique contribution:** First end-to-end formulation of instance-level discrimination as a proxy task; coined the memory bank pattern that MoCo later replaced with a queue.

**Key Hyperparameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Temperature τ | 0.07 | Low temperature sharpens the distribution |
| NCE sample size (negatives) | 4096 | Memory bank size = full dataset |
| Optimizer | SGD, momentum=0.9, wd=1e-4 | |
| LR | 0.03, decay ×0.1 at epochs 120, 160 | |
| Epochs | 200 | |
| Batch size | 256 | |
| Normalization const Z | Estimated on first mini-batch, fixed | |

**Gotchas:**
- The normalization constant Z must be estimated and fixed early; recomputing it each step destabilizes training.
- The memory bank becomes stale: keys in the bank were computed by earlier encoder snapshots, so consistency degrades over long training. This is the fundamental problem MoCo solves.
- ε = 1e-7 must be added for numerical stability in the NCE denominator.

**Reference:** https://github.com/zhirongw/lemniscate.pytorch

---

### Invariant Spread (Ye et al., CVPR 2019)

**Core Algorithm:**
Also called "Unsupervised Embedding Learning via Invariant and Spreading Instance Feature." Directly optimizes a softmax over in-batch instances without a memory bank. For each instance, one augmented copy `T(x_i)` is generated. The embedding must be **invariant** to augmentation and **spread** away from all other instances. Loss is a standard cross-entropy over an in-batch softmax (no queue, no bank).

**Unique contribution:** Showed in-batch softmax can work without a memory bank, trading off the number of negatives for a cleaner, simpler loss. Closer ancestor of SimCLR than of MoCo.

**Key Hyperparameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Temperature | 0.07 | Same as Instance Discrimination |
| Batch size | 128–256 | Limits negatives to batch |
| Optimizer | SGD | |

**Gotchas:**
- Very small batch means very few negatives — performance sensitive to batch size in a way SimCLR later quantified.
- In-batch negatives create implicit correlation between samples in the batch (same-class images can appear as false negatives).

**Reference:** arXiv:1904.03436. No single canonical repo; CMC repo (HobbitLong/CMC) contains an InstDis reimplementation.

---

### CPC (Contrastive Predictive Coding, van den Oord et al., 2018/2019)

**Core Algorithm:**
Learn representations by predicting future latent codes. An encoder maps each input segment to a latent vector `z_t`. An autoregressive model (GRU) summarizes the past into a context vector `c_t`. A learned score function `f_k(z_{t+k}, c_t) = exp(z_{t+k}^T W_k c_t)` scores whether a future code is the "real" one vs. random negatives. The InfoNCE loss maximizes the probability of identifying the true future code among N candidates. This is the origin of the InfoNCE loss.

**Unique contribution:** Introduced InfoNCE as a lower bound on mutual information; demonstrated the same framework on audio, images (patch grid), text, and RL — the only "universal" SSL method of this era.

**Key Hyperparameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Prediction steps k | 12 (audio) / varies | More steps = better features in ablation |
| Context model | GRU, 256-dim hidden | |
| Optimizer | Adam, lr=2e-4 | |
| Batch size | 8 per GPU × 8 GPUs = 64 | |
| Training steps | ~300,000 | |
| Negatives | Drawn from same sequence or same batch | |

**Gotchas:**
- Sequence structure is essential: the GRU context must see a long enough prefix. For images, CPC uses a grid of patches and a row-wise autoregressive model — getting the spatial ordering right is non-trivial.
- CPC v1 (images) uses a pixel-based encoder over 64×64 patches; CPC v2 scales to ResNets with larger crops.
- The InfoNCE loss requires careful normalization: cosine or L2 normalization of `z` before scoring matters.
- Different modalities need different autoregressors; the GRU is audio-specific. For images, PixelCNN-style or row-wise convolutions are used in v1.

**CPC v2 (Hénaff et al., ICML 2020):** Scales to ResNet-50 on ImageNet, reaches 71.5% linear top-1. Key change: larger crops, color jitter, horizontal flips, layer normalization instead of batch norm inside the encoder.

**Reference:** arXiv:1807.03748. CPC v2: http://proceedings.mlr.press/v119/henaff20a/henaff20a.pdf

---

### CMC (Contrastive Multiview Coding, Tian et al., 2019/ECCV 2020)

**Core Algorithm:**
Extends Instance Discrimination to arbitrary numbers of views (modalities). Each image has multiple views (luminance L, chrominance ab, depth, surface normals, semantics). A separate encoder is trained per view. The contrastive loss maximizes agreement between embeddings of the same scene across all view pairs while using a memory bank (one bank per view) for negatives.

**Unique contribution:** First to show the principle that "more views = better representations" and that any paired modality can serve as a view. Directly motivates the multimodal SSL literature.

**Key Hyperparameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Temperature τ | 0.07 | |
| Negatives | 4096 (default), 65536 (improved) | Per memory bank |
| Views | L + ab (default), L + ab + depth + normals | |
| Optimizer | SGD, momentum=0.9, wd=1e-4 | |
| LR | 0.03, decay ×0.1 every 40 epochs after epoch 120 | |
| Epochs | 200 | |

**Gotchas:**
- Each view needs its own memory bank and its own encoder — memory usage scales linearly with number of views.
- Sampling strategy for negatives across multiple banks must be consistent.
- Using YCbCr color space instead of Lab gives ~0.5–1% improvement — a detail not to miss.
- View encoders can have different architectures, but this complicates code sharing.

**Reference:** https://github.com/HobbitLong/CMC (also contains InstDis and MoCo implementations)

---

### Deep Cluster (Caron et al., ECCV 2018)

**Core Algorithm:**
Alternates between two steps each epoch: (1) cluster all image embeddings via k-means to produce pseudo-labels; (2) train the CNN as a standard classifier using those pseudo-labels. The cluster assignments themselves are the supervision signal — no explicit contrastive loss.

**Unique contribution:** Demonstrated that clustering-then-classifying can learn strong visual features. Ancestor of SwAV, which makes this online.

**Key Hyperparameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Number of clusters k | 10,000 | For ImageNet-scale |
| Frequency of re-clustering | Every epoch | |
| PCA dims before k-means | 256 | Required to avoid high-dim issues |
| Whitening | Yes | After PCA |
| Sobel preprocessing | Optional but beneficial | Reduces color bias in first layer |
| Uniform cluster sampling | Yes | Critical to avoid large-cluster dominance |
| Optimizer | SGD | |
| Epochs | 500 | |

**Gotchas:**
- **Trivial solution / empty clusters:** k-means naturally produces imbalanced clusters; some clusters may collapse to near-empty. Must use uniform sampling over cluster labels to rebalance gradients. This is not optional.
- **Cluster stagnation:** On large datasets, one epoch may not be enough to update cluster centroids — representations change faster than clustering can track.
- **PCA + whitening is mandatory:** Raw CNN features in high dimensions produce poor k-means results. Forgetting this preprocessing is a frequent mistake.
- **Sobel filtering:** The paper shows this prevents color statistics from dominating clustering. Include it in early-era ablation comparisons.
- Deep Cluster is **offline** — full k-means over the entire dataset before each epoch. This requires O(N) memory and time per epoch, prohibitive for very large datasets. SwAV solves this by making the assignment online.
- Requires multiple passes to stabilize; 500 epochs is typical vs. 200–300 for contrastive methods.

**Reference:** https://github.com/facebookresearch/deepcluster

---

## Era 2: 2019–2020 MoCo and SimCLR

### MoCo v1 (He et al., CVPR 2020)

**Core Algorithm:**
Frames contrastive learning as dictionary look-up. A query encoder `f_q` processes the anchor view; a momentum encoder `f_k` (EMA copy of `f_q`) encodes negative keys. Keys are stored in a FIFO queue. InfoNCE loss pushes `q` to be similar to its matching key `k+` and dissimilar to all `K` queue keys.

EMA rule: `θ_k ← m·θ_k + (1-m)·θ_q`

The queue decouples dictionary size from batch size — 65,536 keys are used for negatives, far more than any practical batch.

**Unique contribution:** The queue + momentum encoder combination: large, consistent dictionary without needing huge batches or a full memory bank per sample.

**Key Hyperparameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Queue size K | 65,536 | 16× a 4096 batch |
| Momentum m | 0.999 | Must be high for consistency |
| Temperature τ | 0.07 | |
| Optimizer | SGD, momentum=0.9, wd=1e-4 | |
| LR | 0.03, cosine schedule | |
| Epochs | 200 | |
| Projection head | FC only (v1) | |

**Gotchas:**
- m=0.9 (too low) produces much worse results than m=0.999. A common mistake when adapting BYOL's momentum schedule to MoCo.
- The queue must be initialized randomly (not zeros) for the first few batches, but this initialization rarely matters much in practice.
- Shuffling BN: multi-GPU training with BatchNorm leaks information across positive pairs (encoder sees the same BN statistics). MoCo v1 uses **shuffling BN** — shuffle keys across GPUs before encoding. This is easy to forget in custom implementations.

**Reference:** https://github.com/facebookresearch/moco

---

### SimCLR v1 (Chen et al., ICML 2020)

**Core Algorithm:**
Two augmented views of the same image pass through a shared ResNet encoder + 2-layer MLP projection head. NT-Xent loss treats all 2(N-1) other embeddings in the batch as negatives. No queue, no momentum encoder.

**Unique contribution:** The projection head was a key finding — discarding the head and using backbone features for downstream tasks improves performance significantly. Also the first thorough ablation of augmentation type and composition.

**Key Hyperparameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 4096 | Critical — performance degrades sharply below 256 |
| Optimizer | LARS | Required for large batches |
| LR | 4.8 (= 0.3 × batch/256) | Linear scaling rule |
| Weight decay | 1e-6 | |
| Temperature τ | 0.5 (default), 0.1–1.0 range | |
| Epochs | 100–1000 | |
| LR schedule | Linear warmup (10 epochs) + cosine decay | |
| Projection head | 2-layer MLP, 128-dim output | |
| Augmentations | Random crop + resize, color jitter (s=1.0), grayscale, Gaussian blur | |

**Gotchas:**
- LARS is required for batch sizes above ~1024; Adam diverges.
- **Color jitter strength s=1.0 is stronger than default torchvision** — must set explicitly.
- Gaussian blur is often forgotten; it was shown to be important.
- Performance collapses at batch size < 256 due to too few in-batch negatives.
- Projection head output (128-dim) should be used for the loss, not the backbone features.

**Reference:** https://github.com/google-research/simclr (TF), https://github.com/HobbitLong/SupContrast (PyTorch, includes SimCLR)

---

### MoCo v2 (Chen et al., 2020 tech report)

**Core Algorithm:**
Identical to MoCo v1 with three changes borrowed from SimCLR: (1) replace FC projection with 2-layer MLP head, (2) add Gaussian blur augmentation, (3) switch from step LR to cosine LR schedule. Queue and momentum encoder unchanged.

**Unique contribution:** Demonstrated that SimCLR's architecture improvements (MLP head, blur) transfer directly to MoCo, giving +7% ImageNet accuracy over MoCo v1 with no extra compute cost.

**Key Hyperparameters:** Same as MoCo v1, plus:
| Parameter | Value | Notes |
|-----------|-------|-------|
| Projection head | 2-layer MLP, 128-dim | vs. FC in v1 |
| Blur augmentation | σ ∈ [0.1, 2.0] | Gaussian blur, p=0.5 |
| LR schedule | Cosine (no restarts) | vs. step in v1 |

**Gotchas:** Same as MoCo v1. The improvement over v1 is almost purely from the MLP head — a good reminder that architecture details matter as much as algorithm.

**Reference:** arXiv:2003.04297; same repo as MoCo v1: https://github.com/facebookresearch/moco

---

### SimCLR v2 (Chen et al., NeurIPS 2020)

**Core Algorithm:**
Extends SimCLR v1 for semi-supervised learning. Key changes: (1) deeper projection head (3-layer MLP instead of 2), (2) fine-tune using the first MLP layer (not discarding the whole head), (3) knowledge distillation from the large pretrained model to a smaller student. Architecture otherwise identical to v1.

**Unique contribution:** Showed that the projection head is not fully useless — its first layer retains useful representations, especially under low label budget.

**Key Hyperparameters:** Same as SimCLR v1, plus:
| Parameter | Value | Notes |
|-----------|-------|-------|
| Projection head depth | 3 layers | vs. 2 in v1 |
| Fine-tune layer | First layer of MLP head | Not the backbone output |
| Weight decay | Different from v1 | Must re-tune; pretrained weights have different norm scale |

**Gotchas:**
- Pretrained SimCLR v1 and v2 models have **very different weight norm scales** due to different weight decay settings. Using v1 fine-tuning hyperparameters on v2 (or vice versa) gives poor results.
- The distillation stage requires a labeled subset; the semi-supervised setup is more complex than pure self-supervised.

**Reference:** https://github.com/google-research/simclr

---

### SwAV (Caron et al., NeurIPS 2020)

**Core Algorithm:**
Combines clustering with contrastive learning. Prototype vectors `C` (learnable, K=3000 for ImageNet) live alongside the encoder. For each image, two augmented views are encoded to get `z1`, `z2`. "Codes" `q1`, `q2` are computed by assigning features to prototypes using the Sinkhorn-Knopp optimal transport algorithm. The loss swaps assignments: predict `q2` from `z1` and `q1` from `z2`. No pairwise comparison of feature vectors needed.

**Unique contribution:** Online clustering that avoids explicit negative pair computation. Multi-crop strategy: 2 large crops + 6 small crops → dramatically more views per image without proportional memory cost.

**Key Hyperparameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Number of prototypes K | 3000 (ImageNet) | Insensitive — 1000–10000 works similarly |
| Sinkhorn iterations | 3 | More than 3 rarely helps |
| Multi-crop: large crops | 2 × 224×224 | >50% of image |
| Multi-crop: small crops | 6 × 96×96 | <50% of image |
| Temperature τ | 0.1 | |
| Freeze prototypes | First epoch | Prevents degenerate assignments early |
| Optimizer | LARS or SGD | |
| Epochs | 800 (full), 200 (short) | |

**Gotchas:**
- **Prototype collapse:** If prototypes are not frozen for the first epoch, early assignments are meaningless and the loss diverges.
- Sinkhorn-Knopp requires the code matrix to be doubly stochastic (equal average usage of each prototype). If the implementation omits the constraint, some prototypes will dominate.
- Multi-crop with 8 views total makes per-GPU memory usage 4× higher than SimCLR — must reduce batch size accordingly.
- The prototype vectors must be **L2-normalized** at every update step; forgetting this breaks the dot-product score interpretation.
- SwAV is not compatible with standard in-batch negative implementations; the assignment-swapping objective is its own loss function.

**Reference:** https://github.com/facebookresearch/swav

---

### CPC v2 (Hénaff et al., ICML 2020)

**Core Algorithm:**
Scales CPC to ResNet on ImageNet. Key changes from v1: larger patch crops, color jitter + horizontal flips, layer normalization (not batch norm) inside the representation encoder, linear layer for patch prediction rather than bilinear.

**Unique contribution:** Demonstrated CPC is competitive with contrastive methods on ImageNet with the right augmentations and normalization.

**Key Hyperparameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Patch grid | 7×7 (ResNet feature map) | |
| Prediction steps | 5 | Predict 5 rows ahead |
| Normalization | Layer norm (not batch norm) | Critical for patch-level representations |
| Augmentations | Color jitter, horizontal flip, larger crops | New vs. v1 |

**Gotchas:**
- Batch norm inside the patch encoder leaks spatial position information across patches — layer norm is required.
- The autoregressive model must only see past patches (causal), not future ones; a masking mistake turns this into a trivial prediction task.

**Reference:** http://proceedings.mlr.press/v119/henaff20a/henaff20a.pdf

---

### InfoMin (Tian et al., NeurIPS 2020)

**Core Algorithm:**
A theoretical + empirical study of what makes good views for contrastive learning. Proposes the **InfoMin principle**: optimal views minimize I(v1; v2) subject to I(v1; y) = I(v2; y) = I(x; y). In practice, this means using data augmentations that remove as much mutual information as possible while preserving task-relevant information. The method proposes learning data-adaptive views via a semi-supervised view-learning framework.

**Unique contribution:** Theoretical unification of all prior augmentation strategies (SimCLR, MoCo, CMC, PIRL) under a single information-theoretic principle. Achieves 73% top-1 on ImageNet with ResNet-50.

**Key Hyperparameters:** Same as SimCLR/MoCo backbone; the key contribution is augmentation policy.

**Gotchas:**
- InfoMin's semi-supervised view learning requires a small labeled set. Pure unsupervised version is just very strong augmentation.
- As a tutorial method, InfoMin is best presented as "the theory behind why strong augmentation works" rather than as a standalone implementation.

**Reference:** https://github.com/HobbitLong/PyContrast

---

## Era 3: 2020 No-Negative Methods

### BYOL (Bootstrap Your Own Latent, Grill et al., NeurIPS 2020)

**Core Algorithm:**
Two networks: **online** (encoder + projector + predictor) and **target** (encoder + projector, no predictor). Online is trained to predict the target's projection of the same image under a different augmentation. Target parameters are never backpropagated through; they are updated only via EMA of the online parameters.

Loss: MSE between L2-normalized predictor output and L2-normalized target projection.

**Unique contribution:** First method to show that eliminating negative pairs entirely does not cause collapse, provided you use a predictor + EMA target.

**Key Hyperparameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| EMA momentum | Cosine schedule 0.996 → 1.0 | NOT a fixed value |
| Projector MLP | 2 layers, 4096 hidden, 256 output | With batch norm and ReLU |
| Predictor MLP | Same architecture as projector | Also with batch norm |
| Optimizer | LARS | |
| LR | 0.2 × batch/256 | |
| Batch size | 4096 (full), 256 (workable) | Less sensitive than SimCLR |
| Epochs | 1000 | |

**Gotchas:**
- **Momentum schedule matters:** Using a fixed m=0.99 instead of the cosine schedule 0.996→1.0 significantly degrades performance.
- **Predictor is critical:** Removing the predictor from the online network causes immediate collapse. This must be in the implementation.
- **Batch norm in predictor/projector is load-bearing:** BYOL with group norm or layer norm instead of batch norm collapses in some configurations (later work clarified that BN provides implicit negative signal).
- **Target update timing:** The EMA update should occur after the gradient step on the online network, not before. Getting the order wrong introduces a one-step lag that accumulates.
- The MSE loss operates on L2-normalized vectors, so it equals `2 - 2·cos_sim`. Using raw MSE without normalization produces different gradients.

**Reference:** arXiv:2006.07733. PyTorch: https://docs.lightly.ai/self-supervised-learning/examples/byol.html

---

### SimSiam (Chen & He, CVPR 2021)

**Core Algorithm:**
Simplest no-negative method. Two augmented views pass through a **shared encoder** + projector MLP. One view also passes through a predictor MLP. Loss is the negative cosine similarity between the predictor output of view 1 and the **stop-gradient** of the projector output of view 2, computed symmetrically.

Key code pattern:
```python
z1, z2 = projector(encoder(x1)), projector(encoder(x2))
p1, p2 = predictor(z1), predictor(z2)
loss = -(cos_sim(p1, z2.detach()) + cos_sim(p2, z1.detach())) / 2
```

**Unique contribution:** Showed that neither momentum encoder nor large batches nor negative pairs are required. Collapse is prevented by stop-gradient + predictor alone.

**Key Hyperparameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Stop gradient | On z1, z2 | MUST be applied correctly |
| Predictor MLP | 2 layers, 2048 → 512 → 2048 | Bottleneck architecture |
| Projector MLP | 3 layers, 2048 output | With batch norm (no ReLU on output) |
| Optimizer | SGD, momentum=0.9, wd=1e-4 | NOT LARS |
| LR | 0.05 × batch/256 | |
| Batch size | 512 | Works well; not sensitive |
| Epochs | 100–800 | |

**Gotchas:**
- **Stop-gradient application is the single most common bug.** Forgetting `.detach()` on z causes immediate collapse to constant output (loss = -1.0 plateau).
- **Monitoring std of z:** The standard deviation of the L2-normalized z over a batch should be ~0.707 for a uniform sphere. If std → 0, the model has collapsed. Add this as a training diagnostic.
- **Batch norm on projector output (no ReLU):** The final layer of the projector has BN but no ReLU. Adding ReLU to the last layer is a common mistake that hurts performance.
- **Learning rate sensitivity:** Unlike BYOL, SimSiam uses SGD not LARS, and LR 0.05 × batch/256. Switching to Adam without adjusting LR causes divergence.
- **Predictor collapse:** If the predictor is too powerful (too many layers, too high LR), it can learn the identity mapping and the training signal vanishes.

**Reference:** arXiv:2011.10566. https://docs.lightly.ai/self-supervised-learning/examples/simsiam.html

---

### Barlow Twins (Zbontar et al., ICML 2021)

**Core Algorithm:**
Two augmented views pass through a **shared encoder** + high-dimensional projector (8192-dim output). The cross-correlation matrix C between the batch of projections from view 1 and view 2 is computed and normalized by batch size. The loss drives C toward the identity matrix: diagonal entries → 1 (invariance), off-diagonal entries → 0 (redundancy reduction). No stop-gradient, no momentum, no predictor.

Loss:
```
L = Σ_i (1 - C_ii)^2  +  λ · Σ_i Σ_{j≠i} C_ij^2
```

**Unique contribution:** Reframes SSL as decorrelation, not contrastive comparison. Does not require negative samples, large batches, stop-gradient, or EMA. The loss is naturally collapse-resistant because C = I requires each dimension to be uncorrelated.

**Key Hyperparameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Projector output dim | 8192 | Unusually high; method benefits from high-dim |
| Projector MLP | 3 layers: 2048→8192→8192→8192 | BN + ReLU on first 2 layers |
| λ (off-diagonal weight) | 5e-3 | Relative weight of redundancy term |
| Optimizer | LARS | |
| LR | 0.2 × batch/256 | |
| Batch size | 2048 (full), 256 (nearly same performance) | |
| Epochs | 1000 | |

**Gotchas:**
- **Cross-correlation matrix must be normalized by batch size:** Computing C without dividing by N gives a different scale and the loss hyperparameters become meaningless.
- **Projector output must be standardized (zero mean, unit variance) per dimension across the batch before computing C.** Omitting this standardization is a subtle bug; batch norm at the projector output typically handles it, but the standardization must happen at the loss computation, not just during the forward pass.
- **λ is sensitive:** λ too high → the redundancy term dominates and the encoder loses invariance. λ too low → collapse along correlated dimensions.
- Unlike BYOL/SimSiam, gradients flow through both branches symmetrically — no asymmetry tricks needed. This makes Barlow Twins the simplest no-negative method to implement correctly.
- The high-dimensional projector (8192) is unusual and can be a memory bottleneck. Do not reduce below 2048 without performance validation.

**Reference:** https://github.com/facebookresearch/barlowtwins

---

## Era 4: 2021 Transformer Era

### MoCo v3 (Chen et al., ICCV 2021)

**Core Algorithm:**
Extends MoCo to ViT backbones. Abandons the queue entirely (keys co-exist in the same large batch). Uses a symmetric contrastive loss: two query-key pairs from two augmented views. Same InfoNCE/NT-Xent loss as SimCLR but with a momentum key encoder (same as MoCo v1/v2). Architecture: backbone (ViT or ResNet) + projection MLP + prediction MLP.

**Unique contribution:** Identified training instability as the primary challenge for self-supervised ViT. Key stability fix: **freeze the patch projection layer** (first layer of ViT) — random patch projection instead of learned.

**Key Hyperparameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW (ViT) / LARS (ResNet) | ViT requires AdamW |
| LR (ViT-B) | 1.5e-4 for batch 4096 | |
| Weight decay | 0.1 (ViT) | |
| Momentum m | 0.99 (optimal) | Lower = SimCLR-like, 0.999 = worse |
| Temperature τ | 0.2 | |
| Warmup | 40 epochs | |
| Total epochs | 300 (ViT-S), 300 (ViT-B) | |
| Batch size | 4096 | |
| Position embedding | sin-cos fixed | |
| Patch projection | Frozen (random) | Key stability trick |

**Gotchas:**
- **ViT training instability:** Loss spikes mid-training are a known failure mode. Without freezing patch projection, training can silently collapse to partial solutions with deceptively good-looking loss curves but poor linear probe accuracy.
- **Momentum 0.99 not 0.999:** The optimal MoCo v3 momentum (0.99) is different from MoCo v1/v2 (0.999). Using 0.999 gives 1–2% worse accuracy.
- **AdamW for ViT, not SGD:** SGD + LARS gives inferior results for ViT backbones. This is a departure from all ConvNet-era methods.
- The symmetric loss in v3 (two pairs vs. one asymmetric pair in v1/v2) changes the effective learning rate — scale accordingly.

**Reference:** https://github.com/facebookresearch/moco-v3

---

### DINO (Caron et al., ICCV 2021)

**Core Algorithm:**
Knowledge distillation framework without labels. Student and teacher share the same ViT architecture but have different parameters. Teacher is an EMA of the student. Both networks receive augmented views but the teacher only receives **global crops** (>50% of image); the student receives all crops (global + local). Loss is cross-entropy between softmax outputs of student and teacher.

Collapse prevention: **centering** (subtract running mean from teacher output) + **sharpening** (low temperature on teacher softmax, higher on student). These two mechanisms balance each other — centering prevents single-mode collapse, sharpening prevents uniform collapse.

**Unique contribution:** Discovered that self-supervised ViT features contain explicit semantic segmentation in their attention maps ("free" segmentation without any labels). K-NN classifier with frozen features achieves 78.3% on ImageNet — no fine-tuning needed.

**Key Hyperparameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Teacher temperature τ_t | Warmup from 0.04 → 0.07 | Low = sharpening |
| Student temperature τ_s | 0.1 | Higher = softer student distribution |
| EMA momentum | 0.9 → 0.999 (cosine schedule) | |
| Centering | Running mean (EMA) of teacher outputs | Updated each batch |
| Projection head | 3-layer MLP + L2 norm → weight-normalized FC | Bottleneck design |
| Output dim | 65536 | High-dimensional softmax |
| Multi-crop | 2 global (224) + 6–10 local (96) | Teacher: global only |
| Optimizer | AdamW (ViT) | |
| Epochs | 100 (quick) / 800 (full) | |
| Gradient clip | 3.0 | Important for stability |

**Gotchas:**
- **Centering update timing:** Center vector c must be updated with teacher outputs **before** computing the loss. Using old center values makes the loss temporarily inconsistent.
- **Teacher temperature warmup is critical:** Starting with τ_t = 0.07 immediately (without warmup from 0.04) causes early instability.
- **Output dimension 65536 is surprisingly important:** Lower dims (e.g., 2048, 4096) give noticeably worse downstream features. The large softmax is integral to the method.
- **Multi-crop with teacher-global-only:** If local crops are passed to the teacher, the method is essentially CPC-style (predicting global from local) which changes the objective semantics.
- **Gradient clipping:** Without `max_norm=3.0` gradient clipping, ViT training can diverge due to the large projection head.
- DINO is harder to implement correctly than SimCLR or BYOL due to the centering + sharpening interaction.

**Reference:** https://github.com/facebookresearch/dino

---

### DINOv2 (Oquab et al., 2023)

**Core Algorithm:**
Combines multiple SSL objectives on a large curated dataset (LVD-142M). Key components: DINO loss (as above, on CLS token), **iBOT loss** (masked image modeling, on patch tokens), **SwAV-style loss** (prototype assignment), and a regularization term. The joint objective over CLS + patch tokens makes DINOv2 features useful for both image-level and pixel-level tasks.

**Unique contribution:** First self-supervised foundation model for computer vision — features generalize to dense prediction, depth estimation, semantic segmentation without fine-tuning. Added **register tokens** (Oct 2023) to eliminate attention artifacts.

**Key Hyperparameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Model sizes | ViT-S/14, ViT-B/14, ViT-L/14, ViT-g/14 | All patch size 14 |
| Activations | SwiGLU (trained from scratch) | vs. GELU in DINOv1 |
| Normalization | LayerScale | For large-model stability |
| Dataset | LVD-142M (curated) | Not ImageNet-21k |
| Training | FlashAttention, sequence packing | Engineering for scale |
| Register tokens | 4 extra learnable tokens | Eliminates attention artifacts |

**Gotchas:**
- DINOv2 is not practical to train from scratch in a tutorial setting; it requires hundreds of GPU-days. Tutorial use case is **feature extraction / fine-tuning** of pretrained checkpoints.
- The multi-objective loss (DINO + iBOT + SwAV) requires implementing three different loss functions simultaneously; error in any one affects the others.
- Register tokens (added post-publication) change the model API; code written for the original DINOv2 may not handle register tokens correctly.
- Note: There is no official "DINOv3." The progression is DINO (2021) → DINOv2 (2023) → DINOv2 + Registers (Oct 2023).

**Reference:** https://github.com/facebookresearch/dinov2

---

## Supervised: SupCon

### SupCon (Supervised Contrastive Learning, Khosla et al., NeurIPS 2020)

**Core Algorithm:**
Extends SimCLR's self-supervised contrastive loss to the supervised setting. Positives for an anchor are: (a) the other augmented view of the same image, plus (b) all other images in the batch with the **same class label**. Negatives are all images with different labels.

Loss (sum-outside variant):
```
L_sup = Σ_i ( -1/|P(i)| Σ_{p ∈ P(i)} log [ exp(z_i·z_p/τ) / Σ_{k≠i} exp(z_i·z_k/τ) ] )
```
where P(i) is the set of indices of positives for anchor i.

Two-stage training: (1) supervised contrastive pretraining with SupCon loss + projection head; (2) standard cross-entropy fine-tuning with a linear classifier on top of frozen encoder features.

**Unique contribution:** First contrastive loss to consistently outperform cross-entropy on large-scale classification. More robust to label noise and augmentation than cross-entropy. The PyTorch implementation generalizes to SimCLR by passing no labels.

**Key Hyperparameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Temperature τ | 0.07 (default) | Same as SimCLR |
| Projection head | 2-layer MLP, 128-dim output | Features must be L2-normalized |
| Base encoder | ResNet-50 (2048-dim output) | |
| Optimizer (stage 1) | SGD, LARS | |
| Optimizer (stage 2) | SGD | Cross-entropy fine-tune |
| Batch size | 1024 | Need multiple per-class examples in each batch |
| Epochs | 700 (pretraining) + 100 (fine-tuning) | |

**Gotchas:**
- **Features must be L2-normalized before the loss.** The implementation normalizes in the loss function; if you normalize before passing to `SupConLoss`, outputs are double-normalized which is a no-op for unit vectors but confirms the code is doing what you expect.
- **Batch sampling must include multiple instances per class.** If a batch has only one image per class, SupCon degenerates to SimCLR. Use a class-balanced sampler that guarantees at least 2 images per class per batch.
- **Sum-outside vs. sum-inside:** Two loss variants exist (Eq. 2 and Eq. 3 in the paper). The "sum-outside" variant (above) is the one used and recommended. The "sum-inside" variant is mathematically different and performs slightly worse.
- **Two-stage training is important:** Training with SupCon loss end-to-end for classification (adding a classifier during pretraining) does not work as well as the two-stage approach.
- The HobbitLong/SupContrast implementation handles both SupCon and SimCLR modes via the `labels` argument — if labels is None, it behaves as SimCLR. This is a clean design pattern worth replicating.

**Reference:** https://github.com/HobbitLong/SupContrast

---

## Shared Patterns (Code Reuse Opportunities)

### 1. Augmentation Pipeline — Universal Shared Module
All methods from SimCLR onward use a nearly identical augmentation pipeline:
- Random crop + resize (the most important single augmentation)
- Color jitter (strength s=1.0 for strong augmentation)
- Random grayscale (p=0.2)
- Gaussian blur (σ ∈ [0.1, 2.0], p=0.5)
- Random horizontal flip

Implement one `ContrastiveAugmentation(size, strong=True)` class with a configurable `n_views` parameter. Era 1 methods (Instance Discrimination, CMC) use weaker augmentation; era 2+ methods use the strong SimCLR set.

**Shared by:** Instance Discrimination, Invariant Spread, SimCLR v1/v2, MoCo v1/v2/v3, BYOL, SimSiam, Barlow Twins, DINO, SupCon.

---

### 2. InfoNCE / NT-Xent Loss — Shared by All Contrastive Methods
InfoNCE and NT-Xent are the same formula with minor naming differences. A single implementation handles:
- SimCLR (in-batch symmetric)
- MoCo v1/v2 (asymmetric, queue as negatives)
- MoCo v3 (symmetric, in-batch)
- SupCon (multi-positive extension)
- Instance Discrimination (NCE approximation)
- CPC (sequence version)
- CMC (multi-view version)
- InfoMin

One `InfoNCELoss(temperature, reduction='mean')` module covers ~60% of all methods. The SupCon variant needs an additional `labels` parameter.

---

### 3. Projection Head — Shared Architecture
A 2-layer or 3-layer MLP with BN + ReLU (no ReLU on output, optional BN on output) is used by nearly all methods. Parameters differ only in hidden dimension, output dimension, and depth:

| Method | Hidden dim | Output dim | Depth |
|--------|-----------|------------|-------|
| SimCLR v1 | 2048 | 128 | 2 |
| SimCLR v2 | 2048 | 128 | 3 |
| MoCo v2/v3 | 2048 | 128 | 2 |
| BYOL | 4096 | 256 | 2 |
| SimSiam | 2048 | 2048 | 3 |
| Barlow Twins | 8192 | 8192 | 3 |
| DINO | 2048 | 65536 | 3 (+L2+FC) |
| SupCon | 2048 | 128 | 2 |

Implement `ProjectionHead(input_dim, hidden_dim, output_dim, num_layers, use_bn=True)`.

---

### 4. Predictor Head — Shared by BYOL, SimSiam, MoCo v3, DINO
All no-negative methods and DINO use a predictor MLP on top of the projection head. Same architecture as projection head but smaller (bottleneck in SimSiam, same-size in BYOL). Reuse `ProjectionHead` or a separate `PredictorHead`.

---

### 5. EMA / Momentum Encoder Update — Shared by MoCo v1/v2/v3, BYOL, DINO
The same EMA update logic applies to all momentum encoder methods:
```python
def update_ema(online_params, target_params, momentum):
    for online_p, target_p in zip(online_params, target_params):
        target_p.data = momentum * target_p.data + (1 - momentum) * online_p.data
```
MoCo v1/v2 use fixed m=0.999; BYOL and DINO use a cosine schedule from ~0.996 to 1.0.

Implement `EMAUpdater(base_momentum, end_momentum, total_steps)` with a `step()` method.

---

### 6. ResNet Backbone — Universal Encoder
All CNN-era methods use ResNet-50 as the default encoder. A single `get_encoder(arch, pretrained=False)` factory function suffices. For transformer methods (MoCo v3, DINO, DINOv2), add ViT-S/B variants.

---

### 7. Memory Bank / Queue — Shared by Instance Discrimination, CMC, MoCo v1/v2
The memory bank (per-sample key storage) and the FIFO queue (mini-batch rolling window) are structurally similar:
- Memory bank: `torch.nn.Embedding(n_samples, dim)` updated with indices
- Queue: `torch.zeros(K, dim)` with a pointer, FIFO dequeue/enqueue

Implement separate `MemoryBank` and `MomentumQueue` classes; they share a common interface (`get_negatives()`, `update()`).

---

### 8. Multi-Crop Loader — SwAV, DINO, DINOv2
Both SwAV and DINO use multi-resolution crops. A `MultiCropDataset` wrapper that generates `n_large_crops` + `n_small_crops` can be shared.

---

## Implementation Difficulty Rankings

### Tier 1: Straightforward (1–2 days)
Methods with clean, well-understood loss functions and no tricky stability mechanisms.

1. **SupCon** — NT-Xent + labels. Clean reference implementation exists. The main challenge is understanding the multi-positive loss derivation, not the code.
2. **SimCLR v1** — In-batch NT-Xent. No memory management. Best starting point for the tutorial.
3. **Invariant Spread** — Simplest in-batch softmax; precursor to SimCLR.
4. **Barlow Twins** — Cross-correlation matrix loss is more complex than NT-Xent but has no asymmetry or EMA. Fully symmetric, no stop-gradient. The matrix operation is the hardest part.
5. **MoCo v2** — MoCo v1 + MLP head + blur. If MoCo v1 works, v2 is a 5-line change.

---

### Tier 2: Moderate (2–4 days)
Methods with queue management, EMA, or multiple components.

6. **MoCo v1** — Queue management and shuffling BN are the main complications.
7. **SimSiam** — Conceptually simple (stop-gradient), but collapse monitoring and correct `.detach()` placement make it error-prone. High debug risk.
8. **SimCLR v2** — Mostly SimCLR v1 + deeper head. Extra complexity from semi-supervised distillation if included.
9. **Instance Discrimination** — Memory bank + NCE normalization constant estimation.
10. **BYOL** — EMA update + predictor + momentum schedule interaction. Multiple things that must be correct simultaneously.
11. **InfoMin** — If treated as strong augmentation policy (easy), or full semi-supervised view learning (hard).

---

### Tier 3: Hard (4–7 days)
Methods with complex objectives, multiple interacting components, or instability risks.

12. **SwAV** — Sinkhorn-Knopp assignments, prototype normalization, multi-crop, prototype freezing. Each piece has a failure mode.
13. **DeepCluster** — Offline k-means loop, PCA + whitening preprocessing, uniform sampling. The training loop is fundamentally different from all other methods (epoch-level clustering, not batch-level loss).
14. **CMC** — Multiple encoders + multiple memory banks. Code complexity scales with number of views.
15. **DINO** — Centering + sharpening interaction, warmup schedules, high-dim projection head, teacher-global-only crops. Most failure modes are silent (bad representations, not NaN).
16. **CPC** — Autoregressive context model + spatial/temporal ordering logic. Very different architecture from all other methods; the patch-grid setup for images requires careful indexing.
17. **MoCo v3** — ViT instability + frozen patch projection + AdamW tuning. Instability is the main challenge; failures look like partial successes.
18. **DINOv2** — Multi-objective (DINO + iBOT + SwAV) loss, massive scale requirements, curated dataset. Tutorial utility is best as feature extraction, not training from scratch.

---

### Summary: Recommended Tutorial Implementation Order
For a tutorial, implement in this order to progressively build complexity:
1. SimCLR v1 (baseline NT-Xent, in-batch)
2. MoCo v1/v2 (queue + momentum encoder)
3. SupCon (supervised extension, multi-positive)
4. BYOL (no-negative, EMA)
5. SimSiam (no-negative, stop-gradient)
6. Barlow Twins (redundancy reduction)
7. Instance Discrimination (memory bank, historical baseline)
8. SwAV (clustering + multi-crop)
9. DINO (distillation, ViT)
10. MoCo v3 (ViT contrastive)
11. Deep Cluster (offline clustering baseline)
12. CMC (multiview extension)
13. CPC / CPC v2 (sequential/autoregressive)
14. InfoMin (theory + strong augmentation)
15. SimCLR v2 (semi-supervised extension)
16. DINOv2 (scale, multi-objective, fine-tuning)

---

## Sources

- Wu et al. (2018): https://github.com/zhirongw/lemniscate.pytorch
- Tian et al. CMC: https://github.com/HobbitLong/CMC
- Caron et al. DeepCluster: https://github.com/facebookresearch/deepcluster
- van den Oord et al. CPC: https://arxiv.org/abs/1807.03748
- He et al. MoCo: https://github.com/facebookresearch/moco
- Chen et al. SimCLR: https://github.com/google-research/simclr
- Caron et al. SwAV: https://github.com/facebookresearch/swav
- Hénaff et al. CPC v2: http://proceedings.mlr.press/v119/henaff20a/henaff20a.pdf
- Tian et al. InfoMin: https://github.com/HobbitLong/PyContrast
- Grill et al. BYOL: https://arxiv.org/pdf/2006.07733
- Chen & He SimSiam: https://arxiv.org/pdf/2011.10566
- Zbontar et al. Barlow Twins: https://github.com/facebookresearch/barlowtwins
- Khosla et al. SupCon: https://github.com/HobbitLong/SupContrast
- Chen et al. MoCo v3: https://github.com/facebookresearch/moco-v3
- Caron et al. DINO: https://github.com/facebookresearch/dino
- Oquab et al. DINOv2: https://github.com/facebookresearch/dinov2
- Lil'Log Contrastive Survey: https://lilianweng.github.io/posts/2021-05-31-contrastive/
- UvA SimCLR Tutorial: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
- LightlySSL Docs: https://docs.lightly.ai/self-supervised-learning/
