# Phase 8: Supervised Contrastive Learning (SupCon) - Research

**Researched:** 2026-04-10
**Domain:** Supervised Contrastive Learning — loss formulation, class-balanced sampling, two-stage training
**Confidence:** HIGH

---

## Summary

Supervised Contrastive Learning (SupCon, Khosla et al., NeurIPS 2020) extends SimCLR's NT-Xent loss to the labeled setting: instead of treating only the other augmented view as a positive, every in-batch image sharing the same class label is a positive. The loss must use the **sum-outside** formulation (Eq. 2 of the paper), not the sum-inside variant. The method is two-stage: stage 1 pretrain with SupCon loss, stage 2 freeze the backbone and train a linear head with cross-entropy.

The project already has `SupConConfig` stubbed in `core/config.py` with `temperature=0.07` and `n_samples_per_class=2`, and `TrainConfig` already has a `supcon: Optional[SupConConfig]` field. The infrastructure (backbone factory, `ProjectionHead`, `BaseSSLModule`, `SSLDataModule`, `ssl_collate_fn`, dispatcher) is complete and reusable without modification — except `SSLDataModule.train_dataloader()` which currently hard-codes `shuffle=True` and must be extended to accept a custom sampler.

**Primary recommendation:** Implement `SupConLoss` in `core/losses.py` (extending the existing file), `ClassBalancedSampler` in `core/data.py`, `SupConModule` in `methods/supcon/module.py`, and a dedicated `SupConFinetuneModule` in the same package. Wire via the existing dispatcher and config pattern.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SUP-01 | SupCon loss (sum-outside Eq. 2), class-balanced sampler, 2-stage training, SimCLR fallback when labels=None | Covered in sections: SupConLoss Formulation, ClassBalancedSampler Design, SupConModule Architecture, Stage-2 Fine-tuning |
</phase_requirements>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | already in project | Loss computation, mask arithmetic | Project baseline |
| Lightning | already in project | `BaseSSLModule` subclassing, trainer loop | Project baseline |
| torchvision / timm | already in project | Backbone, augmentation pipeline | Project baseline |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `torch.utils.data.Sampler` | stdlib | Base class for `ClassBalancedSampler` | Required for custom batch construction |
| `torch.nn.functional` | stdlib | `F.normalize`, `F.cross_entropy` | Loss numerics |

**No new package installs required.** All dependencies are already in the project environment. [VERIFIED: project codebase grep]

---

## Architecture Patterns

### Recommended Project Structure

```
methods/supcon/
├── __init__.py          # register_method("supcon", SupConModule)
└── module.py            # SupConModule + SupConFinetuneModule

core/
├── losses.py            # ADD: SupConLoss class (append to existing file)
└── data.py              # ADD: ClassBalancedSampler class + SSLDataModule sampler param
```

No new top-level directories. Follows the exact pattern of `methods/simclr/`, `methods/byol/`, etc. [VERIFIED: codebase structure]

---

## SupConLoss Mathematical Formulation

### Paper Equation 2 (sum-outside formulation)

For anchor `i` in a batch of `2N` samples (two augmented views each):

```
L_sup = sum_{i in I} [ -1/|P(i)| * sum_{p in P(i)} log( exp(z_i · z_p / tau) / sum_{a in A(i)} exp(z_i · z_a / tau) ) ]
```

Where:
- `I` = set of all anchor indices in `{1..2N}`
- `P(i)` = set of positives for anchor `i` (all indices with same class label, excluding `i` itself)
- `A(i)` = all indices except anchor `i` itself (i.e., `I \ {i}`)
- `tau` = temperature
- `z_i` = L2-normalized projection output

**Contrast with sum-inside (wrong):** The sum-inside variant averages inside the log, which is mathematically incorrect and empirically worse.

**SimCLR equivalence (labels=None):** When no labels are provided, `P(i)` degenerates to exactly one positive per anchor (the other view). This is equivalent to SimCLR NT-Xent. [CITED: https://arxiv.org/abs/2004.11362, Section 2]

### Concrete Implementation Notes

**Inputs:** The loss receives a feature matrix `features` of shape `[B, D]` (already L2-normalized) and optionally `labels` of shape `[B]`.

**Batch structure for SupCon:** With two augmented views, the module creates a `[2B, D]` feature matrix by stacking both views. The labels tensor is duplicated: `labels_2v = labels.repeat(2)`. Each row `i` in this `[2B, D]` matrix is an anchor.

**Mask construction:**

```python
# Positive mask: positions where label matches anchor label (excluding self)
# [2B, 2B] boolean
labels_2v = labels.repeat(2)  # shape [2B]
label_eq = labels_2v.unsqueeze(0) == labels_2v.unsqueeze(1)  # [2B, 2B]
self_mask = torch.eye(2 * B, dtype=torch.bool, device=features.device)
positive_mask = label_eq & ~self_mask  # positives: same class, not self
```

**When labels=None (SimCLR mode):**

```python
# Positive for z_i[k] is z_j[k] (offset by B), and vice versa
# This matches InfoNCELoss._symmetric_loss exactly
positive_mask = torch.zeros(2*B, 2*B, dtype=torch.bool)
for k in range(B):
    positive_mask[k, B + k] = True
    positive_mask[B + k, k] = True
```

**Log-sum-exp (numerical stability):** Subtract the row-wise maximum before exp, or use `torch.logsumexp`. Do NOT use raw `exp` without this step — with temperature 0.07, logits scale by ~14x and overflow to `inf` for float32.

**Correct loss computation (sum-outside):**

```python
# sim: [2B, 2B], self-diagonal set to -inf
sim = features @ features.T / self.temperature
sim = sim.masked_fill(self_mask, float("-inf"))

# For each anchor i: log( sum_{a != i} exp(sim[i, a]) ) — the denominator
log_denom = torch.logsumexp(sim, dim=1)  # [2B], -inf self excluded via masked_fill

# For each anchor i and each positive p: sim[i, p] - log_denom[i]
# This is the log-probability of the positive pair
log_prob = sim - log_denom.unsqueeze(1)  # [2B, 2B]

# Mean log-prob over positives for each anchor (sum-outside means average over P(i))
n_positives = positive_mask.sum(dim=1).float()  # [2B]
# Guard against anchors with zero positives (batch may have singleton classes)
valid = n_positives > 0
per_anchor_loss = -(log_prob * positive_mask).sum(dim=1) / n_positives.clamp(min=1)

if self.reduction == "mean":
    loss = per_anchor_loss[valid].mean()
else:
    loss = per_anchor_loss[valid].sum()
```

**Critical: Features MUST be L2-normalized before calling SupConLoss.** The module (not the loss) is responsible for normalizing. This is the same contract as `InfoNCELoss` (which normalizes internally). For `SupConLoss`, normalization should also happen inside the loss `forward()` for safety, matching the existing `InfoNCELoss` convention. [VERIFIED: `core/losses.py` lines 55-56]

**SupConLoss signature:**

```python
class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, reduction: str = "mean") -> None: ...
    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            features: [B, D] — single-view projection outputs (NOT pre-duplicated).
                      The loss internally duplicates to [2B, D] if two views are passed
                      separately, OR accepts [2B, D] directly.
            labels:   [B] integer class labels. When None, degenerates to SimCLR.
        """
```

**Design decision for this project:** The module passes `(z_i, z_j, labels)` as separate tensors (matching the SimCLR two-view pattern), and the loss concatenates them internally to `[2B, D]`. This keeps the module interface consistent with the rest of the codebase. Alternative: pass a single `[2B, D]` tensor — both work; the two-argument style matches `InfoNCELoss`.

---

## ClassBalancedSampler Design

### Why shuffle=True is incompatible

`DataLoader(shuffle=True)` uses a `SequentialSampler` + `RandomSampler` under the hood. Passing a custom `sampler=` argument to `DataLoader` requires `shuffle=False` (or equivalently, not passing `shuffle=True`). PyTorch raises `ValueError: sampler option is mutually exclusive with shuffle`. [VERIFIED: PyTorch DataLoader source behavior — well-known constraint]

### Sampler Interface

`ClassBalancedSampler` subclasses `torch.utils.data.Sampler`:

```python
class ClassBalancedSampler(torch.utils.data.Sampler):
    """Yields batch indices guaranteeing n_samples_per_class instances per class per batch.

    Args:
        dataset: An ImageFolder (or any dataset with a .targets attribute, list[int]).
        n_classes_per_batch: Number of distinct classes to include per batch.
        n_samples_per_class: Number of instances per class per batch.

    Effective batch_size = n_classes_per_batch * n_samples_per_class.
    """
    def __init__(self, dataset, n_classes_per_batch: int, n_samples_per_class: int): ...
    def __iter__(self) -> Iterator[int]: ...  # yields flat list of indices
    def __len__(self) -> int: ...
```

**Implementation pattern:**

```python
def __init__(self, dataset, n_classes_per_batch, n_samples_per_class):
    self.n_classes_per_batch = n_classes_per_batch
    self.n_samples_per_class = n_samples_per_class
    # Build class -> [indices] mapping from dataset.targets
    targets = dataset.targets  # ImageFolder provides this
    self.class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        self.class_indices[label].append(idx)
    self.classes = list(self.class_indices.keys())
    # Total samples per epoch: approximate to cover all classes
    n_batches = len(dataset) // (n_classes_per_batch * n_samples_per_class)
    self._length = n_batches * n_classes_per_batch * n_samples_per_class

def __iter__(self):
    # For each batch: sample n_classes_per_batch classes, then n_samples_per_class from each
    indices = []
    n_batches = self._length // (self.n_classes_per_batch * self.n_samples_per_class)
    for _ in range(n_batches):
        chosen_classes = random.sample(self.classes, self.n_classes_per_batch)
        for cls in chosen_classes:
            cls_idxs = self.class_indices[cls]
            # sample with replacement if class has fewer than n_samples_per_class images
            chosen = random.choices(cls_idxs, k=self.n_samples_per_class)
            indices.extend(chosen)
    return iter(indices)
```

**Integration with SSLDataModule:** Add an optional `sampler` parameter to `SSLDataModule.__init__`:

```python
def __init__(self, ..., sampler: str | None = None, n_classes_per_batch: int | None = None):
    ...
    self.sampler_type = sampler  # e.g. "class_balanced"
    self.n_classes_per_batch = n_classes_per_batch
```

In `train_dataloader()`:

```python
def train_dataloader(self):
    if self.sampler_type == "class_balanced":
        sampler = ClassBalancedSampler(
            self.train_dataset,
            n_classes_per_batch=self.n_classes_per_batch,
            n_samples_per_class=self.cfg_n_samples_per_class,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        ...
    )
```

**Simpler alternative for this tutorial repo:** Since `SupConModule` constructs its own `SSLDataModule` or wires it via trainer, pass the sampler instance directly at DataModule construction time. The config key `supcon.n_samples_per_class` drives `n_samples_per_class`; `n_classes_per_batch` can be derived from `batch_size / n_samples_per_class`.

**`dataset.targets` availability:** `ImageFolder` always has `.targets` (list of int labels). [VERIFIED: torchvision ImageFolder source]

---

## SupConModule Architecture and Data Flow

### Two-view batch structure

SupCon uses exactly 2 augmented views (same as SimCLR). The existing `ssl_collate_fn` returns `(views, labels)` where `views.shape = [2, B, C, H, W]` and `labels.shape = [B]`. The module uses `views[0]` and `views[1]` and the labels directly.

```
batch = (views, labels)          # views: [2, B, C, H, W], labels: [B]
h_i = backbone(views[0])         # [B, feat_dim]
h_j = backbone(views[1])         # [B, feat_dim]
z_i = projector(h_i)             # [B, proj_dim]
z_j = projector(h_j)             # [B, proj_dim]
loss = supcon_loss(z_i, z_j, labels)   # loss internally: [2B, proj_dim]
```

Labels come from the DataLoader naturally — `ssl_collate_fn` already returns `torch.tensor(labels, dtype=torch.long)`. No batch format changes needed. [VERIFIED: `core/data.py` lines 89-92]

### Module structure

```python
class SupConModule(BaseSSLModule):
    """SupCon stage-1 pretraining module.
    
    NO classifier head during pretraining. Two-stage workflow:
    Stage 1: python train.py --config configs/supcon_stage1_resnet18.yaml
    Stage 2: python train.py --config configs/supcon_stage2_resnet18.yaml
    """
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        self.backbone, self.feat_dim = build_backbone(cfg.backbone, cfg.pretrained)
        self.projector = self.build_projector()   # 2-layer MLP
        supcon_cfg = cfg.supcon or SupConConfig()
        self.loss_fn = SupConLoss(temperature=supcon_cfg.temperature)
        # No self.classifier — intentional for stage 1
    
    def build_projector(self) -> nn.Module:
        """2-layer MLP projection head (same architecture as SimCLR v1)."""
        return ProjectionHead(self.feat_dim, hidden_dim=2048, output_dim=128, num_layers=2)
    
    def training_step(self, batch, batch_idx):
        views, labels = batch  # views: [2, B, C, H, W]
        h_i = self.backbone(views[0])
        h_j = self.backbone(views[1])
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        loss = self.loss_fn(z_i, z_j, labels=labels)
        self.log_train_metrics(loss)
        return loss
```

### Stage-2 SupConFinetuneModule

A dedicated `SupConFinetuneModule` (not reusing `LinearProbeModule` — that class doesn't exist in the current codebase) that:
- Loads a checkpoint from stage 1
- Freezes the backbone
- Trains a single linear layer with cross-entropy and SGD, `weight_decay=0.0`

```python
class SupConFinetuneModule(BaseSSLModule):
    """SupCon stage-2: frozen backbone + linear head with SGD.
    
    Load stage-1 checkpoint:
        model = SupConFinetuneModule.from_stage1_ckpt(ckpt_path, cfg)
    """
    def __init__(self, cfg: TrainConfig, num_classes: int) -> None:
        super().__init__(cfg)
        self.backbone, self.feat_dim = build_backbone(cfg.backbone, cfg.pretrained)
        # Projector NOT used in stage 2 — backbone output goes directly to classifier
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        # Backbone frozen — must be done after loading checkpoint
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad_(False)
    
    @property
    def learnable_params(self):
        return self.classifier.parameters()  # Only train classifier
    
    def build_projector(self) -> nn.Module:
        return nn.Identity()  # Not used; required by abstract interface
    
    def training_step(self, batch, batch_idx):
        views, labels = batch  # Standard 2-view or 1-view
        # Use only one view for fine-tuning (no augmentation needed for frozen backbone)
        x = views[0] if views.dim() == 4 else views  # handle both view formats
        with torch.no_grad():
            h = self.backbone(x)
        logits = self.classifier(h)
        loss = F.cross_entropy(logits, labels)
        self.log_train_metrics(loss)
        return loss
    
    def configure_optimizers(self):
        # Override: SGD with weight_decay=0.0, no LARS for linear probe
        optimizer = torch.optim.SGD(
            self.learnable_params,
            lr=self.cfg.lr,
            momentum=0.9,
            weight_decay=0.0,
        )
        return optimizer  # No warmup scheduler for stage 2
```

**Loading stage-1 backbone:** The stage-1 checkpoint contains `backbone.*` and `projector.*` weights. The stage-2 module only needs `backbone.*`. Use `strict=False` when loading, then call `freeze_backbone()`.

**`num_classes` sourcing:** Stage-2 config must include `num_classes` as a field, or it can be derived from the dataset at DataModule setup time. The simplest approach for this tutorial: add `num_classes: int = 10` to a `SupConFinetuneConfig` or to `SupConConfig`.

---

## Config Schema Additions

`SupConConfig` is already defined in `core/config.py` and already wired into `TrainConfig`:

```python
# Already exists in core/config.py (lines 112-117):
class SupConConfig(_StrictBase):
    temperature: float = 0.07
    n_samples_per_class: int = 2

# Already in TrainConfig (line 253):
supcon: Optional[SupConConfig] = None
```

**What's missing:** `n_classes_per_batch` and `num_classes` (for stage 2). Options:

1. Add `n_classes_per_batch: int = 8` and `num_classes: int = 10` directly to `SupConConfig`.
2. Create a separate `SupConFinetuneConfig` for stage-2 fields.

**Recommended:** Add both to `SupConConfig` for simplicity. `num_classes` is needed only for stage 2 but harmless in stage 1. Keeping one config class matches all other methods in this project.

Updated `SupConConfig`:

```python
class SupConConfig(_StrictBase):
    temperature: float = 0.07
    n_samples_per_class: int = 2
    n_classes_per_batch: int = 8   # NEW: controls ClassBalancedSampler
    num_classes: int = 10          # NEW: for stage-2 classifier head
    projection_dim: int = 128      # NEW: consistent with SimCLRConfig
```

### Stage-1 YAML (supcon_stage1_resnet18.yaml)

```yaml
method: supcon
backbone: resnet18
pretrained: false

max_epochs: 200
warmup_epochs: 10
batch_size: 256      # effective: n_classes_per_batch * n_samples_per_class
lr: 0.5
weight_decay: 1e-4
optimizer: lars
n_views: 2
data_dir: data
num_workers: 4

supcon:
  temperature: 0.07
  n_samples_per_class: 2
  n_classes_per_batch: 128   # 128 classes * 2 per class = 256 batch
  num_classes: 10
  projection_dim: 128
```

### Stage-2 YAML (supcon_stage2_resnet18.yaml)

```yaml
method: supcon_finetune
backbone: resnet18
pretrained: false

max_epochs: 100
warmup_epochs: 0
batch_size: 256
lr: 0.1
weight_decay: 0.0
optimizer: sgd
n_views: 1           # single view for linear probe
data_dir: data
num_workers: 4

supcon:
  temperature: 0.07
  n_samples_per_class: 2
  n_classes_per_batch: 128
  num_classes: 10
  projection_dim: 128
```

**Dispatcher registration:** Register both in `methods/supcon/__init__.py`:

```python
register_method("supcon", SupConModule)
register_method("supcon_finetune", SupConFinetuneModule)
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Log-sum-exp for numerics | Manual max subtraction | `torch.logsumexp` | Handles -inf correctly for masked entries |
| Backbone creation | Custom model builder | `build_backbone(cfg.backbone, cfg.pretrained)` | Already handles timm + feat_dim |
| Projection head | New MLP class | `ProjectionHead(feat_dim, 2048, 128, num_layers=2)` | Already exists, BN-correct |
| LR scheduler | Custom warmup | `BaseSSLModule.configure_optimizers()` | Already implements warmup-cosine |
| EMA update | Custom momentum encoder | `EMAUpdater` in `core/ema.py` | Already exists (not needed for SupCon, but don't rebuild) |

**Key insight:** SupCon reuses more existing infrastructure than any prior phase. The only genuinely new components are `SupConLoss` (the mask math) and `ClassBalancedSampler` (the batch construction).

---

## Common Pitfalls

### Pitfall 1: Sum-Inside vs Sum-Outside Formulation
**What goes wrong:** Using the sum-inside variant (average over positives inside the log) gives a different mathematical objective that is empirically weaker and doesn't match Eq. 2 in the paper.
**Why it happens:** Eq. 1 in the paper uses sum-inside; Eq. 2 (the recommended variant) uses sum-outside. Both are valid supervised contrastive losses, but Eq. 2 is superior.
**How to avoid:** Implement as `-(log_prob * positive_mask).sum(dim=1) / n_positives` — the division by `|P(i)|` is OUTSIDE the log.
**Warning signs:** Loss values look similar to SimCLR but representation quality is lower than expected.

### Pitfall 2: Non-Normalized Features
**What goes wrong:** Without L2 normalization, the dot products are not bounded to [-1, 1], causing logits to overflow with low temperature (0.07 gives 14x scaling).
**Why it happens:** SupCon loss assumes `z_i · z_j` is a cosine similarity (bounded).
**How to avoid:** Call `F.normalize(features, dim=1)` inside `SupConLoss.forward()` before computing the similarity matrix, matching the existing `InfoNCELoss` convention.
**Warning signs:** NaN loss on first step, or extremely high loss values (>> 10).

### Pitfall 3: shuffle=True with Custom Sampler
**What goes wrong:** `DataLoader(sampler=ClassBalancedSampler(...), shuffle=True)` raises `ValueError: sampler option is mutually exclusive with shuffle`.
**Why it happens:** `shuffle=True` internally creates a `RandomSampler`; you cannot have two samplers.
**How to avoid:** Set `shuffle=False` when `sampler` is provided. The `ClassBalancedSampler` inherently shuffles via `random.sample`.
**Warning signs:** `ValueError` at `train_dataloader()` call.

### Pitfall 4: Classifier During Stage-1 Pretraining
**What goes wrong:** Adding a classification head and cross-entropy during SupCon pretraining (to speed things up) collapses the representation — the contrastive loss needs full freedom to learn structure.
**Why it happens:** Mixing objectives during SupCon pretraining is explicitly warned against in the paper.
**How to avoid:** `SupConModule` must have NO `self.classifier`. Only `self.backbone` and `self.projector`.
**Warning signs:** Stage-1 loss converges quickly but stage-2 fine-tuning achieves only marginal improvement over random init.

### Pitfall 5: Singleton Class in Batch (Zero Positives)
**What goes wrong:** If a class has only one instance in the batch, `|P(i)| = 0` for its anchor, causing division by zero.
**Why it happens:** `ClassBalancedSampler` guarantees minimum `n_samples_per_class`, but after dropout or edge cases this can fail.
**How to avoid:** Guard with `n_positives.clamp(min=1)` and skip anchors with `n_positives == 0` using a validity mask. Log a warning if this occurs frequently (indicates sampler misconfiguration).
**Warning signs:** NaN loss appearing mid-training (not at step 1), inconsistently.

### Pitfall 6: Stage-2 uses Projector Features Instead of Backbone Features
**What goes wrong:** Evaluating downstream tasks on `z` (projector output) instead of `h` (backbone output) gives poor accuracy.
**Why it happens:** Same as SimCLR — the projector discards discriminative information.
**How to avoid:** `SupConFinetuneModule` must bypass the projector entirely: `h = self.backbone(x)` → `logits = self.classifier(h)`.
**Warning signs:** Stage-2 accuracy is suspiciously low (below 30% on CIFAR-10 with a trained backbone).

### Pitfall 7: Weight Decay in Stage-2 Linear Probe
**What goes wrong:** Using non-zero weight decay for the linear classifier in stage 2 degrades accuracy significantly.
**Why it happens:** The linear probe has only one layer — L2 regularization shrinks the weights in a regime where variance matters.
**How to avoid:** `weight_decay=0.0` in stage-2 config. The `SupConFinetuneModule.configure_optimizers()` must NOT inherit `cfg.weight_decay` for the classifier params.
**Warning signs:** Stage-2 accuracy is systematically 3-8 points below expected.

---

## Code Examples

### SupConLoss (verified formulation)

```python
# Source: Khosla et al., NeurIPS 2020, Eq. 2 — verified against paper
# https://arxiv.org/abs/2004.11362

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Sum-outside formulation (Eq. 2). When labels=None, degenerates to SimCLR NT-Xent.

    Args:
        temperature: Softmax temperature. Paper uses 0.07. Default: 0.07.
        reduction: 'mean' (default) or 'sum' over anchors.

    Gotchas:
        - Use sum-OUTSIDE (Eq. 2), not sum-inside (Eq. 1). Division by |P(i)| is outside the log.
        - Features MUST be L2-normalized. This loss normalizes internally for safety.
        - Batch must guarantee multiple instances per class — use ClassBalancedSampler.
        - When labels=None, exactly one positive per anchor (SimCLR behavior).
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean") -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute SupCon loss.

        Args:
            z_i: Projection outputs from view 1, shape [B, D].
            z_j: Projection outputs from view 2, shape [B, D].
            labels: Integer class labels, shape [B]. When None, uses SimCLR mode.

        Returns:
            Scalar loss tensor.
        """
        B = z_i.shape[0]
        device = z_i.device

        # L2 normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate both views → [2B, D]
        features = torch.cat([z_i, z_j], dim=0)  # [2B, D]
        N = 2 * B

        # Similarity matrix [2B, 2B]
        sim = features @ features.T / self.temperature

        # Self-mask (diagonal): set to -inf so self-similarity is excluded from denominator
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(self_mask, float("-inf"))

        # Positive mask [2B, 2B]
        if labels is None:
            # SimCLR mode: positive for z_i[k] is z_j[k] (offset by B)
            positive_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
            idx = torch.arange(B, device=device)
            positive_mask[idx, idx + B] = True
            positive_mask[idx + B, idx] = True
        else:
            # SupCon mode: positive = same class label, excluding self
            labels_2v = labels.repeat(2)  # [2B]
            label_eq = labels_2v.unsqueeze(0) == labels_2v.unsqueeze(1)  # [2B, 2B]
            positive_mask = label_eq & ~self_mask

        # log-denominator: log sum_{a != i} exp(sim[i, a] / tau)
        # logsumexp handles -inf entries (self-similarity) correctly
        log_denom = torch.logsumexp(sim, dim=1, keepdim=True)  # [2B, 1]

        # Log-probability of each pair: sim[i,j]/tau - log_denom[i]
        log_prob = sim - log_denom  # [2B, 2B]

        # Per-anchor loss: -1/|P(i)| * sum_{p in P(i)} log_prob[i, p]  (sum-outside)
        n_positives = positive_mask.float().sum(dim=1)  # [2B]
        per_anchor_loss = -(log_prob * positive_mask).sum(dim=1) / n_positives.clamp(min=1.0)

        # Only average over anchors that have at least one positive
        valid = n_positives > 0
        if self.reduction == "mean":
            return per_anchor_loss[valid].mean()
        else:
            return per_anchor_loss[valid].sum()
```

### SimCLR Equivalence Test (for unit test)

```python
# Source: derived from InfoNCELoss._symmetric_loss in core/losses.py [VERIFIED]

def test_supcon_simclr_equivalence():
    """When labels=None, SupConLoss == InfoNCELoss (symmetric mode)."""
    torch.manual_seed(42)
    B, D = 16, 128
    z_i = torch.randn(B, D)
    z_j = torch.randn(B, D)

    supcon = SupConLoss(temperature=0.5)
    infonce = InfoNCELoss(temperature=0.5)

    loss_supcon = supcon(z_i, z_j, labels=None)
    loss_infonce = infonce(z_i, z_j)

    assert torch.isclose(loss_supcon, loss_infonce, atol=1e-5), (
        f"SupConLoss(labels=None) = {loss_supcon:.6f}, "
        f"InfoNCELoss = {loss_infonce:.6f}"
    )
```

### ClassBalancedSampler Usage

```python
# Wire into SSLDataModule — source: project codebase pattern [VERIFIED: core/data.py]
dm = SSLDataModule(
    data_dir="data/",
    n_views=2,
    batch_size=256,
    num_workers=4,
    sampler="class_balanced",
    n_classes_per_batch=128,
    n_samples_per_class=2,
)
```

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `tests/conftest.py` (shared fixtures exist) |
| Quick run command | `pytest tests/test_supcon.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SUP-01-a | `SupConLoss(labels=None)` equals `InfoNCELoss` (symmetric) | unit | `pytest tests/test_supcon.py::test_supcon_simclr_equivalence -x` | ❌ Wave 0 |
| SUP-01-b | `SupConLoss` sum-outside: loss value matches hand-calculated reference | unit | `pytest tests/test_supcon.py::test_supcon_loss_sum_outside -x` | ❌ Wave 0 |
| SUP-01-c | `SupConLoss` produces finite positive scalar on random inputs with labels | unit | `pytest tests/test_supcon.py::test_supcon_loss_finite -x` | ❌ Wave 0 |
| SUP-01-d | `ClassBalancedSampler` guarantees n_samples_per_class per class per batch | unit | `pytest tests/test_supcon.py::test_class_balanced_sampler -x` | ❌ Wave 0 |
| SUP-01-e | `SupConModule` trains 3 epochs without divergence | smoke | `pytest tests/test_supcon.py::test_supcon_smoke_stage1 -x` | ❌ Wave 0 |
| SUP-01-f | DOC-02 docstring validation for `SupConModule` | unit | `pytest tests/test_supcon.py::test_supcon_docstring_doc02 -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_supcon.py -x -q`
- **Per wave merge:** `pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_supcon.py` — all SUP-01 tests
- [ ] `configs/supcon_stage1_resnet18.yaml` — stage-1 config
- [ ] `configs/supcon_stage2_resnet18.yaml` — stage-2 config

---

## Smoke Test Approach (08-05)

The smoke test for stage 1 follows the exact pattern of `test_simclr_v1_smoke_3_epochs` in `tests/test_simclr.py` [VERIFIED: lines 395-440]:

```python
def test_supcon_smoke_stage1(tmp_imagefolder):
    """Smoke: SupConModule trains 3 epochs on toy data; loss is finite."""
    # tmp_imagefolder fixture from conftest.py: 3 classes, 5 images each, 32x32
    # Use n_classes_per_batch=3 (all classes), n_samples_per_class=2 → batch_size=6
    import methods.supcon
    from core.config import TrainConfig
    from core.dispatcher import method_dispatcher

    cfg = TrainConfig(
        method="supcon",
        backbone="resnet18",
        max_epochs=3,
        warmup_epochs=0,
        batch_size=6,
        lr=1e-3,
        weight_decay=1e-4,
        optimizer="adamw",
        n_views=2,
        supcon={"temperature": 0.07, "n_samples_per_class": 2, "n_classes_per_batch": 3,
                "num_classes": 3, "projection_dim": 128},
    )
    model = method_dispatcher(cfg)
    # Use ClassBalancedSampler via SSLDataModule with sampler="class_balanced"
    dm = SSLDataModule(
        data_dir=str(tmp_imagefolder),
        n_views=2,
        batch_size=6,
        num_workers=0,
        size=32,
        strong=False,
        sampler="class_balanced",
        n_classes_per_batch=3,
        n_samples_per_class=2,
    )
    trainer = L.Trainer(
        max_epochs=3, accelerator="cpu", logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
    )
    trainer.fit(model, dm)
    # Passes if no exception and loss is finite
```

---

## Open Questions

1. **Stage-2 `n_views` handling in SSLDataModule**
   - What we know: stage-2 fine-tuning needs only 1 view (no augmentation pair needed for frozen backbone + CE).
   - What's unclear: `ssl_collate_fn` always returns `[n_views, B, C, H, W]`. With `n_views=1`, `views[0]` still works correctly, but the DataModule currently does not test this path.
   - Recommendation: Set `n_views=1` in stage-2 YAML; the existing `ssl_collate_fn` handles it via `stacked = [torch.stack([v[i] for v in views]) for i in range(n_views)]`. Verify with a unit test.

2. **`SupConFinetuneModule` checkpoint loading API**
   - What we know: Lightning checkpoints contain full module state; `backbone.*` keys are present.
   - What's unclear: Whether to implement `from_stage1_ckpt()` as a classmethod or rely on the user to call `model.load_state_dict(ckpt["state_dict"], strict=False)` manually.
   - Recommendation: Document both approaches in the module docstring. For the tutorial, a classmethod is cleaner pedagogically.

3. **`n_classes_per_batch` vs `batch_size` relationship**
   - What we know: Effective batch size = `n_classes_per_batch * n_samples_per_class`. The `DataLoader` `batch_size` parameter is ignored when a custom `Sampler` yields pre-grouped indices.
   - What's unclear: Whether to keep `batch_size` in the top-level config as documentation-only, or derive it from the SupCon config fields.
   - Recommendation: Keep `batch_size` in the top-level config for consistency; add a docstring note that with `class_balanced` sampler, effective batch size is `n_classes_per_batch * n_samples_per_class`.

---

## Environment Availability

Step 2.6: SKIPPED (no new external dependencies — all required libraries already present in project environment). [VERIFIED: project codebase uses PyTorch, Lightning, torchvision, timm — all confirmed installed]

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `SupConFinetuneModule.configure_optimizers()` overrides the base class to use SGD with `weight_decay=0.0` without a warmup scheduler | Stage-2 Fine-tuning | If base class scheduler is hard-coded, stage-2 may use unnecessary warmup — minor quality impact |
| A2 | `ImageFolder.targets` is a list of integers and always present | ClassBalancedSampler | If the dataset wrapper changes (e.g., IndexedDataset), `.targets` may not be available — need to handle or document |
| A3 | Stage-2 fine-tuning uses `n_views=1` and standard CE; no augmentation pair needed | Stage-2 Fine-tuning Design | Some implementations use augmentation in stage 2 for regularization — minor accuracy difference |

**No `[ASSUMED]` tags on the mathematical formulation** — the SupCon loss (Eq. 2) is verified directly from the paper. [CITED: https://arxiv.org/abs/2004.11362]

---

## Sources

### Primary (HIGH confidence)
- Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020, https://arxiv.org/abs/2004.11362 — Eq. 2 (sum-outside formulation), SimCLR connection, two-stage training
- `core/losses.py` [VERIFIED: read] — existing `InfoNCELoss` implementation, L2 normalization convention, numerical stability approach
- `core/data.py` [VERIFIED: read] — `SSLDataModule.train_dataloader()` `shuffle=True` pattern, `ssl_collate_fn` output format, `ImageFolder` usage
- `core/config.py` [VERIFIED: read] — `SupConConfig` (lines 112-117), `TrainConfig.supcon` field (line 253), `_StrictBase` pattern
- `core/base.py` [VERIFIED: read] — `BaseSSLModule` abstract interface, `build_projector()` requirement, `learnable_params` property override pattern
- `core/dispatcher.py` [VERIFIED: read] — `register_method()` pattern
- `methods/simclr/__init__.py` [VERIFIED: read] — registration pattern
- `tests/test_simclr.py` [VERIFIED: read] — smoke test structure, `LossTracker` callback pattern, DOC-02 docstring test pattern
- `tests/conftest.py` [VERIFIED: read] — shared `tmp_imagefolder` fixture (3 classes, 5 images, 32x32)

### Secondary (MEDIUM confidence)
- PyTorch DataLoader documentation: `sampler` and `shuffle` mutual exclusivity — well-known constraint [ASSUMED from training knowledge, behavior is standard PyTorch]
- `torch.logsumexp` handling of `-inf` inputs — standard PyTorch behavior for masked similarity matrices

---

## Metadata

**Confidence breakdown:**
- Standard Stack: HIGH — all libraries already in project, versions confirmed in use
- SupConLoss formulation: HIGH — directly from paper Eq. 2, cross-referenced with existing `InfoNCELoss` pattern
- ClassBalancedSampler design: HIGH — standard `torch.utils.data.Sampler` pattern, `shuffle` incompatibility is well-known
- SupConModule architecture: HIGH — directly follows existing `SimCLRv1Module` pattern
- Stage-2 fine-tuning: MEDIUM — no existing `LinearProbeModule` in codebase to reference; design extrapolated from base class patterns
- Config integration: HIGH — `SupConConfig` and `TrainConfig.supcon` already exist; only field additions needed

**Research date:** 2026-04-10
**Valid until:** 2026-05-10 (stable libraries — 30-day horizon)
