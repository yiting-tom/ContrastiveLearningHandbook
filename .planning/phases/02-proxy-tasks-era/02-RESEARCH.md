# Phase 2: Proxy Tasks Era - Research

**Researched:** 2026-04-01
**Domain:** Instance Discrimination, Invariant Spread, Memory Bank infrastructure
**Confidence:** HIGH

## Summary

Phase 2 implements the two earliest contrastive SSL methods (Instance Discrimination and Invariant Spread) and the shared `MemoryBank` infrastructure. The foundation from Phase 1 provides `BaseSSLModule`, `InfoNCELoss`, `ProjectionHead`, `ContrastiveAugmentation(strong=False)`, and the `method_dispatcher` registry pattern. All of these are production-ready and well-tested (70/70 tests passing).

The key technical challenges are: (1) implementing `MemoryBank` as `nn.Embedding` with gradient-free update-by-index semantics, (2) a standalone NCE loss with a fixed normalization constant Z stored as a `register_buffer`, and (3) integrating with `SSLDataModule` to pass sample indices through the training pipeline. `InvariantSpreadModule` is simpler -- it reuses `InfoNCELoss` in symmetric mode with no bank and no queue.

**Primary recommendation:** Follow the existing patterns exactly -- sub-configs on `TrainConfig`, registration via `register_method()` in `__init__.py`, and test patterns from Phase 1 (random tensors for unit tests, synthetic ImageFolder for integration tests).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Add `InstanceDiscriminationConfig` and `InvariantSpreadConfig` Pydantic sub-config classes to `core/config.py`, then add them as `Optional` fields on `TrainConfig` -- following the exact same pattern as `SimCLRConfig`, `MoCoConfig`, etc. already present there. The `extra='forbid'` constraint (Phase 1 D-08) means any YAML key not declared as a field raises `ValidationError` at load time; per-method hyper-params (e.g., `n_negatives`, `bank_size`) must be declared here.
- **D-02:** The NCE loss for Instance Discrimination (plan 02-02) is a new standalone class in `methods/instance_discrimination/` -- it does **not** subclass or wrap `InfoNCELoss`. The Z-normalization semantics (fixed scalar estimated on first batch, epsilon=1e-7 in denominator) are incompatible with `InfoNCELoss.forward()`, which L2-normalizes inputs internally and has no slot for a fixed Z.
- **D-03:** `InvariantSpreadModule` (plan 02-04) reuses `InfoNCELoss` from `core/losses.py` directly in symmetric mode -- no new loss class needed for that method.
- **D-04:** `MemoryBank` is implemented as `nn.Embedding` with `self.weight.requires_grad = False` set immediately after construction. The bank is never sent to the optimizer. Updates happen via direct index assignment (`bank.weight.data[indices] = features`) -- not through backprop. This is required because `BaseSSLModule.learnable_params` defaults to `self.parameters()`, which would otherwise include the embedding weight.
- **D-05:** `MemoryBank` is placed in `core/memory_bank.py` (shared infrastructure -- used by Instance Discrimination now, and by CMC in a future phase). It is a standalone utility with no method-specific logic.
- **D-06:** Each method registers itself by calling `register_method()` from within its `__init__.py`: `methods/instance_discrimination/__init__.py` and `methods/invariant_spread/__init__.py`. The top-level `methods/__init__.py` imports both sub-packages to trigger registration before `method_dispatcher` is called.
- **D-07:** Method file layout: `methods/instance_discrimination/module.py` (InstanceDiscriminationModule), `methods/instance_discrimination/losses.py` (NCE loss with fixed Z), `methods/invariant_spread/module.py` (InvariantSpreadModule).

### Claude's Discretion
- Exact NCE sampling strategy for drawing m=4096 negatives from the bank (random without replacement from non-positive indices)
- `MemoryBank` initialization variance (L2-normalized random vectors -- uniform on hypersphere)
- Exact config field names and defaults for both sub-configs
- Where to place the weak-augmentation call in `InstanceDiscriminationModule.training_step`

### Deferred Ideas (OUT OF SCOPE)
None -- analysis stayed within Phase 2 scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ERA1-01 | Instance Discrimination: encoder + memory bank, NCE loss, 4096 negatives, Z fixed after first batch, tau=0.07, epsilon=1e-7, bank staleness documented | MemoryBank as nn.Embedding (D-04/D-05), standalone NCE loss (D-02), config extension (D-01), dispatcher registration (D-06) |
| ERA1-02 | Invariant Spread: in-batch softmax, no bank/queue, one augmented view, InfoNCELoss reuse, batch-size sensitivity documented | InfoNCELoss symmetric mode reuse (D-03), n_views=1 with 2 views produced by MultiViewTransform |
| INFRA-02 | MemoryBank(n_samples, dim) as nn.Embedding with update-by-index, shared by Instance Discrimination and CMC | core/memory_bank.py placement (D-05), gradient-free updates (D-04) |
</phase_requirements>

## Standard Stack

### Core (already installed and verified)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0 | nn.Embedding for MemoryBank, register_buffer for Z | Foundation |
| Lightning | 2.5.2 | BaseSSLModule, Trainer for smoke tests | Foundation |
| timm | 1.0.19 | build_backbone for encoders | Foundation |
| Pydantic | 2.11.7 | Config sub-classes with extra='forbid' | Foundation |
| torchvision | (bundled) | transforms.v2 for ContrastiveAugmentation | Foundation |

### New Components (to build, not install)
| Component | Location | Purpose |
|-----------|----------|---------|
| MemoryBank | `core/memory_bank.py` | nn.Embedding bank with update-by-index |
| NCELossWithFixedZ | `methods/instance_discrimination/losses.py` | (m+1)-way NCE with fixed Z |
| InstanceDiscriminationModule | `methods/instance_discrimination/module.py` | Full method module |
| InvariantSpreadModule | `methods/invariant_spread/module.py` | Full method module |
| InstanceDiscriminationConfig | `core/config.py` | Sub-config for ID method |
| InvariantSpreadConfig | `core/config.py` | Sub-config for IS method |

No new pip packages required.

## Architecture Patterns

### Project Structure (new files for Phase 2)
```
core/
  memory_bank.py            # INFRA-02: shared MemoryBank
  config.py                 # MODIFIED: +2 sub-configs, +2 Optional fields on TrainConfig
  __init__.py               # MODIFIED: +MemoryBank re-export
methods/
  __init__.py               # MODIFIED: import instance_discrimination, invariant_spread
  instance_discrimination/
    __init__.py             # register_method("instance_discrimination", ...)
    module.py               # InstanceDiscriminationModule(BaseSSLModule)
    losses.py               # NCELossWithFixedZ
  invariant_spread/
    __init__.py             # register_method("invariant_spread", ...)
    module.py               # InvariantSpreadModule(BaseSSLModule)
configs/
  instance_discrimination.yaml
  invariant_spread.yaml
tests/
  test_memory_bank.py
  test_nce_loss.py
  test_instance_discrimination.py
  test_invariant_spread.py
```

### Pattern 1: MemoryBank as nn.Embedding (gradient-free)

**What:** `nn.Embedding(n_samples, dim)` with `requires_grad=False` on the weight tensor. Updates via direct data assignment, not backprop.

**Why nn.Embedding:** Provides efficient index-based lookup (`bank(indices)` returns features for those indices) and is a standard PyTorch module that serializes/deserializes with checkpoints automatically.

**Initialization (uniform on hypersphere):**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryBank(nn.Module):
    def __init__(self, n_samples: int, dim: int):
        super().__init__()
        self.bank = nn.Embedding(n_samples, dim)
        # Initialize with L2-normalized random vectors (uniform on hypersphere)
        nn.init.normal_(self.bank.weight.data)
        self.bank.weight.data = F.normalize(self.bank.weight.data, dim=1)
        # CRITICAL: disable gradients — bank is never sent to optimizer
        self.bank.weight.requires_grad = False

    def get(self, indices: torch.Tensor) -> torch.Tensor:
        """Retrieve features by index. Returns L2-normalized vectors."""
        return self.bank(indices)

    @torch.no_grad()
    def update(self, indices: torch.Tensor, features: torch.Tensor) -> None:
        """Update bank entries by index with L2-normalized features."""
        features = F.normalize(features.detach(), dim=1)
        self.bank.weight.data[indices] = features
```

**Key details:**
- `self.bank.weight.requires_grad = False` must be set immediately after construction
- `BaseSSLModule.learnable_params` defaults to `self.parameters()`, which includes all submodules. Since `requires_grad=False`, these parameters will NOT be added to the optimizer by `configure_optimizers()` (PyTorch optimizers only step on parameters with `requires_grad=True` by default, but the list is still populated). The safer approach: override `learnable_params` in `InstanceDiscriminationModule` to explicitly exclude the bank. Example:

```python
@property
def learnable_params(self):
    # Exclude memory bank from optimizer
    yield from self.backbone.parameters()
    yield from self.projector.parameters()
```

- `.detach()` on features before assignment prevents any gradient graph from persisting
- Bank staleness gotcha must be in class docstring: features stored from earlier encoder snapshots become stale as the encoder trains, leading to inconsistent negative samples. MoCo's queue (FIFO with momentum encoder) solves this.

### Pattern 2: NCE Loss with Fixed Z

**Mathematical formulation (Wu et al., 2018):**

The (m+1)-way NCE loss for Instance Discrimination treats each image as its own class. For anchor i with positive feature v_i from the bank and m=4096 sampled negatives:

```
P(i | v) = exp(v^T * f_i / tau) / Z

where Z = n_samples * (mean of exp(v^T * f / tau) over all v in bank)
```

Z is estimated once from the first mini-batch and then fixed for the rest of training. Recomputing Z each step destabilizes training (ERA1-01 gotcha).

**Implementation pattern:**
```python
class NCELossWithFixedZ(nn.Module):
    def __init__(self, temperature: float = 0.07, n_negatives: int = 4096, eps: float = 1e-7):
        super().__init__()
        self.temperature = temperature
        self.n_negatives = n_negatives
        self.eps = eps
        # Z stored as register_buffer so it survives checkpoint/resume
        self.register_buffer("Z", torch.tensor(-1.0))  # sentinel: -1 means not estimated yet
        self.register_buffer("z_initialized", torch.tensor(False))

    def forward(self, query: torch.Tensor, positive: torch.Tensor,
                negatives: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, D] L2-normalized encoder output for current batch
            positive: [B, D] L2-normalized bank features for same indices
            negatives: [B, m, D] L2-normalized sampled negative features
        Returns:
            Scalar loss
        """
        # Positive logits: [B, 1]
        pos_logit = (query * positive).sum(dim=1, keepdim=True) / self.temperature
        # Negative logits: [B, m]
        neg_logits = torch.bmm(negatives, query.unsqueeze(2)).squeeze(2) / self.temperature

        # Estimate Z on first call
        if not self.z_initialized:
            with torch.no_grad():
                all_logits = torch.cat([pos_logit, neg_logits], dim=1)
                self.Z = all_logits.exp().mean() * self.n_negatives
                self.z_initialized.fill_(True)

        # NCE probability
        pos_prob = pos_logit.exp() / (self.Z + self.eps)
        neg_prob = neg_logits.exp() / (self.Z + self.eps)

        # NCE loss: -log(pos) - sum(log(1 - neg))
        loss = -pos_prob.log().mean() - (1 - neg_prob + self.eps).log().sum(dim=1).mean()
        return loss
```

**Key details:**
- `register_buffer("Z", ...)` ensures Z persists across checkpoint save/load cycles
- `register_buffer("z_initialized", ...)` as a boolean tensor tracks initialization state
- The sentinel pattern (check `z_initialized`) is cleaner than checking `Z == -1`
- Z must NOT change after initialization -- the test must verify this explicitly
- `eps = 1e-7` in the denominator prevents division by zero and log(0)

### Pattern 3: Negative Sampling from Bank

```python
def _sample_negatives(self, positive_indices: torch.Tensor, bank: MemoryBank) -> torch.Tensor:
    """Sample m negatives from bank, excluding positive indices."""
    n_samples = bank.bank.weight.shape[0]
    m = self.n_negatives  # 4096
    B = positive_indices.shape[0]

    # For each sample in batch, sample m indices excluding the positive
    neg_features = []
    for i in range(B):
        # Create mask excluding positive index
        mask = torch.ones(n_samples, dtype=torch.bool)
        mask[positive_indices[i]] = False
        valid_indices = torch.where(mask)[0]
        # Sample without replacement
        perm = torch.randperm(valid_indices.shape[0], device=positive_indices.device)[:m]
        sampled_indices = valid_indices[perm]
        neg_features.append(bank.get(sampled_indices))
    return torch.stack(neg_features)  # [B, m, D]
```

**Optimization note:** The per-sample loop is fine for tutorial code. For production, you could batch-sample and just accept that occasionally a positive leaks into negatives (negligible effect with 4096 negatives from 50000 CIFAR-10 samples).

A simpler alternative that is equally valid for tutorial purposes:
```python
# Sample m random indices globally (positive collision is negligible)
neg_indices = torch.randint(0, n_samples, (m,))
negatives = bank.get(neg_indices)  # [m, D]
# Broadcast to [B, m, D]
negatives = negatives.unsqueeze(0).expand(B, -1, -1)
```

### Pattern 4: SSLDataModule Index Passing

**Critical integration point:** `InstanceDiscriminationModule.training_step` needs sample indices to look up and update the memory bank. The current `SSLDataModule` uses `ImageFolder` which returns `(image, label)`. The collate function `ssl_collate_fn` produces `(views_tensor, labels_tensor)`.

**Solution:** The module needs indices from the dataset. Two approaches:

**Approach A (recommended): Wrapper dataset that returns index:**
```python
class IndexedDataset(torch.utils.data.Dataset):
    """Wraps a dataset to also return the sample index."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]  # (views_list, label)
        return (*data, idx)
```

This requires a modified collate function that also stacks indices:
```python
def ssl_collate_with_index(batch):
    views, labels, indices = zip(*batch)
    n_views = len(views[0])
    stacked = [torch.stack([v[i] for v in views]) for i in range(n_views)]
    return torch.stack(stacked), torch.tensor(labels, dtype=torch.long), torch.tensor(indices, dtype=torch.long)
```

**Approach B: Use batch_idx and compute indices from DataLoader state.**
This is fragile and depends on shuffle order. Not recommended.

**Approach C: Access `dataset.targets` or use label as proxy.**
Not applicable -- Instance Discrimination treats each image as its own class.

**Recommendation:** Use Approach A. The `InstanceDiscriminationModule` should create its own DataLoader (or accept a modified one) that wraps the dataset with `IndexedDataset`. This can be done by overriding how the module receives data, or by having the training script wrap the dataset. Since `training_step` receives whatever the DataLoader yields, the module just unpacks `(views, labels, indices)` instead of `(views, labels)`.

The cleanest approach: InstanceDiscriminationModule handles this in its `training_step`:
```python
def training_step(self, batch, batch_idx):
    if len(batch) == 3:
        views, labels, indices = batch
    else:
        views, labels = batch
        indices = None
    # views shape: [n_views, B, C, H, W]
    x = views[0]  # single view for Instance Discrimination
    ...
```

### Pattern 5: InvariantSpreadModule -- InfoNCELoss Reuse

**What:** Invariant Spread uses in-batch cross-entropy with no memory bank. It produces two augmented views (n_views=2) and uses the existing `InfoNCELoss` in symmetric mode.

**How it maps to InfoNCELoss:**
- `InfoNCELoss.forward(z_i, z_j, queue=None)` -> symmetric in-batch mode
- z_i = projection of view 1, z_j = projection of view 2
- All 2B-2 other samples serve as negatives (no external bank/queue)
- InfoNCELoss already L2-normalizes internally

```python
class InvariantSpreadModule(BaseSSLModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg)
        self.backbone, feat_dim = build_backbone(cfg.backbone, cfg.pretrained)
        self.projector = self.build_projector()
        inv_cfg = cfg.invariant_spread or InvariantSpreadConfig()
        self.loss_fn = InfoNCELoss(temperature=inv_cfg.temperature)

    def build_projector(self):
        # Reuse ProjectionHead -- standard 2-layer MLP
        _, feat_dim = build_backbone(self.cfg.backbone, self.cfg.pretrained)
        return ProjectionHead(feat_dim, 2048, 128, num_layers=2)

    def training_step(self, batch, batch_idx):
        views, labels = batch  # views: [2, B, C, H, W]
        z_i = self.projector(self.backbone(views[0]))
        z_j = self.projector(self.backbone(views[1]))
        loss = self.loss_fn(z_i, z_j)  # symmetric in-batch, no queue
        self.log_train_metrics(loss)
        return loss
```

**n_views clarification:**
- Invariant Spread uses `n_views=2` (two augmented views of each image)
- The `SSLDataModule(n_views=2, strong=False)` produces weak augmentations for era-1
- ERA1-02 says "one augmented view per image" -- this means one view beyond the original, i.e., the model sees two views total (original concept). In practice with `MultiViewTransform`, set `n_views=2` to produce two augmented views (both are randomly augmented, neither is the "original").

### Pattern 6: Config Extension

**Existing pattern in `core/config.py`:**
```python
class SimCLRConfig(_StrictBase):
    temperature: float = 0.5
    projection_dim: int = 128

# On TrainConfig:
simclr: Optional[SimCLRConfig] = None
```

**New configs to add:**
```python
class InstanceDiscriminationConfig(_StrictBase):
    """Instance Discrimination method-specific hyper-parameters."""
    temperature: float = 0.07
    n_negatives: int = 4096
    projection_dim: int = 128

class InvariantSpreadConfig(_StrictBase):
    """Invariant Spread method-specific hyper-parameters."""
    temperature: float = 0.07
    projection_dim: int = 128

# On TrainConfig, add:
instance_discrimination: Optional[InstanceDiscriminationConfig] = None
invariant_spread: Optional[InvariantSpreadConfig] = None
```

The `extra='forbid'` from `_StrictBase` means any unknown YAML key under `instance_discrimination:` or `invariant_spread:` raises `ValidationError`.

### Pattern 7: Dispatcher Registration

**From `core/dispatcher.py`:**
```python
register_method(name: str, cls: type[BaseSSLModule]) -> None
```

**Registration in `methods/instance_discrimination/__init__.py`:**
```python
from core.dispatcher import register_method
from methods.instance_discrimination.module import InstanceDiscriminationModule

register_method("instance_discrimination", InstanceDiscriminationModule)
```

**Registration in `methods/invariant_spread/__init__.py`:**
```python
from core.dispatcher import register_method
from methods.invariant_spread.module import InvariantSpreadModule

register_method("invariant_spread", InvariantSpreadModule)
```

**Trigger in `methods/__init__.py`:**
```python
# methods package -- each SSL method gets its own sub-package
import methods.instance_discrimination  # noqa: F401
import methods.invariant_spread         # noqa: F401
```

The `import` triggers the `__init__.py` of each sub-package, which calls `register_method()`. This must happen before `method_dispatcher(cfg)` is called.

**Important:** The existing `test_dispatcher.py` uses a `clean_registry` fixture (autouse) that saves/restores `_METHOD_REGISTRY`. New tests that import methods sub-packages must account for side-effect registration.

### Anti-Patterns to Avoid
- **Subclassing InfoNCELoss for NCE:** D-02 explicitly forbids this. InfoNCELoss L2-normalizes internally and has no Z parameter slot.
- **Sending MemoryBank to optimizer:** The bank weight must have `requires_grad=False`. Even though PyTorch skips zero-grad parameters, explicitly excluding is safer.
- **Recomputing Z every step:** This destabilizes training. Z must be estimated once and fixed.
- **Using `on_train_batch_end` for bank updates:** Bank updates should happen in `training_step`, after computing the loss but before returning. `on_train_batch_end` is reserved for EMA updates (which Instance Discrimination does not use).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| In-batch contrastive loss | Custom cross-entropy | `InfoNCELoss` symmetric mode | Already tested, handles masking and scaling |
| Projection MLP | Custom nn.Sequential | `ProjectionHead` | Handles BN+ReLU intermediate, BN-only final correctly |
| Weak augmentations | Custom transform pipeline | `ContrastiveAugmentation(strong=False)` | Already configured with s=0.4 era-1 defaults |
| Config validation | Manual YAML parsing | `_StrictBase` + Pydantic | extra='forbid' catches typos automatically |
| Method registration | Manual dict updates | `register_method()` | Duplicate detection, sorted error messages |

## Common Pitfalls

### Pitfall 1: Z Recomputation Destabilizes Training
**What goes wrong:** Loss oscillates or diverges because Z changes every step.
**Why it happens:** Z is a normalization constant that should reflect the partition function. Recomputing on each batch makes it noisy.
**How to avoid:** Estimate Z on the first forward pass, store as `register_buffer`, never update again. Test explicitly that `Z` value does not change after initialization.
**Warning signs:** Loss oscillating wildly after epoch 1.

### Pitfall 2: Memory Bank Gradients Leak into Optimizer
**What goes wrong:** Bank parameters appear in optimizer, causing unexpected gradient updates or memory waste.
**Why it happens:** `BaseSSLModule.learnable_params` returns `self.parameters()` by default, which includes all submodule parameters.
**How to avoid:** Set `self.bank.weight.requires_grad = False` immediately after construction. Override `learnable_params` to explicitly yield only backbone + projector parameters.
**Warning signs:** Optimizer has more parameter groups than expected.

### Pitfall 3: Missing Sample Indices in DataLoader
**What goes wrong:** `training_step` receives `(views, labels)` but needs `(views, labels, indices)` for bank lookup/update.
**Why it happens:** Standard `ImageFolder` + `ssl_collate_fn` does not return indices.
**How to avoid:** Wrap dataset with `IndexedDataset` and use a collate function that also returns indices.
**Warning signs:** KeyError or IndexError when trying to access `batch[2]`.

### Pitfall 4: Bank Staleness Not Documented
**What goes wrong:** Success criterion 2 fails -- bank staleness must be documented in class docstring with MoCo cross-reference.
**Why it happens:** Easy to forget documentation in favor of code.
**How to avoid:** Include docstring as an explicit task deliverable. Bank features come from earlier encoder snapshots and become stale as the encoder trains. MoCo's queue (FIFO buffer fed by a slowly-moving momentum encoder) solves this.
**Warning signs:** PR review catches missing docstring.

### Pitfall 5: InvariantSpread Batch-Size Sensitivity Not Documented
**What goes wrong:** Success criterion 3 fails -- batch-size sensitivity must be documented.
**Why it happens:** In-batch negatives mean effective negative count = batch_size - 1. Below ~256, representation quality degrades significantly (unlike queue/bank methods with thousands of negatives).
**How to avoid:** Document in both module docstring and config YAML.
**Warning signs:** Poor accuracy at small batch sizes with no explanation.

### Pitfall 6: register_buffer vs Python Float for Z
**What goes wrong:** Z is lost when saving/loading checkpoints because it was stored as `self.Z = float(...)` instead of `register_buffer`.
**Why it happens:** Python attributes are not serialized by `state_dict()`.
**How to avoid:** Always use `self.register_buffer("Z", torch.tensor(...))`.
**Warning signs:** Training diverges after checkpoint resume.

## Code Examples

### MemoryBank Initialization (L2-normalized on hypersphere)
```python
# Source: PyTorch nn.Embedding docs + standard practice
import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryBank(nn.Module):
    """Non-parametric memory bank storing one L2-normalized feature per sample.

    Implemented as nn.Embedding with requires_grad=False. Features are updated
    via direct index assignment, not through backprop.

    **Staleness gotcha:** Features stored in the bank come from earlier encoder
    snapshots. As the encoder trains, stored features become stale -- they no
    longer match what the current encoder would produce. This means negative
    samples drawn from the bank are "softer" negatives than fresh encoder
    outputs would be. MoCo (He et al., CVPR 2020) addresses this with a FIFO
    queue fed by a slowly-moving momentum encoder, ensuring all negatives come
    from recent (though not current) encoder states.

    Args:
        n_samples: Total number of training samples (bank size).
        dim: Feature dimension.
    """

    def __init__(self, n_samples: int, dim: int):
        super().__init__()
        self.n_samples = n_samples
        self.dim = dim
        self.bank = nn.Embedding(n_samples, dim)
        # Initialize uniform on hypersphere
        nn.init.normal_(self.bank.weight.data)
        self.bank.weight.data = F.normalize(self.bank.weight.data, dim=1)
        self.bank.weight.requires_grad = False

    def get(self, indices: torch.Tensor) -> torch.Tensor:
        return self.bank(indices)

    @torch.no_grad()
    def update(self, indices: torch.Tensor, features: torch.Tensor) -> None:
        features = F.normalize(features.detach(), dim=1)
        self.bank.weight.data[indices] = features
```

### register_buffer for Z
```python
# Source: PyTorch Module.register_buffer docs
class NCELossWithFixedZ(nn.Module):
    def __init__(self, temperature=0.07, n_negatives=4096, eps=1e-7):
        super().__init__()
        self.temperature = temperature
        self.n_negatives = n_negatives
        self.eps = eps
        self.register_buffer("Z", torch.tensor(-1.0))
        self.register_buffer("z_initialized", torch.tensor(False))
```

### IndexedDataset Wrapper
```python
# Source: standard PyTorch pattern for returning indices
class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return (*data, idx)
```

### YAML Config Example (Instance Discrimination)
```yaml
method: instance_discrimination
backbone: resnet18
pretrained: false
max_epochs: 200
warmup_epochs: 10
batch_size: 128
lr: 0.03
weight_decay: 1e-4
optimizer: sgd
n_views: 1

instance_discrimination:
  temperature: 0.07
  n_negatives: 4096
  projection_dim: 128
```

### YAML Config Example (Invariant Spread)
```yaml
method: invariant_spread
backbone: resnet18
pretrained: false
max_epochs: 200
warmup_epochs: 10
batch_size: 256
lr: 0.03
weight_decay: 1e-4
optimizer: sgd
n_views: 2

invariant_spread:
  temperature: 0.07
  projection_dim: 128
```

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (standard, already in use) |
| Config file | None (default pytest discovery) |
| Quick run command | `python -m pytest tests/ -x -q` |
| Full suite command | `python -m pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| INFRA-02 | MemoryBank get/update by index | unit | `python -m pytest tests/test_memory_bank.py -x` | Wave 0 |
| INFRA-02 | MemoryBank L2-normalized init | unit | `python -m pytest tests/test_memory_bank.py::test_init_l2_normalized -x` | Wave 0 |
| INFRA-02 | MemoryBank staleness docstring | manual | grep docstring for "stale" and "MoCo" | manual |
| ERA1-01 | NCE loss finite positive scalar | unit | `python -m pytest tests/test_nce_loss.py -x` | Wave 0 |
| ERA1-01 | Z fixed after first call | unit | `python -m pytest tests/test_nce_loss.py::test_z_fixed_after_first_call -x` | Wave 0 |
| ERA1-01 | Z survives as register_buffer | unit | `python -m pytest tests/test_nce_loss.py::test_z_is_register_buffer -x` | Wave 0 |
| ERA1-01 | InstanceDiscrimination trains 5 epochs no divergence | integration | `python -m pytest tests/test_instance_discrimination.py::test_train_5_epochs -x` | Wave 0 |
| ERA1-01 | InstanceDiscrimination registered in dispatcher | unit | `python -m pytest tests/test_instance_discrimination.py::test_dispatcher_registration -x` | Wave 0 |
| ERA1-02 | InvariantSpread trains 5 epochs loss decreases | integration | `python -m pytest tests/test_invariant_spread.py::test_train_5_epochs -x` | Wave 0 |
| ERA1-02 | InvariantSpread registered in dispatcher | unit | `python -m pytest tests/test_invariant_spread.py::test_dispatcher_registration -x` | Wave 0 |
| ERA1-01/02 | Both methods selectable via YAML config key | unit | `python -m pytest tests/test_instance_discrimination.py tests/test_invariant_spread.py -k dispatcher -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/ -x -q`
- **Per wave merge:** `python -m pytest tests/ -v`
- **Phase gate:** Full suite green (all 70 existing + new tests) before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_memory_bank.py` -- covers INFRA-02 (init, get, update, L2-norm, staleness doc)
- [ ] `tests/test_nce_loss.py` -- covers ERA1-01 NCE loss (Z fixed, finite, gradients)
- [ ] `tests/test_instance_discrimination.py` -- covers ERA1-01 module (5-epoch train, dispatcher)
- [ ] `tests/test_invariant_spread.py` -- covers ERA1-02 module (5-epoch train, dispatcher, loss decrease)

### Test Patterns from Phase 1
The existing test suite (70 tests) establishes these patterns:
1. **Unit tests:** Random tensors, assert shape/finiteness/gradients (see `test_losses.py`)
2. **Integration tests:** `DummySSLModule` pattern -- minimal concrete subclass, `L.Trainer` with `max_epochs=1, accelerator="cpu", enable_checkpointing=False, logger=False, enable_progress_bar=False`
3. **Fixtures:** `random_tensor` factory, `tmp_imagefolder` (3 classes x 5 images at 32x32), `toy_config_dict`
4. **Dispatcher tests:** `clean_registry` autouse fixture saves/restores `_METHOD_REGISTRY`
5. **Config tests:** `TrainConfig(**minimal_dict)` with only required fields

### Smoke Test Pattern for 5-Epoch Training
```python
def test_instance_discrimination_trains_5_epochs(tmp_imagefolder):
    cfg = TrainConfig(
        method="instance_discrimination",
        backbone="resnet18",
        max_epochs=5,
        warmup_epochs=0,
        batch_size=4,
        lr=0.03,
        optimizer="sgd",
        n_views=1,
        data_dir=str(tmp_imagefolder),
        instance_discrimination=InstanceDiscriminationConfig(),
    )
    # Need IndexedDataset wrapping for indices
    model = InstanceDiscriminationModule(cfg)
    # ... setup DataLoader with IndexedDataset ...
    trainer = L.Trainer(
        max_epochs=5,
        accelerator="cpu",
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, dataloader)
    # Assert loss is finite (not NaN/Inf)
```

## Open Questions

1. **n_views for Instance Discrimination**
   - What we know: ERA1-01 says "one augmented view per image" meaning the encoder sees one augmented view, and the bank provides the comparison target.
   - What's unclear: Should `n_views=1` in config? The `MultiViewTransform` with `n_views=1` produces a single view list. The collate function would produce `views` of shape `[1, B, C, H, W]`.
   - Recommendation: Use `n_views=1`. The module extracts `views[0]` as the single augmented view. The memory bank provides the "second view" (stored features). This matches the paper's design where only one view is encoded per step.

2. **IndexedDataset Integration Strategy**
   - What we know: Instance Discrimination needs sample indices. Current SSLDataModule does not return them.
   - What's unclear: Should we modify SSLDataModule globally, or handle it in the method's training script?
   - Recommendation: Create an `IndexedDataset` wrapper and a matching collate function. The `InstanceDiscriminationModule` can document that it expects `(views, labels, indices)` batches. A helper function or a small modification to SSLDataModule can be provided. Do NOT modify the shared `ssl_collate_fn` -- add a new `ssl_collate_with_index` alongside it.

3. **MemoryBank n_samples at construction time**
   - What we know: MemoryBank needs `n_samples` (total dataset size) at construction.
   - What's unclear: Where does n_samples come from before the datamodule is set up?
   - Recommendation: Pass it as a config parameter or compute it in `setup()` after the datamodule is prepared. For CIFAR-10, n_samples=50000. The config could include a `bank_size` field (defaulting to dataset size), or the module can query `len(trainer.datamodule.train_dataset)` during `setup()`.

## Sources

### Primary (HIGH confidence)
- `core/base.py` -- BaseSSLModule interface (learnable_params, configure_optimizers, on_train_batch_end)
- `core/losses.py` -- InfoNCELoss (symmetric + asymmetric modes, internal L2-norm)
- `core/config.py` -- _StrictBase, TrainConfig, existing sub-config pattern
- `core/dispatcher.py` -- register_method(), method_dispatcher(), _METHOD_REGISTRY
- `core/data.py` -- SSLDataModule, ssl_collate_fn, MultiViewTransform, ContrastiveAugmentation
- `core/projection.py` -- ProjectionHead (2-layer MLP with BN)
- `tests/conftest.py` -- Shared fixtures (random_tensor, tmp_imagefolder, toy_config_dict)
- `tests/test_dispatcher.py` -- clean_registry fixture pattern for isolated registry tests
- PyTorch nn.Embedding docs -- index-based lookup and weight assignment
- PyTorch Module.register_buffer docs -- persistent buffer for Z

### Secondary (MEDIUM confidence)
- Wu et al., "Unsupervised Feature Learning via Non-Parametric Instance Discrimination" (CVPR 2018) -- NCE loss formulation, Z estimation, tau=0.07
- Ye et al., "Unsupervised Embedding Learning via Invariant and Spreading Instance Feature" (CVPR 2019) -- in-batch cross-entropy, batch-size sensitivity

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all dependencies already installed and tested in Phase 1
- Architecture: HIGH -- patterns directly observed from existing codebase
- Pitfalls: HIGH -- derived from REQUIREMENTS.md explicit gotchas and codebase analysis
- NCE math: MEDIUM -- based on paper descriptions and training data knowledge; exact implementation details should be verified against reference implementations during coding

**Research date:** 2026-04-01
**Valid until:** 2026-05-01 (stable domain, no fast-moving dependencies)
