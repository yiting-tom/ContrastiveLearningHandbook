# Phase 4: MoCo - Research

**Researched:** 2026-04-05
**Domain:** Momentum Contrast (MoCo v1/v2) — queue-based contrastive learning with momentum encoder
**Confidence:** HIGH

## Summary

Phase 4 implements MoCo v1 and MoCo v2, the queue-based contrastive learning methods that solve the memory bank staleness problem from Phase 2 (Instance Discrimination). The core new component is `MomentumQueue` (FIFO buffer in `core/queue.py`), which stores recent momentum-encoded keys for use as negatives. MoCo v1 uses a bare `nn.Linear` projection; MoCo v2 subclasses v1 and swaps in a 2-layer MLP projection head — mirroring the SimCLR v1/v2 subclass pattern already established in Phase 3.

The foundation codebase provides all required building blocks: `InfoNCELoss` with `_asymmetric_loss` (queue mode), `EMAUpdater` with flat schedule support (`base_momentum == end_momentum`), `BaseSSLModule` with EMA hooks (`on_train_batch_end`), `MoCoConfig` in the config schema, and `ProjectionHead` for the v2 MLP head. The implementation is primarily assembly of existing components plus the new `MomentumQueue` class.

**Primary recommendation:** Follow the SimCLR v1/v2 subclass pattern exactly — `MoCoV2Module(MoCoV1Module)` overriding only `build_projector()`. The `MomentumQueue` should use `register_buffer` for the queue tensor and pointer so they survive checkpointing. Keys must be detached before the loss; queue updated after loss computation.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** `MoCoV1Module` uses `nn.Linear(feat_dim, 128)` — bare linear, no BN, matching original paper. Do NOT use `ProjectionHead(num_layers=1)`.
- **D-02:** `MoCoV2Module(MoCoV1Module)` subclasses v1, overrides `build_projector()` to return `ProjectionHead(feat_dim, 2048, 128, num_layers=2)`. Docstring documents "5-line diff from v1".
- **D-03:** `MomentumQueue` lives in `core/queue.py` — separate from `core/memory_bank.py`.
- **D-04:** Use `EMAUpdater(base_momentum=0.999, end_momentum=0.999, total_steps=...)` for fixed-momentum schedule. Do NOT inline manual EMA update.
- **D-05:** Shuffled-BN documented in class docstring only — no code placeholder or stub. Single-GPU tutorial scope.
- **D-06:** Both modules in `methods/moco/module.py`. Registration in `methods/moco/__init__.py` as `moco_v1` and `moco_v2`. Top-level `methods/__init__.py` imports `moco` sub-package.
- **D-07:** Use `InfoNCELoss._asymmetric_loss` (queue mode) — pass queue via `queue` argument. No new loss class. Keys detached before loss; queue updated after loss computation.

### Claude's Discretion
- `MomentumQueue` initialization: `torch.randn` (L2-normalized) as specified in plan 04-01
- Queue pointer wrap-around implementation details
- `assert_no_ema_in_optimizer` utility placement (test helper in `tests/` or inline assertion in module)
- YAML config defaults (AdamW vs SGD, default LR, scheduler choice for v2 cosine)
- Exact wording of m=0.9 vs m=0.999 sensitivity gotcha in docstring

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within Phase 4 scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ERA2-01 | MoCo v1: query encoder + momentum encoder (EMA copy), FIFO queue K=65536, InfoNCE loss, EMA momentum m=0.999, shuffled-BN documented as limitation | `MoCoV1Module` using `BaseSSLModule` EMA hooks, `MomentumQueue(65536, dim)`, `InfoNCELoss(temperature=0.07, queue=...)`, `EMAUpdater(0.999, 0.999, total_steps)` |
| ERA2-02 | MoCo v2: MoCo v1 + 2-layer MLP head + Gaussian blur + cosine LR schedule | `MoCoV2Module(MoCoV1Module)` overriding `build_projector()` to use `ProjectionHead(feat_dim, 2048, 128, num_layers=2)`; cosine LR already provided by `BaseSSLModule.configure_optimizers()` |
| INFRA-03 | `MomentumQueue(queue_size, dim)` FIFO buffer with `get_negatives()` and `update(keys)` interface | New `core/queue.py` with `register_buffer` for queue tensor and pointer; L2-normalized `torch.randn` initialization |
</phase_requirements>

## Standard Stack

### Core (already installed)
| Library | Purpose | Why Standard |
|---------|---------|--------------|
| `torch` | Queue buffer, EMA operations, tensor ops | Foundation dependency |
| `lightning` | Training loop, `on_train_batch_end` hook | Base class provides EMA hook |
| `timm` | Backbone factory (`build_backbone`) | Already wired |

### Existing Components (from Phase 1)
| Component | File | Purpose for MoCo |
|-----------|------|-----------------|
| `InfoNCELoss` | `core/losses.py` | `_asymmetric_loss(z_i, z_j, queue)` — queue shape `[D, K]` |
| `EMAUpdater` | `core/ema.py` | `EMAUpdater(0.999, 0.999, total_steps)` — flat schedule |
| `BaseSSLModule` | `core/base.py` | `on_train_batch_end` EMA hook, `configure_optimizers` |
| `ProjectionHead` | `core/projection.py` | MoCo v2 2-layer MLP head |
| `MoCoConfig` | `core/config.py` | `temperature=0.07`, `queue_size=65536`, `momentum=0.999` |
| `register_method` | `core/dispatcher.py` | Registry pattern |

**No new pip dependencies required.** All standard library and existing project dependencies.

## Architecture Patterns

### Recommended Project Structure
```
core/
  queue.py                  # NEW: MomentumQueue (FIFO buffer)
methods/
  moco/
    __init__.py             # NEW: register moco_v1, moco_v2
    module.py               # NEW: MoCoV1Module, MoCoV2Module
methods/__init__.py         # MODIFY: add `import methods.moco`
configs/
  moco_v1_resnet18.yaml    # NEW
  moco_v2_resnet18.yaml    # NEW
tests/
  test_queue.py             # NEW: MomentumQueue unit tests
  test_moco.py              # NEW: MoCo v1/v2 integration tests
```

### Pattern 1: MomentumQueue as register_buffer
**What:** FIFO queue stored as a `register_buffer` (not a `nn.Parameter`) so it persists in checkpoints but has no gradients.
**When to use:** Any fixed-size buffer that must survive save/load but is not learnable.
**Example:**
```python
class MomentumQueue(nn.Module):
    def __init__(self, queue_size: int, dim: int) -> None:
        super().__init__()
        self.queue_size = queue_size
        self.dim = dim
        # L2-normalized random init
        queue = torch.randn(dim, queue_size)
        queue = F.normalize(queue, dim=0)
        self.register_buffer("queue", queue)
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update(self, keys: torch.Tensor) -> None:
        """Enqueue new keys (FIFO). keys shape: [B, D]."""
        keys = keys.T  # [D, B]
        batch_size = keys.shape[1]
        ptr = int(self.ptr)
        # Handle wrap-around
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys
        else:
            overflow = (ptr + batch_size) - self.queue_size
            self.queue[:, ptr:] = keys[:, :batch_size - overflow]
            self.queue[:, :overflow] = keys[:, batch_size - overflow:]
        self.ptr[0] = (ptr + batch_size) % self.queue_size

    def get_negatives(self) -> torch.Tensor:
        """Return current queue contents detached. Shape: [D, K]."""
        return self.queue.clone().detach()
```

### Pattern 2: Momentum Encoder Setup (copy.deepcopy + deactivate_requires_grad)
**What:** Deep copy backbone, then disable all gradients on the copy.
**When to use:** Any method with a momentum/teacher encoder (MoCo, BYOL, DINO).
**Example:**
```python
import copy

# In __init__:
self.backbone_ema = copy.deepcopy(self.backbone)
self.backbone_ema.requires_grad_(False)

# Also deepcopy the projection head for momentum path
self.projector_ema = copy.deepcopy(self.projector)  # v1: nn.Linear copy
self.projector_ema.requires_grad_(False)

# Wire EMA updater (flat schedule for MoCo)
moco_cfg = cfg.moco or MoCoConfig()
self.ema_updater = EMAUpdater(
    base_momentum=moco_cfg.momentum,
    end_momentum=moco_cfg.momentum,
    total_steps=1,  # placeholder — updated later or irrelevant for flat
)
self._online_params = list(self.backbone.parameters()) + list(self.projector.parameters())
self._target_params = list(self.backbone_ema.parameters()) + list(self.projector_ema.parameters())
```

### Pattern 3: MoCo v1/v2 Subclass Relationship
**What:** `MoCoV2Module(MoCoV1Module)` overrides only `build_projector()`.
**When to use:** Same pattern as `SimCLRv2Module(SimCLRv1Module)`.
**Example:**
```python
class MoCoV1Module(BaseSSLModule):
    def build_projector(self) -> nn.Module:
        return nn.Linear(self.feat_dim, 128)  # D-01: bare linear, no BN

class MoCoV2Module(MoCoV1Module):
    def build_projector(self) -> nn.Module:
        return ProjectionHead(self.feat_dim, 2048, 128, num_layers=2)  # D-02
```

### Pattern 4: learnable_params Override (Exclude EMA)
**What:** Override `learnable_params` property to exclude momentum encoder parameters.
**When to use:** Any method with EMA target network.
**Example:**
```python
@property
def learnable_params(self):
    return list(self.backbone.parameters()) + list(self.projector.parameters())
```

### Pattern 5: Training Step — Asymmetric Forward
**What:** Query goes through online encoder; key goes through momentum encoder. Key is detached. Queue updated after loss.
**Example:**
```python
def training_step(self, batch, batch_idx):
    views, _ = batch  # views: [2, B, C, H, W]
    # Query path (online encoder)
    q = self.projector(self.backbone(views[0]))
    # Key path (momentum encoder, no grad)
    with torch.no_grad():
        k = self.projector_ema(self.backbone_ema(views[1]))
        k = k.detach()
    # Loss with queue negatives
    loss = self.loss_fn(q, k, queue=self.momentum_queue.get_negatives())
    # Update queue AFTER loss computation (D-07)
    self.momentum_queue.update(k)
    self.log_train_metrics(loss)
    return loss
```

### Anti-Patterns to Avoid
- **Updating queue before loss computation:** Queue must be updated AFTER the loss is computed — otherwise positive keys would appear as negatives in the same batch.
- **EMA update in training_step:** Must use `on_train_batch_end` (already wired in `BaseSSLModule`). Doing EMA in `training_step` means gradients may be corrupted.
- **Including EMA params in optimizer:** Use `requires_grad_(False)` immediately after `deepcopy` AND override `learnable_params`.
- **Using `ProjectionHead` for v1:** D-01 explicitly forbids this — v1 uses bare `nn.Linear(feat_dim, 128)` with no BN to match the paper exactly.
- **Inline EMA update:** D-04 mandates using the shared `EMAUpdater` class, not a manual loop.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| EMA momentum update | Manual parameter loop | `EMAUpdater` from `core/ema.py` | Cosine schedule support, consistency across methods |
| Contrastive loss with queue | Custom loss function | `InfoNCELoss(queue=...)` from `core/losses.py` | Already tested, handles normalization internally |
| Warmup-cosine scheduler | Manual LR scheduling | `BaseSSLModule.configure_optimizers()` | Step-based interval already wired |
| MLP projection head | Manual Sequential layers | `ProjectionHead` from `core/projection.py` | BN+ReLU conventions handled correctly |
| Method registration | Manual registry dict | `register_method()` from `core/dispatcher.py` | Duplicate-name guard built in |

## Common Pitfalls

### Pitfall 1: Queue Update Timing
**What goes wrong:** Updating the queue before computing the loss causes the current batch's positive keys to appear as negatives.
**Why it happens:** Intuitive to "store results then use them" — but in MoCo the queue provides negatives for the current batch.
**How to avoid:** Always compute loss FIRST, then call `self.momentum_queue.update(k)`.
**Warning signs:** Loss is abnormally high or training diverges from the start.

### Pitfall 2: Momentum Encoder Gradients Leaking
**What goes wrong:** EMA parameters appear in optimizer, causing them to be updated by both EMA and gradient descent simultaneously.
**Why it happens:** `self.parameters()` includes all submodules — if `requires_grad_(False)` is not called, they appear in param groups.
**How to avoid:** Call `requires_grad_(False)` immediately after `deepcopy`. Override `learnable_params` to return only online network params.
**Warning signs:** Test `assert_no_ema_in_optimizer` fails; training instability after a few epochs.

### Pitfall 3: EMA Momentum Sensitivity (m=0.9 vs m=0.999)
**What goes wrong:** Using m=0.9 (which seems reasonable) produces significantly worse representations than m=0.999.
**Why it happens:** Lower momentum means the target encoder diverges faster from the online encoder, reducing key consistency in the queue. The queue stores keys from many steps — they need to be consistent.
**How to avoid:** Default to m=0.999 in `MoCoConfig`. Document the sensitivity in the class docstring.
**Warning signs:** Accuracy drops ~10% from expected baselines.

### Pitfall 4: Queue Pointer Wrap-Around Bug
**What goes wrong:** When `ptr + batch_size > queue_size`, naive slicing causes index-out-of-bounds or silently drops keys.
**Why it happens:** Batch size may not evenly divide queue size.
**How to avoid:** Implement explicit wrap-around logic that handles partial writes at both ends of the buffer.
**Warning signs:** Queue contains stale uninitialized values; unit tests for wrap-around catch this.

### Pitfall 5: Forgetting to Detach Keys
**What goes wrong:** Not detaching momentum encoder outputs before passing to the loss allows gradient flow through the target network.
**Why it happens:** The momentum encoder forward pass happens in the same computation graph.
**How to avoid:** Use `torch.no_grad()` context manager around the momentum encoder forward AND explicitly `.detach()` the output.
**Warning signs:** Training is slower than expected (unnecessary gradient computation); EMA params show non-zero gradients.

### Pitfall 6: BN Leakage (Shuffled BN)
**What goes wrong:** In multi-GPU settings, BatchNorm in the momentum encoder leaks information about positive pairs through batch statistics.
**Why it happens:** Both views go through the same BN layer, sharing running mean/variance.
**How to avoid:** In production multi-GPU: shuffle samples across GPUs before momentum encoding. In this tutorial: document as a known limitation (single-GPU only).
**Warning signs:** Unrealistically high accuracy in multi-GPU training.

## Code Examples

### MomentumQueue Complete Implementation
```python
# core/queue.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MomentumQueue(nn.Module):
    """FIFO queue for MoCo negative keys.

    Stores L2-normalized feature vectors in a fixed-size circular buffer.
    Used by MoCo v1 and MoCo v2.

    Args:
        queue_size: Maximum number of keys in the queue (K).
        dim: Feature dimensionality.
    """

    def __init__(self, queue_size: int, dim: int) -> None:
        super().__init__()
        self.queue_size = queue_size
        self.dim = dim
        queue = torch.randn(dim, queue_size)
        queue = F.normalize(queue, dim=0)
        self.register_buffer("queue", queue)
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update(self, keys: torch.Tensor) -> None:
        """Enqueue new keys FIFO. keys: [B, D], L2-normalized."""
        keys = F.normalize(keys.detach(), dim=1)
        batch_size = keys.shape[0]
        ptr = int(self.ptr)

        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T

        self.ptr[0] = (ptr + batch_size) % self.queue_size

    def get_negatives(self) -> torch.Tensor:
        """Return queue contents detached. Shape: [D, K]."""
        return self.queue.clone().detach()
```

### MoCoV1Module Skeleton
```python
# methods/moco/module.py
import copy
import torch
import torch.nn as nn

from core.backbone import build_backbone
from core.base import BaseSSLModule
from core.config import MoCoConfig, TrainConfig
from core.ema import EMAUpdater
from core.losses import InfoNCELoss
from core.queue import MomentumQueue


class MoCoV1Module(BaseSSLModule):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        moco_cfg = cfg.moco or MoCoConfig()

        self.backbone, self.feat_dim = build_backbone(cfg.backbone, cfg.pretrained)
        self.projector = self.build_projector()

        # Momentum encoder
        self.backbone_ema = copy.deepcopy(self.backbone)
        self.backbone_ema.requires_grad_(False)
        self.projector_ema = copy.deepcopy(self.projector)
        self.projector_ema.requires_grad_(False)

        # Queue
        self.momentum_queue = MomentumQueue(moco_cfg.queue_size, 128)

        # Loss
        self.loss_fn = InfoNCELoss(temperature=moco_cfg.temperature)

        # EMA (flat schedule: base == end)
        self.ema_updater = EMAUpdater(
            base_momentum=moco_cfg.momentum,
            end_momentum=moco_cfg.momentum,
            total_steps=1,
        )
        self._online_params = (
            list(self.backbone.parameters()) + list(self.projector.parameters())
        )
        self._target_params = (
            list(self.backbone_ema.parameters()) + list(self.projector_ema.parameters())
        )

    def build_projector(self) -> nn.Module:
        return nn.Linear(self.feat_dim, 128)  # D-01: bare linear, no BN

    @property
    def learnable_params(self):
        return list(self.backbone.parameters()) + list(self.projector.parameters())

    def training_step(self, batch, batch_idx):
        views, _ = batch
        q = self.projector(self.backbone(views[0]))
        with torch.no_grad():
            k = self.projector_ema(self.backbone_ema(views[1]))
            k = k.detach()
        loss = self.loss_fn(q, k, queue=self.momentum_queue.get_negatives())
        self.momentum_queue.update(k)
        self.log_train_metrics(loss)
        return loss
```

### Registration Pattern
```python
# methods/moco/__init__.py
from core.dispatcher import register_method
from methods.moco.module import MoCoV1Module, MoCoV2Module

register_method("moco_v1", MoCoV1Module)
register_method("moco_v2", MoCoV2Module)
```

### YAML Config Example
```yaml
# configs/moco_v1_resnet18.yaml
method: moco_v1
backbone: resnet18
pretrained: false
max_epochs: 200
warmup_epochs: 10
batch_size: 256
lr: 0.03
weight_decay: 1e-4
optimizer: sgd
moco:
  temperature: 0.07
  queue_size: 65536
  momentum: 0.999
```

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (via pyproject.toml) |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `python -m pytest tests/test_queue.py tests/test_moco.py -x -q` |
| Full suite command | `python -m pytest -x -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| INFRA-03 | MomentumQueue FIFO with pointer wrap-around | unit | `python -m pytest tests/test_queue.py -x` | No — Wave 0 |
| INFRA-03 | Queue size invariant after filling | unit | `python -m pytest tests/test_queue.py::test_queue_size_invariant -x` | No — Wave 0 |
| ERA2-01 | EMA params not in optimizer | unit | `python -m pytest tests/test_moco.py::test_ema_params_not_in_optimizer -x` | No — Wave 0 |
| ERA2-01 | EMA update in on_train_batch_end | unit (mock) | `python -m pytest tests/test_moco.py::test_ema_update_call_order -x` | No — Wave 0 |
| ERA2-01 | MoCo v1 trains 5 epochs without divergence | integration | `python -m pytest tests/test_moco.py::test_moco_v1_train_5_epochs -x` | No — Wave 0 |
| ERA2-02 | MoCo v2 trains 5 epochs without divergence | integration | `python -m pytest tests/test_moco.py::test_moco_v2_train_5_epochs -x` | No — Wave 0 |
| ERA2-02 | v2 projector is 2-layer MLP (not bare linear) | unit | `python -m pytest tests/test_moco.py::test_moco_v2_has_mlp_head -x` | No — Wave 0 |
| ERA2-01/02 | Dispatcher registration for moco_v1 and moco_v2 | unit | `python -m pytest tests/test_moco.py::test_dispatcher_registration -x` | No — Wave 0 |
| DOC-02 | Docstrings meet DOC-02 standard | unit | `python -m pytest tests/test_moco.py::test_docstring -x` | No — Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_queue.py tests/test_moco.py -x -q`
- **Per wave merge:** `python -m pytest -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_queue.py` -- covers INFRA-03 (queue FIFO, wrap-around, size invariant, get_negatives detached)
- [ ] `tests/test_moco.py` -- covers ERA2-01, ERA2-02 (EMA exclusion, training smoke, v2 MLP head, dispatcher, docstrings)

## Open Questions

1. **MoCo v2 Gaussian blur augmentation scope**
   - What we know: D-02 and ERA2-02 specify Gaussian blur as a v2 addition
   - What's unclear: Whether to create a separate augmentation config for v2 or rely on `ContrastiveAugmentation(strong=True)` which already includes Gaussian blur (p=0.5)
   - Recommendation: `ContrastiveAugmentation(strong=True)` already provides Gaussian blur. Document in v2 YAML config comments that strong augmentation includes the blur required by MoCo v2. No new augmentation code needed.

2. **EMAUpdater total_steps for flat schedule**
   - What we know: `base_momentum == end_momentum == 0.999` makes the cosine schedule irrelevant (m is constant)
   - What's unclear: What to pass for `total_steps` when the schedule is flat
   - Recommendation: Pass `total_steps=1` (any positive value works since base==end). The momentum is constant regardless.

3. **assert_no_ema_in_optimizer placement**
   - What we know: This utility verifies EMA params are disjoint from optimizer param groups
   - What's unclear: Whether it lives in `tests/` as a test helper or in `core/` as a runtime assertion
   - Recommendation: Implement as a test helper function in `tests/test_moco.py` — it is a verification tool, not a runtime guard. The `requires_grad_(False)` call is the actual guard.

## Sources

### Primary (HIGH confidence)
- Existing codebase: `core/losses.py` lines 87-108 — `_asymmetric_loss` with queue shape `[D, K]`
- Existing codebase: `core/ema.py` — `EMAUpdater` with cosine schedule, flat when `base==end`
- Existing codebase: `core/base.py` lines 184-197 — `on_train_batch_end` EMA hook
- Existing codebase: `core/config.py` lines 43-49 — `MoCoConfig(temperature=0.07, queue_size=65536, momentum=0.999)`
- Existing codebase: `methods/simclr/module.py` — v1/v2 subclass pattern reference
- Existing codebase: `core/memory_bank.py` — `nn.Embedding` buffer pattern reference (queue uses `register_buffer` instead)

### Secondary (HIGH confidence)
- CONTEXT.md D-01 through D-07 — locked implementation decisions
- REQUIREMENTS.md ERA2-01, ERA2-02, INFRA-03 — method specifications

### MoCo Papers (HIGH confidence)
- He et al., "Momentum Contrast for Unsupervised Visual Representation Learning" (CVPR 2020) — MoCo v1 https://arxiv.org/abs/1911.05722
- Chen et al., "Improved Baselines with Momentum Contrastive Learning" (2020) — MoCo v2 https://arxiv.org/abs/2003.04297

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all components already exist in codebase; no new dependencies
- Architecture: HIGH - follows established SimCLR v1/v2 subclass pattern exactly
- Pitfalls: HIGH - well-documented in MoCo papers and CONTEXT.md decisions

**Research date:** 2026-04-05
**Valid until:** 2026-05-05 (stable — no external dependency changes expected)
