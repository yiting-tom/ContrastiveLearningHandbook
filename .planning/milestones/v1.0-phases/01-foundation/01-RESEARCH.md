# Phase 1: Foundation - Research

**Researched:** 2026-03-31
**Domain:** PyTorch Lightning + timm + Pydantic SSL training infrastructure
**Confidence:** HIGH

## Summary

Phase 1 builds the shared infrastructure for a contrastive learning tutorial repo: a `BaseSSLModule` Lightning base class, Pydantic v2 config schema, timm backbone factory, projection head MLP, augmentation pipeline, data module, EMA updater, InfoNCE loss, LARS optimizer, and method dispatcher. This is a greenfield project -- no existing code.

The standard stack is well-established: PyTorch 2.x, Lightning 2.x, timm 1.x, Pydantic v2, torchvision transforms v2. All components have clear, battle-tested patterns from the SSL literature and well-documented APIs. The primary risk is subtle API misuse (e.g., wrong scheduler interval, incorrect EMA hook placement, missing `extra='forbid'` propagation in nested Pydantic models).

**Primary recommendation:** Use native PyTorch schedulers (`LinearLR` + `CosineAnnealingLR` via `SequentialLR`) instead of `pl_bolts` for warmup-cosine scheduling -- zero extra dependencies, stable API. Implement LARS from scratch per user decision D-04. Use `torchvision.transforms.v2` for augmentations.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Top-level `core/` package for all shared foundation code, `methods/` directory where each SSL method gets its own package (e.g., `methods/simclr/`, `methods/moco/`).
- **D-02:** `core/` contains: `base.py` (BaseSSLModule), `backbone.py` (build_backbone), `losses.py` (InfoNCELoss), `projection.py` (ProjectionHead), `data.py` (SSLDataModule + ContrastiveAugmentation), `ema.py` (EMAUpdater), `dispatcher.py` (method_dispatcher), `config.py` (TrainConfig/EvalConfig), `optimizers.py` (LARS).
- **D-03:** Top-level layout: `core/`, `methods/`, `tests/`, `configs/`. No wrapping package -- `core` and `methods` are importable directly.
- **D-04:** Implement LARS from scratch in `core/optimizers.py` (~50-60 lines of pure PyTorch). No dependency on `lightly`. Class docstring includes paper reference (https://arxiv.org/abs/1708.03888). This is a tutorial repo -- the implementation should be readable.
- **D-05:** `LARS.__init__` signature: `(params, lr, momentum=0.9, weight_decay=1e-6, eta=0.001, exclude_bias_and_norm=True)`.
- **D-06:** Unit tests (backbone, projection head, losses, EMA, config, dispatcher) use random tensors -- no I/O, fast, offline-safe.
- **D-07:** `SSLDataModule` and `ContrastiveAugmentation` tests use a temporary synthetic `ImageFolder` (a few dummy `.jpg` images in per-class subdirectories, created by the test fixture and cleaned up after).
- **D-08:** `TrainConfig` (and all sub-configs) use `model_config = ConfigDict(extra='forbid')`. Unknown YAML keys raise a `ValidationError` immediately.

### Claude's Discretion
- Exact warmup-cosine scheduler implementation (linear warmup then cosine decay is standard)
- LARS `exclude_bias_and_norm` filtering logic (exclude 1-D parameter tensors)
- Synthetic ImageFolder fixture helper (shared pytest fixture or per-test)
- `__init__.py` exports for `core/` (re-export public API vs. leave explicit)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within Phase 1 scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FOUND-01 | `BaseSSLModule(LightningModule)` abstract base with `build_projector()`, `learnable_params`, `configure_optimizers()`, EMA hook, TensorBoard logging | Lightning 2.x `LightningModule` API, `SequentialLR` for warmup-cosine, `self.log()` for TensorBoard |
| FOUND-02 | Pydantic v2 `TrainConfig` with namespaced per-method sub-configs, YAML loading | Pydantic v2 `model_validate()`, `ConfigDict(extra='forbid')`, `yaml.safe_load()` |
| FOUND-03 | `build_backbone(model_name, pretrained=False)` via timm | `timm.create_model(..., num_classes=0)`, `backbone.num_features` |
| FOUND-04 | `ProjectionHead` reusable MLP with BN+ReLU intermediate, BN-only final | Standard `nn.Sequential` construction pattern |
| FOUND-05 | `ContrastiveAugmentation` with strong/weak paths using `torchvision.transforms.v2` | `v2.Compose`, `v2.ColorJitter`, `v2.GaussianBlur`, `v2.RandomResizedCrop` |
| FOUND-06 | `SSLDataModule(LightningDataModule)` wrapping ImageFolder with configurable `n_views` | Lightning `LightningDataModule`, custom collate or multi-view wrapper |
| FOUND-07 | `method_dispatcher(cfg) -> BaseSSLModule` factory with `ValueError` on unknown | Simple dict-based registry pattern |
| FOUND-08 | `EvalConfig` Pydantic sub-schema under `eval:` key | Nested Pydantic models with `Optional` fields |
| FOUND-09 | TensorBoard logging via `self.log()` | Lightning built-in TensorBoard logger, `self.log()` / `self.log_dict()` |
| FOUND-10 | `EMAUpdater` with cosine-scheduled momentum and `step()` method | `torch.no_grad()`, `param.data.mul_().add_()` pattern |
| INFRA-01 | `InfoNCELoss(temperature, reduction)` -- symmetric and asymmetric modes | Cross-entropy over cosine similarity matrix pattern |
| INFRA-06 | `LARS` optimizer from scratch | `torch.optim.Optimizer` subclass, per-layer trust ratio |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.11.0 | Deep learning framework | Foundation for everything |
| lightning | 2.6.1 | Training loop, callbacks, logging | Reduces boilerplate, handles device management |
| timm | 1.0.26 | Backbone model zoo | Largest vision model collection, clean `create_model` API |
| pydantic | 2.12.5 | Config validation | Type-safe YAML config with `model_validate()` |
| torchvision | 0.26.0 | Augmentation transforms | `transforms.v2` API for SSL augmentation pipelines |
| pyyaml | 6.0.3 | YAML parsing | Load config files via `yaml.safe_load()` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | 8.4.1 | Testing framework | All unit tests |
| Pillow | (bundled with torchvision) | Image I/O | Synthetic ImageFolder test fixtures |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Native `SequentialLR` | `pl_bolts.LinearWarmupCosineAnnealingLR` | pl_bolts is "under review", adds dependency; native PyTorch is stable and zero-dep |
| LARS from scratch | `torchlars` or `lightly` | User decision D-04 locks this: from-scratch for tutorial readability |
| `torchvision.transforms.v2` | `albumentations` | v2 is sufficient for SSL augmentations and avoids extra dependency |

**Installation:**
```bash
pip install torch==2.11.0 torchvision==0.26.0 lightning==2.6.1 timm==1.0.26 pydantic==2.12.5 pyyaml==6.0.3 pytest==8.4.1
```

**Version verification:** Versions verified against PyPI on 2026-03-31. The environment will need a dedicated setup (see Environment Availability section).

## Architecture Patterns

### Recommended Project Structure
```
core/
    __init__.py          # Re-export public API
    base.py              # BaseSSLModule(LightningModule)
    backbone.py          # build_backbone() factory
    config.py            # TrainConfig, EvalConfig, sub-configs
    data.py              # SSLDataModule, ContrastiveAugmentation
    dispatcher.py        # method_dispatcher()
    ema.py               # EMAUpdater
    losses.py            # InfoNCELoss
    optimizers.py        # LARS
    projection.py        # ProjectionHead
methods/
    __init__.py
tests/
    __init__.py
    test_backbone.py
    test_base.py
    test_config.py
    test_data.py
    test_dispatcher.py
    test_ema.py
    test_losses.py
    test_optimizers.py
    test_projection.py
configs/
    example.yaml         # Example config for testing/reference
```

### Pattern 1: Lightning configure_optimizers with Warmup-Cosine

**What:** Use native PyTorch `SequentialLR` combining `LinearLR` (warmup) and `CosineAnnealingLR` (decay).

**When to use:** Always -- this is the standard scheduler for SSL methods.

**Example:**
```python
# Source: PyTorch docs (torch.optim.lr_scheduler)
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

def configure_optimizers(self):
    optimizer = self._build_optimizer()  # AdamW/SGD/LARS dispatch

    warmup_steps = self.cfg.warmup_epochs * self.trainer.estimated_stepping_batches // self.cfg.max_epochs
    total_steps = self.trainer.estimated_stepping_batches

    warmup = LinearLR(optimizer, start_factor=1e-4, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
        },
    }
```

**Pitfall:** `self.trainer.estimated_stepping_batches` is only available after `trainer.fit()` is called. Access it inside `configure_optimizers()` -- Lightning makes it available at that point. If warmup is epoch-based, convert to steps.

### Pattern 2: timm Backbone Factory

**What:** Use `timm.create_model(..., num_classes=0)` to get a backbone with global pooling but no classifier head.

**Example:**
```python
# Source: timm docs (huggingface.co/docs/timm/feature_extraction)
import timm

def build_backbone(model_name: str, pretrained: bool = False) -> tuple:
    backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    feat_dim = backbone.num_features
    return backbone, feat_dim
```

**Key:** `num_classes=0` replaces the classifier with `nn.Identity` while keeping global pooling. `backbone.num_features` gives the correct feature dimension for any architecture (ResNet, ViT, etc.). Never use `backbone.inplanes` (ResNet-only) or hard-coded values.

### Pattern 3: Pydantic v2 Config with extra='forbid'

**What:** Nested Pydantic models with strict validation to catch YAML typos.

**Example:**
```python
# Source: Pydantic v2 docs (docs.pydantic.dev/latest/concepts/models/)
from pydantic import BaseModel, ConfigDict
from typing import Optional

class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

class SimCLRConfig(_StrictBase):
    temperature: float = 0.5
    projection_dim: int = 128

class TrainConfig(_StrictBase):
    method: str
    backbone: str = "resnet50"
    pretrained: bool = False
    max_epochs: int = 100
    warmup_epochs: int = 10
    batch_size: int = 256
    lr: float = 0.3
    weight_decay: float = 1e-6
    optimizer: str = "lars"  # "adamw" | "sgd" | "lars"

    # Per-method sub-configs (optional -- only the active method's config is used)
    simclr: Optional[SimCLRConfig] = None
    # ... other method configs
```

**Key:** `extra='forbid'` MUST be on every sub-config class, not just the root. Use a shared `_StrictBase` to avoid repetition.

### Pattern 4: Multi-View Data Loading

**What:** A wrapper that applies augmentation `n_views` times per image and stacks the results.

**Example:**
```python
# Standard SSL data loading pattern
class MultiViewTransform:
    def __init__(self, base_transform, n_views: int = 2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, img):
        return [self.base_transform(img) for _ in range(self.n_views)]

# In SSLDataModule, use a custom collate_fn to stack views:
# Output shape: [n_views, B, C, H, W]
def ssl_collate_fn(batch):
    views, labels = zip(*batch)
    # views is list of lists: [[v1, v2], [v1, v2], ...]
    n_views = len(views[0])
    stacked = [torch.stack([v[i] for v in views]) for i in range(n_views)]
    return torch.stack(stacked), torch.tensor(labels)
```

### Pattern 5: EMA Update in on_train_batch_end

**What:** Momentum update of target network parameters, scheduled with cosine ramp.

**Example:**
```python
# Standard EMA pattern for SSL (BYOL, MoCo, DINO)
import math
import torch

class EMAUpdater:
    def __init__(self, base_momentum: float, end_momentum: float, total_steps: int):
        self.base_momentum = base_momentum
        self.end_momentum = end_momentum
        self.total_steps = total_steps
        self._step = 0

    @property
    def current_momentum(self) -> float:
        # Cosine schedule from base_momentum to end_momentum
        t = self._step / max(self.total_steps, 1)
        return self.end_momentum - (self.end_momentum - self.base_momentum) * (
            math.cos(math.pi * t) + 1
        ) / 2

    @torch.no_grad()
    def step(self, online_params, target_params):
        m = self.current_momentum
        for p_online, p_target in zip(online_params, target_params):
            p_target.data.mul_(m).add_(p_online.data, alpha=1 - m)
        self._step += 1
```

**Key:** Always use `@torch.no_grad()` and operate on `.data` directly. Target params must have `requires_grad=False`.

### Anti-Patterns to Avoid
- **EMA in training_step:** Causes gradient leakage. Always use `on_train_batch_end`.
- **Hard-coded feature dims:** Use `backbone.num_features`, never `2048` or `768`.
- **Missing extra='forbid' on sub-configs:** Typos in nested YAML sections pass silently.
- **Using `pl_bolts` scheduler:** Marked "under review", may break. Use native PyTorch `SequentialLR`.
- **Forgetting `requires_grad_(False)` on target network:** Target params leak into optimizer, corrupting EMA.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Backbone models | Custom ResNet/ViT implementations | `timm.create_model()` | Hundreds of tested architectures with correct `num_features` |
| Image transforms | Manual PIL/numpy augmentations | `torchvision.transforms.v2` | GPU-accelerated, composable, handles all standard SSL augmentations |
| Training loop | Manual epoch/batch loops | Lightning `Trainer` | Handles distributed, mixed precision, logging, checkpointing |
| YAML parsing | Custom config parser | `yaml.safe_load()` + Pydantic `model_validate()` | Type validation, error messages, nested model support |
| Warmup-cosine schedule | Custom LR lambda function | `SequentialLR(LinearLR, CosineAnnealingLR)` | Native PyTorch, well-tested, handles edge cases |

**Key insight:** The LARS optimizer is the only component that must be hand-rolled (per user decision D-04), and it is deliberately chosen for pedagogical value. Everything else should use established libraries.

## Common Pitfalls

### Pitfall 1: SequentialLR Milestone Units
**What goes wrong:** `milestones` parameter expects step counts (not epoch counts) when `interval="step"`. Passing epoch counts causes warmup to last the wrong duration.
**Why it happens:** Confusion between epoch-based and step-based scheduling.
**How to avoid:** Compute `warmup_steps = warmup_epochs * steps_per_epoch` and pass that as the milestone. Use `self.trainer.estimated_stepping_batches` to get total steps.
**Warning signs:** Learning rate plot shows abrupt jump at wrong point.

### Pitfall 2: Pydantic extra='forbid' Not Propagating to Nested Models
**What goes wrong:** Only the root `TrainConfig` rejects unknown keys; nested sub-configs silently accept typos.
**Why it happens:** Each Pydantic model needs its own `model_config`. Child models don't inherit parent config.
**How to avoid:** Define a `_StrictBase(BaseModel)` with `model_config = ConfigDict(extra='forbid')` and have ALL config classes inherit from it.
**Warning signs:** Typos in YAML sub-sections don't raise errors.

### Pitfall 3: EMA Target Params in Optimizer
**What goes wrong:** Target network parameters appear in `learnable_params`, get updated by both optimizer and EMA, causing training instability.
**Why it happens:** Using `self.parameters()` instead of a curated `learnable_params` property.
**How to avoid:** `BaseSSLModule.learnable_params` must explicitly return only online network params. Call `requires_grad_(False)` on target network immediately after creation.
**Warning signs:** Target network gradients are non-None; optimizer has more param groups than expected.

### Pitfall 4: torchvision.transforms.v2 Import Path
**What goes wrong:** Using `from torchvision import transforms` gives v1 API. Some v2 transforms have different signatures.
**Why it happens:** Old tutorials use v1 imports.
**How to avoid:** Always use `from torchvision.transforms import v2` or `from torchvision.transforms.v2 import ...`.
**Warning signs:** Missing `v2`-specific features, deprecation warnings.

### Pitfall 5: ColorJitter Strength for SSL
**What goes wrong:** Using default `ColorJitter` parameters (weak) instead of s=1.0 strength. SimCLR performance degrades significantly.
**Why it happens:** Default torchvision jitter is tuned for supervised learning.
**How to avoid:** For strong augmentation: `v2.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)` where `s=1.0`. This matches the SimCLR paper.
**Warning signs:** Representations cluster poorly; downstream accuracy is low.

### Pitfall 6: GaussianBlur Kernel Size Must Be Odd
**What goes wrong:** Even kernel sizes cause errors or incorrect blurring.
**Why it happens:** Gaussian convolution requires odd kernel dimensions for symmetric padding.
**How to avoid:** Use `kernel_size=23` (SimCLR standard for 224x224 images: 10% of image size, rounded to nearest odd).
**Warning signs:** Runtime error from torchvision.

### Pitfall 7: InfoNCE Numerical Stability
**What goes wrong:** Loss becomes NaN or Inf with large batch sizes or extreme temperatures.
**Why it happens:** Softmax over large similarity values overflows.
**How to avoid:** Subtract the maximum logit before exp (log-sum-exp trick). Use `F.cross_entropy` which handles this internally.
**Warning signs:** NaN loss in early training steps.

### Pitfall 8: LARS Trust Ratio for Bias/Norm Parameters
**What goes wrong:** Applying LARS scaling to bias and batch norm parameters destabilizes training.
**Why it happens:** These parameters have different gradient dynamics than weight tensors.
**How to avoid:** Per D-05, `exclude_bias_and_norm=True` by default. Filter by parameter dimensionality: exclude 1-D tensors (biases, BN weights, BN biases).
**Warning signs:** Training diverges with LARS on small models.

## Code Examples

### InfoNCE Loss (Symmetric -- SimCLR Style)
```python
# Verified pattern from SimCLR paper + Lightning tutorial
import torch
import torch.nn.functional as F

class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.5, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, z_i, z_j, queue=None):
        """
        Symmetric InfoNCE.
        z_i, z_j: [B, D] L2-normalized embeddings from two views.
        queue: optional [D, K] tensor for asymmetric (MoCo) mode.
        """
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        if queue is None:
            # Symmetric in-batch (SimCLR)
            B = z_i.shape[0]
            z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
            sim = z @ z.T / self.temperature    # [2B, 2B]

            # Mask out self-similarity
            mask = ~torch.eye(2 * B, dtype=torch.bool, device=z.device)
            sim = sim.masked_fill(~mask, float("-inf"))

            # Labels: positive pairs are at offset B
            labels = torch.cat([
                torch.arange(B, 2 * B, device=z.device),
                torch.arange(0, B, device=z.device),
            ])

            loss = F.cross_entropy(sim, labels, reduction=self.reduction)
        else:
            # Asymmetric with queue (MoCo)
            # z_i: queries [B, D], z_j: positive keys [B, D]
            # queue: [D, K] negative keys
            l_pos = (z_i * z_j).sum(dim=1, keepdim=True) / self.temperature  # [B, 1]
            l_neg = z_i @ queue / self.temperature  # [B, K]
            logits = torch.cat([l_pos, l_neg], dim=1)  # [B, 1+K]
            labels = torch.zeros(z_i.shape[0], dtype=torch.long, device=z_i.device)
            loss = F.cross_entropy(logits, labels, reduction=self.reduction)

        return loss
```

### LARS Optimizer Skeleton
```python
# Based on: https://arxiv.org/abs/1708.03888
# Pattern from torchlars, flash.core.optimizers, adapted for from-scratch per D-04
import torch
from torch.optim import Optimizer

class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling optimizer.

    Reference: https://arxiv.org/abs/1708.03888
    """
    def __init__(self, params, lr, momentum=0.9, weight_decay=1e-6,
                 eta=0.001, exclude_bias_and_norm=True):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        eta=eta, exclude_bias_and_norm=exclude_bias_and_norm)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                # LARS scaling: skip for bias/norm (1-D params)
                if group["exclude_bias_and_norm"] and p.ndim == 1:
                    trust_ratio = 1.0
                else:
                    p_norm = p.norm()
                    g_norm = grad.norm()
                    if p_norm > 0 and g_norm > 0:
                        trust_ratio = group["eta"] * p_norm / g_norm
                    else:
                        trust_ratio = 1.0

                scaled_lr = group["lr"] * trust_ratio

                # Momentum update
                if group["momentum"] != 0:
                    buf = self.state[p].get("momentum_buffer", None)
                    if buf is None:
                        buf = torch.clone(grad).detach()
                        self.state[p]["momentum_buffer"] = buf
                    else:
                        buf.mul_(group["momentum"]).add_(grad)
                    grad = buf

                p.add_(grad, alpha=-scaled_lr)

        return loss
```

### ProjectionHead MLP
```python
# Standard SSL projection head pattern
import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, use_bn: bool = True):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            out_dim = output_dim if is_last else hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_dim))
            if not is_last:
                layers.append(nn.ReLU(inplace=True))
            # Final layer: BN only, no ReLU

            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
```

### Warmup-Cosine Scheduler in Lightning
```python
# Using native PyTorch schedulers with Lightning
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

def configure_optimizers(self):
    # Optimizer dispatch
    if self.cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(self.learnable_params, lr=self.cfg.lr,
                                       weight_decay=self.cfg.weight_decay)
    elif self.cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(self.learnable_params, lr=self.cfg.lr,
                                     momentum=0.9, weight_decay=self.cfg.weight_decay)
    elif self.cfg.optimizer == "lars":
        optimizer = LARS(self.learnable_params, lr=self.cfg.lr,
                         weight_decay=self.cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {self.cfg.optimizer}")

    total_steps = self.trainer.estimated_stepping_batches
    warmup_steps = int(total_steps * self.cfg.warmup_epochs / self.cfg.max_epochs)

    warmup = LinearLR(optimizer, start_factor=1e-4, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
    }
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torchvision.transforms` (v1) | `torchvision.transforms.v2` | torchvision 0.15+ (2023) | New API is recommended; v1 still works but v2 is preferred |
| `pl_bolts` schedulers | Native `SequentialLR` | PyTorch 1.13+ (2022) | `LinearLR` + `CosineAnnealingLR` via `SequentialLR` is zero-dep |
| Pydantic v1 `parse_obj()` | Pydantic v2 `model_validate()` | Pydantic v2 (2023) | Different API, `ConfigDict` replaces `class Config` |
| `timm.create_model(num_classes=0, global_pool='')` | `timm.create_model(num_classes=0)` | timm 0.9+ | `num_classes=0` retains pooling but removes classifier |
| Manual Lightning callbacks for logging | `self.log()` / `self.log_dict()` | Lightning 1.x+ | Built-in, automatic TensorBoard integration |

**Deprecated/outdated:**
- `pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR`: Marked "under review" in Lightning Bolts 0.7.0. Use native PyTorch `SequentialLR` instead.
- `torchvision.transforms.Compose` (v1): Still works but `v2.Compose` is the recommended path forward.
- `Pydantic BaseSettings` for YAML config: Not needed; `BaseModel.model_validate(yaml.safe_load(...))` is simpler and sufficient.

## Open Questions

1. **Scheduler step granularity (epoch vs step)**
   - What we know: `SequentialLR` with `interval="step"` gives smooth warmup. Epoch-based warmup is coarser but simpler.
   - What's unclear: Whether step-based is necessary for short training runs (e.g., 10 epochs on CIFAR-10 toy examples).
   - Recommendation: Default to step-based (`interval="step"`) for correctness. It works for both short and long runs.

2. **`core/__init__.py` export strategy**
   - What we know: User wants `from core.base import BaseSSLModule` to work (D-03).
   - What's unclear: Whether to also support `from core import BaseSSLModule` via `__init__.py` re-exports.
   - Recommendation: Add `__init__.py` with public API re-exports for convenience. Keep explicit imports as the documented style.

3. **ContrastiveAugmentation: class vs function**
   - What we know: Needs to produce `n_views` augmented copies. `MultiViewTransform` wrapper is the standard pattern.
   - What's unclear: Whether `ContrastiveAugmentation` should be the transform itself or a factory that returns a `MultiViewTransform`.
   - Recommendation: Make `ContrastiveAugmentation` a callable class that applies a single augmentation. Wrap it in `MultiViewTransform(ContrastiveAugmentation(...), n_views=N)` for multi-view. This keeps concerns separated.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.10+ | Project constraint | Partial | 3.13.5 (conda base) / 3.9.6 (system) | Create conda env with Python 3.10+ |
| PyTorch | All components | Yes (conda base) | 2.10.0 | Need 2.11.0 in project env |
| Lightning | BaseSSLModule, SSLDataModule | No | -- | `pip install lightning` |
| timm | build_backbone | No | -- | `pip install timm` |
| Pydantic v2 | TrainConfig | Yes (conda base) | 2.11.7 | Need in project env |
| torchvision | Augmentations | No | -- | `pip install torchvision` |
| PyYAML | Config loading | Yes (conda base) | -- | Need in project env |
| pytest | Testing | Yes (conda base) | 8.4.1 | Already available |

**Missing dependencies with no fallback:**
- A dedicated project environment must be created. The conda base env has PyTorch and Pydantic but is missing lightning, timm, and torchvision.

**Missing dependencies with fallback:**
- None -- all dependencies are pip-installable.

**Recommendation:** Wave 0 should include creating a `requirements.txt` with pinned versions, and the planner should note that a virtual environment or conda env is needed.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.4.1 |
| Config file | none -- see Wave 0 |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FOUND-01 | BaseSSLModule subclass trains 1 epoch on toy data | integration | `pytest tests/test_base.py -x` | Wave 0 |
| FOUND-02 | TrainConfig validates YAML, rejects unknown keys | unit | `pytest tests/test_config.py -x` | Wave 0 |
| FOUND-03 | build_backbone returns (backbone, feat_dim) for resnet50 and vit_small | unit | `pytest tests/test_backbone.py -x` | Wave 0 |
| FOUND-04 | ProjectionHead BN+ReLU pattern correct for 2 and 3 layers | unit | `pytest tests/test_projection.py -x` | Wave 0 |
| FOUND-05 | ContrastiveAugmentation strong/weak paths produce correct output | unit | `pytest tests/test_data.py::test_augmentation -x` | Wave 0 |
| FOUND-06 | SSLDataModule yields [n_views, B, C, H, W] batches | integration | `pytest tests/test_data.py::test_datamodule -x` | Wave 0 |
| FOUND-07 | method_dispatcher raises ValueError on unknown method | unit | `pytest tests/test_dispatcher.py -x` | Wave 0 |
| FOUND-08 | EvalConfig sub-schema validates correctly | unit | `pytest tests/test_config.py::test_eval_config -x` | Wave 0 |
| FOUND-09 | self.log() produces TensorBoard entries | integration | `pytest tests/test_base.py::test_logging -x` | Wave 0 |
| FOUND-10 | EMAUpdater.step() updates target, target not in learnable_params | unit | `pytest tests/test_ema.py -x` | Wave 0 |
| INFRA-01 | InfoNCELoss finite for symmetric and asymmetric modes | unit | `pytest tests/test_losses.py -x` | Wave 0 |
| INFRA-06 | LARS optimizer step runs without error, respects exclude_bias_and_norm | unit | `pytest tests/test_optimizers.py -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/ -x -q`
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/__init__.py` -- package init
- [ ] `tests/conftest.py` -- shared fixtures (synthetic ImageFolder, random tensor helpers, toy config dict)
- [ ] `pytest.ini` or `pyproject.toml [tool.pytest.ini_options]` -- pytest config
- [ ] `requirements.txt` -- pinned dependency versions
- [ ] All test files listed above (test_backbone.py, test_base.py, test_config.py, test_data.py, test_dispatcher.py, test_ema.py, test_losses.py, test_optimizers.py, test_projection.py)

## Sources

### Primary (HIGH confidence)
- [PyTorch Lightning 2.6.1 Optimization docs](https://lightning.ai/docs/pytorch/stable/common/optimization.html) -- configure_optimizers return format, scheduler interval
- [PyTorch Lightning 2.6.1 LightningModule API](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html) -- self.log(), on_train_batch_end, estimated_stepping_batches
- [timm Feature Extraction docs (HuggingFace)](https://huggingface.co/docs/timm/feature_extraction) -- num_classes=0, num_features, forward_features
- [Pydantic v2 Models docs](https://docs.pydantic.dev/latest/concepts/models/) -- model_validate, ConfigDict, extra='forbid'
- [Pydantic v2 Configuration docs](https://docs.pydantic.dev/latest/api/config/) -- ConfigDict options
- [torchvision transforms v2 docs](https://docs.pytorch.org/vision/main/transforms.html) -- v2 API, GaussianBlur, ColorJitter
- [PyTorch CosineAnnealingLR docs](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html) -- scheduler API
- [PyTorch LinearLR docs](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html) -- warmup via start_factor

### Secondary (MEDIUM confidence)
- [Lightning SimCLR tutorial](https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html) -- SimCLR implementation pattern in Lightning
- [LARS paper](https://arxiv.org/abs/1708.03888) -- original algorithm specification
- [torchlars GitHub](https://github.com/kakaobrain/torchlars) -- reference LARS implementation
- [flash.core.optimizers.LARS](https://lightning-flash.readthedocs.io/en/stable/api/generated/flash.core.optimizers.LARS.html) -- Lightning Flash LARS reference
- PyPI version checks (2026-03-31) -- torch 2.11.0, lightning 2.6.1, timm 1.0.26, pydantic 2.12.5, torchvision 0.26.0, pyyaml 6.0.3

### Tertiary (LOW confidence)
- None -- all findings verified against official documentation.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries are stable, well-documented, versions verified against PyPI
- Architecture: HIGH -- patterns are well-established in SSL literature and Lightning ecosystem
- Pitfalls: HIGH -- drawn from official docs, known issues, and SSL community experience

**Research date:** 2026-03-31
**Valid until:** 2026-04-30 (stable libraries, slow-moving domain)
