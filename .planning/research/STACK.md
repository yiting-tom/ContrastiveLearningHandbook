# Stack Research

**Project:** SSL / Contrastive Learning Tutorial Repository
**Researched:** 2026-03-29
**Confidence:** HIGH (PyTorch Lightning, timm APIs verified against official docs; solo-learn patterns verified against source; pitfall claims corroborated by multiple papers and library implementations)

---

## Recommended Architecture

### Layered Module Hierarchy

The canonical pattern used by solo-learn (the most actively maintained SSL research library as of 2024) is a two-level class hierarchy:

```
lightning.LightningModule
    └── BaseSSLModule          # shared: backbone, projector interface, optimizer, scheduler
            ├── SimCLRModule   # adds: NTXent loss, symmetric forward
            ├── MoCoModule     # adds: momentum encoder, memory bank
            ├── BYOLModule     # adds: predictor head, EMA update, stop-gradient
            ├── SimSiamModule  # adds: predictor, stop-gradient, no negatives
            └── BarlowTwins    # adds: cross-correlation loss
```

The `BaseSSLModule` owns:
- Backbone instantiation via timm (see timm section)
- Projection head interface (`build_projector()` method to override)
- `learnable_params` property for optimizer param group construction
- `configure_optimizers()` with warmup-cosine scheduler
- `validation_step()` with optional linear probe / KNN evaluation
- Logging helpers (`self.log(...)`)

Each method subclass owns only what differs: the forward pass, the loss, and any additional components (momentum encoder, predictor, prototypes).

**Source:** solo-learn `solo/methods/base.py` exposes `BaseMethod(LightningModule)` — all 13+ methods extend it. Verified via GitHub source inspection.

---

## Lightning Integration Patterns

### Pattern 1: Shared Base Class with `learnable_params` Property

Solo-learn's key design decision: optimizers are configured from a `learnable_params` property rather than hardcoding `self.parameters()`. This allows subclasses to inject extra parameter groups (e.g., a predictor head with different lr, a prototypes tensor):

```python
import lightning as L
from torch import nn

class BaseSSLModule(L.LightningModule):
    def __init__(self, backbone, projector_dim, lr, weight_decay, max_epochs, warmup_epochs):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        self.projector = self.build_projector(projector_dim)

    def build_projector(self, dim: int) -> nn.Module:
        # override in subclasses
        raise NotImplementedError

    @property
    def learnable_params(self) -> list[dict]:
        # subclasses append extra param dicts here
        return [
            {"name": "backbone", "params": self.backbone.parameters()},
            {"name": "projector", "params": self.projector.parameters()},
        ]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.learnable_params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
```

**Why this pattern:** It eliminates the common mistake of passing `self.parameters()` to the optimizer when momentum encoders are present (momentum encoder params must be excluded — they should not receive gradient updates).

### Pattern 2: Momentum Encoder Updates

For MoCo and BYOL, the momentum encoder is a deep copy of the online encoder with gradients disabled. The update location matters:

- **Inside `training_step` (before forward pass):** Used by LightlySSL. Simple, but the EMA update happens on weights from the previous step — semantically consistent.
- **Inside `on_train_batch_end` (after optimizer.step):** Ensures EMA is computed on the freshly updated online encoder weights. Preferred for BYOL-style methods where the target network guides learning.

```python
import copy
from lightly.models.utils import deactivate_requires_grad, update_momentum

class MoCoModule(BaseSSLModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone_ema = copy.deepcopy(self.backbone)
        self.projector_ema = copy.deepcopy(self.projector)
        deactivate_requires_grad(self.backbone_ema)
        deactivate_requires_grad(self.projector_ema)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update after optimizer step — uses freshly updated online weights
        m = self.hparams.momentum
        update_momentum(self.backbone, self.backbone_ema, m)
        update_momentum(self.projector, self.projector_ema, m)
```

**Critical:** Never include `backbone_ema` or `projector_ema` in `learnable_params`. They are purely inference networks.

### Pattern 3: Manual vs. Automatic Optimization

Most SSL methods work with automatic optimization (`self.automatic_optimization = True`, the default). Use manual optimization only when:
- You need to clip gradients on different parameter groups with different thresholds.
- You implement a GAN-style alternating update (rare in SSL).

For standard methods (SimCLR, BYOL, Barlow Twins), stick with automatic optimization and configure gradient clipping via `Trainer(gradient_clip_val=1.0)`.

**Gradient clipping note (Lightning 2.x verified):**
- `Trainer(gradient_clip_val=1.0)` clips global norm by default (`gradient_clip_algorithm="norm"`).
- Mixed precision is safe: Lightning unscales gradients before clipping.
- With manual optimization, `configure_gradient_clipping()` is NOT called — use `self.clip_gradients(opt, ...)` explicitly.

### Pattern 4: `save_hyperparameters()` for Checkpoint Reproducibility

Call `self.save_hyperparameters()` in `__init__`. This stores all constructor arguments under `self.hparams` and serializes them into Lightning checkpoints. Access them as `self.hparams.lr`, etc.

```python
def __init__(self, lr, weight_decay, temperature, max_epochs, warmup_epochs):
    super().__init__()
    self.save_hyperparameters()  # all args now in self.hparams
```

**Exception:** Pass `ignore=["backbone"]` if backbone is an `nn.Module` (not serializable as a plain hyperparameter).

### Pattern 5: Distributed Training

```python
trainer = L.Trainer(
    max_epochs=200,
    accelerator="gpu",
    devices="auto",
    strategy="ddp",
    sync_batchnorm=True,          # critical for SimCLR, BYOL with BN in projector
    precision="16-mixed",         # safe with Lightning's auto gradient unscaling
)
```

`sync_batchnorm=True` is required whenever BatchNorm appears in the projector or prediction head — without it, BN statistics are per-GPU, causing inconsistent normalization across devices and potential collapse in methods that implicitly rely on batch statistics (especially BYOL).

---

## timm Integration

### Recommended API: `timm.create_model` with `num_classes=0`

For SSL, the backbone must output pooled feature vectors (not class logits). The cleanest approach:

```python
import timm

def build_backbone(model_name: str, pretrained: bool = False) -> tuple[nn.Module, int]:
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,        # removes classifier head, keeps global pooling
    )
    feature_dim = backbone.num_features  # works for ALL timm architectures
    return backbone, feature_dim
```

`backbone.num_features` is the universal attribute across timm architectures (ResNet, ViT, ConvNeXt, EfficientNet, etc.) that gives the embedding dimension without querying a dummy forward pass.

**Do NOT use `backbone.inplanes`** — this is ResNet-specific and does not generalize. Solo-learn's older code makes this mistake for legacy ResNets; the correct cross-architecture attribute is `num_features`.

### Feature Extraction API Options (from official timm docs, HIGH confidence)

| API | Returns | Use Case |
|-----|---------|----------|
| `create_model(..., num_classes=0)` | Pooled vector `[B, D]` | SSL pretraining (recommended) |
| `model.forward_features(x)` | Unpooled spatial features `[B, D, H, W]` | Dense / pixel tasks |
| `create_model(..., num_classes=0, global_pool='')` | Unpooled `[B, D, H, W]` | Explicit control |
| `create_model(..., features_only=True)` | List of multi-scale maps | Detection / segmentation |
| `model.forward_intermediates(x)` | Intermediate layer outputs | ViT layer-wise analysis |

For SSL tutorials, `num_classes=0` with pooled output is the right default.

### Projector Head Separation Pattern

```python
class SSLBackboneWithProjector(nn.Module):
    def __init__(self, model_name: str, proj_hidden_dim: int, proj_out_dim: int):
        super().__init__()
        self.backbone, feat_dim = build_backbone(model_name)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_out_dim),
        )

    def forward(self, x):
        h = self.backbone(x)          # pooled features [B, feat_dim]
        z = self.projector(h)         # projected embeddings [B, proj_out_dim]
        return h, z                   # return BOTH for downstream use
```

**Why return both `h` and `z`:** The pre-projection representation `h` consistently outperforms post-projection `z` on downstream tasks. Returning both lets the `LightningModule.training_step` use `z` for loss computation, while the `validation_step` evaluates `h` with a linear probe or KNN. This is the pattern used by both solo-learn and LightlySSL.

### Backbone Name Convention (timm 1.x)

timm 1.0+ uses `architecture.pretrained_tag` naming:

```python
# Old (timm 0.x)
timm.create_model("vit_base_patch16_224_in21k", ...)

# New (timm 1.x) — preferred
timm.create_model("vit_base_patch16_224.augreg_in21k", ...)
timm.create_model("resnet50.a1_in1k", ...)   # specific weight variant
timm.create_model("resnet50", ...)            # still works, defaults to first entry in default_cfgs
```

For a tutorial, using bare architecture names (`resnet50`, `vit_small_patch16_224`) is fine — they resolve to the best default weights.

---

## Config System

### Recommended: Pydantic v2 + PyYAML (no Hydra, no OmegaConf)

Pydantic v2 provides type validation, IDE completion, nested model support, and helpful error messages — all without adding the heavy Hydra dependency with its complex override syntax.

**Pattern: discriminated union on `method` field**

```yaml
# configs/simclr_resnet50.yaml
method: simclr
backbone: resnet50
pretrained: false
max_epochs: 200
warmup_epochs: 10
batch_size: 256
lr: 3.0e-4
weight_decay: 1.0e-6
optimizer: adamw
scheduler: warmup_cosine

simclr:
  temperature: 0.07
  proj_hidden_dim: 2048
  proj_out_dim: 128
```

```yaml
# configs/byol_resnet50.yaml
method: byol
backbone: resnet50
max_epochs: 200
warmup_epochs: 10
batch_size: 256
lr: 3.0e-4
weight_decay: 1.5e-6
optimizer: adamw
scheduler: warmup_cosine

byol:
  proj_hidden_dim: 4096
  proj_out_dim: 256
  pred_hidden_dim: 4096
  pred_out_dim: 256
  momentum: 0.996
```

```python
# configs/schema.py
from __future__ import annotations
from typing import Literal, Annotated, Union
from pydantic import BaseModel, Field
import yaml

class SimCLRConfig(BaseModel):
    temperature: float = 0.07
    proj_hidden_dim: int = 2048
    proj_out_dim: int = 128

class BYOLConfig(BaseModel):
    proj_hidden_dim: int = 4096
    proj_out_dim: int = 256
    pred_hidden_dim: int = 4096
    pred_out_dim: int = 256
    momentum: float = 0.996

class TrainConfig(BaseModel):
    method: str
    backbone: str = "resnet50"
    pretrained: bool = False
    max_epochs: int = 200
    warmup_epochs: int = 10
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-6
    optimizer: Literal["sgd", "adamw", "lars"] = "adamw"
    scheduler: Literal["warmup_cosine", "cosine", "none"] = "warmup_cosine"
    # method-specific blocks are optional at the top level
    simclr: SimCLRConfig = Field(default_factory=SimCLRConfig)
    byol: BYOLConfig = Field(default_factory=BYOLConfig)

def load_config(path: str) -> TrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return TrainConfig.model_validate(raw)
```

**Why this pattern:**
- Single YAML file per experiment (easy to track in git)
- Method-specific blocks are namespaced (`simclr.temperature`) — no key collision
- Unused method blocks have Pydantic defaults, so YAML is minimal
- `TrainConfig.method` is a plain string used to dispatch to the right `LightningModule` subclass at runtime
- No Hydra multirun magic needed for a tutorial repo

**Dispatcher pattern:**

```python
def build_module(cfg: TrainConfig) -> BaseSSLModule:
    backbone, feat_dim = build_backbone(cfg.backbone, cfg.pretrained)
    dispatch = {
        "simclr": lambda: SimCLRModule(backbone, feat_dim, cfg.simclr, cfg),
        "byol":   lambda: BYOLModule(backbone, feat_dim, cfg.byol, cfg),
        "moco":   lambda: MoCoModule(backbone, feat_dim, cfg.moco, cfg),
    }
    if cfg.method not in dispatch:
        raise ValueError(f"Unknown method: {cfg.method}. Available: {list(dispatch)}")
    return dispatch[cfg.method]()
```

### Alternative: OmegaConf without Hydra

If you want variable interpolation (`${lr}` references inside YAML), OmegaConf alone (without Hydra) is viable:

```python
from omegaconf import OmegaConf
cfg = OmegaConf.load("config.yaml")
lr = cfg.lr  # DictConfig, not validated
```

Drawback: No type validation, no IDE completion on DictConfig attributes. Use Pydantic instead for a tutorial repo where clarity matters.

---

## Common Pitfalls

### Pitfall 1: Including Momentum Encoder in Optimizer (CRITICAL)

**What goes wrong:** If `backbone_ema` or `projector_ema` parameters are included in `self.parameters()` (or any optimizer param group), they receive gradient updates. This corrupts the EMA and causes training instability or silent incorrect behavior.

**Prevention:** Explicitly use the `learnable_params` property pattern (Pattern 1 above). Never call `optimizer = Adam(self.parameters())` when momentum encoders are present.

**Detection:** Check that `deactivate_requires_grad(self.backbone_ema)` is called immediately after `deepcopy`. Log `sum(p.requires_grad for p in self.parameters())` at init to verify parameter count.

---

### Pitfall 2: Representation Collapse in Non-Contrastive Methods

**What goes wrong:** BYOL and SimSiam can trivially minimize their loss by outputting a constant embedding for all inputs (the "constant collapse" mode). The predictor head and stop-gradient are not optional — removing either causes immediate collapse.

**Root cause:** Without negative samples, the loss has a degenerate global minimum (all embeddings equal). The predictor head + stop-gradient provide an implicit mechanism to avoid it. Batch normalization in the projection head also implicitly prevents collapse.

**Prevention:**
- For BYOL: predictor head on the online branch only; stop-gradient on the target branch output.
- For SimSiam: same, but no EMA — use stop-gradient in the symmetric loss.
- Never remove BN from the projector in non-contrastive methods without replacement regularization.

**Detection:** Monitor the standard deviation of the batch embeddings (`z.std(dim=0).mean()`). A collapsing model will show this approaching 0 within 1-2 epochs.

---

### Pitfall 3: Wrong EMA Update Timing

**What goes wrong:** Updating the momentum encoder inside `on_before_optimizer_step` (before `optimizer.step()`) means the EMA sees weights from before the gradient update — semantically equivalent to a one-step lag, which is fine. However, some implementations accidentally update it in `on_train_batch_start` (before the forward pass), using weights from two steps ago.

**Prevention:** Use `on_train_batch_end` for EMA updates in the `LightningModule`. This ensures the EMA is always computed on the most recently updated online weights.

**Note from Lightning docs (verified):** Inside `on_before_optimizer_step`, gradients are unscaled but NOT clipped. Do not apply EMA logic here — use `on_train_batch_end`.

---

### Pitfall 4: Gradient Clipping Disabled in Manual Optimization

**What goes wrong:** Setting `Trainer(gradient_clip_val=1.0)` only works with automatic optimization. If `self.automatic_optimization = False`, the Trainer silently ignores `gradient_clip_val`.

**Prevention:** In manual optimization, call `self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")` explicitly inside `training_step`.

---

### Pitfall 5: BatchNorm Behavior in DDP

**What goes wrong:** Without `sync_batchnorm=True`, each GPU computes BN statistics independently. For small per-GPU batch sizes (common in SSL with large models), this means noisy statistics. Methods that implicitly rely on BN for collapse prevention (BYOL) can become unstable.

**Prevention:** Always use `trainer = L.Trainer(..., sync_batchnorm=True)` when training with multiple GPUs and any BN layers in the network.

---

### Pitfall 6: Not Using `backbone.num_features` for Feature Dim

**What goes wrong:** Hard-coding the projector input dimension (e.g., `nn.Linear(2048, ...)` for ResNet-50) breaks when switching backbones. Using `backbone.inplanes` works only for ResNet.

**Prevention:** Use `timm.create_model(..., num_classes=0).num_features` — this attribute is available on all timm architectures and returns the correct pooled feature dimension.

---

### Pitfall 7: Dimensional Collapse (Partial Collapse)

**What goes wrong:** Even when trivial collapse is avoided, the representation can undergo dimensional collapse — most of the embedding dimensions carry no signal (near-zero singular values in the embedding covariance matrix).

**Who is affected:** SimSiam with small backbones (ResNet-18 on large datasets). Less of a problem for SimCLR, BYOL, and Barlow Twins.

**Prevention:** For a tutorial repo, include Barlow Twins (which explicitly decorrelates embedding dimensions) as a demonstration of an approach designed against dimensional collapse. For SimSiam tutorials, recommend ResNet-50 minimum.

---

### Pitfall 8: `model_name` Breaking Changes in timm 1.x

**What goes wrong:** timm 1.0 renamed many models to the `architecture.pretrained_tag` convention. Code written for timm 0.x using names like `vit_base_patch16_224_in21k` raises `RuntimeError: model vit_base_patch16_224_in21k not found` in timm 1.x.

**Prevention:** Use architecture-only names (`vit_base_patch16_224`, `resnet50`) which still resolve correctly in timm 1.x via `default_cfgs`. Document the timm version pinned in `requirements.txt`.

---

## Key Dependencies (with versions)

| Package | Recommended Version | Purpose | Notes |
|---------|-------------------|---------|-------|
| `torch` | `>=2.1, <2.7` | Core tensor ops | Lightning 2.6.x supports last 5 PyTorch minors |
| `lightning` | `>=2.4, <3.0` | Training framework | Use `import lightning as L` (not `pytorch_lightning`) |
| `timm` | `>=1.0.0` | Backbone models | 1.x has breaking renames vs 0.x; pin major version |
| `pydantic` | `>=2.0` | Config validation | v2 uses Rust core, 10-50x faster than v1 |
| `pyyaml` | `>=6.0` | YAML loading | Used with Pydantic `model_validate` |
| `lightly` | `>=1.5` (optional) | Loss functions, EMA utils | Use `lightly.loss`, `lightly.models.utils`; skip if building from scratch |
| `torchvision` | matches torch | Augmentations | `transforms.v2` API preferred in PyTorch 2.x |

```bash
# Minimal install
pip install torch>=2.1 lightning>=2.4 timm>=1.0 pydantic>=2.0 pyyaml>=6.0

# With lightly utilities (recommended for tutorials)
pip install lightly>=1.5

# Dev / optional
pip install tensorboard einops matplotlib
```

**Python requirement:** `>=3.10` (Lightning 2.4+ requires Python 3.10+, matching the project spec).

---

## Sources

- solo-learn source `solo/methods/base.py`: https://github.com/vturrisi/solo-learn (HIGH confidence — inspected directly)
- LightlySSL MoCo example: https://docs.lightly.ai/self-supervised-learning/examples/moco.html (HIGH confidence)
- timm feature extraction API: https://huggingface.co/docs/timm/feature_extraction (HIGH confidence — official docs)
- timm changelog / 1.x naming: https://huggingface.co/docs/timm/changes (HIGH confidence — official docs)
- PyTorch Lightning optimization: https://lightning.ai/docs/pytorch/stable/common/optimization.html (HIGH confidence — official docs)
- PyTorch Lightning training tricks: https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html (HIGH confidence — official docs)
- PyTorch Lightning LightningModule API: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html (HIGH confidence)
- Pydantic v2 docs: https://docs.pydantic.dev/latest/ (HIGH confidence)
- SSL collapse analysis (BYOL/SimSiam): https://www.cs.cmu.edu/~dpathak/papers/eccv22.pdf (MEDIUM confidence — academic paper)
- Preventing dimensional collapse NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/file/ad7922fd4650f8aba5d8b067e622ca84-Paper-Conference.pdf (MEDIUM confidence)
- PyTorch Lightning changelog 2.6.x: https://lightning.ai/docs/pytorch/stable/generated/CHANGELOG.html (HIGH confidence)
