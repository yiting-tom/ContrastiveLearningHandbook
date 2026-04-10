# Phase 9: Evaluation Suite - Research

**Researched:** 2026-04-10
**Domain:** ML evaluation pipelines (k-NN, linear probing, visualization, CAM) on top of PyTorch Lightning SSL checkpoints
**Confidence:** HIGH

## Summary

Phase 9 builds six evaluation components: an in-training `KNNCallback`, four offline scripts (`linear_probe.py`, `tsne_vis.py`, `umap_vis.py`, `finetune.py`), and a CAM visualization script (`cam_vis.py`), plus an integration test. The existing codebase already provides all config schemas (`EvalConfig` + 6 sub-configs in `core/config.py`), the `SSLDataModule.val_dataloader()` integration point, and `build_backbone()` / `method_dispatcher()` / `get_method()` for checkpoint loading. Three new pip dependencies are needed: `faiss-cpu`, `umap-learn`, and `grad-cam` (PyPI name for pytorch-grad-cam).

All eval scripts follow the same invocation pattern: `python eval/<script>.py configs/<method>.yaml --ckpt <path>`. They load configs via `TrainConfig.model_validate(yaml.safe_load(...))` and access `cfg.eval.*` sub-configs. The `KNNCallback` is the only component that runs during training; everything else is post-hoc.

**Primary recommendation:** Follow the established `dinov2_demo.py` pattern for script structure (importable modules with `__main__` guard, argparse for `--ckpt`). Use `get_method(cfg.method).load_from_checkpoint(ckpt_path)` for checkpoint loading (Lightning's native mechanism). Keep each script self-contained with no shared eval utility module -- the scripts are simple enough that DRY is less important than tutorial readability.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** All five offline eval scripts use the same invocation pattern: `python eval/<script>.py configs/simclr.yaml --ckpt outputs/run1/checkpoints/epoch-99.ckpt`. YAML config supplies eval settings. Each script is standalone.
- **D-02:** `KNNCallback` accesses labeled val data via `trainer.datamodule.val_dataloader()`. No new DataModule interface required. Must guard against `val_dataset is None`.
- **D-03:** All three eval-specific libraries go into `requirements.txt` (not a separate extras file): `faiss-cpu>=1.7`, `umap-learn>=0.5`, `pytorch-grad-cam>=1.4`.
- **D-04:** Linear probe caches pre-extracted features in a sibling `cache/` directory next to the checkpoint file, keyed to checkpoint filename.
- **D-05:** Integration test uses synthetic checkpoint + synthetic ImageFolder (no network access). knn_acc threshold relaxed to `>= 0.0`.

### Claude's Discretion
- ArgumentParser argument names for `--ckpt` and secondary flags (`--output-dir`, `--device`)
- Whether eval scripts are importable modules or pure `__main__` scripts (prefer importable for testability)
- Exact PNG filename conventions for t-SNE (keep perplexity in filename)
- FAISS brute-force fallback threshold implementation details (>100K detection)
- EigenCAM target layer selection logic (ResNet vs ViT detection)

### Deferred Ideas (OUT OF SCOPE)
None.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EVAL-01 | k-NN callback with FAISS for >100K, weighted voting, configurable k/temperature/every_n_epochs | KNNCallback pattern, FAISS IndexFlatIP API, Lightning Callback hooks |
| EVAL-02 | Linear probe script: frozen backbone, SGD, weight_decay=0.0, MultiStepLR [60,80], feature caching | LinearProbeModule as LightningModule, feature extraction + torch.save caching |
| EVAL-03 | t-SNE visualization: PCA to 50 dims, cosine metric, 3 perplexity sweep, PNG output | scikit-learn TSNE API (already installed v1.7.1), matplotlib (v3.10.3) |
| EVAL-04 | UMAP visualization: cosine metric, 5000 samples, torchdr suggestion for >50K | umap-learn API, new dependency |
| EVAL-05 | Fine-tuning script: separate LR groups, AdamW, freeze_bn option | FinetuneModule as LightningModule, param group configuration |
| EVAL-06 | CAM visualization: EigenCAM default, GradCAM fallback, architecture-aware target layers | grad-cam library API, ViT reshape transform |
| FOUND-08 | EvalConfig schema in TrainConfig | Already implemented in core/config.py lines 153-211 -- SATISFIED |
</phase_requirements>

## Standard Stack

### Core (already installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.10.0 | Tensor ops, model loading | Project foundation [VERIFIED: pip show] |
| lightning | 2.5.2 | Callback system, Trainer, checkpoint loading | Project foundation [VERIFIED: pip show] |
| scikit-learn | 1.7.1 | t-SNE, PCA pre-reduction | Standard ML toolkit, `TSNE(init='pca', metric='cosine', learning_rate='auto')` [VERIFIED: pip show] |
| matplotlib | 3.10.3 | Plot generation for t-SNE, UMAP, CAM overlay | Standard plotting [VERIFIED: pip show] |
| timm | 1.0.26 | Backbone factory | Already used project-wide [VERIFIED: pip show] |

### New Dependencies (D-03)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| faiss-cpu | 1.13.2 | k-NN search at scale (>100K samples) | Facebook AI standard for similarity search; `IndexFlatIP` for exact inner product [VERIFIED: pip index] |
| umap-learn | 0.5.12 | UMAP dimensionality reduction visualization | De facto standard UMAP implementation [VERIFIED: pip index] |
| grad-cam | 1.5.5 | CAM visualization (EigenCAM, GradCAM) | PyPI package name for pytorch-grad-cam; supports ViT reshape [VERIFIED: pypi.org] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| faiss-cpu for k-NN | torch cdist/mm brute force | Brute force is fine for <100K; FAISS needed only for scale. Both paths needed per EVAL-01 |
| umap-learn | torchdr | torchdr is GPU-accelerated but less mature; umap-learn is the standard. Suggest torchdr for >50K only |
| sklearn TSNE | openTSNE | openTSNE is faster but adds another dependency; sklearn is already installed |

**Installation:**
```bash
pip install faiss-cpu>=1.7 umap-learn>=0.5 grad-cam>=1.4
```

Then add these three lines to `requirements.txt`.

## Architecture Patterns

### Recommended Project Structure
```
eval/
    __init__.py          # existing (empty)
    dinov2_demo.py       # existing reference script
    knn_callback.py      # NEW: KNNCallback(L.Callback)
    linear_probe.py      # NEW: LinearProbeModule + __main__
    tsne_vis.py          # NEW: t-SNE visualization + __main__
    umap_vis.py          # NEW: UMAP visualization + __main__
    finetune.py          # NEW: FinetuneModule + __main__
    cam_vis.py           # NEW: CAM visualization + __main__
tests/
    test_eval_knn.py     # NEW
    test_eval_linear_probe.py  # NEW
    test_eval_tsne.py    # NEW
    test_eval_umap.py    # NEW
    test_eval_finetune.py # NEW
    test_eval_cam.py     # NEW
    test_eval_integration.py   # NEW: 09-07 integration test
```

### Pattern 1: Eval Script Structure (all 5 offline scripts)
**What:** Each script is an importable module with argument parsing, a `main()` function, and `__main__` guard.
**When to use:** All offline eval scripts.
**Example:**
```python
# Source: Derived from existing eval/dinov2_demo.py pattern [VERIFIED: codebase]
"""Eval script docstring."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import torch
from core.config import TrainConfig
from core.dispatcher import get_method

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("config", type=str, help="Path to YAML config")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def load_model(config_path: str, ckpt_path: str, device: str):
    """Load config + checkpoint -> frozen backbone on device."""
    cfg = TrainConfig.model_validate(yaml.safe_load(open(config_path)))
    # Import methods to register them in dispatcher
    import methods  # noqa: F401
    MethodClass = get_method(cfg.method)
    model = MethodClass.load_from_checkpoint(ckpt_path, cfg=cfg)
    model.eval()
    model.to(device)
    return model, cfg

def main():
    args = get_args()
    model, cfg = load_model(args.config, args.ckpt, args.device)
    # ... script-specific logic ...

if __name__ == "__main__":
    main()
```

### Pattern 2: KNNCallback (Lightning Callback)
**What:** Callback that runs k-NN evaluation at configurable epoch intervals.
**When to use:** During training via `callbacks=[KNNCallback(cfg.eval.knn)]`.
**Example:**
```python
# Source: Lightning Callback API + DINO/MoCo k-NN protocol [ASSUMED]
class KNNCallback(L.Callback):
    def __init__(self, knn_config: KNNConfig):
        self.k = knn_config.k
        self.temperature = knn_config.temperature
        self.every_n_epochs = knn_config.every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.every_n_epochs == 0:
            # Only run at end of training
            if trainer.current_epoch != trainer.max_epochs - 1:
                return
        elif (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        val_loader = trainer.datamodule.val_dataloader()
        if val_loader is None:
            return  # Guard: no val split available

        # Extract features, run k-NN, log result
        acc = self._run_knn(pl_module, val_loader, trainer)
        pl_module.log("eval/knn_acc", acc, prog_bar=True)
```

### Pattern 3: Feature Extraction + Caching (Linear Probe)
**What:** Extract backbone features once, cache to disk, train linear head on cached tensors.
**When to use:** Linear probe (D-04 cache pattern).
**Example:**
```python
# Source: D-04 decision + standard practice [VERIFIED: CONTEXT.md]
def extract_and_cache(model, dataloader, cache_dir: Path, device: str):
    """Extract features if cache doesn't exist, else load from cache."""
    feat_path = cache_dir / "features_train.pt"
    label_path = cache_dir / "labels_train.pt"
    if feat_path.exists() and label_path.exists():
        return torch.load(feat_path), torch.load(label_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            feats = model.backbone(imgs.to(device))
            features.append(feats.cpu())
            labels.append(lbls)
    features = torch.cat(features)
    labels = torch.cat(labels)
    # L2 normalize
    features = torch.nn.functional.normalize(features, dim=1)
    torch.save(features, feat_path)
    torch.save(labels, label_path)
    return features, labels
```

### Pattern 4: Checkpoint Loading via Dispatcher
**What:** Load any SSL method checkpoint using `get_method()` + Lightning's `load_from_checkpoint`.
**When to use:** All eval scripts that need to reconstruct a trained model.
**Key detail:** Lightning's `load_from_checkpoint` requires the class. The config YAML contains `method:` which maps to the class via the dispatcher registry. Must `import methods` first to trigger registration. [VERIFIED: core/dispatcher.py]
```python
import methods  # triggers register_method() calls in all method __init__.py files
from core.dispatcher import get_method

MethodClass = get_method(cfg.method)
model = MethodClass.load_from_checkpoint(ckpt_path, cfg=cfg)
```

### Anti-Patterns to Avoid
- **Hand-rolling k-NN for large datasets:** Use FAISS `IndexFlatIP` for >100K samples; brute-force torch matmul is fine below that threshold.
- **Re-defining config schemas:** All eval configs already exist in `core/config.py`. Scripts must use `cfg.eval.knn`, `cfg.eval.linear_probe`, etc. Adding fields to existing schemas requires updating the Pydantic model (which has `extra='forbid'`).
- **Weight decay on linear probe head:** EVAL-02 explicitly requires `weight_decay=0.0`. Any nonzero value suppresses accuracy when backbone is frozen.
- **Forgetting `import methods`:** Without it, the dispatcher registry is empty and `get_method()` will raise `ValueError`. Every eval script must import the methods package before using the dispatcher.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| k-NN similarity search | Custom pairwise distance | `faiss.IndexFlatIP(dim)` | Handles millions of vectors efficiently; exact inner product search [CITED: faiss.ai] |
| t-SNE | Custom implementation | `sklearn.manifold.TSNE` | Numerically stable, supports cosine metric, PCA init [VERIFIED: sklearn 1.7.1 installed] |
| UMAP | Custom implementation | `umap.UMAP` | Standard implementation with reproducible `random_state` [VERIFIED: pip index] |
| CAM visualization | Manual gradient hooks | `grad-cam` library (`EigenCAM`, `GradCAM`) | Handles ViT reshape, supports multiple CAM methods, maintained [VERIFIED: pypi.org v1.5.5] |
| PCA pre-reduction | Manual SVD | `sklearn.decomposition.PCA` | Already available, standard preprocessing before t-SNE [VERIFIED: sklearn installed] |
| Linear LR schedule | Manual step decay | `torch.optim.lr_scheduler.MultiStepLR` | Built into PyTorch, matches EVAL-02 spec [VERIFIED: torch 2.10.0] |

**Key insight:** The eval suite is a collection of well-established ML evaluation techniques. Every component has a standard library implementation. The value is in correct integration with the project's config system and checkpoint format, not in novel algorithms.

## Common Pitfalls

### Pitfall 1: FAISS Requires Normalized Vectors for Cosine Similarity
**What goes wrong:** `IndexFlatIP` computes raw inner product. If vectors are not L2-normalized, results are not cosine similarity.
**Why it happens:** FAISS has no built-in cosine metric; you must normalize before adding to the index.
**How to avoid:** Always `F.normalize(features, dim=1)` before `index.add(features.numpy())`. The KNNCallback and linear probe both operate on L2-normalized backbone features.
**Warning signs:** k-NN accuracy much lower than expected; nearest neighbors are wrong.

### Pitfall 2: scikit-learn TSNE `learning_rate='auto'` Requires sklearn >= 1.2
**What goes wrong:** Older sklearn versions don't support `learning_rate='auto'`.
**Why it happens:** `'auto'` was added in sklearn 1.2 (Dec 2022).
**How to avoid:** Project has sklearn 1.7.1 -- this is safe. [VERIFIED: pip show scikit-learn]
**Warning signs:** `TypeError` on TSNE construction.

### Pitfall 3: pytorch-grad-cam ViT Reshape Transform
**What goes wrong:** EigenCAM/GradCAM on ViT produces 1D activations (sequence of patches) instead of 2D spatial maps.
**Why it happens:** ViT output at target layer is `[B, N_patches+1, D]` (includes CLS token), not `[B, C, H, W]`.
**How to avoid:** Use `reshape_transform` that removes CLS token and reshapes to `[B, D, H, W]` where `H = W = sqrt(N_patches)`. The grad-cam library's ViT tutorial shows this pattern. [CITED: github.com/jacobgil/pytorch-grad-cam]
**Warning signs:** Error about tensor shape mismatch in CAM computation.

### Pitfall 4: Linear Probe Weight Decay Must Be 0.0
**What goes wrong:** Adding weight decay to the linear head when the backbone is frozen suppresses accuracy.
**Why it happens:** With frozen features, regularizing the only learnable layer limits expressiveness unnecessarily.
**How to avoid:** Assert `weight_decay == 0.0` in the SGD optimizer for the linear head. This is in EVAL-02 spec.
**Warning signs:** Linear probe accuracy 2-5% below expected.

### Pitfall 5: val_dataloader() Returns None When No val/ Directory Exists
**What goes wrong:** `KNNCallback` crashes with `TypeError: 'NoneType' object is not iterable`.
**Why it happens:** `SSLDataModule.val_dataloader()` returns `None` when `self.val_dataset is None` (no `val/` subdirectory). [VERIFIED: core/data.py line 382-384]
**How to avoid:** Guard with `if val_loader is None: return` at the start of `on_validation_epoch_end`. D-02 explicitly requires this.
**Warning signs:** Crash during training when data directory lacks `val/` subdirectory.

### Pitfall 6: EigenCAM Does Not Need a Target/Class -- GradCAM Does
**What goes wrong:** Passing `targets=None` to GradCAM raises an error; passing targets to EigenCAM is ignored.
**Why it happens:** EigenCAM uses PCA on activations (gradient-free); GradCAM needs a differentiable target to compute gradients. For SSL models without a classifier, EigenCAM is the correct default.
**How to avoid:** Default to EigenCAM for SSL checkpoints. Only switch to GradCAM when a classifier is present (e.g., after fine-tuning). [CITED: github.com/jacobgil/pytorch-grad-cam]
**Warning signs:** Error about missing targets in CAM computation.

## Code Examples

### k-NN Weighted Voting (DINO/MoCo v3 Protocol)
```python
# Source: DINO paper protocol + FAISS docs [ASSUMED]
import faiss
import torch
import torch.nn.functional as F

def knn_predict(
    train_features: torch.Tensor,  # [N, D] L2-normalized
    train_labels: torch.Tensor,    # [N] integer class labels
    test_features: torch.Tensor,   # [M, D] L2-normalized
    k: int = 200,
    temperature: float = 0.07,
    num_classes: int = 10,
) -> float:
    """Weighted k-NN classification with temperature scaling."""
    n_train = train_features.shape[0]

    if n_train > 100_000:
        # FAISS path for large datasets
        dim = train_features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(train_features.numpy())
        similarities, indices = index.search(test_features.numpy(), k)
        similarities = torch.from_numpy(similarities)
        indices = torch.from_numpy(indices)
    else:
        # Brute-force path
        similarities = test_features @ train_features.T  # [M, N]
        similarities, indices = similarities.topk(k, dim=1)

    # Temperature-scaled weighted voting
    weights = (similarities / temperature).exp()  # [M, k]
    neighbor_labels = train_labels[indices]         # [M, k]

    # Accumulate votes per class
    votes = torch.zeros(test_features.shape[0], num_classes)
    votes.scatter_add_(1, neighbor_labels.long(), weights)

    predicted = votes.argmax(dim=1)
    correct = (predicted == test_features_labels).sum().item()
    return correct / test_features.shape[0]
```

### CAM Target Layer Selection (Architecture-Aware)
```python
# Source: pytorch-grad-cam docs + EVAL-06 spec [CITED: github.com/jacobgil/pytorch-grad-cam]
from pytorch_grad_cam import EigenCAM, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_target_layer(backbone, backbone_name: str):
    """Select target layer based on architecture."""
    if "resnet" in backbone_name.lower():
        # ResNet: last block of layer4
        return [backbone.layer4[-1]]
    elif "vit" in backbone_name.lower():
        # ViT: last transformer block's LayerNorm
        return [backbone.blocks[-1].norm1]
    else:
        raise ValueError(f"Unsupported backbone for CAM: {backbone_name}")

def vit_reshape_transform(tensor, height=14, width=14):
    """Reshape ViT [B, N+1, D] -> [B, D, H, W] for CAM."""
    # Remove CLS token, reshape patches to spatial grid
    result = tensor[:, 1:, :].reshape(
        tensor.size(0), height, width, tensor.size(2)
    )
    # [B, H, W, D] -> [B, D, H, W]
    return result.permute(0, 3, 1, 2)
```

### Linear Probe Module
```python
# Source: EVAL-02 spec + Lightning pattern [VERIFIED: core/base.py, core/config.py]
import lightning as L
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

class LinearProbeModule(L.LightningModule):
    def __init__(self, feat_dim: int, num_classes: int, cfg):
        super().__init__()
        self.linear = nn.Linear(feat_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.cfg = cfg.eval.linear_probe  # LinearProbeConfig

    def training_step(self, batch, batch_idx):
        features, labels = batch
        logits = self.linear(features)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean()
        self.log("train/loss", loss)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # weight_decay=0.0 is CRITICAL (EVAL-02)
        optimizer = SGD(self.linear.parameters(), lr=self.cfg.lr, weight_decay=0.0)
        scheduler = MultiStepLR(optimizer, milestones=self.cfg.milestones)
        return [optimizer], [scheduler]
```

### Fine-tune Module (Separate LR Groups)
```python
# Source: EVAL-05 spec [VERIFIED: core/config.py FinetuneConfig]
class FinetuneModule(L.LightningModule):
    def __init__(self, backbone, feat_dim, num_classes, cfg):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feat_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.ft_cfg = cfg.eval.finetune  # FinetuneConfig

    def configure_optimizers(self):
        param_groups = [
            {"params": self.backbone.parameters(), "lr": self.ft_cfg.backbone_lr},
            {"params": self.head.parameters(), "lr": self.ft_cfg.head_lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        # Warmup-cosine scheduler
        ...
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def train(self, mode=True):
        """Override to keep BN in eval mode when freeze_bn=True."""
        super().train(mode)
        if self.ft_cfg.freeze_bn and mode:
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()
        return self
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| sklearn NearestNeighbors for k-NN | FAISS IndexFlatIP | Standard since ~2019 | 10-100x faster for >100K samples |
| Manual gradient hooks for CAM | pytorch-grad-cam library | Mature since v1.3+ | Handles ViT, multiple methods, tested |
| Fixed learning_rate in t-SNE | `learning_rate='auto'` | sklearn 1.2 (Dec 2022) | Better default, avoids bad embeddings |
| Separate eval config files | Unified YAML with `eval:` key | This project's design | One file per experiment |

**Deprecated/outdated:**
- `sklearn.manifold.TSNE(learning_rate=200)`: Use `'auto'` instead (project has sklearn 1.7.1)
- `pytorch-grad-cam` < 1.4: Missing EigenCAM support for newer architectures

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Lightning `load_from_checkpoint` accepts `cfg=cfg` as a keyword arg for BaseSSLModule subclasses | Pattern 4 | Checkpoint loading fails; may need `strict=False` or different kwarg name |
| A2 | k-NN weighted voting protocol matches DINO paper exactly (temperature-scaled softmax over similarities) | Code Examples | k-NN accuracy slightly different from reference; not a correctness issue |
| A3 | ViT backbone in timm exposes `backbone.blocks[-1].norm1` for CAM target layer | Code Examples | Need to inspect actual timm ViT attribute names at implementation time |
| A4 | `faiss-cpu` numpy array requirements (float32, C-contiguous) | Pitfall 1 | Need `.astype(np.float32)` and `.contiguous()` before FAISS calls |

## Open Questions

1. **Checkpoint kwarg name for `load_from_checkpoint`**
   - What we know: Lightning's `load_from_checkpoint` passes extra kwargs to `__init__`. BaseSSLModule takes `cfg: TrainConfig`.
   - What's unclear: Whether the checkpoint stores `cfg` in `hparams` automatically or needs explicit passing.
   - Recommendation: Test at implementation time; if `cfg` is saved via `self.save_hyperparameters()`, it may auto-restore. Otherwise, pass explicitly.

2. **FAISS numpy dtype requirement**
   - What we know: FAISS requires float32 numpy arrays.
   - What's unclear: Whether `torch.Tensor.numpy()` from float32 tensors is always C-contiguous.
   - Recommendation: Always call `np.ascontiguousarray(features.numpy().astype(np.float32))` before FAISS.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| scikit-learn | t-SNE, PCA | Yes | 1.7.1 | -- |
| matplotlib | All visualization scripts | Yes | 3.10.3 | -- |
| faiss-cpu | KNNCallback (>100K path) | No | -- | Brute-force torch matmul for small datasets; must install for full feature |
| umap-learn | umap_vis.py | No | -- | Must install; no fallback |
| grad-cam | cam_vis.py | No | -- | Must install; no fallback |

**Missing dependencies with no fallback:**
- `umap-learn` and `grad-cam` must be installed before implementation. Without them, UMAP and CAM scripts cannot function.

**Missing dependencies with fallback:**
- `faiss-cpu`: The brute-force torch path works for datasets <100K. FAISS is only needed for the scale path. Must install for completeness.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.4.1 |
| Config file | None (uses pytest defaults) |
| Quick run command | `pytest tests/test_eval_knn.py tests/test_eval_linear_probe.py tests/test_eval_tsne.py tests/test_eval_umap.py tests/test_eval_finetune.py tests/test_eval_cam.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-01 | KNNCallback logs eval/knn_acc at correct intervals | unit | `pytest tests/test_eval_knn.py -x` | No -- Wave 0 |
| EVAL-02 | LinearProbeModule trains with weight_decay=0.0, MultiStepLR | unit | `pytest tests/test_eval_linear_probe.py -x` | No -- Wave 0 |
| EVAL-03 | t-SNE produces 3 PNGs with perplexity in filename | unit | `pytest tests/test_eval_tsne.py -x` | No -- Wave 0 |
| EVAL-04 | UMAP produces PNG, prints torchdr note for >50K | unit | `pytest tests/test_eval_umap.py -x` | No -- Wave 0 |
| EVAL-05 | FinetuneModule uses 2 LR groups, freeze_bn works | unit | `pytest tests/test_eval_finetune.py -x` | No -- Wave 0 |
| EVAL-06 | CAM vis uses EigenCAM default, GradCAM with classifier | unit | `pytest tests/test_eval_cam.py -x` | No -- Wave 0 |
| D-05 | Integration test: synthetic ckpt + all eval scripts | integration | `pytest tests/test_eval_integration.py -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_eval_*.py -x --timeout=60`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_eval_knn.py` -- covers EVAL-01
- [ ] `tests/test_eval_linear_probe.py` -- covers EVAL-02
- [ ] `tests/test_eval_tsne.py` -- covers EVAL-03
- [ ] `tests/test_eval_umap.py` -- covers EVAL-04
- [ ] `tests/test_eval_finetune.py` -- covers EVAL-05
- [ ] `tests/test_eval_cam.py` -- covers EVAL-06
- [ ] `tests/test_eval_integration.py` -- covers D-05 integration test
- [ ] Install new dependencies: `pip install faiss-cpu umap-learn grad-cam`

## Security Domain

Security is minimal for this phase -- eval scripts are offline tools that process local checkpoints and data. No network access, no user input beyond CLI args, no secrets.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | N/A |
| V3 Session Management | No | N/A |
| V4 Access Control | No | N/A |
| V5 Input Validation | Minimal | argparse validates CLI args; Pydantic validates YAML config (`extra='forbid'`) |
| V6 Cryptography | No | N/A |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malicious pickle in checkpoint | Tampering | Lightning loads via `torch.load` with `weights_only` when available; accept risk for tutorial repo |
| Path traversal via --ckpt | Tampering | Use `pathlib.Path` for path resolution; tutorial context, not production |

## Sources

### Primary (HIGH confidence)
- `core/config.py` lines 153-261 -- All eval sub-configs and TrainConfig verified [VERIFIED: codebase]
- `core/data.py` lines 333-390 -- SSLDataModule.val_dataloader() returns None guard [VERIFIED: codebase]
- `core/dispatcher.py` -- get_method() and register_method() API [VERIFIED: codebase]
- `eval/dinov2_demo.py` -- Reference eval script structure [VERIFIED: codebase]
- `tests/conftest.py` -- tmp_imagefolder fixture pattern [VERIFIED: codebase]
- pip index versions: faiss-cpu 1.13.2, umap-learn 0.5.12 [VERIFIED: pip index]
- PyPI: grad-cam 1.5.5 [VERIFIED: pypi.org/project/grad-cam/]

### Secondary (MEDIUM confidence)
- [pytorch-grad-cam GitHub](https://github.com/jacobgil/pytorch-grad-cam) -- EigenCAM API, ViT reshape transform
- [FAISS wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) -- IndexFlatIP usage

### Tertiary (LOW confidence)
- k-NN weighted voting protocol details (temperature scaling) -- based on training knowledge of DINO paper

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all packages verified via pip index/show, configs verified in codebase
- Architecture: HIGH -- follows established project patterns (dinov2_demo.py, conftest.py fixtures)
- Pitfalls: HIGH -- derived from verified code (val_dataloader None guard, config extra='forbid') and documented requirements (weight_decay=0.0)
- CAM integration: MEDIUM -- pytorch-grad-cam API details from GitHub, not verified via installed package

**Research date:** 2026-04-10
**Valid until:** 2026-05-10 (stable libraries, no rapid changes expected)
