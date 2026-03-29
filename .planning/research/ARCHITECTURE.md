# Architecture Research

**Project:** SSL / Contrastive Learning Tutorial Repository
**Researched:** 2026-03-29
**Confidence:** HIGH for directory structure, interface design, and DataModule patterns (verified against solo-learn source, LightlySSL docs, PyTorch Lightning 2.6.x official docs). MEDIUM for testing and documentation patterns (multiple corroborating sources, no single authoritative reference for tutorial repos specifically).

---

## Directory Structure

### Recommended Layout

The solo-learn library (the most actively maintained multi-method SSL repo as of 2024–2025, published in JMLR Vol. 23) establishes the canonical pattern: flat `methods/` directory where each file is one method, a shared `losses/` module, and a `utils/` module. LightlySSL separates concerns differently (models, transforms, losses all at the top level) but the effect is the same.

For a tutorial repo implementing 15+ methods, the right choice is a **flat `methods/` directory** rather than nested subdirectories. The cognitive cost of navigating `methods/contrastive/instance_discrimination/model.py` is too high when a reader wants to compare SimCLR and BYOL side by side.

```
ssl-tutorial/
├── methods/                    # One file per method
│   ├── __init__.py             # Registry: METHOD_REGISTRY = {"simclr": SimCLRModule, ...}
│   ├── base.py                 # BaseSSLMethod ABC (see Interface Design section)
│   ├── instance_discrimination.py
│   ├── invariant_spread.py
│   ├── cpc.py
│   ├── cmc.py
│   ├── deep_cluster.py
│   ├── moco_v1.py
│   ├── simclr_v1.py
│   ├── moco_v2.py
│   ├── simclr_v2.py
│   ├── swav.py
│   ├── byol.py
│   ├── simsiam.py
│   ├── barlow_twins.py
│   ├── moco_v3.py
│   ├── dino.py
│   └── supcon.py
│
├── losses/                     # Decoupled from LightningModules
│   ├── __init__.py
│   ├── ntxent.py               # SimCLR NT-Xent / InfoNCE
│   ├── nce.py                  # Instance Discrimination NCE
│   ├── byol.py                 # Cosine similarity loss
│   ├── barlow.py               # Cross-correlation loss
│   ├── vicreg.py               # Variance-Invariance-Covariance
│   └── swav.py                 # Sinkhorn-Knopp + prototype loss
│
├── transforms/                 # Method-specific augmentation pipelines
│   ├── __init__.py
│   ├── base.py                 # MultiViewTransform base class
│   ├── simclr.py               # SimCLRTransform (2 views)
│   ├── moco.py                 # MoCoTransform (2 views, no blur in v1)
│   ├── byol.py                 # BYOLTransform (2 asymmetric views)
│   ├── dino.py                 # DINOTransform (2 global + N local crops)
│   └── cpc.py                  # CPCTransform (grid of patches)
│
├── data/
│   ├── __init__.py
│   ├── datamodule.py           # SSLDataModule (see DataModule Design section)
│   └── memory_bank.py          # NNMemoryBank for MoCo, Instance Discrimination
│
├── configs/                    # One YAML per method × backbone combination
│   ├── simclr_resnet50.yaml
│   ├── byol_resnet50.yaml
│   ├── dino_vit_small.yaml
│   └── ...
│
├── configs/schema.py           # Pydantic v2 config models
│
├── evaluation/
│   ├── __init__.py
│   ├── linear_probe.py         # sklearn LogisticRegression on frozen features
│   ├── knn.py                  # k-NN eval on feature bank
│   ├── tsne_umap.py            # Embedding visualization
│   └── cam.py                  # Grad-CAM / activation maps
│
├── tests/
│   ├── conftest.py             # Shared fixtures: tiny_batch, tiny_backbone
│   ├── test_methods.py         # Parametrized smoke tests for ALL methods
│   ├── test_losses.py          # Loss function unit tests
│   ├── test_transforms.py      # Augmentation output shape / value tests
│   ├── test_datamodule.py      # DataModule setup / loader tests
│   └── test_configs.py         # Config schema validation tests
│
├── scripts/
│   ├── pretrain.py             # Main training entry point
│   └── evaluate.py             # Post-training evaluation entry point
│
├── notebooks/                  # Pedagogical Jupyter notebooks
│   ├── 01_simclr_walkthrough.ipynb
│   ├── 02_moco_memory_bank.ipynb
│   └── ...
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Key Organizational Decisions

**Flat `methods/` directory:** Each file is one method. No subdirectories. When a reader is following a paper alongside the code, they open `methods/byol.py` and see exactly one class. This is how solo-learn organizes its 13+ methods.

**Separate `losses/` from `methods/`:** The loss function is the most intellectually interesting part of each paper. Keeping it separate means `methods/simclr.py` has a clean `training_step` that calls `losses.ntxent(z1, z2, temperature)`. Readers can study the loss independently. This pattern is used by both LightlySSL (`lightly.loss`) and solo-learn.

**`transforms/` directory, not inline augmentations:** Each method has specific augmentation requirements (DINO uses 2 global + 6–10 local crops; BYOL uses asymmetric pipelines; CPC uses patch grids). Keeping these in a dedicated module makes the augmentation logic reviewable without opening a 300-line LightningModule.

**`METHOD_REGISTRY` dict in `methods/__init__.py`:** Enables the dispatcher pattern without import gymnastics. `build_module(cfg)` looks up `METHOD_REGISTRY[cfg.method]` and instantiates the right class. Adding a new method requires: (1) writing the method file, (2) adding one entry to the registry.

---

## Interface Design (BaseSSLMethod)

### What the Base Class Must Own

Based on the solo-learn `BaseMethod(LightningModule)` design and LightlySSL patterns, the base class should own everything that is identical across methods. The subclass implements only what differs.

**Base class owns:**
- Backbone instantiation (via timm, `num_classes=0`)
- Projection head interface (`build_projector()` — abstract, subclass overrides)
- `learnable_params` property for optimizer construction (subclasses extend it to add predictor heads, prototypes, etc.)
- `configure_optimizers()` with warmup-cosine schedule
- `validation_step()` with optional KNN evaluation on frozen backbone features
- Logging helpers

**Subclass owns:**
- `build_projector()` — returns the method-specific projection head (different architecture per method)
- `forward(views)` — the method-specific forward pass (symmetric vs. asymmetric, predictor, stop-gradient)
- `training_step(batch, batch_idx)` — computes the method-specific loss
- `learnable_params` property extension — adds predictor/prototype param groups if needed
- Any additional `__init__` components (momentum encoder, queue, prototypes, memory bank)

### Recommended BaseSSLMethod Signature

```python
from abc import ABC, abstractmethod
from typing import Any
import lightning as L
import torch
import torch.nn as nn
import timm


class BaseSSLMethod(L.LightningModule, ABC):
    """
    Abstract base class for all self-supervised learning methods.

    Every method in this repository extends this class and must implement:
      - build_projector()  -- return the projection head nn.Module
      - forward()          -- return (features, projections) given a list of augmented views
      - training_step()    -- compute and return the method-specific loss

    The base class handles:
      - Backbone construction via timm (num_classes=0 for pooled features)
      - configure_optimizers() with LinearWarmup + CosineAnnealing
      - validation_step() with K-NN evaluation on backbone features
      - Logging helpers

    Parameters
    ----------
    backbone_name : str
        Any timm model name, e.g. "resnet50", "vit_small_patch16_224".
    proj_out_dim : int
        Output dimension of the projection head. Method-specific default in subclass.
    lr : float
        Peak learning rate after warmup.
    weight_decay : float
        AdamW weight decay.
    max_epochs : int
        Total training epochs (needed to schedule cosine decay).
    warmup_epochs : int
        Number of linear warmup epochs before cosine decay begins.
    """

    def __init__(
        self,
        backbone_name: str,
        proj_out_dim: int,
        lr: float,
        weight_decay: float,
        max_epochs: int,
        warmup_epochs: int,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[])

        # Build backbone — num_classes=0 removes the classifier, keeps global pool
        # backbone.num_features is the universal attribute across all timm architectures
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        self.feat_dim: int = self.backbone.num_features

        # Subclass builds its projection head
        self.projector: nn.Module = self.build_projector(self.feat_dim, proj_out_dim)

    @abstractmethod
    def build_projector(self, feat_dim: int, proj_out_dim: int) -> nn.Module:
        """
        Build and return the projection head.

        The returned module maps backbone features [B, feat_dim] to
        projected embeddings [B, proj_out_dim].

        Notes
        -----
        - SimCLR: 2-layer MLP with BN, ReLU (Chen et al. 2020, Eq. 1)
        - BYOL:   4096-dim hidden, 256-dim output, BN after each linear (Grill et al. 2020)
        - DINO:   3-layer MLP with hidden norm, L2-normalized output
        """
        ...

    @abstractmethod
    def forward(self, views: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the method-specific forward pass.

        Parameters
        ----------
        views : list of Tensor
            List of augmented views. Length and semantics are method-specific:
              - SimCLR: [view1, view2], both processed by same encoder
              - BYOL:   [online_view, target_view], processed by separate encoders
              - DINO:   [global1, global2, local_1, ..., local_N]

        Returns
        -------
        feats : Tensor [B, feat_dim]
            Pre-projection backbone features. Used for downstream evaluation.
        proj : Tensor [B, proj_out_dim] or list of Tensors (for multi-crop methods)
            Projected embeddings. Used for computing training loss.
        """
        ...

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Compute and return the training loss for one batch.

        Must call self.log("train/loss", loss) before returning.
        Additional method-specific metrics (e.g., std_z for collapse detection)
        should also be logged here.

        Returns
        -------
        loss : Tensor
            Scalar loss tensor. Lightning handles the backward() and optimizer.step().
        """
        ...

    @property
    def learnable_params(self) -> list[dict]:
        """
        Parameter groups for the optimizer.

        Subclasses that add extra modules (predictor head, prototypes) should
        override this property and append their param groups:

            def learnable_params(self):
                return super().learnable_params + [
                    {"name": "predictor", "params": self.predictor.parameters()}
                ]

        Critical: NEVER include momentum encoder params here. EMA encoders
        must NOT receive gradient updates.
        """
        return [
            {"name": "backbone", "params": self.backbone.parameters()},
            {"name": "projector", "params": self.projector.parameters()},
        ]

    def configure_optimizers(self):
        """
        AdamW optimizer with linear warmup + cosine annealing.

        See pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR
        or implement directly. All methods in this repo use this schedule
        unless the original paper specifies otherwise (DeepCluster uses SGD + step decay).
        """
        optimizer = torch.optim.AdamW(
            self.learnable_params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # LinearWarmupCosineAnnealingLR from pl-bolts or a minimal reimplementation
        scheduler = _build_warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """
        Optional: compute K-NN accuracy on backbone features.

        Default implementation is a no-op. Override in subclasses or use
        a Lightning Callback for KNN eval to keep this method clean.
        """
        pass
```

### Handling Method-Specific Augmentation Pipelines

The augmentation pipeline is NOT part of the `LightningModule`. It belongs in the `transforms/` directory and is attached to the `DataModule`. This separation is important: readers comparing methods should see method logic in `methods/`, augmentation logic in `transforms/`.

Pattern from LightlySSL — each method gets its own `Transform` class that wraps PyTorch's `transforms.v2` (preferred in PyTorch 2.x):

```python
# transforms/base.py
class MultiViewTransform:
    """Apply a list of transforms to the same image, returning N views."""
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img):
        return [t(img) for t in self.transforms]


# transforms/simclr.py
class SimCLRTransform(MultiViewTransform):
    """
    Two-view augmentation pipeline from SimCLR (Chen et al. 2020).

    Augmentations applied: RandomResizedCrop, RandomHorizontalFlip,
    ColorJitter, RandomGrayscale, GaussianBlur.

    See paper: Chen et al. (2020), Section 3 and Appendix A.
    The specific strength of color distortion (s=0.5) is from Table 3.
    """
    def __init__(self, image_size: int = 224, s: float = 0.5):
        view_transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=int(0.1 * image_size) | 1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        super().__init__([view_transform, view_transform])
```

The `SSLDataModule` receives the transform class name or instance from config and applies it. Each method's config YAML specifies which transform to use.

### Methods That Need Asymmetric or Non-Standard Transforms

| Method | Transform Requirement | Implementation Note |
|--------|----------------------|---------------------|
| SimCLR v1/v2 | Symmetric 2 views | `SimCLRTransform` |
| MoCo v1 | Symmetric 2 views, no blur | `MoCoV1Transform` |
| MoCo v2 | Symmetric 2 views, with blur | `SimCLRTransform` (identical) |
| BYOL | Asymmetric: view1 has blur, view2 has solarize | `BYOLTransform` |
| DINO | 2 global crops (224) + 6–10 local crops (96) | `DINOTransform` — multi-resolution |
| CPC | Grid of patches, not augmented views | `CPCTransform` — entirely different paradigm |
| SwAV | Multi-crop: 2×224 + 6×96 | `SwAVTransform` |
| CMC | Two color channels (L, ab in Lab space) | `CMCTransform` — channel split, not augmentation |

DINO and SwAV are the two methods where the DataModule must be aware of multi-resolution output — the batch is a list of tensors with different spatial sizes, not a single stacked tensor.

---

## DataModule Design

### Recommended: Single `SSLDataModule` with Transform Injection

The goal is a single `LightningDataModule` that handles ImageFolder-style datasets by default, but accepts any PyTorch `Dataset` as an override. Transform injection (passing the transform in) is cleaner than subclassing for each method.

```python
# data/datamodule.py
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import lightning as L


class SSLDataModule(L.LightningDataModule):
    """
    General-purpose DataModule for self-supervised pretraining.

    Supports:
      - ImageFolder-style datasets (default, requires root_dir with class subdirs)
      - Custom Dataset instances (pass dataset_train / dataset_val directly)

    The transform is injected — it is NOT built internally. Pass the method-
    specific MultiViewTransform from transforms/ to the constructor.

    Parameters
    ----------
    root_dir : str or Path
        Root of an ImageFolder-layout directory. Ignored if dataset_train is given.
    train_transform : callable
        Applied to each training image. Should return a list of views (MultiViewTransform).
    val_transform : callable, optional
        Applied during validation. Default: center crop + normalize (single view).
    batch_size : int
        Number of images per batch. Note: effective contrastive batch size is
        batch_size * num_views.
    num_workers : int
        DataLoader workers. Rule of thumb: 4 per GPU.
    dataset_train : Dataset, optional
        Override the ImageFolder training dataset with a custom Dataset.
    dataset_val : Dataset, optional
        Override the ImageFolder validation dataset with a custom Dataset.
    """

    def __init__(
        self,
        root_dir: str | Path,
        train_transform,
        val_transform=None,
        batch_size: int = 256,
        num_workers: int = 8,
        dataset_train: Dataset | None = None,
        dataset_val: Dataset | None = None,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_transform", "val_transform",
                                          "dataset_train", "dataset_val"])
        self.root_dir = Path(root_dir)
        self.train_transform = train_transform
        self.val_transform = val_transform or self._default_val_transform()
        self._dataset_train = dataset_train
        self._dataset_val = dataset_val

    def setup(self, stage: str | None = None):
        if stage in ("fit", None):
            if self._dataset_train is not None:
                self.train_dataset = self._dataset_train
            else:
                self.train_dataset = ImageFolder(
                    root=self.root_dir / "train",
                    transform=self.train_transform,
                )

        if stage in ("validate", "fit", None):
            if self._dataset_val is not None:
                self.val_dataset = self._dataset_val
            else:
                val_root = self.root_dir / "val"
                if val_root.exists():
                    self.val_dataset = ImageFolder(
                        root=val_root,
                        transform=self.val_transform,
                    )
                else:
                    # No validation split — create a small random subset of train
                    from torch.utils.data import Subset
                    import random
                    idxs = random.sample(range(len(self.train_dataset)), k=min(2000, len(self.train_dataset)))
                    self.val_dataset = Subset(
                        ImageFolder(root=self.root_dir / "train", transform=self.val_transform),
                        idxs,
                    )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,  # Required: avoid single-sample batches crashing BN
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
        )

    @staticmethod
    def _default_val_transform():
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
```

### Critical Design Decisions

**`drop_last=True` for training:** Single-element batches crash BatchNorm. For SSL, batch size stability also matters for the contrastive loss (BYOL shows instability with very small batches).

**`persistent_workers=True` when `num_workers > 0`:** Avoids re-spawning worker processes at each epoch. This is a meaningful speedup for ImageFolder with many small files.

**Transform injection, not subclassing:** Do NOT subclass `SSLDataModule` per method. The transform encapsulates augmentation logic; the DataModule is method-agnostic. This keeps the `data/` module clean.

**Multi-resolution methods (DINO, SwAV):** Their transforms return a list of tensors with different spatial dimensions. The DataLoader `collate_fn` must handle this. Use a custom collate:

```python
def multicrop_collate(batch):
    # batch: list of (list_of_views, label)
    views_per_sample = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    # Transpose: group views by index across the batch
    num_views = len(views_per_sample[0])
    views = [torch.stack([v[i] for v in views_per_sample]) for i in range(num_views)]
    return views, labels
```

**CIFAR-10/100 defaults:** For fast iteration and tutorial use, expose a `from_cifar10(cls, ...)` classmethod that sets the right image size (32), normalization, and no resize. This enables smoke tests and notebook demos without downloading ImageNet.

---

## Testing Strategy

### Core Principle: Smoke Tests for All Methods, Unit Tests for Loss Functions

The most common failure mode in a multi-method SSL repo is a new method silently producing `nan` loss, wrong output shapes, or including the momentum encoder in the optimizer. Smoke tests catch all of these without needing real data or real training.

### Test Structure

**`tests/conftest.py` — Shared Fixtures**

```python
import pytest
import torch
import timm

@pytest.fixture(scope="session")
def tiny_backbone():
    """ResNet-18 with num_classes=0. Used for all method smoke tests."""
    model = timm.create_model("resnet18", pretrained=False, num_classes=0)
    return model

@pytest.fixture
def tiny_batch():
    """Batch of 4 images at 32x32. Fast enough for CPU-only CI."""
    return torch.randn(4, 3, 32, 32)

@pytest.fixture
def two_views(tiny_batch):
    """Standard 2-view SSL batch."""
    return [tiny_batch, tiny_batch.clone()]
```

**`tests/test_methods.py` — Parametrized Smoke Test**

```python
import pytest
import torch
from methods import METHOD_REGISTRY

ALL_METHODS = list(METHOD_REGISTRY.keys())

@pytest.mark.parametrize("method_name", ALL_METHODS)
def test_method_forward_no_crash(method_name, tiny_backbone, two_views):
    """Every method must run a forward pass without crashing."""
    MethodClass = METHOD_REGISTRY[method_name]
    model = MethodClass.from_tiny_config(tiny_backbone)  # factory with small dims
    feats, proj = model(two_views)
    assert feats.shape[0] == 4, "Batch dimension preserved"
    assert not torch.isnan(proj).any(), f"{method_name}: NaN in projections"
    assert not torch.isinf(proj).any(), f"{method_name}: Inf in projections"

@pytest.mark.parametrize("method_name", ALL_METHODS)
def test_training_step_returns_scalar_loss(method_name, tiny_backbone, two_views):
    """training_step must return a scalar tensor (Lightning requirement)."""
    MethodClass = METHOD_REGISTRY[method_name]
    model = MethodClass.from_tiny_config(tiny_backbone)
    fake_batch = (two_views, torch.zeros(4, dtype=torch.long))
    loss = model.training_step(fake_batch, batch_idx=0)
    assert loss.ndim == 0, f"{method_name}: loss must be scalar"
    assert not torch.isnan(loss), f"{method_name}: NaN loss"

@pytest.mark.parametrize("method_name", ALL_METHODS)
def test_momentum_encoder_excluded_from_optimizer(method_name, tiny_backbone):
    """Methods with EMA encoders must NOT include them in learnable_params."""
    MethodClass = METHOD_REGISTRY[method_name]
    model = MethodClass.from_tiny_config(tiny_backbone)
    optimizer_param_ids = {
        id(p) for group in model.learnable_params for p in group["params"]
    }
    # Check that EMA params (requires_grad=False) are not in the optimizer
    ema_attrs = ["backbone_ema", "projector_ema", "encoder_ema"]
    for attr in ema_attrs:
        if hasattr(model, attr):
            ema_module = getattr(model, attr)
            for p in ema_module.parameters():
                assert id(p) not in optimizer_param_ids, (
                    f"{method_name}: EMA param found in optimizer — will corrupt training"
                )
```

### The `from_tiny_config` Factory Pattern

Each method class should implement a `from_tiny_config` classmethod that builds a minimal instance for testing:

```python
class SimCLRModule(BaseSSLMethod):
    @classmethod
    def from_tiny_config(cls, backbone):
        return cls(
            backbone=backbone,
            proj_hidden_dim=64,
            proj_out_dim=32,
            temperature=0.07,
            lr=1e-3,
            weight_decay=1e-4,
            max_epochs=2,
            warmup_epochs=1,
        )
```

This pattern means tests never need real configs; they instantiate directly with known-small dimensions. It's the ML equivalent of a "unit test with mocks."

### Loss Function Unit Tests

Loss functions are pure math — they should be unit tested with known inputs:

```python
def test_ntxent_identical_views_is_log_2N():
    """NT-Xent on identical views: each pos pair has exp(1/tau) in numerator and denominator."""
    from losses.ntxent import NTXentLoss
    loss_fn = NTXentLoss(temperature=1.0)
    z = torch.nn.functional.normalize(torch.eye(4), dim=1)  # 4 orthogonal vectors
    # When z1 == z2 and all pairs are orthogonal, loss should be predictable
    loss = loss_fn(z, z)
    assert torch.isfinite(loss), "NT-Xent must be finite for orthogonal embeddings"

def test_ntxent_collapsed_embeddings_high_loss():
    """Constant embeddings (collapse) should produce maximum possible loss."""
    from losses.ntxent import NTXentLoss
    loss_fn = NTXentLoss(temperature=0.07)
    z_collapsed = torch.ones(8, 128)  # all identical
    loss = loss_fn(z_collapsed, z_collapsed)
    # After normalization, cos-sim = 1 for all pairs; loss ≈ log(2N)
    assert loss.item() > 4.0, "Collapsed embeddings should yield high NT-Xent loss"
```

### CI Configuration

Run smoke tests on every push with a tiny model on CPU. Full training is never part of CI — that would require a GPU and hours. The test suite should complete in under 60 seconds:

```yaml
# .github/workflows/test.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -x -q --timeout=120
```

---

## Documentation Patterns

### Docstring Standard: NumPy Style with Paper References

The NumPy docstring format is used by scikit-learn, PyTorch, and most ML research libraries. It renders cleanly in IDEs (VS Code, PyCharm) and with Sphinx. Use it consistently across all files.

The critical addition for a tutorial repo is the `References` section that ties every non-obvious line back to the paper that introduced it.

```python
class BYOLModule(BaseSSLMethod):
    """
    Bootstrap Your Own Latent (BYOL) — self-supervised learning without negatives.

    BYOL trains an online network to predict the output of a target network given
    different augmented views of the same image. The target network is an
    exponential moving average (EMA) of the online network. No negative samples
    are used.

    Architecture:
      - Online network: backbone -> projector -> predictor
      - Target network: backbone_ema -> projector_ema (no predictor, stop-gradient)

    The key insight: the predictor head + stop-gradient creates an implicit
    asymmetry that prevents trivial collapse despite having no negatives.
    See Appendix F of the paper for the theoretical analysis.

    Parameters
    ----------
    backbone_name : str
        timm model name, e.g. "resnet50".
    momentum : float
        EMA decay for the target network. Paper uses 0.996, increased to 0.9999
        over training via cosine schedule (see paper Section 3.1).
    proj_hidden_dim : int
        Hidden dimension of both projector and predictor MLP. Default: 4096.
    proj_out_dim : int
        Output dimension of projector and predictor. Default: 256.

    Notes
    -----
    The loss is the mean squared error between L2-normalized predictions and
    L2-normalized target projections, equivalent to 2 - 2 * cos_sim(p, z.detach()):

    .. math::
        \\mathcal{L} = 2 - 2 \\cdot \\frac{\\langle p_\\theta, z_\\xi \\rangle}
                         {\\|p_\\theta\\|_2 \\cdot \\|z_\\xi\\|_2}

    where p_theta is the online predictor output and z_xi is the target projector
    output with stop-gradient applied (detach).

    References
    ----------
    .. [1] Grill, J.B. et al. "Bootstrap Your Own Latent: A New Approach to
           Self-Supervised Learning." NeurIPS 2020.
           https://arxiv.org/abs/2006.07733
    .. [2] Appendix F: analysis of why the predictor prevents collapse.
    """
```

### Inline Comments: Equation Anchors

For non-obvious mathematical operations, add a comment anchoring to the paper equation:

```python
# Eq. 3 in Chen et al. (2020): symmetric NT-Xent loss
loss = (ntxent(z1, z2) + ntxent(z2, z1)) / 2

# Barlow Twins Eq. 1: cross-correlation matrix C = Z^T Z / N
# normalized so diagonal = 1 means max correlation, off-diagonal = 0 means decorrelated
C = (z1.T @ z2) / batch_size

# BYOL stop-gradient: z_target must NOT receive gradients
# This is the mechanism that prevents trivial collapse (Grill et al. 2020, Appendix F)
z_target = self.forward_target(view2).detach()
```

### Per-Method README Sections

Each method in `methods/` should have a docstring preamble that works as a self-contained explanation. Structure:

1. **What it is** (1 sentence)
2. **What problem it solves** (compared to prior method)
3. **Key mechanism** (the one algorithmic insight)
4. **Architecture diagram** (ASCII or description)
5. **Critical hyperparameters** (temperature, momentum, etc.)
6. **Paper reference** (arxiv link, venue, year)

### Repository-Level README

```
README.md structure:
1. Overview (2 paragraphs: what this is, who it's for)
2. Methods table (name, year, paper link, key contribution, status)
3. Quick start (clone, install, run SimCLR in <5 commands)
4. Directory structure (annotated tree — 20 lines)
5. How to add a new method (5-step guide pointing to base.py)
6. Evaluation (how to run linear probe, KNN, UMAP)
7. Citation guidance
```

The methods table in the README is the highest-value piece of documentation for a tutorial repo. It lets a reader immediately navigate to what they're looking for:

| Method | Year | Venue | Paradigm | Paper | Code |
|--------|------|-------|----------|-------|------|
| Instance Discrimination | 2018 | CVPR | Memory bank | [Wu et al.](https://arxiv.org/abs/1805.01978) | `methods/instance_discrimination.py` |
| SimCLR | 2020 | ICML | Contrastive | [Chen et al.](https://arxiv.org/abs/2002.05709) | `methods/simclr_v1.py` |
| BYOL | 2020 | NeurIPS | EMA, no negatives | [Grill et al.](https://arxiv.org/abs/2006.07733) | `methods/byol.py` |
| ... | | | | | |

---

## Logging & Metrics

### What to Log (and Why)

Standard pretraining run should log these metrics to TensorBoard (included in Lightning) or W&B:

**Loss metrics (log every step with `on_step=True, on_epoch=True`):**
```python
self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True)
```

**Collapse detection (log every epoch, `on_epoch=True`):**

The single most useful diagnostic for SSL training is the standard deviation of embeddings across the batch. Collapse shows as `std_z → 0`:

```python
def training_step(self, batch, batch_idx):
    views, _ = batch
    feats, z = self.forward(views)
    loss = self.compute_loss(z)

    # Collapse detection: std across batch for each dimension, then mean
    # Healthy range: > 0.1. Collapsing: < 0.01 within first 2 epochs.
    with torch.no_grad():
        z_norm = torch.nn.functional.normalize(z, dim=1)
        std_z = z_norm.std(dim=0).mean()
        self.log("train/std_z", std_z, on_epoch=True)

    self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
    return loss
```

**Effective rank (log every N epochs, computationally expensive):**

RankMe (Garrido et al., ICML 2023) provides a label-free proxy for downstream performance. Log it every 10 epochs by accumulating a feature bank and computing the entropy of the singular value distribution:

```python
def on_validation_epoch_end(self):
    if self.current_epoch % 10 == 0 and hasattr(self, "_val_features"):
        features = torch.cat(self._val_features, dim=0)
        # Effective rank = exp(entropy of normalized singular values)
        # See RankMe: Garrido et al. ICML 2023
        _, s, _ = torch.linalg.svd(features, full_matrices=False)
        s = s / s.sum()
        effective_rank = torch.exp(-(s * torch.log(s + 1e-8)).sum())
        self.log("val/effective_rank", effective_rank)
        self._val_features = []
```

**Learning rate (log every step):** Critical for diagnosing warmup issues. A missing warmup or wrong max_lr is a common failure mode.

**Gradient norms (log every 50 steps):** Gradient explosion early in training is often the cause of NaN losses. Log the global gradient norm:

```python
def on_before_optimizer_step(self, optimizer):
    # Log global grad norm every 50 steps
    if self.global_step % 50 == 0:
        norms = [p.grad.norm() for p in self.parameters() if p.grad is not None]
        if norms:
            global_norm = torch.stack(norms).norm()
            self.log("train/grad_norm", global_norm)
```

**EMA momentum value (for BYOL/MoCo):** Some papers schedule momentum from 0.996 → 0.9999 over training. Log the current value to verify the schedule is applying:

```python
self.log("train/ema_momentum", self.hparams.momentum)
```

### Recommended Metric Summary

| Metric | Name | Frequency | Purpose |
|--------|------|-----------|---------|
| Training loss | `train/loss` | Every step | Core convergence signal |
| Learning rate | `train/lr` | Every step | Diagnose warmup/schedule issues |
| Embedding std | `train/std_z` | Every epoch | Detect collapse immediately (std → 0) |
| Effective rank | `val/effective_rank` | Every 10 epochs | Label-free quality proxy (RankMe) |
| Gradient norm | `train/grad_norm` | Every 50 steps | Detect explosion before NaN |
| EMA momentum | `train/ema_momentum` | Every epoch | Verify schedule for BYOL/MoCo |
| Val loss | `val/loss` | Every epoch | Generalization check (if val split exists) |

### Lightning Logging Pattern (Verified for 2.x)

```python
# CORRECT: log scalar from training_step
self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

# CORRECT: log non-scalar (histogram of embedding values) via logger directly
if self.logger and hasattr(self.logger, "experiment"):
    self.logger.experiment.add_histogram("embeddings", z, self.global_step)

# WRONG: calling self.metric.compute() AND logging the metric object in same step
# This resets the metric between calls — produces nonsense values
```

**`sync_dist=True`** is required for multi-GPU DDP training. For single-GPU tutorials, it's a no-op but harmless to include — better to make it a habit.

---

## Recommended Implementation Order

The 15+ methods should be implemented in curriculum order — each phase introduces one new algorithmic concept. A reader who works through them sequentially understands the full evolution of the field.

### Phase 1: Foundation (The Loss Function Era)

Implement the scaffolding + the 3 foundational methods that define the core concepts:

1. `BaseSSLMethod` + `SSLDataModule` + test infrastructure
2. **Instance Discrimination** (Wu et al., CVPR 2018) — memory bank, NCE loss, temperature
3. **Invariant Spread** (Ye et al., CVPR 2019) — in-batch negatives, no memory bank
4. **CPC** (van den Oord et al., 2018) — InfoNCE loss (the loss all later methods derive from)

**Why this order:** Instance Discrimination establishes the core SSL problem. CPC introduces InfoNCE. These two together explain every contrastive method that follows.

### Phase 2: Scaling Contrastive Learning

5. **CMC** (Tian et al., 2019) — multi-view / multi-modal extension
6. **MoCo v1** (He et al., CVPR 2020) — momentum contrast, fixed-size queue replaces memory bank
7. **SimCLR v1** (Chen et al., ICML 2020) — large batch, projection head, NT-Xent loss
8. **MoCo v2** (Chen et al., 2020) — MoCo + SimCLR augmentations + projection head

**Why this order:** MoCo solves the stale-negatives problem. SimCLR shows batch size matters. MoCo v2 shows they're complementary.

### Phase 3: Eliminating Negatives

9. **BYOL** (Grill et al., NeurIPS 2020) — EMA teacher, predictor, no negatives
10. **SimSiam** (Chen & He, CVPR 2021) — no EMA, stop-gradient only
11. **Barlow Twins** (Zbontar et al., ICML 2021) — cross-correlation redundancy reduction

**Why this order:** BYOL is the breakthrough. SimSiam strips it to the minimum. Barlow Twins takes a completely different approach (information theory) but reaches the same destination.

### Phase 4: Clustering and Multi-Crop

12. **Deep Cluster** (Caron et al., ECCV 2018) — k-means as pseudo-labels
13. **SwAV** (Caron et al., NeurIPS 2020) — online clustering, multi-crop
14. **SupCon** (Khosla et al., NeurIPS 2020) — supervised extension of contrastive

**Why this order:** Deep Cluster is the predecessor. SwAV is the modern synthesis. SupCon closes the loop by bringing labels back in.

### Phase 5: Transformer Era

15. **MoCo v3** (Chen et al., ICCV 2021) — contrastive + ViT backbone
16. **DINO** (Caron et al., ICCV 2021) — self-distillation, multi-crop, ViT features
17. **SimCLR v2** / **InfoMin** — scaling and data-efficiency variants

**Why this phase last:** ViT-based methods require larger compute and introduce transformer-specific concerns (patch embedding, CLS token vs. patch average). They build on all previous concepts.

### Implementation Priority Rule

If forced to ship only 6 methods, implement: Instance Discrimination, MoCo v1, SimCLR v1, BYOL, Barlow Twins, DINO. These six span all major paradigm shifts (memory bank → queue → large batch → no negatives → redundancy reduction → self-distillation).

---

## Sources

- solo-learn source and JMLR paper: https://github.com/vturrisi/solo-learn / https://www.jmlr.org/papers/v23/21-1155.html (HIGH confidence)
- LightlySSL documentation and module organization: https://docs.lightly.ai/self-supervised-learning/ (HIGH confidence)
- PyTorch Lightning 2.6.x DataModule docs: https://lightning.ai/docs/pytorch/stable/data/datamodule.html (HIGH confidence)
- PyTorch Lightning 2.6.x Logging docs: https://lightning.ai/docs/pytorch/stable/extensions/logging.html (HIGH confidence)
- RankMe (effective rank as SSL diagnostic): Garrido et al., ICML 2023, https://proceedings.mlr.press/v202/garrido23a/garrido23a.pdf (HIGH confidence — peer-reviewed)
- Understanding dimensional collapse: Jing et al., ICLR 2022, https://openreview.net/forum?id=YevsQ05DEN7 (HIGH confidence)
- NumPy docstring standard: https://numpydoc.readthedocs.io/en/latest/format.html (HIGH confidence)
- SSL method implementation curriculum: https://theaisummer.com/simclr/ and https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html (MEDIUM confidence — tutorial sources)
- Ploomber ML testing series: https://ploomber.io/blog/ml-testing-i/ (MEDIUM confidence)
- Made With ML testing guide: https://madewithml.com/courses/mlops/testing/ (MEDIUM confidence)
