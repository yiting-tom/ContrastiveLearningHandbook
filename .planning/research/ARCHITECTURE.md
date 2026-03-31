# Architecture Research

**Project:** SSL / Contrastive Learning Tutorial Repository
**Researched:** 2026-03-29
**Confidence:** HIGH — patterns verified against solo-learn source, LightlySSL docs, official PyTorch Lightning 2.6.x docs, and VICReg/Barlow Twins literature.

---

## Directory Structure

### Recommendation: Flat `methods/` with One File Per Method

The solo-learn pattern is the gold standard for this kind of repo. After inspecting both solo-learn (`solo/methods/` — 22 files) and LightlySSL (`lightly/models/` — 13 files), the correct layout for 15+ methods is a flat directory with era-grouped comments in the registry, not subdirectories.

```
ml_topic_contrastive_learning/
├── methods/
│   ├── __init__.py               # METHODS dict registry; explicit exports
│   ├── base.py                   # BaseSSLMethod (LightningModule)
│   ├── base_momentum.py          # BaseMomentumMethod (adds EMA encoder)
│   │
│   ├── # --- Era 1: Proxy Tasks (2018-2019) ---
│   ├── instance_discrimination.py
│   ├── invariant_spread.py
│   ├── cpc.py
│   ├── amdim.py
│   │
│   ├── # --- Era 2: Contrastive (2019-2020) ---
│   ├── cmc.py
│   ├── moco_v1.py
│   ├── simclr.py
│   ├── moco_v2.py
│   │
│   ├── # --- Era 3: No-Negatives (2020-2021) ---
│   ├── byol.py
│   ├── simsiam.py
│   ├── swav.py
│   ├── barlow_twins.py
│   │
│   ├── # --- Era 4: Regularization-Based (2021-2022) ---
│   ├── vicreg.py
│   ├── moco_v3.py
│   ├── dino.py
│   │
│   └── # --- Era 5: Masked / ViT-Native (2021-2022) ---
│       ├── mae.py
│       └── ibot.py
│
├── losses/
│   ├── __init__.py
│   ├── ntxent.py                 # NT-Xent / InfoNCE (SimCLR, MoCo, NNCLR)
│   ├── byol_loss.py              # negative cosine similarity
│   ├── barlow_loss.py            # cross-correlation matrix loss
│   ├── vicreg_loss.py            # variance + invariance + covariance
│   └── swav_loss.py              # swapped assignments + Sinkhorn-Knopp
│
├── transforms/
│   ├── __init__.py
│   ├── base_transform.py         # two-view: RandomResizedCrop + ColorJitter + GaussianBlur
│   ├── simclr_transform.py       # strong color jitter per Chen 2020
│   ├── byol_transform.py         # asymmetric: different strengths for view1/view2
│   ├── dino_transform.py         # multi-crop: 2 global (224) + N local (96)
│   └── mae_transform.py          # minimal: just center crop, normalization
│
├── data/
│   ├── __init__.py
│   ├── datamodule.py             # SSLDataModule (LightningDataModule)
│   └── multi_view_dataset.py     # wraps any Dataset; applies multi-view transform
│
├── eval/
│   ├── __init__.py
│   ├── linear_probe.py           # offline linear evaluation script
│   ├── knn_eval.py               # online KNNEvalCallback (Lightning Callback)
│   └── online_linear.py          # optional online linear head (solo-learn style)
│
├── configs/
│   ├── schema.py                 # Pydantic v2 config models
│   ├── simclr_resnet50.yaml
│   ├── byol_resnet50.yaml
│   └── ...                       # one YAML per method
│
├── tests/
│   ├── conftest.py               # shared fixtures: tiny_batch, tiny_model, gen_trainer
│   ├── test_simclr.py
│   ├── test_byol.py
│   └── ...                       # one test file per method
│
├── notebooks/
│   └── 01_simclr_walkthrough.ipynb
│
├── scripts/
│   ├── pretrain.py               # main entry point
│   ├── linear_eval.py
│   └── knn_eval.py
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Key Structural Decisions

**Why one file per method, not grouped in subdirectories.** Files are searched and opened individually. A flat list is easier to navigate and link from documentation. Era grouping via comments in `__init__.py` provides logical structure without adding import path complexity.

**Why `losses/` is separate from `methods/`.** NT-Xent is used by SimCLR, MoCo v1/v2, and NNCLR. A separate `losses/` directory makes each loss independently testable and avoids circular imports. LightlySSL uses the same separation pattern.

**Why `transforms/` is separate.** DINO's multi-crop pipeline (2 global views at 224px + N local views at 96px) is fundamentally different from SimCLR's two-view pipeline. Keeping transforms separate makes it clear which augmentation belongs to which method and allows `SSLDataModule` to accept any callable transform.

**`__init__.py` registry pattern (from solo-learn source, HIGH confidence):**

```python
# methods/__init__.py
from .simclr import SimCLR
from .byol import BYOL
from .moco_v2 import MoCoV2
# ... all methods ...

METHODS = {
    "simclr": SimCLR,
    "byol": BYOL,
    "moco_v2": MoCoV2,
    # ...
}

__all__ = ["METHODS", "SimCLR", "BYOL", "MoCoV2", ...]
```

Solo-learn uses explicit manual registration (not filesystem scanning or `importlib` magic). This gives full control, clear visibility in `__init__.py`, and avoids fragile dynamic import paths. The `METHODS` dict enables string-based dispatch from a config file.

---

## Interface Design (BaseSSLMethod)

### Two-Level Class Hierarchy

```
lightning.LightningModule
    └── BaseSSLMethod          (methods/base.py)
            └── BaseMomentumMethod   (methods/base_momentum.py)
```

Methods without a momentum encoder inherit `BaseSSLMethod` directly: SimCLR, SimSiam, Barlow Twins, VICReg, SwAV, MAE, Instance Discrimination, Invariant Spread, CPC.

Methods with a momentum encoder inherit `BaseMomentumMethod`: MoCo v1/v2/v3, BYOL, DINO, iBOT.

This matches the solo-learn hierarchy exactly (`BaseMethod` / `BaseMomentumMethod`), verified by source inspection.

### BaseSSLMethod Contract

```python
# methods/base.py
from __future__ import annotations
from abc import abstractmethod
import torch
import torch.nn as nn
import lightning as L


class BaseSSLMethod(L.LightningModule):
    """
    Abstract base class for all self-supervised learning methods.

    Subclasses MUST implement:
        build_projector(feat_dim)  -- returns the projection MLP
        forward(x)                 -- single-view: returns dict with "h" and "z"
        ssl_loss(outputs, batch)   -- computes the training loss

    Subclasses MAY override:
        learnable_params           -- inject extra optimizer param groups
        default_transform()        -- return method-specific augmentation
        on_train_batch_end(...)    -- momentum update, queue update
    """

    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        lr: float,
        weight_decay: float,
        max_epochs: int,
        warmup_epochs: int = 10,
        num_classes: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        self.projector = self.build_projector(feat_dim)
        if num_classes > 0:
            # Online linear head for monitoring (not used in ssl_loss)
            self.online_classifier = nn.Linear(feat_dim, num_classes)

    @abstractmethod
    def build_projector(self, feat_dim: int) -> nn.Module:
        """Return the projection head (MLP from feat_dim to embedding space)."""
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Single-view forward pass. Returns a dict with at minimum:
            "h": backbone features  [B, feat_dim]
            "z": projected embeddings  [B, proj_dim]
        Methods like BYOL that have a predictor also return "p" here.
        Multi-view collation for training is handled in training_step.
        """
        ...

    @abstractmethod
    def ssl_loss(
        self, outputs: dict[str, list[torch.Tensor]], batch: tuple
    ) -> torch.Tensor:
        """Compute the SSL training loss from multi-view forward outputs."""
        ...

    @property
    def learnable_params(self) -> list[dict]:
        """
        Return optimizer parameter groups.
        Subclasses MUST call super().learnable_params and extend the list.
        Never include momentum encoder parameters here.
        """
        params = [
            {"name": "backbone", "params": self.backbone.parameters()},
            {"name": "projector", "params": self.projector.parameters()},
        ]
        if hasattr(self, "online_classifier"):
            params.append(
                {"name": "classifier", "params": self.online_classifier.parameters()}
            )
        return params

    def training_step(self, batch, batch_idx):
        views, labels = batch  # views is a list of tensors, one per augmented view
        outputs: dict[str, list] = {}
        for v in views:
            out = self.forward(v)
            for k, val in out.items():
                outputs.setdefault(k, []).append(val)
        loss = self.ssl_loss(outputs, batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if not hasattr(self, "online_classifier"):
            return
        x, labels = batch[0][0], batch[1]   # first view only, not augmented
        with torch.no_grad():
            h = self.backbone(x)
        logits = self.online_classifier(h)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("val/acc_top1", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.learnable_params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # LinearWarmupCosineAnnealingLR from pl_bolts or custom implementation
        from methods._schedulers import build_warmup_cosine_scheduler
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    @classmethod
    def default_transform(cls):
        """
        Return the canonical augmentation pipeline for this method.
        Used by SSLDataModule when no custom transform is provided.
        Subclasses should override.
        """
        from transforms.base_transform import BaseSSLTransform
        return BaseSSLTransform()
```

### BaseMomentumMethod Extension

```python
# methods/base_momentum.py
import copy
from lightly.models.utils import deactivate_requires_grad, update_momentum
from .base import BaseSSLMethod


class BaseMomentumMethod(BaseSSLMethod):
    """
    Extends BaseSSLMethod with an EMA (momentum) encoder.
    Used by: MoCo v1/v2/v3, BYOL, DINO, iBOT.

    The momentum encoder MUST NOT appear in learnable_params.
    EMA update is placed in on_train_batch_end (after optimizer.step),
    ensuring the EMA is computed on freshly updated online weights.
    """

    def __init__(self, *args, momentum: float = 0.996, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum
        self.backbone_ema = copy.deepcopy(self.backbone)
        self.projector_ema = copy.deepcopy(self.projector)
        deactivate_requires_grad(self.backbone_ema)
        deactivate_requires_grad(self.projector_ema)
        # Do NOT include *_ema in learnable_params — they must never receive gradients.

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # After optimizer.step() has updated online weights
        update_momentum(self.backbone, self.backbone_ema, self.momentum)
        update_momentum(self.projector, self.projector_ema, self.momentum)
```

**Why `on_train_batch_end` and not `training_step`.** The EMA update should see the freshly optimizer-stepped online weights. `on_train_batch_end` is called after `optimizer.step()`, making the update semantically correct. Placing it inside `training_step` before the loss computation would use stale online weights. This is the pattern used by both solo-learn and LightlySSL.

### Per-Method Augmentation Handling

Each method ships its own transform class. The `SSLDataModule` receives a transform callable and passes it to `MultiViewDataset`. The transform is applied per-sample in `__getitem__` and returns a tuple of views.

```
transforms/simclr_transform.py   → SimCLRTransform()   returns (view1, view2)
transforms/byol_transform.py     → BYOLTransform()     returns (view1, view2) with asymmetric strength
transforms/dino_transform.py     → DINOTransform()     returns (global1, global2, local1, ..., localN)
transforms/mae_transform.py      → MAETransform()      returns (x,) — masking happens inside the model
```

The `default_transform()` classmethod on each method subclass returns the correct transform object. This allows using a method without any config file:

```python
model = SimCLR(backbone=..., ...)
dm = SSLDataModule(data_dir="data/", transform=SimCLR.default_transform())
```

---

## DataModule Design

### SSLDataModule Interface

```python
# data/datamodule.py
import lightning as L
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


class SSLDataModule(L.LightningDataModule):
    """
    Flexible LightningDataModule for SSL pretraining.

    Supports three usage modes:
        1. Path to an ImageFolder-style directory  (data_dir="path/to/root")
        2. Named built-in dataset                 (dataset_name="cifar10")
        3. Explicit torch Dataset                 (train_dataset=my_dataset)

    The transform must return a tuple of views: (view1, view2) or more.
    If not provided, defaults to BaseSSLTransform (two-view, standard augmentations).

    Parameters
    ----------
    data_dir : str, optional
        Path to a folder containing train/ and (optionally) val/ subdirectories
        in ImageFolder format: root/class_name/image.jpg.
    dataset_name : str, optional
        One of "cifar10", "cifar100", "stl10". Downloads automatically.
    train_dataset : Dataset, optional
        Explicit Dataset override. Mutually exclusive with data_dir/dataset_name.
    transform : callable, optional
        Multi-view transform. Must return a tuple of tensors.
    batch_size : int
        Per-device batch size. Default 256.
    num_workers : int
        DataLoader workers. Default 8.
    drop_last : bool
        Drop the last incomplete batch. MUST be True for NT-Xent and BatchNorm.
        Default True.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        dataset_name: str | None = None,
        train_dataset: Dataset | None = None,
        transform=None,
        val_transform=None,
        batch_size: int = 256,
        num_workers: int = 8,
        val_split: float = 0.0,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_dataset", "transform", "val_transform"])
        self._explicit_dataset = train_dataset
        self.transform = transform
        self.val_transform = val_transform

    def setup(self, stage: str | None = None):
        from data.multi_view_dataset import MultiViewDataset
        if self._explicit_dataset is not None:
            base = self._explicit_dataset
        elif self.hparams.dataset_name:
            base = self._build_builtin_dataset()
        else:
            base = ImageFolder(root=self.hparams.data_dir)
        self.train_ds = MultiViewDataset(base, self.transform or BaseSSLTransform())
        # Validation: single-view with labels for online kNN/linear monitoring
        self.val_ds = self._build_val_dataset()

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        if not hasattr(self, "val_ds") or self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size * 2,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
        )
```

### MultiViewDataset Wrapper

```python
# data/multi_view_dataset.py
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):
    """
    Wraps any Dataset and applies a multi-view transform in __getitem__.

    The transform must return a tuple of tensors (the views).
    Labels from the underlying dataset are preserved.

    Separating view generation from the LightningModule is critical:
    it allows the same method to be evaluated with different augmentation
    strategies without changing the model code.
    """

    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        views = self.transform(img)     # returns (view1, view2) or more
        return views, label

    def __len__(self):
        return len(self.dataset)
```

### Non-Negotiable DataModule Constraints

**`drop_last=True` is required.** NT-Xent breaks with a partial batch (batch_size=1 has 0 negatives). BatchNorm in the projector produces incorrect statistics with a single-sample batch, causing NaN loss or silent training failure.

**`val_split` vs. separate val directory.** ImageFolder-style datasets typically have `train/` and `val/` at the same root. `SSLDataModule` should check for `{data_dir}/val` first; fall back to a random `val_split` only if absent. STL-10 is the canonical SSL benchmark: 100K unlabeled images for training, 5K labeled for evaluation.

**`persistent_workers=True`** prevents worker process respawn between epochs. Measurable speedup when `num_workers >= 4`.

**Recommended configurations:**

| Scenario | dataset_name | batch_size | num_workers |
|----------|-------------|------------|-------------|
| Laptop smoke test | "cifar10" | 64 | 2 |
| STL-10 SSL | (data_dir) | 256 | 8 |
| ImageNet-100 | (data_dir) | 512 | 16 |
| Custom ImageFolder | (data_dir) | 256 | 8 |

---

## Testing Strategy

### Layered Test Pyramid

For a 15+ method tutorial repo, tests follow four layers:

| Layer | Scope | Lightning Flag | CI Trigger |
|-------|-------|---------------|-----------|
| **Smoke** | 1 batch, no assert on values | `fast_dev_run=True` | Every commit |
| **Shape** | Output tensor shapes correct | None (unit test) | Every commit |
| **Sanity** | Loss finite, no collapse in 2 epochs | `max_epochs=2, limit_train_batches=5` | PR |
| **Integration** | Pretrain → kNN eval runs end-to-end | Full run on tiny data | Nightly |

### Shared Fixtures (conftest.py)

```python
# tests/conftest.py
import pytest
import torch
import timm
import lightning as L


@pytest.fixture(scope="session")
def tiny_backbone():
    """ResNet-18, no classifier head. session-scoped for speed across all test files."""
    backbone = timm.create_model("resnet18", num_classes=0, pretrained=False)
    return backbone, backbone.num_features


@pytest.fixture
def tiny_batch():
    """4 images at 32x32, 2 views each. Sufficient for all methods."""
    B, C, H, W = 4, 3, 32, 32
    view1 = torch.randn(B, C, H, W)
    view2 = torch.randn(B, C, H, W)
    labels = torch.randint(0, 10, (B,))
    return [view1, view2], labels


@pytest.fixture
def fast_trainer():
    """Runs 1 train batch. Pure smoke test — disables all logging and checkpointing."""
    return L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )


@pytest.fixture
def sanity_trainer():
    """Runs 2 epochs x 5 batches. Catches NaN loss and collapse within seconds."""
    return L.Trainer(
        max_epochs=2,
        limit_train_batches=5,
        limit_val_batches=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
```

### Per-Method Test Template

```python
# tests/test_simclr.py
import torch
import pytest
from methods.simclr import SimCLR


@pytest.mark.smoke
def test_simclr_smoke(tiny_backbone, fast_trainer, tmp_path):
    """One forward + backward pass completes without error."""
    backbone, feat_dim = tiny_backbone
    model = SimCLR(
        backbone=backbone, feat_dim=feat_dim, lr=1e-3, weight_decay=1e-6,
        max_epochs=100, temperature=0.07, proj_out_dim=128,
    )
    from data.datamodule import SSLDataModule
    dm = SSLDataModule(dataset_name="cifar10", batch_size=4, num_workers=0,
                       transform=SimCLR.default_transform())
    fast_trainer.fit(model, datamodule=dm)


@pytest.mark.smoke
def test_simclr_output_shapes(tiny_backbone, tiny_batch):
    """forward() returns h and z with correct shapes."""
    backbone, feat_dim = tiny_backbone
    model = SimCLR(backbone=backbone, feat_dim=feat_dim, lr=1e-3, weight_decay=1e-6,
                   max_epochs=100, temperature=0.07, proj_out_dim=128)
    views, _ = tiny_batch
    out = model.forward(views[0])
    assert out["h"].shape == (4, feat_dim)
    assert out["z"].shape == (4, 128)


@pytest.mark.sanity
def test_simclr_loss_finite(tiny_backbone, tiny_batch):
    """Loss is a finite positive scalar."""
    backbone, feat_dim = tiny_backbone
    model = SimCLR(backbone=backbone, feat_dim=feat_dim, lr=1e-3, weight_decay=1e-6,
                   max_epochs=100, temperature=0.07, proj_out_dim=128)
    views, labels = tiny_batch
    outputs = {"z": [model.forward(v)["z"] for v in views]}
    loss = model.ssl_loss(outputs, (views, labels))
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0, "Loss should be positive"


@pytest.mark.sanity
def test_simclr_no_collapse(tiny_backbone, tiny_batch):
    """Embeddings have non-trivial variance (not constant output)."""
    backbone, feat_dim = tiny_backbone
    model = SimCLR(backbone=backbone, feat_dim=feat_dim, lr=1e-3, weight_decay=1e-6,
                   max_epochs=100, temperature=0.07, proj_out_dim=128)
    views, _ = tiny_batch
    with torch.no_grad():
        z = model.forward(views[0])["z"]
    std = z.std(dim=0).mean().item()
    assert std > 1e-4, f"Embeddings appear collapsed: per-dim std = {std:.6f}"
```

### pytest Markers Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
markers = [
    "smoke: fast smoke tests using fast_dev_run=True (always run, < 30s total)",
    "sanity: 2-epoch sanity checks (run on PR, < 5min total)",
    "slow: full integration tests (run nightly)",
]
```

CI pipeline runs `pytest -m smoke` on every push (< 2 minutes on CPU), and `pytest -m smoke or sanity` on PRs.

### What NOT to Assert in Tests

Do not assert that `val/acc_top1` reaches a specific threshold — this requires full training and is not reproducible on CPU. Do not assert exact loss values — they depend on random initialization and batch sampling. Do not write `assert loss < some_threshold` — correct SSL loss values are batch-size-dependent and method-specific. Test structure and finiteness, not magnitude.

---

## Documentation Patterns

### Docstring Standard: NumPy Style with Mandatory References Section

NumPy docstrings are the standard for scientific Python (NumPy, SciPy, scikit-learn, PyTorch). They render correctly with Sphinx + numpydoc and are readable in plain text terminals. For a tutorial repo, every method class docstring must include a **References** section that links to the original paper.

```python
class SimCLR(BaseSSLMethod):
    """
    SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.

    Chen et al. (ICML 2020) showed that contrastive learning with a projection head,
    large batch size, and strong augmentation produces representations competitive
    with supervised baselines. NT-Xent loss maximizes agreement between two augmented
    views of the same image while pushing apart representations of different images.

    Key design choices in this implementation:
    - 3-layer MLP projector (paper uses 2-layer; 3-layer is stronger per SimCLR v2 [2])
    - Temperature τ = 0.07 (paper default)
    - Symmetric loss: loss(z1, z2) + loss(z2, z1)
    - No momentum encoder, no memory bank — batch negatives only

    Parameters
    ----------
    backbone : nn.Module
        Feature extractor. Any timm backbone with num_classes=0.
    feat_dim : int
        Backbone output dimension (use backbone.num_features).
    temperature : float, optional
        NT-Xent temperature τ. Default 0.07. Lower = sharper distribution
        and harder negatives. Typical range: [0.05, 0.2].
    proj_out_dim : int, optional
        Projector output dimension. Default 128. SimCLR v2 uses 256.
    proj_hidden_dim : int, optional
        Projector hidden dimension. Default 2048.

    Notes
    -----
    The projection head is discarded after pretraining. Downstream evaluations
    use backbone features ``h``, not projected ``z``.

    SimCLR requires batch_size >= 256 for sufficient negatives. With
    batch_size=256 you have 510 negatives per anchor (2 * 256 - 2).

    References
    ----------
    .. [1] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).
           A simple framework for contrastive learning of visual representations.
           ICML 2020. https://arxiv.org/abs/2002.05709

    .. [2] Chen, T., Kornblith, S., Sohl-Dickstein, J., & Hinton, G. (2020).
           Big self-supervised models are strong semi-supervised learners.
           NeurIPS 2020. https://arxiv.org/abs/2006.10029

    Examples
    --------
    >>> import timm
    >>> backbone = timm.create_model("resnet50", num_classes=0, pretrained=False)
    >>> model = SimCLR(
    ...     backbone=backbone,
    ...     feat_dim=backbone.num_features,
    ...     temperature=0.07,
    ...     lr=3e-4,
    ...     weight_decay=1e-6,
    ...     max_epochs=200,
    ... )
    """
```

### Inline Comment Pattern: Paper-Equation Anchors

Every non-obvious line in loss functions should cite the paper equation:

```python
def ssl_loss(self, outputs, batch):
    z1, z2 = outputs["z"]  # [B, D] each

    # L2-normalize before cosine similarity (Chen 2020, Sec 2, Eq. 1)
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)

    # Similarity matrix [2B, 2B] -- both views concatenated
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / self.hparams.temperature

    # Mask diagonal (self-similarity is not a valid negative pair)
    B = z1.shape[0]
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))

    # Positive pairs: (i, i+B) and (i+B, i) -- symmetric formulation (Chen 2020, Eq. 2)
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).to(z.device)
    loss = nn.functional.cross_entropy(sim, labels)
    return loss
```

### README / Notebook Method Section Structure

Each method gets a section (or notebook cell group) following this order:

1. **What problem does this method solve?** Plain English, one paragraph, no math.
2. **The core idea.** One diagram or the key equation, cited to the paper.
3. **Key implementation components.** Bulleted list with file links.
4. **Run command.** Copy-pasteable one-liner.
5. **Expected behavior.** What the loss curve looks like, what kNN accuracy is reasonable.

This "theory → implementation → run" structure is the pattern used by the UvA Deep Learning course tutorials (the most cited single-method SSL tutorial in the community).

---

## Logging & Metrics

### What to Log During Pretraining

**Every step (on_step=True):**

| Key | Why |
|-----|-----|
| `train/loss` | Primary signal; confirms training is progressing |
| `train/lr` | Confirms warmup and cosine decay are working |

**Every epoch (on_epoch=True):**

| Key | How to Compute | What It Tells You |
|-----|---------------|------------------|
| `train/embed_std_mean` | `z.std(dim=0).mean()` | Total collapse → approaches 0 |
| `train/embed_std_min` | `z.std(dim=0).min()` | Dimensional collapse → one dim near 0 |
| `train/embed_eff_rank` | Fraction of singular values > 1% of max | Rank drop → dimensional collapse |
| `train/grad_norm` | L2 norm of all parameter gradients | Spikes indicate instability |
| `val/knn_acc_top1` | KNNEvalCallback (k=20) | Best label-free quality proxy |

**For VICReg specifically, decompose the loss:**

```python
self.log("train/vicreg_invariance", loss_inv)   # should decrease
self.log("train/vicreg_variance",   loss_var)   # should stay near 0 (variance enforced)
self.log("train/vicreg_covariance", loss_cov)   # should decrease toward 0
```

This decomposition is essential for debugging VICReg: if `loss_var` is large, the variance term is not being satisfied and collapse risk is high.

### Collapse Detection Reference Implementation

This helper belongs in `BaseSSLMethod` and is called from `on_train_epoch_end` with the last batch's embeddings:

```python
def _log_embedding_health(self, z: torch.Tensor):
    """
    Log collapse detection metrics for a batch of embeddings z [B, D].

    Per-dimension std is the canonical VICReg variance monitoring metric [1].
    Effective rank tracks dimensional collapse [2].

    References
    ----------
    .. [1] Bardes et al. (2022). VICReg. https://arxiv.org/abs/2105.04906
    .. [2] Understanding Dimensional Collapse. OpenReview 2022.
    """
    with torch.no_grad():
        z_std = z.std(dim=0)                          # [D]
        self.log("train/embed_std_mean", z_std.mean())
        self.log("train/embed_std_min",  z_std.min())
        # Effective rank: how many singular values carry significant energy
        _, s, _ = torch.linalg.svd(z - z.mean(dim=0), full_matrices=False)
        eff_rank = (s > 0.01 * s[0]).sum().float()
        self.log("train/embed_eff_rank", eff_rank)
```

### kNN Evaluation Callback

Online kNN evaluation is the best label-free quality proxy (r = 0.96 with out-of-domain kNN accuracy across 26 SSL models, IJCV 2025). Run it as a Lightning callback every N epochs to track representation quality without any labeled data.

```python
# eval/knn_eval.py
from lightning.pytorch.callbacks import Callback
import torch, torch.nn.functional as F


class KNNEvalCallback(Callback):
    """
    Evaluates backbone representations with a weighted k-NN classifier
    at the end of each epoch. Requires a labeled val_dataloader.

    Parameters
    ----------
    k : int
        Number of nearest neighbors. Default 20.
    temperature : float
        Softmax temperature for weighted kNN. Default 0.07.
    every_n_epochs : int
        Run kNN eval every N epochs. Default 5 (avoids overhead each epoch).
    """

    def __init__(self, k: int = 20, temperature: float = 0.07,
                 every_n_epochs: int = 5):
        self.k = k
        self.temperature = temperature
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        # Collect train and val embeddings, run kNN, log accuracy
        ...
```

### Logging Infrastructure

Use TensorBoard as the default (zero extra dependencies, built into Lightning). Add WandB as an opt-in:

```python
# scripts/pretrain.py
if cfg.logger == "wandb":
    logger = WandbLogger(project="ssl-tutorial", name=cfg.run_name)
else:
    logger = TensorBoardLogger("logs/", name=cfg.run_name)
```

**What NOT to log:** Raw embedding tensors per step (enormous storage), confusion matrices during pretraining (labels are not used), per-sample loss values (redundant with mean loss).

---

## Recommended Implementation Order

Build methods in this order. Each group introduces one new infrastructure concept. Later methods reuse infrastructure from earlier ones without modification.

### Group 1: Foundation (build the scaffolding here)

| # | Method | New Infrastructure | Reuses |
|---|--------|-------------------|--------|
| 1 | **SimCLR** | BaseSSLMethod, SSLDataModule, MultiViewDataset, KNNEvalCallback, NT-Xent loss | — |
| 2 | **Instance Discrimination** | Memory bank pattern, NCE loss | BaseSSLMethod, backbone, NT-Xent variant |

Start with SimCLR because it has no moving parts (no momentum encoder, no predictor, no queue). It exercises the entire infrastructure stack with the simplest possible method. Once SimCLR works end-to-end with kNN evaluation, all subsequent methods plug into the same scaffolding.

### Group 2: Momentum Encoder

| # | Method | New Infrastructure | Reuses |
|---|--------|-------------------|--------|
| 3 | **MoCo v1** | BaseMomentumMethod, FIFO queue | NT-Xent loss |
| 4 | **MoCo v2** | Stronger augmentation only (SimCLR-style) | MoCo v1 — minimal delta |
| 5 | **CMC** | Two encoders, multi-view from different channels | BaseMomentumMethod |

Introduce `BaseMomentumMethod` with MoCo v1. All subsequent momentum methods (BYOL, DINO, iBOT) get the EMA infrastructure for free.

### Group 3: No-Negatives

| # | Method | New Infrastructure | Reuses |
|---|--------|-------------------|--------|
| 6 | **BYOL** | Predictor head, stop-gradient, collapse detection logging | BaseMomentumMethod |
| 7 | **SimSiam** | Stop-gradient without EMA (same predictor, no momentum) | BYOL predictor pattern |

Implement BYOL before SimSiam. SimSiam is pedagogically "BYOL minus the momentum encoder." The conceptual relationship is clearer in this order. Add `_log_embedding_health` here — BYOL collapse is the canonical demonstration.

### Group 4: Clustering and Redundancy-Reduction

| # | Method | New Infrastructure | Reuses |
|---|--------|-------------------|--------|
| 8 | **SwAV** | Prototype assignments, Sinkhorn-Knopp, multi-crop | DINOTransform multi-crop |
| 9 | **Barlow Twins** | Cross-correlation matrix loss | BaseSSLMethod, two-view aug |
| 10 | **VICReg** | Variance + invariance + covariance loss | Barlow Twins projector |

Barlow Twins before VICReg. VICReg is partially motivated as a simpler variant of Barlow Twins — the cross-correlation loss and the covariance loss share the same motivation (decorrelation).

### Group 5: Historical Proxy Tasks

| # | Method | New Infrastructure | Reuses |
|---|--------|-------------------|--------|
| 11 | **Invariant Spread** | In-batch softmax, single augmented view | SimCLR NT-Xent |
| 12 | **CPC** | Autoregressive prediction, grid patch encoder | Separate PixelConv encoder |
| 13 | **AMDIM** | Multi-scale feature prediction | CPC patch encoder |

These are valuable for historical context. They can be implemented in any order and don't share infrastructure with Groups 3–4.

### Group 6: ViT-Native Methods

| # | Method | New Infrastructure | Reuses |
|---|--------|-------------------|--------|
| 14 | **MoCo v3** | ViT backbone config, no memory queue | BaseMomentumMethod, NT-Xent |
| 15 | **DINO** | Centering, softmax sharpening, DINOTransform multi-crop | BaseMomentumMethod |
| 16 | **MAE** | Encoder-decoder ViT, patch masking, reconstruction loss | Separate MAE architecture |
| 17 | **iBOT** | DINO + MAE objectives combined | DINO + MAE |

MoCo v3 before DINO. DINO's centering and sharpening are easier to understand after confirming that a simple momentum contrastive method already works well with ViT backbones.

MAE is architecturally self-contained (needs encoder-decoder ViT). Implement it independently; it does not inherit from `BaseMomentumMethod`.

### Ordering Rationale Summary

1. **Infrastructure is built once on the simplest method.** SimCLR forces you to implement `BaseSSLMethod`, `SSLDataModule`, and `KNNEvalCallback` before the complexity of queues or predictors.
2. **`BaseMomentumMethod` is introduced once and reused.** Adding it for MoCo v1 gives it to BYOL, DINO, and iBOT for free.
3. **Collapse detection logging is added with BYOL.** The canonical collapse scenario is non-contrastive methods. Once added, the metric is available for all methods.
4. **ViT methods are last.** They require different backbone config, different augmentation, and different projector architecture. Keeping them in Group 6 avoids polluting the ResNet-oriented tutorial with ViT-specific code paths.

---

## Sources

- solo-learn source (`solo/methods/`): https://github.com/vturrisi/solo-learn (HIGH confidence — source inspected directly)
- solo-learn `__init__.py` registry pattern: source confirmed manual explicit dict registration
- LightlySSL transform-per-method and model organization: https://docs.lightly.ai/self-supervised-learning/ (HIGH confidence — official docs)
- PyTorch Lightning `fast_dev_run` / `limit_train_batches` / `on_train_batch_end`: https://lightning.ai/docs/pytorch/stable/common/trainer.html (HIGH confidence — official docs)
- PyTorch Lightning LightningDataModule lifecycle: https://lightning.ai/docs/pytorch/stable/data/datamodule.html (HIGH confidence)
- VICReg per-dimension std as collapse detection: https://arxiv.org/abs/2105.04906 (HIGH confidence — original paper, Bardes et al. 2022)
- Dimensional collapse survey: https://openreview.net/forum?id=YevsQ05DEN7 (MEDIUM confidence — peer reviewed)
- Stepwise SSL / embedding rank tracking: https://bair.berkeley.edu/blog/2023/07/10/stepwise-ssl/ (MEDIUM confidence — BAIR blog, backed by NeurIPS 2023 paper)
- kNN vs. linear probe as evaluation proxy: https://arxiv.org/abs/2407.12210 (MEDIUM confidence — IJCV 2025)
- NumPy docstring format: https://numpydoc.readthedocs.io/en/latest/format.html (HIGH confidence)
- UvA DL SimCLR tutorial (theory→implementation structure): https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html (MEDIUM confidence)
