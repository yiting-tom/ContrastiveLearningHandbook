# Tutorial Section (a): How to Add a New Method

This section walks through adding a new self-supervised contrastive method to the
repository, using a small "toy contrastive" example as the worked sample. By the
end you will have:

1. A new `LightningModule` subclass under `methods/<newmethod>/`.
2. The method registered in `core.dispatcher` so `method: <key>` works in any YAML.
3. A YAML config in `configs/<newmethod>_resnet18.yaml`.
4. A successful `python train.py --config configs/<newmethod>_resnet18.yaml` run.

The whole loop is six small steps. The goal is for you to be able to copy any
existing method (e.g., `methods/invariant_spread/`) as a starting point and swap
in your own loss in under 30 lines.

## The interface: what BaseSSLModule expects

Every method in this repo subclasses `core.base.BaseSSLModule`, which is a thin
wrapper around `lightning.LightningModule` that handles the parts that are the
same for every contrastive method:

- `configure_optimizers()` dispatches AdamW / SGD / LARS based on `cfg.optimizer`,
  and attaches a warmup-cosine LR scheduler.
- `on_train_batch_end()` calls EMA updates on any registered momentum encoders
  (no-op for methods without an EMA target).
- `log_train_metrics(loss)` logs `train/loss`, `train/lr`, and per-method
  diagnostics via `self.log(...)`.
- `learnable_params` defaults to `self.parameters()`; override it to exclude EMA
  target params from the optimizer (see `methods/byol/module.py`).

A subclass MUST override two methods:

| Method | Purpose |
|--------|---------|
| `build_projector(self) -> nn.Module` | Construct the projection head (often `core.projection.ProjectionHead`) given `self.feat_dim` |
| `training_step(self, batch, batch_idx) -> torch.Tensor` | Compute the contrastive loss for one mini-batch |

Everything else (optimizer choice, augmentations, data loading) comes from the
YAML config plus the shared `SSLDataModule`.

## Step 1: Create the package directory

Each method lives in its own sub-package under `methods/`. The convention is:

```
methods/
  my_toy_contrastive/
    __init__.py    # registers the method with the dispatcher
    module.py      # the LightningModule subclass
```

```bash
mkdir methods/my_toy_contrastive
touch methods/my_toy_contrastive/__init__.py methods/my_toy_contrastive/module.py
```

## Step 2: Implement the LightningModule subclass

The minimal subclass that works with the existing infrastructure looks like this.
Copy it into `methods/my_toy_contrastive/module.py`:

```python
"""My toy contrastive method — a minimal SimCLR-like example for the tutorial.

Paper: (this is a tutorial example — no paper)
Authors: you
Venue: tutorial
arXiv: n/a

Algorithm:
1. Augment each image into 2 views.
2. Encode with shared backbone, project via 2-layer MLP.
3. Compute symmetric InfoNCE loss with temperature 0.5.

Gotchas:
- This is the smallest possible BaseSSLModule subclass. It uses InfoNCELoss
  in symmetric mode (the same path SimCLR uses).

Reference implementation: this file.
"""
from __future__ import annotations

import torch.nn as nn

from core.backbone import build_backbone
from core.base import BaseSSLModule
from core.config import TrainConfig
from core.losses import InfoNCELoss
from core.projection import ProjectionHead


class MyToyContrastiveModule(BaseSSLModule):
    """Minimal contrastive method for the tutorial."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        self.backbone, self.feat_dim = build_backbone(cfg.backbone, cfg.pretrained)
        self.projector = self.build_projector()
        self.loss_fn = InfoNCELoss(temperature=0.5)

    def build_projector(self) -> nn.Module:
        return ProjectionHead(
            input_dim=self.feat_dim,
            hidden_dim=2048,
            output_dim=128,
            num_layers=2,
        )

    def training_step(self, batch, batch_idx):
        views, _ = batch                       # SSLDataModule yields (views, labels)
        z_i = self.projector(self.backbone(views[0]))
        z_j = self.projector(self.backbone(views[1]))
        loss = self.loss_fn(z_i, z_j)          # InfoNCELoss L2-normalizes internally
        self.log_train_metrics(loss)
        return loss
```

Things to notice:

- **Always use `backbone.num_features`** via `build_backbone(...)` — never hard-code
  2048 or 512. timm exposes `num_features` consistently across ResNets and ViTs.
- **`InfoNCELoss` L2-normalizes inputs internally**. Do not pre-normalize.
- **`views, _ = batch`** — `SSLDataModule` yields a list of view tensors plus class
  labels (the labels are unused for SSL methods; SupCon is the exception).
- **`log_train_metrics`** logs `train/loss` and `train/lr` for free; you can add
  method-specific scalars by calling `self.log("train/<name>", value)` directly
  after `log_train_metrics(loss)`.

## Step 3: Register with the dispatcher

Methods become discoverable to `core.dispatcher.method_dispatcher` via a side-effect
import. Add the registration call in `methods/my_toy_contrastive/__init__.py`:

```python
"""my_toy_contrastive package — registers MyToyContrastiveModule with the dispatcher."""
from core.dispatcher import register_method
from methods.my_toy_contrastive.module import MyToyContrastiveModule

register_method("my_toy_contrastive", MyToyContrastiveModule)
```

Then add ONE line to the top-level `methods/__init__.py` so the side-effect import
fires when `train.py` runs `import methods`:

```python
# methods/__init__.py — add this line at the bottom:
import methods.my_toy_contrastive  # noqa: F401
```

Verify the registration worked:

```bash
python -c "import methods; from core.dispatcher import available_methods; print('my_toy_contrastive' in available_methods())"
# expected: True
```

## Step 4: Write a YAML config

Configs live in `configs/`. Copy `configs/simclr_v1_resnet18.yaml` as a starting
point and rename the method key:

```yaml
# configs/my_toy_contrastive_resnet18.yaml
method: my_toy_contrastive
backbone: resnet18
pretrained: false

max_epochs: 200
warmup_epochs: 10
batch_size: 256
lr: 1e-3
weight_decay: 1e-6
optimizer: adamw
n_views: 2
data_dir: data
num_workers: 4
```

A few rules the config schema enforces (`extra='forbid'` on `TrainConfig`):

- **Unknown top-level keys raise ValidationError.** If you misspell `optimizer` as
  `optimiser`, `load_config` raises `pydantic.ValidationError` immediately — by design.
- **Per-method sub-configs are namespaced.** SimCLR puts its temperature under
  `simclr.temperature`; if your method needs config knobs, add a `MyToyContrastiveConfig`
  class to `core/config.py` (mirror `SimCLRConfig` / `MoCoConfig`).
- **Optimizer must be one of `adamw`, `sgd`, `lars`**; the value is dispatched in
  `BaseSSLModule.configure_optimizers`.

## Step 5: Run training

Train your new method exactly the same way you would train any of the 14 built-in
methods:

```bash
python train.py --config configs/my_toy_contrastive_resnet18.yaml --data-dir data/
```

For a quick sanity check, you can override `data_dir` to a small ImageFolder
fixture (or use the toy fixture from `tests/conftest.py::tmp_imagefolder`).

## Step 6: Add a smoke test (recommended)

Drop a minimal test in `tests/test_my_toy_contrastive.py`:

```python
from __future__ import annotations

import lightning as L

from core.config import TrainConfig
from core.data import SSLDataModule
from core.dispatcher import method_dispatcher


def test_my_toy_contrastive_smoke(tmp_imagefolder, toy_config_dict):
    cfg_raw = dict(toy_config_dict)
    cfg_raw["method"] = "my_toy_contrastive"
    cfg_raw["data_dir"] = str(tmp_imagefolder)
    cfg = TrainConfig.model_validate(cfg_raw)

    model = method_dispatcher(cfg)
    dm = SSLDataModule(
        data_dir=cfg.data_dir,
        n_views=cfg.n_views,
        batch_size=4,
        num_workers=0,
        size=32,
        strong=True,
    )
    trainer = L.Trainer(
        max_epochs=1, limit_train_batches=1,
        accelerator="cpu", logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
    )
    trainer.fit(model, dm)
```

Run it:

```bash
pytest tests/test_my_toy_contrastive.py -x -q
```

## Where to go next

- To add a method that uses an **EMA momentum encoder** (BYOL / MoCo / DINO style),
  study `methods/byol/module.py`. It overrides `learnable_params` to exclude EMA
  target params and wires the EMA update through `on_train_batch_end`.
- To add a method that uses a **memory bank or queue** (Instance Discrimination,
  MoCo), see `core/queue.py` (`MomentumQueue`) and
  `methods/instance_discrimination/module.py` (`MemoryBank`).
- To add a **multi-crop method** (SwAV, DINO), set `n_views > 2` in the YAML;
  `SSLDataModule` automatically switches to `MultiCropDataset`.

Section (b) — *Running an experiment end-to-end* — picks up from here and walks
through training one of the 14 built-in methods, evaluating it via the `eval/`
suite, and reading the results.
