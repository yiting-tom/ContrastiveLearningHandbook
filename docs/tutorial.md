# Contrastive Learning Tutorial — Walkthrough

This document is the long-form companion to the README. It begins with an
era-by-era narrative explaining what conceptually changed between the methods
in this repo, then walks through three concrete activities: adding a new
method (section a), running an experiment end-to-end (section b), and
comparing two methods on the same dataset (section c).

If you have not yet read the [README](../README.md), start there.

## Contents

- [The four eras of contrastive SSL](#the-four-eras-of-contrastive-ssl)
- [Section (a): How to Add a New Method](#section-a-how-to-add-a-new-method)
- [Section (b): Running an Experiment End-to-End](#section-b-running-an-experiment-end-to-end)
- [Section (c): Comparing Two Methods](#section-c-comparing-two-methods)
- [Where to next](#where-to-next)

## The four eras of contrastive SSL

The 14 methods in this repo span four conceptually distinct eras. The narrative
below summarizes what each era contributed and what the next era replaced.

### Era 1: Proxy Tasks (2018–2019)

**Methods:** Instance Discrimination, Invariant Spread.

The earliest contrastive methods treated SSL as a *classification* problem with
a peculiar twist: each individual training image is its own class. Instance
Discrimination (Wu et al., CVPR 2018) maintains a memory bank — one
L2-normalized feature vector per training sample — and computes an NCE loss
that pulls each image's encoded feature toward its bank entry while pushing
away from a sample of others. Invariant Spread (Ye et al., CVPR 2019) replaces
the memory bank with an in-batch softmax: positives come from the augmented
view of the same image, negatives from the other images in the batch.

**What was hard:** Memory bank entries are *stale* — they were written by an
earlier snapshot of the encoder, so the loss is computed against feature
vectors that no longer reflect the current model. Training is fragile, and
Invariant Spread's in-batch softmax is sensitive to batch size.

**What the next era solved:** SimCLR proved that with strong augmentation and
a large batch, the in-batch softmax beats memory banks. MoCo replaced the
memory bank with a *momentum-encoded* queue, which gives lots of fresh
negatives without big batches.

### Era 2: In-Batch / Queue / Prototype (2019–2020)

**Methods:** SimCLR v1/v2, MoCo v1/v2, SwAV, InfoMin.

This is the era when contrastive SSL became competitive with supervised
pretraining on ImageNet. Three sub-paradigms emerged in parallel:

- **In-batch contrastive (SimCLR).** Strong augmentation (s=1.0 color jitter,
  Gaussian blur), an MLP projection head, and the symmetric NT-Xent loss on
  large batches. Negatives are the other 2*(B-1) images in the batch.
  Performance scales sharply with batch size — below 256 it degrades; above
  4096 it saturates.
- **Queue-based contrastive (MoCo).** A momentum-EMA encoder produces *keys*
  that go into a FIFO queue (size 65,536). The query encoder uses these
  queued keys as negatives. Decouples the negative count from the batch size
  — works well on a single GPU. MoCo v2 adds SimCLR's MLP head + Gaussian
  blur to MoCo v1's queue.
- **Prototype clustering (SwAV).** Replaces the explicit pairwise contrastive
  loss with online clustering: features are assigned to a fixed codebook of
  K=3000 prototypes via Sinkhorn-Knopp optimal transport, and a
  *swapped-prediction* loss enforces that the cluster assignment for one view
  is predictable from another view's features. Adds the multi-crop trick:
  2 large + 6 small crops per image.

InfoMin sits within Era 2 as an *augmentation-policy* result: views should
share minimal mutual information except for task-relevant content. It is
demonstrated here as a config tweak on top of SimCLR/MoCo.

**What was hard:** SimCLR needs huge batches; MoCo's shuffled-BN is finicky in
distributed settings; SwAV's prototypes need careful initialization (frozen
for the first epoch) to avoid degenerate clusters.

**What the next era solved:** All Era 2 methods rely on negatives in some form.
The next era asked whether negatives are needed at all.

### Era 3: No-Negative Methods (2020–2021)

**Methods:** BYOL, SimSiam, Barlow Twins.

The provocative claim of this era: collapse can be prevented without any
explicit negatives, given the right architectural asymmetries.

- **BYOL (Grill et al., NeurIPS 2020)** uses two networks: an *online* network
  with an extra predictor MLP, and a *target* network updated by EMA. The MSE
  loss between the L2-normalized predictor output and L2-normalized target
  projection — combined with the predictor asymmetry and the EMA target —
  was enough to avoid collapse without negatives.
- **SimSiam (Chen & He, CVPR 2021)** went further: even the EMA target is not
  needed. A single shared encoder, a predictor MLP, and a *stop-gradient* on
  the projector output prevent collapse. This was a striking result — the
  stop-gradient is the only thing standing between SimSiam and a degenerate
  solution.
- **Barlow Twins (Zbontar et al., ICML 2021)** removed the predictor and the
  EMA target entirely. Two views go through a shared encoder and a high-
  dimensional projector (8192-d). The cross-correlation matrix between the
  two batches of projected features is driven toward the identity:
  decorrelation as the anti-collapse mechanism.

**What is hard:** Collapse can hide. The mandatory diagnostic is to log
`z.std(dim=0).mean()` — if this drops toward 0 during training, you have
collapsed. Forgetting the stop-gradient in SimSiam, or using constant EMA
momentum in BYOL, both produce silent quality degradation rather than loud
errors.

**What the next era did:** Showed that all of these tricks transfer to
Vision Transformers.

### Era 4: Transformer Era (2021+)

**Methods:** MoCo v3, DINO, DINOv2 (feature extraction only).

Vision Transformers replaced ResNets as the dominant backbone. Methods needed
adaptation:

- **MoCo v3 (Chen et al., ICCV 2021)** dropped the queue (in-batch keys
  suffice on modern hardware), used symmetric loss, and froze the patch-
  projection layer for stability — without that freeze, ViT training silently
  produces poor representations. AdamW replaced SGD for ViT optimization.
- **DINO (Caron et al., ICCV 2021)** combined a student-teacher framework
  (teacher = EMA of student) with two anti-collapse mechanisms: *centering*
  (subtract running mean from teacher output) and *sharpening* (low teacher
  temperature with warmup 0.04 → 0.07). Multi-crop is reused from SwAV.
  Teacher receives only global crops; student receives all crops.
- **DINOv2 (Oquab et al., 2023)** is a scale story: same DINO recipe + iBOT
  patch-level objective, trained on the curated LVD-142M dataset for hundreds
  of GPU-days. This repo includes DINOv2 only as a feature-extraction / fine-
  tuning demo (`eval/dinov2_demo.py`); training from scratch is out of scope.

**Key insight across eras:** the path from Era 1 to Era 4 is largely about
*reducing the role of negatives* (from millions in Era 1 memory banks, to
batches in Era 2, to none in Era 3, to teacher-student distillation in Era 4)
while *increasing the role of architectural asymmetries* (predictor MLPs,
stop-gradients, EMA targets, multi-crop, centering).

For implementation details on any method, see its class docstring (e.g.,
`help(SimCLRv1Module)`) or the per-method source under `methods/`.

---

## Section (a): How to Add a New Method

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

### The interface: what BaseSSLModule expects

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

### Step 1: Create the package directory

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

### Step 2: Implement the LightningModule subclass

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

### Step 3: Register with the dispatcher

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

### Step 4: Write a YAML config

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

### Step 5: Run training

Train your new method exactly the same way you would train any of the 14 built-in
methods:

```bash
python train.py --config configs/my_toy_contrastive_resnet18.yaml --data-dir data/
```

For a quick sanity check, you can override `data_dir` to a small ImageFolder
fixture (or use the toy fixture from `tests/conftest.py::tmp_imagefolder`).

### Step 6: Add a smoke test (recommended)

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

### Where to go next

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

---

## Section (b): Running an Experiment End-to-End

This section walks through the complete experimental loop: choose a config,
prepare data, train, inspect the checkpoint, run evaluations, and read results.
The worked example trains SimCLR v1 on CIFAR-10 with a ResNet-18 backbone — the
canonical quickstart config — and shows you what to expect at every step so you
can recognize when something has gone wrong.

The full sequence is the same for any of the 14 built-in methods: swap
`configs/simclr_v1_resnet18.yaml` for any other config in `configs/` and the
rest of the pipeline is identical.

### Step 1: Choose a config

Configs in `configs/` are named `<method>_<backbone>[_optimizer].yaml`. For this
walkthrough we use `configs/simclr_v1_resnet18.yaml`:

```yaml
# Excerpt — see configs/simclr_v1_resnet18.yaml for the full file
method: simclr_v1
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

simclr:
  temperature: 0.5
  projection_dim: 128
```

A few orientation notes before training:

- `method: simclr_v1` is the dispatcher key — it must match exactly one entry from
  `core.dispatcher.available_methods()`. Typos raise `ValueError` listing the
  available keys.
- `n_views: 2` makes `SSLDataModule` produce 2 augmented views per image.
- `optimizer: adamw` is the default for SimCLR at small batch sizes; switch to
  `lars` for batch sizes above ~1024 (use `configs/simclr_v1_resnet50_lars.yaml`).
- `data_dir: data` is overridable from the CLI via `--data-dir`.

### Step 2: Prepare an ImageFolder dataset

`SSLDataModule` wraps `torchvision.datasets.ImageFolder`, so it expects a
directory tree with one subfolder per class. CIFAR-10 via `torchvision` does not
ship in this layout — convert it once with:

```bash
python - <<'EOF'
from pathlib import Path
from torchvision.datasets import CIFAR10

out_train = Path("data/cifar10_imagefolder/train")
out_test = Path("data/cifar10_imagefolder/test")
for out, train in [(out_train, True), (out_test, False)]:
    out.mkdir(parents=True, exist_ok=True)
    ds = CIFAR10(root="data/cifar10_raw", train=train, download=True)
    for i, (img, y) in enumerate(ds):
        d = out / ds.classes[y]
        d.mkdir(exist_ok=True)
        img.save(d / f"{i:05d}.png")
EOF
```

After this script you should have:

```
data/cifar10_imagefolder/
  train/
    airplane/  automobile/  bird/  cat/  deer/
    dog/  frog/  horse/  ship/  truck/
  test/
    airplane/  ... (same 10 classes)
```

For larger datasets (ImageNet-100, STL-10), follow the same convention. Any
ImageFolder-formatted directory will plug into the pipeline.

### Step 3: Train

Run training with one command:

```bash
python train.py --config configs/simclr_v1_resnet18.yaml --data-dir data/cifar10_imagefolder/train
```

What happens:

1. `train.py` calls `core.config.load_config()` which validates the YAML against
   `TrainConfig` (Pydantic v2). Unknown keys raise `ValidationError` — useful for
   catching typos.
2. `import methods` triggers `register_method()` for all 14 built-in methods.
3. `core.dispatcher.method_dispatcher(cfg)` returns a `SimCLRv1Module(cfg)` instance.
4. `SSLDataModule(data_dir=..., n_views=2, ...)` builds the multi-view DataLoader.
5. Lightning's `Trainer.fit()` runs for `cfg.max_epochs` epochs, writing
   TensorBoard logs and checkpoints under `lightning_logs/version_<N>/`.

While training, watch:

| TensorBoard scalar | What it tells you |
|--------------------|-------------------|
| `train/loss` | Should decrease monotonically; sudden spikes mean LR too high |
| `train/lr` | Confirms warmup + cosine schedule (rises 0->lr over 10 epochs, then cosine decays) |
| `eval/knn_acc` | Logged every `eval.knn.every_n_epochs` epochs (if KNNCallback enabled in YAML) |

Open TensorBoard with:

```bash
tensorboard --logdir lightning_logs/
```

Expected wall-time on a single A100 / RTX 3090 GPU: roughly 30-45 minutes for
CIFAR-10 / ResNet-18 / 200 epochs / batch 256. CPU-only is feasible for the smoke
test (1 epoch + tiny batch) but impractical for full training.

### Step 4: Locate the checkpoint

Lightning's default checkpoint callback writes the latest epoch's weights to:

```
lightning_logs/version_<N>/checkpoints/epoch=<E>-step=<S>.ckpt
```

Find the latest checkpoint:

```bash
ls -1 lightning_logs/version_*/checkpoints/*.ckpt | sort | tail -1
```

For the rest of this walkthrough we will assume the path is
`lightning_logs/version_0/checkpoints/last.ckpt`. Substitute your real path.

### Step 5: Evaluate — linear probe

The linear probe freezes the backbone, trains a single linear layer on top with
SGD and `weight_decay=0.0`, and reports top-1 accuracy:

```bash
python eval/linear_probe.py configs/simclr_v1_resnet18.yaml \
    --ckpt lightning_logs/version_0/checkpoints/last.ckpt
```

What you will see:

- The script extracts features from the frozen backbone, caches them to disk for
  reuse (set in `eval.linear_probe.cache_features` in YAML), and trains the
  linear head for `eval.linear_probe.epochs` epochs.
- Final stdout line includes `linear_probe/top1: <fraction>` (e.g., `0.892` for
  a fully-trained SimCLR / ResNet-18 / CIFAR-10).
- Linear probe accuracy on CIFAR-10 typically lands in the 0.85–0.92 range for
  SimCLR / ResNet-18 / 200 epochs. Significantly lower (<0.7) suggests collapse
  or undertraining; check `train/embedding_std` in TensorBoard if available.

### Step 6: Evaluate — UMAP visualization

Visualize the feature space:

```bash
python eval/umap_vis.py configs/simclr_v1_resnet18.yaml \
    --ckpt lightning_logs/version_0/checkpoints/last.ckpt
```

This writes a PNG to a path under the eval output directory (see
`eval.umap.output_dir` in YAML; defaults to `eval_outputs/umap/`). Each point is
colored by class; well-trained features cluster cleanly, with class boundaries
visible.

For a perplexity sweep, the t-SNE script writes three PNGs (perplexity 10, 30,
50) — useful as a sanity check that the structure is robust to the choice of
perplexity:

```bash
python eval/tsne_vis.py configs/simclr_v1_resnet18.yaml \
    --ckpt lightning_logs/version_0/checkpoints/last.ckpt
```

### Step 7: Evaluate — k-NN (in-training or post-hoc)

The k-NN evaluation can run either as a Lightning callback during training (set
`eval.knn` in the YAML before `python train.py`) or as a standalone script.
During training, the scalar `eval/knn_acc` appears in TensorBoard at every
`eval.knn.every_n_epochs` epochs. Typical k-NN accuracy on CIFAR-10 with
SimCLR / ResNet-18 / 200 epochs is in the 0.80–0.88 range — a few points below
the linear probe is expected.

### Step 8: (Optional) Fine-tune the full network

If a downstream labeled task is available, fine-tune the backbone with separate
LR groups (backbone 1e-4, head 1e-3):

```bash
python eval/finetune.py configs/simclr_v1_resnet18.yaml \
    --ckpt lightning_logs/version_0/checkpoints/last.ckpt
```

Fine-tuning typically gains 1-3 percentage points over the linear probe on
small datasets like CIFAR-10. The `freeze_bn=True` option (see
`eval.finetune` in YAML) keeps batch-norm layers in eval mode, which helps when
the downstream batch is small.

### Step 9: (Optional) CAM visualization

For ResNet backbones, run EigenCAM (no classifier required) to see what
spatial regions the model has learned to attend to:

```bash
python eval/cam_vis.py configs/simclr_v1_resnet18.yaml \
    --ckpt lightning_logs/version_0/checkpoints/last.ckpt
```

The script writes PNGs of the original images overlaid with CAM heatmaps. For
ViT backbones the script automatically uses the appropriate target layer
(`backbone.blocks[-1].norm1`) and reshape transform.

### Sanity checks: when things go wrong

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `train/loss` plateaus immediately | LR too low or BN issue | Confirm `train/lr` schedule visible in TensorBoard; for ViT, check patch-projection freeze (MoCo v3 / DINO) |
| `train/loss` collapses to a fixed negative value | Stop-gradient missing (SimSiam) or BN replaced with LN (BYOL) | Check `.detach()` placement; verify projector BN |
| `eval/knn_acc` stuck near 0.1 (random for 10 classes) | Bank/queue not updating, or feature normalization missing | Confirm `MemoryBank.update` / `MomentumQueue.update` runs each step |
| Linear probe accuracy << k-NN accuracy | Linear head undertrained or weight decay > 0 | Verify `weight_decay=0.0` in `eval.linear_probe` config |
| Out-of-memory at high `n_views` (SwAV/DINO) | Multi-crop memory grows ~4x | Reduce `batch_size` to 1/4 of SimCLR baseline |

### Comparing methods

To compare two methods on the same dataset, repeat steps 3–6 with the second
config (e.g., `configs/moco_v2_resnet18.yaml`) and compare TensorBoard scalars
side by side. Section (c) walks through the comparison workflow in detail.

---

## Section (c): Comparing Two Methods

This section walks through using the evaluation suite to compare two SSL methods
on the same dataset. The worked example compares **SimCLR v1** (in-batch
contrastive) and **MoCo v2** (queue-based contrastive), both with ResNet-18 on
CIFAR-10. The same procedure applies to any pair from the 14 built-in methods.

After this section you will know how to:

1. Train two methods to comparable checkpoints (matched backbone, dataset, budget).
2. Run k-NN, linear probe, and t-SNE on both checkpoints.
3. Build a comparison table and interpret the differences.

### Step 1: Train both methods with matched hyperparameters

For a fair comparison, the two runs must share:

- **Backbone** (`backbone: resnet18` for both).
- **Dataset** (same `--data-dir`).
- **Training budget** (same `max_epochs`; effective batch size matched as
  closely as method-specific constraints allow).

Both `configs/simclr_v1_resnet18.yaml` and `configs/moco_v2_resnet18.yaml` ship
with `max_epochs: 200` and ResNet-18 + CIFAR-style settings, so they are
already matched out of the box.

```bash
# Train SimCLR v1
python train.py --config configs/simclr_v1_resnet18.yaml \
    --data-dir data/cifar10_imagefolder/train

# Note the version number Lightning assigns; e.g., version_0
# Save the checkpoint path:
SIMCLR_CKPT=lightning_logs/version_0/checkpoints/last.ckpt

# Train MoCo v2
python train.py --config configs/moco_v2_resnet18.yaml \
    --data-dir data/cifar10_imagefolder/train

MOCO_CKPT=lightning_logs/version_1/checkpoints/last.ckpt
```

If you want to run the two trainings in parallel on separate GPUs, set
`CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1` before each `python train.py`
invocation.

#### What is NOT comparable

- **Different backbones** — comparing SimCLR / ResNet-18 against DINO / ViT-S
  conflates the method with the architecture. Stick to the same backbone for
  primary comparisons.
- **Different epoch counts** — methods like BYOL benefit substantially from
  longer training; comparing 200 vs 800 epochs is a budget comparison, not a
  method comparison.
- **Different effective batch sizes** — SimCLR's `train/loss` magnitude depends
  on the number of in-batch negatives; MoCo's depends on the queue size.
  Compare downstream metrics (linear probe / k-NN), not training losses.

### Step 2: Run the linear probe on both checkpoints

The linear probe is the headline metric for SSL representation quality. Run it
on each checkpoint:

```bash
python eval/linear_probe.py configs/simclr_v1_resnet18.yaml \
    --ckpt $SIMCLR_CKPT
# stdout includes: linear_probe/top1: 0.<XYZ>

python eval/linear_probe.py configs/moco_v2_resnet18.yaml \
    --ckpt $MOCO_CKPT
# stdout includes: linear_probe/top1: 0.<XYZ>
```

Both runs use the same eval recipe (frozen backbone, SGD, `weight_decay=0.0`,
MultiStepLR at epochs `[60, 80]`, 100 total epochs) defined under
`eval.linear_probe` in each YAML.

### Step 3: Run k-NN evaluation on both checkpoints

If you enabled `KNNCallback` during training (set `eval.knn` in YAML before
running `python train.py`), the k-NN accuracy is already logged in TensorBoard
as `eval/knn_acc`. Read the final-epoch value from each run:

```bash
tensorboard --logdir lightning_logs/
# Open the UI, navigate to the eval/knn_acc scalar, hover the last point for each run.
```

If you did not enable the callback, you can run k-NN post-hoc by training with a
modified config that includes the callback (or by reusing the linear probe
feature cache).

### Step 4: Run t-SNE on both checkpoints

t-SNE reveals the qualitative structure of the feature space. Run the perplexity
sweep on each checkpoint:

```bash
python eval/tsne_vis.py configs/simclr_v1_resnet18.yaml --ckpt $SIMCLR_CKPT
# Outputs PNGs under eval_outputs/tsne/<run-id>/ for perplexity 10, 30, 50

python eval/tsne_vis.py configs/moco_v2_resnet18.yaml --ckpt $MOCO_CKPT
```

Open the resulting PNGs side by side. What to look for:

- **Class separation:** clear cluster boundaries on both → both methods learned
  class-discriminative features. Blurry boundaries on one but not the other →
  signal that the blurry one is undertrained or has collapsed.
- **Robustness across perplexity values:** a method that produces clean clusters
  at perplexity 10, 30, AND 50 has more reliable structure than one that only
  looks good at perplexity 30.
- **Cluster shape:** SimCLR-style methods often produce more uniform / spherical
  clusters; methods with prototypes or memory banks may show denser cluster
  cores.

### Step 5: Optional — UMAP and CAM

UMAP is generally faster and more global-structure-preserving than t-SNE; run
it for additional confirmation:

```bash
python eval/umap_vis.py configs/simclr_v1_resnet18.yaml --ckpt $SIMCLR_CKPT
python eval/umap_vis.py configs/moco_v2_resnet18.yaml --ckpt $MOCO_CKPT
```

CAM visualization (EigenCAM by default for SSL — no classifier required) shows
which spatial regions each method relies on:

```bash
python eval/cam_vis.py configs/simclr_v1_resnet18.yaml --ckpt $SIMCLR_CKPT
python eval/cam_vis.py configs/moco_v2_resnet18.yaml --ckpt $MOCO_CKPT
```

Differences in CAM heatmaps reveal whether the methods focus on similar
discriminative regions (e.g., object center vs context).

### Step 6: Build the comparison table

Collect the numbers from steps 2–4 into a Markdown table. The exact values
depend on your training run; what matters is the pattern.

```markdown
| Metric | SimCLR v1 / ResNet-18 | MoCo v2 / ResNet-18 |
|--------|-----------------------|---------------------|
| Linear probe top-1 (CIFAR-10) | 0.<XYZ> | 0.<XYZ> |
| k-NN top-1 (k=200, t=0.07) | 0.<XYZ> | 0.<XYZ> |
| Training epochs | 200 | 200 |
| Batch size | 256 | 256 |
| Effective negatives | 2*(B-1) = 510 | queue_size = 4096 |
| Optimizer | AdamW | SGD |
| EMA momentum | n/a | 0.999 |
```

Things this table reveals:

- The **effective negatives** column makes the structural difference between the
  methods explicit. SimCLR's negatives come from the batch only; MoCo's queue
  decouples the negative count from the batch size.
- The **optimizer + EMA** columns show the secondary differences that may
  influence results — useful when methods have similar accuracy but you want
  to understand why.

### Step 7: Interpreting differences

A few rules of thumb:

- **Linear probe deltas <1 percentage point** are typically within run-to-run
  noise. Re-train with a different seed before drawing conclusions.
- **Linear probe vs k-NN gap** is informative: a method with high linear probe
  accuracy but lower k-NN often has features that are linearly separable but
  not perfectly clustered (the linear classifier compensates).
- **t-SNE patterns matching the linear probe ranking** is reassuring. t-SNE
  showing one method clearly cleaner while linear probe says they are tied is
  a sign of either undertrained probe or t-SNE artifact — try multiple
  perplexity values + multiple t-SNE seeds.

### Other comparison pairings worth running

| Pairing | What it isolates |
|---------|------------------|
| SimCLR v1 vs MoCo v2 | In-batch negatives vs queue (this section) |
| SimCLR v1 vs BYOL | With-negatives vs no-negatives |
| MoCo v2 vs MoCo v3 | ResNet vs ViT (different backbones — adjust expectations) |
| BYOL vs SimSiam | EMA target vs stop-gradient-only |
| Barlow Twins vs SimSiam | Decorrelation loss vs cosine loss |
| SwAV vs DINO | Prototype clustering: ResNet vs ViT |

For each pairing, repeat steps 1–6 with the appropriate `configs/<method>.yaml`
files — the rest of the workflow is identical.

---

## Where to next

- **Looking up a specific method?** Read its class docstring directly:
  `python -c "from methods.simclr.module import SimCLRv1Module; help(SimCLRv1Module)"`.
  All 14 method classes have DOC-02-compliant docstrings (paper, arXiv, gotchas,
  reference implementation URL).
- **Adding a method that uses an EMA target / momentum encoder?** Study
  `methods/byol/module.py` (no-negative + EMA) or `methods/moco/module.py`
  (queue + EMA).
- **Adding a multi-crop method?** Set `n_views > 2` in YAML and `SSLDataModule`
  switches to `MultiCropDataset` automatically.
- **Running on something other than ImageFolder?** Wrap your dataset in an
  `ImageFolder`-compatible adapter or open an issue describing the layout.

This is a tutorial repository — its purpose is to make every method readable
and runnable, not to win benchmarks. Hyperparameters favor clarity (small
backbones, short schedules, configs that fit on one GPU) over peak accuracy.
For research-grade reproductions of any individual method, follow the
reference implementation URL in that method's docstring.
