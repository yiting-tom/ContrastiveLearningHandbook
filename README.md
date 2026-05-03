# Contrastive Learning Tutorial Repository

A self-contained, reproducible tutorial implementation of 14 self-supervised
contrastive learning methods spanning four eras (proxy tasks → in-batch
contrastive → no-negative → transformer-based). Every method is a thin
`LightningModule` subclass that plugs into a single shared training script,
data module, and evaluation suite, so you can read each method's loss
function in isolation and compare methods on identical pipelines.

**Audience:** ML practitioners who want to understand how each contrastive
learning method actually works — not a research-grade benchmark suite.

---

## Installation

Requires Python 3.10+ and PyTorch 2.0+. CPU-only and CUDA both supported.

```bash
pip install -r requirements.txt
```

This installs `torch`, `lightning`, `timm`, `pydantic`, `pyyaml`, `faiss-cpu`,
`umap-learn`, `pytorch-grad-cam`, and other tutorial dependencies.

## Quickstart

Train SimCLR v1 on a small ImageFolder dataset and run a k-NN evaluation in
five commands. The example uses `data/imagenet100` — substitute any
`ImageFolder`-format directory (one subfolder per class).

```bash
# 1. Install
pip install -r requirements.txt

# 2. Prepare an ImageFolder dataset (CIFAR-10 example below)
#    See "Preparing CIFAR-10 as ImageFolder" section.

# 3. Train SimCLR v1 (200 epochs, ResNet-18, AdamW)
python train.py --config configs/simclr_v1_resnet18.yaml --data-dir data/cifar10_imagefolder

# 4. Linear probe evaluation
python eval/linear_probe.py configs/simclr_v1_resnet18.yaml --ckpt checkpoints/last.ckpt

# 5. Visualize feature space (UMAP)
python eval/umap_vis.py configs/simclr_v1_resnet18.yaml --ckpt checkpoints/last.ckpt
```

### Preparing CIFAR-10 as ImageFolder

`SSLDataModule` wraps `torchvision.datasets.ImageFolder`, which requires a
directory tree with one subdirectory per class. `torchvision.datasets.CIFAR10`
does not produce this layout automatically. Convert it once:

```bash
python - <<'EOF'
from pathlib import Path
from torchvision.datasets import CIFAR10
from PIL import Image

root = Path("data/cifar10_imagefolder")
for split, train_flag in [("train", True), ("val", False)]:
    out = root / split
    out.mkdir(parents=True, exist_ok=True)
    ds = CIFAR10(root="data/cifar10_raw", train=train_flag, download=True)
    for i, (img, y) in enumerate(ds):
        d = out / ds.classes[y]
        d.mkdir(exist_ok=True)
        img.save(d / f"{i:05d}.png")
EOF
```

Then point `--data-dir data/cifar10_imagefolder` at the result. `SSLDataModule`
auto-detects the `train/` and `val/` subdirectories (see `core/data.py`); the
`val/` split is required by `eval/linear_probe.py` and `eval/finetune.py`.

## Config System

Every experiment is described by a single YAML file in `configs/`. The YAML
is parsed by `yaml.safe_load` and validated by a Pydantic v2 model
(`TrainConfig`) that uses `extra='forbid'` — unknown keys raise a
`ValidationError` immediately, which catches typos in tutorial copy-paste.

Top-level fields (see `core/config.py` for the full schema):

| Field | Type | Description |
|-------|------|-------------|
| `method` | str | Dispatcher key — must match one of the 14 methods below |
| `backbone` | str | Any timm model name, e.g., `resnet18`, `vit_small_patch16_224` |
| `pretrained` | bool | Load timm pretrained weights |
| `max_epochs` | int | Training epochs |
| `warmup_epochs` | int | Cosine warmup duration |
| `batch_size` | int | Per-device batch size |
| `lr` | float | Base learning rate |
| `weight_decay` | float | AdamW / SGD weight decay |
| `optimizer` | str | `adamw`, `sgd`, or `lars` |
| `n_views` | int | Augmented views per image (2 default; 8 for SwAV/DINO) |
| `data_dir` | str | ImageFolder root |
| `num_workers` | int | DataLoader worker count |
| `gradient_clip_val` | float \| null | Optional gradient clipping (recommended for ViT methods) |

Per-method sub-configs (e.g., `simclr.temperature`, `moco.queue_size`) live
under namespaced keys. See `configs/example.yaml` for a complete annotated
config and any `configs/<method>_resnet18.yaml` for a real example.

## Methods (14 v1 methods)

| Method | Dispatcher Key | Era | Venue | Year | Primary Contribution |
|--------|----------------|-----|-------|------|----------------------|
| Instance Discrimination | `instance_discrimination` | Era 1: Proxy Tasks | CVPR 2018 | 2018 | Non-parametric memory bank; each image as its own class |
| Invariant Spread | `invariant_spread` | Era 1: Proxy Tasks | CVPR 2019 | 2019 | In-batch softmax contrastive; direct ancestor of SimCLR |
| MoCo v1 | `moco_v1` | Era 2: Queue-Based | CVPR 2020 | 2020 | Momentum encoder + FIFO queue for large negative set |
| MoCo v2 | `moco_v2` | Era 2: Queue-Based | arXiv 2020 | 2020 | MoCo + SimCLR architecture improvements (MLP head, blur, cosine LR) |
| SimCLR v1 | `simclr_v1` | Era 2: In-Batch | ICML 2020 | 2020 | Strong augmentation + in-batch symmetric NT-Xent loss |
| SimCLR v2 | `simclr_v2` | Era 2: In-Batch | NeurIPS 2020 | 2020 | Deeper 3-layer projection head (pretraining stage only) |
| SwAV | `swav` | Era 2: Prototype | NeurIPS 2020 | 2020 | Online clustering via Sinkhorn-Knopp OT; multi-crop |
| InfoMin | `infomin` | Era 2: Augmentation | NeurIPS 2020 | 2020 | Minimal-MI view design; augmentation-policy principle |
| BYOL | `byol` | Era 3: No-Negative | NeurIPS 2020 | 2020 | Bootstrap without negatives via predictor asymmetry + EMA |
| SimSiam | `simsiam` | Era 3: No-Negative | CVPR 2021 | 2021 | Stop-gradient as the only collapse prevention; no EMA |
| Barlow Twins | `barlow_twins` | Era 3: No-Negative | ICML 2021 | 2021 | Redundancy reduction via cross-correlation matrix toward identity |
| MoCo v3 | `moco_v3` | Era 4: Transformer | ICCV 2021 | 2021 | MoCo for ViTs; patch-projection freeze for stability |
| DINO | `dino` | Era 4: Transformer | ICCV 2021 | 2021 | Student-teacher with centering + sharpening; no contrastive negatives |
| DINOv2* | `eval/dinov2_demo.py` | Era 4: Transformer | TMLR 2024 | 2023 | Large-scale pretraining; tutorial = feature extraction only |

*DINOv2 is provided as a feature-extraction / fine-tuning demo (`eval/dinov2_demo.py`),
not as a from-scratch training implementation. Training DINOv2 requires the
LVD-142M dataset and hundreds of GPU-days.

Every method's `LightningModule` class docstring contains the paper title,
authors, venue, arXiv link, algorithm summary, gotchas, and reference
implementation URL. See `methods/simclr/module.py` (`SimCLRv1Module`) as the
canonical example.

## Evaluation

The `eval/` directory contains six evaluation tools that work from any
pretrained checkpoint produced by `train.py`. Each script accepts the same
YAML config used for pretraining plus a `--ckpt` path.

```bash
# k-NN — runs in-training when eval.knn is set in YAML, OR via callback
python train.py --config configs/<method>.yaml  # KNNCallback runs every N epochs

# Linear probe (frozen backbone, weight_decay=0.0 on linear head)
python eval/linear_probe.py configs/<method>.yaml --ckpt <path>

# t-SNE feature visualization (perplexity sweep: 10, 30, 50)
python eval/tsne_vis.py configs/<method>.yaml --ckpt <path>

# UMAP feature visualization (preferred for >5K samples)
python eval/umap_vis.py configs/<method>.yaml --ckpt <path>

# Fine-tuning (separate LR groups: backbone 1e-4, head 1e-3)
python eval/finetune.py configs/<method>.yaml --ckpt <path>

# CAM visualization (EigenCAM by default for SSL; GradCAM with classifier)
python eval/cam_vis.py configs/<method>.yaml --ckpt <path>

# DINOv2 zero-shot k-NN + linear probe (no SSL training needed)
python eval/dinov2_demo.py --dataset cifar10
```

Per-eval configuration lives under `eval:` in the same YAML file (see
`core/config.py::EvalConfig`).

## Tutorial

A guided walkthrough covering (a) how to add a new method, (b) running a full
experiment from config to evaluation, and (c) comparing two methods on the
same dataset is in [`docs/tutorial.md`](docs/tutorial.md).

## Project Layout

```
core/         Shared infrastructure: BaseSSLModule, config schema, dispatcher,
              data module, augmentations, EMA updater, projection heads, losses
methods/      One sub-package per SSL method (14 methods)
configs/      One YAML per (method, backbone) combination (17 configs)
eval/         Six evaluation tools + DINOv2 demo
tests/        326+ unit and integration tests
train.py      Single CLI entry point for all 14 methods
```

## Citation

If you use this tutorial repository in your work, please cite the original
papers (linked in each method's class docstring) rather than this repo.
