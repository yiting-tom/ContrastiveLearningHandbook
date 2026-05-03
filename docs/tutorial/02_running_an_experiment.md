# Tutorial Section (b): Running an Experiment End-to-End

This section walks through the complete experimental loop: choose a config,
prepare data, train, inspect the checkpoint, run evaluations, and read results.
The worked example trains SimCLR v1 on CIFAR-10 with a ResNet-18 backbone — the
canonical quickstart config — and shows you what to expect at every step so you
can recognize when something has gone wrong.

The full sequence is the same for any of the 14 built-in methods: swap
`configs/simclr_v1_resnet18.yaml` for any other config in `configs/` and the
rest of the pipeline is identical.

## Step 1: Choose a config

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

## Step 2: Prepare an ImageFolder dataset

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

## Step 3: Train

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

## Step 4: Locate the checkpoint

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

## Step 5: Evaluate — linear probe

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

## Step 6: Evaluate — UMAP visualization

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

## Step 7: Evaluate — k-NN (in-training or post-hoc)

The k-NN evaluation can run either as a Lightning callback during training (set
`eval.knn` in the YAML before `python train.py`) or as a standalone script.
During training, the scalar `eval/knn_acc` appears in TensorBoard at every
`eval.knn.every_n_epochs` epochs. Typical k-NN accuracy on CIFAR-10 with
SimCLR / ResNet-18 / 200 epochs is in the 0.80–0.88 range — a few points below
the linear probe is expected.

## Step 8: (Optional) Fine-tune the full network

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

## Step 9: (Optional) CAM visualization

For ResNet backbones, run EigenCAM (no classifier required) to see what
spatial regions the model has learned to attend to:

```bash
python eval/cam_vis.py configs/simclr_v1_resnet18.yaml \
    --ckpt lightning_logs/version_0/checkpoints/last.ckpt
```

The script writes PNGs of the original images overlaid with CAM heatmaps. For
ViT backbones the script automatically uses the appropriate target layer
(`backbone.blocks[-1].norm1`) and reshape transform.

## Sanity checks: when things go wrong

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `train/loss` plateaus immediately | LR too low or BN issue | Confirm `train/lr` schedule visible in TensorBoard; for ViT, check patch-projection freeze (MoCo v3 / DINO) |
| `train/loss` collapses to a fixed negative value | Stop-gradient missing (SimSiam) or BN replaced with LN (BYOL) | Check `.detach()` placement; verify projector BN |
| `eval/knn_acc` stuck near 0.1 (random for 10 classes) | Bank/queue not updating, or feature normalization missing | Confirm `MemoryBank.update` / `MomentumQueue.update` runs each step |
| Linear probe accuracy << k-NN accuracy | Linear head undertrained or weight decay > 0 | Verify `weight_decay=0.0` in `eval.linear_probe` config |
| Out-of-memory at high `n_views` (SwAV/DINO) | Multi-crop memory grows ~4x | Reduce `batch_size` to 1/4 of SimCLR baseline |

## Comparing methods

To compare two methods on the same dataset, repeat steps 3–6 with the second
config (e.g., `configs/moco_v2_resnet18.yaml`) and compare TensorBoard scalars
side by side. Section (c) walks through the comparison workflow in detail.
