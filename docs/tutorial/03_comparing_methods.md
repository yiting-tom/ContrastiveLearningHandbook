# Tutorial Section (c): Comparing Two Methods

This section walks through using the evaluation suite to compare two SSL methods
on the same dataset. The worked example compares **SimCLR v1** (in-batch
contrastive) and **MoCo v2** (queue-based contrastive), both with ResNet-18 on
CIFAR-10. The same procedure applies to any pair from the 14 built-in methods.

After this section you will know how to:

1. Train two methods to comparable checkpoints (matched backbone, dataset, budget).
2. Run k-NN, linear probe, and t-SNE on both checkpoints.
3. Build a comparison table and interpret the differences.

## Step 1: Train both methods with matched hyperparameters

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

### What is NOT comparable

- **Different backbones** — comparing SimCLR / ResNet-18 against DINO / ViT-S
  conflates the method with the architecture. Stick to the same backbone for
  primary comparisons.
- **Different epoch counts** — methods like BYOL benefit substantially from
  longer training; comparing 200 vs 800 epochs is a budget comparison, not a
  method comparison.
- **Different effective batch sizes** — SimCLR's `train/loss` magnitude depends
  on the number of in-batch negatives; MoCo's depends on the queue size.
  Compare downstream metrics (linear probe / k-NN), not training losses.

## Step 2: Run the linear probe on both checkpoints

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

## Step 3: Run k-NN evaluation on both checkpoints

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

## Step 4: Run t-SNE on both checkpoints

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

## Step 5: Optional — UMAP and CAM

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

## Step 6: Build the comparison table

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

## Step 7: Interpreting differences

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

## Other comparison pairings worth running

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
