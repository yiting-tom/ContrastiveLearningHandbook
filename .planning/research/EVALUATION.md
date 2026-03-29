# Evaluation Suite Research

**Project:** SSL / Contrastive Learning Tutorial Repository
**Researched:** 2026-03-29
**Overall confidence:** HIGH (implementation patterns verified against solo-learn source, official PyTorch Lightning docs, and published benchmark protocols)

---

## Linear Probing

### What It Is

Freeze the backbone; extract features once; train a linear classifier on the frozen features. The resulting accuracy is the canonical proxy metric for representation quality in SSL. Research confirms linear probing is the best OOD predictor (r = 0.84 across 26 SSL models on 11 datasets, IJCV 2025).

### Standard Protocol

| Hyperparameter | CIFAR-10/100 | ImageNet-100 | ImageNet-1K |
|----------------|-------------|--------------|-------------|
| Epochs | 100 | 90–100 | 90–100 |
| Optimizer | SGD (momentum 0.9) | SGD or LARS | LARS |
| Learning rate | 0.1 (step decay) | 0.1 → cosine | 0.3 × (BS/256) |
| LR schedule | Step: decay at 60, 80 | Cosine | Linear warmup + cosine |
| Batch size | 128–256 | 256 | 256–4096 |
| Weight decay | 0 (frozen BN: critical) | 0 | 0 |
| Backbone | Frozen (`.requires_grad_(False)`) | Frozen | Frozen |

**Critical:** Weight decay must be 0 for the linear head when the backbone is frozen. Any regularization on the linear layer hurts accuracy.

### Implementation Pattern (solo-learn style)

solo-learn separates linear evaluation into a dedicated `main_linear.py` script launched after pretraining. This is the recommended approach — not an online callback — because it avoids cluttering the pretraining loop and allows proper hyperparameter sweeps on the evaluator.

```python
# eval/linear_probe.py
import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import DataLoader

class LinearProbeModule(L.LightningModule):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int,
                 lr: float = 0.1, epochs: int = 100):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        # Freeze backbone completely
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        # Only the linear head is trainable
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            h = self.backbone(x)
        return self.classifier(h)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log_dict({"train/loss": loss, "train/acc": acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log_dict({"val/loss": loss, "val/acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.SGD(
            self.classifier.parameters(),   # only the head
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=0.0,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[60, 80], gamma=0.1
        )
        return {"optimizer": opt, "lr_scheduler": scheduler}
```

**Memory optimization for large datasets:** Pre-extract all features to disk (or CPU RAM) once, then train the linear head on the cached tensors. This avoids running backbone inference on every epoch.

```python
# Pre-extract features once, cache to disk
def extract_features(backbone, loader, device):
    backbone.eval().to(device)
    all_feats, all_labels = [], []
    with torch.inference_mode():
        for x, y in loader:
            feats = backbone(x.to(device)).cpu()
            all_feats.append(feats)
            all_labels.append(y)
    return torch.cat(all_feats), torch.cat(all_labels)
```

### Online vs. Offline

| Mode | Approach | When to Use |
|------|----------|-------------|
| Online callback | Lightweight sklearn `LogisticRegression` on val set at end of each pretraining epoch | Training-time monitoring only; rough proxy |
| Offline script | Full 100-epoch linear training on frozen backbone after pretraining | Final evaluation, reported number |

Use online probing for monitoring during pretraining, offline probing for any number you report. The two give different numbers; do not conflate them.

### Integration with YAML Config

```yaml
eval:
  linear_probe:
    enabled: true
    epochs: 100
    lr: 0.1
    batch_size: 256
    optimizer: sgd
    weight_decay: 0.0
    lr_decay_steps: [60, 80]
```

---

## k-NN Evaluation

### What It Is

Encode the entire training set into a feature bank; for each test sample, find the k nearest neighbors by cosine similarity and aggregate their labels via temperature-weighted voting. No gradient, no training — just inference. Extremely fast to run and provides a training-free quality signal.

### Weighted k-NN Protocol (DINO / MoCo v3 style)

The canonical implementation from DINO and MoCo v3:

1. Build a feature bank `F_train ∈ R^{N × D}` (L2-normalized) from the training set.
2. For each test sample `q`, compute cosine similarities to all training embeddings.
3. Take the top-k most similar; compute softmax with temperature τ over their similarities.
4. Return the weighted vote across labels.

```python
# eval/knn_eval.py
import torch
import torch.nn.functional as F

@torch.inference_mode()
def knn_evaluate(
    backbone,
    train_loader,
    val_loader,
    device,
    k: int = 200,
    temperature: float = 0.07,
    num_classes: int = 10,
) -> float:
    backbone.eval().to(device)

    # 1. Build feature bank
    train_feats, train_labels = [], []
    for x, y in train_loader:
        feats = F.normalize(backbone(x.to(device)), dim=1)
        train_feats.append(feats.cpu())
        train_labels.append(y)
    feature_bank = torch.cat(train_feats).T  # [D, N]
    feature_labels = torch.cat(train_labels)  # [N]

    # 2. Classify test samples
    correct, total = 0, 0
    for x, y in val_loader:
        q = F.normalize(backbone(x.to(device)), dim=1)  # [B, D]
        sim = q @ feature_bank.to(device)               # [B, N]
        sim_topk, idx_topk = sim.topk(k, dim=1)         # [B, k]

        # Temperature-scaled weights
        weights = F.softmax(sim_topk / temperature, dim=1)  # [B, k]
        # One-hot neighbor labels
        neighbor_labels = feature_labels[idx_topk]          # [B, k]
        # Weighted vote
        votes = torch.zeros(x.size(0), num_classes, device=device)
        for cls in range(num_classes):
            mask = (neighbor_labels.to(device) == cls).float()
            votes[:, cls] = (weights * mask).sum(dim=1)
        preds = votes.argmax(dim=1)
        correct += (preds.cpu() == y).sum().item()
        total += y.size(0)

    return correct / total
```

### Key Hyperparameters

| Parameter | Typical Value | Effect |
|-----------|--------------|--------|
| `k` | 20 (CIFAR), 200 (ImageNet) | Larger k = more stable but slower; 200 is standard for ImageNet-scale |
| `temperature` τ | 0.07 | Lower = sharper (trusts nearest neighbor more); 0.07 is standard from MoCo |
| `metric` | Cosine (L2-normalize then dot product) | Always use cosine for SSL embeddings; Euclidean works poorly |

### Memory-Efficient Variant with FAISS

For datasets > 100K samples, use FAISS for approximate nearest neighbor search instead of brute-force matrix multiplication:

```python
import faiss
import numpy as np

# After L2-normalizing all features:
index = faiss.IndexFlatIP(feat_dim)   # inner product on normalized = cosine
index.add(feature_bank_np)            # shape [N, D]
distances, indices = index.search(query_np, k)  # shape [B, k] each
```

FAISS `IndexFlatIP` on L2-normalized vectors is exact cosine similarity with O(1) memory overhead vs. full similarity matrix. Essential for ImageNet-scale (1.2M samples × 2048 dims would be ~9 GB as float32 matrix).

### Running as Lightning Callback

k-NN evaluation is lightweight enough to run every N epochs as a callback:

```python
class KNNCallback(L.Callback):
    def __init__(self, train_loader, val_loader, k=200, every_n_epochs=5):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.k = k
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        acc = knn_evaluate(
            pl_module.backbone,
            self.train_loader,
            self.val_loader,
            device=pl_module.device,
            k=self.k,
        )
        pl_module.log("eval/knn_acc", acc, prog_bar=True)
```

### Integration with YAML Config

```yaml
eval:
  knn:
    enabled: true
    k: 200
    temperature: 0.07
    every_n_epochs: 5   # 0 = only at end
```

---

## t-SNE Visualization

### What It Is

t-SNE projects high-dimensional embeddings into 2D for qualitative assessment of cluster structure. Well-separated, tight clusters indicate discriminative representations. Run on a fixed subset (1000–5000 samples) to keep it tractable.

### Key Hyperparameters

| Parameter | Range | Recommendation |
|-----------|-------|----------------|
| `perplexity` | 5–50 | Start at 30; scale up for larger subsets (rule: perplexity ≈ sqrt(N)) |
| `n_iter` | 1000–5000 | Never use < 1000; 2000 is safe; increase until shapes stabilize |
| `learning_rate` | 10–1000 | `'auto'` in scikit-learn 1.2+ sets it to max(N/12, 50) — use it |
| `init` | `'pca'` | Always use PCA init (not random) for reproducibility and better global structure |
| `metric` | cosine | For SSL embeddings; NOT Euclidean |

**Pitfall:** Never conclude a cluster structure exists or does not exist from a single perplexity value. Run at least 3 perplexities (e.g., 10, 30, 50) and check for consistency.

**Pitfall:** t-SNE does not preserve global distances. Two well-separated clusters in a t-SNE plot does not mean large inter-cluster distance in embedding space.

### Implementation Pattern

```python
# eval/tsne_vis.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch, torch.nn.functional as F

def run_tsne(
    backbone,
    loader,
    device,
    n_samples: int = 2000,
    perplexity: float = 30.0,
    n_iter: int = 2000,
    save_path: str = "tsne.png",
    class_names: list[str] | None = None,
):
    # 1. Extract features
    backbone.eval().to(device)
    feats, labels = [], []
    with torch.inference_mode():
        for x, y in loader:
            h = backbone(x.to(device)).cpu()
            feats.append(h)
            labels.append(y)
            if sum(f.shape[0] for f in feats) >= n_samples:
                break
    feats = torch.cat(feats)[:n_samples]
    labels = torch.cat(labels)[:n_samples].numpy()

    # 2. PCA pre-reduction (recommended before t-SNE)
    feats_np = F.normalize(feats, dim=1).numpy()
    from sklearn.decomposition import PCA
    if feats_np.shape[1] > 50:
        feats_np = PCA(n_components=50).fit_transform(feats_np)

    # 3. t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        learning_rate="auto",
        init="pca",
        metric="cosine",
        random_state=42,
    )
    embedding_2d = tsne.fit_transform(feats_np)

    # 4. Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                         c=labels, cmap="tab10", s=4, alpha=0.7)
    if class_names:
        legend = ax.legend(*scatter.legend_elements(),
                           title="Classes", labels=class_names)
        ax.add_artist(legend)
    ax.set_title(f"t-SNE (perplexity={perplexity})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return embedding_2d
```

### When to Run

t-SNE is expensive (O(N^2) in naive implementation; O(N log N) with BH approximation). Do not run as a per-epoch callback. Recommended trigger points:

- End of pretraining
- After major phase milestones (e.g., epoch 50, 100, 200)
- As a one-off diagnostic script: `python eval/tsne_vis.py --checkpoint path/to/ckpt.ckpt`

### Integration with YAML Config

```yaml
eval:
  tsne:
    enabled: true
    n_samples: 2000
    perplexity: 30.0
    n_iter: 2000
    perplexities_to_sweep: [10, 30, 50]  # run multiple for robustness check
    save_dir: "outputs/tsne/"
```

---

## UMAP Visualization

### What It Is

UMAP is the preferred alternative to t-SNE for SSL embeddings. It is faster, more stable across runs, preserves global structure better, and supports new sample mapping (transform new points into an existing fit). Use UMAP as the default visualization; use t-SNE as a cross-check.

### Key Hyperparameters

| Parameter | Default | Recommendation |
|-----------|---------|----------------|
| `n_neighbors` | 15 | Controls local vs. global balance; increase to 30–50 for larger datasets or to reveal global topology |
| `min_dist` | 0.1 | Lower (0.0–0.05) for tight clusters; higher (0.3–0.5) for continuous manifold layout |
| `n_components` | 2 | Use 2 for visualization; 3 for interactive 3D plots |
| `metric` | `'cosine'` | Always for SSL embeddings; UMAP natively supports cosine |
| `random_state` | 42 | Set for reproducibility |

**Parameter interaction:** `n_neighbors` is the more important parameter. Start with 15 (default) and 30 (global structure). Keep `min_dist=0.1` until you have a good `n_neighbors`.

### Implementation Pattern

```python
# eval/umap_vis.py
import numpy as np
import matplotlib.pyplot as plt
import umap
import torch, torch.nn.functional as F

def run_umap(
    backbone,
    loader,
    device,
    n_samples: int = 5000,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    save_path: str = "umap.png",
    class_names: list[str] | None = None,
):
    # 1. Feature extraction (same as t-SNE)
    backbone.eval().to(device)
    feats, labels = [], []
    with torch.inference_mode():
        for x, y in loader:
            h = backbone(x.to(device)).cpu()
            feats.append(h)
            labels.append(y)
            if sum(f.shape[0] for f in feats) >= n_samples:
                break
    feats = F.normalize(torch.cat(feats)[:n_samples], dim=1).numpy()
    labels = torch.cat(labels)[:n_samples].numpy()

    # 2. UMAP reduction
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=42,
    )
    embedding_2d = reducer.fit_transform(feats)

    # 3. Plot (same as t-SNE)
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                         c=labels, cmap="tab10", s=4, alpha=0.7)
    if class_names:
        ax.legend(*scatter.legend_elements(), title="Classes", labels=class_names)
    ax.set_title(f"UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return embedding_2d, reducer  # return reducer for new-sample mapping
```

### t-SNE vs. UMAP Comparison

| Criterion | t-SNE | UMAP |
|-----------|-------|------|
| Speed (5K samples, 512-dim) | ~20–60s (BH) | ~5–15s |
| Global structure | Poor | Good |
| Local cluster separation | Excellent | Good |
| Reproducibility | Stochastic (set seed) | More stable |
| New sample mapping | No | Yes (`reducer.transform(new_feats)`) |
| Recommended default | Cross-check | Primary |

### GPU-Accelerated Option

`torchdr` provides a PyTorch-backed UMAP that runs on GPU:

```python
from torchdr import UMAP as TorchUMAP
reducer = TorchUMAP(n_neighbors=15, min_dist=0.1, device="cuda")
embedding = reducer.fit_transform(feats_tensor)  # feats_tensor on GPU
```

Use `torchdr` when dataset exceeds 50K samples or when CPU UMAP is too slow.

### Integration with YAML Config

```yaml
eval:
  umap:
    enabled: true
    n_samples: 5000
    n_neighbors: 15
    min_dist: 0.1
    save_dir: "outputs/umap/"
```

---

## Fine-tuning Protocol

### What It Is

Unfreeze the backbone (or a portion of it) and jointly train backbone + classification head on the downstream labeled dataset. Measures how much the SSL representations can be adapted to a specific task. This is distinct from linear probing — it allows the backbone to change.

### Standard Protocol

| Hyperparameter | Small dataset (< 10K) | ImageNet fine-tune |
|----------------|----------------------|--------------------|
| Epochs | 50–100 | 30–90 |
| Optimizer | AdamW | SGD or AdamW |
| LR (head) | 1e-3 | 1e-3 |
| LR (backbone) | 1e-4 (10× lower) | 1e-4 |
| LR schedule | Cosine w/ warmup | Cosine w/ warmup (500 steps) |
| Weight decay | 1e-4 | 1e-4 |
| Warmup | 5–10 epochs | 500 steps |
| BatchNorm | Keep frozen (eval mode) if dataset is small | Unfreeze if dataset > 50K |

**Key design decision: separate LR for backbone and head.** The backbone has already learned good features — using the same LR as the head causes catastrophic forgetting of pretraining.

### Implementation Pattern

```python
# eval/finetune.py
import torch
import torch.nn as nn
import lightning as L

class FinetuneModule(L.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        num_classes: int,
        backbone_lr: float = 1e-4,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        warmup_epochs: int = 10,
        freeze_bn: bool = True,     # keep BN in eval mode to avoid noisy stats
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.freeze_bn = freeze_bn

    def on_train_epoch_start(self):
        # Keep BN layers in eval mode if dataset is small
        if self.freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()

    def forward(self, x):
        h = self.backbone(x)
        return self.classifier(h)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log_dict({"train/loss": loss, "train/acc": acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log_dict({"val/loss": loss, "val/acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        # Separate parameter groups with different LRs
        optimizer = torch.optim.AdamW([
            {"name": "backbone", "params": self.backbone.parameters(),
             "lr": self.hparams.backbone_lr},
            {"name": "head",     "params": self.classifier.parameters(),
             "lr": self.hparams.head_lr},
        ], weight_decay=self.hparams.weight_decay)
        from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
```

### Gradual Unfreezing Option

For very small downstream datasets (< 1K samples), gradual unfreezing helps prevent catastrophic forgetting:

1. Epoch 1–10: freeze backbone, train head only (linear probe phase)
2. Epoch 11–30: unfreeze last 2 backbone blocks + head
3. Epoch 31+: unfreeze all layers

PyTorch Lightning's `BackboneFinetuning` callback implements this pattern. However, for a tutorial repository, a simpler two-phase approach (linear probe then full fine-tune) is clearer.

### Integration with YAML Config

```yaml
eval:
  finetune:
    enabled: true
    epochs: 100
    backbone_lr: 1.0e-4
    head_lr: 1.0e-3
    weight_decay: 1.0e-4
    warmup_epochs: 10
    freeze_bn: true     # set false for large datasets (> 50K)
```

---

## CAM Visualization

### What It Is

Class Activation Maps (CAM and its Grad-CAM variants) produce a spatial heatmap showing which image regions the model attends to when making a prediction. For SSL models, this answers: "does the backbone focus on semantically meaningful regions?"

### Recommended Library

Use `pytorch-grad-cam` (jacobgil/pytorch-grad-cam). It is the most feature-complete library, supports CNNs and Vision Transformers natively, and has an actively maintained API (HIGH confidence — verified against GitHub source).

```bash
pip install grad-cam
```

### Method Selection by Architecture

| Architecture | Recommended Method | Why |
|-------------|-------------------|-----|
| ResNet, ConvNet | `GradCAM` | Standard; targets `layer4[-1]` |
| ViT (patch-based) | `EigenCAM` | Does not need gradient (more stable for ViT); no class discrimination |
| ViT (class-discriminative) | `EigenGradCAM` or `GradCAM` with reshape_transform | Uses first PC of activations × gradients |
| Any (no labels needed) | `EigenCAM` | Works without a classifier; good for SSL where no downstream head exists yet |

**For SSL specifically:** EigenCAM is the right default because you often do not have a downstream classifier attached yet. It uses the first principal component of the final conv/attention output — no gradients needed.

### Implementation Pattern

```python
# eval/cam_vis.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_target_layer(backbone, arch: str):
    """Return the appropriate target layer for CAM."""
    if "resnet" in arch:
        return [backbone.layer4[-1]]
    elif "vit" in arch:
        # ViT: use the last transformer block's norm/attention output
        return [backbone.blocks[-1].norm1]
    else:
        raise ValueError(f"Unknown architecture: {arch}")

def visualize_cam(
    backbone: torch.nn.Module,
    classifier: torch.nn.Module | None,  # None → use EigenCAM (no labels needed)
    images: torch.Tensor,                # [B, C, H, W] normalized
    raw_images: np.ndarray,              # [B, H, W, C] in [0, 1] for overlay
    arch: str = "resnet50",
    method: str = "eigen",               # "grad" or "eigen"
    save_path: str = "cam.png",
    targets=None,
):
    # Build full model (backbone + optional classifier head)
    if classifier is not None:
        model = torch.nn.Sequential(backbone, classifier)
    else:
        model = backbone

    target_layers = get_target_layer(backbone, arch)

    CamClass = EigenCAM if method == "eigen" else GradCAM
    with CamClass(model=model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=images, targets=targets)

    # Overlay heatmaps on raw images
    fig, axes = plt.subplots(2, len(images), figsize=(4 * len(images), 8))
    for i, (raw, gcam) in enumerate(zip(raw_images, grayscale_cams)):
        visualization = show_cam_on_image(raw, gcam, use_rgb=True)
        axes[0, i].imshow(raw)
        axes[0, i].axis("off")
        axes[1, i].imshow(visualization)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Input")
    axes[1, 0].set_ylabel("CAM")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
```

### ViT Reshape Transform

ViT attention outputs have shape `[B, N_patches + 1, D]` (the +1 is the CLS token), not `[B, D, H, W]`. GradCAM needs spatial dimensions. Use the reshape transform:

```python
def vit_reshape_transform(tensor, height=14, width=14):
    # Remove CLS token, reshape patch tokens to spatial grid
    result = tensor[:, 1:, :]                    # [B, N, D] → remove CLS
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)  # [B, D, H, W]
    return result

# Usage:
with GradCAM(model=model, target_layers=target_layers,
             reshape_transform=vit_reshape_transform) as cam:
    ...
```

### When to Run

CAM visualization is a diagnostic tool, not a metric. Run it:

- On a fixed set of 8–16 reference images from each class
- After pretraining (sanity check: does the backbone look at the right things?)
- After fine-tuning (compare: does fine-tuning sharpen attention?)
- As a one-off script; do not run inside a callback

### Integration with YAML Config

```yaml
eval:
  cam:
    enabled: true
    method: "eigen"          # "eigen" (no classifier needed) or "grad" (needs classifier)
    n_images: 8              # how many reference images per class
    save_dir: "outputs/cam/"
```

---

## Integration with YAML Config

### Design Principle

The evaluation config sits under a top-level `eval:` key in the same YAML used for pretraining. This keeps experiment configs self-contained — one YAML file, one run, one set of results.

### Full Config Schema (Pydantic v2)

Consistent with the existing STACK.md config pattern:

```python
# configs/schema.py  (extend existing TrainConfig)
from pydantic import BaseModel, Field
from typing import Literal

class LinearProbeConfig(BaseModel):
    enabled: bool = True
    epochs: int = 100
    lr: float = 0.1
    batch_size: int = 256
    optimizer: Literal["sgd", "lars"] = "sgd"
    weight_decay: float = 0.0
    lr_decay_steps: list[int] = Field(default=[60, 80])

class KNNConfig(BaseModel):
    enabled: bool = True
    k: int = 200
    temperature: float = 0.07
    every_n_epochs: int = 5   # 0 = end-of-training only

class TSNEConfig(BaseModel):
    enabled: bool = True
    n_samples: int = 2000
    perplexity: float = 30.0
    n_iter: int = 2000
    perplexities_to_sweep: list[float] = Field(default=[10.0, 30.0, 50.0])
    save_dir: str = "outputs/tsne/"

class UMAPConfig(BaseModel):
    enabled: bool = True
    n_samples: int = 5000
    n_neighbors: int = 15
    min_dist: float = 0.1
    save_dir: str = "outputs/umap/"

class FinetuneConfig(BaseModel):
    enabled: bool = False
    epochs: int = 100
    backbone_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    freeze_bn: bool = True

class CAMConfig(BaseModel):
    enabled: bool = False
    method: Literal["eigen", "grad"] = "eigen"
    n_images: int = 8
    save_dir: str = "outputs/cam/"

class EvalConfig(BaseModel):
    linear_probe: LinearProbeConfig = Field(default_factory=LinearProbeConfig)
    knn: KNNConfig = Field(default_factory=KNNConfig)
    tsne: TSNEConfig = Field(default_factory=TSNEConfig)
    umap: UMAPConfig = Field(default_factory=UMAPConfig)
    finetune: FinetuneConfig = Field(default_factory=FinetuneConfig)
    cam: CAMConfig = Field(default_factory=CAMConfig)

# Add to TrainConfig:
# eval: EvalConfig = Field(default_factory=EvalConfig)
```

### Full Example YAML

```yaml
# configs/simclr_cifar10.yaml
method: simclr
backbone: resnet50
max_epochs: 200
batch_size: 256
lr: 3.0e-4
weight_decay: 1.0e-6

simclr:
  temperature: 0.07
  proj_hidden_dim: 2048
  proj_out_dim: 128

eval:
  linear_probe:
    enabled: true
    epochs: 100
    lr: 0.1
    lr_decay_steps: [60, 80]
  knn:
    enabled: true
    k: 200
    temperature: 0.07
    every_n_epochs: 10
  tsne:
    enabled: true
    n_samples: 2000
    perplexity: 30.0
  umap:
    enabled: true
    n_samples: 5000
  finetune:
    enabled: false    # disabled by default; enable explicitly
  cam:
    enabled: false    # disabled by default; run as separate diagnostic
```

### Entry Point Pattern

```python
# eval/run_eval.py
import yaml
from configs.schema import TrainConfig

def run_evaluation(cfg: TrainConfig, checkpoint_path: str):
    backbone, feat_dim = load_backbone_from_checkpoint(checkpoint_path, cfg)

    if cfg.eval.knn.enabled:
        run_knn(backbone, cfg.eval.knn, ...)

    if cfg.eval.linear_probe.enabled:
        run_linear_probe(backbone, feat_dim, cfg.eval.linear_probe, ...)

    if cfg.eval.tsne.enabled:
        run_tsne(backbone, cfg.eval.tsne, ...)

    if cfg.eval.umap.enabled:
        run_umap(backbone, cfg.eval.umap, ...)

    if cfg.eval.finetune.enabled:
        run_finetune(backbone, feat_dim, cfg.eval.finetune, ...)

    if cfg.eval.cam.enabled:
        run_cam(backbone, cfg.eval.cam, ...)
```

---

## Standard Benchmarks and Protocols

### Published Numbers to Target

These are the reference numbers from major SSL papers on standard benchmarks. Use them to sanity-check your implementation.

#### CIFAR-10 (ResNet-18, 200 epochs pretraining)

| Method | Linear Probe Top-1 | k-NN Top-1 |
|--------|--------------------|-----------|
| SimCLR | ~91–92% | ~89% |
| MoCo v2 | ~92% | ~89% |
| BYOL | ~92–93% | ~90% |
| Barlow Twins | ~92% | ~89% |
| SimSiam | ~90–91% | ~88% |

Source: solo-learn benchmark tables (ImageNet-100 is closer to reported paper numbers; CIFAR-10 numbers vary by augmentation).

#### ImageNet-100 (ResNet-50, 400 epochs pretraining)

| Method | Linear Probe Top-1 |
|--------|--------------------|
| SimCLR | ~79–80% |
| MoCo v2 | ~79% |
| BYOL | ~80–81% |
| Barlow Twins | ~80% |

#### ImageNet-1K (ResNet-50, 800 epochs pretraining)

| Method | Linear Probe Top-1 | Fine-tune Top-1 |
|--------|--------------------|-----------------|
| SimCLR v2 | 71.7% | 79.8% |
| MoCo v3 | 73.0% | — |
| BYOL | 74.3% | — |
| DINO (ViT-S/16) | 77.0% | — |

### Linear Probe Training Standard

The de facto standard as of 2025 (used by SimCLR, MoCo v2, BYOL, SwAV, VICReg):

- **Epochs:** 90–100
- **Optimizer:** SGD (momentum 0.9) or LARS
- **LR:** 0.1 with step decay at epochs [60, 80] × 0.1, OR cosine decay from 0.1
- **Batch size:** 256 (scale LR proportionally if changing)
- **Weight decay:** 0.0
- **BatchNorm:** Keep in eval mode (frozen BN stats)
- **Data augmentation:** Standard (RandomCrop + HorizontalFlip only — no color jitter)

### k-NN Evaluation Standard

- **k:** 20 (CIFAR), 200 (ImageNet-scale)
- **Temperature:** 0.07
- **Feature normalization:** L2-normalize before cosine similarity
- **Feature bank:** Full training set encoded with frozen backbone
- **No training involved** — pure retrieval

### Fine-tuning Standard

For ImageNet transfer (from SSL → supervised downstream):

- **Epochs:** 50–100
- **Optimizer:** AdamW or SGD
- **LR ratio (backbone/head):** 1/10 (backbone 10× lower than head)
- **Warmup:** 5–10 epochs linear warmup
- **Schedule:** Cosine decay after warmup
- **BatchNorm:** Unfreeze for large datasets; keep frozen for small

### Evaluation Execution Order (Recommended)

Run evaluations in this order to minimize redundant forward passes:

1. **k-NN** (zero training, cheapest — run first as sanity check during pretraining)
2. **t-SNE / UMAP** (one-time feature extraction, shared with k-NN if features are cached)
3. **Linear probe** (100 epochs, offline after pretraining)
4. **Fine-tuning** (most expensive — run last, only when linear probe is solid)
5. **CAM** (diagnostic — run once on a few reference images)

### Shared Feature Extraction

To avoid extracting features multiple times for k-NN, t-SNE, and UMAP, extract once and reuse:

```python
# Extract features once, pass to all downstream evaluators
train_feats, train_labels = extract_features(backbone, train_loader, device)
val_feats, val_labels     = extract_features(backbone, val_loader, device)

# Pass to k-NN
knn_acc = knn_from_features(train_feats, train_labels, val_feats, val_labels)

# Pass to t-SNE (subsample)
run_tsne_from_features(val_feats[:2000], val_labels[:2000], ...)

# Pass to UMAP (subsample)
run_umap_from_features(val_feats[:5000], val_labels[:5000], ...)
```

This pattern reduces total evaluation runtime significantly: one forward pass through the dataset instead of three.

---

## Additional Dependencies

The following packages must be added to requirements for the evaluation suite:

```bash
# Core evaluation
pip install scikit-learn>=1.2          # t-SNE, LogisticRegression, metrics
pip install umap-learn>=0.5            # UMAP
pip install grad-cam>=1.4              # pytorch-grad-cam (GradCAM, EigenCAM)
pip install faiss-cpu>=1.7             # or faiss-gpu for large-scale k-NN

# Optional: GPU-accelerated UMAP
pip install torchdr                    # PyTorch-backed UMAP on GPU

# Visualization
pip install matplotlib>=3.7
pip install seaborn>=0.12              # for polished scatter plots (optional)
```

| Package | Purpose | Confidence |
|---------|---------|-----------|
| `scikit-learn` | t-SNE, LogisticRegression, metrics | HIGH |
| `umap-learn` | UMAP dimensionality reduction | HIGH |
| `grad-cam` (jacobgil) | GradCAM + EigenCAM | HIGH |
| `faiss-cpu` | Efficient k-NN at scale | HIGH |
| `torchdr` | GPU UMAP | MEDIUM (newer library, less battle-tested) |

---

## Sources

- solo-learn offline linear eval tutorial: https://github.com/vturrisi/solo-learn/blob/main/docs/source/tutorials/offline_linear_eval.rst (HIGH confidence)
- solo-learn README + evaluation scripts: https://github.com/vturrisi/solo-learn (HIGH confidence)
- "A Closer Look at Benchmarking SSL Pre-training" (IJCV 2025): https://arxiv.org/abs/2407.12210 (HIGH confidence — peer-reviewed)
- PyTorch Lightning evaluation / predict: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html (HIGH confidence)
- PyTorch Lightning transfer learning: https://lightning.ai/docs/pytorch/stable/advanced/pretrained.html (HIGH confidence)
- pytorch-grad-cam library: https://github.com/jacobgil/pytorch-grad-cam (HIGH confidence)
- "Attention Guided CAM" for ViT (arXiv 2402.04563, 2024): https://arxiv.org/abs/2402.04563 (MEDIUM confidence — preprint)
- UMAP documentation (basic parameters): https://umap-learn.readthedocs.io/en/latest/parameters.html (HIGH confidence)
- "How to Use t-SNE Effectively" (distill.pub): https://distill.pub/2016/misread-tsne/ (HIGH confidence — canonical reference)
- MoCo v3 empirical study (ICCV 2021): https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_An_Empirical_Study_of_Training_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf (HIGH confidence)
- LightlySSL benchmark docs: https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html (HIGH confidence)
- PyTorch Lightning KNN callback (gist): https://gist.github.com/isaaccorley/92d32c1cd818251f70996ea04ba83d1b (MEDIUM confidence — community code)
- FAISS documentation: https://faiss.ai/index.html (HIGH confidence)
- scDEED hyperparameter optimization for t-SNE/UMAP (Nature Comms 2024): https://pmc.ncbi.nlm.nih.gov/articles/PMC10897166/ (HIGH confidence — peer-reviewed)
