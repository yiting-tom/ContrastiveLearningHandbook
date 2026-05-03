"""t-SNE visualization of SSL representations.

Runs PCA pre-reduction to 50 dimensions, then t-SNE with cosine metric.
Sweeps multiple perplexity values and saves one PNG per perplexity.

WARNING: t-SNE preserves local structure only. Do NOT interpret global
distances or cluster spacing. Different perplexity values emphasize
different scales of structure -- always compare multiple values.

Usage:
    python eval/tsne_vis.py configs/simclr_v1_resnet18.yaml --ckpt outputs/run/checkpoints/epoch-99.ckpt
"""
from __future__ import annotations

# B1 fix (phase 10.1): allow `python eval/tsne_vis.py ...` from repo root
# to find sibling `core` and `methods` packages without an editable install.
# Reference: https://alex.dzyoba.com/blog/python-import/
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def extract_features(
    backbone: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    n_samples: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract L2-normalized features from backbone using a dataloader.

    Iterates the dataloader until n_samples are collected, then subsamples
    to exactly n_samples. Handles multi-view batches (5-dim tensors) by
    taking the first view.

    Args:
        backbone: Feature extractor (no classification head).
        dataloader: DataLoader yielding (images, labels) batches.
        n_samples: Maximum number of samples to collect.
        device: Device string ('cpu', 'cuda', etc.).

    Returns:
        Tuple of (features, labels) as float32 numpy arrays.
        features shape: [min(dataset_size, n_samples), feat_dim]
        labels shape: [min(dataset_size, n_samples)]
    """
    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            imgs, labels = batch[0], batch[1]
            # Handle multi-view augmented batches (shape [B, n_views, C, H, W])
            if imgs.ndim == 5:
                imgs = imgs[:, 0]  # take first view
            feats = backbone(imgs.to(device))
            feats = F.normalize(feats, dim=1)
            all_features.append(feats.cpu())
            all_labels.append(labels.cpu())

            if sum(f.shape[0] for f in all_features) >= n_samples:
                break

    features = torch.cat(all_features)
    labels = torch.cat(all_labels)

    # Subsample to exactly n_samples if we collected more
    if features.shape[0] > n_samples:
        idx = torch.randperm(features.shape[0])[:n_samples]
        features = features[idx]
        labels = labels[idx]

    return features.numpy(), labels.numpy()


def run_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    perplexities: list[int],
    output_dir: Path,
    n_samples: Optional[int] = None,
) -> list[Path]:
    """Run t-SNE on features and save one PNG per perplexity value.

    Applies PCA pre-reduction to 50 dimensions when input dim > 50 for
    faster and more stable t-SNE convergence. Uses cosine metric with
    PCA initialization and automatic learning rate.

    Args:
        features: Feature matrix, shape [N, D].
        labels: Integer class labels, shape [N].
        perplexities: List of perplexity values to sweep.
        output_dir: Directory where PNG files will be saved.
        n_samples: If set, subsample features to this many rows before
            running t-SNE. Useful when calling run_tsne directly with a
            large pre-extracted feature set.

    Returns:
        List of Path objects for the saved PNG files (one per perplexity).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional subsampling (for direct calls with large arrays)
    if n_samples is not None and features.shape[0] > n_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(features.shape[0], size=n_samples, replace=False)
        features = features[idx]
        labels = labels[idx]

    # PCA pre-reduction: reduces dimensionality before t-SNE for speed and stability
    # Cap n_components to min(50, n_samples-1) to handle small synthetic datasets
    if features.shape[1] > 50:
        n_pca = min(50, features.shape[0] - 1, features.shape[1])
        if n_pca >= 2:
            features = PCA(n_components=n_pca, random_state=42).fit_transform(features)

    # B4 fix (phase 10.1): scikit-learn >= 1.2 enforces perplexity < n_samples
    # strictly. Guard against pathological inputs and clamp per-perplexity below.
    n = features.shape[0]
    if n < 4:
        raise ValueError(
            f"t-SNE requires at least 4 samples; got {n}. "
            "Increase n_samples or use a larger dataset."
        )

    saved_paths: list[Path] = []
    for perplexity in perplexities:
        # B4 fix (phase 10.1): clamp to n_samples - 1 to avoid sklearn ValueError
        # on small datasets. The original requested perplexity is preserved in
        # the output filename for backward compatibility; the actual run uses
        # safe_perplexity. Reference:
        # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        safe_perplexity = min(perplexity, n - 1)
        if safe_perplexity != perplexity:
            print(
                f"warn: requested perplexity={perplexity} >= n_samples={n}; "
                f"clamping to {safe_perplexity}"
            )
        embedding = TSNE(
            n_components=2,
            perplexity=safe_perplexity,
            init="pca",
            metric="cosine",
            learning_rate="auto",
            random_state=42,
        ).fit_transform(features)

        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            cmap="tab10",
            s=5,
            alpha=0.7,
        )
        ax.set_title(f"t-SNE (perplexity={perplexity})")
        plt.colorbar(scatter, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])

        path = output_dir / f"tsne_perp{perplexity}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths


def get_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="t-SNE visualization of SSL representations"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML training config",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for output PNGs. "
            "Defaults to <ckpt_parent_parent>/tsne/"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for feature extraction (default: cuda if available)",
    )
    return parser.parse_args()


def main() -> None:
    """Run t-SNE visualization pipeline.

    Loads config + checkpoint, extracts features, runs t-SNE sweep, and
    saves PNGs to the output directory.
    """
    args = get_args()

    from core.config import TrainConfig, TSNEConfig
    from core.dispatcher import get_method
    import methods  # noqa: F401 -- triggers register_method() calls

    with open(args.config) as f:
        cfg = TrainConfig.model_validate(yaml.safe_load(f))
    tsne_cfg: TSNEConfig = (
        cfg.eval.tsne if cfg.eval and cfg.eval.tsne else TSNEConfig()
    )

    ckpt_path = Path(args.ckpt)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ckpt_path.parent.parent / "tsne"
    )

    print(f"Loading checkpoint: {ckpt_path}")
    MethodClass = get_method(cfg.method)
    model = MethodClass.load_from_checkpoint(str(ckpt_path), cfg=cfg)
    model.eval()
    model.to(args.device)
    backbone = model.backbone

    from torchvision.datasets import ImageFolder
    from torchvision import transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Auto-detect train/ subdir so data_dir can point to the train+val parent (matches SSLDataModule)
    data_root = Path(cfg.data_dir)
    if (data_root / "train").is_dir():
        data_root = data_root / "train"
    dataset = ImageFolder(root=str(data_root), transform=transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

    print(f"Extracting up to {tsne_cfg.n_samples} features ...")
    features, labels = extract_features(backbone, loader, tsne_cfg.n_samples, args.device)
    print(f"Features shape: {features.shape}")

    print(f"Running t-SNE sweep (perplexities={tsne_cfg.perplexities}) ...")
    paths = run_tsne(features, labels, tsne_cfg.perplexities, output_dir)

    for p in paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
