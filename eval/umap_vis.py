"""UMAP visualization of SSL representations.

Preferred over t-SNE as the primary visualization: faster, preserves more
global structure, and the reducer object can map new samples.

Usage:
    python eval/umap_vis.py configs/simclr_v1_resnet18.yaml --ckpt outputs/run/checkpoints/epoch-99.ckpt
"""
from __future__ import annotations

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
import umap


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


def run_umap(
    features: np.ndarray,
    labels: np.ndarray,
    metric: str,
    output_dir: Path,
    n_samples: Optional[int] = None,
) -> tuple[Path, umap.UMAP]:
    """Run UMAP on features and save a PNG visualization.

    Uses cosine metric with fixed random_state=42 for reproducibility.
    For datasets with > 50K samples, prints a note about torchdr for
    GPU-accelerated UMAP.

    Args:
        features: Feature matrix, shape [N, D].
        labels: Integer class labels, shape [N].
        metric: Distance metric for UMAP (e.g. 'cosine', 'euclidean').
        output_dir: Directory where the PNG will be saved.
        n_samples: If set, subsample features to this many rows before
            running UMAP. Useful when calling run_umap directly with a
            large pre-extracted feature set.

    Returns:
        Tuple of (path, reducer) where path is the saved PNG Path and
        reducer is the fitted umap.UMAP object (for mapping new samples).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional subsampling (for direct calls with large arrays)
    if n_samples is not None and features.shape[0] > n_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(features.shape[0], size=n_samples, replace=False)
        features = features[idx]
        labels = labels[idx]

    # Print torchdr suggestion for large datasets
    if features.shape[0] > 50_000:
        print(
            "Note: For datasets >50K samples, consider torchdr for "
            "GPU-accelerated UMAP: pip install torchdr"
        )

    reducer = umap.UMAP(metric=metric, random_state=42, n_components=2)
    embedding = reducer.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab10",
        s=5,
        alpha=0.7,
    )
    ax.set_title("UMAP")
    plt.colorbar(scatter, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])

    path = output_dir / "umap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return path, reducer


def get_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="UMAP visualization of SSL representations"
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
            "Directory for output PNG. "
            "Defaults to <ckpt_parent_parent>/umap/"
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
    """Run UMAP visualization pipeline.

    Loads config + checkpoint, extracts features, runs UMAP, and saves
    PNG to the output directory.
    """
    args = get_args()

    from core.config import TrainConfig, UMAPConfig
    from core.dispatcher import get_method
    import methods  # noqa: F401 -- triggers register_method() calls

    cfg = TrainConfig.model_validate(yaml.safe_load(open(args.config)))
    umap_cfg: UMAPConfig = (
        cfg.eval.umap if cfg.eval and cfg.eval.umap else UMAPConfig()
    )

    ckpt_path = Path(args.ckpt)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ckpt_path.parent.parent / "umap"
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
    dataset = ImageFolder(root=cfg.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

    print(f"Extracting up to {umap_cfg.n_samples} features ...")
    features, labels = extract_features(backbone, loader, umap_cfg.n_samples, args.device)
    print(f"Features shape: {features.shape}")

    print(f"Running UMAP (metric={umap_cfg.metric}) ...")
    path, _ = run_umap(features, labels, metric=umap_cfg.metric, output_dir=output_dir)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
