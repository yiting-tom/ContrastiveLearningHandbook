"""Visualize the SSL augmentation pipeline.

Standalone CLI script that applies ContrastiveAugmentation to a single
image and saves a grid of augmented views. Use this to verify that
strong color jitter (s=1.0) and Gaussian blur are present in the
SimCLR augmentation pipeline.

Usage:
    python tools/visualize_augmentations.py --data-dir data --n-views 8
    python tools/visualize_augmentations.py --output tools/output/augmentation_grid.png
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torchvision.datasets import ImageFolder  # noqa: E402

# Add project root to path so core.data is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from core.data import ContrastiveAugmentation  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization: (C, H, W) float tensor -> (H, W, C) numpy."""
    # tensor shape: (C, H, W)
    for t, m, s in zip(tensor, IMAGENET_MEAN, IMAGENET_STD):
        t.mul_(s).add_(m)
    return tensor.clamp(0, 1).permute(1, 2, 0).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize SSL augmentation pipeline on a single image.",
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Path to ImageFolder-style dataset directory (default: data).",
    )
    parser.add_argument(
        "--output", default="tools/output/augmentation_grid.png",
        help="Output path for the augmentation grid image (default: tools/output/augmentation_grid.png).",
    )
    parser.add_argument(
        "--n-views", type=int, default=8,
        help="Number of augmented views to generate (default: 8).",
    )
    parser.add_argument(
        "--size", type=int, default=224,
        help="Crop size for augmentations (default: 224).",
    )
    parser.add_argument(
        "--strong", action=argparse.BooleanOptionalAction, default=True,
        help="Use strong (SimCLR) augmentation. --no-strong for weak/era-1 (default: True).",
    )
    args = parser.parse_args()

    # Load the first image from the dataset
    dataset = ImageFolder(args.data_dir)
    img, _ = dataset[0]  # PIL Image

    # Create augmentation (single-view mode)
    aug = ContrastiveAugmentation(size=args.size, strong=args.strong)

    # Generate n_views augmented versions
    views = [aug(img) for _ in range(args.n_views)]

    # Create grid
    n_cols = min(4, args.n_views)
    n_rows = (args.n_views + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Flatten axes for easy iteration
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    s_val = "1.0" if args.strong else "0.4"
    for i, ax in enumerate(axes):
        if i < len(views):
            img_np = denormalize(views[i].clone())
            ax.imshow(img_np)
            if i == 0:
                ax.set_title(f"Strong={args.strong}, s={s_val}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()

    # Save
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=100, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved augmentation grid to {args.output}")


if __name__ == "__main__":
    main()
