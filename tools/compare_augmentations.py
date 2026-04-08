"""Side-by-side augmentation comparison: SimCLR vs InfoMin.

Standalone CLI script that applies SimCLR and InfoMin augmentations to a
single source image and saves a side-by-side grid. Use this to visualize
the InfoMin principle: more aggressive augmentation (no blur, s=1.5) strips
low-level texture correlations while preserving semantic content.

Usage:
    python tools/compare_augmentations.py --image path/to/image.jpg
    python tools/compare_augmentations.py --image img.jpg --output /tmp/out.png --n-samples 4 --size 224
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")  # headless -- same as visualize_augmentations.py
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

# Add project root to path so core/methods are importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from core.data import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from methods.simclr.module import SimCLRv1Module  # noqa: E402
from methods.infomin.module import InfoMinModule  # noqa: E402

IMAGENET_MEAN_T = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
IMAGENET_STD_T = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def denormalize(tensor: torch.Tensor) -> "np.ndarray":
    """Reverse ImageNet normalization: (C, H, W) float tensor -> (H, W, C) numpy."""
    import numpy as np
    img = tensor.clone().float()
    img = img * IMAGENET_STD_T + IMAGENET_MEAN_T
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def compare(
    image_path: str,
    output_path: str,
    n_samples: int = 4,
    size: int = 224,
) -> None:
    """Generate side-by-side SimCLR vs InfoMin augmentation grid.

    Args:
        image_path: Path to source image file.
        output_path: Output path for the comparison grid PNG.
        n_samples: Number of augmented views per method.
        size: Crop size for augmentations.
    """
    # Load source image
    img = Image.open(image_path).convert("RGB")

    # Build augmentations using the build_augmentation() hooks
    simclr_aug = SimCLRv1Module.build_augmentation(size=size)
    infomin_aug = InfoMinModule.build_augmentation(size=size)

    # Generate augmented views
    simclr_views = [simclr_aug(img) for _ in range(n_samples)]
    infomin_views = [infomin_aug(img) for _ in range(n_samples)]

    # Build figure: 2 rows (SimCLR, InfoMin), (1 + n_samples) columns
    # Column 0 = original image, columns 1..n_samples = augmented views
    n_cols = 1 + n_samples
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    # Row 0: SimCLR
    # Row 1: InfoMin
    row_labels = [
        "SimCLR (s=1.0, blur=yes)",
        "InfoMin (s=1.5, blur=no)",
    ]
    rows_views = [simclr_views, infomin_views]

    for row_idx, (views, label) in enumerate(zip(rows_views, row_labels)):
        # Column 0: original image
        axes[row_idx, 0].imshow(img.resize((size, size)))
        if row_idx == 0:
            axes[row_idx, 0].set_title("Original", fontsize=10)
        axes[row_idx, 0].set_ylabel(label, fontsize=9, rotation=90, labelpad=5)
        axes[row_idx, 0].axis("off")

        # Columns 1..n_samples: augmented views
        for col_idx, view_tensor in enumerate(views):
            ax = axes[row_idx, 1 + col_idx]
            img_np = denormalize(view_tensor)
            ax.imshow(img_np)
            if row_idx == 0:
                ax.set_title(f"View {col_idx + 1}", fontsize=10)
            ax.axis("off")

    plt.tight_layout()

    # Save output
    out = pathlib.Path(output_path)
    os.makedirs(str(out.parent), exist_ok=True)
    fig.savefig(str(out), dpi=100, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved augmentation comparison to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Side-by-side SimCLR vs InfoMin augmentation comparison.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to source image file.",
    )
    parser.add_argument(
        "--output",
        default="tools/output/augmentation_comparison.png",
        help="Output path for the comparison grid (default: tools/output/augmentation_comparison.png).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=4,
        help="Number of augmented views to show per method (default: 4).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Crop size for augmentations (default: 224).",
    )
    args = parser.parse_args()

    compare(
        image_path=args.image,
        output_path=args.output,
        n_samples=args.n_samples,
        size=args.size,
    )


if __name__ == "__main__":
    main()
