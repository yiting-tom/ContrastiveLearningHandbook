"""Linear probe evaluation for SSL checkpoints.

Freezes the pretrained backbone, pre-extracts features (cached to disk per D-04),
and trains a linear classifier head with SGD (weight_decay=0.0).

Cache location: {ckpt_path.parent.parent}/cache/
Cache filenames: {ckpt_stem}_features_{split}.pt, {ckpt_stem}_labels_{split}.pt

Usage:
    python eval/linear_probe.py configs/simclr_v1_resnet18.yaml --ckpt outputs/run/checkpoints/epoch-99.ckpt
"""
from __future__ import annotations

# B1 fix (phase 10.1): allow `python eval/linear_probe.py ...` from repo root
# to find sibling `core` and `methods` packages without an editable install.
# Reference: https://alex.dzyoba.com/blog/python-import/
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset
import yaml

from core.config import TrainConfig, LinearProbeConfig
from core.dispatcher import get_method


# ---------------------------------------------------------------------------
# Feature extraction + caching
# ---------------------------------------------------------------------------

def extract_and_cache(
    backbone: nn.Module,
    dataloader: DataLoader,
    cache_dir: Path,
    split: str,
    device: str,
    ckpt_path: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract backbone features and cache to disk; load from cache if already exists.

    Cache filenames include checkpoint stem to prevent collisions (D-04):
        {ckpt_stem}_features_{split}.pt
        {ckpt_stem}_labels_{split}.pt

    Args:
        backbone: Pretrained backbone (nn.Module) with ``requires_grad`` already set.
        dataloader: DataLoader yielding (imgs, labels) or ([view1, view2, ...], labels).
        cache_dir: Directory to store cached .pt files.
        split: Split name, e.g. "train" or "val".
        device: Device string for backbone inference.
        ckpt_path: Path to the checkpoint file (used to derive cache key stem).

    Returns:
        Tuple of (features, labels) tensors. Features are L2-normalized.
    """
    ckpt_stem = Path(ckpt_path).stem
    feat_path = cache_dir / f"{ckpt_stem}_features_{split}.pt"
    label_path = cache_dir / f"{ckpt_stem}_labels_{split}.pt"

    if feat_path.exists() and label_path.exists():
        features = torch.load(feat_path, weights_only=True)
        labels = torch.load(label_path, weights_only=True)
        return features, labels

    cache_dir.mkdir(parents=True, exist_ok=True)
    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            imgs, lbls = batch[0], batch[1]
            # Handle SSL multi-view batches: ssl_collate_fn produces [n_views, B, C, H, W]
            if isinstance(imgs, torch.Tensor) and imgs.ndim == 5:
                imgs = imgs[0]  # take the first view: shape [B, C, H, W]
            feats = backbone(imgs.to(device))
            all_features.append(feats.cpu())
            all_labels.append(lbls.cpu())

    features = F.normalize(torch.cat(all_features), dim=1)
    labels = torch.cat(all_labels)

    torch.save(features, feat_path)
    torch.save(labels, label_path)

    return features, labels


# ---------------------------------------------------------------------------
# LinearProbeModule
# ---------------------------------------------------------------------------

class LinearProbeModule(L.LightningModule):
    """Lightning module for linear probe evaluation.

    Trains a linear classification head on top of pre-extracted, frozen backbone
    features. Uses SGD with weight_decay=0.0 and MultiStepLR schedule.

    Args:
        feat_dim: Dimensionality of the pre-extracted features.
        num_classes: Number of output classes.
        lp_cfg: LinearProbeConfig containing lr, milestones, max_epochs.
    """

    def __init__(self, feat_dim: int, num_classes: int, lp_cfg: LinearProbeConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(feat_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lp_cfg = lp_cfg
        self.save_hyperparameters()

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Compute cross-entropy loss and log train/loss and train/acc."""
        features, labels = batch
        logits = self.linear(features)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean()
        self.log("train/loss", loss)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Compute cross-entropy loss and log val/loss and val/acc."""
        features, labels = batch
        logits = self.linear(features)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean()
        self.log("val/loss", loss)
        self.log("val/acc", acc, prog_bar=True)

    def configure_optimizers(self):
        """Return SGD (weight_decay=0.0) + MultiStepLR scheduler.

        weight_decay MUST be 0.0 for linear probing (EVAL-02 requirement).
        With frozen backbone features, regularizing the only learnable layer
        limits expressiveness unnecessarily.
        """
        optimizer = SGD(
            self.linear.parameters(),
            lr=self.lp_cfg.lr,
            weight_decay=0.0,
        )
        assert optimizer.param_groups[0]["weight_decay"] == 0.0, (
            "Linear probe requires weight_decay=0.0"
        )
        scheduler = MultiStepLR(optimizer, milestones=self.lp_cfg.milestones)
        return [optimizer], [scheduler]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with config, ckpt, output_dir, device.
    """
    parser = argparse.ArgumentParser(
        description="Linear probe evaluation for SSL checkpoints."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML training config (supplies eval.linear_probe settings)",
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
        help="Output directory (default: derived from checkpoint path)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for feature extraction (default: cuda if available, else cpu)",
    )
    return parser.parse_args()


def main() -> None:
    """Run linear probe evaluation end-to-end.

    1. Load config + checkpoint (frozen backbone).
    2. Pre-extract train/val features with checkpoint-keyed caching (D-04).
    3. Train LinearProbeModule on cached features.
    4. Print final val/acc.
    """
    args = get_args()

    # Load config
    with open(args.config) as f:
        cfg = TrainConfig.model_validate(yaml.safe_load(f))

    # Import methods package to trigger dispatcher registration
    import methods  # noqa: F401

    # Load checkpoint
    MethodClass = get_method(cfg.method)
    model = MethodClass.load_from_checkpoint(args.ckpt, cfg=cfg)
    model.eval()
    model.to(args.device)
    model.backbone.requires_grad_(False)

    # Resolve cache directory: sibling to checkpoints/ directory (D-04)
    cache_dir = Path(args.ckpt).parent.parent / "cache"

    # Build dataloaders via SSLDataModule using the same kwargs train.py uses.
    # B2 fix (phase 10.1): the previous call `SSLDataModule(cfg)` passed the
    # entire TrainConfig as data_dir, silently using default n_views/batch_size
    # and crashing later in setup() with "expected str, bytes or os.PathLike object".
    from core.data import SSLDataModule

    dm = SSLDataModule(
        data_dir=cfg.data_dir,
        n_views=cfg.n_views,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    dm.setup("fit")

    # Extract and cache features for train and val splits
    train_feats, train_labels = extract_and_cache(
        model.backbone,
        dm.train_dataloader(),
        cache_dir,
        "train",
        args.device,
        args.ckpt,
    )
    # B5 fix (phase 10.1): linear probe requires a val/ split for evaluation.
    # Without this guard, dm.val_dataloader() returns None when data_dir/val/
    # is missing, and the next extract_and_cache() crashes with the unhelpful
    # `'NoneType' object is not iterable` from inside the for-loop.
    val_loader = dm.val_dataloader()
    if val_loader is None:
        raise FileNotFoundError(
            f"Linear probe requires a 'val/' subdirectory under data_dir='{cfg.data_dir}'. "
            f"Either add data_dir/val/ with the same class structure as train/, or "
            f"reorganize your data to include a held-out split."
        )

    val_feats, val_labels = extract_and_cache(
        model.backbone,
        val_loader,
        cache_dir,
        "val",
        args.device,
        args.ckpt,
    )

    # Infer shapes
    num_classes = int(train_labels.max().item()) + 1
    feat_dim = train_feats.shape[1]

    print(f"Feature dim: {feat_dim}, num classes: {num_classes}")
    print(f"Train: {train_feats.shape}, Val: {val_feats.shape}")

    # Build cached-feature DataLoaders
    train_ds = TensorDataset(train_feats, train_labels)
    val_ds = TensorDataset(val_feats, val_labels)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    # Get linear probe config (use defaults if not in YAML)
    lp_cfg = (
        cfg.eval.linear_probe
        if cfg.eval and cfg.eval.linear_probe
        else LinearProbeConfig()
    )

    # Create and train linear probe
    lp_module = LinearProbeModule(feat_dim, num_classes, lp_cfg)
    trainer = L.Trainer(
        max_epochs=lp_cfg.max_epochs,
        accelerator="auto",
        logger=True,
    )
    trainer.fit(lp_module, train_loader, val_loader)

    # Report final accuracy
    results = trainer.validate(lp_module, val_loader, verbose=False)
    val_acc = results[0].get("val/acc", float("nan"))
    print(f"Final val/acc: {val_acc:.4f}")


if __name__ == "__main__":
    main()
