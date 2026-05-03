"""Fine-tuning evaluation for SSL checkpoints.

Unfreezes the backbone and trains the full model with differentiated
learning rates: backbone at a lower LR to prevent catastrophic forgetting,
classification head at a higher LR.

Usage:
    python eval/finetune.py configs/simclr_v1_resnet18.yaml --ckpt outputs/run/checkpoints/epoch-99.ckpt
"""
from __future__ import annotations

# B1 fix (phase 10.1): allow `python eval/finetune.py ...` from repo root
# to find sibling `core` and `methods` packages without an editable install.
# Reference: https://alex.dzyoba.com/blog/python-import/
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import math
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from core.config import TrainConfig, FinetuneConfig
from core.data import SSLDataModule
from core.dispatcher import get_method


# ---------------------------------------------------------------------------
# FinetuneModule
# ---------------------------------------------------------------------------

class FinetuneModule(L.LightningModule):
    """Lightning module for fine-tuning an SSL backbone on a classification task.

    Uses differentiated learning rates:
      - Backbone at ``ft_cfg.backbone_lr`` (default 1e-4) to prevent
        catastrophic forgetting of SSL-learned representations.
      - Classification head at ``ft_cfg.head_lr`` (default 1e-3) for
        faster adaptation.

    Optionally keeps BatchNorm layers frozen (``ft_cfg.freeze_bn=True``)
    to preserve batch statistics learned during SSL pre-training.

    Args:
        backbone: Pretrained backbone module (e.g., ResNet from SSL checkpoint).
        feat_dim: Feature dimension output by the backbone.
        num_classes: Number of target classes for the classification head.
        ft_cfg: Fine-tuning configuration (LRs, freeze_bn flag).
        max_epochs: Total training epochs (used for cosine schedule).
        warmup_epochs: Number of warmup epochs (proportional to total steps).
    """

    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        num_classes: int,
        ft_cfg: FinetuneConfig,
        max_epochs: int = 100,
        warmup_epochs: int = 10,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feat_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.ft_cfg = ft_cfg
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.save_hyperparameters(ignore=["backbone"])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    # ------------------------------------------------------------------
    # Training / Validation steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        imgs, labels = batch[0], batch[1]
        # Handle multi-view batches: SSLDataModule ssl_collate_fn produces
        # shape [n_views, B, C, H, W]. Use first view: imgs[0] -> [B, C, H, W].
        if imgs.ndim == 5:
            imgs = imgs[0]
        logits = self(imgs)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch[0], batch[1]
        # Handle multi-view batches: SSLDataModule ssl_collate_fn produces
        # shape [n_views, B, C, H, W]. Use first view: imgs[0] -> [B, C, H, W].
        if imgs.ndim == 5:
            imgs = imgs[0]
        logits = self(imgs)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """AdamW with separate LR groups + warmup-cosine schedule."""
        param_groups = [
            {"params": list(self.backbone.parameters()), "lr": self.ft_cfg.backbone_lr},
            {"params": list(self.head.parameters()), "lr": self.ft_cfg.head_lr},
        ]
        optimizer = AdamW(param_groups, weight_decay=1e-4)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.warmup_epochs / max(self.max_epochs, 1))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    # ------------------------------------------------------------------
    # freeze_bn override
    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        """Override to keep BatchNorm layers in eval mode when freeze_bn=True.

        After calling super().train(mode), iterate through backbone modules
        and set any BatchNorm layer back to eval mode. This preserves batch
        statistics from SSL pre-training during fine-tuning.
        """
        super().train(mode)
        if self.ft_cfg.freeze_bn and mode:
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
        return self


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an SSL checkpoint on a classification task."
    )
    parser.add_argument("config", type=str, help="Path to YAML training config")
    parser.add_argument("--ckpt", type=str, required=True, help="SSL checkpoint path")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for fine-tuned checkpoint (default: sibling of --ckpt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available, else cpu)",
    )
    return parser.parse_args()


def main() -> None:
    """Fine-tune an SSL checkpoint and report final val/acc."""
    args = get_args()

    # Load config
    with open(args.config) as f:
        cfg = TrainConfig.model_validate(yaml.safe_load(f))

    # Import methods to populate the dispatcher registry
    import methods  # noqa: F401

    # Load SSL checkpoint
    MethodClass = get_method(cfg.method)
    model = MethodClass.load_from_checkpoint(args.ckpt, cfg=cfg)
    model.eval()
    backbone = model.backbone

    # Feature dimension from backbone
    feat_dim: int = backbone.num_features

    # Build data module using the same kwargs train.py uses.
    # B2 fix (phase 10.1): the previous call `SSLDataModule(cfg)` passed the
    # entire TrainConfig as data_dir, silently using default n_views/batch_size
    # and crashing later in setup() with "expected str, bytes or os.PathLike object".
    dm = SSLDataModule(
        data_dir=cfg.data_dir,
        n_views=cfg.n_views,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    dm.setup("fit")

    # Infer num_classes from val dataset
    val_ds = dm.val_dataset
    if val_ds is None:
        raise RuntimeError(
            "No val/ split found in data_dir. Fine-tuning requires a labeled val split."
        )
    num_classes = len(val_ds.classes)

    # Fine-tuning config from YAML or defaults
    ft_cfg = (
        cfg.eval.finetune
        if cfg.eval is not None and cfg.eval.finetune is not None
        else FinetuneConfig()
    )

    # Output directory
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.ckpt).parent.parent / "finetune"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create fine-tune module
    ft_module = FinetuneModule(
        backbone=backbone,
        feat_dim=feat_dim,
        num_classes=num_classes,
        ft_cfg=ft_cfg,
        max_epochs=cfg.max_epochs,
        warmup_epochs=cfg.warmup_epochs,
    )

    # Train
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=args.device if args.device == "cpu" else "auto",
        default_root_dir=str(output_dir),
        enable_progress_bar=True,
    )
    trainer.fit(ft_module, datamodule=dm)

    # Report final val/acc
    results = trainer.logged_metrics
    val_acc = results.get("val/acc", results.get("val/acc_epoch", None))
    if val_acc is not None:
        print(f"Final val/acc: {val_acc:.4f}")


if __name__ == "__main__":
    main()
