"""
Smoke test: SupCon stage-1 trains for 3 epochs on synthetic data.

Run with:
    python -m pytest tests/test_supcon_smoke.py -v -s
or directly:
    python tests/test_supcon_smoke.py
"""

import os
import tempfile

import lightning as L
import torch
from torch.utils.data import DataLoader

from core.config import TrainConfig
from core.data import ClassBalancedSampler, SSLDataModule, ssl_collate_fn
from core.dispatcher import get_method


class FakeTinyDataset(torch.utils.data.Dataset):
    """Synthetic 4-class dataset for smoke testing. No disk I/O required."""

    def __init__(self, n_per_class: int = 20, n_classes: int = 4, n_views: int = 2):
        self.targets = [c for c in range(n_classes) for _ in range(n_per_class)]
        self.n_views = n_views

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        label = self.targets[idx]
        # Random 3x32x32 image — values don't matter for smoke test
        views = [torch.randn(3, 32, 32) for _ in range(self.n_views)]
        return views, label


def test_supcon_stage1_three_epochs():
    """Stage-1 SupCon runs 3 epochs without NaN loss on synthetic data."""
    import methods.supcon  # trigger dispatcher registration

    cfg = TrainConfig.model_validate({
        "method": "supcon",
        "backbone": "resnet18",
        "pretrained": False,
        "max_epochs": 3,
        "warmup_epochs": 0,
        "batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "optimizer": "adamw",
        "n_views": 2,
        "data_dir": "data",
        "num_workers": 0,
        "supcon": {
            "temperature": 0.07,
            "n_samples_per_class": 2,
            "n_classes_per_batch": 4,
            "num_classes": 4,
            "projection_dim": 128,
        },
    })

    SupConModule = get_method("supcon")
    module = SupConModule(cfg)

    # Build synthetic dataset with ClassBalancedSampler
    dataset = FakeTinyDataset(n_per_class=10, n_classes=4, n_views=2)
    sampler = ClassBalancedSampler(
        dataset,
        n_classes_per_batch=4,
        n_samples_per_class=2,
    )
    loader = DataLoader(
        dataset,
        batch_size=8,
        sampler=sampler,
        shuffle=False,
        collate_fn=ssl_collate_fn,
        drop_last=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = L.Trainer(
            max_epochs=3,
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=True,
            default_root_dir=tmpdir,
        )
        trainer.fit(module, train_dataloaders=loader)

    # Verify the loss logged during training was finite
    train_losses = [
        v for k, v in trainer.logged_metrics.items()
        if "train/loss" in k or "train_loss" in k
    ]
    if train_losses:
        assert all(torch.isfinite(torch.tensor(v)) for v in train_losses), (
            f"NaN/Inf loss detected: {train_losses}"
        )
    print("Stage-1 3-epoch smoke test PASSED.")


if __name__ == "__main__":
    test_supcon_stage1_three_epochs()
