"""Tests verifying collapse monitoring metrics are logged by all three no-negative modules."""
from __future__ import annotations

import torch
import pytest
from unittest.mock import MagicMock, patch

from core.config import TrainConfig, BYOLConfig, SimSiamConfig, BarlowTwinsConfig
from methods.byol.module import BYOLModule
from methods.simsiam.module import SimSiamModule
from methods.barlow_twins.module import BarlowTwinsModule


def make_batch(B: int = 4, C: int = 3, H: int = 32, W: int = 32):
    """Synthetic two-view batch: ((v1, v2), labels)."""
    v1 = torch.randn(B, C, H, W)
    v2 = torch.randn(B, C, H, W)
    return (v1, v2), torch.zeros(B, dtype=torch.long)


def collect_logs(module, batch):
    """Run training_step and collect all logged metrics without a trainer.

    Patches module.log to capture (key, value) pairs, and also patches
    log_train_metrics to avoid trainer dependency.
    """
    logged = {}

    def capture_log(key, val, **kwargs):
        logged[key] = val.item() if hasattr(val, "item") else float(val)

    module.train()
    module.log = capture_log
    module.log_train_metrics = MagicMock()
    module.training_step(batch, 0)
    return logged


def test_byol_logs_embedding_std():
    """BYOLModule.training_step must log 'train/embedding_std'."""
    cfg = TrainConfig(
        method="byol",
        backbone="resnet18",
        pretrained=False,
        max_epochs=1,
        batch_size=4,
        lr=0.001,
        byol=BYOLConfig(),
    )
    module = BYOLModule(cfg)
    logged = collect_logs(module, make_batch())
    assert "train/embedding_std" in logged, (
        "BYOLModule must log 'train/embedding_std' to enable collapse monitoring"
    )
    assert torch.isfinite(torch.tensor(logged["train/embedding_std"]))
    assert logged["train/embedding_std"] >= 0


def test_simsiam_logs_embedding_std():
    """SimSiamModule.training_step must log 'train/embedding_std'."""
    cfg = TrainConfig(
        method="simsiam",
        backbone="resnet18",
        pretrained=False,
        max_epochs=1,
        batch_size=4,
        lr=0.001,
        simsiam=SimSiamConfig(),
    )
    module = SimSiamModule(cfg)
    logged = collect_logs(module, make_batch())
    assert "train/embedding_std" in logged, (
        "SimSiamModule must log 'train/embedding_std' to enable collapse monitoring"
    )
    assert torch.isfinite(torch.tensor(logged["train/embedding_std"]))
    assert logged["train/embedding_std"] >= 0


def test_barlow_twins_logs_corr_diag_mean():
    """BarlowTwinsModule.training_step must log 'train/corr_diag_mean'."""
    cfg = TrainConfig(
        method="barlow_twins",
        backbone="resnet18",
        pretrained=False,
        max_epochs=1,
        batch_size=4,
        lr=0.001,
        barlow_twins=BarlowTwinsConfig(),
    )
    module = BarlowTwinsModule(cfg)
    logged = collect_logs(module, make_batch())
    assert "train/corr_diag_mean" in logged, (
        "BarlowTwinsModule must log 'train/corr_diag_mean' to enable collapse monitoring"
    )
    assert torch.isfinite(torch.tensor(logged["train/corr_diag_mean"]))


def test_embedding_std_is_finite_positive():
    """embedding_std logged by BYOLModule must be a finite non-negative scalar.

    Verifies with a larger batch (B=8) that the collapse metric is healthy
    (not NaN from degenerate std computation on edge cases).
    """
    cfg = TrainConfig(
        method="byol",
        backbone="resnet18",
        pretrained=False,
        max_epochs=1,
        batch_size=8,
        lr=0.001,
        byol=BYOLConfig(),
    )
    module = BYOLModule(cfg)
    logged = collect_logs(module, make_batch(B=8))
    val = logged["train/embedding_std"]
    assert not (val != val), f"embedding_std is NaN — check z.std(dim=0).mean() computation"
    assert val >= 0, f"embedding_std must be non-negative, got {val}"


def test_corr_diag_mean_in_valid_range():
    """corr_diag_mean logged by BarlowTwinsModule must be in [-1, 1].

    The cross-correlation matrix diagonal represents cosine-similarity-style
    correlation between the two views' embeddings. Each entry is bounded by
    [-1, 1] because inputs are L2-normalized before matrix multiplication.
    """
    cfg = TrainConfig(
        method="barlow_twins",
        backbone="resnet18",
        pretrained=False,
        max_epochs=1,
        batch_size=8,
        lr=0.001,
        barlow_twins=BarlowTwinsConfig(),
    )
    module = BarlowTwinsModule(cfg)
    logged = collect_logs(module, make_batch(B=8))
    val = logged["train/corr_diag_mean"]
    assert -1.0 <= val <= 1.0, (
        f"corr_diag_mean must be in [-1, 1] (L2-normalized inputs), got {val}"
    )
