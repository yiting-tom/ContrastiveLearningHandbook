"""Unit tests for eval/linear_probe.py — LinearProbeModule and feature caching.

Tests cover:
  - LinearProbeModule init, optimizer config, training/validation steps
  - extract_and_cache: file creation, cache loading, L2-normalization
  - Checkpoint-keyed cache filenames (D-04)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

import pytest

from core.config import LinearProbeConfig
from eval.linear_probe import LinearProbeModule, extract_and_cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lp_cfg():
    """Return default LinearProbeConfig."""
    return LinearProbeConfig(max_epochs=100, lr=0.1, milestones=[60, 80])


@pytest.fixture
def module(lp_cfg):
    """Return a LinearProbeModule with feat_dim=32 and 3 classes."""
    return LinearProbeModule(feat_dim=32, num_classes=3, lp_cfg=lp_cfg)


@pytest.fixture
def simple_dataloader():
    """DataLoader yielding (imgs, labels) — plain 2D tensors, no SSL multi-view."""
    feats = torch.randn(20, 32)
    labels = torch.randint(0, 3, (20,))
    ds = TensorDataset(feats, labels)
    return DataLoader(ds, batch_size=5)


@pytest.fixture
def ssl_multiview_dataloader():
    """DataLoader yielding ([view1, ...], labels) — SSL multi-view format."""
    # Simulate two-view batch: batch[0] has shape [B, n_views, C, H, W] = [10, 2, 3, 8, 8]
    class FakeSSLDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, idx):
            # Return (views_tensor, label) where views_tensor is [n_views, C, H, W]
            views = torch.randn(2, 3, 8, 8)  # 2 views
            label = torch.tensor(idx % 3)
            return views, label

    return DataLoader(FakeSSLDataset(), batch_size=5)


@pytest.fixture
def tiny_backbone():
    """Tiny backbone that returns 32-dim features."""
    class TinyBackbone(nn.Module):
        def forward(self, x):
            # Flatten to 32-dim, normalize so output is reproducible
            b = x.shape[0]
            return torch.ones(b, 32)

    return TinyBackbone()


# ---------------------------------------------------------------------------
# Test 1: LinearProbeModule.__init__
# ---------------------------------------------------------------------------

def test_init_creates_linear_layer(module):
    """LinearProbeModule.__init__ creates nn.Linear(feat_dim, num_classes) and stores lp_cfg."""
    assert isinstance(module.linear, nn.Linear)
    assert module.linear.in_features == 32
    assert module.linear.out_features == 3
    assert isinstance(module.lp_cfg, LinearProbeConfig)


# ---------------------------------------------------------------------------
# Test 2: SGD with weight_decay=0.0
# ---------------------------------------------------------------------------

def test_configure_optimizers_sgd_weight_decay_zero(module):
    """configure_optimizers returns SGD with weight_decay=0.0."""
    result = module.configure_optimizers()
    optimizers, schedulers = result
    optimizer = optimizers[0]
    assert isinstance(optimizer, SGD)
    assert optimizer.param_groups[0]["weight_decay"] == 0.0


# ---------------------------------------------------------------------------
# Test 3: MultiStepLR with correct milestones
# ---------------------------------------------------------------------------

def test_configure_optimizers_multisteplr(module):
    """configure_optimizers returns MultiStepLR with milestones [60, 80]."""
    result = module.configure_optimizers()
    optimizers, schedulers = result
    scheduler = schedulers[0]
    assert isinstance(scheduler, MultiStepLR)
    assert scheduler.milestones == {60: 1, 80: 1}


# ---------------------------------------------------------------------------
# Test 4: training_step logs train/loss and train/acc
# ---------------------------------------------------------------------------

def test_training_step_logs_metrics(module):
    """training_step computes cross-entropy loss and logs train/loss and train/acc."""
    logged = {}

    def mock_log(name, value, **kwargs):
        logged[name] = value

    module.log = mock_log

    feats = torch.randn(8, 32)
    labels = torch.randint(0, 3, (8,))
    batch = (feats, labels)
    loss = module.training_step(batch, 0)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad or loss.item() >= 0.0
    assert "train/loss" in logged
    assert "train/acc" in logged
    assert 0.0 <= logged["train/acc"].item() <= 1.0


# ---------------------------------------------------------------------------
# Test 5: validation_step logs val/loss and val/acc
# ---------------------------------------------------------------------------

def test_validation_step_logs_metrics(module):
    """validation_step logs val/loss and val/acc."""
    logged = {}

    def mock_log(name, value, **kwargs):
        logged[name] = value

    module.log = mock_log

    feats = torch.randn(8, 32)
    labels = torch.randint(0, 3, (8,))
    batch = (feats, labels)
    module.validation_step(batch, 0)

    assert "val/loss" in logged
    assert "val/acc" in logged
    assert 0.0 <= logged["val/acc"].item() <= 1.0


# ---------------------------------------------------------------------------
# Test 6: extract_and_cache saves checkpoint-keyed files
# ---------------------------------------------------------------------------

def test_extract_and_cache_saves_files(tmp_path, tiny_backbone, simple_dataloader):
    """extract_and_cache saves {ckpt_stem}_features_{split}.pt and {ckpt_stem}_labels_{split}.pt."""
    ckpt_path = str(tmp_path / "epoch-42.ckpt")
    cache_dir = tmp_path / "cache"

    feats, labels = extract_and_cache(
        tiny_backbone, simple_dataloader, cache_dir, "train", "cpu", ckpt_path
    )

    feat_file = cache_dir / "epoch-42_features_train.pt"
    label_file = cache_dir / "epoch-42_labels_train.pt"

    assert feat_file.exists(), f"Expected {feat_file} to exist"
    assert label_file.exists(), f"Expected {label_file} to exist"
    assert feats.shape[1] == 32
    assert labels.shape[0] == feats.shape[0]


# ---------------------------------------------------------------------------
# Test 7: extract_and_cache loads from cache on second call
# ---------------------------------------------------------------------------

def test_extract_and_cache_loads_from_cache(tmp_path, tiny_backbone, simple_dataloader):
    """extract_and_cache loads from cache on second call (files already exist)."""
    ckpt_path = str(tmp_path / "epoch-42.ckpt")
    cache_dir = tmp_path / "cache"

    # First call — populates cache
    feats1, labels1 = extract_and_cache(
        tiny_backbone, simple_dataloader, cache_dir, "train", "cpu", ckpt_path
    )

    # Corrupt the backbone so if it's called again, output differs
    class BrokenBackbone(nn.Module):
        def forward(self, x):
            raise RuntimeError("Backbone should not be called on cache hit")

    feats2, labels2 = extract_and_cache(
        BrokenBackbone(), simple_dataloader, cache_dir, "train", "cpu", ckpt_path
    )

    assert torch.allclose(feats1, feats2)
    assert torch.equal(labels1, labels2)


# ---------------------------------------------------------------------------
# Test 8: Cached features are L2-normalized
# ---------------------------------------------------------------------------

def test_cached_features_are_l2_normalized(tmp_path, tiny_backbone, simple_dataloader):
    """Cached features are L2-normalized: torch.allclose(F.normalize(feats, dim=1), feats)."""
    ckpt_path = str(tmp_path / "epoch-42.ckpt")
    cache_dir = tmp_path / "cache"

    feats, _ = extract_and_cache(
        tiny_backbone, simple_dataloader, cache_dir, "train", "cpu", ckpt_path
    )

    # Check L2 normalization: each row should have norm ≈ 1.0
    norms = feats.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"Features are not L2-normalized; norms: {norms}"
    # Also check via F.normalize comparison
    assert torch.allclose(F.normalize(feats, dim=1), feats, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 9: Multi-view SSL batch handling in extract_and_cache
# ---------------------------------------------------------------------------

def test_extract_and_cache_handles_multiview_batch(tmp_path, tiny_backbone, ssl_multiview_dataloader):
    """extract_and_cache handles SSL multi-view batches (batch[0].ndim == 5)."""
    ckpt_path = str(tmp_path / "epoch-42.ckpt")
    cache_dir = tmp_path / "cache"

    # Override backbone to accept 4D input from multi-view batch
    class FlatBackbone(nn.Module):
        def forward(self, x):
            b = x.shape[0]
            return torch.ones(b, 32)

    feats, labels = extract_and_cache(
        FlatBackbone(), ssl_multiview_dataloader, cache_dir, "train", "cpu", ckpt_path
    )

    assert feats.shape[1] == 32
    assert labels.shape[0] == feats.shape[0]
