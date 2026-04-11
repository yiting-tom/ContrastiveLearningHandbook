"""Unit tests for KNNCallback (EVAL-01).

Tests:
    1. KNNCallback.__init__ accepts KNNConfig and stores k, temperature, every_n_epochs
    2. on_validation_epoch_end skips when val_dataloader returns None
    3. on_validation_epoch_end skips when epoch interval not met
    4. on_validation_epoch_end runs when epoch interval is met
    5. every_n_epochs=0 only runs at final epoch
    6. knn_predict brute-force path returns correct accuracy on separable data
    7. knn_predict FAISS path produces same results as brute-force on identical data
    8. Features are L2-normalized before k-NN search
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from unittest.mock import MagicMock, patch

from core.config import KNNConfig
from eval.knn_callback import KNNCallback, knn_predict


# ---------------------------------------------------------------------------
# Test 1: KNNCallback.__init__ stores config fields
# ---------------------------------------------------------------------------

def test_knn_callback_init_stores_config():
    """KNNCallback.__init__ accepts KNNConfig and stores k, temperature, every_n_epochs."""
    cfg = KNNConfig(k=50, temperature=0.1, every_n_epochs=10)
    cb = KNNCallback(cfg)
    assert cb.k == 50
    assert cb.temperature == 0.1
    assert cb.every_n_epochs == 10


def test_knn_callback_init_default_config():
    """KNNCallback works with default KNNConfig values."""
    cfg = KNNConfig()
    cb = KNNCallback(cfg)
    assert cb.k == 200
    assert cb.temperature == 0.07
    assert cb.every_n_epochs == 5


# ---------------------------------------------------------------------------
# Test 2: Skips when val_dataloader returns None
# ---------------------------------------------------------------------------

def test_on_validation_epoch_end_skips_none_val_loader():
    """on_validation_epoch_end returns without error when val_dataloader returns None."""
    cfg = KNNConfig(k=5, temperature=0.07, every_n_epochs=1)
    cb = KNNCallback(cfg)

    trainer = MagicMock()
    trainer.current_epoch = 0
    trainer.max_epochs = 10
    trainer.datamodule.val_dataloader.return_value = None

    pl_module = MagicMock()

    # Should not raise
    cb.on_validation_epoch_end(trainer, pl_module)
    # pl_module.log should NOT have been called
    pl_module.log.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: Skips when epoch interval not met
# ---------------------------------------------------------------------------

def test_on_validation_epoch_end_skips_wrong_interval():
    """on_validation_epoch_end skips when (current_epoch+1) % every_n_epochs != 0."""
    cfg = KNNConfig(k=5, temperature=0.07, every_n_epochs=5)
    cb = KNNCallback(cfg)

    trainer = MagicMock()
    trainer.current_epoch = 2  # epoch 3 — not divisible by 5
    trainer.max_epochs = 20
    # val_dataloader returns something non-None
    trainer.datamodule.val_dataloader.return_value = MagicMock()

    pl_module = MagicMock()
    cb.on_validation_epoch_end(trainer, pl_module)
    pl_module.log.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: Runs when epoch interval is met
# ---------------------------------------------------------------------------

def _make_simple_loader(n_samples: int = 30, n_classes: int = 3, feat_dim: int = 8):
    """Create a simple dataloader-like object with synthetic (imgs, labels) batches."""
    feats = torch.randn(n_samples, 3, 4, 4)  # fake images
    labels = torch.arange(n_samples) % n_classes
    dataset = list(zip(
        feats.split(10),
        labels.split(10),
    ))
    return dataset


def test_on_validation_epoch_end_runs_at_interval():
    """on_validation_epoch_end calls pl_module.log when interval is met."""
    cfg = KNNConfig(k=3, temperature=0.07, every_n_epochs=5)
    cb = KNNCallback(cfg)

    # Build simple synthetic backbone
    backbone = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 4 * 4, 16),
    )
    backbone.eval()

    n_samples = 30
    n_classes = 3
    fake_imgs = torch.randn(n_samples, 3, 4, 4)
    fake_labels = torch.arange(n_samples) % n_classes

    # Batches of 10: list of (imgs_batch, labels_batch) tuples
    train_batches = list(zip(fake_imgs.split(10), fake_labels.split(10)))
    val_batches = list(zip(fake_imgs.split(10), fake_labels.split(10)))

    trainer = MagicMock()
    trainer.current_epoch = 4  # epoch 5, divisible by 5
    trainer.max_epochs = 20
    trainer.datamodule.val_dataloader.return_value = val_batches
    trainer.datamodule.train_dataloader.return_value = train_batches

    pl_module = MagicMock()
    pl_module.backbone = backbone
    pl_module.device = torch.device("cpu")

    cb.on_validation_epoch_end(trainer, pl_module)
    pl_module.log.assert_called_once()
    call_args = pl_module.log.call_args
    assert call_args[0][0] == "eval/knn_acc"
    acc = call_args[0][1]
    assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# Test 5: every_n_epochs=0 only runs at final epoch
# ---------------------------------------------------------------------------

def test_every_n_epochs_zero_runs_only_at_final():
    """When every_n_epochs=0, callback only runs at trainer.max_epochs - 1."""
    cfg = KNNConfig(k=3, temperature=0.07, every_n_epochs=0)
    cb = KNNCallback(cfg)

    trainer = MagicMock()
    trainer.max_epochs = 10
    trainer.datamodule.val_dataloader.return_value = None  # skip actual k-NN

    pl_module = MagicMock()

    # Should skip at epoch 0, 1, ..., 8
    for epoch in range(9):
        trainer.current_epoch = epoch
        cb.on_validation_epoch_end(trainer, pl_module)
    pl_module.log.assert_not_called()

    # Should run at epoch 9 (max_epochs - 1), but val is None so log still not called
    trainer.current_epoch = 9
    cb.on_validation_epoch_end(trainer, pl_module)
    pl_module.log.assert_not_called()


def test_every_n_epochs_zero_would_run_at_final():
    """When every_n_epochs=0, _should_run returns True only at max_epochs-1."""
    cfg = KNNConfig(k=3, temperature=0.07, every_n_epochs=0)
    cb = KNNCallback(cfg)

    trainer = MagicMock()
    trainer.max_epochs = 10

    trainer.current_epoch = 8
    assert cb._should_run(trainer) is False

    trainer.current_epoch = 9  # max_epochs - 1
    assert cb._should_run(trainer) is True


# ---------------------------------------------------------------------------
# Test 6: knn_predict brute-force path returns correct accuracy on separable data
# ---------------------------------------------------------------------------

def test_knn_predict_brute_force_separable():
    """knn_predict with brute-force returns high accuracy on trivially separable data."""
    torch.manual_seed(42)
    n_classes = 3
    n_per_class = 20
    feat_dim = 16

    # Create well-separated cluster centers
    centers = torch.eye(n_classes, feat_dim)
    train_feats = []
    train_labels = []
    for c in range(n_classes):
        feats = centers[c].unsqueeze(0) + 0.01 * torch.randn(n_per_class, feat_dim)
        train_feats.append(feats)
        train_labels.extend([c] * n_per_class)

    train_feats = F.normalize(torch.cat(train_feats), dim=1)
    train_labels = torch.tensor(train_labels)

    # Test features: same distribution
    test_feats = []
    test_labels = []
    for c in range(n_classes):
        feats = centers[c].unsqueeze(0) + 0.01 * torch.randn(10, feat_dim)
        test_feats.append(feats)
        test_labels.extend([c] * 10)

    test_feats = F.normalize(torch.cat(test_feats), dim=1)
    test_labels = torch.tensor(test_labels)

    # Force brute-force path: n_train=60 < 100_000
    assert train_feats.shape[0] < 100_000
    acc = knn_predict(
        train_feats,
        train_labels,
        test_feats,
        test_labels,
        k=5,
        temperature=0.07,
        num_classes=n_classes,
    )
    assert acc > 0.9, f"Expected high accuracy on separable data, got {acc}"


# ---------------------------------------------------------------------------
# Test 7: knn_predict FAISS path produces same results as brute-force
# ---------------------------------------------------------------------------

def test_knn_predict_faiss_vs_brute_force():
    """knn_predict with FAISS path should produce same result as brute-force on identical data."""
    torch.manual_seed(0)
    n_classes = 3
    n_train = 30
    n_test = 10
    feat_dim = 16
    k = 5
    temperature = 0.07

    train_feats = F.normalize(torch.randn(n_train, feat_dim), dim=1)
    train_labels = torch.arange(n_train) % n_classes
    test_feats = F.normalize(torch.randn(n_test, feat_dim), dim=1)
    test_labels = torch.arange(n_test) % n_classes

    # Brute-force result (n_train < 100_000)
    acc_brute = knn_predict(
        train_feats,
        train_labels,
        test_feats,
        test_labels,
        k=k,
        temperature=temperature,
        num_classes=n_classes,
    )

    # FAISS result: patch threshold to force FAISS path on small data
    with patch("eval.knn_callback.FAISS_THRESHOLD", 0):
        acc_faiss = knn_predict(
            train_feats,
            train_labels,
            test_feats,
            test_labels,
            k=k,
            temperature=temperature,
            num_classes=n_classes,
        )

    assert abs(acc_faiss - acc_brute) < 1e-4, (
        f"FAISS and brute-force paths should give same result. "
        f"FAISS: {acc_faiss}, brute: {acc_brute}"
    )


# ---------------------------------------------------------------------------
# Test 8: Features are L2-normalized before k-NN search
# ---------------------------------------------------------------------------

def test_features_are_l2_normalized():
    """KNNCallback._extract_features returns L2-normalized features."""
    cfg = KNNConfig(k=3, temperature=0.07, every_n_epochs=1)
    cb = KNNCallback(cfg)

    # Simple backbone that returns unnormalized features
    class UnnormalizedBackbone(torch.nn.Module):
        def forward(self, x):
            # Return unnormalized vectors (large magnitude)
            return torch.ones(x.shape[0], 8) * 100.0

    backbone = UnnormalizedBackbone()

    fake_imgs = torch.randn(10, 3, 4, 4)
    fake_labels = torch.zeros(10, dtype=torch.long)
    dataloader = [(fake_imgs, fake_labels)]

    pl_module = MagicMock()
    pl_module.backbone = backbone
    pl_module.device = torch.device("cpu")

    features, labels = cb._extract_features(pl_module, dataloader)

    # Features should be L2-normalized (norm ~= 1.0)
    norms = features.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        f"Expected L2-normalized features (norm=1), got norms: {norms}"
    )
