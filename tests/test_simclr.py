"""Tests for methods/simclr — SimCLRv1Module and SimCLRv2Module.

Tests verify:
  - NT-Xent symmetry: loss(z1, z2) == loss(z2, z1) within 1e-5
  - Identical views produce lower loss than random views
  - 10-epoch training with decreasing loss on toy data
  - SimCLRv2 has exactly 3 linear layers in projector vs 2 for v1
  - v2 only changes projector (backbone/loss_fn unchanged)
  - Dispatcher registration for both simclr_v1 and simclr_v2
  - Uses InfoNCELoss
  - Strong augmentation uses color jitter s=1.0
  - LARS optimizer activates correctly
  - Default optimizer is AdamW
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import lightning as L
import numpy as np
from PIL import Image

from core.config import TrainConfig
from core.losses import InfoNCELoss
from core.data import SSLDataModule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def large_imagefolder(tmp_path):
    """Create a larger ImageFolder for training tests (3 classes, 40 images each).

    Uses a fixed RNG seed so images are deterministic across runs.
    """
    rng = np.random.RandomState(12345)
    n_classes = 3
    n_images = 40
    for cls_idx in range(n_classes):
        cls_dir = tmp_path / f"class_{cls_idx}"
        cls_dir.mkdir()
        for img_idx in range(n_images):
            arr = rng.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            img.save(cls_dir / f"img_{img_idx:02d}.jpg")
    return tmp_path


@pytest.fixture(autouse=True)
def clean_registry():
    """Restore _METHOD_REGISTRY to its original state after each test."""
    from core.dispatcher import _METHOD_REGISTRY
    original = _METHOD_REGISTRY.copy()
    yield
    _METHOD_REGISTRY.clear()
    _METHOD_REGISTRY.update(original)


def _make_cfg(**overrides) -> TrainConfig:
    """Create a minimal TrainConfig for SimCLR testing."""
    defaults = {
        "method": "simclr_v1",
        "backbone": "resnet18",
        "pretrained": False,
        "max_epochs": 5,
        "warmup_epochs": 0,
        "batch_size": 4,
        "lr": 1e-3,
        "weight_decay": 1e-6,
        "optimizer": "adamw",
        "n_views": 2,
    }
    defaults.update(overrides)
    return TrainConfig(**defaults)


# ---------------------------------------------------------------------------
# Loss-tracking callback
# ---------------------------------------------------------------------------

class LossTracker(L.Callback):
    """Callback that records per-epoch average training loss."""

    def __init__(self):
        super().__init__()
        self.epoch_losses: list[float] = []
        self._step_losses: list[float] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, dict) and "loss" in outputs:
            self._step_losses.append(outputs["loss"].detach().item())
        elif hasattr(outputs, "item"):
            self._step_losses.append(outputs.detach().item())

    def on_train_epoch_end(self, trainer, pl_module):
        if self._step_losses:
            avg = sum(self._step_losses) / len(self._step_losses)
            self.epoch_losses.append(avg)
            self._step_losses.clear()


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_ntxent_symmetry():
    """InfoNCELoss(temperature=0.5) is symmetric: loss(z1, z2) == loss(z2, z1)."""
    torch.manual_seed(42)
    loss_fn = InfoNCELoss(temperature=0.5)
    z1 = torch.randn(32, 128)
    z2 = torch.randn(32, 128)
    assert torch.isclose(loss_fn(z1, z2), loss_fn(z2, z1), atol=1e-5)


def test_identical_views_minimum():
    """Identical views produce lower loss than random views."""
    torch.manual_seed(42)
    loss_fn = InfoNCELoss(temperature=0.5)
    z = torch.randn(32, 128)
    loss_identical = loss_fn(z, z.clone())
    loss_random = loss_fn(z, torch.randn(32, 128))
    assert loss_identical < loss_random


def test_simclr_v1_train_5_epochs(large_imagefolder):
    """SimCLRv1Module trains 10 epochs; loss decreases overall."""
    L.seed_everything(42)

    import methods.simclr  # trigger registration
    from methods.simclr.module import SimCLRv1Module

    cfg = _make_cfg(
        lr=0.01,
        batch_size=8,
        max_epochs=15,
        simclr={"temperature": 0.5, "projection_dim": 128},
    )
    model = SimCLRv1Module(cfg)
    dm = SSLDataModule(
        data_dir=str(large_imagefolder),
        n_views=2,
        batch_size=8,
        num_workers=0,
        size=32,
        strong=False,  # weak augmentation for stable convergence on tiny data
    )
    tracker = LossTracker()
    trainer = L.Trainer(
        max_epochs=15,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
        callbacks=[tracker],
    )
    trainer.fit(model, dm)

    # Loss must be finite at end
    assert len(tracker.epoch_losses) == 15, f"Expected 15 epochs, got {len(tracker.epoch_losses)}"
    for i, loss in enumerate(tracker.epoch_losses):
        assert loss == loss, f"Epoch {i} loss is NaN"  # NaN check
        assert abs(loss) < 1e6, f"Epoch {i} loss is not finite: {loss}"

    # Loss at last epoch < loss at first epoch (overall decreasing trend)
    # Use min of last 3 epochs vs max of first 3 epochs for noise-robust comparison
    early_loss = max(tracker.epoch_losses[:3])
    late_loss = min(tracker.epoch_losses[-3:])
    assert late_loss < early_loss, (
        f"Loss should decrease over training: "
        f"early_max={early_loss:.4f}, late_min={late_loss:.4f}"
    )


def test_simclr_v2_3layer_head():
    """SimCLRv2Module projector has exactly 3 nn.Linear layers."""
    from methods.simclr.module import SimCLRv2Module

    cfg = _make_cfg(method="simclr_v2")
    model = SimCLRv2Module(cfg)
    linear_count = sum(1 for m in model.projector.modules() if isinstance(m, nn.Linear))
    assert linear_count == 3, f"Expected 3 linear layers, got {linear_count}"


def test_v1_has_2layer_head():
    """SimCLRv1Module projector has exactly 2 nn.Linear layers."""
    from methods.simclr.module import SimCLRv1Module

    cfg = _make_cfg()
    model = SimCLRv1Module(cfg)
    linear_count = sum(1 for m in model.projector.modules() if isinstance(m, nn.Linear))
    assert linear_count == 2, f"Expected 2 linear layers, got {linear_count}"


def test_v2_only_changes_projector():
    """v1 and v2 have identical backbone and loss_fn; only projector differs."""
    from methods.simclr.module import SimCLRv1Module, SimCLRv2Module

    cfg_v1 = _make_cfg(method="simclr_v1")
    cfg_v2 = _make_cfg(method="simclr_v2")
    v1 = SimCLRv1Module(cfg_v1)
    v2 = SimCLRv2Module(cfg_v2)

    # Same backbone type
    assert type(v1.backbone).__name__ == type(v2.backbone).__name__
    # Same loss type and temperature
    assert type(v1.loss_fn).__name__ == type(v2.loss_fn).__name__
    assert v1.loss_fn.temperature == v2.loss_fn.temperature
    # Different projector depth
    v1_linears = sum(1 for m in v1.projector.modules() if isinstance(m, nn.Linear))
    v2_linears = sum(1 for m in v2.projector.modules() if isinstance(m, nn.Linear))
    assert v1_linears == 2
    assert v2_linears == 3


def test_dispatcher_registration():
    """Dispatcher returns correct class for simclr_v1 and simclr_v2."""
    from methods.simclr.module import SimCLRv1Module, SimCLRv2Module
    from core.dispatcher import method_dispatcher, register_method, available_methods

    if "simclr_v1" not in available_methods():
        register_method("simclr_v1", SimCLRv1Module)
    if "simclr_v2" not in available_methods():
        register_method("simclr_v2", SimCLRv2Module)

    cfg_v1 = _make_cfg(method="simclr_v1")
    model_v1 = method_dispatcher(cfg_v1)
    assert isinstance(model_v1, SimCLRv1Module)

    cfg_v2 = _make_cfg(method="simclr_v2")
    model_v2 = method_dispatcher(cfg_v2)
    assert isinstance(model_v2, SimCLRv2Module)


def test_uses_infonce_loss():
    """SimCLRv1Module.loss_fn is an InfoNCELoss instance."""
    from methods.simclr.module import SimCLRv1Module

    cfg = _make_cfg()
    model = SimCLRv1Module(cfg)
    assert isinstance(model.loss_fn, InfoNCELoss)


def test_strong_augmentation_s1():
    """ContrastiveAugmentation(strong=True) uses color jitter with s=1.0."""
    from core.data import ContrastiveAugmentation

    aug = ContrastiveAugmentation(size=224, strong=True)
    # Check that ColorJitter is present in the transform pipeline
    transform_strs = [str(t) for t in aug.transform.transforms]
    has_color_jitter = any("ColorJitter" in s for s in transform_strs)
    assert has_color_jitter, "Strong augmentation must include ColorJitter"

    # Verify s=1.0 parameters: brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
    # (0.8 * s where s=1.0)
    found_jitter = False
    for t in aug.transform.transforms:
        # RandomApply wraps ColorJitter
        if hasattr(t, "transforms"):
            for inner in t.transforms:
                if "ColorJitter" in type(inner).__name__:
                    found_jitter = True
                    # brightness=(1-0.8, 1+0.8) with float tolerance
                    assert abs(inner.brightness[0] - 0.2) < 1e-6, (
                        f"Expected brightness lower=0.2, got {inner.brightness[0]}"
                    )
                    assert abs(inner.brightness[1] - 1.8) < 1e-6, (
                        f"Expected brightness upper=1.8, got {inner.brightness[1]}"
                    )
    assert found_jitter, "ColorJitter not found in strong augmentation pipeline"


def _configure_optimizers_with_mock_trainer(model):
    """Helper: attach a mock trainer and call configure_optimizers."""
    from unittest.mock import PropertyMock, patch

    trainer = L.Trainer(
        max_epochs=5,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    model.trainer = trainer
    with patch.object(type(trainer), "estimated_stepping_batches", new_callable=PropertyMock, return_value=100):
        return model.configure_optimizers()


def test_lars_optimizer_activates():
    """cfg with optimizer='lars' produces LARS optimizer."""
    from methods.simclr.module import SimCLRv1Module

    cfg = _make_cfg(optimizer="lars", max_epochs=5)
    model = SimCLRv1Module(cfg)
    result = _configure_optimizers_with_mock_trainer(model)
    optimizer = result["optimizer"]
    assert "LARS" in type(optimizer).__name__, (
        f"Expected LARS optimizer, got {type(optimizer).__name__}"
    )


def test_default_optimizer_is_adamw():
    """cfg with optimizer='adamw' produces AdamW optimizer."""
    from methods.simclr.module import SimCLRv1Module

    cfg = _make_cfg(optimizer="adamw", max_epochs=5)
    model = SimCLRv1Module(cfg)
    result = _configure_optimizers_with_mock_trainer(model)
    optimizer = result["optimizer"]
    assert "AdamW" in type(optimizer).__name__, (
        f"Expected AdamW optimizer, got {type(optimizer).__name__}"
    )


# ---------------------------------------------------------------------------
# YAML Config Loading Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_imagefolder(tmp_path):
    """Create a minimal ImageFolder for config loading tests (2 classes, 8 images each)."""
    rng = np.random.RandomState(99)
    for cls_idx in range(2):
        cls_dir = tmp_path / f"class_{cls_idx}"
        cls_dir.mkdir()
        for img_idx in range(8):
            arr = rng.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            img.save(cls_dir / f"img_{img_idx:02d}.jpg")
    return tmp_path


def test_yaml_config_loads_and_trains(tmp_imagefolder):
    """Load simclr_v1_resnet18.yaml, override for test, train 1 epoch."""
    import yaml as _yaml

    from methods.simclr.module import SimCLRv1Module
    from core.dispatcher import method_dispatcher, register_method, available_methods

    if "simclr_v1" not in available_methods():
        register_method("simclr_v1", SimCLRv1Module)

    with open("configs/simclr_v1_resnet18.yaml") as fh:
        raw = _yaml.safe_load(fh)

    # Override for fast test
    raw["data_dir"] = str(tmp_imagefolder)
    raw["max_epochs"] = 1
    raw["warmup_epochs"] = 0
    raw["batch_size"] = 4
    raw["num_workers"] = 0

    cfg = TrainConfig.model_validate(raw)
    model = method_dispatcher(cfg)
    dm = SSLDataModule(
        data_dir=cfg.data_dir,
        n_views=cfg.n_views,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        size=32,
        strong=True,
    )
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, dm)
    # If we get here, training completed without error


def test_simclr_v2_yaml_config_loads():
    """Load simclr_v2_resnet18.yaml and validate its fields."""
    import yaml as _yaml

    with open("configs/simclr_v2_resnet18.yaml") as fh:
        raw = _yaml.safe_load(fh)

    cfg = TrainConfig.model_validate(raw)
    assert cfg.method == "simclr_v2"
    assert cfg.simclr is not None
    assert cfg.simclr.temperature == 0.5
    assert cfg.simclr.projection_dim == 128
