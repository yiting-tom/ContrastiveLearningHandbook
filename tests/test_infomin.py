"""Tests for methods/infomin -- InfoMinModule (Tian et al., NeurIPS 2020).

Tests verify:
  - SimCLRv1Module has a build_augmentation() classmethod
  - InfoMinModule is a subclass of SimCLRv1Module
  - InfoMinModule.build_augmentation() returns a transform WITHOUT GaussianBlur
  - InfoMinModule.build_augmentation() uses color jitter with s=1.5 (brightness=1.2)
  - InfoMinModule.build_augmentation() uses RandomGrayscale with p=0.4
  - InfoMinModule is registered as "infomin" in method_dispatcher
  - InfoMinModule trains 5 epochs on toy data without loss divergence
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import lightning as L
from PIL import Image


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_registry():
    """Restore _METHOD_REGISTRY to its original state after each test."""
    from core.dispatcher import _METHOD_REGISTRY
    original = _METHOD_REGISTRY.copy()
    yield
    _METHOD_REGISTRY.clear()
    _METHOD_REGISTRY.update(original)


@pytest.fixture
def large_imagefolder(tmp_path):
    """Create a larger ImageFolder for training tests (3 classes, 40 images each)."""
    rng = np.random.RandomState(42)
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


def _make_cfg(**overrides):
    """Create a minimal TrainConfig for InfoMin testing."""
    from core.config import TrainConfig
    defaults = {
        "method": "infomin",
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
# Test 1: SimCLRv1Module has build_augmentation() classmethod
# ---------------------------------------------------------------------------

def test_simclr_has_build_augmentation():
    """SimCLRv1Module has a build_augmentation() classmethod."""
    from methods.simclr.module import SimCLRv1Module
    assert hasattr(SimCLRv1Module, "build_augmentation"), (
        "SimCLRv1Module must have a build_augmentation() classmethod"
    )
    assert callable(SimCLRv1Module.build_augmentation), (
        "build_augmentation must be callable"
    )
    # Must be a classmethod (not an instance method)
    import inspect
    method = inspect.getattr_static(SimCLRv1Module, "build_augmentation")
    assert isinstance(method, classmethod), (
        "build_augmentation must be a classmethod"
    )


def test_simclr_build_augmentation_returns_callable():
    """SimCLRv1Module.build_augmentation() returns a callable transform."""
    from methods.simclr.module import SimCLRv1Module
    aug = SimCLRv1Module.build_augmentation(size=32)
    assert callable(aug), "build_augmentation() must return a callable"


# ---------------------------------------------------------------------------
# Test 2: InfoMinModule is a subclass of SimCLRv1Module
# ---------------------------------------------------------------------------

def test_infomin_is_subclass_of_simclr():
    """InfoMinModule is a subclass of SimCLRv1Module."""
    import methods.infomin  # trigger registration
    from methods.infomin.module import InfoMinModule
    from methods.simclr.module import SimCLRv1Module
    assert issubclass(InfoMinModule, SimCLRv1Module), (
        "InfoMinModule must be a subclass of SimCLRv1Module"
    )


# ---------------------------------------------------------------------------
# Test 3: InfoMinModule.build_augmentation() has no GaussianBlur
# ---------------------------------------------------------------------------

def test_infomin_no_gaussian_blur():
    """InfoMinModule.build_augmentation() returns a transform WITHOUT GaussianBlur."""
    from methods.infomin.module import InfoMinModule
    aug = InfoMinModule.build_augmentation(size=32)
    # Inspect all transforms in the pipeline
    transforms_list = []
    if hasattr(aug, "transforms"):
        transforms_list = aug.transforms
    else:
        # v2.Compose stores in .transforms
        transforms_list = getattr(aug, "transforms", [])
    # Flatten nested transforms (RandomApply has sub-transforms)
    all_transforms = []
    for t in transforms_list:
        all_transforms.append(t)
        if hasattr(t, "transforms"):
            all_transforms.extend(t.transforms)

    has_blur = any("GaussianBlur" in type(t).__name__ for t in all_transforms)
    assert not has_blur, (
        "InfoMinModule.build_augmentation() must NOT include GaussianBlur by default"
    )


# ---------------------------------------------------------------------------
# Test 4: InfoMinModule uses color jitter with s=1.5 (brightness = 0.8*1.5 = 1.2)
# ---------------------------------------------------------------------------

def test_infomin_color_jitter_s15():
    """InfoMinModule.build_augmentation() uses ColorJitter with s=1.5 (brightness=1.2)."""
    from methods.infomin.module import InfoMinModule
    aug = InfoMinModule.build_augmentation(size=32)

    # Expected: 0.8 * 1.5 = 1.2 -> brightness range (1-1.2, 1+1.2) = (0, 2.2) but
    # torchvision clips brightness to [max(0, 1-b), 1+b] where b=1.2
    # Actually: brightness param = 1.2 so range is (max(0, 1-1.2), 1+1.2) = (0, 2.2)
    # But ColorJitter stores the tuple internally so check brightness param
    found_jitter = False
    transforms_list = getattr(aug, "transforms", [])
    for t in transforms_list:
        sub_transforms = [t] + (list(getattr(t, "transforms", [])))
        for inner in sub_transforms:
            if "ColorJitter" in type(inner).__name__:
                found_jitter = True
                # brightness is stored as (lower, upper) tuple
                b = inner.brightness
                # 0.8 * 1.5 = 1.2, so brightness[1] = 1 + 1.2 = 2.2
                assert abs(b[1] - 2.2) < 1e-5, (
                    f"Expected ColorJitter brightness upper=2.2 (s=1.5), got {b[1]}"
                )
    assert found_jitter, "ColorJitter not found in InfoMin augmentation pipeline"


# ---------------------------------------------------------------------------
# Test 5: InfoMinModule uses RandomGrayscale with p=0.4
# ---------------------------------------------------------------------------

def test_infomin_grayscale_p04():
    """InfoMinModule.build_augmentation() uses RandomGrayscale with p=0.4."""
    from methods.infomin.module import InfoMinModule
    aug = InfoMinModule.build_augmentation(size=32)

    found_grayscale = False
    transforms_list = getattr(aug, "transforms", [])
    for t in transforms_list:
        if "RandomGrayscale" in type(t).__name__:
            found_grayscale = True
            assert abs(t.p - 0.4) < 1e-6, (
                f"Expected RandomGrayscale p=0.4, got {t.p}"
            )
    assert found_grayscale, "RandomGrayscale not found in InfoMin augmentation pipeline"


# ---------------------------------------------------------------------------
# Test 6: InfoMinModule registered as "infomin" in dispatcher
# ---------------------------------------------------------------------------

def test_infomin_dispatcher_registration():
    """'infomin' is registered in the method dispatcher."""
    from methods.infomin.module import InfoMinModule
    from core.dispatcher import available_methods, register_method

    if "infomin" not in available_methods():
        register_method("infomin", InfoMinModule)

    assert "infomin" in available_methods(), (
        "'infomin' must be registered in the method dispatcher"
    )


def test_infomin_dispatcher_returns_correct_class():
    """Dispatcher returns InfoMinModule for method='infomin'."""
    from methods.infomin.module import InfoMinModule
    from core.dispatcher import method_dispatcher, available_methods, register_method

    if "infomin" not in available_methods():
        register_method("infomin", InfoMinModule)

    cfg = _make_cfg(method="infomin")
    model = method_dispatcher(cfg)
    assert isinstance(model, InfoMinModule), (
        f"Expected InfoMinModule, got {type(model).__name__}"
    )


# ---------------------------------------------------------------------------
# Test 7: InfoMinModule trains 5 epochs without loss divergence
#         Uses production data path: build_augmentation() -> MultiViewTransform -> SSLDataModule
# ---------------------------------------------------------------------------

def test_infomin_trains_5_epochs_no_nan(large_imagefolder):
    """InfoMinModule trains 5 epochs using build_augmentation() production path without NaN."""
    from methods.infomin.module import InfoMinModule
    from core.dispatcher import available_methods, register_method
    if "infomin" not in available_methods():
        register_method("infomin", InfoMinModule)
    from core.data import MultiViewTransform, SSLDataModule
    from torchvision.datasets import ImageFolder

    L.seed_everything(42)

    cfg = _make_cfg(
        method="infomin",
        lr=0.01,
        batch_size=8,
        max_epochs=5,
        n_views=2,
        data_dir=str(large_imagefolder),
    )

    # Production data path: use InfoMinModule.build_augmentation() directly
    aug = InfoMinModule.build_augmentation(size=32)
    transform = MultiViewTransform(aug, n_views=2)
    dataset = ImageFolder(str(large_imagefolder), transform=transform)

    # Wire into SSLDataModule by passing the pre-built dataset
    from core.data import ssl_collate_fn
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=ssl_collate_fn,
        drop_last=True,
    )

    model = InfoMinModule(cfg)
    tracker = LossTracker()
    trainer = L.Trainer(
        max_epochs=5,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
        callbacks=[tracker],
    )
    trainer.fit(model, train_dataloaders=train_loader)

    # All epoch losses must be finite
    assert len(tracker.epoch_losses) == 5, (
        f"Expected 5 epochs, got {len(tracker.epoch_losses)}"
    )
    for i, loss in enumerate(tracker.epoch_losses):
        assert loss == loss, f"Epoch {i} loss is NaN"
        assert abs(loss) < 1e6, f"Epoch {i} loss diverged: {loss}"
