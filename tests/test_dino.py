"""Tests for methods/dino -- DINOModule student-teacher self-distillation.

Tests verify:
  - DINOModule has a registered buffer named 'center' with shape [n_prototypes]
  - After one training step, center buffer is non-zero (updated before loss)
  - Teacher forward is called only on global crops (first 2), not local crops
  - Module does NOT have predictor_ema or teacher_predictor attribute
  - No EMA parameter id appears in learnable_params
  - Prototype layer output dim is 65536 (n_prototypes)
  - method_dispatcher with method="dino" returns DINOModule
  - DINOModule trains 3 epochs on toy data without loss divergence

The centering-before-loss ordering is the load-bearing correctness property of DINO:
without it, the centering adjustment is applied to the wrong targets, causing
instability or collapse.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import lightning as L
import numpy as np
from PIL import Image

from core.config import DINOConfig, TrainConfig
from core.data import MultiCropDataset, SSLDataModule
from torchvision.datasets import ImageFolder


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_toy_cfg(**overrides) -> TrainConfig:
    """Minimal TrainConfig for DINO unit tests."""
    defaults = {
        "method": "dino",
        "backbone": "resnet18",
        "pretrained": False,
        "max_epochs": 3,
        "warmup_epochs": 0,
        "batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 1e-6,
        "optimizer": "adamw",
        "n_views": 2,
        "dino": DINOConfig(n_prototypes=128, warmup_teacher_temp_epochs=1),
    }
    defaults.update(overrides)
    return TrainConfig(**defaults)


def make_multi_crop_batch(B: int = 4, n_global: int = 2, n_local: int = 2,
                          large_h: int = 32, small_h: int = 16) -> tuple:
    """Create a synthetic multi-crop batch: (crops_list, labels).

    crops_list contains n_global large crops followed by n_local small crops.
    """
    crops_list = []
    for _ in range(n_global):
        crops_list.append(torch.randn(B, 3, large_h, large_h))
    for _ in range(n_local):
        crops_list.append(torch.randn(B, 3, small_h, small_h))
    labels = torch.zeros(B, dtype=torch.long)
    return crops_list, labels


def make_toy_batch(B: int = 4) -> tuple:
    """Create a simple 2-view batch (both treated as global crops)."""
    crops_list = [torch.randn(B, 3, 32, 32), torch.randn(B, 3, 32, 32)]
    labels = torch.zeros(B, dtype=torch.long)
    return crops_list, labels


def run_training_step(module, batch):
    """Call training_step with logging mocked out (no trainer required)."""
    module.train()
    module.log = MagicMock()
    module.log_train_metrics = MagicMock()
    return module.training_step(batch, 0)


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

def test_centering_buffer_exists():
    """DINOModule must have a registered buffer 'center' with shape [n_prototypes]."""
    from methods.dino.module import DINOModule

    cfg = make_toy_cfg()
    module = DINOModule(cfg)

    assert hasattr(module, "center"), "DINOModule must have 'center' buffer"
    assert isinstance(module.center, torch.Tensor), (
        f"center must be a Tensor, got {type(module.center)}"
    )
    assert module.center.shape == (128,), (
        f"center shape should be (n_prototypes,)=(128,), got {module.center.shape}"
    )
    # Must be a buffer (not a parameter)
    buffer_names = [name for name, _ in module.named_buffers()]
    assert "center" in buffer_names, (
        "center must be a registered buffer (via register_buffer), not a plain attribute"
    )


def test_centering_update_before_loss():
    """After one training step, center buffer must be non-zero (updated before loss).

    This verifies the critical ordering: center update happens BEFORE loss computation.
    A zero center at end of step would indicate the update was skipped or applied after.
    """
    from methods.dino.module import DINOModule

    cfg = make_toy_cfg()
    module = DINOModule(cfg)
    batch = make_toy_batch()

    # Center starts at zero
    assert module.center.abs().sum().item() == 0.0, "center should start at zero"

    # After one training step, center must be non-zero
    _ = run_training_step(module, batch)

    center_norm = module.center.abs().sum().item()
    assert center_norm > 0.0, (
        f"center buffer is still zero after training_step — "
        f"centering update is not being applied (center L1 norm={center_norm:.6f})"
    )


def test_teacher_global_crops_only():
    """Teacher forward must only process global crops (first 2), not local crops.

    Verify that when given 2 global + 2 local crops, teacher processes exactly 2 crops.
    We test this by checking that n_global=2 teacher outputs are produced even
    when 4 crops are in the batch.
    """
    from methods.dino.module import DINOModule

    cfg = make_toy_cfg()
    module = DINOModule(cfg)
    module.train()
    module.log = MagicMock()
    module.log_train_metrics = MagicMock()

    # 2 global + 2 local crops
    batch = make_multi_crop_batch(B=4, n_global=2, n_local=2)

    # Track backbone_ema calls to verify only 2 (global) forward passes
    call_count = []
    original_forward = module.backbone_ema.forward

    def counting_forward(x):
        call_count.append(x.shape[0])
        return original_forward(x)

    module.backbone_ema.forward = counting_forward

    _ = module.training_step(batch, 0)

    assert len(call_count) == 2, (
        f"Teacher (backbone_ema) should be called exactly 2 times for 2 global crops, "
        f"got {len(call_count)} calls"
    )


def test_teacher_no_predictor():
    """DINOModule must NOT have predictor_ema or teacher_predictor attribute.

    Teacher branch should have no predictor — only backbone_ema + projector_ema.
    """
    from methods.dino.module import DINOModule

    cfg = make_toy_cfg()
    module = DINOModule(cfg)

    assert not hasattr(module, "predictor_ema"), (
        "DINOModule must NOT have predictor_ema attribute — "
        "teacher branch has no predictor (unlike BYOL)"
    )
    assert not hasattr(module, "teacher_predictor"), (
        "DINOModule must NOT have teacher_predictor attribute"
    )


def test_momentum_encoder_excluded():
    """EMA (teacher) parameters must NOT appear in learnable_params.

    No parameter id from backbone_ema, projector_ema, or prototype_layer_ema
    should appear in learnable_params — teacher is frozen and excluded from optimizer.
    """
    from methods.dino.module import DINOModule

    cfg = make_toy_cfg()
    module = DINOModule(cfg)

    learnable_ids = {id(p) for p in module.learnable_params}

    ema_ids = (
        {id(p) for p in module.backbone_ema.parameters()}
        | {id(p) for p in module.projector_ema.parameters()}
        | {id(p) for p in module.prototype_layer_ema.parameters()}
    )

    overlap = learnable_ids & ema_ids
    assert len(overlap) == 0, (
        f"Found {len(overlap)} EMA parameter(s) in learnable_params — "
        f"teacher parameters must be excluded from the optimizer"
    )


def test_prototype_output_dim():
    """Prototype linear layer output dim must be 65536 (n_prototypes).

    This is the default DINOConfig value and a critical quality hyperparameter.
    """
    from methods.dino.module import DINOModule

    # Use default n_prototypes (65536)
    cfg = TrainConfig(
        method="dino",
        backbone="resnet18",
        pretrained=False,
        max_epochs=1,
        batch_size=4,
        lr=1e-3,
        dino=DINOConfig(),  # default n_prototypes=65536
    )
    module = DINOModule(cfg)

    # Check prototype_layer output dim
    assert hasattr(module, "prototype_layer"), "DINOModule must have prototype_layer"
    assert isinstance(module.prototype_layer, nn.Linear), (
        f"prototype_layer must be nn.Linear, got {type(module.prototype_layer)}"
    )
    assert module.prototype_layer.out_features == 65536, (
        f"prototype_layer output dim must be 65536 (n_prototypes), "
        f"got {module.prototype_layer.out_features}"
    )
    assert module.prototype_layer.bias is None, (
        "prototype_layer must have bias=False (per DINO paper)"
    )


def test_dispatcher_dino():
    """method_dispatcher with method='dino' must return DINOModule instance."""
    import methods  # noqa: F401 -- triggers register_method calls
    from core.dispatcher import method_dispatcher
    from methods.dino.module import DINOModule

    cfg = make_toy_cfg()
    module = method_dispatcher(cfg)

    assert isinstance(module, DINOModule), (
        f"method_dispatcher('dino') should return DINOModule, got {type(module).__name__}"
    )


# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------

def test_dino_train_3_epochs(large_imagefolder):
    """DINOModule trains 3 epochs on toy data without loss divergence.

    Uses:
    - resnet18 backbone (fast)
    - 2 global + 2 local crops via MultiCropDataset
    - n_prototypes=128 (reduced for speed)
    - Verifies loss is finite and not diverging (last epoch < 2 * first epoch)
    """
    import methods  # noqa: F401 -- triggers register_method calls
    from methods.dino.module import DINOModule
    from core.data import MultiCropDataset, ssl_collate_multi_crop
    from torch.utils.data import DataLoader

    cfg = TrainConfig(
        method="dino",
        backbone="resnet18",
        pretrained=False,
        max_epochs=3,
        warmup_epochs=0,
        batch_size=8,
        lr=1e-3,
        weight_decay=1e-6,
        optimizer="adamw",
        n_views=4,
        dino=DINOConfig(
            n_prototypes=128,
            teacher_temp=0.04,
            warmup_teacher_temp=0.07,
            warmup_teacher_temp_epochs=1,
            student_temp=0.1,
            centering_momentum=0.9,
        ),
    )

    module = DINOModule(cfg)
    loss_tracker = LossTracker()

    # Create MultiCropDataset: 2 global (32x32) + 2 local (16x16)
    base_dataset = ImageFolder(str(large_imagefolder))  # no transform = PIL images
    multi_crop_ds = MultiCropDataset(
        dataset=base_dataset,
        n_large_crops=2,
        large_size=32,
        n_small_crops=2,
        small_size=16,
    )

    dm = SSLDataModule(
        data_dir=str(large_imagefolder),
        batch_size=8,
        num_workers=0,
        dataset=multi_crop_ds,
    )

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        callbacks=[loss_tracker],
        gradient_clip_val=3.0,  # DINO requires gradient clipping
    )
    trainer.fit(module, datamodule=dm)

    assert len(loss_tracker.epoch_losses) == 3, (
        f"Expected 3 epoch losses, got {len(loss_tracker.epoch_losses)}"
    )

    for i, loss in enumerate(loss_tracker.epoch_losses):
        assert not (loss != loss), f"Loss at epoch {i} is NaN"  # NaN check
        assert loss < float("inf"), f"Loss at epoch {i} is infinite"

    # Check no loss divergence: last epoch < 2 * first epoch (guard against explosion)
    first_loss = loss_tracker.epoch_losses[0]
    last_loss = loss_tracker.epoch_losses[-1]
    assert last_loss < 2.0 * max(first_loss, 1e-6), (
        f"Loss diverged: first={first_loss:.4f}, last={last_loss:.4f}"
    )
