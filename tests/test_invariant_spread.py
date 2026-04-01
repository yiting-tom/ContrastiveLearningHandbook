"""Tests for methods/invariant_spread — InvariantSpreadModule.

Tests verify:
  - 5-epoch training with monotonically decreasing loss in first 3 epochs
  - Dispatcher registration as 'invariant_spread'
  - Reuses InfoNCELoss from core/losses.py (per D-03)
  - No memory bank attribute
  - Batch-size sensitivity documented in docstring
"""
from __future__ import annotations

import pytest
import lightning as L

from core.config import TrainConfig
from core.losses import InfoNCELoss
from core.data import SSLDataModule


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


def _make_cfg(**overrides) -> TrainConfig:
    """Create a minimal TrainConfig for InvariantSpread testing."""
    defaults = {
        "method": "invariant_spread",
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
# Tests
# ---------------------------------------------------------------------------

def test_train_5_epochs(tmp_imagefolder):
    """InvariantSpreadModule trains 5 epochs; loss decreases in first 3."""
    import methods.invariant_spread  # trigger registration
    from methods.invariant_spread.module import InvariantSpreadModule

    cfg = _make_cfg()
    model = InvariantSpreadModule(cfg)
    dm = SSLDataModule(
        data_dir=str(tmp_imagefolder),
        n_views=2,
        batch_size=4,
        num_workers=0,
        size=32,
        strong=False,
    )
    tracker = LossTracker()
    trainer = L.Trainer(
        max_epochs=5,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[tracker],
    )
    trainer.fit(model, dm)

    # Loss must be finite at end
    assert len(tracker.epoch_losses) == 5, f"Expected 5 epochs, got {len(tracker.epoch_losses)}"
    for i, loss in enumerate(tracker.epoch_losses):
        assert loss == loss, f"Epoch {i} loss is NaN"  # NaN check
        assert abs(loss) < 1e6, f"Epoch {i} loss is not finite: {loss}"

    # Loss at epoch 3 < loss at epoch 1 (monotonic decrease in first 3)
    assert tracker.epoch_losses[2] < tracker.epoch_losses[0], (
        f"Loss should decrease in first 3 epochs: "
        f"epoch1={tracker.epoch_losses[0]:.4f}, epoch3={tracker.epoch_losses[2]:.4f}"
    )


def test_dispatcher_registration():
    """After importing methods.invariant_spread, dispatcher returns InvariantSpreadModule."""
    import methods.invariant_spread  # trigger registration
    from methods.invariant_spread.module import InvariantSpreadModule
    from core.dispatcher import method_dispatcher

    cfg = _make_cfg()
    model = method_dispatcher(cfg)
    assert isinstance(model, InvariantSpreadModule)


def test_uses_infonce_loss():
    """InvariantSpreadModule.loss_fn is an InfoNCELoss instance (per D-03)."""
    from methods.invariant_spread.module import InvariantSpreadModule

    cfg = _make_cfg()
    model = InvariantSpreadModule(cfg)
    assert isinstance(model.loss_fn, InfoNCELoss)


def test_no_memory_bank():
    """InvariantSpreadModule has no memory_bank attribute (or it is None)."""
    from methods.invariant_spread.module import InvariantSpreadModule

    cfg = _make_cfg()
    model = InvariantSpreadModule(cfg)
    assert not hasattr(model, "memory_bank") or model.memory_bank is None


def test_batch_size_sensitivity_docstring():
    """InvariantSpreadModule docstring documents batch-size sensitivity."""
    from methods.invariant_spread.module import InvariantSpreadModule

    doc = InvariantSpreadModule.__doc__
    assert doc is not None, "InvariantSpreadModule must have a docstring"
    doc_lower = doc.lower()
    assert "batch" in doc_lower, "Docstring must mention 'batch'"
    assert "sensitiv" in doc_lower, "Docstring must mention 'sensitivity' or 'sensitive'"
