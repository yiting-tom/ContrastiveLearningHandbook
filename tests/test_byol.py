"""Tests for methods/byol -- BYOLModule stop-gradient validation.

Tests verify:
  - Target branch parameters (backbone_ema, projector_ema) receive zero
    gradient after training_step backward pass (stop-gradient correctness)
  - Online branch parameters (backbone, projector, predictor) receive
    non-zero gradient after training_step backward pass
  - BYOLModule has backbone_ema, projector_ema, and predictor attributes
  - training_step returns a finite scalar loss
  - Target network parameters have requires_grad=False at initialisation

The stop-gradient mechanism is the load-bearing correctness property of BYOL:
without it, the optimisation trivially collapses to a constant representation.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from core.config import BYOLConfig, TrainConfig
from methods.byol.module import BYOLModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_toy_cfg() -> TrainConfig:
    """Minimal TrainConfig for BYOL unit tests."""
    return TrainConfig(
        method="byol",
        backbone="resnet18",
        pretrained=False,
        max_epochs=1,
        batch_size=4,
        lr=0.001,
        byol=BYOLConfig(),
    )


def make_toy_batch(B: int = 4, C: int = 3, H: int = 32, W: int = 32):
    """Synthetic two-view batch: ((v1, v2), labels)."""
    v1 = torch.randn(B, C, H, W)
    v2 = torch.randn(B, C, H, W)
    labels = torch.zeros(B, dtype=torch.long)
    return (v1, v2), labels


def run_training_step(module: BYOLModule, batch):
    """Call training_step with logging mocked out (no trainer required).

    BYOLModule.training_step() calls self.log() and self.log_train_metrics()
    which normally require a Lightning Trainer context.  We patch both at the
    module-instance level so the gradient-flow logic is tested in isolation.
    """
    module.train()
    # Patch instance-level log and log_train_metrics to avoid trainer dependency.
    module.log = MagicMock()
    module.log_train_metrics = MagicMock()
    return module.training_step(batch, 0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_byol_target_zero_grad():
    """Target branch parameters must receive zero gradient during training_step.

    This is the core correctness test for the stop-gradient mechanism.
    The target network (backbone_ema, projector_ema) is computed inside a
    torch.no_grad() context and its outputs are additionally .detach()ed
    before the loss.  Both guards together guarantee no gradient reaches
    the target branch.
    """
    cfg = make_toy_cfg()
    module = BYOLModule(cfg)
    batch = make_toy_batch()

    loss = run_training_step(module, batch)
    loss.backward()

    # Target params must have no gradient (None or exactly zero).
    for name, p in module.backbone_ema.named_parameters():
        grad_max = p.grad.abs().max().item() if p.grad is not None else 0.0
        assert grad_max == 0.0, (
            f"backbone_ema.{name} has non-zero gradient ({grad_max:.6f}) "
            f"— stop-gradient is broken"
        )
    for name, p in module.projector_ema.named_parameters():
        grad_max = p.grad.abs().max().item() if p.grad is not None else 0.0
        assert grad_max == 0.0, (
            f"projector_ema.{name} has non-zero gradient ({grad_max:.6f}) "
            f"— stop-gradient is broken"
        )


def test_byol_online_params_have_grad():
    """Online params must receive gradient so learning can happen.

    At least one parameter in backbone, projector, and predictor must have a
    non-zero gradient after backward — confirming the online branch is
    connected to the computation graph.
    """
    cfg = make_toy_cfg()
    module = BYOLModule(cfg)
    batch = make_toy_batch()

    loss = run_training_step(module, batch)
    loss.backward()

    online_params_with_grad = [
        p
        for p in (
            list(module.backbone.parameters())
            + list(module.projector.parameters())
            + list(module.predictor.parameters())
        )
        if p.grad is not None and p.grad.abs().max() > 0
    ]
    assert len(online_params_with_grad) > 0, (
        "No online branch parameters have a gradient after backward — "
        "the online branch is not connected to the loss"
    )


def test_byol_instantiation():
    """BYOLModule must expose backbone_ema, projector_ema, and predictor."""
    cfg = make_toy_cfg()
    module = BYOLModule(cfg)

    assert hasattr(module, "backbone_ema"), "BYOLModule must have backbone_ema attribute"
    assert hasattr(module, "projector_ema"), "BYOLModule must have projector_ema attribute"
    assert hasattr(module, "predictor"), "BYOLModule must have predictor attribute"


def test_byol_training_step_returns_finite_loss():
    """training_step must return a finite scalar tensor."""
    cfg = make_toy_cfg()
    module = BYOLModule(cfg)
    batch = make_toy_batch()

    loss = run_training_step(module, batch)

    assert isinstance(loss, torch.Tensor), (
        f"training_step must return a Tensor, got {type(loss).__name__}"
    )
    assert loss.ndim == 0, (
        f"training_step must return a scalar (0-d) Tensor, got shape {loss.shape}"
    )
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


def test_byol_target_frozen_at_init():
    """Target network params must have requires_grad=False immediately after __init__."""
    cfg = make_toy_cfg()
    module = BYOLModule(cfg)

    for name, p in module.backbone_ema.named_parameters():
        assert not p.requires_grad, (
            f"backbone_ema.{name} has requires_grad=True — "
            f"target params must be frozen to exclude them from the optimiser"
        )
    for name, p in module.projector_ema.named_parameters():
        assert not p.requires_grad, (
            f"projector_ema.{name} has requires_grad=True — "
            f"target params must be frozen to exclude them from the optimiser"
        )
