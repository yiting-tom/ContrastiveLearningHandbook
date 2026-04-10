"""Tests for SupConFinetuneModule — stage-2 frozen backbone + linear head."""
import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from core.config import TrainConfig
from methods.supcon.module import SupConFinetuneModule, SupConModule


def run_training_step(module: SupConFinetuneModule, batch):
    """Call training_step with logging mocked out (no trainer required).

    SupConFinetuneModule.training_step() calls self.log_train_metrics()
    which normally requires a Lightning Trainer context. We patch both at the
    module-instance level so the loss computation is tested in isolation.
    """
    module.train()
    module.log = MagicMock()
    module.log_train_metrics = MagicMock()
    return module.training_step(batch, 0)


def run_validation_step(module: SupConFinetuneModule, batch):
    """Call validation_step with logging mocked out (no trainer required)."""
    module.eval()
    module.log = MagicMock()
    return module.validation_step(batch, 0)


def make_cfg(num_classes: int = 4) -> TrainConfig:
    return TrainConfig.model_validate({
        "method": "supcon_finetune",
        "backbone": "resnet18",
        "pretrained": False,
        "max_epochs": 1,
        "warmup_epochs": 0,
        "batch_size": 8,
        "lr": 0.1,
        "weight_decay": 0.0,
        "optimizer": "sgd",
        "n_views": 1,
        "data_dir": "data",
        "num_workers": 0,
        "supcon": {
            "temperature": 0.07,
            "n_samples_per_class": 2,
            "n_classes_per_batch": 4,
            "num_classes": num_classes,
            "projection_dim": 128,
        },
    })


# ---------------------------------------------------------------------------
# Test 1: Only classifier parameters are returned by learnable_params  (SC-4)
# ---------------------------------------------------------------------------
def test_only_classifier_params_trained():
    """learnable_params must return ONLY classifier parameters, not backbone."""
    cfg = make_cfg(num_classes=4)
    module = SupConFinetuneModule(cfg)
    module.freeze_backbone()

    learnable_param_ids = {
        id(p)
        for group in module.learnable_params
        for p in group["params"]
    }
    classifier_param_ids = {id(p) for p in module.classifier.parameters()}
    backbone_param_ids = {id(p) for p in module.backbone.parameters()}

    assert learnable_param_ids == classifier_param_ids, (
        "learnable_params must equal classifier.parameters() exactly"
    )
    assert learnable_param_ids.isdisjoint(backbone_param_ids), (
        "Backbone parameters must NOT appear in learnable_params"
    )


# ---------------------------------------------------------------------------
# Test 2: SGD optimizer has weight_decay=0.0
# ---------------------------------------------------------------------------
def test_sgd_weight_decay_zero():
    """configure_optimizers must return SGD with weight_decay=0.0."""
    cfg = make_cfg()
    module = SupConFinetuneModule(cfg)
    module.freeze_backbone()
    optimizer = module.configure_optimizers()
    assert isinstance(optimizer, torch.optim.SGD), (
        f"Expected SGD, got {type(optimizer).__name__}"
    )
    for group in optimizer.param_groups:
        assert group["weight_decay"] == 0.0, (
            f"Expected weight_decay=0.0, got {group['weight_decay']}"
        )


# ---------------------------------------------------------------------------
# Test 3: Backbone parameters do NOT receive gradients after freeze
# ---------------------------------------------------------------------------
def test_backbone_frozen_no_gradients():
    """After freeze_backbone(), no backbone parameter should require grad."""
    cfg = make_cfg()
    module = SupConFinetuneModule(cfg)
    module.freeze_backbone()

    for name, param in module.backbone.named_parameters():
        assert not param.requires_grad, (
            f"Backbone param '{name}' still requires grad after freeze_backbone()"
        )


# ---------------------------------------------------------------------------
# Test 4: training_step produces finite scalar loss
# ---------------------------------------------------------------------------
def test_training_step_finite_loss():
    cfg = make_cfg(num_classes=4)
    module = SupConFinetuneModule(cfg)
    module.freeze_backbone()

    B = 8
    # views shape [2, B, C, H, W] — module uses views[0]
    views = torch.randn(2, B, 3, 32, 32)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    loss = run_training_step(module, (views, labels))
    assert loss.shape == (), "Loss must be scalar"
    assert torch.isfinite(loss), "Loss must be finite"


# ---------------------------------------------------------------------------
# Test 5: from_stage1_ckpt loads backbone and discards projector
# ---------------------------------------------------------------------------
def test_from_stage1_ckpt_backbone_only():
    """from_stage1_ckpt should load backbone.* keys and ignore projector.*."""
    cfg_stage1 = TrainConfig.model_validate({
        "method": "supcon",
        "backbone": "resnet18",
        "pretrained": False,
        "max_epochs": 1,
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

    # Build a fake stage-1 checkpoint with backbone.* and projector.* keys
    stage1_module = SupConModule(cfg_stage1)
    fake_ckpt = {"state_dict": stage1_module.state_dict()}

    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        torch.save(fake_ckpt, f.name)
        ckpt_path = f.name

    cfg_stage2 = make_cfg(num_classes=4)
    module = SupConFinetuneModule.from_stage1_ckpt(ckpt_path, cfg_stage2)

    # Backbone should be frozen
    for param in module.backbone.parameters():
        assert not param.requires_grad, "Backbone must be frozen after from_stage1_ckpt"

    # Classifier should still require grad
    for param in module.classifier.parameters():
        assert param.requires_grad, "Classifier must still require grad"

    # Verify backbone weights match the stage-1 module
    for name, param in module.backbone.named_parameters():
        expected = stage1_module.backbone.state_dict()[name]
        assert torch.allclose(param.data, expected), (
            f"Backbone param '{name}' does not match stage-1 checkpoint"
        )

    Path(ckpt_path).unlink()  # cleanup


# ---------------------------------------------------------------------------
# Test 6: ValueError when checkpoint has no backbone.* keys
# ---------------------------------------------------------------------------
def test_from_stage1_ckpt_no_backbone_raises():
    fake_ckpt = {"state_dict": {"random_key": torch.tensor(1.0)}}
    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        torch.save(fake_ckpt, f.name)
        ckpt_path = f.name

    cfg = make_cfg()
    with pytest.raises(ValueError, match="backbone"):
        SupConFinetuneModule.from_stage1_ckpt(ckpt_path, cfg)

    Path(ckpt_path).unlink()
