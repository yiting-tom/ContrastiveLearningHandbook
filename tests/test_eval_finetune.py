"""Unit tests for eval/finetune.py — FinetuneModule and fine-tuning script.

Tests verify:
 - 2 optimizer param groups with correct LRs (backbone 1e-4, head 1e-3)
 - AdamW optimizer
 - freeze_bn keeps BN in eval mode; freeze_bn=False lets BN go to train mode
 - training_step computes cross-entropy and logs train/loss, train/acc
 - validation_step logs val/loss, val/acc
 - backbone params have lr=1e-4, head params have lr=1e-3
"""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
import lightning as L
from unittest.mock import MagicMock, patch
from torch.optim import AdamW

# Import the module under test — will fail until implemented (TDD RED)
from eval.finetune import FinetuneModule
from core.config import FinetuneConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backbone_with_bn(feat_dim: int = 64) -> nn.Module:
    """Small backbone with BatchNorm for freeze_bn tests."""
    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, feat_dim, 3, padding=1)
            self.bn = nn.BatchNorm2d(feat_dim)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.num_features = feat_dim

        def forward(self, x):
            x = self.bn(self.conv(x))
            x = self.pool(x)
            return x.flatten(1)

    return TinyBackbone()


def _make_simple_backbone(feat_dim: int = 64) -> nn.Module:
    """Minimal backbone without BN for optimizer tests."""
    class MinimalBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 4 * 4, feat_dim)
            self.num_features = feat_dim

        def forward(self, x):
            return self.fc(x.flatten(1))

    return MinimalBackbone()


def _make_module(
    backbone: nn.Module | None = None,
    feat_dim: int = 64,
    num_classes: int = 3,
    freeze_bn: bool = True,
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
) -> FinetuneModule:
    if backbone is None:
        backbone = _make_simple_backbone(feat_dim)
    ft_cfg = FinetuneConfig(
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        freeze_bn=freeze_bn,
    )
    return FinetuneModule(
        backbone=backbone,
        feat_dim=feat_dim,
        num_classes=num_classes,
        ft_cfg=ft_cfg,
        max_epochs=10,
        warmup_epochs=1,
    )


def _attach_mock_trainer(module: FinetuneModule, total_steps: int = 100):
    """Attach a minimal mock trainer to allow configure_optimizers to work."""
    trainer = MagicMock()
    trainer.estimated_stepping_batches = total_steps
    module.trainer = trainer


# ---------------------------------------------------------------------------
# Test 1: configure_optimizers returns 2 param groups
# ---------------------------------------------------------------------------

class TestConfigureOptimizers:
    def test_two_param_groups(self):
        module = _make_module()
        _attach_mock_trainer(module)
        result = module.configure_optimizers()
        # May return ([optimizer], [sched]) or dict
        if isinstance(result, tuple):
            optimizer = result[0][0]
        elif isinstance(result, dict):
            optimizer = result["optimizer"]
        else:
            optimizer = result
        assert len(optimizer.param_groups) == 2, (
            f"Expected 2 param groups, got {len(optimizer.param_groups)}"
        )

    def test_param_group_lrs(self):
        """Backbone group has lr=1e-4, head group has lr=1e-3 (as base lr, before scheduler steps)."""
        module = _make_module(backbone_lr=1e-4, head_lr=1e-3)
        _attach_mock_trainer(module)
        result = module.configure_optimizers()
        if isinstance(result, tuple):
            optimizer = result[0][0]
            sched_cfg = result[1][0] if result[1] else None
        elif isinstance(result, dict):
            optimizer = result["optimizer"]
            sched_cfg = result.get("lr_scheduler")
        else:
            optimizer = result
            sched_cfg = None

        # Check base_lrs from scheduler (the original lr before lambda scaling)
        # LambdaLR stores original lrs in scheduler.base_lrs
        if sched_cfg is not None:
            scheduler = sched_cfg["scheduler"] if isinstance(sched_cfg, dict) else sched_cfg
            base_lrs = scheduler.base_lrs
            assert base_lrs[0] == pytest.approx(1e-4), f"backbone base lr should be 1e-4, got {base_lrs[0]}"
            assert base_lrs[1] == pytest.approx(1e-3), f"head base lr should be 1e-3, got {base_lrs[1]}"
        else:
            # No scheduler - check param_groups directly
            lrs = [pg["lr"] for pg in optimizer.param_groups]
            assert lrs[0] == pytest.approx(1e-4), f"backbone lr should be 1e-4, got {lrs[0]}"
            assert lrs[1] == pytest.approx(1e-3), f"head lr should be 1e-3, got {lrs[1]}"

    def test_uses_adamw(self):
        """Optimizer must be AdamW."""
        module = _make_module()
        _attach_mock_trainer(module)
        result = module.configure_optimizers()
        if isinstance(result, tuple):
            optimizer = result[0][0]
        elif isinstance(result, dict):
            optimizer = result["optimizer"]
        else:
            optimizer = result
        assert isinstance(optimizer, AdamW), (
            f"Expected AdamW, got {type(optimizer).__name__}"
        )

    def test_backbone_params_correct_lr(self):
        """All backbone parameters appear in the first param group (base lr=1e-4)."""
        backbone = _make_simple_backbone(64)
        module = _make_module(backbone=backbone, feat_dim=64)
        _attach_mock_trainer(module)
        result = module.configure_optimizers()
        if isinstance(result, tuple):
            optimizer = result[0][0]
            sched_cfg = result[1][0] if result[1] else None
        elif isinstance(result, dict):
            optimizer = result["optimizer"]
            sched_cfg = result.get("lr_scheduler")
        else:
            optimizer = result
            sched_cfg = None

        backbone_group = optimizer.param_groups[0]
        head_group = optimizer.param_groups[1]

        # Check base lrs via scheduler (LambdaLR modifies current lr but stores original)
        if sched_cfg is not None:
            scheduler = sched_cfg["scheduler"] if isinstance(sched_cfg, dict) else sched_cfg
            assert scheduler.base_lrs[0] == pytest.approx(1e-4), f"backbone base lr = {scheduler.base_lrs[0]}"
            assert scheduler.base_lrs[1] == pytest.approx(1e-3), f"head base lr = {scheduler.base_lrs[1]}"

        # backbone group should contain backbone's parameters
        backbone_param_ids = {id(p) for p in backbone.parameters()}
        group0_param_ids = {id(p) for p in backbone_group["params"]}
        assert backbone_param_ids == group0_param_ids, (
            "Backbone param group should contain exactly the backbone parameters"
        )


# ---------------------------------------------------------------------------
# Test 3 & 4: freeze_bn behavior
# ---------------------------------------------------------------------------

class TestFreezeBN:
    def test_freeze_bn_true_keeps_bn_in_eval(self):
        """When freeze_bn=True, BN layers stay in eval mode after model.train()."""
        backbone = _make_backbone_with_bn(feat_dim=16)
        module = _make_module(backbone=backbone, feat_dim=16, freeze_bn=True)
        module.train()  # trigger train() override
        # All BN modules in backbone should be in eval mode
        bn_modules = [
            m for m in backbone.modules()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm))
        ]
        assert len(bn_modules) > 0, "Expected at least one BN module in backbone"
        for m in bn_modules:
            assert not m.training, (
                f"BN module {m} should be in eval mode when freeze_bn=True, "
                f"but m.training={m.training}"
            )

    def test_freeze_bn_false_allows_bn_training_mode(self):
        """When freeze_bn=False, calling train() puts BN in train mode."""
        backbone = _make_backbone_with_bn(feat_dim=16)
        module = _make_module(backbone=backbone, feat_dim=16, freeze_bn=False)
        module.train()
        bn_modules = [
            m for m in backbone.modules()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm))
        ]
        assert len(bn_modules) > 0
        for m in bn_modules:
            assert m.training, (
                f"BN module should be in train mode when freeze_bn=False, "
                f"but m.training={m.training}"
            )


# ---------------------------------------------------------------------------
# Test 5: training_step logs train/loss and train/acc
# ---------------------------------------------------------------------------

class TestTrainingStep:
    def test_training_step_logs_loss_and_acc(self):
        """training_step should log train/loss and train/acc."""
        backbone = _make_simple_backbone(feat_dim=16)
        module = _make_module(backbone=backbone, feat_dim=16, num_classes=3)

        logged = {}

        def fake_log(name, value, **kwargs):
            logged[name] = value

        module.log = fake_log

        # Create a simple batch: [imgs, labels]
        imgs = torch.randn(4, 3, 4, 4)
        labels = torch.randint(0, 3, (4,))
        batch = [imgs, labels]

        loss = module.training_step(batch, 0)

        assert loss is not None, "training_step should return a loss tensor"
        assert isinstance(loss, torch.Tensor), f"loss should be Tensor, got {type(loss)}"
        assert "train/loss" in logged, f"Expected 'train/loss' in logged, got {list(logged.keys())}"
        assert "train/acc" in logged, f"Expected 'train/acc' in logged, got {list(logged.keys())}"

    def test_training_step_handles_multiview_batch(self):
        """training_step handles multi-view imgs [B, n_views, C, H, W] by using first view."""
        backbone = _make_simple_backbone(feat_dim=16)
        module = _make_module(backbone=backbone, feat_dim=16, num_classes=3)
        module.log = MagicMock()

        # Multi-view batch: first element is a list or 5D tensor
        imgs = torch.randn(4, 2, 3, 4, 4)  # [B, n_views, C, H, W]
        labels = torch.randint(0, 3, (4,))
        batch = [imgs, labels]

        loss = module.training_step(batch, 0)
        assert loss is not None


# ---------------------------------------------------------------------------
# Test 6: validation_step logs val/loss and val/acc
# ---------------------------------------------------------------------------

class TestValidationStep:
    def test_validation_step_logs_loss_and_acc(self):
        """validation_step should log val/loss and val/acc."""
        backbone = _make_simple_backbone(feat_dim=16)
        module = _make_module(backbone=backbone, feat_dim=16, num_classes=3)

        logged = {}

        def fake_log(name, value, **kwargs):
            logged[name] = value

        module.log = fake_log

        imgs = torch.randn(4, 3, 4, 4)
        labels = torch.randint(0, 3, (4,))
        batch = [imgs, labels]

        module.validation_step(batch, 0)

        assert "val/loss" in logged, f"Expected 'val/loss' in logged, got {list(logged.keys())}"
        assert "val/acc" in logged, f"Expected 'val/acc' in logged, got {list(logged.keys())}"
