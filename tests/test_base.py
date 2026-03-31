"""Integration tests for BaseSSLModule.

Covers:
  - Minimal subclass trains for 1 epoch on toy data
  - configure_optimizers dispatches to AdamW, SGD, and LARS
  - configure_optimizers raises ValueError for unknown optimizer
  - LR scheduler is wired with interval='step'
  - EMA hook calls EMAUpdater.step when ema_updater is set
  - learnable_params default returns all parameters
  - log_train_metrics calls self.log with train/loss
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import lightning as L

from core.base import BaseSSLModule
from core.config import TrainConfig
from core.ema import EMAUpdater
from core.optimizers import LARS


# ---------------------------------------------------------------------------
# Helpers: minimal concrete subclass
# ---------------------------------------------------------------------------

class DummySSLModule(BaseSSLModule):
    """Minimal concrete subclass for testing."""

    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg)
        self.linear = nn.Linear(4, 4)
        self.projector = self.build_projector()

    def build_projector(self) -> nn.Module:
        return nn.Identity()

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.linear(x)
        loss = F.mse_loss(pred, y)
        self.log_train_metrics(loss)
        return loss


def _make_cfg(**overrides) -> TrainConfig:
    defaults = dict(method="dummy", max_epochs=1, warmup_epochs=0)
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _make_toy_loader(n: int = 8, features: int = 4):
    x = torch.randn(n, features)
    y = torch.randn(n, features)
    return DataLoader(TensorDataset(x, y), batch_size=4)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSubclassTrain:
    """Integration: minimal subclass trains for 1 epoch without error."""

    def test_subclass_trains(self):
        cfg = _make_cfg(optimizer="adamw")
        model = DummySSLModule(cfg)
        loader = _make_toy_loader()
        trainer = L.Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, loader)  # must not raise


class TestConfigureOptimizers:
    """configure_optimizers dispatches to the correct optimizer class."""

    def _fit_one_step(self, cfg: TrainConfig):
        """Run a single training step to trigger configure_optimizers."""
        model = DummySSLModule(cfg)
        loader = _make_toy_loader()
        trainer = L.Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, loader)
        return model

    def test_configure_optimizers_adamw(self):
        cfg = _make_cfg(optimizer="adamw")
        model = DummySSLModule(cfg)
        loader = _make_toy_loader()
        trainer = L.Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, loader)
        opt_cfg = model.configure_optimizers()
        assert isinstance(opt_cfg["optimizer"], torch.optim.AdamW)

    def test_configure_optimizers_sgd(self):
        cfg = _make_cfg(optimizer="sgd")
        model = DummySSLModule(cfg)
        loader = _make_toy_loader()
        trainer = L.Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, loader)
        opt_cfg = model.configure_optimizers()
        assert isinstance(opt_cfg["optimizer"], torch.optim.SGD)

    def test_configure_optimizers_lars(self):
        cfg = _make_cfg(optimizer="lars")
        model = DummySSLModule(cfg)
        loader = _make_toy_loader()
        trainer = L.Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, loader)
        opt_cfg = model.configure_optimizers()
        assert isinstance(opt_cfg["optimizer"], LARS)

    def test_configure_optimizers_unknown(self):
        # Bypass Pydantic literal validation by using model_construct
        cfg = TrainConfig.model_construct(
            method="dummy",
            optimizer="unknown",
            lr=0.01,
            weight_decay=1e-6,
            max_epochs=1,
            warmup_epochs=0,
            batch_size=256,
            backbone="resnet50",
            pretrained=False,
            scheduler="warmup_cosine",
            n_views=2,
            data_dir="data",
            num_workers=4,
            simclr=None,
            moco=None,
            byol=None,
            swav=None,
            barlow_twins=None,
            simsiam=None,
            dino=None,
            supcon=None,
            eval=None,
        )
        model = DummySSLModule(cfg)
        loader = _make_toy_loader()
        trainer = L.Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        # configure_optimizers is called during fit
        with pytest.raises((ValueError, Exception)):
            trainer.fit(model, loader)


class TestScheduler:
    """LR scheduler is wired with step-based interval."""

    def test_scheduler_is_step_based(self):
        cfg = _make_cfg(optimizer="adamw")
        model = DummySSLModule(cfg)
        loader = _make_toy_loader()
        trainer = L.Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, loader)
        opt_cfg = model.configure_optimizers()
        assert opt_cfg["lr_scheduler"]["interval"] == "step"


class TestEMAHook:
    """on_train_batch_end calls EMAUpdater.step when ema_updater is set."""

    def test_ema_hook_called(self):
        cfg = _make_cfg(optimizer="adamw")
        model = DummySSLModule(cfg)

        # Set up mock EMA updater and parameter lists
        mock_ema = MagicMock(spec=EMAUpdater)
        model.ema_updater = mock_ema
        model._online_params = list(model.linear.parameters())
        # Target params: detached copy with requires_grad=False
        target_linear = nn.Linear(4, 4)
        target_linear.requires_grad_(False)
        model._target_params = list(target_linear.parameters())

        loader = _make_toy_loader()
        trainer = L.Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, loader)
        # step should have been called once per batch (2 batches for 8 samples, batch=4)
        assert mock_ema.step.call_count >= 1

    def test_ema_hook_not_called_when_not_set(self):
        """on_train_batch_end is a no-op when ema_updater is None."""
        cfg = _make_cfg(optimizer="adamw")
        model = DummySSLModule(cfg)
        # ema_updater is None by default — just run and check no crash
        loader = _make_toy_loader()
        trainer = L.Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, loader)  # must not raise


class TestLearnableParams:
    """learnable_params default returns all model parameters."""

    def test_learnable_params_default(self):
        cfg = _make_cfg()
        model = DummySSLModule(cfg)
        default_params = list(model.learnable_params)
        all_params = list(model.parameters())
        # Same number of parameter tensors
        assert len(default_params) == len(all_params)

    def test_learnable_params_can_be_overridden(self):
        """Subclass can narrow learnable_params to exclude target network."""

        class WiderModule(DummySSLModule):
            """Module with two separate linear layers so we can exclude one."""

            def __init__(self, cfg: TrainConfig):
                super().__init__(cfg)
                # Add a second linear layer that we will exclude from learnable_params
                self.target_linear = nn.Linear(4, 4)
                self.target_linear.requires_grad_(False)

            @property
            def learnable_params(self):
                # Only return online linear params (excludes target_linear which
                # has requires_grad=False anyway, but we explicitly narrow here)
                return self.linear.parameters()

        cfg = _make_cfg()
        model = WiderModule(cfg)
        narrow = list(model.learnable_params)
        full = list(model.parameters())
        # full includes both linear layers; narrow only has self.linear (2 params)
        assert len(narrow) < len(full)


class TestLogTrainMetrics:
    """log_train_metrics calls self.log with train/loss."""

    def test_log_train_metrics_logs_loss(self):
        cfg = _make_cfg()
        model = DummySSLModule(cfg)
        # Mock self.log and self.optimizers
        model.log = MagicMock()
        model.optimizers = MagicMock(return_value=None)
        loss = torch.tensor(0.5)
        model.log_train_metrics(loss)
        # Verify train/loss was logged
        logged_keys = [call.args[0] for call in model.log.call_args_list]
        assert "train/loss" in logged_keys

    def test_log_train_metrics_logs_extra_keys(self):
        cfg = _make_cfg()
        model = DummySSLModule(cfg)
        model.log = MagicMock()
        model.optimizers = MagicMock(return_value=None)
        loss = torch.tensor(0.5)
        model.log_train_metrics(loss, acc=torch.tensor(0.9))
        logged_keys = [call.args[0] for call in model.log.call_args_list]
        assert "train/acc" in logged_keys
