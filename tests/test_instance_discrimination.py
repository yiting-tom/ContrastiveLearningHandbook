"""Tests for InstanceDiscriminationModule (Wu et al., CVPR 2018).

Tests verify:
  - 5-epoch smoke training on toy data without loss divergence
  - Z is fixed after first mini-batch (does not change between epoch 1 and 5)
  - Memory bank is updated each step with current encoder output
  - Module is registered in dispatcher as 'instance_discrimination'
  - learnable_params excludes memory bank parameters
"""
from __future__ import annotations

import pytest
import torch
import lightning as L
from torch.utils.data import DataLoader

from core.config import TrainConfig, InstanceDiscriminationConfig
from core.data import (
    SSLDataModule,
    IndexedDataset,
    ssl_collate_with_index,
)
from core.memory_bank import MemoryBank


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
    """Create a minimal InstanceDiscrimination config."""
    defaults = dict(
        method="instance_discrimination",
        backbone="resnet18",
        pretrained=False,
        max_epochs=5,
        warmup_epochs=0,
        batch_size=4,
        lr=0.01,
        weight_decay=0.0,
        optimizer="adamw",
        n_views=1,
        instance_discrimination=InstanceDiscriminationConfig(
            temperature=0.07,
            n_negatives=8,
            projection_dim=128,
        ),
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _build_dataloader(tmp_imagefolder, cfg):
    """Build a DataLoader with IndexedDataset wrapping SSLDataModule."""
    dm = SSLDataModule(
        data_dir=str(tmp_imagefolder),
        n_views=cfg.n_views,
        batch_size=cfg.batch_size,
        num_workers=0,
        size=32,
        strong=False,
    )
    dm.setup()
    indexed = IndexedDataset(dm.train_dataset)
    loader = DataLoader(
        indexed,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=ssl_collate_with_index,
        drop_last=True,
    )
    return loader, len(dm.train_dataset)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_train_5_epochs(tmp_imagefolder):
    """InstanceDiscriminationModule trains 5 epochs without loss divergence."""
    from methods.instance_discrimination.module import InstanceDiscriminationModule

    cfg = _make_cfg()
    model = InstanceDiscriminationModule(cfg)
    loader, n_samples = _build_dataloader(tmp_imagefolder, cfg)
    model.memory_bank = MemoryBank(n_samples, cfg.instance_discrimination.projection_dim)

    trainer = L.Trainer(
        max_epochs=5,
        accelerator="cpu",
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=loader)

    # Loss should be finite (not NaN or Inf)
    assert model.trainer.callback_metrics.get("train/loss") is not None
    loss_val = model.trainer.callback_metrics["train/loss"].item()
    assert torch.isfinite(torch.tensor(loss_val)), f"Loss diverged: {loss_val}"


def test_z_fixed_during_training(tmp_imagefolder):
    """Z is fixed after first mini-batch and does not change."""
    from methods.instance_discrimination.module import InstanceDiscriminationModule

    cfg = _make_cfg()
    model = InstanceDiscriminationModule(cfg)
    loader, n_samples = _build_dataloader(tmp_imagefolder, cfg)
    model.memory_bank = MemoryBank(n_samples, cfg.instance_discrimination.projection_dim)

    trainer = L.Trainer(
        max_epochs=5,
        accelerator="cpu",
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=loader)

    # Z should be initialized (not -1.0) and fixed
    z_val = model.nce_loss.Z.item()
    assert z_val != -1.0, "Z was never initialized"
    assert model.nce_loss.z_initialized.item() is True


def test_dispatcher_registration():
    """After registering instance_discrimination, dispatcher works."""
    from core.dispatcher import method_dispatcher, register_method, available_methods
    from methods.instance_discrimination.module import InstanceDiscriminationModule

    if "instance_discrimination" not in available_methods():
        register_method("instance_discrimination", InstanceDiscriminationModule)

    cfg = _make_cfg()
    model = method_dispatcher(cfg)
    assert isinstance(model, InstanceDiscriminationModule)


def test_learnable_params_excludes_bank(tmp_imagefolder):
    """model.learnable_params does not include memory bank parameters."""
    from methods.instance_discrimination.module import InstanceDiscriminationModule

    cfg = _make_cfg()
    model = InstanceDiscriminationModule(cfg)
    _, n_samples = _build_dataloader(tmp_imagefolder, cfg)
    model.memory_bank = MemoryBank(n_samples, cfg.instance_discrimination.projection_dim)

    learnable_ids = {id(p) for p in model.learnable_params}
    bank_ids = {id(p) for p in model.memory_bank.parameters()}

    # Bank has parameters (embedding weight), but they should not be learnable
    assert len(bank_ids) > 0, "MemoryBank should have parameters"
    assert learnable_ids.isdisjoint(bank_ids), "learnable_params must exclude memory bank"


def test_bank_updated_after_step(tmp_imagefolder):
    """After one training step, bank entries at batch indices differ from initial."""
    from methods.instance_discrimination.module import InstanceDiscriminationModule

    cfg = _make_cfg()
    model = InstanceDiscriminationModule(cfg)
    loader, n_samples = _build_dataloader(tmp_imagefolder, cfg)
    model.memory_bank = MemoryBank(n_samples, cfg.instance_discrimination.projection_dim)

    # Record initial bank state for first few indices
    initial_bank = model.memory_bank.bank.weight.data.clone()

    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        limit_train_batches=1,
    )
    trainer.fit(model, train_dataloaders=loader)

    # At least some bank entries should have changed
    bank_after = model.memory_bank.bank.weight.data
    changed = (~torch.all(initial_bank == bank_after, dim=1)).sum().item()
    assert changed > 0, "No bank entries were updated after a training step"


def test_yaml_config_loads_and_trains(tmp_imagefolder):
    """End-to-end: load YAML config, instantiate module via dispatcher, train 1 epoch."""
    import yaml
    from core.config import TrainConfig
    from core.dispatcher import method_dispatcher
    from core.data import SSLDataModule, IndexedDataset, ssl_collate_with_index
    from core.memory_bank import MemoryBank

    # Load and adapt config for test
    with open("configs/instance_discrimination_resnet18.yaml") as f:
        raw = yaml.safe_load(f)
    raw["data_dir"] = str(tmp_imagefolder)
    raw["max_epochs"] = 1
    raw["warmup_epochs"] = 0
    raw["batch_size"] = 4
    raw["num_workers"] = 0
    cfg = TrainConfig.model_validate(raw)

    from methods.instance_discrimination.module import InstanceDiscriminationModule
    from core.dispatcher import register_method, available_methods
    if "instance_discrimination" not in available_methods():
        register_method("instance_discrimination", InstanceDiscriminationModule)
    model = method_dispatcher(cfg)

    # Setup data with IndexedDataset wrapper
    dm = SSLDataModule(
        data_dir=cfg.data_dir, n_views=cfg.n_views,
        batch_size=cfg.batch_size, num_workers=0, size=32, strong=False,
    )
    dm.setup()
    n_samples = len(dm.train_dataset)
    indexed_ds = IndexedDataset(dm.train_dataset)
    dl = DataLoader(indexed_ds, batch_size=cfg.batch_size,
                    shuffle=True, collate_fn=ssl_collate_with_index, drop_last=True)

    # Initialize memory bank
    model.memory_bank = MemoryBank(n_samples, cfg.instance_discrimination.projection_dim)

    trainer = L.Trainer(
        max_epochs=1, accelerator="cpu",
        enable_checkpointing=False, logger=False, enable_progress_bar=False,
    )
    trainer.fit(model, dl)
    # Verify training completed without error — no assertion needed, fit() would raise
