"""3-epoch smoke tests for BYOL, SimSiam, and Barlow Twins.

Verifies that each method trains for 3 epochs on toy random data without
error and produces a finite loss. Also verifies that:
  - All three YAML configs load via TrainConfig.model_validate without error.
  - EMAUpdater cosine momentum schedule returns correct boundary values.

These are integration-level tests that exercise the full Lightning training
loop (backbone -> projector -> loss -> optimizer -> EMA). Each test runs on
CPU with a small synthetic dataset (64 samples, 32x32 images) in about 5-15s.
"""
from __future__ import annotations

import math

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

import lightning as L

from core.config import BYOLConfig, BarlowTwinsConfig, SimSiamConfig, TrainConfig
from core.ema import EMAUpdater
from methods.barlow_twins.module import BarlowTwinsModule
from methods.byol.module import BYOLModule
from methods.simsiam.module import SimSiamModule


# ---------------------------------------------------------------------------
# Shared dataset helper
# ---------------------------------------------------------------------------

class PairedDataset(Dataset):
    """Synthetic dataset that yields ((view1, view2), label) tuples.

    Matches the batch format expected by all three modules' training_step:
        (v1, v2), _ = batch
    """

    def __init__(self, n: int = 64, C: int = 3, H: int = 32, W: int = 32):
        self.v1 = torch.randn(n, C, H, W)
        self.v2 = torch.randn(n, C, H, W)

    def __len__(self) -> int:
        return len(self.v1)

    def __getitem__(self, i: int):
        return (self.v1[i], self.v2[i]), 0


def make_toy_dataloader(n: int = 64, batch_size: int = 16) -> DataLoader:
    """Return a DataLoader over PairedDataset with deterministic order."""
    return DataLoader(PairedDataset(n=n), batch_size=batch_size, shuffle=False)


def run_smoke(module: L.LightningModule, epochs: int = 3) -> float:
    """Train module for `epochs` epochs and return the final training loss.

    Uses CPU-only Lightning Trainer with minimal logging to keep tests fast.
    Returns the final loss as a float for finiteness assertions.
    """
    dl = make_toy_dataloader()
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(module, dl)
    # callback_metrics contains the most recent logged value for "train/loss"
    loss_tensor = trainer.callback_metrics.get("train/loss")
    assert loss_tensor is not None, "train/loss was not logged — check log_train_metrics"
    return loss_tensor.item()


# ---------------------------------------------------------------------------
# Smoke tests: 3-epoch training loop
# ---------------------------------------------------------------------------

def test_byol_smoke():
    """BYOL trains for 3 epochs on toy data — loss is finite, no NaN."""
    cfg = TrainConfig(
        method="byol",
        backbone="resnet18",
        pretrained=False,
        max_epochs=3,
        batch_size=16,
        lr=3e-4,
        byol=BYOLConfig(base_momentum=0.996, end_momentum=1.0),
    )
    module = BYOLModule(cfg)
    final_loss = run_smoke(module, epochs=3)
    assert math.isfinite(final_loss), f"BYOL final loss is not finite: {final_loss}"


def test_simsiam_smoke():
    """SimSiam trains for 3 epochs on toy data — loss is finite, no NaN."""
    cfg = TrainConfig(
        method="simsiam",
        backbone="resnet18",
        pretrained=False,
        max_epochs=3,
        batch_size=16,
        lr=0.05,
        weight_decay=1e-4,
        optimizer="sgd",
        simsiam=SimSiamConfig(predictor_hidden_dim=512),
    )
    module = SimSiamModule(cfg)
    final_loss = run_smoke(module, epochs=3)
    assert math.isfinite(final_loss), f"SimSiam final loss is not finite: {final_loss}"


def test_barlow_twins_smoke():
    """Barlow Twins trains for 3 epochs on toy data — loss is finite, no NaN."""
    cfg = TrainConfig(
        method="barlow_twins",
        backbone="resnet18",
        pretrained=False,
        max_epochs=3,
        batch_size=16,
        lr=3e-4,
        barlow_twins=BarlowTwinsConfig(lambda_coeff=5e-3, projection_dim=512),
    )
    module = BarlowTwinsModule(cfg)
    final_loss = run_smoke(module, epochs=3)
    assert math.isfinite(final_loss), f"Barlow Twins final loss is not finite: {final_loss}"


# ---------------------------------------------------------------------------
# EMA momentum schedule boundary test
# ---------------------------------------------------------------------------

def test_byol_ema_momentum_schedule():
    """EMA momentum at step 0 is base_momentum; at total_steps is end_momentum.

    Verifies the cosine schedule boundary conditions from the BYOL paper:
      m(0) = base_momentum  (start of training, slower updates)
      m(T) = end_momentum   (end of training, nearly frozen target)
    """
    base, end, total = 0.996, 1.0, 1000
    ema = EMAUpdater(base_momentum=base, end_momentum=end, total_steps=total)

    # At step 0 (freshly constructed)
    assert abs(ema.current_momentum - base) < 1e-6, (
        f"At step 0, momentum should be {base}, got {ema.current_momentum}"
    )

    # Advance to final step by setting _step directly (no param tensors needed)
    ema._step = total
    assert abs(ema.current_momentum - end) < 1e-6, (
        f"At step {total}, momentum should be {end}, got {ema.current_momentum}"
    )


# ---------------------------------------------------------------------------
# YAML config loading test
# ---------------------------------------------------------------------------

def test_yaml_configs_load():
    """All three YAML configs load via TrainConfig.model_validate without error."""
    configs = [
        "configs/byol_resnet18.yaml",
        "configs/simsiam_resnet18.yaml",
        "configs/barlow_twins_resnet18.yaml",
    ]
    for path in configs:
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = TrainConfig.model_validate(raw)
        assert cfg.method in {"byol", "simsiam", "barlow_twins"}, (
            f"{path}: unexpected method={cfg.method!r}"
        )
