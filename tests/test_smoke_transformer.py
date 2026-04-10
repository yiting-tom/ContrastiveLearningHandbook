"""Smoke tests for MoCo v3, DINO, YAML config loading, and DOC-02 docstrings.

Verifies:
  - moco_v3_vit_small.yaml loads via load_config() with correct field values.
  - dino_vit_small.yaml loads via load_config() with correct field values.
  - MoCoV3Module trains 3 epochs on toy random data (CPU) without error; loss is finite.
  - DINOModule trains 3 epochs on toy random data (CPU) without error; loss is finite.
  - MoCoV3Module module-level docstring satisfies DOC-02: author, venue, arXiv, patch gotcha.
  - DINOModule module-level docstring satisfies DOC-02: author, venue, arXiv, centering gotcha.

All smoke tests run on CPU with small synthetic datasets (16 samples, 32x32 images)
and complete in about 10-30s each.
"""
from __future__ import annotations

import math
import inspect

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import lightning as L

from core.config import MoCoV3Config, DINOConfig, TrainConfig
from core.config import load_config


# ---------------------------------------------------------------------------
# Shared test dataset
# ---------------------------------------------------------------------------

class PairedDataset(Dataset):
    """Synthetic dataset returning ((view1, view2), label) tuples.

    Batch format expected by MoCoV3Module.training_step:
        views, labels = batch  # views is tuple of (v1, v2)
    """

    def __init__(self, n: int = 64, C: int = 3, H: int = 32, W: int = 32):
        self.v1 = torch.randn(n, C, H, W)
        self.v2 = torch.randn(n, C, H, W)

    def __len__(self) -> int:
        return len(self.v1)

    def __getitem__(self, i: int):
        return (self.v1[i], self.v2[i]), 0


class MultiCropToyDataset(Dataset):
    """Synthetic dataset returning (crops_list, label) tuples for DINO.

    Returns 2 global crops only (simple 2-view mode for CPU smoke test).
    Batch format expected by DINOModule.training_step:
        crops_list, labels = batch
    """

    def __init__(self, n: int = 64, n_views: int = 2, C: int = 3, H: int = 32, W: int = 32):
        self.n = n
        self.n_views = n_views
        self.data = torch.randn(n_views, n, C, H, W)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        crops = [self.data[v, i] for v in range(self.n_views)]
        return crops, 0


def multicrop_collate_fn(batch):
    """Collate list of (crops_list, label) into (list_of_stacked_crops, labels_tensor)."""
    all_crops, labels = zip(*batch)
    n_crops = len(all_crops[0])
    crops_list = [torch.stack([sample[c] for sample in all_crops]) for c in range(n_crops)]
    return crops_list, torch.tensor(labels, dtype=torch.long)


def make_paired_dataloader(n: int = 64, batch_size: int = 16) -> DataLoader:
    """DataLoader over PairedDataset with deterministic order."""
    return DataLoader(PairedDataset(n=n), batch_size=batch_size, shuffle=False)


def make_multicrop_dataloader(n: int = 64, batch_size: int = 16, n_views: int = 2) -> DataLoader:
    """DataLoader over MultiCropToyDataset with deterministic order."""
    return DataLoader(
        MultiCropToyDataset(n=n, n_views=n_views),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multicrop_collate_fn,
    )


# ---------------------------------------------------------------------------
# YAML config loading tests
# ---------------------------------------------------------------------------

def test_moco_v3_yaml_valid():
    """configs/moco_v3_vit_small.yaml loads via load_config() and has correct values."""
    cfg = load_config("configs/moco_v3_vit_small.yaml")
    assert cfg.method == "moco_v3", f"Expected method='moco_v3', got {cfg.method!r}"
    assert cfg.backbone == "vit_small_patch16_224", (
        f"Expected backbone='vit_small_patch16_224', got {cfg.backbone!r}"
    )
    assert cfg.optimizer == "adamw", f"Expected optimizer='adamw', got {cfg.optimizer!r}"
    assert cfg.moco_v3 is not None, "moco_v3 sub-config must be present"
    assert cfg.moco_v3.temperature == 0.2, (
        f"Expected moco_v3.temperature=0.2, got {cfg.moco_v3.temperature}"
    )
    assert cfg.gradient_clip_val == 1.0, (
        f"Expected gradient_clip_val=1.0, got {cfg.gradient_clip_val}"
    )


def test_dino_yaml_valid():
    """configs/dino_vit_small.yaml loads via load_config() and has correct values."""
    cfg = load_config("configs/dino_vit_small.yaml")
    assert cfg.method == "dino", f"Expected method='dino', got {cfg.method!r}"
    assert cfg.backbone == "vit_small_patch16_224", (
        f"Expected backbone='vit_small_patch16_224', got {cfg.backbone!r}"
    )
    assert cfg.dino is not None, "dino sub-config must be present"
    assert cfg.dino.n_prototypes == 65536, (
        f"Expected dino.n_prototypes=65536, got {cfg.dino.n_prototypes}"
    )
    assert cfg.gradient_clip_val == 3.0, (
        f"Expected gradient_clip_val=3.0, got {cfg.gradient_clip_val}"
    )


# ---------------------------------------------------------------------------
# Smoke tests: 3-epoch training loop
# ---------------------------------------------------------------------------

def test_smoke_moco_v3_train():
    """MoCoV3Module trains 3 epochs on toy CPU data — loss is finite, no NaN."""
    from methods.moco_v3.module import MoCoV3Module

    cfg = TrainConfig(
        method="moco_v3",
        backbone="resnet18",
        pretrained=False,
        max_epochs=3,
        warmup_epochs=1,
        batch_size=16,
        lr=1e-3,
        optimizer="adamw",
        moco_v3=MoCoV3Config(temperature=0.2, momentum=0.99, predictor_hidden_dim=256),
    )
    module = MoCoV3Module(cfg)
    dl = make_paired_dataloader(n=64, batch_size=16)

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(module, dl)

    loss_tensor = trainer.callback_metrics.get("train/loss")
    assert loss_tensor is not None, "train/loss was not logged — check log_train_metrics"
    final_loss = loss_tensor.item()
    assert math.isfinite(final_loss), f"MoCoV3 final loss is not finite: {final_loss}"


def test_smoke_dino_train():
    """DINOModule trains 3 epochs on toy CPU data (2-view mode) — loss is finite, no NaN."""
    from methods.dino.module import DINOModule

    cfg = TrainConfig(
        method="dino",
        backbone="resnet18",
        pretrained=False,
        max_epochs=3,
        warmup_epochs=0,
        batch_size=16,
        lr=1e-3,
        optimizer="adamw",
        n_views=2,
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
    dl = make_multicrop_dataloader(n=64, batch_size=16, n_views=2)

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        gradient_clip_val=3.0,
    )
    trainer.fit(module, dl)

    loss_tensor = trainer.callback_metrics.get("train/loss")
    assert loss_tensor is not None, "train/loss was not logged — check log_train_metrics"
    final_loss = loss_tensor.item()
    assert math.isfinite(final_loss), f"DINO final loss is not finite: {final_loss}"


# ---------------------------------------------------------------------------
# DOC-02 docstring verification
# ---------------------------------------------------------------------------

def test_doc02_moco_v3():
    """MoCoV3Module (or its module docstring) satisfies DOC-02 requirements.

    DOC-02: Each method module must document author, venue, arXiv link, and gotchas.
    For MoCo v3: Chen (author), ICCV 2021 (venue), arxiv link, patch projection freeze gotcha.
    """
    import methods.moco_v3.module as mod
    from methods.moco_v3.module import MoCoV3Module

    # Collect all available docstring text (module + class)
    docs = []
    if mod.__doc__:
        docs.append(mod.__doc__)
    if MoCoV3Module.__doc__:
        docs.append(MoCoV3Module.__doc__)
    full_doc = "\n".join(docs)

    assert "Chen" in full_doc, (
        "MoCoV3 docstring must mention author 'Chen' (Xinlei Chen et al., ICCV 2021)"
    )
    assert "ICCV 2021" in full_doc, (
        "MoCoV3 docstring must mention venue 'ICCV 2021'"
    )
    assert "arxiv" in full_doc.lower(), (
        "MoCoV3 docstring must include arXiv link (case-insensitive)"
    )
    assert "patch" in full_doc.lower(), (
        "MoCoV3 docstring must mention 'patch' (frozen patch_embed.proj gotcha)"
    )


def test_doc02_dino():
    """DINOModule (or its module docstring) satisfies DOC-02 requirements.

    DOC-02: Each method module must document author, venue, arXiv link, and gotchas.
    For DINO: Caron (author), ICCV 2021 (venue), arxiv link, centering gotcha.
    """
    import methods.dino.module as mod
    from methods.dino.module import DINOModule

    # Collect all available docstring text (module + class)
    docs = []
    if mod.__doc__:
        docs.append(mod.__doc__)
    if DINOModule.__doc__:
        docs.append(DINOModule.__doc__)
    full_doc = "\n".join(docs)

    assert "Caron" in full_doc, (
        "DINO docstring must mention author 'Caron' (Mathilde Caron et al., ICCV 2021)"
    )
    assert "ICCV 2021" in full_doc, (
        "DINO docstring must mention venue 'ICCV 2021'"
    )
    assert "arxiv" in full_doc.lower(), (
        "DINO docstring must include arXiv link (case-insensitive)"
    )
    assert "centering" in full_doc.lower(), (
        "DINO docstring must mention 'centering' (collapse prevention gotcha)"
    )
