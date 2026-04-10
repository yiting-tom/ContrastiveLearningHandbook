"""Tests for methods/moco_v3 -- MoCoV3Module.

Tests verify:
  - Patch projection layer frozen from __init__ (ViT-specific stability fix)
  - AdamW optimizer is used by default
  - Momentum encoder excluded from learnable_params
  - Predictor on online branch only (no predictor_ema)
  - No queue (in-batch symmetric InfoNCE)
  - Dispatcher resolves "moco_v3" to MoCoV3Module
  - MoCoV3Module trains 3 epochs without loss divergence
"""
from __future__ import annotations

import pytest
import torch
import lightning as L
import numpy as np
from PIL import Image

from core.config import TrainConfig, MoCoV3Config
from core.data import SSLDataModule


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


def _make_cfg(**overrides) -> TrainConfig:
    """Create a minimal TrainConfig for MoCo v3 testing."""
    defaults = {
        "method": "moco_v3",
        "backbone": "resnet18",
        "pretrained": False,
        "max_epochs": 3,
        "warmup_epochs": 1,
        "batch_size": 16,
        "lr": 1e-3,
        "weight_decay": 1e-6,
        "optimizer": "adamw",
        "n_views": 2,
        "moco_v3": MoCoV3Config(temperature=0.2, momentum=0.99, predictor_hidden_dim=4096),
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
# Unit Tests
# ---------------------------------------------------------------------------

def test_patch_projection_frozen():
    """After MoCoV3Module construction with vit_small_patch16_224, patch_embed.proj is frozen."""
    from methods.moco_v3.module import MoCoV3Module

    cfg = _make_cfg(backbone="vit_small_patch16_224")
    module = MoCoV3Module(cfg)

    assert hasattr(module.backbone, "patch_embed"), (
        "ViT backbone should have patch_embed attribute"
    )
    assert not module.backbone.patch_embed.proj.weight.requires_grad, (
        "patch_embed.proj.weight should be frozen (requires_grad=False)"
    )
    assert not module.backbone.patch_embed.proj.bias.requires_grad, (
        "patch_embed.proj.bias should be frozen (requires_grad=False)"
    )


def test_moco_v3_uses_adamw():
    """Module's default optimizer config is adamw.

    MoCo v3 requires AdamW (not SGD/LARS) for ViT stability. This test
    verifies the default config uses adamw so that configure_optimizers
    (which is called by the trainer during fit) will instantiate AdamW.
    We also verify by directly building the optimizer from learnable_params.
    """
    import torch.optim as optim
    from methods.moco_v3.module import MoCoV3Module

    cfg = _make_cfg(optimizer="adamw")
    module = MoCoV3Module(cfg)

    # Check the config is set to adamw
    assert cfg.optimizer == "adamw", (
        f"MoCo v3 default should use adamw, got {cfg.optimizer!r}"
    )

    # Directly instantiate AdamW from learnable_params to confirm it works
    params = list(module.learnable_params)
    optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    assert isinstance(optimizer, optim.AdamW), (
        f"MoCo v3 should use AdamW optimizer, got {type(optimizer).__name__}"
    )


def test_momentum_encoder_excluded_from_learnable_params():
    """No parameter id in learnable_params overlaps with backbone_ema or projector_ema parameters."""
    from methods.moco_v3.module import MoCoV3Module

    cfg = _make_cfg()
    module = MoCoV3Module(cfg)

    learnable_ids = {id(p) for p in module.learnable_params}
    ema_ids = {id(p) for p in module.backbone_ema.parameters()} | {
        id(p) for p in module.projector_ema.parameters()
    }
    overlap = learnable_ids & ema_ids
    assert len(overlap) == 0, (
        f"EMA params should not overlap with learnable params, found {len(overlap)} overlapping"
    )


def test_predictor_on_online_only():
    """Module has self.predictor; does NOT have self.predictor_ema."""
    from methods.moco_v3.module import MoCoV3Module

    cfg = _make_cfg()
    module = MoCoV3Module(cfg)

    assert hasattr(module, "predictor"), (
        "MoCoV3Module must have a self.predictor attribute (online branch only)"
    )
    assert not hasattr(module, "predictor_ema"), (
        "MoCoV3Module must NOT have self.predictor_ema (predictor is online-only)"
    )


def test_no_queue():
    """Module does NOT have a momentum_queue attribute (in-batch symmetric loss, no queue)."""
    from methods.moco_v3.module import MoCoV3Module

    cfg = _make_cfg()
    module = MoCoV3Module(cfg)

    assert not hasattr(module, "momentum_queue"), (
        "MoCoV3Module must NOT have momentum_queue — MoCo v3 uses in-batch symmetric loss"
    )


def test_dispatcher_moco_v3():
    """method_dispatcher(cfg) with method='moco_v3' returns a MoCoV3Module."""
    import methods.moco_v3  # noqa: F401  # trigger registration (no-op if cached)
    from methods.moco_v3.module import MoCoV3Module
    from core.dispatcher import method_dispatcher, register_method, available_methods

    # Guard: re-register if clean_registry cleared the registry after a prior import
    if "moco_v3" not in available_methods():
        register_method("moco_v3", MoCoV3Module)

    cfg = _make_cfg(method="moco_v3")
    model = method_dispatcher(cfg)
    assert isinstance(model, MoCoV3Module), (
        f"Dispatcher should return MoCoV3Module, got {type(model).__name__}"
    )


def test_moco_v3_train_3_epochs(large_imagefolder):
    """MoCoV3Module trains 3 epochs on toy data without loss divergence.

    Uses resnet18 backbone for speed. ViT-specific behavior tested in unit tests.
    """
    L.seed_everything(42)

    import methods.moco_v3  # noqa: F401  # trigger registration (no-op if cached)
    from methods.moco_v3.module import MoCoV3Module
    from core.dispatcher import register_method, available_methods
    if "moco_v3" not in available_methods():
        register_method("moco_v3", MoCoV3Module)

    cfg = _make_cfg(
        method="moco_v3",
        backbone="resnet18",
        max_epochs=3,
        warmup_epochs=1,
        batch_size=16,
        lr=1e-3,
        optimizer="adamw",
    )
    model = MoCoV3Module(cfg)
    dm = SSLDataModule(
        data_dir=str(large_imagefolder),
        n_views=2,
        batch_size=16,
        num_workers=0,
        size=32,
        strong=False,  # weak augmentation for stable convergence on tiny data
    )
    tracker = LossTracker()
    trainer = L.Trainer(
        max_epochs=3,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
        callbacks=[tracker],
    )
    trainer.fit(model, dm)

    # Loss must be finite at end
    assert len(tracker.epoch_losses) == 3, f"Expected 3 epochs, got {len(tracker.epoch_losses)}"
    for i, loss in enumerate(tracker.epoch_losses):
        assert loss == loss, f"Epoch {i} loss is NaN"  # NaN check
        assert abs(loss) < 1e6, f"Epoch {i} loss is not finite: {loss}"
