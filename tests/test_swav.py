"""Tests for SwAV Sinkhorn-Knopp optimal transport, swapped-prediction loss, and SwAVModule.

Reference:
    Caron et al., "Unsupervised Learning of Visual Features by Contrasting
    Cluster Assignments" (NeurIPS 2020). https://arxiv.org/abs/2006.09882
"""
import torch
import torch.nn as nn
import pytest
import numpy as np
from PIL import Image
import lightning as L

from methods.swav.losses import sinkhorn_knopp, swav_loss


# ---------------------------------------------------------------------------
# Sinkhorn-Knopp tests
# ---------------------------------------------------------------------------

def test_sinkhorn_output_shape():
    """Test 1: sinkhorn_knopp returns tensor of same shape as input [B, K]."""
    B, K = 32, 50
    scores = torch.randn(B, K)
    Q = sinkhorn_knopp(scores, n_iters=10)
    assert Q.shape == (B, K), f"Expected shape ({B}, {K}), got {Q.shape}"


def test_sinkhorn_row_sums_uniform():
    """Test 2: Row sums are approximately equal (all close to 1.0, atol=0.05).

    Note: Row sums converge quickly (within a few iterations). n_iters=100 used
    here to also satisfy the tight atol=0.05 for column sums in the next test.
    """
    B, K = 64, 100
    scores = torch.randn(B, K)
    Q = sinkhorn_knopp(scores, n_iters=100)
    row_sums = Q.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(B), atol=0.05), (
        f"Row sums not uniform: min={row_sums.min():.4f}, max={row_sums.max():.4f}"
    )


def test_sinkhorn_column_sums_uniform():
    """Test 3: Column sums are approximately equal (all close to B/K, atol=0.05).

    Note: Column sums require more Sinkhorn iterations to converge than row sums.
    With random scores, n_iters=100 achieves atol=0.05 reliably. The production
    default of n_iters=3 trades accuracy for speed; this test validates convergence.
    """
    B, K = 64, 100
    scores = torch.randn(B, K)
    Q = sinkhorn_knopp(scores, n_iters=100)
    col_sums = Q.sum(dim=0)
    expected = torch.full((K,), B / K)
    assert torch.allclose(col_sums, expected, atol=0.05), (
        f"Column sums not uniform: min={col_sums.min():.4f}, max={col_sums.max():.4f}, expected={B/K:.4f}"
    )


def test_sinkhorn_nonnegative():
    """Test 4: Q values are all non-negative."""
    B, K = 32, 50
    scores = torch.randn(B, K)
    Q = sinkhorn_knopp(scores, n_iters=10)
    assert (Q >= 0).all(), "sinkhorn_knopp returned negative values"


def test_sinkhorn_different_shapes():
    """Test 5: Function works with different B, K combinations."""
    for B, K in [(16, 50), (64, 200)]:
        scores = torch.randn(B, K)
        Q = sinkhorn_knopp(scores, n_iters=100)
        assert Q.shape == (B, K)
        row_sums = Q.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(B), atol=0.05), (
            f"Failed for B={B}, K={K}: row sums not ~1.0"
        )


def test_sinkhorn_no_grad():
    """Test 6: Decorated with @torch.no_grad -- output has requires_grad=False."""
    B, K = 32, 50
    scores = torch.randn(B, K, requires_grad=True)
    Q = sinkhorn_knopp(scores, n_iters=10)
    assert not Q.requires_grad, "sinkhorn_knopp output should not require grad"


def test_sinkhorn_doubly_stochastic():
    """Test that sinkhorn_knopp produces a doubly stochastic matrix (both row and col uniform).

    Uses n_iters=100 for tight convergence verification (atol=0.05).
    """
    B, K = 48, 80
    scores = torch.randn(B, K)
    Q = sinkhorn_knopp(scores, n_iters=100)
    # Row sums ~ 1
    assert torch.allclose(Q.sum(dim=1), torch.ones(B), atol=0.05)
    # Col sums ~ B/K
    expected_col = torch.full((K,), B / K)
    assert torch.allclose(Q.sum(dim=0), expected_col, atol=0.05)


# ---------------------------------------------------------------------------
# swav_loss tests
# ---------------------------------------------------------------------------

class _DummyPrototype(nn.Module):
    """Prototype layer that returns scores by matrix multiply."""

    def __init__(self, in_dim: int, n_prototypes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_prototypes, in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is already L2-normalized in swav_loss
        w = nn.functional.normalize(self.weight, dim=1)
        return x @ w.t()


def test_swav_loss_finite_scalar():
    """Test swav_loss: returns finite scalar loss."""
    B, D, K = 16, 64, 50
    n_large_crops = 2
    n_total_crops = 4
    z_list = [torch.randn(B, D) for _ in range(n_total_crops)]
    prototype_layer = _DummyPrototype(D, K)

    loss = swav_loss(
        z_list=z_list,
        prototype_layer=prototype_layer,
        temperature=0.1,
        n_large_crops=n_large_crops,
        sinkhorn_fn=sinkhorn_knopp,
    )
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


def test_swav_loss_gradient_flow():
    """Test swav_loss: gradients flow through prediction side but not code side (q)."""
    B, D, K = 8, 32, 20
    n_large_crops = 2
    n_total_crops = 3
    z_list = [torch.randn(B, D, requires_grad=True) for _ in range(n_total_crops)]
    prototype_layer = _DummyPrototype(D, K)

    loss = swav_loss(
        z_list=z_list,
        prototype_layer=prototype_layer,
        temperature=0.1,
        n_large_crops=n_large_crops,
        sinkhorn_fn=sinkhorn_knopp,
    )
    loss.backward()
    # All z tensors should have gradients flowing (prediction side)
    for i, z in enumerate(z_list):
        assert z.grad is not None, f"z_list[{i}] has no gradient"


def test_swav_loss_cross_entropy_terms():
    """Test swav_loss: with n_large_crops=2 and 4 total crops, produces valid loss.

    With n_large_crops=2 and n_crops=4:
    - Each large crop i computes codes q_i
    - All OTHER crops v != i predict q_i
    - Total terms = n_large_crops * (n_crops - 1) = 2 * 3 = 6
    The loss should be averaged over these 6 terms.
    """
    B, D, K = 8, 32, 20
    n_large_crops = 2
    n_total_crops = 4
    torch.manual_seed(42)
    z_list = [torch.randn(B, D) for _ in range(n_total_crops)]
    prototype_layer = _DummyPrototype(D, K)

    loss = swav_loss(
        z_list=z_list,
        prototype_layer=prototype_layer,
        temperature=0.1,
        n_large_crops=n_large_crops,
        sinkhorn_fn=sinkhorn_knopp,
    )
    # Loss should be positive (cross-entropy is non-negative) and finite
    assert loss.item() > 0, "Expected positive cross-entropy loss"
    assert torch.isfinite(loss), "Loss should be finite"


# ---------------------------------------------------------------------------
# SwAVModule tests
# ---------------------------------------------------------------------------

@pytest.fixture
def large_imagefolder(tmp_path):
    """Create an ImageFolder (3 classes, 40 images each, 32x32)."""
    rng = np.random.RandomState(42)
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


def _make_swav_cfg(**overrides):
    """Create a minimal TrainConfig for SwAV testing."""
    from core.config import TrainConfig, SwAVConfig
    defaults = {
        "method": "swav",
        "backbone": "resnet18",
        "pretrained": False,
        "max_epochs": 5,
        "warmup_epochs": 0,
        "batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 1e-6,
        "optimizer": "adamw",
        "n_views": 2,
        "swav": SwAVConfig(
            n_prototypes=10,
            freeze_prototypes_epochs=1,
            sinkhorn_iterations=3,
            temperature=0.1,
            epsilon=0.05,
            n_large_crops=2,
            large_size=32,
            n_small_crops=2,
            small_size=16,
        ),
    }
    defaults.update(overrides)
    return TrainConfig(**defaults)


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


def test_swav_dispatcher_registration():
    """Test 1: SwAVModule registered as 'swav' in method_dispatcher."""
    import methods.swav  # noqa: F401 -- trigger registration
    from core.dispatcher import available_methods
    from methods.swav.module import SwAVModule

    assert "swav" in available_methods(), (
        f"'swav' not in available methods: {available_methods()}"
    )


def test_swav_dispatcher_returns_swav_module():
    """Test 2: method_dispatcher with method='swav' returns SwAVModule instance."""
    import methods.swav  # noqa: F401
    from core.dispatcher import method_dispatcher
    from methods.swav.module import SwAVModule

    cfg = _make_swav_cfg()
    model = method_dispatcher(cfg)
    assert isinstance(model, SwAVModule), (
        f"Expected SwAVModule, got {type(model).__name__}"
    )


def test_swav_train_5_epochs(large_imagefolder):
    """Test 3: SwAVModule trains 5 epochs without loss divergence on toy data."""
    L.seed_everything(42)

    import methods.swav  # noqa: F401
    from methods.swav.module import SwAVModule
    from core.data import SSLDataModule, MultiCropDataset
    from torchvision.datasets import ImageFolder

    cfg = _make_swav_cfg(max_epochs=5)

    base_ds = ImageFolder(str(large_imagefolder))
    multi_crop_ds = MultiCropDataset(
        base_ds,
        n_large_crops=2,
        large_size=32,
        n_small_crops=2,
        small_size=16,
        strong=True,
    )
    dm = SSLDataModule(
        data_dir=str(large_imagefolder),
        dataset=multi_crop_ds,
        batch_size=8,
        num_workers=0,
    )

    model = SwAVModule(cfg)
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

    assert len(tracker.epoch_losses) == 5, f"Expected 5 epochs, got {len(tracker.epoch_losses)}"
    for i, loss in enumerate(tracker.epoch_losses):
        assert loss == loss, f"Epoch {i} loss is NaN"
        assert abs(loss) < 1e6, f"Epoch {i} loss diverged: {loss}"

    # Noise-robust convergence check: min of last 3 < max of first 3
    early_loss = max(tracker.epoch_losses[:3])
    late_loss = min(tracker.epoch_losses[-3:])
    assert late_loss < early_loss, (
        f"Loss should decrease over training: "
        f"early_max={early_loss:.4f}, late_min={late_loss:.4f}"
    )


def test_swav_prototype_normalization(large_imagefolder):
    """Test 4: Prototype vectors remain L2-normalized (norm ≈ 1.0) after training."""
    L.seed_everything(42)

    import methods.swav  # noqa: F401
    from methods.swav.module import SwAVModule
    from core.data import SSLDataModule, MultiCropDataset
    from torchvision.datasets import ImageFolder

    cfg = _make_swav_cfg(max_epochs=2)
    n_prototypes = 10

    base_ds = ImageFolder(str(large_imagefolder))
    multi_crop_ds = MultiCropDataset(
        base_ds,
        n_large_crops=2,
        large_size=32,
        n_small_crops=2,
        small_size=16,
        strong=True,
    )
    dm = SSLDataModule(
        data_dir=str(large_imagefolder),
        dataset=multi_crop_ds,
        batch_size=8,
        num_workers=0,
    )

    model = SwAVModule(cfg)
    trainer = L.Trainer(
        max_epochs=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, dm)

    # All prototype rows should have L2 norm ≈ 1.0
    norms = torch.norm(model.prototype_layer.linear.weight, dim=1)
    assert torch.allclose(norms, torch.ones(n_prototypes), atol=0.01), (
        f"Prototype norms not ~1.0 after training: min={norms.min():.4f}, max={norms.max():.4f}"
    )


def test_swav_learnable_params_includes_prototypes():
    """Test 5: learnable_params includes prototype layer parameters."""
    from methods.swav.module import SwAVModule

    cfg = _make_swav_cfg()
    model = SwAVModule(cfg)

    learnable_ids = {id(p) for p in model.learnable_params}
    proto_ids = {id(p) for p in model.prototype_layer.parameters()}

    assert proto_ids.issubset(learnable_ids), (
        "Prototype layer parameters not found in learnable_params"
    )


def test_swav_prototype_gradients_frozen_during_freeze_epoch():
    """Test 6: Prototype gradients are zeroed during freeze_prototypes_epochs."""
    from methods.swav.module import SwAVModule

    cfg = _make_swav_cfg()
    model = SwAVModule(cfg)

    # Simulate a gradient on prototype weights
    model.prototype_layer.linear.weight.grad = torch.ones_like(
        model.prototype_layer.linear.weight
    )

    # At epoch 0 with freeze_prototypes_epochs=1, gradients should be zeroed
    assert model.current_epoch == 0, "Expected epoch 0 at start"
    mock_optimizer = None
    model.on_before_optimizer_step(mock_optimizer)

    grad = model.prototype_layer.linear.weight.grad
    assert torch.all(grad == 0), (
        "Prototype gradients should be zeroed during freeze epochs"
    )
