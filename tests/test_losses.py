"""Tests for InfoNCELoss — symmetric and asymmetric modes."""
import torch
import torch.nn.functional as F
import pytest

from core.losses import InfoNCELoss


def make_normalized(batch_size: int, dim: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return F.normalize(torch.randn(batch_size, dim), dim=1)


# ---------------------------------------------------------------------------
# Test 1: Symmetric mode produces finite, positive scalar
# ---------------------------------------------------------------------------
def test_symmetric_finite_positive_scalar():
    loss_fn = InfoNCELoss(temperature=0.5)
    z_i = make_normalized(32, 128, seed=0)
    z_j = make_normalized(32, 128, seed=1)
    result = loss_fn(z_i, z_j)
    assert result.shape == (), "Expected scalar output"
    assert torch.isfinite(result), "Loss must be finite"
    assert result.item() > 0, "Loss must be positive"


# ---------------------------------------------------------------------------
# Test 2: Symmetric mode is symmetric: loss(z_i, z_j) == loss(z_j, z_i)
# ---------------------------------------------------------------------------
def test_symmetric_symmetry_property():
    loss_fn = InfoNCELoss(temperature=0.5)
    z_i = make_normalized(32, 128, seed=0)
    z_j = make_normalized(32, 128, seed=1)
    loss_ij = loss_fn(z_i, z_j)
    loss_ji = loss_fn(z_j, z_i)
    assert torch.isclose(loss_ij, loss_ji, atol=1e-5), (
        f"loss(z_i, z_j)={loss_ij.item():.6f} != loss(z_j, z_i)={loss_ji.item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3: Asymmetric mode with queue produces finite, positive scalar
# ---------------------------------------------------------------------------
def test_asymmetric_with_queue_finite_positive_scalar():
    loss_fn = InfoNCELoss(temperature=0.5)
    z_i = make_normalized(32, 128, seed=0)
    z_j = make_normalized(32, 128, seed=1)
    # Queue shape: [D, K]
    queue = torch.randn(128, 4096)
    result = loss_fn(z_i, z_j, queue=queue)
    assert result.shape == (), "Expected scalar output"
    assert torch.isfinite(result), "Loss must be finite"
    assert result.item() > 0, "Loss must be positive"


# ---------------------------------------------------------------------------
# Test 4: Identical views produce lower loss than random unrelated views
# ---------------------------------------------------------------------------
def test_identical_views_lower_loss_than_random():
    loss_fn = InfoNCELoss(temperature=0.5)
    z = make_normalized(32, 128, seed=0)
    z_identical = z.clone()
    z_random = make_normalized(32, 128, seed=99)
    loss_identical = loss_fn(z, z_identical)
    loss_random = loss_fn(z, z_random)
    assert loss_identical.item() < loss_random.item(), (
        f"Identical views loss {loss_identical.item():.4f} should be < random views loss {loss_random.item():.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5: Lower temperature produces higher loss for non-perfect pairs
# ---------------------------------------------------------------------------
def test_temperature_scaling_lower_temp_higher_loss():
    z_i = make_normalized(32, 128, seed=0)
    z_j = make_normalized(32, 128, seed=1)
    loss_low_temp = InfoNCELoss(temperature=0.07)(z_i, z_j)
    loss_high_temp = InfoNCELoss(temperature=0.5)(z_i, z_j)
    assert loss_low_temp.item() > loss_high_temp.item(), (
        f"Lower temperature should produce higher loss for non-perfect pairs: "
        f"low_temp={loss_low_temp.item():.4f}, high_temp={loss_high_temp.item():.4f}"
    )


# ---------------------------------------------------------------------------
# Test 6: Batch size 1 does not crash (edge case)
# ---------------------------------------------------------------------------
def test_batch_size_1_does_not_crash():
    loss_fn = InfoNCELoss(temperature=0.5)
    z_i = make_normalized(1, 128, seed=0)
    z_j = make_normalized(1, 128, seed=1)
    result = loss_fn(z_i, z_j)
    assert torch.isfinite(result), "Loss must be finite for batch size 1"


# ---------------------------------------------------------------------------
# Test 7: Gradients flow through z_i and z_j in symmetric mode
# ---------------------------------------------------------------------------
def test_gradients_flow_both_inputs():
    loss_fn = InfoNCELoss(temperature=0.5)
    z_i = make_normalized(32, 128, seed=0).requires_grad_(True)
    z_j = make_normalized(32, 128, seed=1).requires_grad_(True)
    result = loss_fn(z_i, z_j)
    result.backward()
    assert z_i.grad is not None, "Gradient must flow through z_i"
    assert z_j.grad is not None, "Gradient must flow through z_j"
    assert torch.isfinite(z_i.grad).all(), "z_i gradients must be finite"
    assert torch.isfinite(z_j.grad).all(), "z_j gradients must be finite"
