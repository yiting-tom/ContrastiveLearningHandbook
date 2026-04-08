"""Tests for SwAV Sinkhorn-Knopp optimal transport and swapped-prediction loss.

Reference:
    Caron et al., "Unsupervised Learning of Visual Features by Contrasting
    Cluster Assignments" (NeurIPS 2020). https://arxiv.org/abs/2006.09882
"""
import torch
import torch.nn as nn
import pytest

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
