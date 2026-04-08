"""Unit tests for PrototypeLayer normalization and freezing behavior.

Tests cover:
- Construction creates correct nn.Linear layer
- Weight rows are L2-normalized after construction
- normalize_prototypes() restores L2-norm after arbitrary modification
- should_freeze_prototypes() returns correct bool based on epoch and freeze threshold
- zero_prototype_gradients() zeros the weight gradient
- Prototype weight has requires_grad=True
- normalize_prototypes() is idempotent
- forward() produces correct output shape
"""
import torch
import pytest

from methods.swav.prototype import PrototypeLayer


FEAT_DIM = 128
N_PROTOTYPES = 100


# -------------------------------------------------------------------------
# Fixture
# -------------------------------------------------------------------------

@pytest.fixture
def layer():
    """A freshly-constructed PrototypeLayer."""
    return PrototypeLayer(feat_dim=FEAT_DIM, n_prototypes=N_PROTOTYPES)


# -------------------------------------------------------------------------
# Test 1: Constructor creates correct nn.Linear
# -------------------------------------------------------------------------

def test_prototype_layer_creates_linear(layer):
    """PrototypeLayer(128, 100) creates nn.Linear(128, 100, bias=False)."""
    import torch.nn as nn
    assert hasattr(layer, "linear"), "PrototypeLayer must have a 'linear' attribute"
    assert isinstance(layer.linear, nn.Linear)
    assert layer.linear.in_features == FEAT_DIM
    assert layer.linear.out_features == N_PROTOTYPES
    assert layer.linear.bias is None, "Linear must have bias=False"


# -------------------------------------------------------------------------
# Test 2: Rows are L2-normalized after construction
# -------------------------------------------------------------------------

def test_prototype_rows_l2_normalized_after_construction(layer):
    """After construction, each prototype row has L2-norm ≈ 1.0."""
    norms = layer.linear.weight.data.norm(dim=1, p=2)
    assert torch.allclose(norms, torch.ones(N_PROTOTYPES), atol=1e-5), (
        f"Expected all row norms ≈ 1.0, got min={norms.min():.6f} max={norms.max():.6f}"
    )


# -------------------------------------------------------------------------
# Test 3: normalize_prototypes() restores L2-norm after arbitrary modification
# -------------------------------------------------------------------------

def test_normalize_prototypes_restores_unit_norm(layer):
    """normalize_prototypes() restores unit-norm rows after weight modification."""
    # Corrupt weights
    with torch.no_grad():
        layer.linear.weight.mul_(3.7)

    # Verify they are NOT unit-norm before normalization
    norms_before = layer.linear.weight.data.norm(dim=1, p=2)
    assert not torch.allclose(norms_before, torch.ones(N_PROTOTYPES), atol=1e-4), (
        "Corruption should break unit norms"
    )

    layer.normalize_prototypes()

    norms_after = layer.linear.weight.data.norm(dim=1, p=2)
    assert torch.allclose(norms_after, torch.ones(N_PROTOTYPES), atol=1e-5), (
        f"normalize_prototypes should restore unit norms, "
        f"got min={norms_after.min():.6f} max={norms_after.max():.6f}"
    )


# -------------------------------------------------------------------------
# Test 4: should_freeze_prototypes epoch logic
# -------------------------------------------------------------------------

def test_should_freeze_prototypes_true_when_epoch_less_than_freeze():
    """should_freeze_prototypes(current_epoch=0, freeze_epochs=1) returns True."""
    assert PrototypeLayer.should_freeze_prototypes(current_epoch=0, freeze_epochs=1) is True


def test_should_freeze_prototypes_false_when_epoch_equals_freeze():
    """should_freeze_prototypes(current_epoch=1, freeze_epochs=1) returns False."""
    assert PrototypeLayer.should_freeze_prototypes(current_epoch=1, freeze_epochs=1) is False


def test_should_freeze_prototypes_false_when_epoch_greater_than_freeze():
    """should_freeze_prototypes(current_epoch=5, freeze_epochs=1) returns False."""
    assert PrototypeLayer.should_freeze_prototypes(current_epoch=5, freeze_epochs=1) is False


def test_should_freeze_prototypes_multiple_epochs():
    """should_freeze_prototypes works correctly across a range of epochs."""
    freeze_epochs = 3
    for epoch in range(freeze_epochs):
        assert PrototypeLayer.should_freeze_prototypes(epoch, freeze_epochs) is True
    for epoch in range(freeze_epochs, freeze_epochs + 5):
        assert PrototypeLayer.should_freeze_prototypes(epoch, freeze_epochs) is False


# -------------------------------------------------------------------------
# Test 5: zero_prototype_gradients() zeros the weight gradient
# -------------------------------------------------------------------------

def test_zero_prototype_gradients_zeros_grad(layer):
    """zero_prototype_gradients() zeros the linear weight gradient."""
    # Create a fake gradient
    layer.linear.weight.grad = torch.ones(N_PROTOTYPES, FEAT_DIM)

    layer.zero_prototype_gradients()

    assert layer.linear.weight.grad is not None, "Gradient tensor should still exist"
    assert torch.all(layer.linear.weight.grad == 0), "Gradient should be all zeros"


def test_zero_prototype_gradients_no_error_when_grad_none(layer):
    """zero_prototype_gradients() does not raise when grad is None."""
    assert layer.linear.weight.grad is None
    layer.zero_prototype_gradients()  # Should not raise


# -------------------------------------------------------------------------
# Test 6: Prototype weight has requires_grad=True
# -------------------------------------------------------------------------

def test_prototype_weight_is_trainable(layer):
    """Prototype weight has requires_grad=True (trainable, included in optimizer)."""
    assert layer.linear.weight.requires_grad is True, (
        "Prototype weight must be trainable (requires_grad=True)"
    )


# -------------------------------------------------------------------------
# Extra: normalize_prototypes is idempotent
# -------------------------------------------------------------------------

def test_normalize_prototypes_is_idempotent(layer):
    """Calling normalize_prototypes() twice gives the same result as calling it once."""
    layer.normalize_prototypes()
    weights_after_first = layer.linear.weight.data.clone()

    layer.normalize_prototypes()
    weights_after_second = layer.linear.weight.data.clone()

    assert torch.allclose(weights_after_first, weights_after_second, atol=1e-7), (
        "normalize_prototypes should be idempotent"
    )


# -------------------------------------------------------------------------
# Extra: forward produces correct output shape
# -------------------------------------------------------------------------

def test_forward_output_shape(layer):
    """forward([B, feat_dim]) produces [B, n_prototypes] output."""
    batch_size = 16
    x = torch.randn(batch_size, FEAT_DIM)
    out = layer(x)
    assert out.shape == (batch_size, N_PROTOTYPES), (
        f"Expected output shape {(batch_size, N_PROTOTYPES)}, got {out.shape}"
    )


def test_forward_output_is_prototype_scores(layer):
    """forward output values are dot products (prototype scores)."""
    x = torch.randn(4, FEAT_DIM)
    out = layer(x)
    # Manually compute expected: x @ W.T (since bias=False)
    expected = x @ layer.linear.weight.t()
    assert torch.allclose(out, expected, atol=1e-5), (
        "forward should return x @ prototypes.T"
    )
