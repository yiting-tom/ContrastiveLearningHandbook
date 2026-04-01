"""Tests for NCELossWithFixedZ -- standalone NCE loss for Instance Discrimination."""
import torch
import torch.nn.functional as F
import pytest

from methods.instance_discrimination.losses import NCELossWithFixedZ


def _normalized(shape, seed=0):
    """Create L2-normalized random tensor."""
    torch.manual_seed(seed)
    return F.normalize(torch.randn(*shape), dim=-1)


# ---------------------------------------------------------------------------
# Test 1: Forward pass produces finite positive scalar
# ---------------------------------------------------------------------------
def test_output_finite_positive():
    loss_fn = NCELossWithFixedZ(temperature=0.07, n_negatives=128)
    query = _normalized((4, 64), seed=0)
    positive = _normalized((4, 64), seed=1)
    negatives = _normalized((4, 128, 64), seed=2)
    result = loss_fn(query, positive, negatives)
    assert result.shape == (), "Expected scalar output"
    assert torch.isfinite(result), "Loss must be finite"
    assert result.item() > 0, "Loss must be positive"


# ---------------------------------------------------------------------------
# Test 2: Z is fixed after first forward call
# ---------------------------------------------------------------------------
def test_z_fixed_after_first_call():
    loss_fn = NCELossWithFixedZ(temperature=0.07, n_negatives=128)
    query1 = _normalized((4, 64), seed=0)
    pos1 = _normalized((4, 64), seed=1)
    neg1 = _normalized((4, 128, 64), seed=2)
    loss_fn(query1, pos1, neg1)
    z_after_first = loss_fn.Z.clone()

    # Second forward with completely different data
    query2 = _normalized((4, 64), seed=10)
    pos2 = _normalized((4, 64), seed=11)
    neg2 = _normalized((4, 128, 64), seed=12)
    loss_fn(query2, pos2, neg2)
    z_after_second = loss_fn.Z.clone()

    assert torch.equal(z_after_first, z_after_second), (
        f"Z changed between calls: {z_after_first.item()} -> {z_after_second.item()}"
    )


# ---------------------------------------------------------------------------
# Test 3: Z and z_initialized are register_buffers (in state_dict)
# ---------------------------------------------------------------------------
def test_z_is_register_buffer():
    loss_fn = NCELossWithFixedZ()
    keys = loss_fn.state_dict().keys()
    assert "Z" in keys, "'Z' must be a registered buffer"
    assert "z_initialized" in keys, "'z_initialized' must be a registered buffer"


# ---------------------------------------------------------------------------
# Test 4: Z survives state_dict save/load round-trip
# ---------------------------------------------------------------------------
def test_z_survives_state_dict_roundtrip():
    loss_fn = NCELossWithFixedZ(temperature=0.07, n_negatives=128)
    query = _normalized((4, 64), seed=0)
    positive = _normalized((4, 64), seed=1)
    negatives = _normalized((4, 128, 64), seed=2)
    loss_fn(query, positive, negatives)

    state = loss_fn.state_dict()
    z_original = state["Z"].clone()

    # Create fresh instance and load state
    loss_fn2 = NCELossWithFixedZ(temperature=0.07, n_negatives=128)
    loss_fn2.load_state_dict(state)

    assert torch.equal(loss_fn2.Z, z_original), "Z must survive state_dict round-trip"
    assert loss_fn2.z_initialized.item() is True, "z_initialized must be True after load"


# ---------------------------------------------------------------------------
# Test 5: Z not initialized before forward
# ---------------------------------------------------------------------------
def test_z_not_initialized_before_forward():
    loss_fn = NCELossWithFixedZ()
    assert loss_fn.z_initialized.item() is False, "z_initialized should be False before forward"
    assert loss_fn.Z.item() == -1.0, "Z should be -1.0 before forward"


# ---------------------------------------------------------------------------
# Test 6: eps attribute is stored correctly
# ---------------------------------------------------------------------------
def test_eps_in_denominator():
    loss_fn = NCELossWithFixedZ(eps=1e-7)
    assert loss_fn.eps == 1e-7, f"eps should be 1e-7, got {loss_fn.eps}"


# ---------------------------------------------------------------------------
# Test 7: Gradient flows to query
# ---------------------------------------------------------------------------
def test_gradient_flows_to_query():
    loss_fn = NCELossWithFixedZ(temperature=0.07, n_negatives=128)
    query = _normalized((4, 64), seed=0).requires_grad_(True)
    positive = _normalized((4, 64), seed=1)
    negatives = _normalized((4, 128, 64), seed=2)
    result = loss_fn(query, positive, negatives)
    result.backward()
    assert query.grad is not None, "Gradient must flow through query"
    assert torch.isfinite(query.grad).all(), "Query gradients must be finite"


# ---------------------------------------------------------------------------
# Test 8: Loss decreases for aligned pairs vs orthogonal pairs
# ---------------------------------------------------------------------------
def test_loss_decreases_for_aligned_pairs():
    loss_fn_aligned = NCELossWithFixedZ(temperature=0.07, n_negatives=128)
    loss_fn_ortho = NCELossWithFixedZ(temperature=0.07, n_negatives=128)

    # Perfectly aligned: query == positive (cosine sim = 1.0)
    query_aligned = _normalized((4, 64), seed=0)
    positive_aligned = query_aligned.clone()
    negatives = _normalized((4, 128, 64), seed=2)
    loss_aligned = loss_fn_aligned(query_aligned, positive_aligned, negatives)

    # Orthogonal: construct orthogonal positive
    query_ortho = _normalized((4, 64), seed=0)
    # Random positive (very unlikely to be aligned)
    positive_ortho = _normalized((4, 64), seed=99)
    loss_ortho = loss_fn_ortho(query_ortho, positive_ortho, negatives)

    assert loss_aligned.item() < loss_ortho.item(), (
        f"Aligned loss {loss_aligned.item():.4f} should be < orthogonal loss {loss_ortho.item():.4f}"
    )
