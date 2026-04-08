"""Tests for core.projection.PredictorHead MLP module.

Covers both predictor variants used in no-negative SSL methods:
- 'standard': 2-layer MLP, BN+ReLU on first layer, BN-only on output (BYOL)
- 'bottleneck': 2048->512->2048 bottleneck MLP, BN on ALL layers, no ReLU on output (SimSiam)
"""
import pytest
import torch
import torch.nn as nn

from core.projection import PredictorHead


# ---------------------------------------------------------------------------
# Test 1: Standard variant forward shape
# ---------------------------------------------------------------------------

def test_standard_forward_shape():
    """PredictorHead('standard', 128, 4096, 128) forward produces shape [B, 128]."""
    pred = PredictorHead(predictor_type="standard", input_dim=128, hidden_dim=4096, output_dim=128)
    pred.eval()
    x = torch.randn(4, 128)
    with torch.no_grad():
        out = pred(x)
    assert out.shape == (4, 128), f"Expected (4, 128), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 2: Standard variant layer architecture
# ---------------------------------------------------------------------------

def test_standard_layer_architecture():
    """Standard variant has exactly 2 Linear layers, BN on both, ReLU only on first."""
    pred = PredictorHead(predictor_type="standard", input_dim=128, hidden_dim=4096, output_dim=128)
    modules = list(pred.mlp.children())

    # First block: Linear -> BN -> ReLU
    assert isinstance(modules[0], nn.Linear), f"modules[0] should be Linear, got {type(modules[0])}"
    assert isinstance(modules[1], nn.BatchNorm1d), f"modules[1] should be BatchNorm1d, got {type(modules[1])}"
    assert isinstance(modules[2], nn.ReLU), f"modules[2] should be ReLU, got {type(modules[2])}"

    # Final block: Linear -> BN (no ReLU)
    assert isinstance(modules[3], nn.Linear), f"modules[3] should be Linear, got {type(modules[3])}"
    assert isinstance(modules[4], nn.BatchNorm1d), f"modules[4] should be BatchNorm1d, got {type(modules[4])}"

    assert len(modules) == 5, f"Standard variant should have 5 modules total, got {len(modules)}"

    # Verify no ReLU on output (only 1 ReLU total, on first layer)
    relu_modules = [m for m in pred.mlp.children() if isinstance(m, nn.ReLU)]
    assert len(relu_modules) == 1, f"Standard variant should have exactly 1 ReLU (first layer only), got {len(relu_modules)}"


# ---------------------------------------------------------------------------
# Test 3: Bottleneck variant forward shape
# ---------------------------------------------------------------------------

def test_bottleneck_forward_shape():
    """PredictorHead('bottleneck', 2048, output_dim=2048) forward produces shape [B, 2048]."""
    pred = PredictorHead(predictor_type="bottleneck", input_dim=2048, output_dim=2048)
    pred.eval()
    x = torch.randn(4, 2048)
    with torch.no_grad():
        out = pred(x)
    assert out.shape == (4, 2048), f"Expected (4, 2048), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 4: Bottleneck variant architecture — BN on ALL layers, no ReLU on output
# ---------------------------------------------------------------------------

def test_bottleneck_layer_architecture():
    """Bottleneck variant: 2048->512->2048, BN on ALL layers including output, no ReLU on output."""
    pred = PredictorHead(predictor_type="bottleneck", input_dim=2048, output_dim=2048, bottleneck_dim=512)
    modules = list(pred.mlp.children())

    # First block: Linear(2048->512) -> BN -> ReLU
    assert isinstance(modules[0], nn.Linear), f"modules[0] should be Linear, got {type(modules[0])}"
    assert modules[0].in_features == 2048, f"First linear in_features should be 2048, got {modules[0].in_features}"
    assert modules[0].out_features == 512, f"First linear out_features should be 512, got {modules[0].out_features}"
    assert isinstance(modules[1], nn.BatchNorm1d), f"modules[1] should be BatchNorm1d, got {type(modules[1])}"
    assert isinstance(modules[2], nn.ReLU), f"modules[2] should be ReLU, got {type(modules[2])}"

    # Final block: Linear(512->2048) -> BN (no ReLU)
    assert isinstance(modules[3], nn.Linear), f"modules[3] should be Linear, got {type(modules[3])}"
    assert modules[3].in_features == 512, f"Second linear in_features should be 512, got {modules[3].in_features}"
    assert modules[3].out_features == 2048, f"Second linear out_features should be 2048, got {modules[3].out_features}"
    assert isinstance(modules[4], nn.BatchNorm1d), f"modules[4] should be BatchNorm1d, got {type(modules[4])}"

    assert len(modules) == 5, f"Bottleneck variant should have 5 modules total, got {len(modules)}"

    # BN on all layers: 2 BatchNorm1d modules total
    bn_modules = [m for m in pred.mlp.children() if isinstance(m, nn.BatchNorm1d)]
    assert len(bn_modules) == 2, f"Bottleneck variant should have 2 BatchNorm1d (including output), got {len(bn_modules)}"

    # No ReLU on output: exactly 1 ReLU (on first block only)
    relu_modules = [m for m in pred.mlp.children() if isinstance(m, nn.ReLU)]
    assert len(relu_modules) == 1, f"Bottleneck variant should have 1 ReLU (no ReLU on output), got {len(relu_modules)}"


# ---------------------------------------------------------------------------
# Test 5: Bottleneck bottleneck_dim defaults to 512
# ---------------------------------------------------------------------------

def test_bottleneck_dim_default():
    """Bottleneck bottleneck_dim defaults to 512 (SimSiam: 2048->512->2048)."""
    pred = PredictorHead(predictor_type="bottleneck", input_dim=2048, output_dim=2048)
    modules = list(pred.mlp.children())

    # First linear: 2048 -> 512
    first_linear = modules[0]
    assert isinstance(first_linear, nn.Linear)
    assert first_linear.out_features == 512, (
        f"Default bottleneck_dim should be 512, got {first_linear.out_features}"
    )


# ---------------------------------------------------------------------------
# Test 6: PredictorHead is importable from core.projection
# ---------------------------------------------------------------------------

def test_importable_from_core_projection():
    """PredictorHead is importable from core.projection."""
    # This test passes if the import at the top of the file succeeded.
    # Verify it's the class we expect.
    assert PredictorHead is not None
    assert hasattr(PredictorHead, "__init__")
    # Instantiate to confirm it's the right class
    pred = PredictorHead(predictor_type="standard", input_dim=64, hidden_dim=256, output_dim=64)
    assert isinstance(pred, nn.Module)


# ---------------------------------------------------------------------------
# Test 7: Invalid predictor_type raises ValueError
# ---------------------------------------------------------------------------

def test_invalid_predictor_type_raises_value_error():
    """Invalid predictor_type raises ValueError with helpful message."""
    with pytest.raises(ValueError, match="predictor_type"):
        PredictorHead(predictor_type="unknown_type", input_dim=128, output_dim=128)
