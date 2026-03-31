"""Tests for core.projection.ProjectionHead MLP module.

These tests cover:
- BN+ReLU on intermediate layers, BN-only on final layer (2-layer config)
- BN+ReLU on first N-1 layers, BN-only on final layer (3-layer config)
- Forward pass output shape for 2-layer and 3-layer configs
- use_bn=False produces no BatchNorm1d modules
- Parameter count consistency
"""
import pytest
import torch
import torch.nn as nn

from core.projection import ProjectionHead


def _get_layer_module_types(proj: ProjectionHead) -> list:
    """Extract list of module type names from the MLP sequential."""
    return [type(m).__name__ for m in proj.mlp.children()]


def test_bn_relu_pattern_2layer():
    """2-layer ProjectionHead: intermediate has BN+ReLU, final has BN only.

    Architecture for num_layers=2:
      Linear(2048 -> 2048) -> BatchNorm1d -> ReLU
      Linear(2048 -> 128)  -> BatchNorm1d
    """
    proj = ProjectionHead(2048, 2048, 128, num_layers=2)
    modules = list(proj.mlp.children())

    # First block: Linear, BN, ReLU
    assert isinstance(modules[0], nn.Linear), "Layer 0 should be Linear"
    assert isinstance(modules[1], nn.BatchNorm1d), "Layer 1 should be BatchNorm1d"
    assert isinstance(modules[2], nn.ReLU), "Layer 2 should be ReLU"

    # Final block: Linear, BN (no ReLU)
    assert isinstance(modules[3], nn.Linear), "Layer 3 should be Linear"
    assert isinstance(modules[4], nn.BatchNorm1d), "Layer 4 should be BatchNorm1d"
    assert len(modules) == 5, f"Expected 5 modules total, got {len(modules)}"


def test_bn_relu_pattern_3layer():
    """3-layer ProjectionHead: first two layers have BN+ReLU, final has BN only.

    Architecture for num_layers=3:
      Linear(2048 -> 2048) -> BatchNorm1d -> ReLU
      Linear(2048 -> 2048) -> BatchNorm1d -> ReLU
      Linear(2048 -> 128)  -> BatchNorm1d
    """
    proj = ProjectionHead(2048, 2048, 128, num_layers=3)
    modules = list(proj.mlp.children())

    # First block: Linear, BN, ReLU
    assert isinstance(modules[0], nn.Linear), "Layer 0 should be Linear"
    assert isinstance(modules[1], nn.BatchNorm1d), "Layer 1 should be BatchNorm1d"
    assert isinstance(modules[2], nn.ReLU), "Layer 2 should be ReLU"

    # Second block: Linear, BN, ReLU
    assert isinstance(modules[3], nn.Linear), "Layer 3 should be Linear"
    assert isinstance(modules[4], nn.BatchNorm1d), "Layer 4 should be BatchNorm1d"
    assert isinstance(modules[5], nn.ReLU), "Layer 5 should be ReLU"

    # Final block: Linear, BN (no ReLU)
    assert isinstance(modules[6], nn.Linear), "Layer 6 should be Linear"
    assert isinstance(modules[7], nn.BatchNorm1d), "Layer 7 should be BatchNorm1d"
    assert len(modules) == 8, f"Expected 8 modules total, got {len(modules)}"


def test_forward_shape_2layer():
    """Forward pass [4, 2048] -> [4, 128] for num_layers=2."""
    proj = ProjectionHead(2048, 2048, 128, num_layers=2)
    proj.eval()
    x = torch.randn(4, 2048)
    with torch.no_grad():
        out = proj(x)
    assert out.shape == (4, 128), f"Expected (4, 128), got {out.shape}"


def test_forward_shape_3layer():
    """Forward pass [4, 2048] -> [4, 256] for num_layers=3, output_dim=256."""
    proj = ProjectionHead(2048, 2048, 256, num_layers=3)
    proj.eval()
    x = torch.randn(4, 2048)
    with torch.no_grad():
        out = proj(x)
    assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"


def test_no_bn_when_use_bn_false():
    """ProjectionHead(768, 768, 128, num_layers=2, use_bn=False) has no BatchNorm1d."""
    proj = ProjectionHead(768, 768, 128, num_layers=2, use_bn=False)
    bn_modules = [m for m in proj.mlp.modules() if isinstance(m, nn.BatchNorm1d)]
    assert len(bn_modules) == 0, (
        f"Expected no BatchNorm1d modules when use_bn=False, found {len(bn_modules)}"
    )


def test_parameter_count_consistent():
    """Parameter count is consistent with architecture spec (no unexpected extras)."""
    proj_2 = ProjectionHead(2048, 2048, 128, num_layers=2)
    proj_3 = ProjectionHead(2048, 2048, 128, num_layers=3)

    # Count all parameter tensors
    params_2 = sum(1 for _ in proj_2.parameters())
    params_3 = sum(1 for _ in proj_3.parameters())

    # 2-layer with BN: each layer has weight + bias for Linear, weight + bias for BN
    # Linear has weight + bias, BN has weight + bias
    # 2 layers * 4 params = 8 params
    # But BN running stats are buffers not params (running_mean, running_var)
    # Linear weight + bias = 2 per layer, BN weight + bias = 2 per layer
    # 2 layers * (2 + 2) = 8
    assert params_2 == 8, f"2-layer ProjectionHead should have 8 parameter tensors, got {params_2}"
    # 3-layer: 3 layers * (2 + 2) = 12
    assert params_3 == 12, f"3-layer ProjectionHead should have 12 parameter tensors, got {params_3}"
