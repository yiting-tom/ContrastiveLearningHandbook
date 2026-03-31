"""Tests for core.backbone.build_backbone factory function.

These tests cover:
- Correct (backbone, feat_dim) return type
- Correct feat_dim for ResNet50 and ViT-Small
- No classifier head (num_classes=0)
- Forward pass output shape
- Unknown model name raises an error
"""
import pytest
import torch
import timm

from core.backbone import build_backbone


@pytest.mark.parametrize(
    "model_name, expected_feat_dim",
    [
        ("resnet50", 2048),
        ("vit_small_patch16_224", 384),
    ],
)
def test_build_backbone_feat_dim(model_name, expected_feat_dim):
    """build_backbone returns (backbone, feat_dim) with correct feat_dim."""
    backbone, feat_dim = build_backbone(model_name, pretrained=False)
    assert feat_dim == expected_feat_dim, (
        f"Expected feat_dim={expected_feat_dim}, got {feat_dim}"
    )
    assert feat_dim == backbone.num_features, (
        "feat_dim must equal backbone.num_features (never hardcoded)"
    )


def test_resnet50_no_classifier():
    """build_backbone('resnet50', pretrained=False) — no classifier head."""
    backbone, feat_dim = build_backbone("resnet50", pretrained=False)
    # With num_classes=0, timm removes the classifier head.
    # The backbone should have num_classes == 0.
    assert getattr(backbone, "num_classes", 0) == 0, (
        "Backbone should have num_classes=0 (no classifier head)"
    )


@pytest.mark.parametrize(
    "model_name, expected_feat_dim",
    [
        ("resnet50", 2048),
        ("vit_small_patch16_224", 384),
    ],
)
def test_forward_pass_output_shape(model_name, expected_feat_dim):
    """Forward pass with [2, 3, 224, 224] produces [2, feat_dim]."""
    backbone, feat_dim = build_backbone(model_name, pretrained=False)
    backbone.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = backbone(x)
    assert out.shape == (2, feat_dim), (
        f"Expected output shape (2, {feat_dim}), got {out.shape}"
    )


def test_unknown_model_name_raises():
    """Unknown model name raises a timm error (not silently returns None)."""
    with pytest.raises(Exception):
        build_backbone("this_model_does_not_exist_xyz123", pretrained=False)
