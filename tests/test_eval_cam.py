"""Unit tests for eval/cam_vis.py — CAM visualization script.

Tests verify:
 - get_target_layer returns correct layer for ResNet and ViT architectures
 - get_target_layer raises ValueError for unsupported architectures
 - vit_reshape_transform converts [B, N+1, D] -> [B, D, H, W]
 - Default CAM method is EigenCAM
 - GradCAM is used when method='gradcam'
 - WrapperModule forward returns logits with correct shape
 - Script saves n_images overlay PNGs
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

# Import from module under test — will fail until implemented (TDD RED)
from eval.cam_vis import (
    get_target_layer,
    vit_reshape_transform,
    get_cam_method,
    WrapperModule,
)
from core.config import CAMConfig


# ---------------------------------------------------------------------------
# Minimal fake backbones for architecture tests
# ---------------------------------------------------------------------------

class FakeResNetBlock(nn.Module):
    """Minimal block to simulate ResNet layer4."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 1)

    def forward(self, x):
        return self.conv(x)


class FakeResNetBackbone(nn.Module):
    """Minimal backbone simulating ResNet structure with layer4."""
    def __init__(self):
        super().__init__()
        self.layer4 = nn.ModuleList([FakeResNetBlock(), FakeResNetBlock()])
        self.num_features = 64

    def forward(self, x):
        return x.mean(dim=[2, 3])


class FakeViTBlock(nn.Module):
    """Minimal ViT-style block with norm1."""
    def __init__(self, dim: int = 32):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc(self.norm1(x))


class FakeViTBackbone(nn.Module):
    """Minimal backbone simulating ViT structure with blocks."""
    def __init__(self, num_blocks: int = 2, dim: int = 32):
        super().__init__()
        self.blocks = nn.ModuleList([FakeViTBlock(dim) for _ in range(num_blocks)])
        self.num_features = dim

    def forward(self, x):
        return x.mean(dim=1)


class FakeUnsupportedBackbone(nn.Module):
    """Backbone with no recognized architecture."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)
        self.num_features = 16

    def forward(self, x):
        return x


# ---------------------------------------------------------------------------
# Test 1: get_target_layer for ResNet
# ---------------------------------------------------------------------------

class TestGetTargetLayer:
    def test_resnet_returns_layer4_last_block(self):
        """get_target_layer returns [backbone.layer4[-1]] for resnet18."""
        backbone = FakeResNetBackbone()
        target = get_target_layer(backbone, "resnet18")
        assert isinstance(target, list), "Should return a list of layers"
        assert len(target) == 1, f"Should return 1 target layer, got {len(target)}"
        assert target[0] is backbone.layer4[-1], (
            "Target layer should be backbone.layer4[-1]"
        )

    def test_resnet50_returns_layer4_last_block(self):
        """get_target_layer works for resnet50 (case-insensitive 'resnet' check)."""
        backbone = FakeResNetBackbone()
        target = get_target_layer(backbone, "resnet50")
        assert target[0] is backbone.layer4[-1]

    def test_vit_returns_blocks_last_norm1(self):
        """get_target_layer returns [backbone.blocks[-1].norm1] for vit_small_patch16_224."""
        backbone = FakeViTBackbone(num_blocks=3)
        target = get_target_layer(backbone, "vit_small_patch16_224")
        assert isinstance(target, list)
        assert len(target) == 1
        assert target[0] is backbone.blocks[-1].norm1, (
            "Target layer should be backbone.blocks[-1].norm1"
        )

    def test_unsupported_backbone_raises_value_error(self):
        """get_target_layer raises ValueError for unsupported backbone names."""
        backbone = FakeUnsupportedBackbone()
        with pytest.raises(ValueError, match="Unsupported backbone"):
            get_target_layer(backbone, "efficientnet_b0")


# ---------------------------------------------------------------------------
# Test 4: vit_reshape_transform
# ---------------------------------------------------------------------------

class TestViTReshapeTransform:
    def test_removes_cls_token_and_reshapes(self):
        """vit_reshape_transform converts [B, N+1, D] -> [B, D, H, W]."""
        B, H, W, D = 2, 14, 14, 32
        N_patches = H * W
        # Input: [B, N_patches + 1, D] (includes CLS token)
        tensor = torch.randn(B, N_patches + 1, D)
        result = vit_reshape_transform(tensor, height=H, width=W)
        assert result.shape == (B, D, H, W), (
            f"Expected shape ({B}, {D}, {H}, {W}), got {result.shape}"
        )

    def test_cls_token_removed(self):
        """vit_reshape_transform removes exactly the first token (CLS)."""
        B, H, W, D = 1, 4, 4, 8
        N_patches = H * W
        tensor = torch.randn(B, N_patches + 1, D)
        # Set CLS token to a distinctive value
        tensor[:, 0, :] = 999.0
        result = vit_reshape_transform(tensor, height=H, width=W)
        # Result should not contain 999.0 (CLS was discarded)
        assert not (result == 999.0).any(), "CLS token value should not appear in output"

    def test_output_is_bchw(self):
        """Result is in BCHW format (channels-first)."""
        B, H, W, D = 2, 7, 7, 16
        tensor = torch.randn(B, H * W + 1, D)
        result = vit_reshape_transform(tensor, height=H, width=W)
        # Should be [B, D, H, W] where D=channels
        assert result.shape[1] == D, f"Channel dim should be {D}, got {result.shape[1]}"
        assert result.shape[2] == H
        assert result.shape[3] == W


# ---------------------------------------------------------------------------
# Test 5 & 6: get_cam_method
# ---------------------------------------------------------------------------

class TestGetCamMethod:
    def test_default_eigencam(self):
        """get_cam_method returns EigenCAM for 'eigencam'."""
        from pytorch_grad_cam import EigenCAM
        CamClass = get_cam_method("eigencam")
        assert CamClass is EigenCAM, f"Expected EigenCAM, got {CamClass}"

    def test_gradcam(self):
        """get_cam_method returns GradCAM for 'gradcam'."""
        from pytorch_grad_cam import GradCAM
        CamClass = get_cam_method("gradcam")
        assert CamClass is GradCAM, f"Expected GradCAM, got {CamClass}"

    def test_unknown_method_raises(self):
        """get_cam_method raises ValueError for unknown CAM method."""
        with pytest.raises(ValueError):
            get_cam_method("scorecam_custom")


# ---------------------------------------------------------------------------
# Test 8: WrapperModule
# ---------------------------------------------------------------------------

class TestWrapperModule:
    def test_wrapper_forward_returns_logits(self):
        """WrapperModule(backbone, head) forward returns logits with correct shape."""
        feat_dim = 32
        num_classes = 5
        batch_size = 4

        # Simple backbone that outputs feat_dim features
        backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, feat_dim),
        )
        backbone.num_features = feat_dim

        head = nn.Linear(feat_dim, num_classes)
        wrapper = WrapperModule(backbone, head)

        x = torch.randn(batch_size, 3, 8, 8)
        logits = wrapper(x)

        assert logits.shape == (batch_size, num_classes), (
            f"Expected logits shape ({batch_size}, {num_classes}), got {logits.shape}"
        )

    def test_wrapper_stores_backbone_and_head(self):
        """WrapperModule stores backbone and head as submodules."""
        backbone = nn.Linear(16, 8)
        head = nn.Linear(8, 3)
        wrapper = WrapperModule(backbone, head)
        assert wrapper.backbone is backbone
        assert wrapper.head is head

    def test_wrapper_is_nn_module(self):
        """WrapperModule is an nn.Module subclass."""
        backbone = nn.Linear(16, 8)
        head = nn.Linear(8, 3)
        wrapper = WrapperModule(backbone, head)
        assert isinstance(wrapper, nn.Module)


# ---------------------------------------------------------------------------
# Test 7: run_cam saves n_images PNGs (integration-style test)
# ---------------------------------------------------------------------------

class TestRunCAM:
    def test_run_cam_saves_pngs(self, tmp_path):
        """run_cam saves n_images overlay PNG files."""
        from eval.cam_vis import run_cam
        from core.config import CAMConfig

        # Build a tiny resnet-like model
        feat_dim = 16
        n_classes = 3
        n_images = 3

        backbone = FakeResNetBackbone()

        # Patch run_cam to avoid actual grad-cam computation (test just file saving)
        # We create minimal numpy images
        images = [
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            for _ in range(n_images)
        ]

        cam_cfg = CAMConfig(method="eigencam", n_images=n_images)

        # Create a minimal model-like object for run_cam
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Wrap backbone so run_cam can access model.backbone
                self.backbone = FakeResNetBackbone()
                self.backbone.layer4 = nn.ModuleList([
                    nn.Sequential(nn.Conv2d(4, 4, 1))
                ])

            def forward(self, x):
                return self.backbone(x)

        # Test with EigenCAM on a real (tiny) resnet backbone
        backbone, feat_dim = None, None
        try:
            import timm
            backbone = timm.create_model("resnet18", pretrained=False, num_classes=0)
            backbone_name = "resnet18"
        except Exception:
            pytest.skip("timm not available for this test")

        class SimpleModel(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone

        model = SimpleModel(backbone)
        model.eval()

        output_dir = tmp_path / "cam_output"
        saved_paths = run_cam(
            model=model,
            backbone_name=backbone_name,
            images=images,
            cam_cfg=cam_cfg,
            output_dir=output_dir,
        )

        # Check correct number of files saved
        assert len(saved_paths) == n_images, (
            f"Expected {n_images} saved PNGs, got {len(saved_paths)}"
        )
        for p in saved_paths:
            assert Path(p).exists(), f"Expected PNG file at {p}"
            assert str(p).endswith(".png"), f"Expected .png file, got {p}"
