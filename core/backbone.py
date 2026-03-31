"""Backbone factory using timm.

Provides a single factory function, build_backbone(), that wraps
timm.create_model() with global pooling enabled and no classifier head.
Feature dimension is always read from backbone.num_features — never hardcoded.
"""
import timm


def build_backbone(model_name: str, pretrained: bool = False) -> tuple:
    """Build a backbone using timm with global pooling, no classifier head.

    Args:
        model_name: Any valid timm model name (e.g., 'resnet50', 'vit_small_patch16_224').
        pretrained: Whether to load pretrained weights.

    Returns:
        Tuple of (backbone, feat_dim) where feat_dim = backbone.num_features.
        Never hard-code feature dimensions — always use backbone.num_features.

    Raises:
        RuntimeError: If model_name is not a valid timm model.
    """
    backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    feat_dim = backbone.num_features
    return backbone, feat_dim
