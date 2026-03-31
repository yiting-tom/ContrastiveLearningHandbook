"""Reusable MLP projection head for SSL methods.

Provides ProjectionHead, a configurable MLP used by SimCLR, MoCo, BYOL,
Barlow Twins, and other contrastive learning methods. The architecture uses
BN+ReLU on intermediate layers and BN-only (no ReLU) on the final layer,
following the standard SSL literature convention.
"""
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Reusable MLP projection head for SSL methods.

    Architecture: Linear -> [BN -> ReLU] on intermediate layers,
                  Linear -> [BN] on final layer (no ReLU on output).

    Used by SimCLR (2-layer), SimCLR v2 (3-layer), MoCo v2 (2-layer),
    BYOL (2-layer), Barlow Twins (3-layer), etc.

    Args:
        input_dim: Backbone feature dimension (from build_backbone).
        hidden_dim: Hidden layer width.
        output_dim: Output embedding dimension.
        num_layers: Number of linear layers (minimum 2).
        use_bn: Whether to include BatchNorm1d layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        use_bn: bool = True,
    ):
        super().__init__()
        assert num_layers >= 2, "ProjectionHead requires at least 2 layers"

        layers = []
        in_dim = input_dim

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            out_dim = output_dim if is_last else hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))

            if use_bn:
                layers.append(nn.BatchNorm1d(out_dim))

            if not is_last:
                layers.append(nn.ReLU(inplace=True))

            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
