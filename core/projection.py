"""Reusable MLP projection and predictor heads for SSL methods.

Provides:
- ProjectionHead: configurable MLP used by SimCLR, MoCo, BYOL, Barlow Twins,
  and other contrastive learning methods. Uses BN+ReLU on intermediate layers
  and BN-only (no ReLU) on the final layer, following the standard SSL convention.
- PredictorHead: predictor MLP for BYOL and SimSiam online branch. Supports
  'standard' (BYOL) and 'bottleneck' (SimSiam) variants.
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


class PredictorHead(nn.Module):
    """Predictor MLP for BYOL and SimSiam online branch.

    Sits on top of the projector on the online (student) branch only.
    The target (teacher/momentum) branch never has a predictor.

    Two variants:

    - ``'standard'``: 2-layer MLP used by BYOL (Grill et al., 2020).
      Architecture: Linear -> BN -> ReLU -> Linear -> BN (no ReLU on output).

    - ``'bottleneck'``: 2-layer bottleneck MLP used by SimSiam (Chen & He, 2021).
      Architecture: Linear -> BN -> ReLU -> Linear -> BN (no ReLU on output).
      Dims: input_dim -> bottleneck_dim -> input_dim (default 2048->512->2048).
      BN is applied on ALL layers including the output layer.

    Args:
        predictor_type: ``'standard'`` (BYOL) or ``'bottleneck'`` (SimSiam).
        input_dim: Input dimension. For ``'bottleneck'``, this is also the output dim.
        hidden_dim: Hidden layer width. Used only for ``'standard'`` variant.
        output_dim: Output dimension. Used only for ``'standard'`` variant.
        bottleneck_dim: Bottleneck hidden dimension. Used only for ``'bottleneck'``
            variant. Default: 512 (matching SimSiam paper: 2048->512->2048).

    Raises:
        ValueError: If ``predictor_type`` is not ``'standard'`` or ``'bottleneck'``.
    """

    def __init__(
        self,
        predictor_type: str = "standard",
        input_dim: int = 256,
        hidden_dim: int = 4096,
        output_dim: int = 256,
        bottleneck_dim: int = 512,
    ):
        super().__init__()
        if predictor_type == "standard":
            # BYOL predictor: Linear -> BN -> ReLU -> Linear -> BN
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
                nn.BatchNorm1d(output_dim),
            )
        elif predictor_type == "bottleneck":
            # SimSiam predictor: 2048->512->2048, BN on all layers, no ReLU on output
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU(inplace=True),
                nn.Linear(bottleneck_dim, input_dim),
                nn.BatchNorm1d(input_dim),
            )
        else:
            raise ValueError(
                f"Unknown predictor_type '{predictor_type}'. "
                f"Choose 'standard' (BYOL) or 'bottleneck' (SimSiam)."
            )

    def forward(self, x):
        return self.mlp(x)
