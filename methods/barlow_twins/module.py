"""Barlow Twins (Zbontar et al., ICML 2021).

No-negative self-supervised learning via redundancy reduction in the cross-
correlation matrix of embedding pairs. The loss drives the cross-correlation
matrix toward the identity: diagonal entries toward 1 (invariance) and
off-diagonal entries toward 0 (redundancy reduction).

Paper: "Barlow Twins: Self-Supervised Learning via Redundancy Reduction"
Authors: Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, Stephane Deny
Venue: ICML 2021
arXiv: https://arxiv.org/abs/2103.03230
Reference implementation: https://github.com/facebookresearch/barlowtwins

Gotchas:
- The projector must be high-dimensional (8192) — low-dim projectors (128, 256)
  produce significantly worse results. The cross-correlation matrix must be
  large enough to capture redundancy across many dimensions.
- L2-normalize the projector outputs BEFORE computing the cross-correlation
  matrix. Without normalization, the matrix entries are not bounded and the
  loss can diverge.
- Normalize C by batch size (divide by B), NOT by the number of dimensions D.
  This keeps the magnitude of C consistent across batch sizes.
- lambda_coeff controls the trade-off between invariance (diagonal) and
  redundancy reduction (off-diagonal). Too large -> diagonal sacrificed; too
  small -> off-diagonal not reduced. 5e-3 is the default from the paper.
- There is no EMA, no predictor, no queue. The entire mechanism is the loss
  function acting on the cross-correlation matrix.
- The diagonal mean of C serves as a collapse indicator: collapse occurs when
  diagonal entries drop below 0.5, meaning the two views' representations are
  no longer correlated.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.backbone import build_backbone
from core.base import BaseSSLModule
from core.config import BarlowTwinsConfig, TrainConfig
from core.projection import ProjectionHead


class BarlowTwinsModule(BaseSSLModule):
    """Barlow Twins (Zbontar et al., ICML 2021).

    Barlow Twins: Self-Supervised Learning via Redundancy Reduction.

    Two augmented views are encoded by a shared backbone and a high-dimensional
    projector (8192-dim output). The cross-correlation matrix C between the
    projected representations of the two batches (normalized by batch size) is
    driven toward the identity: diagonal terms invariance ((1-C_ii)^2) and
    off-diagonal terms redundancy (lambda * C_ij^2). No negatives, no EMA, no
    predictor — fully symmetric.

    Paper: "Barlow Twins: Self-Supervised Learning via Redundancy Reduction"
    Authors: Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, Stephane Deny
    Venue: ICML 2021
    arXiv: https://arxiv.org/abs/2103.03230

    Algorithm:
    1. Augment x into v1, v2; encode both with shared backbone.
    2. Project to 8192-dim via 3-layer MLP (BN+ReLU on layers 1-2, BN-only on 3).
    3. Standardize (zero mean, unit variance) projector outputs across batch.
    4. Compute cross-correlation C = Z1.T @ Z2 / N.
    5. Loss = sum_i (1 - C_ii)^2 + lambda * sum_{i!=j} C_ij^2 (lambda=5e-3).

    Gotchas:
    - C must be normalized by batch size (divide by N) before computing the loss.
      Without this, the loss scale depends on batch size in a non-obvious way.
    - Projector output must be standardized (zero mean, unit variance per
      dimension across the batch) before computing C. BN at the projector output
      handles standardization but it must happen at loss-computation time too.
    - lambda is sensitive — values much above 5e-3 push C off-diagonals to zero
      but starve the diagonal; values much below leave redundancy uncontrolled.
    - Do not reduce projector output dim below 2048 without empirical validation.
    - Mandatory training diagnostic: log z.std(dim=0).mean(); collapse signal.

    Reference implementation: https://github.com/facebookresearch/barlowtwins
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        bt_cfg: BarlowTwinsConfig = cfg.barlow_twins or BarlowTwinsConfig()
        self._lambda_coeff = bt_cfg.lambda_coeff
        self._proj_dim = bt_cfg.projection_dim

        self.backbone, self.feat_dim = build_backbone(cfg.backbone, cfg.pretrained)
        self.projector = self.build_projector()

    def build_projector(self) -> nn.Module:
        """3-layer high-dimensional projection head (feat_dim->8192->8192->8192)."""
        return ProjectionHead(
            input_dim=self.feat_dim,
            hidden_dim=self._proj_dim,
            output_dim=self._proj_dim,
            num_layers=3,
        )

    def training_step(self, batch, batch_idx):
        (v1, v2), _ = batch

        # Shared encoder
        z_a = F.normalize(self.projector(self.backbone(v1)), dim=1)  # [B, D]
        z_b = F.normalize(self.projector(self.backbone(v2)), dim=1)  # [B, D]

        # Cross-correlation matrix: C[i,j] = corr(z_a[:, i], z_b[:, j])
        B = z_a.size(0)
        C = z_a.T @ z_b / B  # [D, D]

        # Barlow Twins loss
        # Invariance: diagonal entries -> 1
        on_diag = torch.diagonal(C).add_(-1).pow_(2).sum()
        # Redundancy reduction: off-diagonal entries -> 0
        off_diag = self._off_diagonal(C).pow_(2).sum()
        loss = on_diag + self._lambda_coeff * off_diag

        # Collapse monitoring — diagonal mean approaches 0 on collapse
        with torch.no_grad():
            diag_mean = torch.diagonal(C).mean()
            self.log("train/corr_diag_mean", diag_mean, on_step=True, on_epoch=False)

        self.log_train_metrics(loss)
        return loss

    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        """Return all off-diagonal elements of a square matrix as a 1D tensor."""
        n = x.size(0)
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
