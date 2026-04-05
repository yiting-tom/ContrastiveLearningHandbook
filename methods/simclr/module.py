"""SimCLR v1 and v2 — contrastive learning with in-batch negatives.

SimCLR v1 uses a 2-layer projection head and symmetric NT-Xent loss (InfoNCE
in symmetric mode with no queue).  SimCLR v2 is identical except for a 3-layer
projection head that improves representation quality at the cost of more
parameters.

Paper (v1): "A Simple Framework for Contrastive Learning of Visual Representations"
Authors: Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
Venue: ICML 2020
arXiv: https://arxiv.org/abs/2002.05709

Paper (v2): "Big Self-Supervised Models are Strong Semi-Supervised Learners"
Authors: Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, Geoffrey Hinton
Venue: NeurIPS 2020
arXiv: https://arxiv.org/abs/2006.10029

Key properties:
- Uses InfoNCELoss in symmetric mode (NT-Xent): loss on z, eval on h.
- Strong augmentation (color jitter s=1.0) is critical for performance.
- LARS optimizer recommended for large batch sizes (>= 256).
- Batch size sensitivity: below ~256, representation quality degrades because
  all negatives come from the current batch (no external queue or memory bank).

Gotchas:
- Batch size sensitivity below 256 — fewer in-batch negatives hurts quality.
- Color jitter strength s=1.0 is NOT the torchvision default (s=0.4).
- Temperature is sensitive: 0.5 works for batch_size=256, tune for other sizes.
"""
from __future__ import annotations

import torch.nn as nn

from core.backbone import build_backbone
from core.base import BaseSSLModule
from core.config import SimCLRConfig, TrainConfig
from core.losses import InfoNCELoss
from core.projection import ProjectionHead


class SimCLRv1Module(BaseSSLModule):
    """SimCLR v1 — contrastive learning with 2-layer projection head.

    Paper: "A Simple Framework for Contrastive Learning of Visual Representations"
    Authors: Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    Venue: ICML 2020
    arXiv: https://arxiv.org/abs/2002.05709

    Batch-size sensitivity: Because all negatives come from the current batch
    (no external bank or queue), the effective number of negatives is
    batch_size - 1.  Performance degrades significantly below batch size ~256.
    Use LARS optimizer for large-batch training (>= 256).

    Gotchas:
    - Batch size sensitivity: below ~256, representation quality degrades.
    - Color jitter strength s=1.0 (not the torchvision default s=0.4).
    - n_views=2 required in config.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        self.backbone, self.feat_dim = build_backbone(cfg.backbone, cfg.pretrained)
        self.projector = self.build_projector()
        simclr_cfg = cfg.simclr or SimCLRConfig()
        self.loss_fn = InfoNCELoss(temperature=simclr_cfg.temperature)

    def build_projector(self) -> nn.Module:
        """Build 2-layer MLP projection head (SimCLR v1)."""
        simclr_cfg = self.cfg.simclr or SimCLRConfig()
        return ProjectionHead(
            self.feat_dim,
            2048,
            simclr_cfg.projection_dim,
            num_layers=2,
        )

    def training_step(self, batch, batch_idx):
        """Forward pass: encode both views, compute symmetric NT-Xent loss.

        Args:
            batch: Tuple of (views, labels) where views has shape [2, B, C, H, W].
            batch_idx: Index of the current batch.

        Returns:
            Scalar loss tensor.
        """
        views, labels = batch  # views: [2, B, C, H, W]
        h_i = self.backbone(views[0])  # representations (for eval)
        h_j = self.backbone(views[1])
        z_i = self.projector(h_i)  # projections (for loss)
        z_j = self.projector(h_j)
        loss = self.loss_fn(z_i, z_j)  # symmetric NT-Xent, no queue (per D-01)
        self.log_train_metrics(loss)
        return loss


class SimCLRv2Module(SimCLRv1Module):
    """SimCLR v2 — 3-layer projection head, otherwise identical to v1.

    Paper: "Big Self-Supervised Models are Strong Semi-Supervised Learners"
    Authors: Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, Geoffrey Hinton
    Venue: NeurIPS 2020
    arXiv: https://arxiv.org/abs/2006.10029

    The only difference from v1 is a deeper (3-layer) projection head, which
    the paper shows improves representation quality, especially for larger
    backbones.  Weight decay sensitivity differs from v1 — tune carefully.
    """

    def build_projector(self) -> nn.Module:
        """Build 3-layer MLP projection head (SimCLR v2)."""
        simclr_cfg = self.cfg.simclr or SimCLRConfig()
        return ProjectionHead(
            self.feat_dim,
            2048,
            simclr_cfg.projection_dim,
            num_layers=3,
        )
