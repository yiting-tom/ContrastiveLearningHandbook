"""Invariant and Spreading Instance Feature (Ye et al., CVPR 2019).

In-batch contrastive learning without memory bank or queue. Two augmented
views of each image are produced; the model learns to bring views of the
same image together while pushing views of different images apart using
symmetric cross-entropy (InfoNCE) loss over in-batch instances.

Paper: "Unsupervised Embedding Learning via Invariant and Spreading Instance Feature"
Authors: Mang Ye, Xu Zhang, Pong C. Yuen, Shih-Fu Chang
Venue: CVPR 2019
arXiv: https://arxiv.org/abs/1904.03436

Reference implementation: https://github.com/mangye16/Unsupervised_Embedding_Learning
"""
from __future__ import annotations

import torch.nn as nn

from core.backbone import build_backbone
from core.base import BaseSSLModule
from core.config import InvariantSpreadConfig, TrainConfig
from core.losses import InfoNCELoss
from core.projection import ProjectionHead


class InvariantSpreadModule(BaseSSLModule):
    """Invariant and Spreading Instance Feature (Ye et al., CVPR 2019).

    In-batch contrastive learning without memory bank or queue. Two augmented
    views of each image are produced; the model learns to bring views of the
    same image together while pushing views of different images apart using
    symmetric cross-entropy (InfoNCE) loss over in-batch instances.

    Paper: "Unsupervised Embedding Learning via Invariant and Spreading Instance Feature"
    Authors: Mang Ye, Xu Zhang, Pong C. Yuen, Shih-Fu Chang
    Venue: CVPR 2019
    arXiv: https://arxiv.org/abs/1904.03436

    Batch-size sensitivity: Because all negatives come from the current batch
    (no external bank or queue), the effective number of negatives is
    batch_size - 1. Performance degrades significantly below batch size ~256,
    unlike queue-based methods (MoCo) or bank-based methods (Instance
    Discrimination) which maintain thousands of negatives regardless of batch
    size. This method is the direct ancestor of SimCLR, which addresses the
    batch-size sensitivity with stronger augmentations and LARS optimizer.

    Gotchas:
    - Batch size sensitivity: below ~256, representation quality degrades.
    - Uses weak augmentation (era-1 defaults: color jitter s=0.4, no blur).
    - n_views=2 required in config.

    Reference implementation: https://github.com/mangye16/Unsupervised_Embedding_Learning
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        self.backbone, self.feat_dim = build_backbone(cfg.backbone, cfg.pretrained)
        self.projector = self.build_projector()
        inv_cfg = cfg.invariant_spread or InvariantSpreadConfig()
        self.loss_fn = InfoNCELoss(temperature=inv_cfg.temperature)

    def build_projector(self) -> nn.Module:
        """Build 2-layer MLP projection head."""
        inv_cfg = self.cfg.invariant_spread or InvariantSpreadConfig()
        return ProjectionHead(
            self.feat_dim,
            2048,
            inv_cfg.projection_dim,
            num_layers=2,
        )

    def training_step(self, batch, batch_idx):
        """Forward pass: encode both views, compute symmetric InfoNCE loss.

        Args:
            batch: Tuple of (views, labels) where views has shape [2, B, C, H, W].
            batch_idx: Index of the current batch.

        Returns:
            Scalar loss tensor.
        """
        views, labels = batch  # views: [2, B, C, H, W]
        z_i = self.projector(self.backbone(views[0]))
        z_j = self.projector(self.backbone(views[1]))
        loss = self.loss_fn(z_i, z_j)  # symmetric in-batch, no queue (per D-03)
        self.log_train_metrics(loss)
        return loss
