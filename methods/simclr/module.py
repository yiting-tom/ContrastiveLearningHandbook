"""SimCLR v1 and v2 (Chen et al., ICML 2020 / NeurIPS 2020).

Two-view in-batch contrastive learning with symmetric NT-Xent loss. Both views
are encoded by a shared backbone and projected through an MLP head. Loss is
computed on the projection output z; downstream evaluation uses the backbone
representation h. Strong augmentation (color jitter s=1.0, Gaussian blur) is
critical for performance.

Paper (v1): "A Simple Framework for Contrastive Learning of Visual Representations"
Authors: Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
Venue: ICML 2020
arXiv: https://arxiv.org/abs/2002.05709

Paper (v2): "Big Self-Supervised Models are Strong Semi-Supervised Learners"
Authors: Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, Geoffrey Hinton
Venue: NeurIPS 2020
arXiv: https://arxiv.org/abs/2006.10029

Reference implementation: https://github.com/google-research/simclr
"""
from __future__ import annotations

import torch.nn as nn

from core.backbone import build_backbone
from core.base import BaseSSLModule
from core.config import SimCLRConfig, TrainConfig
from core.losses import InfoNCELoss
from core.projection import ProjectionHead


class SimCLRv1Module(BaseSSLModule):
    """SimCLR v1 (Chen et al., ICML 2020).

    A Simple Framework for Contrastive Learning of Visual Representations.

    Two augmented views of each image are encoded by a shared backbone and
    projected through a 2-layer MLP (2048->2048->128). The symmetric NT-Xent
    loss (implemented via InfoNCELoss with queue=None) brings views of the same
    image together while pushing views of different images apart within the
    batch. Loss is computed on the projection output z; downstream evaluation
    should use the backbone representation h.

    Paper: "A Simple Framework for Contrastive Learning of Visual Representations"
    Authors: Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    Venue: ICML 2020
    arXiv: https://arxiv.org/abs/2002.05709

    Algorithm:
    1. Augment each image twice with strong augmentation (s=1.0 color jitter,
       Gaussian blur, random grayscale, random crop + resize).
    2. Encode both views: h_i = backbone(x_i), h_j = backbone(x_j).
    3. Project: z_i = projector(h_i), z_j = projector(h_j).
    4. Compute symmetric NT-Xent loss on (z_i, z_j).

    Gotchas:
    - Color jitter strength must be s=1.0 (not torchvision default ~0.4).
      ContrastiveAugmentation(strong=True) handles this.
    - Performance degrades sharply below batch_size=256 because effective
      negatives = 2*(batch_size-1). Use LARS optimizer for batch sizes >1024.
    - Loss is computed on z (projection), but evaluation must use h (backbone).
      Do not evaluate downstream tasks on z.
    - InfoNCELoss internally L2-normalizes inputs; do not pre-normalize.

    Reference implementation: https://github.com/google-research/simclr
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
    """SimCLR v2 (Chen et al., NeurIPS 2020).

    Big Self-Supervised Models are Strong Semi-Supervised Learners.

    Extends SimCLR v1 with a deeper 3-layer projection head
    (2048->2048->2048->128). Only the projection head depth changes;
    backbone, loss, and augmentation are identical to v1. This
    implementation covers the pretraining stage only; the semi-supervised
    distillation stage is deferred to v2 scope.

    Paper: "Big Self-Supervised Models are Strong Semi-Supervised Learners"
    Authors: Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, Geoffrey Hinton
    Venue: NeurIPS 2020
    arXiv: https://arxiv.org/abs/2006.10029

    Gotchas:
    - Weight decay scales with projection head depth -- v1 fine-tune
      hyperparameters do not transfer directly to v2. The larger projection
      head (3 layers vs 2) requires more regularization.
    - The num_layers=2 -> num_layers=3 switch is controlled by this subclass,
      not by YAML config (per D-02).

    Reference implementation: https://github.com/google-research/simclr
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
