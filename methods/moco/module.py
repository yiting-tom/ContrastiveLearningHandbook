"""MoCo v1 and v2 (He et al., CVPR 2020 / Chen et al., arXiv 2020).

Queue-based momentum contrastive learning. A momentum-updated encoder produces
key representations that are stored in a large FIFO queue, providing a rich set
of negatives for InfoNCE loss without requiring a large batch size.

MoCo v1 uses a bare nn.Linear projection; MoCo v2 upgrades to a 2-layer MLP
projection head (matching SimCLR's improvement) while keeping everything else
identical.

Paper (v1): "Momentum Contrast for Unsupervised Visual Representation Learning"
Authors: Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick
Venue: CVPR 2020
arXiv: https://arxiv.org/abs/1911.05722

Paper (v2): "Improved Baselines with Momentum Contrastive Learning"
Authors: Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He
arXiv: https://arxiv.org/abs/2003.04297

Reference implementation: https://github.com/facebookresearch/moco
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from core.backbone import build_backbone
from core.base import BaseSSLModule
from core.config import MoCoConfig, TrainConfig
from core.ema import EMAUpdater
from core.losses import InfoNCELoss
from core.projection import ProjectionHead
from core.queue import MomentumQueue


class MoCoV1Module(BaseSSLModule):
    """MoCo v1 (He et al., CVPR 2020).

    Momentum Contrast for Unsupervised Visual Representation Learning.

    Uses a momentum-updated encoder to produce consistent key representations
    stored in a large FIFO queue. The query encoder is trained via InfoNCE loss
    against the queue negatives, decoupling the number of negatives from batch
    size.

    Paper: "Momentum Contrast for Unsupervised Visual Representation Learning"
    Authors: Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick
    Venue: CVPR 2020
    arXiv: https://arxiv.org/abs/1911.05722

    Algorithm:
    1. Augment each image twice (query view and key view).
    2. Query path: q = projector(backbone(x_q))  -- gradient flows.
    3. Key path (no grad): k = projector_ema(backbone_ema(x_k))  -- detached.
    4. Compute InfoNCE loss with queue negatives.
    5. Update queue FIFO with current keys (after loss computation).
    6. EMA update of momentum encoder in on_train_batch_end.

    Gotchas:
    - Shuffled BN is required for multi-GPU training to prevent BN statistics
      leakage between query and key encoders. This tutorial implementation is
      single-GPU only and does not implement shuffled BN. On multi-GPU setups,
      the key encoder's BN would leak information about other samples in the
      batch, effectively creating a shortcut that bypasses contrastive learning.
    - m=0.9 produces significantly worse results than m=0.999. Lower momentum
      means the target encoder diverges faster from the online encoder, reducing
      key consistency across the queue. The queue stores keys produced at
      different training steps, so a fast-changing encoder makes older keys
      stale and inconsistent with recent ones.
    - Queue must be updated AFTER loss computation, not before. Enqueueing
      current keys before computing loss would make them appear as both
      positive and negative, breaking the contrastive objective.
    - Momentum encoder params must have requires_grad=False and must NOT
      appear in the optimizer. The learnable_params property enforces this.
    - MoCo v1 uses a bare nn.Linear projection (no BN, no hidden layer).
      The MLP projection head upgrade is MoCo v2.
    - Temperature 0.07 (not 0.5 like SimCLR) -- MoCo uses a much sharper
      distribution because the queue provides many more negatives.

    Reference implementation: https://github.com/facebookresearch/moco
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        moco_cfg = cfg.moco or MoCoConfig()

        # Online encoder
        self.backbone, self.feat_dim = build_backbone(cfg.backbone, cfg.pretrained)
        self.projector = self.build_projector()

        # Momentum encoder (deep copy, no gradients)
        self.backbone_ema = copy.deepcopy(self.backbone)
        self.backbone_ema.requires_grad_(False)
        self.projector_ema = copy.deepcopy(self.projector)
        self.projector_ema.requires_grad_(False)

        # Queue for negative keys
        self.momentum_queue = MomentumQueue(moco_cfg.queue_size, 128)

        # Loss
        self.loss_fn = InfoNCELoss(temperature=moco_cfg.temperature)

        # Wire EMA -- constant momentum (base == end) per MoCo v1/v2 convention
        self.ema_updater = EMAUpdater(
            base_momentum=moco_cfg.momentum,
            end_momentum=moco_cfg.momentum,
            total_steps=1,
        )
        self._online_params = list(self.backbone.parameters()) + list(
            self.projector.parameters()
        )
        self._target_params = list(self.backbone_ema.parameters()) + list(
            self.projector_ema.parameters()
        )

    def build_projector(self) -> nn.Module:
        """Build bare linear projection head (MoCo v1 -- no BN, no hidden layer)."""
        return nn.Linear(self.feat_dim, 128)

    @property
    def learnable_params(self):
        """Parameters for the optimizer -- excludes momentum encoder."""
        return list(self.backbone.parameters()) + list(self.projector.parameters())

    def training_step(self, batch, batch_idx):
        """Forward pass: query via online encoder, key via momentum encoder.

        Args:
            batch: Tuple of (views, labels) where views has shape [2, B, C, H, W].
            batch_idx: Index of the current batch.

        Returns:
            Scalar loss tensor.
        """
        views, labels = batch

        # Query path (gradients flow)
        q = self.projector(self.backbone(views[0]))

        # Key path (no gradients -- momentum encoder)
        with torch.no_grad():
            k = self.projector_ema(self.backbone_ema(views[1]))
            k = k.detach()

        # Compute loss with queue negatives
        loss = self.loss_fn(q, k, queue=self.momentum_queue.get_negatives())

        # Update queue AFTER loss computation (D-07)
        self.momentum_queue.update(k)

        self.log_train_metrics(loss)
        return loss


class MoCoV2Module(MoCoV1Module):
    """MoCo v2 (Chen et al., arXiv 2020).

    Improved Baselines with Momentum Contrastive Learning.

    This is a "5-line diff from v1": SimCLR's architecture improvements (2-layer
    MLP projection head, stronger augmentation with Gaussian blur, cosine LR
    schedule) applied to MoCo's momentum-queue framework. All other components
    -- momentum encoder, queue, InfoNCE loss, EMA schedule -- are inherited
    unchanged from MoCoV1Module.

    Paper: "Improved Baselines with Momentum Contrastive Learning"
    Authors: Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He
    Venue: arXiv 2020 (tech report)
    arXiv: https://arxiv.org/abs/2003.04297

    Changes from v1:
    1. 2-layer MLP projection head replaces bare nn.Linear (128-d output kept).
    2. Strong augmentation: s=1.0 color jitter + Gaussian blur (from SimCLR).
    3. Cosine learning rate schedule (warmup_cosine).

    Gotchas:
    - MoCo v2 is essentially "SimCLR's architecture improvements applied to
      MoCo". The paper demonstrates that these improvements are orthogonal to
      the contrastive mechanism (in-batch vs. queue).
    - The only code change from v1 is the projection head: nn.Linear -> 2-layer
      MLP. Augmentation and scheduler changes are config-level, not code-level.
    - Weight decay and learning rate may need re-tuning when switching from v1
      to v2 due to the additional projection head parameters.

    Reference implementation: https://github.com/facebookresearch/moco
    """

    def build_projector(self) -> nn.Module:
        """Build 2-layer MLP projection head (MoCo v2)."""
        return ProjectionHead(self.feat_dim, 2048, 128, num_layers=2)
