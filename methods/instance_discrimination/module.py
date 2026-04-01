"""Instance Discrimination SSL module (Wu et al., CVPR 2018).

Non-parametric instance-level discrimination using a memory bank.
Each image is treated as its own class. The encoder produces a feature
which is compared against stored features in the bank via NCE loss.

The memory bank stores one L2-normalized feature vector per training
sample. Bank entries are updated with current encoder outputs each step
(no EMA). The NCE normalization constant Z is estimated on the first
mini-batch and fixed thereafter.

Paper: "Unsupervised Feature Learning via Non-Parametric Instance Discrimination"
Authors: Zhirong Wu, Yuanjun Xiong, Stella X. Yu, Dahua Lin
Venue: CVPR 2018
arXiv: https://arxiv.org/abs/1805.01978

Gotchas:
- Z must be fixed after initial estimation; recomputing destabilizes training.
- Bank features become stale as the encoder trains. See MemoryBank docstring.
- Uses weak augmentation (ContrastiveAugmentation strong=False), not SimCLR-strength.
- n_views=1 in config; bank provides the "second view".

Reference implementation: https://github.com/zhirongw/lemniscate.pytorch
"""
from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import BaseSSLModule
from core.backbone import build_backbone
from core.config import InstanceDiscriminationConfig, TrainConfig
from core.memory_bank import MemoryBank
from core.projection import ProjectionHead
from methods.instance_discrimination.losses import NCELossWithFixedZ


class InstanceDiscriminationModule(BaseSSLModule):
    """Instance Discrimination (Wu et al., CVPR 2018).

    Non-parametric instance-level discrimination using a memory bank.
    Each image is treated as its own class. The encoder produces a feature
    which is compared against stored features in the bank via NCE loss.

    The memory bank stores one L2-normalized feature vector per training
    sample. Bank entries are updated with current encoder outputs each step
    (no EMA). The NCE normalization constant Z is estimated on the first
    mini-batch and fixed thereafter.

    Paper: "Unsupervised Feature Learning via Non-Parametric Instance Discrimination"
    Authors: Zhirong Wu, Yuanjun Xiong, Stella X. Yu, Dahua Lin
    Venue: CVPR 2018
    arXiv: https://arxiv.org/abs/1805.01978

    Gotchas:
    - Z must be fixed after initial estimation; recomputing destabilizes training.
    - Bank features become stale as the encoder trains. See MemoryBank docstring.
    - Uses weak augmentation (ContrastiveAugmentation strong=False), not SimCLR-strength.
    - n_views=1 in config; bank provides the "second view".

    Reference implementation: https://github.com/zhirongw/lemniscate.pytorch
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        self.id_cfg = cfg.instance_discrimination or InstanceDiscriminationConfig()

        # Backbone encoder
        self.backbone, self.feat_dim = build_backbone(cfg.backbone, cfg.pretrained)

        # Projection head
        self.projector = self.build_projector()

        # NCE loss with fixed Z normalization
        self.nce_loss = NCELossWithFixedZ(
            temperature=self.id_cfg.temperature,
            n_negatives=self.id_cfg.n_negatives,
        )

        # Memory bank -- created lazily in setup() or set externally in tests
        self.memory_bank: MemoryBank | None = None

    def build_projector(self) -> nn.Module:
        """Build 2-layer MLP projection head."""
        return ProjectionHead(
            self.feat_dim,
            2048,
            self.id_cfg.projection_dim,
            num_layers=2,
        )

    def setup(self, stage: str | None = None) -> None:
        """Initialize memory bank when trainer.datamodule is available."""
        if self.memory_bank is None:
            n_samples = len(self.trainer.datamodule.train_dataset)
            self.memory_bank = MemoryBank(n_samples, self.id_cfg.projection_dim)

    @property
    def learnable_params(self):
        """Parameters for the optimizer -- excludes memory bank."""
        return chain(self.backbone.parameters(), self.projector.parameters())

    def _sample_negatives(
        self, indices: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Sample random negatives from the memory bank.

        Args:
            indices: Current batch indices (not used for exclusion in this
                simple implementation -- global random sampling).
            batch_size: Number of samples in the current batch.

        Returns:
            Negatives tensor of shape [B, m, D].
        """
        n = self.memory_bank.n_samples
        m = self.id_cfg.n_negatives
        neg_indices = torch.randint(0, n, (m,), device=indices.device)
        negatives = self.memory_bank.get(neg_indices)  # [m, D]
        return negatives.unsqueeze(0).expand(batch_size, -1, -1)  # [B, m, D]

    def training_step(self, batch, batch_idx):
        """Forward pass + NCE loss + bank update.

        Args:
            batch: (views, labels, indices) from ssl_collate_with_index.
                views shape: [n_views, B, C, H, W] (n_views=1).
            batch_idx: Index of the current batch.

        Returns:
            Scalar loss tensor.
        """
        views, _labels, indices = batch
        x = views[0]  # n_views=1, take the single view

        # Forward through encoder + projector
        h = self.backbone(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)

        # Get positive from bank (stored feature for same index)
        positive = self.memory_bank.get(indices)

        # Sample negatives from bank
        negatives = self._sample_negatives(indices, x.size(0))

        # NCE loss
        loss = self.nce_loss(z, positive, negatives)

        # Update bank with current encoder output
        self.memory_bank.update(indices, z)

        self.log_train_metrics(loss)
        return loss
