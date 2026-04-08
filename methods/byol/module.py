"""BYOL: Bootstrap Your Own Latent (Grill et al., NeurIPS 2020).

No-negative self-supervised learning via online/target bootstrap.
The predictor on the online branch prevents representational collapse
without requiring any negative pairs.

Paper: "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning"
Authors: Jean-Bastien Grill, Florian Strub, Florent Altche, Corentin Tallec,
         Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila
         Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot,
         Koray Kavukcuoglu, Remi Munos, Michal Valko
Venue: NeurIPS 2020
arXiv: https://arxiv.org/abs/2006.07733
Reference implementation: https://github.com/deepmind/deepmind-research/tree/master/byol

Gotchas:
- The predictor is on the ONLINE branch only. The target branch has no predictor.
  Adding a predictor to the target branch breaks the asymmetry that prevents collapse.
- EMA momentum must be cosine-scheduled (0.996->1.0), not constant. Constant momentum
  leads to suboptimal representations.
- Target branch parameters must have requires_grad=False and be excluded from the
  optimizer via the learnable_params property. Including them causes silent waste.
- L2-normalize projector outputs BEFORE the loss and BEFORE the predictor input.
  The original paper normalizes at the projector output stage.
- The stop-gradient is implicit: target branch output is detached via no_grad context
  during the forward pass. There is no explicit .detach() call needed if the target
  network parameters have requires_grad=False.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.backbone import build_backbone
from core.base import BaseSSLModule
from core.config import BYOLConfig, TrainConfig
from core.ema import EMAUpdater
from core.projection import PredictorHead, ProjectionHead


class BYOLModule(BaseSSLModule):
    """BYOL (Grill et al., NeurIPS 2020).

    Bootstrap Your Own Latent — no-negative self-supervised learning.

    Architecture:
    - Online network: backbone -> projector -> predictor
    - Target network: backbone_ema -> projector_ema (NO predictor)

    The MSE loss (2 - 2*cosine_similarity) between online predictions and
    target projections, combined with EMA updates and the predictor asymmetry,
    prevents representational collapse without negatives.

    Args:
        cfg: TrainConfig with cfg.byol populated.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        byol_cfg: BYOLConfig = cfg.byol or BYOLConfig()

        # Online network
        self.backbone, self.feat_dim = build_backbone(cfg.backbone, cfg.pretrained)
        self.projector = self.build_projector()
        self.predictor = PredictorHead(
            predictor_type="standard",
            input_dim=256,
            hidden_dim=4096,
            output_dim=256,
        )

        # Target network — deep copy, freeze
        self.backbone_ema = copy.deepcopy(self.backbone)
        self.projector_ema = copy.deepcopy(self.projector)
        for p in self.backbone_ema.parameters():
            p.requires_grad_(False)
        for p in self.projector_ema.parameters():
            p.requires_grad_(False)

        # EMA updater with cosine-scheduled momentum — initialized in setup()
        # once total_steps is known from the trainer.
        self._byol_base_momentum = byol_cfg.base_momentum
        self._byol_end_momentum = byol_cfg.end_momentum
        self.ema: EMAUpdater | None = None

    def build_projector(self) -> nn.Module:
        """2-layer projection head (BYOL default: feat_dim->4096->256)."""
        return ProjectionHead(
            input_dim=self.feat_dim,
            hidden_dim=4096,
            output_dim=256,
            num_layers=2,
        )

    def setup(self, stage: str) -> None:
        """Initialize EMA updater once total_steps is known from trainer."""
        if stage == "fit" and self.trainer is not None:
            total_steps = self.trainer.estimated_stepping_batches
            self.ema = EMAUpdater(
                base_momentum=self._byol_base_momentum,
                end_momentum=self._byol_end_momentum,
                total_steps=int(total_steps),
            )

    @property
    def learnable_params(self):
        """Exclude target network from optimizer.

        Target params have requires_grad=False so this filter is both
        explicit (intent) and redundant (safety). The base class default
        of self.parameters() would technically work because frozen params
        are excluded by the optimizer, but overriding makes intent clear.
        """
        return (p for p in self.parameters() if p.requires_grad)

    def training_step(self, batch, batch_idx):
        """Symmetric BYOL loss between online predictions and target projections.

        Args:
            batch: Tuple of (views, labels) where views[0] and views[1] are
                the two augmented views of shape [B, C, H, W].
            batch_idx: Index of the current batch.

        Returns:
            Scalar loss tensor.
        """
        (v1, v2), _ = batch

        # Online forward — gradient flows through backbone, projector, predictor
        z1 = F.normalize(self.projector(self.backbone(v1)), dim=1)
        z2 = F.normalize(self.projector(self.backbone(v2)), dim=1)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # Target forward — no gradient (stop-gradient via no_grad context)
        with torch.no_grad():
            t1 = F.normalize(self.projector_ema(self.backbone_ema(v1)), dim=1)
            t2 = F.normalize(self.projector_ema(self.backbone_ema(v2)), dim=1)

        # Stop-gradient: target outputs are detached — gradients flow only through the
        # online predictor, not the target branch. This asymmetry (predictor on online
        # branch only) is the mechanism that prevents representational collapse without
        # any negative pairs. Removing .detach() here (or the torch.no_grad() context
        # above) causes immediate collapse because the optimization can trivially satisfy
        # the loss by making both branches output the same constant vector.
        loss = (
            (2 - 2 * F.cosine_similarity(p1, t2.detach(), dim=1)).mean()  # stop-gradient
            + (2 - 2 * F.cosine_similarity(p2, t1.detach(), dim=1)).mean()  # stop-gradient
        ) / 2

        # Collapse monitoring — log embedding_std from online projector output
        with torch.no_grad():
            embedding_std = z1.std(dim=0).mean()
            self.log("train/embedding_std", embedding_std, on_step=True, on_epoch=False)

        self.log_train_metrics(loss)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """EMA update of target network after each optimizer step.

        Passes parameter iterables (not modules) to EMAUpdater.step() as
        required by the EMAUpdater API.
        """
        if self.ema is not None:
            self.ema.step(
                self.backbone.parameters(),
                self.backbone_ema.parameters(),
            )
            self.ema.step(
                self.projector.parameters(),
                self.projector_ema.parameters(),
            )
