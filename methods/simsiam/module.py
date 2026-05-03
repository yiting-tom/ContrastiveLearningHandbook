"""SimSiam: Exploring Simple Siamese Representation Learning (Chen & He, 2021).

No-negative self-supervised learning with a shared encoder and stop-gradient.
The stop-gradient on the projection (z) side is the ONLY mechanism preventing
collapse — there is no momentum encoder, no queue, no negatives.

Paper: "Exploring Simple Siamese Representation Learning"
Authors: Xinlei Chen, Kaiming He
Venue: CVPR 2021
arXiv: https://arxiv.org/abs/2011.10566
Reference implementation: https://github.com/facebookresearch/simsiam

Gotchas:
- The .detach() on z2 (and z1) is load-bearing. Removing it causes immediate
  collapse: the loss drops to exactly -1.0 within 2 epochs because the network
  learns the trivial solution of mapping all inputs to the same point.
  This is documented by a unit test in tests/test_simsiam.py.
- The projector uses BN+ReLU on intermediate layers and BN-only on the output
  (standard ProjectionHead convention). The predictor uses a bottleneck MLP
  (2048->512->2048) with BN on ALL layers including output, no ReLU on output.
- There is no EMA or momentum encoder. Both views share the exact same backbone
  and projector weights. The stop-gradient is the ONLY asymmetry.
- L2-normalization before cosine_similarity is NOT needed because F.cosine_similarity
  normalizes internally. However, you may still normalize for numerical stability.
- Default projection dim is 2048 (not 128 like SimCLR). SimSiam uses higher-dim
  projections to stabilize training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.backbone import build_backbone
from core.base import BaseSSLModule
from core.config import SimSiamConfig, TrainConfig
from core.projection import PredictorHead, ProjectionHead


class SimSiamModule(BaseSSLModule):
    """SimSiam (Chen & He, CVPR 2021).

    Exploring Simple Siamese Representation Learning.

    SimSiam is BYOL without the EMA target — a single shared encoder produces
    both views, and the only thing preventing collapse is the predictor MLP
    plus a stop-gradient on the projector output. The symmetric negative-cosine
    loss `-(cos_sim(p1, z2.detach()) + cos_sim(p2, z1.detach())) / 2` is
    computed on L2-normalized vectors.

    Paper: "Exploring Simple Siamese Representation Learning"
    Authors: Xinlei Chen, Kaiming He
    Venue: CVPR 2021
    arXiv: https://arxiv.org/abs/2011.10566

    Algorithm:
    1. Augment x into v1, v2; encode both with the shared backbone.
    2. Project: z1 = projector(h1); z2 = projector(h2). 3-layer projector with
       BN on all layers including the output (no ReLU on output).
    3. Predict: p1 = predictor(z1); p2 = predictor(z2). Bottleneck 2-layer MLP
       (2048 -> 512 -> 2048).
    4. Loss = -(cos_sim(p1, z2.detach()) + cos_sim(p2, z1.detach())) / 2.

    Gotchas:
    - Forgetting `.detach()` causes immediate collapse to loss = -1.0 plateau —
      this is the single most common SimSiam implementation bug.
    - The final projector layer has BN but NO ReLU. Adding ReLU here measurably
      hurts performance.
    - Use SGD (NOT LARS); paper uses lr = 0.05 * batch_size / 256.
    - Mandatory training diagnostic: log z.std(dim=0).mean(); should stay near
      0.707. Approaching 0 indicates collapse.

    Reference implementation: https://github.com/facebookresearch/simsiam
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        simsiam_cfg: SimSiamConfig = cfg.simsiam or SimSiamConfig()

        self.backbone, self.feat_dim = build_backbone(cfg.backbone, cfg.pretrained)
        self.projector = self.build_projector()
        self.predictor = PredictorHead(
            predictor_type="bottleneck",
            input_dim=2048,
            bottleneck_dim=simsiam_cfg.predictor_hidden_dim,
        )

    def build_projector(self) -> nn.Module:
        """3-layer projection head (SimSiam default: feat_dim->2048->2048->2048)."""
        return ProjectionHead(
            input_dim=self.feat_dim,
            hidden_dim=2048,
            output_dim=2048,
            num_layers=3,
        )

    def training_step(self, batch, batch_idx):
        (v1, v2), _ = batch

        # Shared encoder — both views through the same backbone and projector
        z1 = self.projector(self.backbone(v1))
        z2 = self.projector(self.backbone(v2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # Stop-gradient loss: detach z (projection), gradient flows through p (prediction)
        # COLLAPSE WARNING: removing .detach() causes loss=-1.0 collapse within 2 epochs.
        # The stop-gradient is the only mechanism preventing collapse in SimSiam.
        loss = -(
            F.cosine_similarity(p1, z2.detach(), dim=1).mean()  # stop-gradient on z2
            + F.cosine_similarity(p2, z1.detach(), dim=1).mean()  # stop-gradient on z1
        ) / 2

        # Collapse monitoring — embedding_std approaches 0 on collapse
        with torch.no_grad():
            embedding_std = z1.std(dim=0).mean()
            self.log("train/embedding_std", embedding_std, on_step=True, on_epoch=False)

        self.log_train_metrics(loss)
        return loss
