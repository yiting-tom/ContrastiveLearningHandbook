"""MoCo v3 (Chen, Xie, He -- ICCV 2021).

Momentum Contrast adapted for Vision Transformers with three critical changes:

1. **Patch projection freeze** (key stability fix): `backbone.patch_embed.proj`
   weights and biases are frozen immediately at construction time. Without this,
   the patch embedding layer becomes unstable during early training, causing loss
   spikes and training collapse with ViT backbones.

2. **In-batch symmetric InfoNCE (no queue)**: MoCo v3 drops the queue entirely.
   With large-batch ViT training (4096+), the batch itself provides enough
   negatives. The symmetric loss (average of both directions) provides gradient
   signal from both views simultaneously.

3. **AdamW optimizer (not SGD/LARS)**: ViT training requires AdamW with weight
   decay and cosine LR schedule for stable optimization. SGD/LARS patterns
   from ResNet-based methods do not transfer to ViTs.

Paper: "An Empirical Study of Training Self-Supervised Vision Transformers"
Authors: Xinlei Chen, Saining Xie, Kaiming He
Venue: ICCV 2021
arXiv: https://arxiv.org/abs/2104.02057

Gotchas:
1. **Freeze patch_embed.proj** -- the single most important stability fix. The
   patch embedding projection (a Conv2d) maps raw pixels to patch tokens. If
   this layer is allowed to adapt freely, it becomes the "easy shortcut" for
   the optimizer: the model learns to create trivially discriminative patches
   rather than meaningful semantic representations. Freezing it forces the
   Transformer layers to do the representational heavy-lifting.
2. **Use AdamW, not SGD or LARS**. ViTs have no inductive biases (no conv
   locality, no pooling), so they rely on the weight decay and adaptive
   learning rates of AdamW for stable optimization. SGD diverges or converges
   slowly on ViT.
3. **m=0.99 not m=0.999**. MoCo v3 uses lower momentum (slower EMA) compared
   to MoCo v1/v2. With large batches and shorter training schedules, a faster
   target network update (m=0.99 vs 0.999) provides better gradient signal.
4. **Gradient clipping recommended**: Add `gradient_clip_val=1.0` to
   L.Trainer when training with ViT backbones for additional stability.
5. **No queue -- in-batch keys only**: With batch sizes ≥ 1024, in-batch
   negatives are sufficient. The queue (MoCo v1/v2 feature) added complexity
   without benefit at large batch scales.

Reference implementation: https://github.com/facebookresearch/moco-v3
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from core.backbone import build_backbone
from core.base import BaseSSLModule
from core.config import MoCoV3Config, TrainConfig
from core.ema import EMAUpdater
from core.losses import InfoNCELoss
from core.projection import PredictorHead, ProjectionHead


class MoCoV3Module(BaseSSLModule):
    """MoCo v3 (Chen, Xie, He -- ICCV 2021).

    Momentum contrastive learning adapted for Vision Transformers.

    Architecture:
    - Online network: backbone -> projector -> predictor
    - Momentum network: backbone_ema -> projector_ema (NO predictor)

    The symmetric in-batch InfoNCE loss is computed between online predictions
    and momentum keys from both views, with no queue. The patch projection
    layer of the ViT backbone is frozen at construction time.

    Paper: "An Empirical Study of Training Self-Supervised Vision Transformers"
    Authors: Xinlei Chen, Saining Xie, Kaiming He
    Venue: ICCV 2021
    arXiv: https://arxiv.org/abs/2104.02057

    Algorithm:
    1. Augment each image twice (v1, v2).
    2. Freeze backbone.patch_embed.proj (ViT stability trick).
    3. Online branch: q1 = predictor(projector(backbone(v1))),
                      q2 = predictor(projector(backbone(v2))).
    4. Momentum branch (no grad): k1 = projector_ema(backbone_ema(v1)),
                                   k2 = projector_ema(backbone_ema(v2)).
    5. Symmetric loss: (InfoNCE(q1, k2) + InfoNCE(q2, k1)) / 2.
    6. EMA update of momentum encoder in on_train_batch_end.

    Gotchas:
    - Freeze patch_embed.proj in __init__ -- key stability fix for ViT training.
    - Predictor is on the ONLINE branch only. Momentum branch has no predictor.
    - Momentum encoder params must have requires_grad=False and be excluded
      from the optimizer via the learnable_params property.
    - Use AdamW, not SGD/LARS. ViTs require adaptive optimizer with weight decay.
    - m=0.99 (not 0.999) -- MoCo v3 uses slower EMA update rate.
    - No queue -- in-batch keys are sufficient with large batches.
    - Gradient clipping (gradient_clip_val=1.0) improves stability with ViT.

    Reference implementation: https://github.com/facebookresearch/moco-v3

    Args:
        cfg: TrainConfig with cfg.moco_v3 populated (or defaults applied).
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        moco_v3_cfg: MoCoV3Config = cfg.moco_v3 or MoCoV3Config()

        # Online network
        self.backbone, self.feat_dim = build_backbone(cfg.backbone, cfg.pretrained)

        # Freeze patch projection layer (ViT-specific stability fix, D-07)
        # This must happen BEFORE building the momentum encoder (deepcopy)
        # so the freeze propagates to the EMA backbone as well.
        if hasattr(self.backbone, "patch_embed"):
            self.backbone.patch_embed.proj.weight.requires_grad_(False)
            self.backbone.patch_embed.proj.bias.requires_grad_(False)

        self.projector = self.build_projector()

        # Predictor on online branch only (no predictor on momentum branch)
        self.predictor = PredictorHead(
            predictor_type="standard",
            input_dim=256,
            hidden_dim=moco_v3_cfg.predictor_hidden_dim,
            output_dim=256,
        )

        # Momentum encoder -- deep copy, all params frozen
        self.backbone_ema = copy.deepcopy(self.backbone)
        self.backbone_ema.requires_grad_(False)
        self.projector_ema = copy.deepcopy(self.projector)
        self.projector_ema.requires_grad_(False)

        # Loss (symmetric in-batch InfoNCE -- no queue)
        self.loss_fn = InfoNCELoss(temperature=moco_v3_cfg.temperature)

        # EMA updater -- initialized in setup() once total_steps is known
        self._moco_v3_momentum = moco_v3_cfg.momentum
        self.ema: EMAUpdater | None = None

    def build_projector(self) -> nn.Module:
        """3-layer MLP projection head (feat_dim -> 4096 -> 256)."""
        return ProjectionHead(
            input_dim=self.feat_dim,
            hidden_dim=4096,
            output_dim=256,
            num_layers=3,
        )

    def setup(self, stage: str) -> None:
        """Initialize EMA updater once total_steps is known from trainer.

        Uses constant momentum (base==end) per MoCo v3 convention.
        For cosine-scheduled momentum, use BYOL/DINO patterns.
        """
        if stage == "fit" and self.trainer is not None:
            total_steps = self.trainer.estimated_stepping_batches
            self.ema = EMAUpdater(
                base_momentum=self._moco_v3_momentum,
                end_momentum=self._moco_v3_momentum,
                total_steps=int(total_steps),
            )

    @property
    def learnable_params(self):
        """Parameters for the optimizer -- excludes momentum encoder.

        Chains online backbone, projector, and predictor parameters.
        backbone_ema and projector_ema are explicitly excluded (requires_grad=False).
        """
        import itertools
        return itertools.chain(
            self.backbone.parameters(),
            self.projector.parameters(),
            self.predictor.parameters(),
        )

    def training_step(self, batch, batch_idx):
        """Symmetric in-batch InfoNCE loss between online predictions and momentum keys.

        Args:
            batch: Tuple of (views, labels) where views[0] and views[1] are
                the two augmented views of shape [B, C, H, W].
            batch_idx: Index of the current batch.

        Returns:
            Scalar loss tensor.
        """
        views, labels = batch
        # Handle both stacked tensor [2, B, C, H, W] and list/tuple of 2 views
        if isinstance(views, torch.Tensor):
            v1, v2 = views[0], views[1]
        else:
            v1, v2 = views[0], views[1]

        # Online branch -- gradient flows through backbone, projector, predictor
        z1 = self.projector(self.backbone(v1))
        z2 = self.projector(self.backbone(v2))
        q1 = self.predictor(z1)
        q2 = self.predictor(z2)

        # Momentum branch -- no gradient (stop-gradient via no_grad context)
        with torch.no_grad():
            k1 = self.projector_ema(self.backbone_ema(v1))
            k2 = self.projector_ema(self.backbone_ema(v2))

        # Symmetric InfoNCE (in-batch, no queue)
        # Cross-predict: q1 predicts k2, q2 predicts k1
        loss = (self.loss_fn(q1, k2) + self.loss_fn(q2, k1)) / 2

        self.log_train_metrics(loss)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """EMA update of momentum encoder after each optimizer step."""
        if self.ema is not None:
            self.ema.step(
                self.backbone.parameters(),
                self.backbone_ema.parameters(),
            )
            self.ema.step(
                self.projector.parameters(),
                self.projector_ema.parameters(),
            )
