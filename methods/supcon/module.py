"""Supervised Contrastive Learning — Khosla et al., NeurIPS 2020.

Two-stage training workflow:

Stage 1 (this module — SupConModule):
    Pretrains backbone + 2-layer projection head with SupCon loss.
    Class-balanced sampling ensures at least n_samples_per_class instances
    per class per batch, which is required for the supervised positives to
    be meaningful.

    Command:
        python train.py --config configs/supcon_stage1_resnet18.yaml

Stage 2 (SupConFinetuneModule):
    Loads stage-1 checkpoint, FREEZES the backbone, trains a linear
    classification head with SGD (weight_decay=0.0).

    Command:
        python train.py --config configs/supcon_stage2_resnet18.yaml \\
            --ckpt_path logs/supcon/version_0/checkpoints/last.ckpt

Paper: "Supervised Contrastive Learning"
Authors: Prannay Khosla, Piyush Tian, Chen Wang, Aaron Neimark,
         Piyush Rai, Chen Xu, Dilip Krishnan, Serge Belongie
Venue: NeurIPS 2020
arXiv: https://arxiv.org/abs/2004.11362

Reference implementation: https://github.com/HobbitLong/SupContrast
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.backbone import build_backbone
from core.base import BaseSSLModule
from core.config import SupConConfig, TrainConfig
from core.data import ContrastiveAugmentation, SSLDataModule
from core.losses import SupConLoss
from core.projection import ProjectionHead


class SupConModule(BaseSSLModule):
    """SupCon stage-1 pretraining module (Khosla et al., NeurIPS 2020).

    Encodes two augmented views through a shared backbone and 2-layer MLP
    projection head. The SupCon loss treats every in-batch image with the
    same class label as a positive for each anchor, not just the other
    augmented view. Use ClassBalancedSampler (via sampler_type='class_balanced'
    in SSLDataModule) to ensure >= n_samples_per_class instances per class
    appear in each batch — without this, many classes will be singletons and
    the supervised term degrades to SimCLR.

    NO classifier head is used during stage 1. Adding cross-entropy here
    collapses the contrastive representation (warned against in the paper).

    Algorithm:
    1. Sample batch with ClassBalancedSampler.
    2. Encode both views: h_i = backbone(x_i), h_j = backbone(x_j).
    3. Project: z_i = projector(h_i), z_j = projector(h_j).
    4. SupConLoss(z_i, z_j, labels) — sum-outside formulation (Eq. 2).

    Gotchas:
    - Do NOT add a classifier here — it will collapse the representation.
    - ClassBalancedSampler is required; without it, singleton classes
      in batches have no positives and the supervised term is zero.
    - projection_dim=128 (not the backbone feature dim) is the target for
      downstream evaluation comparisons with SimCLR.
    - SupConLoss normalizes features internally; do not pre-normalize.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        self.backbone, self.feat_dim = build_backbone(cfg.backbone, cfg.pretrained)
        supcon_cfg = cfg.supcon or SupConConfig()
        self.projector = self.build_projector(supcon_cfg)
        self.loss_fn = SupConLoss(temperature=supcon_cfg.temperature)
        # No self.classifier — intentional for stage 1

    def build_projector(self, supcon_cfg: SupConConfig | None = None) -> nn.Module:
        """2-layer MLP projection head (same architecture as SimCLR v1).

        Architecture: feat_dim -> 2048 (BN+ReLU) -> projection_dim (BN only).
        """
        if supcon_cfg is None:
            supcon_cfg = self.cfg.supcon or SupConConfig()
        return ProjectionHead(
            self.feat_dim,
            hidden_dim=2048,
            output_dim=supcon_cfg.projection_dim,
            num_layers=2,
        )

    @classmethod
    def build_augmentation(cls, size: int = 224) -> ContrastiveAugmentation:
        """Strong augmentation — same as SimCLR v1 (s=1.0 color jitter, blur)."""
        return ContrastiveAugmentation(size=size, strong=True)

    @property
    def learnable_params(self) -> list:
        """All parameters (backbone + projector) are trained in stage 1."""
        return [
            {"params": self.backbone.parameters()},
            {"params": self.projector.parameters()},
        ]

    def training_step(self, batch, batch_idx):
        """Encode two views, compute SupCon loss with class labels.

        Args:
            batch: Tuple of (views, labels) from ssl_collate_fn.
                   views shape: [2, B, C, H, W], labels shape: [B].
            batch_idx: Current batch index.

        Returns:
            Scalar SupCon loss.
        """
        views, labels = batch  # views: [2, B, C, H, W]
        h_i = self.backbone(views[0])  # [B, feat_dim]
        h_j = self.backbone(views[1])  # [B, feat_dim]
        z_i = self.projector(h_i)       # [B, projection_dim]
        z_j = self.projector(h_j)       # [B, projection_dim]
        loss = self.loss_fn(z_i, z_j, labels=labels)
        self.log_train_metrics(loss)
        return loss

    @classmethod
    def build_datamodule(cls, cfg: TrainConfig) -> SSLDataModule:
        """Build SSLDataModule with ClassBalancedSampler for SupCon stage 1.

        Reads n_classes_per_batch and n_samples_per_class from cfg.supcon.
        Effective batch_size ~= n_classes_per_batch * n_samples_per_class.
        """
        supcon_cfg = cfg.supcon or SupConConfig()
        return SSLDataModule(
            data_dir=cfg.data_dir,
            n_views=2,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            strong=True,
            sampler_type="class_balanced",
            n_classes_per_batch=supcon_cfg.n_classes_per_batch,
        )


class SupConFinetuneModule(BaseSSLModule):
    """SupCon stage-2: frozen backbone + linear head trained with SGD.

    Loads a stage-1 SupConModule checkpoint, freezes the backbone, and
    trains a single linear layer with cross-entropy and SGD (weight_decay=0.0).
    The projector weights from stage 1 are discarded — the classifier takes
    the backbone's raw feature vector h (not the projection z).

    Two-stage workflow:
        Stage 1 (pretraining):
            python train.py --config configs/supcon_stage1_resnet18.yaml

        Stage 2 (fine-tuning — this module):
            python train.py --config configs/supcon_stage2_resnet18.yaml \\
                --ckpt_path logs/supcon/version_0/checkpoints/last.ckpt

    Loading stage-1 backbone into stage-2 module:
        module = SupConFinetuneModule(cfg)
        state = torch.load(ckpt_path, map_location='cpu')['state_dict']
        # Keep only backbone.* keys; projector.* are discarded
        backbone_state = {k: v for k, v in state.items() if k.startswith('backbone.')}
        module.load_state_dict(backbone_state, strict=False)
        module.freeze_backbone()

    Gotchas:
    - SGD with weight_decay=0.0 is correct for linear probe. Do NOT use LARS
      or AdamW with nonzero weight decay — it regularizes away signal.
    - The backbone must be frozen AFTER loading the checkpoint, not before.
    - Use only one view during fine-tuning (no multi-view needed for CE loss).
    - BN layers in the backbone remain in eval() mode (set by freeze_backbone).
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        self.backbone, self.feat_dim = build_backbone(cfg.backbone, cfg.pretrained)
        supcon_cfg = cfg.supcon or SupConConfig()
        # Linear head: backbone_features -> num_classes
        self.classifier = nn.Linear(self.feat_dim, supcon_cfg.num_classes)
        # Projector not used in stage 2; build_projector required by abstract base
        # Implementation filled in Plan 4 (configure_optimizers override)

    def build_projector(self) -> nn.Module:
        """Not used in stage 2 (classifier operates on h, not z)."""
        return nn.Identity()

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters and set BN layers to eval mode.

        Call this AFTER loading the stage-1 checkpoint:
            module.load_state_dict(backbone_state, strict=False)
            module.freeze_backbone()
        """
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        # Keep BN in eval mode so running stats aren't updated
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()

    @property
    def learnable_params(self) -> list:
        """Only the linear classifier is trained in stage 2."""
        return [{"params": self.classifier.parameters()}]

    def training_step(self, batch, batch_idx):
        """Forward pass with frozen backbone — CE loss on linear head.

        Args:
            batch: Tuple of (views, labels). Uses views[0] only.
            batch_idx: Current batch index.

        Returns:
            Scalar cross-entropy loss.
        """
        views, labels = batch
        # Use first view only — no multi-view augmentation needed for linear probe
        x = views[0] if views.dim() == 5 else views  # handle [2,B,C,H,W] or [B,C,H,W]
        with torch.no_grad():
            h = self.backbone(x)  # [B, feat_dim] — backbone is frozen
        logits = self.classifier(h)  # [B, num_classes]
        loss = F.cross_entropy(logits, labels)
        self.log_train_metrics(loss)
        return loss

    def configure_optimizers(self):
        """SGD optimizer with weight_decay=0.0 for linear probe.

        Overrides BaseSSLModule.configure_optimizers() to use plain SGD
        without LARS or warmup. weight_decay=0.0 is required — the linear
        probe should not be regularized.
        """
        optimizer = torch.optim.SGD(
            [p for g in self.learnable_params for p in g["params"]],
            lr=self.cfg.lr,
            momentum=0.9,
            weight_decay=0.0,
        )
        return optimizer

    @classmethod
    def from_stage1_ckpt(
        cls,
        ckpt_path: str,
        cfg: TrainConfig,
        map_location: str = "cpu",
    ) -> "SupConFinetuneModule":
        """Construct a stage-2 module from a stage-1 SupConModule checkpoint.

        Only backbone.* weights are loaded from the checkpoint; projector.*
        weights are discarded (the classifier operates on backbone output h,
        not projection z).

        Args:
            ckpt_path: Path to the Lightning checkpoint from stage-1 training.
            cfg: TrainConfig for stage-2 (must have cfg.supcon.num_classes set).
            map_location: torch.load map_location. Default: 'cpu'.

        Returns:
            SupConFinetuneModule with backbone loaded from stage-1 checkpoint
            and backbone frozen.

        Example::

            module = SupConFinetuneModule.from_stage1_ckpt(
                ckpt_path="logs/supcon/version_0/checkpoints/last.ckpt",
                cfg=cfg,
            )
            trainer.fit(module, datamodule=dm)
        """
        module = cls(cfg)
        state = torch.load(ckpt_path, map_location=map_location)

        # Lightning checkpoints wrap state_dict under 'state_dict' key
        raw_state = state.get("state_dict", state)

        # Extract only backbone.* keys
        backbone_state = {
            k: v for k, v in raw_state.items()
            if k.startswith("backbone.")
        }

        if not backbone_state:
            raise ValueError(
                f"No 'backbone.*' keys found in checkpoint at {ckpt_path}. "
                "Ensure the checkpoint is from a SupConModule stage-1 run."
            )

        missing, unexpected = module.load_state_dict(backbone_state, strict=False)
        # Report any unexpected keys (projector.* missing from strict=False is expected)
        backbone_unexpected = [k for k in unexpected if k.startswith("backbone.")]
        if backbone_unexpected:
            import warnings
            warnings.warn(
                f"Unexpected backbone keys in checkpoint: {backbone_unexpected}"
            )

        module.freeze_backbone()
        return module

    def validation_step(self, batch, batch_idx):
        """Compute validation accuracy for stage-2 fine-tuning monitoring.

        Args:
            batch: Tuple of (views, labels). Single view expected.
            batch_idx: Current batch index.
        """
        views, labels = batch
        x = views[0] if views.dim() == 5 else views
        with torch.no_grad():
            h = self.backbone(x)
        logits = self.classifier(h)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        return loss
