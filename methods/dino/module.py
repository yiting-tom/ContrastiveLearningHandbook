"""DINO: Emerging Properties in Self-Supervised Vision Transformers (Caron et al., ICCV 2021).

Student-teacher self-distillation where the teacher is an EMA of the student.
No labels, no contrastive negatives — collapse is prevented by (1) centering of
teacher outputs (momentum-averaged batch mean) and (2) teacher temperature sharpening.

Paper: "Emerging Properties in Self-Supervised Vision Transformers"
Authors: Mathilde Caron, Hugo Touvron, Ishan Misra, Herve Jegou, Julien Mairal,
         Piotr Bojanowski, Armand Joulin
Venue: ICCV 2021
arXiv: https://arxiv.org/abs/2104.14294
Reference implementation: https://github.com/facebookresearch/dino

Algorithm:
1. Student processes ALL crops (global + local) through backbone -> projector -> prototype layer.
2. Teacher (EMA of student) processes ONLY global crops (first 2) through backbone_ema -> projector_ema -> prototype_layer_ema.
3. Teacher outputs are CENTERED (subtract momentum-averaged batch mean) then SHARPENED (low temperature softmax).
4. Cross-entropy loss between teacher probabilities and student log-softmax predictions.
5. Same-view pairs are skipped (student view i vs teacher view i).
6. EMA update: teacher network = m*teacher + (1-m)*student after each batch.

Gotchas:
- Centering MUST be updated BEFORE loss computation each step. Applying the old center
  to compute loss (and updating center after) leads to instability.
- Teacher temperature warmup is critical (start at 0.07, end at 0.04 after warmup_epochs).
  Starting cold (0.04) causes early collapse; warmup stabilizes initial training.
- Teacher receives GLOBAL CROPS ONLY (first n_global in crops_list). Feeding local crops
  to the teacher reduces representation quality.
- Prototype output dim 65536 matters — smaller dims (~1000) lead to significantly
  worse downstream accuracy with ViT backbones.
- Gradient clipping (max_norm=3.0) is required for stable ViT training.
- Teacher has NO predictor. The asymmetry (student has backbone+projector+prototype,
  teacher has same but via EMA) is not from a predictor — it's from the EMA itself.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.backbone import build_backbone
from core.base import BaseSSLModule
from core.config import DINOConfig, TrainConfig
from core.ema import EMAUpdater
from core.projection import ProjectionHead


class DINOModule(BaseSSLModule):
    """DINO: Self-distillation with student-teacher networks and centering (Caron et al., ICCV 2021).

    Architecture:
    - Student (online) network: backbone -> projector (3-layer MLP, 256-dim) -> prototype_layer (256->65536)
    - Teacher (EMA) network: backbone_ema -> projector_ema -> prototype_layer_ema (same dims, no grad)

    Collapse prevention:
    - Centering: teacher logits are shifted by a momentum-updated batch mean (``center`` buffer).
    - Sharpening: teacher uses low temperature (0.04) for sharper softmax distributions.
    - EMA: teacher slowly tracks student; avoids trivial constant collapse.

    Multi-crop:
    - Teacher forward: first ``n_global`` crops only (default 2).
    - Student forward: all crops (global + local).

    Args:
        cfg: TrainConfig with cfg.dino populated (or default DINOConfig).
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        dino_cfg: DINOConfig = cfg.dino or DINOConfig()

        # Store DINO hyper-parameters
        self.n_prototypes: int = dino_cfg.n_prototypes
        self.teacher_temp: float = dino_cfg.teacher_temp
        self.warmup_teacher_temp: float = dino_cfg.warmup_teacher_temp
        self.warmup_teacher_temp_epochs: int = dino_cfg.warmup_teacher_temp_epochs
        self.student_temp: float = dino_cfg.student_temp
        self.centering_momentum: float = dino_cfg.centering_momentum

        # Student (online) network
        self.backbone, self.feat_dim = build_backbone(cfg.backbone, cfg.pretrained)
        self.projector = self.build_projector()
        self.prototype_layer = nn.Linear(256, self.n_prototypes, bias=False)

        # Teacher (EMA) network — deep copy, freeze all params
        self.backbone_ema = copy.deepcopy(self.backbone)
        self.projector_ema = copy.deepcopy(self.projector)
        self.prototype_layer_ema = copy.deepcopy(self.prototype_layer)
        self.backbone_ema.requires_grad_(False)
        self.projector_ema.requires_grad_(False)
        self.prototype_layer_ema.requires_grad_(False)

        # Centering buffer — updated BEFORE loss computation each step (D-02)
        self.register_buffer("center", torch.zeros(self.n_prototypes))

        # EMA updater — initialized in setup() once total_steps is known from trainer
        self._dino_base_momentum: float = 0.996
        self._dino_end_momentum: float = 1.0
        self.ema: EMAUpdater | None = None

    def build_projector(self) -> nn.Module:
        """3-layer projection head (DINO: feat_dim->2048->256, per D-06)."""
        return ProjectionHead(
            input_dim=self.feat_dim,
            hidden_dim=2048,
            output_dim=256,
            num_layers=3,
        )

    def setup(self, stage: str) -> None:
        """Initialize EMA updater once total_steps is known from the trainer.

        Follows BYOL setup pattern with cosine-scheduled momentum 0.996 -> 1.0.
        """
        if stage == "fit" and self.trainer is not None:
            total_steps = self.trainer.estimated_stepping_batches
            self.ema = EMAUpdater(
                base_momentum=self._dino_base_momentum,
                end_momentum=self._dino_end_momentum,
                total_steps=int(total_steps),
            )

    @property
    def learnable_params(self):
        """Exclude teacher network from optimizer.

        Chains backbone, projector, and prototype_layer parameters.
        Teacher (EMA) parameters have requires_grad=False and are explicitly excluded.
        """
        import itertools
        return itertools.chain(
            self.backbone.parameters(),
            self.projector.parameters(),
            self.prototype_layer.parameters(),
        )

    def _get_teacher_temp(self) -> float:
        """Compute teacher temperature with linear warmup.

        Warms up from ``teacher_temp`` (0.04) to ``warmup_teacher_temp`` (0.07)
        over ``warmup_teacher_temp_epochs`` epochs, then stays at ``teacher_temp``.

        Note: The warmup starts HIGH (0.07) and ends LOW (0.04) — counterintuitively,
        this is correct per the DINO paper. High teacher temperature early = softer
        distributions = more stable initial training.
        """
        epoch = self.current_epoch
        warmup_epochs = self.warmup_teacher_temp_epochs
        if warmup_epochs > 0 and epoch < warmup_epochs:
            # Linear interpolation: start at warmup_teacher_temp (0.07), end at teacher_temp (0.04)
            alpha = epoch / warmup_epochs
            return self.warmup_teacher_temp + alpha * (self.teacher_temp - self.warmup_teacher_temp)
        return self.teacher_temp

    def training_step(self, batch, batch_idx):
        """DINO cross-entropy loss between centered/sharpened teacher and student outputs.

        Teacher sees global crops only; student sees all crops.
        Centering is updated BEFORE loss computation (D-02).

        Args:
            batch: Tuple of (crops_list, labels) where crops_list is a list of tensors
                of shape [B, C, H, W] (first n_global are global crops, rest are local).
            batch_idx: Index of the current batch.

        Returns:
            Scalar loss tensor.
        """
        crops_list, labels = batch

        # Determine number of global crops (default 2, but handle 2-view case)
        n_global = 2 if len(crops_list) > 2 else len(crops_list)

        # Teacher forward (no grad) -- global crops ONLY
        with torch.no_grad():
            teacher_outs = []
            for crop in crops_list[:n_global]:
                h = self.backbone_ema(crop)
                z = self.projector_ema(h)
                z = F.normalize(z, dim=-1)
                logits = self.prototype_layer_ema(z)  # [B, n_prototypes]
                teacher_outs.append(logits)
            teacher_out_cat = torch.cat(teacher_outs, dim=0)  # [n_global*B, n_prototypes]

            # Update center BEFORE loss computation (D-02 — critical ordering)
            self.center = (
                self.centering_momentum * self.center
                + (1 - self.centering_momentum) * teacher_out_cat.mean(dim=0)
            )

            # Get current teacher temperature (with warmup)
            t_temp = self._get_teacher_temp()

            # Centering + sharpening: subtract center, then sharpen with low temperature
            teacher_probs_list = []
            for t_out in teacher_outs:
                probs = F.softmax((t_out - self.center) / t_temp, dim=-1)
                teacher_probs_list.append(probs)

        # Student forward -- ALL crops (global + local)
        total_loss = 0.0
        n_loss_terms = 0
        for i, crop in enumerate(crops_list):
            h = self.backbone(crop)
            z = self.projector(h)
            z = F.normalize(z, dim=-1)
            student_logits = self.prototype_layer(z)  # [B, n_prototypes]
            student_log_probs = F.log_softmax(student_logits / self.student_temp, dim=-1)

            # Cross-entropy with each global teacher output; skip same-view pairs
            for j, t_probs in enumerate(teacher_probs_list):
                if i == j:
                    continue  # skip same-view pairs
                total_loss = total_loss + (-torch.sum(t_probs * student_log_probs, dim=-1).mean())
                n_loss_terms += 1

        loss = total_loss / max(n_loss_terms, 1)

        self.log_train_metrics(loss)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """EMA update of teacher network after each optimizer step.

        Updates backbone_ema, projector_ema, and prototype_layer_ema toward
        their student counterparts using cosine-scheduled momentum.
        """
        if self.ema is not None:
            self.ema.step(self.backbone.parameters(), self.backbone_ema.parameters())
            self.ema.step(self.projector.parameters(), self.projector_ema.parameters())
            self.ema.step(
                self.prototype_layer.parameters(), self.prototype_layer_ema.parameters()
            )
