"""SwAV (Caron et al., NeurIPS 2020).

[DOC-02 docstring added in Plan 07]
"""
from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.backbone import build_backbone
from core.base import BaseSSLModule
from core.config import SwAVConfig, TrainConfig
from core.projection import ProjectionHead
from methods.swav.losses import sinkhorn_knopp, swav_loss
from methods.swav.prototype import PrototypeLayer


class SwAVModule(BaseSSLModule):
    """SwAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        swav_cfg = cfg.swav or SwAVConfig()

        # Backbone
        self.backbone, self.feat_dim = build_backbone(cfg.backbone, cfg.pretrained)

        # Projector: 2-layer MLP, 128-dim output (standard SwAV)
        self.projector = self.build_projector()

        # Prototype layer
        self.prototype_layer = PrototypeLayer(
            feat_dim=128,  # projection output dim
            n_prototypes=swav_cfg.n_prototypes,
        )

        # Hyper-parameters stored for training_step
        self.n_large_crops = swav_cfg.n_large_crops
        self.temperature = swav_cfg.temperature
        self.sinkhorn_iters = swav_cfg.sinkhorn_iterations
        self.epsilon = swav_cfg.epsilon
        self.freeze_prototypes_epochs = swav_cfg.freeze_prototypes_epochs

    def build_projector(self) -> nn.Module:
        return ProjectionHead(self.feat_dim, 2048, 128, num_layers=2)

    @property
    def learnable_params(self):
        """Include backbone + projector + prototype layer parameters."""
        import itertools
        return itertools.chain(
            self.backbone.parameters(),
            self.projector.parameters(),
            self.prototype_layer.parameters(),
        )

    def training_step(self, batch, batch_idx):
        crops_list, labels = batch  # crops_list: list of n_crops tensors

        # Encode all crops through backbone + projector
        z_list = []
        for crop in crops_list:
            h = self.backbone(crop)
            z = self.projector(h)
            z_list.append(z)

        # Compute swapped prediction loss
        sinkhorn_fn = partial(
            sinkhorn_knopp,
            n_iters=self.sinkhorn_iters,
            epsilon=self.epsilon,
        )
        loss = swav_loss(
            z_list=z_list,
            prototype_layer=self.prototype_layer,
            temperature=self.temperature,
            n_large_crops=self.n_large_crops,
            sinkhorn_fn=sinkhorn_fn,
        )
        self.log_train_metrics(loss)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """L2-normalize prototype vectors after optimizer step (per D-06)."""
        super().on_train_batch_end(outputs, batch, batch_idx)
        self.prototype_layer.normalize_prototypes()

    def on_before_optimizer_step(self, optimizer):
        """Zero prototype gradients during freeze epochs (per D-07)."""
        if PrototypeLayer.should_freeze_prototypes(
            self.current_epoch, self.freeze_prototypes_epochs
        ):
            self.prototype_layer.zero_prototype_gradients()
