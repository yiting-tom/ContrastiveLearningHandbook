"""Prototype layer for SwAV with L2-normalization and epoch-based gradient freezing.

Paper: SwAV — "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments"
Authors: Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, Armand Joulin
Venue: NeurIPS 2020
arXiv: https://arxiv.org/abs/2006.09882

Design decisions (from D-06, D-07, D-09):

  L2-renormalization (D-06):
    ``normalize_prototypes()`` is called in ``on_train_batch_end`` (after
    ``optimizer.step()``).  Normalization MUST happen post-step so that
    the optimizer update is applied first; renormalizing beforehand would
    interfere with gradient flow.

  Gradient freezing (D-07):
    ``zero_prototype_gradients()`` is called inside
    ``on_before_optimizer_step`` (before ``optimizer.step()``).  During
    ``freeze_prototypes_epochs`` (default 1), the prototype gradients are
    zeroed so the optimizer step has no effect on prototype weights.
    This lets the encoder warm up before the prototypes start moving.

  Trainable linear layer (D-09):
    Prototypes are represented as the weight matrix of an
    ``nn.Linear(feat_dim, n_prototypes, bias=False)`` module.  Setting
    ``bias=False`` ensures the forward pass computes pure dot products
    (cosine similarities when both inputs and weights are L2-normalized).

Gotchas:
  - Always call ``normalize_prototypes()`` from ``on_train_batch_end``
    in the host ``SwAVModule``, NOT before the loss computation.
  - The prototype weight rows, not columns, correspond to individual
    prototypes.  ``F.normalize(..., dim=1)`` normalizes along
    ``feat_dim`` (i.e., each row independently).
  - ``zero_prototype_gradients()`` is a no-op when ``grad`` is None
    (e.g., at the very first step before any backward pass).

Reference implementation:
  https://github.com/facebookresearch/swav
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeLayer(nn.Module):
    """Learnable prototype layer with L2-normalization and freeze helpers.

    The prototype weight matrix is ``[n_prototypes, feat_dim]``.  On
    construction the rows are initialized with uniform random values then
    immediately L2-normalized so that the initial cosine similarities are
    well-conditioned.

    Args:
        feat_dim: Dimensionality of the (already L2-normalized) feature
            vectors produced by the projector head.
        n_prototypes: Number of prototype vectors (cluster centroids) K.
            Typical values: 3000 (CIFAR) to 3000–65536 (ImageNet).

    Example::

        proto = PrototypeLayer(feat_dim=128, n_prototypes=3000)
        scores = proto(z_normalized)           # [B, 3000]
        q = sinkhorn_knopp(scores)             # soft assignments

        # In on_before_optimizer_step:
        if PrototypeLayer.should_freeze_prototypes(epoch, cfg.freeze_prototypes_epochs):
            proto.zero_prototype_gradients()

        # In on_train_batch_end:
        proto.normalize_prototypes()
    """

    def __init__(self, feat_dim: int, n_prototypes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feat_dim, n_prototypes, bias=False)
        # Initialize with uniform random, then L2-normalize each row so
        # initial prototype directions are meaningful (not magnitude-dominated).
        nn.init.uniform_(self.linear.weight)
        with torch.no_grad():
            self.linear.weight.copy_(
                F.normalize(self.linear.weight.data, dim=1, p=2)
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute prototype scores for a batch of feature vectors.

        Args:
            x: Feature tensor of shape ``[B, feat_dim]``.  Should be
               L2-normalized before passing to this layer so that the
               scores are cosine similarities.

        Returns:
            Tensor of shape ``[B, n_prototypes]`` containing raw
            prototype scores (dot products).
        """
        return self.linear(x)

    # ------------------------------------------------------------------
    # Normalization helper — called from on_train_batch_end
    # ------------------------------------------------------------------

    def normalize_prototypes(self) -> None:
        """L2-normalize each prototype row to unit norm.

        Call this method in ``on_train_batch_end`` (after optimizer.step)
        so that prototype directions remain on the unit hypersphere
        throughout training.  Normalization is applied in-place via
        ``torch.no_grad()`` to avoid creating spurious computation graph
        nodes.
        """
        with torch.no_grad():
            self.linear.weight.copy_(
                F.normalize(self.linear.weight.data, dim=1, p=2)
            )

    # ------------------------------------------------------------------
    # Gradient freeze helper — called from on_before_optimizer_step
    # ------------------------------------------------------------------

    def zero_prototype_gradients(self) -> None:
        """Zero the prototype weight gradient (freeze effect).

        Call this method in ``on_before_optimizer_step`` when
        ``should_freeze_prototypes`` returns True.  Zeroing the gradient
        before the optimizer step means the optimizer update has no effect
        on the prototype weights for that step.

        This is a no-op if the gradient tensor is None (e.g., during the
        very first backward pass before any accumulation).
        """
        if self.linear.weight.grad is not None:
            self.linear.weight.grad.zero_()

    # ------------------------------------------------------------------
    # Epoch-based freeze gate — pure logic, no state
    # ------------------------------------------------------------------

    @staticmethod
    def should_freeze_prototypes(current_epoch: int, freeze_epochs: int) -> bool:
        """Return True if prototypes should be frozen this epoch.

        Prototypes are frozen during the first ``freeze_epochs`` epochs
        (epochs 0, 1, …, freeze_epochs-1) so the encoder representation
        can warm up before prototypes start moving.  Default for
        ``freeze_prototypes_epochs`` is 1 (only the very first epoch).

        Args:
            current_epoch: Zero-based current training epoch index
                (Lightning's ``self.current_epoch``).
            freeze_epochs: Number of initial epochs to freeze prototypes
                (``cfg.swav.freeze_prototypes_epochs``).

        Returns:
            True if ``current_epoch < freeze_epochs``, False otherwise.
        """
        return current_epoch < freeze_epochs
