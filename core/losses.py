"""InfoNCE loss for contrastive self-supervised learning.

Supports:
- Symmetric in-batch mode (SimCLR, MoCo v3, Instance Discrimination, etc.)
- Asymmetric queue mode (MoCo v1/v2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE (Noise Contrastive Estimation) loss.

    Covers symmetric in-batch (SimCLR) and asymmetric queue (MoCo) modes.

    Args:
        temperature: Softmax temperature scaling factor. Lower values produce
            sharper distributions. Default: 0.5.
        reduction: Reduction to apply to the output. One of 'mean' or 'sum'.
            Default: 'mean'.

    References:
        - van den Oord et al., "Representation Learning with Contrastive Predictive Coding"
          https://arxiv.org/abs/1807.03748
        - Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations"
          https://arxiv.org/abs/2002.05709 (symmetric mode / SimCLR)
        - He et al., "Momentum Contrast for Unsupervised Visual Representation Learning"
          https://arxiv.org/abs/1911.05722 (asymmetric queue mode / MoCo)
    """

    def __init__(self, temperature: float = 0.5, reduction: str = "mean") -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        queue: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            z_i: Embeddings from view 1, shape [B, D].
            z_j: Embeddings from view 2 (or positive keys), shape [B, D].
            queue: Optional negative key buffer for asymmetric mode, shape [D, K].
                   When None, uses symmetric in-batch mode. When provided, uses
                   asymmetric MoCo-style mode with queue as additional negatives.

        Returns:
            Scalar loss tensor.
        """
        # Always L2-normalize inputs for numerical stability
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        if queue is None:
            return self._symmetric_loss(z_i, z_j)
        else:
            return self._asymmetric_loss(z_i, z_j, queue)

    def _symmetric_loss(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Symmetric in-batch InfoNCE (SimCLR / NT-Xent style).

        Concatenates both views and treats each sample's counterpart view as
        the positive. All other 2B-2 samples are negatives.
        """
        B = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
        sim = z @ z.T / self.temperature   # [2B, 2B]

        # Mask self-similarity (diagonal) with -inf so softmax ignores it
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, float("-inf"))

        # Positive pair for z_i[k] is z_j[k] (offset by B), and vice versa
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])

        # F.cross_entropy handles log-sum-exp internally (Pitfall 7: numerical stability)
        return F.cross_entropy(sim, labels, reduction=self.reduction)

    def _asymmetric_loss(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        queue: torch.Tensor,
    ) -> torch.Tensor:
        """Asymmetric InfoNCE with negative queue (MoCo style).

        Args:
            z_i: Query embeddings, shape [B, D].
            z_j: Positive key embeddings, shape [B, D].
            queue: Negative key buffer, shape [D, K].
        """
        # Positive logits: dot product between query and its positive key
        l_pos = (z_i * z_j).sum(dim=1, keepdim=True) / self.temperature  # [B, 1]
        # Negative logits: query against all queue keys
        l_neg = z_i @ queue / self.temperature  # [B, K]
        # Concatenate: class 0 is the positive, classes 1..K are negatives
        logits = torch.cat([l_pos, l_neg], dim=1)  # [B, 1+K]
        labels = torch.zeros(z_i.shape[0], dtype=torch.long, device=z_i.device)

        return F.cross_entropy(logits, labels, reduction=self.reduction)
