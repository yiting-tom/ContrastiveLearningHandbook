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


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Extends SimCLR's NT-Xent loss to the labeled setting: every in-batch
    sample sharing the same class label is treated as a positive for an anchor,
    not just the other augmented view.

    Uses the **sum-outside** formulation (Eq. 2 of the paper):

        L_i = -1/|P(i)| * sum_{p in P(i)} [s_{ip}/tau - log sum_{a != i} exp(s_{ia}/tau)]

    When ``labels=None``, degenerates exactly to SimCLR NT-Xent: the only
    positive for anchor z_i[k] is z_j[k] (and vice versa), giving one
    positive per anchor.

    Paper: "Supervised Contrastive Learning"
    Authors: Prannay Khosla, Piotr Tian, Chen Wang, Aaron Neimark,
             Piyush Rai, Chen Xu, Dilip Krishnan, Serge Belongie
    Venue: NeurIPS 2020
    arXiv: https://arxiv.org/abs/2004.11362

    Args:
        temperature: Softmax temperature. Default: 0.07 (paper recommendation).
        reduction: 'mean' or 'sum'. Default: 'mean'.

    Gotchas:
    - Use sum-outside (this implementation), NOT sum-inside (Eq. 1). Eq. 2 is
      empirically stronger and is the recommended variant.
    - Features are L2-normalized inside forward(); do NOT pre-normalize inputs.
    - With temperature=0.07, logits scale by ~14x — never use raw exp without
      logsumexp stabilization (handled here via torch.logsumexp).
    - Anchors whose class has no other in-batch sample (singleton class) have
      zero positives and are excluded from the mean to avoid dividing by zero.
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean") -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute SupCon loss.

        Args:
            z_i: Projection outputs from view 1, shape [B, D].
            z_j: Projection outputs from view 2, shape [B, D].
            labels: Integer class labels, shape [B]. When None, degenerates
                    to SimCLR NT-Xent (one positive per anchor).

        Returns:
            Scalar loss tensor.
        """
        B = z_i.shape[0]
        device = z_i.device

        # L2-normalize both views (matching InfoNCELoss convention)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Stack to [2B, D]: first B rows = view-i, last B rows = view-j
        features = torch.cat([z_i, z_j], dim=0)  # [2B, D]

        # --- Build positive mask [2B, 2B] ---
        self_mask = torch.eye(2 * B, dtype=torch.bool, device=device)

        if labels is None:
            # SimCLR mode: one positive per anchor (the other view)
            positive_mask = torch.zeros(2 * B, 2 * B, dtype=torch.bool, device=device)
            for k in range(B):
                positive_mask[k, B + k] = True
                positive_mask[B + k, k] = True
        else:
            # Supervised mode: all same-class anchors (excluding self)
            labels_2v = labels.repeat(2)  # [2B]
            label_eq = labels_2v.unsqueeze(0) == labels_2v.unsqueeze(1)  # [2B, 2B]
            positive_mask = label_eq & ~self_mask

        # --- Similarity matrix [2B, 2B] scaled by temperature ---
        sim = features @ features.T / self.temperature  # [2B, 2B]

        # Exclude self from denominator by masking diagonal to -inf
        sim_no_self = sim.masked_fill(self_mask, float("-inf"))

        # log( sum_{a != i} exp(sim[i,a] / tau) ) — stable via logsumexp
        log_denom = torch.logsumexp(sim_no_self, dim=1)  # [2B]

        # log-probability of each positive pair for anchor i
        log_prob = sim - log_denom.unsqueeze(1)  # [2B, 2B]

        # Per-anchor loss: -mean over positives (sum-outside: division outside log)
        n_positives = positive_mask.sum(dim=1).float()  # [2B]
        valid = n_positives > 0  # exclude singleton-class anchors

        # Sum log-prob over positives; divide by |P(i)| (outside the log)
        per_anchor_loss = -(log_prob * positive_mask).sum(dim=1) / n_positives.clamp(min=1)

        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        if self.reduction == "mean":
            return per_anchor_loss[valid].mean()
        else:
            return per_anchor_loss[valid].sum()
