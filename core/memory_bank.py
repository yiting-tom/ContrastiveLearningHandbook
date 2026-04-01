"""Feature memory bank for instance-level contrastive learning.

Stores L2-normalized feature vectors indexed by dataset sample index.  Used by
Instance Discrimination (Wu et al., CVPR 2018) and CMC (Tian et al., ECCV 2020)
to supply negative samples without recomputing the full dataset every step.

Staleness gotcha: Features stored in the bank come from earlier encoder
snapshots. As the encoder trains, stored features become stale -- they no longer
match what the current encoder would produce. This means negative samples drawn
from the bank are softer negatives than fresh encoder outputs would be. MoCo
(He et al., CVPR 2020) addresses this with a FIFO queue fed by a slowly-moving
momentum encoder, ensuring all negatives come from recent (though not current)
encoder states.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    """L2-normalized feature memory bank backed by ``nn.Embedding``.

    Stores one feature vector per dataset sample.  Vectors are always kept
    L2-normalized so that cosine similarity reduces to a dot product.

    Staleness gotcha: Features stored in the bank come from earlier encoder
    snapshots. As the encoder trains, stored features become stale -- they no
    longer match what the current encoder would produce. This means negative
    samples drawn from the bank are softer negatives than fresh encoder outputs
    would be. MoCo (He et al., CVPR 2020) addresses this with a FIFO queue fed
    by a slowly-moving momentum encoder, ensuring all negatives come from recent
    (though not current) encoder states.

    Args:
        n_samples: Number of dataset samples (bank rows).
        dim: Feature dimensionality (bank columns).
    """

    def __init__(self, n_samples: int, dim: int) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.dim = dim
        self.bank = nn.Embedding(n_samples, dim)
        nn.init.normal_(self.bank.weight.data)
        self.bank.weight.data = F.normalize(self.bank.weight.data, dim=1)
        self.bank.weight.requires_grad = False

    def get(self, indices: torch.Tensor) -> torch.Tensor:
        """Retrieve feature vectors for the given sample indices.

        Args:
            indices: 1-D integer tensor of sample indices.

        Returns:
            L2-normalized feature matrix of shape ``(len(indices), dim)``.
        """
        return self.bank(indices)

    @torch.no_grad()
    def update(self, indices: torch.Tensor, features: torch.Tensor) -> None:
        """Write new feature vectors into the bank (L2-normalized).

        Args:
            indices: 1-D integer tensor of sample indices to update.
            features: Feature matrix of shape ``(len(indices), dim)``.
                Will be detached and L2-normalized before storage.
        """
        normalized = F.normalize(features.detach(), dim=1)
        self.bank.weight.data[indices] = normalized
