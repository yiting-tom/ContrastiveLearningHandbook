"""FIFO negative-key queue for momentum contrastive learning.

Implements the queue component from MoCo (He et al., CVPR 2020).  The momentum
encoder produces key representations that are enqueued into a fixed-size FIFO
buffer.  At each training step the queue supplies a large pool of negative keys
for the InfoNCE loss without requiring a large batch size.

Key properties:
  - Fixed size K: after filling, the queue always contains exactly K keys.
  - FIFO order: oldest keys are overwritten first (circular buffer via pointer).
  - L2-normalized storage: all keys are normalized before enqueueing so that
    cosine similarity reduces to a dot product.
  - Checkpoint safe: queue and pointer are ``register_buffer`` tensors that
    survive ``state_dict`` save / load.

Reference:
    He et al., "Momentum Contrast for Unsupervised Visual Representation
    Learning", CVPR 2020.  arXiv: https://arxiv.org/abs/1911.05722
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MomentumQueue(nn.Module):
    """Fixed-size FIFO queue of L2-normalized key vectors.

    Stores keys in a tensor of shape ``[dim, queue_size]`` (columns are
    individual keys) so that computing logits via matrix multiply is a single
    ``torch.mm(query, queue)`` call — matching the convention used by
    ``InfoNCELoss._asymmetric_loss``.

    Args:
        queue_size: Maximum number of keys stored (K).
        dim: Dimensionality of each key vector (D).
    """

    def __init__(self, queue_size: int, dim: int) -> None:
        super().__init__()
        self.queue_size = queue_size
        self.dim = dim

        # Initialize with L2-normalized random vectors
        queue = torch.randn(dim, queue_size)
        queue = F.normalize(queue, dim=0)
        self.register_buffer("queue", queue)
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update(self, keys: torch.Tensor) -> None:
        """Enqueue a batch of key vectors (FIFO).

        Keys are detached and L2-normalized before storage.  If the batch
        straddles the end of the buffer the write wraps around to the
        beginning.

        Args:
            keys: Key matrix of shape ``(batch_size, dim)``.
        """
        keys = F.normalize(keys.detach(), dim=1)
        batch_size = keys.shape[0]
        ptr = int(self.ptr.item())

        if ptr + batch_size <= self.queue_size:
            # Simple case: entire batch fits without wrapping
            self.queue[:, ptr : ptr + batch_size] = keys.T
        else:
            # Split write across buffer boundary
            tail = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:tail].T
            self.queue[:, : batch_size - tail] = keys[tail:].T

        self.ptr[0] = (ptr + batch_size) % self.queue_size

    def get_negatives(self) -> torch.Tensor:
        """Return a detached clone of the queue.

        Returns:
            Tensor of shape ``[dim, queue_size]`` (D x K).  The returned
            tensor is a clone — mutations do not affect the internal buffer.
        """
        return self.queue.clone().detach()
