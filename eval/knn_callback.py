"""KNNCallback: In-training k-NN evaluation via the DINO/MoCo v3 weighted k-NN protocol.

This module provides a Lightning Callback that computes weighted k-NN accuracy
(logged as ``eval/knn_acc``) at configurable epoch intervals. It uses FAISS
``IndexFlatIP`` for large datasets (>100K samples) and brute-force torch matmul
for smaller datasets.

Reference:
    - DINO: Caron et al., "Emerging Properties in Self-Supervised Vision Transformers",
      ICCV 2021. https://arxiv.org/abs/2104.14294
    - MoCo v3: Chen et al., "An Empirical Study of Training Self-Supervised Vision
      Transformers", ICCV 2021. https://arxiv.org/abs/2104.02057

Usage::

    from core.config import KNNConfig
    from eval.knn_callback import KNNCallback

    knn_cb = KNNCallback(KNNConfig(k=200, temperature=0.07, every_n_epochs=5))
    trainer = L.Trainer(callbacks=[knn_cb], ...)
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn.functional as F
import lightning as L

from core.config import KNNConfig

# macOS: FAISS ships its own OpenMP runtime which can conflict with PyTorch's.
# Setting KMP_DUPLICATE_LIB_OK=TRUE prevents a segfault on macOS ARM/Intel.
# This is a known macOS-specific workaround; safe for tutorial use.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Threshold for switching from brute-force torch matmul to FAISS IndexFlatIP.
# Exposed as a module-level constant so tests can patch it.
FAISS_THRESHOLD: int = 100_000


def knn_predict(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    k: int,
    temperature: float,
    num_classes: int,
) -> float:
    """Weighted k-NN classification following the DINO/MoCo v3 protocol.

    Both ``train_features`` and ``test_features`` must be **L2-normalized** before
    calling this function.  When ``train_features.shape[0] > FAISS_THRESHOLD``,
    FAISS ``IndexFlatIP`` is used; otherwise brute-force torch matmul is used.

    Args:
        train_features: Shape [N, D], L2-normalized float32 tensors.
        train_labels:   Shape [N], integer class labels.
        test_features:  Shape [M, D], L2-normalized float32 tensors.
        test_labels:    Shape [M], integer class labels (ground-truth for accuracy).
        k:              Number of nearest neighbours.
        temperature:    Temperature for scaling similarities before softmax-like
                        weighted voting.
        num_classes:    Number of distinct classes.

    Returns:
        Accuracy in [0, 1] as a Python float.
    """
    n_train = train_features.shape[0]
    m_test = test_features.shape[0]

    if n_train > FAISS_THRESHOLD:
        # ------------------------------------------------------------------
        # FAISS path — used for large datasets (>100K training samples)
        # Mitigation for T-09-03 (OOM on large k-NN)
        # ------------------------------------------------------------------
        import faiss  # local import: optional at module level

        dim = train_features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(
            np.ascontiguousarray(train_features.numpy().astype(np.float32))
        )
        raw_sims, raw_idx = index.search(
            np.ascontiguousarray(test_features.numpy().astype(np.float32)),
            k,
        )
        similarities = torch.from_numpy(raw_sims)   # [M, k]
        indices = torch.from_numpy(raw_idx.astype(np.int64))  # [M, k]
    else:
        # ------------------------------------------------------------------
        # Brute-force path — used for smaller datasets
        # ------------------------------------------------------------------
        similarities = test_features @ train_features.T  # [M, N]
        similarities, indices = similarities.topk(k, dim=1)  # [M, k] each

    # Temperature-scaled weighted voting
    weights = (similarities / temperature).exp()          # [M, k]
    neighbor_labels = train_labels[indices.long()]        # [M, k]

    # Accumulate votes per class
    votes = torch.zeros(m_test, num_classes)
    votes.scatter_add_(1, neighbor_labels.long(), weights)

    predicted = votes.argmax(dim=1)
    accuracy = (predicted == test_labels).sum().item() / m_test
    return float(accuracy)


class KNNCallback(L.Callback):
    """Lightning Callback that logs weighted k-NN accuracy at configured epochs.

    Computes k-NN accuracy using the DINO/MoCo v3 weighted voting protocol and
    logs the result as ``eval/knn_acc`` to TensorBoard (and any other loggers
    attached to the trainer).

    Args:
        knn_config: ``KNNConfig`` instance with ``k``, ``temperature``, and
            ``every_n_epochs`` fields.

    Example::

        from core.config import KNNConfig
        from eval.knn_callback import KNNCallback

        cb = KNNCallback(KNNConfig(k=200, temperature=0.07, every_n_epochs=5))
        trainer = L.Trainer(callbacks=[cb], ...)
    """

    def __init__(self, knn_config: KNNConfig) -> None:
        super().__init__()
        self.k = knn_config.k
        self.temperature = knn_config.temperature
        self.every_n_epochs = knn_config.every_n_epochs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_run(self, trainer: L.Trainer) -> bool:
        """Return True if k-NN should run this epoch.

        When ``every_n_epochs == 0``, k-NN runs only at the final epoch
        (``trainer.current_epoch == trainer.max_epochs - 1``).  Otherwise,
        k-NN runs whenever ``(current_epoch + 1) % every_n_epochs == 0``.
        """
        if self.every_n_epochs == 0:
            return trainer.current_epoch == trainer.max_epochs - 1
        return (trainer.current_epoch + 1) % self.every_n_epochs == 0

    def _extract_features(
        self,
        pl_module: L.LightningModule,
        dataloader,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Iterate a dataloader, extract backbone features, and L2-normalize.

        Handles both standard ``(imgs, labels)`` batches and multi-view SSL
        batches where the first element is a 5-D tensor
        ``[n_views, B, C, H, W]``.  In the multi-view case, only the first
        view is used.

        Args:
            pl_module: The Lightning module (provides ``backbone`` and ``device``).
            dataloader: Iterable of batches.

        Returns:
            Tuple of ``(features, labels)`` where:
                - ``features``: [N, D] float32, L2-normalized, on CPU.
                - ``labels``:   [N] long, on CPU.
        """
        all_features: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in dataloader:
                imgs = batch[0]
                labels = batch[1]

                # Handle multi-view SSL batches: shape [n_views, B, C, H, W]
                if isinstance(imgs, torch.Tensor) and imgs.ndim == 5:
                    imgs = imgs[0]  # take the first view

                imgs = imgs.to(pl_module.device)
                feats = pl_module.backbone(imgs)
                all_features.append(feats.cpu())
                all_labels.append(labels.cpu() if isinstance(labels, torch.Tensor) else torch.tensor(labels))

        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # L2-normalize features for cosine similarity via inner product
        features = F.normalize(features, dim=1)
        return features, labels

    # ------------------------------------------------------------------
    # Lightning hook
    # ------------------------------------------------------------------

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Compute and log k-NN accuracy at the configured epoch interval.

        Skips silently when:
        - The current epoch does not meet the configured interval.
        - The datamodule's ``val_dataloader()`` returns ``None`` (no val split).
        """
        if not self._should_run(trainer):
            return

        val_loader = trainer.datamodule.val_dataloader()
        if val_loader is None:
            # D-02: guard against missing val/ directory — no crash
            return

        # Extract train features (for the k-NN feature bank)
        train_loader = trainer.datamodule.train_dataloader()
        train_feats, train_labels = self._extract_features(pl_module, train_loader)

        # Extract val features (queries)
        val_feats, val_labels = self._extract_features(pl_module, val_loader)

        num_classes = int(train_labels.max().item()) + 1

        acc = knn_predict(
            train_feats,
            train_labels,
            val_feats,
            val_labels,
            k=self.k,
            temperature=self.temperature,
            num_classes=num_classes,
        )

        pl_module.log("eval/knn_acc", acc, prog_bar=True)
