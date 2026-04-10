"""Tests for ClassBalancedSampler — per-class guarantee and integration."""
import random
from collections import Counter

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from core.data import ClassBalancedSampler


class FakeImageFolder:
    """Minimal ImageFolder stand-in with a .targets attribute."""

    def __init__(self, targets: list[int]):
        self.targets = targets
        # Build simple (tensor, label) pairs
        self._data = [(torch.zeros(3, 4, 4), t) for t in targets]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


def make_balanced_dataset(n_classes: int = 10, n_per_class: int = 20) -> FakeImageFolder:
    targets = []
    for c in range(n_classes):
        targets.extend([c] * n_per_class)
    return FakeImageFolder(targets)


# ---------------------------------------------------------------------------
# Test 1: min(class_counts_per_batch) >= n_samples_per_class  (SC-3)
# ---------------------------------------------------------------------------
def test_min_class_count_per_batch():
    """Every batch must contain >= n_samples_per_class instances of each sampled class."""
    n_classes_per_batch = 5
    n_samples_per_class = 2
    dataset = make_balanced_dataset(n_classes=10, n_per_class=20)
    sampler = ClassBalancedSampler(dataset, n_classes_per_batch, n_samples_per_class)

    batch_size = n_classes_per_batch * n_samples_per_class
    targets = dataset.targets

    indices = list(sampler)
    n_batches = len(indices) // batch_size

    for b in range(n_batches):
        batch_indices = indices[b * batch_size : (b + 1) * batch_size]
        batch_labels = [targets[i] for i in batch_indices]
        counts = Counter(batch_labels)
        min_count = min(counts.values())
        assert min_count >= n_samples_per_class, (
            f"Batch {b}: min class count {min_count} < {n_samples_per_class}"
        )


# ---------------------------------------------------------------------------
# Test 2: Sampler produces correct total length
# ---------------------------------------------------------------------------
def test_sampler_length():
    n_classes_per_batch = 4
    n_samples_per_class = 3
    dataset = make_balanced_dataset(n_classes=8, n_per_class=10)
    sampler = ClassBalancedSampler(dataset, n_classes_per_batch, n_samples_per_class)

    indices = list(sampler)
    assert len(indices) == len(sampler), (
        f"__len__={len(sampler)} but __iter__ yielded {len(indices)} indices"
    )
    # Length must be a multiple of batch_size
    batch_size = n_classes_per_batch * n_samples_per_class
    assert len(indices) % batch_size == 0


# ---------------------------------------------------------------------------
# Test 3: ValueError when n_classes_per_batch > number of dataset classes
# ---------------------------------------------------------------------------
def test_too_many_classes_raises():
    dataset = make_balanced_dataset(n_classes=3, n_per_class=10)
    with pytest.raises(ValueError, match="n_classes_per_batch"):
        ClassBalancedSampler(dataset, n_classes_per_batch=5, n_samples_per_class=2)


# ---------------------------------------------------------------------------
# Test 4: DataLoader with sampler and shuffle=False works without error
# ---------------------------------------------------------------------------
def test_dataloader_integration_no_error():
    dataset = make_balanced_dataset(n_classes=5, n_per_class=10)
    sampler = ClassBalancedSampler(dataset, n_classes_per_batch=5, n_samples_per_class=2)
    loader = DataLoader(dataset, batch_size=10, sampler=sampler, shuffle=False)
    batch = next(iter(loader))
    assert batch is not None


# ---------------------------------------------------------------------------
# Test 5: shuffle=True with sampler raises ValueError (PyTorch guard)
# ---------------------------------------------------------------------------
def test_shuffle_true_with_sampler_raises():
    dataset = make_balanced_dataset(n_classes=5, n_per_class=10)
    sampler = ClassBalancedSampler(dataset, n_classes_per_batch=5, n_samples_per_class=2)
    with pytest.raises(ValueError):
        DataLoader(dataset, batch_size=10, sampler=sampler, shuffle=True)
