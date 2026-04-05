"""Tests for core/queue.py — MomentumQueue FIFO negative-key buffer.

Tests verify:
  - Queue initializes with L2-normalized random vectors of shape [dim, queue_size]
  - After enqueuing queue_size vectors, queue size is exactly queue_size (size invariant)
  - After enqueuing queue_size + batch_size vectors, queue size remains queue_size
  - Pointer wrap-around: enqueue vectors when ptr near end correctly wraps
  - get_negatives() returns tensor of shape [D, K] that is detached
  - get_negatives() returns a clone (mutating returned tensor does not affect internal queue)
  - FIFO order — newest keys replace oldest keys
  - Queue and pointer survive state_dict save/load (register_buffer)
  - update() L2-normalizes keys before storing
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def queue():
    """Create a small MomentumQueue for testing."""
    from core.queue import MomentumQueue
    torch.manual_seed(42)
    return MomentumQueue(queue_size=8, dim=4)


@pytest.fixture
def large_queue():
    """Create a larger MomentumQueue for wrap-around tests."""
    from core.queue import MomentumQueue
    torch.manual_seed(42)
    return MomentumQueue(queue_size=16, dim=4)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_queue_shape(self, queue):
        """Queue tensor has shape [dim, queue_size]."""
        assert queue.queue.shape == (4, 8)

    def test_queue_l2_normalized(self, queue):
        """Initial queue vectors are L2-normalized along dim=0."""
        norms = queue.queue.norm(dim=0)
        torch.testing.assert_close(norms, torch.ones(8), atol=1e-5, rtol=1e-5)

    def test_pointer_starts_at_zero(self, queue):
        """Pointer initializes to zero."""
        assert queue.ptr.item() == 0


# ---------------------------------------------------------------------------
# Size invariant tests
# ---------------------------------------------------------------------------

class TestSizeInvariant:
    def test_queue_size_invariant(self, queue):
        """After enqueuing exactly queue_size vectors, queue size stays at queue_size."""
        keys = F.normalize(torch.randn(8, 4), dim=1)
        queue.update(keys)
        assert queue.queue.shape == (4, 8)

    def test_queue_size_after_overflow(self, queue):
        """After enqueuing queue_size + batch_size vectors, queue size remains queue_size."""
        # Fill queue
        keys1 = F.normalize(torch.randn(8, 4), dim=1)
        queue.update(keys1)
        # Overflow
        keys2 = F.normalize(torch.randn(4, 4), dim=1)
        queue.update(keys2)
        assert queue.queue.shape == (4, 8)


# ---------------------------------------------------------------------------
# Pointer wrap-around tests
# ---------------------------------------------------------------------------

class TestWrapAround:
    def test_pointer_wraps(self, queue):
        """Pointer wraps around after filling the queue."""
        keys = F.normalize(torch.randn(8, 4), dim=1)
        queue.update(keys)
        assert queue.ptr.item() == 0  # wraps back to 0

    def test_pointer_wrap_mid_batch(self, queue):
        """When ptr + batch_size > queue_size, pointer wraps correctly."""
        # Fill 6 slots
        keys1 = F.normalize(torch.randn(6, 4), dim=1)
        queue.update(keys1)
        assert queue.ptr.item() == 6

        # Enqueue 4 more: 2 fit at end, 2 wrap to beginning
        keys2 = F.normalize(torch.randn(4, 4), dim=1)
        queue.update(keys2)
        assert queue.ptr.item() == 2

        # Verify the wrapped keys are in the right positions
        # Last 2 of keys2 should be at positions 0 and 1
        expected_start = F.normalize(keys2[2:], dim=1).T
        torch.testing.assert_close(
            queue.queue[:, :2], expected_start, atol=1e-5, rtol=1e-5
        )


# ---------------------------------------------------------------------------
# get_negatives tests
# ---------------------------------------------------------------------------

class TestGetNegatives:
    def test_shape(self, queue):
        """get_negatives() returns shape [D, K]."""
        neg = queue.get_negatives()
        assert neg.shape == (4, 8)

    def test_detached(self, queue):
        """get_negatives() returns a detached tensor."""
        neg = queue.get_negatives()
        assert not neg.requires_grad

    def test_clone_independence(self, queue):
        """Mutating get_negatives() result does not affect internal queue."""
        original = queue.queue.clone()
        neg = queue.get_negatives()
        neg.fill_(999.0)
        torch.testing.assert_close(queue.queue, original)


# ---------------------------------------------------------------------------
# FIFO order tests
# ---------------------------------------------------------------------------

class TestFIFOOrder:
    def test_newest_replaces_oldest(self, queue):
        """FIFO: newest keys replace oldest keys, not newest-first."""
        torch.manual_seed(99)
        # Fill entire queue with batch A
        batch_a = F.normalize(torch.randn(8, 4), dim=1)
        queue.update(batch_a)

        # Now enqueue batch B of size 4 — should replace positions 0-3
        batch_b = F.normalize(torch.randn(4, 4), dim=1)
        queue.update(batch_b)

        # Positions 0-3 should have batch_b
        expected_b = F.normalize(batch_b, dim=1).T
        torch.testing.assert_close(
            queue.queue[:, :4], expected_b, atol=1e-5, rtol=1e-5
        )

        # Positions 4-7 should still have batch_a[4:8]
        expected_a = F.normalize(batch_a[4:], dim=1).T
        torch.testing.assert_close(
            queue.queue[:, 4:], expected_a, atol=1e-5, rtol=1e-5
        )


# ---------------------------------------------------------------------------
# State dict persistence tests
# ---------------------------------------------------------------------------

class TestStateDictPersistence:
    def test_queue_survives_save_load(self, queue):
        """Queue and pointer survive state_dict save/load (register_buffer)."""
        # Enqueue some data
        keys = F.normalize(torch.randn(4, 4), dim=1)
        queue.update(keys)

        # Save state
        state = queue.state_dict()

        # Create new queue and load state
        from core.queue import MomentumQueue
        new_queue = MomentumQueue(queue_size=8, dim=4)
        new_queue.load_state_dict(state)

        torch.testing.assert_close(new_queue.queue, queue.queue)
        assert new_queue.ptr.item() == queue.ptr.item()


# ---------------------------------------------------------------------------
# L2 normalization on update tests
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_update_normalizes_keys(self, queue):
        """update() L2-normalizes keys before storing (even if input not normalized)."""
        # Create unnormalized keys
        keys = torch.randn(4, 4) * 5.0  # intentionally not normalized
        queue.update(keys)

        # Check that stored vectors are L2-normalized (along dim=0, which is
        # the feature dimension in [D, K] storage)
        norms = queue.queue[:, :4].norm(dim=0)
        torch.testing.assert_close(norms, torch.ones(4), atol=1e-5, rtol=1e-5)
