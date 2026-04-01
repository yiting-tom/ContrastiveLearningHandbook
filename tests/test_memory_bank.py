"""Unit tests for core.memory_bank.MemoryBank."""
import torch
import pytest

from core.memory_bank import MemoryBank


class TestMemoryBankInit:
    """Tests for MemoryBank initialization."""

    def test_init_shape(self):
        """MemoryBank(100, 64).bank.weight.shape == (100, 64)."""
        mb = MemoryBank(100, 64)
        assert mb.bank.weight.shape == (100, 64)

    def test_init_l2_normalized(self):
        """All row norms of bank.weight.data are 1.0 (within atol=1e-5)."""
        mb = MemoryBank(100, 64)
        norms = mb.bank.weight.data.norm(dim=1)
        assert torch.allclose(norms, torch.ones(100), atol=1e-5)

    def test_requires_grad_false(self):
        """bank.bank.weight.requires_grad is False."""
        mb = MemoryBank(100, 64)
        assert mb.bank.weight.requires_grad is False

    def test_bank_not_in_learnable_params_pattern(self):
        """bank.bank.weight not in list of params with requires_grad=True."""
        mb = MemoryBank(100, 64)
        learnable = [p for p in mb.parameters() if p.requires_grad]
        assert mb.bank.weight not in learnable


class TestMemoryBankGetUpdate:
    """Tests for MemoryBank get/update operations."""

    def test_get_returns_correct_features(self):
        """After update(indices, features), get(indices) returns L2-normalized features."""
        mb = MemoryBank(100, 64)
        features = torch.randn(2, 64)
        indices = torch.tensor([0, 1])
        mb.update(indices, features)
        retrieved = mb.get(indices)
        expected = torch.nn.functional.normalize(features, dim=1)
        assert torch.allclose(retrieved, expected, atol=1e-5)

    def test_update_l2_normalizes(self):
        """Features passed to update are stored L2-normalized regardless of input norm."""
        mb = MemoryBank(50, 32)
        # Create features with large norms
        features = torch.randn(3, 32) * 100.0
        indices = torch.tensor([5, 10, 15])
        mb.update(indices, features)
        stored = mb.get(indices)
        norms = stored.norm(dim=1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)

    def test_update_does_not_affect_other_indices(self):
        """After update(indices=[0], features), bank entries at other indices are unchanged."""
        mb = MemoryBank(10, 16)
        original = mb.bank.weight.data.clone()
        features = torch.randn(1, 16)
        indices = torch.tensor([0])
        mb.update(indices, features)
        # Entries 1..9 should be unchanged
        assert torch.equal(mb.bank.weight.data[1:], original[1:])


class TestMemoryBankDocstring:
    """Tests for MemoryBank documentation."""

    def test_staleness_docstring(self):
        """MemoryBank.__doc__ contains 'stale' and 'MoCo'."""
        doc = MemoryBank.__doc__
        assert doc is not None
        assert "stale" in doc.lower()
        assert "MoCo" in doc
