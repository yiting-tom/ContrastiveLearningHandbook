"""Tests for SupConLoss — sum-outside formulation, SimCLR equivalence, label mode."""
import torch
import torch.nn.functional as F
import pytest

from core.losses import InfoNCELoss, SupConLoss


def make_features(B: int, D: int, seed: int = 0) -> torch.Tensor:
    """Return random [B, D] tensor (not normalized — loss normalizes internally)."""
    torch.manual_seed(seed)
    return torch.randn(B, D)


# ---------------------------------------------------------------------------
# Test 1: SimCLR equivalence — labels=None must match InfoNCELoss
# ---------------------------------------------------------------------------
def test_simclr_equivalence_labels_none():
    """SupConLoss(labels=None) must produce identical output to InfoNCELoss."""
    B, D = 16, 128
    z_i = make_features(B, D, seed=0)
    z_j = make_features(B, D, seed=1)

    supcon = SupConLoss(temperature=0.5, reduction="mean")
    infonce = InfoNCELoss(temperature=0.5, reduction="mean")

    loss_supcon = supcon(z_i, z_j, labels=None)
    loss_infonce = infonce(z_i, z_j)

    assert torch.isclose(loss_supcon, loss_infonce, atol=1e-5), (
        f"SupCon(labels=None)={loss_supcon.item():.6f} != InfoNCE={loss_infonce.item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 2: Finite positive scalar with labels
# ---------------------------------------------------------------------------
def test_supcon_finite_positive_scalar_with_labels():
    B, D = 32, 128
    z_i = make_features(B, D, seed=2)
    z_j = make_features(B, D, seed=3)
    # 4 classes, 8 samples each
    labels = torch.repeat_interleave(torch.arange(4), 8)

    loss_fn = SupConLoss(temperature=0.07)
    loss = loss_fn(z_i, z_j, labels=labels)

    assert loss.shape == (), "Expected scalar output"
    assert torch.isfinite(loss), "Loss must be finite"
    assert loss.item() > 0, "Loss must be positive"


# ---------------------------------------------------------------------------
# Test 3: More same-class positives in batch → lower loss
# ---------------------------------------------------------------------------
def test_more_positives_lower_loss():
    """A batch where many samples share the same class has lower loss than
    a batch where every sample has a unique class (one positive = SimCLR mode)."""
    B, D = 32, 128
    torch.manual_seed(42)
    z_i = F.normalize(torch.randn(B, D), dim=1)
    z_j = F.normalize(torch.randn(B, D), dim=1)

    loss_fn = SupConLoss(temperature=0.07)

    # All unique labels → one positive per anchor (same as SimCLR)
    labels_unique = torch.arange(B)
    loss_unique = loss_fn(z_i, z_j, labels=labels_unique)

    # 4 classes × 8 samples each → 15 positives per anchor
    labels_grouped = torch.repeat_interleave(torch.arange(4), B // 4)
    loss_grouped = loss_fn(z_i, z_j, labels=labels_grouped)

    assert loss_grouped.item() < loss_unique.item(), (
        f"Expected grouped loss ({loss_grouped.item():.4f}) < "
        f"unique loss ({loss_unique.item():.4f})"
    )


# ---------------------------------------------------------------------------
# Test 4: All-unique labels matches labels=None result
# ---------------------------------------------------------------------------
def test_all_unique_labels_matches_no_labels():
    """When every sample has a unique class, SupCon must equal SimCLR mode."""
    B, D = 16, 128
    z_i = make_features(B, D, seed=10)
    z_j = make_features(B, D, seed=11)

    loss_fn = SupConLoss(temperature=0.5)
    loss_unique = loss_fn(z_i, z_j, labels=torch.arange(B))
    loss_none = loss_fn(z_i, z_j, labels=None)

    assert torch.isclose(loss_unique, loss_none, atol=1e-5), (
        f"All-unique labels ({loss_unique.item():.6f}) != labels=None ({loss_none.item():.6f})"
    )


# ---------------------------------------------------------------------------
# Test 5: Temperature sensitivity — lower tau → higher loss magnitude
# ---------------------------------------------------------------------------
def test_temperature_sensitivity():
    B, D = 16, 128
    z_i = make_features(B, D, seed=20)
    z_j = make_features(B, D, seed=21)
    labels = torch.repeat_interleave(torch.arange(4), 4)

    loss_low_tau = SupConLoss(temperature=0.07)(z_i, z_j, labels=labels)
    loss_high_tau = SupConLoss(temperature=1.0)(z_i, z_j, labels=labels)

    assert loss_low_tau.item() > loss_high_tau.item(), (
        "Lower temperature should produce higher loss"
    )
