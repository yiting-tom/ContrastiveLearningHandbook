"""Unit tests for eval/tsne_vis.py.

Tests verify:
- run_tsne produces 3 PNGs with perplexity in filename
- PCA pre-reduction is applied when feature dim > 50
- TSNE is called with init='pca', metric='cosine', learning_rate='auto'
- Script respects n_samples config (subsamples if larger)
- Each PNG file is >0 bytes (valid image)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

# Ensure repo root is on path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def random_features():
    """Return random feature matrix [200, 128] and integer labels [200]."""
    rng = np.random.default_rng(42)
    features = rng.standard_normal((200, 128)).astype(np.float32)
    labels = rng.integers(0, 10, size=200)
    return features, labels


@pytest.fixture
def low_dim_features():
    """Return random feature matrix [200, 30] (< 50 dims -- no PCA needed)."""
    rng = np.random.default_rng(0)
    features = rng.standard_normal((200, 30)).astype(np.float32)
    labels = rng.integers(0, 5, size=200)
    return features, labels


# ---------------------------------------------------------------------------
# Test 1: run_tsne produces 3 PNG files named with perplexity
# ---------------------------------------------------------------------------

def test_run_tsne_produces_three_pngs(tmp_path, random_features):
    """run_tsne with 3 perplexity values should save 3 PNG files."""
    from eval.tsne_vis import run_tsne

    features, labels = random_features
    perplexities = [10, 30, 50]
    paths = run_tsne(features, labels, perplexities, tmp_path)

    assert len(paths) == 3, f"Expected 3 output paths, got {len(paths)}"
    for p in paths:
        assert p.exists(), f"Expected file {p} to exist"
        assert p.suffix == ".png", f"Expected .png suffix, got {p.suffix}"


def test_run_tsne_filenames_contain_perplexity(tmp_path, random_features):
    """Output PNG filenames must include the perplexity value."""
    from eval.tsne_vis import run_tsne

    features, labels = random_features
    perplexities = [10, 30, 50]
    paths = run_tsne(features, labels, perplexities, tmp_path)

    for perp, path in zip(perplexities, paths):
        assert str(perp) in path.name, (
            f"Expected perplexity {perp} in filename {path.name}"
        )


# ---------------------------------------------------------------------------
# Test 2: PCA pre-reduction is applied when feature dim > 50
# ---------------------------------------------------------------------------

def test_pca_prereduction_applied_for_high_dim(tmp_path, random_features):
    """PCA should be applied when features.shape[1] > 50."""
    from eval.tsne_vis import run_tsne

    features, labels = random_features
    assert features.shape[1] > 50, "Test requires features with dim > 50"

    with patch("eval.tsne_vis.PCA") as mock_pca_cls:
        mock_pca = MagicMock()
        mock_pca.fit_transform.return_value = np.random.randn(200, 50).astype(np.float32)
        mock_pca_cls.return_value = mock_pca

        with patch("eval.tsne_vis.TSNE") as mock_tsne_cls:
            mock_tsne = MagicMock()
            mock_tsne.fit_transform.return_value = np.random.randn(200, 2).astype(np.float32)
            mock_tsne_cls.return_value = mock_tsne

            run_tsne(features, labels, [10], tmp_path)

        mock_pca_cls.assert_called_once()
        call_kwargs = mock_pca_cls.call_args
        # n_components=50 must be set
        assert call_kwargs is not None
        args, kwargs = call_kwargs
        n_components = kwargs.get("n_components", args[0] if args else None)
        assert n_components == 50, f"PCA n_components should be 50, got {n_components}"


def test_pca_not_applied_for_low_dim(tmp_path, low_dim_features):
    """PCA should NOT be applied when features.shape[1] <= 50."""
    from eval.tsne_vis import run_tsne

    features, labels = low_dim_features
    assert features.shape[1] <= 50, "Test requires features with dim <= 50"

    with patch("eval.tsne_vis.PCA") as mock_pca_cls:
        with patch("eval.tsne_vis.TSNE") as mock_tsne_cls:
            mock_tsne = MagicMock()
            mock_tsne.fit_transform.return_value = np.random.randn(200, 2).astype(np.float32)
            mock_tsne_cls.return_value = mock_tsne

            run_tsne(features, labels, [10], tmp_path)

        mock_pca_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: TSNE is called with correct parameters
# ---------------------------------------------------------------------------

def test_tsne_parameters(tmp_path, random_features):
    """TSNE must be constructed with init='pca', metric='cosine', learning_rate='auto'."""
    from eval.tsne_vis import run_tsne

    features, labels = random_features

    with patch("eval.tsne_vis.PCA") as mock_pca_cls:
        mock_pca = MagicMock()
        mock_pca.fit_transform.return_value = np.random.randn(200, 50).astype(np.float32)
        mock_pca_cls.return_value = mock_pca

        with patch("eval.tsne_vis.TSNE") as mock_tsne_cls:
            mock_tsne = MagicMock()
            mock_tsne.fit_transform.return_value = np.random.randn(200, 2).astype(np.float32)
            mock_tsne_cls.return_value = mock_tsne

            run_tsne(features, labels, [30], tmp_path)

    assert mock_tsne_cls.called, "TSNE was not instantiated"
    _, kwargs = mock_tsne_cls.call_args
    assert kwargs.get("init") == "pca", f"init should be 'pca', got {kwargs.get('init')!r}"
    assert kwargs.get("metric") == "cosine", f"metric should be 'cosine', got {kwargs.get('metric')!r}"
    assert kwargs.get("learning_rate") == "auto", (
        f"learning_rate should be 'auto', got {kwargs.get('learning_rate')!r}"
    )


# ---------------------------------------------------------------------------
# Test 4: n_samples respected (subsampling)
# ---------------------------------------------------------------------------

def test_n_samples_subsampling(tmp_path):
    """run_tsne should subsample to n_samples if dataset is larger than n_samples."""
    from eval.tsne_vis import run_tsne

    # Create a large feature matrix (300 samples) but request only 100
    rng = np.random.default_rng(7)
    features = rng.standard_normal((300, 20)).astype(np.float32)
    labels = rng.integers(0, 5, size=300)

    captured_input = []

    with patch("eval.tsne_vis.TSNE") as mock_tsne_cls:
        mock_tsne = MagicMock()
        def fake_fit_transform(X):
            captured_input.append(X.shape[0])
            return np.random.randn(X.shape[0], 2).astype(np.float32)
        mock_tsne.fit_transform.side_effect = fake_fit_transform
        mock_tsne_cls.return_value = mock_tsne

        run_tsne(features, labels, [10], tmp_path, n_samples=100)

    # All TSNE calls should receive exactly 100 rows
    assert captured_input, "TSNE.fit_transform was never called"
    for n in captured_input:
        assert n <= 100, f"Expected at most 100 samples passed to TSNE, got {n}"


# ---------------------------------------------------------------------------
# Test 5: Each PNG file is > 0 bytes
# ---------------------------------------------------------------------------

def test_png_files_are_nonempty(tmp_path, random_features):
    """Each output PNG must be a non-empty file (valid image)."""
    from eval.tsne_vis import run_tsne

    features, labels = random_features
    paths = run_tsne(features, labels, [10, 30, 50], tmp_path)

    for p in paths:
        size = p.stat().st_size
        assert size > 0, f"PNG file {p} is empty (0 bytes)"
