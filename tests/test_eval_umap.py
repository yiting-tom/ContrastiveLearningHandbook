"""Unit tests for eval/umap_vis.py.

Tests verify:
- run_umap with random features produces a PNG file
- UMAP is called with metric='cosine' and random_state=42
- run_umap returns the reducer object for reuse
- Script prints torchdr suggestion when n_samples > 50000
- PNG file is >0 bytes
- Script respects n_samples config (subsamples if larger)
"""
from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure repo root is on path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def random_features():
    """Return random feature matrix [200, 64] and integer labels [200]."""
    rng = np.random.default_rng(42)
    features = rng.standard_normal((200, 64)).astype(np.float32)
    labels = rng.integers(0, 10, size=200)
    return features, labels


@pytest.fixture
def large_features():
    """Return feature matrix with 60000 samples (> 50K torchdr threshold)."""
    rng = np.random.default_rng(1)
    features = rng.standard_normal((60000, 32)).astype(np.float32)
    labels = rng.integers(0, 10, size=60000)
    return features, labels


# ---------------------------------------------------------------------------
# Test 1: run_umap produces a PNG file
# ---------------------------------------------------------------------------

def test_run_umap_produces_png(tmp_path, random_features):
    """run_umap should save a PNG file and return its path."""
    from eval.umap_vis import run_umap

    features, labels = random_features
    path, reducer = run_umap(features, labels, metric="cosine", output_dir=tmp_path)

    assert path.exists(), f"Expected output PNG at {path}"
    assert path.suffix == ".png", f"Expected .png file, got {path.suffix}"


def test_run_umap_output_filename(tmp_path, random_features):
    """Output PNG filename should be 'umap.png'."""
    from eval.umap_vis import run_umap

    features, labels = random_features
    path, _ = run_umap(features, labels, metric="cosine", output_dir=tmp_path)

    assert path.name == "umap.png", f"Expected 'umap.png', got {path.name!r}"


# ---------------------------------------------------------------------------
# Test 2: UMAP is called with metric='cosine' and random_state=42
# ---------------------------------------------------------------------------

def test_umap_parameters(tmp_path, random_features):
    """UMAP must be constructed with metric='cosine' and random_state=42."""
    from eval.umap_vis import run_umap

    features, labels = random_features

    with patch("eval.umap_vis.umap.UMAP") as mock_umap_cls:
        mock_reducer = MagicMock()
        mock_reducer.fit_transform.return_value = np.random.randn(200, 2).astype(np.float32)
        mock_umap_cls.return_value = mock_reducer

        run_umap(features, labels, metric="cosine", output_dir=tmp_path)

    assert mock_umap_cls.called, "umap.UMAP was not instantiated"
    _, kwargs = mock_umap_cls.call_args
    assert kwargs.get("random_state") == 42, (
        f"random_state should be 42, got {kwargs.get('random_state')!r}"
    )
    # metric is passed via parameter, check it was used
    metric_used = kwargs.get("metric")
    assert metric_used == "cosine", (
        f"metric should be 'cosine', got {metric_used!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: run_umap returns the reducer object for reuse
# ---------------------------------------------------------------------------

def test_run_umap_returns_reducer(tmp_path, random_features):
    """run_umap must return a (path, reducer) tuple; reducer must be a umap.UMAP instance."""
    from eval.umap_vis import run_umap
    import umap

    features, labels = random_features
    result = run_umap(features, labels, metric="cosine", output_dir=tmp_path)

    assert isinstance(result, tuple) and len(result) == 2, (
        f"Expected (path, reducer) tuple, got {type(result)}"
    )
    path, reducer = result
    assert isinstance(reducer, umap.UMAP), (
        f"Expected umap.UMAP instance, got {type(reducer)}"
    )


# ---------------------------------------------------------------------------
# Test 4: torchdr suggestion printed for > 50000 samples
# ---------------------------------------------------------------------------

def test_torchdr_suggestion_for_large_dataset(tmp_path, large_features, capsys):
    """run_umap should print a torchdr suggestion when n_samples > 50000."""
    from eval.umap_vis import run_umap

    features, labels = large_features
    assert features.shape[0] > 50_000, "Test requires > 50K samples"

    with patch("eval.umap_vis.umap.UMAP") as mock_umap_cls:
        mock_reducer = MagicMock()
        mock_reducer.fit_transform.return_value = np.random.randn(
            features.shape[0], 2
        ).astype(np.float32)
        mock_umap_cls.return_value = mock_reducer

        run_umap(features, labels, metric="cosine", output_dir=tmp_path)

    captured = capsys.readouterr()
    assert "torchdr" in captured.out.lower(), (
        "Expected torchdr mention in stdout for datasets > 50K samples"
    )


def test_no_torchdr_suggestion_for_small_dataset(tmp_path, random_features, capsys):
    """run_umap should NOT print torchdr suggestion for small datasets."""
    from eval.umap_vis import run_umap

    features, labels = random_features
    assert features.shape[0] < 50_000, "Test requires < 50K samples"

    run_umap(features, labels, metric="cosine", output_dir=tmp_path)

    captured = capsys.readouterr()
    assert "torchdr" not in captured.out.lower(), (
        "Did not expect torchdr mention for small datasets"
    )


# ---------------------------------------------------------------------------
# Test 5: PNG file is > 0 bytes
# ---------------------------------------------------------------------------

def test_png_file_is_nonempty(tmp_path, random_features):
    """Output PNG must be a non-empty file."""
    from eval.umap_vis import run_umap

    features, labels = random_features
    path, _ = run_umap(features, labels, metric="cosine", output_dir=tmp_path)

    size = path.stat().st_size
    assert size > 0, f"PNG file {path} is empty (0 bytes)"


# ---------------------------------------------------------------------------
# Test 6: n_samples respected (subsampling)
# ---------------------------------------------------------------------------

def test_n_samples_subsampling(tmp_path):
    """run_umap should subsample to n_samples if dataset is larger."""
    from eval.umap_vis import run_umap

    rng = np.random.default_rng(7)
    features = rng.standard_normal((500, 20)).astype(np.float32)
    labels = rng.integers(0, 5, size=500)

    captured_input = []

    with patch("eval.umap_vis.umap.UMAP") as mock_umap_cls:
        mock_reducer = MagicMock()
        def fake_fit_transform(X):
            captured_input.append(X.shape[0])
            return np.random.randn(X.shape[0], 2).astype(np.float32)
        mock_reducer.fit_transform.side_effect = fake_fit_transform
        mock_umap_cls.return_value = mock_reducer

        run_umap(features, labels, metric="cosine", output_dir=tmp_path, n_samples=100)

    assert captured_input, "umap.UMAP.fit_transform was never called"
    assert captured_input[0] <= 100, (
        f"Expected at most 100 samples passed to UMAP, got {captured_input[0]}"
    )
