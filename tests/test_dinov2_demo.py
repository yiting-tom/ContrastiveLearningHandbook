"""Tests for eval/dinov2_demo.py.

Verifies:
  - Script is importable as a module.
  - argparse defaults are correct (--dataset defaults to 'cifar10').
  - timm knows about DINOv2 models (verifies timm installation).

Intentionally does NOT test actual model download or feature extraction —
those require internet access and significant time/memory.
"""
from __future__ import annotations

import pytest
import timm


def test_dinov2_demo_importable():
    """eval.dinov2_demo can be imported without error."""
    import eval.dinov2_demo  # noqa: F401


def test_dinov2_demo_argparse_defaults():
    """argparse defaults: dataset='cifar10', data_dir='data', k=20, batch_size=256."""
    from eval.dinov2_demo import get_args
    import sys

    # Temporarily replace argv to simulate empty CLI invocation
    original_argv = sys.argv
    try:
        sys.argv = ["dinov2_demo.py"]
        args = get_args()
    finally:
        sys.argv = original_argv

    assert args.dataset == "cifar10", f"Expected default dataset='cifar10', got {args.dataset!r}"
    assert args.data_dir == "data", f"Expected default data_dir='data', got {args.data_dir!r}"
    assert args.k == 20, f"Expected default k=20, got {args.k}"
    assert args.batch_size == 256, f"Expected default batch_size=256, got {args.batch_size}"


def test_dinov2_demo_dataset_choices():
    """--dataset accepts exactly cifar10, stl10, imagefolder."""
    from eval.dinov2_demo import get_args
    import sys

    for dataset in ("cifar10", "stl10", "imagefolder"):
        original_argv = sys.argv
        try:
            sys.argv = ["dinov2_demo.py", "--dataset", dataset]
            args = get_args()
        finally:
            sys.argv = original_argv
        assert args.dataset == dataset


def test_timm_knows_dinov2():
    """timm.list_models('*dinov2*') returns at least one model containing 'dinov2'.

    This verifies that the timm installation supports DINOv2 model families.
    Does NOT download weights.
    """
    dinov2_models = timm.list_models("*dinov2*")
    assert len(dinov2_models) > 0, (
        "timm.list_models('*dinov2*') returned no models — "
        "ensure timm >= 0.9 is installed for DINOv2 support."
    )
    # At least one model should contain 'dinov2' in its name
    assert any("dinov2" in m for m in dinov2_models), (
        f"No model containing 'dinov2' found in: {dinov2_models[:5]}"
    )


def test_dinov2_demo_docstring_lineage():
    """Docstring mentions correct DINO lineage and notes 'DINOv3' does not exist."""
    import eval.dinov2_demo as mod

    doc = mod.__doc__
    assert doc is not None, "dinov2_demo module should have a docstring"
    assert "DINOv3" in doc, "Docstring should mention 'DINOv3'"
    assert "does not exist" in doc, "Docstring should note that 'DINOv3' does not exist"
    assert "DINO -> DINOv2" in doc, "Docstring should show the correct lineage: DINO -> DINOv2"
    # Check for register tokens note (case-insensitive match for both forms)
    doc_lower = doc.lower()
    assert "register token" in doc_lower, "Docstring should mention 'Register tokens'"
