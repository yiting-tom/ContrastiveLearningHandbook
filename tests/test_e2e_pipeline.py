"""End-to-end CLI integration tests for Phase 10.1.

These tests subprocess-invoke train.py and eval/*.py scripts to catch
integration bugs that in-process tests (test_eval_integration.py) bypass.

Wave 0 stubs raise NotImplementedError. Bodies are filled in plan 10.1-05
after the bug fixes in plans 10.1-02, 10.1-03, 10.1-04 are committed.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = REPO_ROOT / "train.py"
EVAL_DIR = REPO_ROOT / "eval"


# ---------------------------------------------------------------------------
# Regression-B1: every eval script's --help exits 0 (sys.path bootstrap fix)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "script",
    ["linear_probe.py", "finetune.py", "tsne_vis.py", "umap_vis.py", "cam_vis.py"],
)
def test_each_eval_script_help_exits_zero(script: str) -> None:
    """Regression for B1: every eval script imports successfully via `python eval/X.py --help`."""
    raise NotImplementedError("Wave 0 stub — filled in plan 10.1-05")


# ---------------------------------------------------------------------------
# Regression-B2: no SSLDataModule(cfg) misuse anywhere in eval/*.py
# ---------------------------------------------------------------------------

def test_no_ssldatamodule_cfg_misuse() -> None:
    """Regression for B2: greps eval/*.py source for `SSLDataModule(cfg)` mis-construction."""
    raise NotImplementedError("Wave 0 stub — filled in plan 10.1-05")


# ---------------------------------------------------------------------------
# SC-1: train.py runs end-to-end on toy config + produces a checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_train_produces_checkpoint(tmp_path: Path, tmp_imagefolder_with_val: Path) -> None:
    """SC-1: train.py runs end-to-end on a toy CIFAR-10-shaped config and produces a checkpoint."""
    raise NotImplementedError("Wave 0 stub — filled in plan 10.1-05")


# ---------------------------------------------------------------------------
# SC-2a-e: each eval script accepts a checkpoint produced by train.py
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_linear_probe_accepts_train_checkpoint(
    tmp_path: Path, tmp_imagefolder_with_val: Path
) -> None:
    """SC-2a: eval/linear_probe.py accepts a checkpoint produced by train.py."""
    raise NotImplementedError("Wave 0 stub — filled in plan 10.1-05")


@pytest.mark.slow
def test_tsne_accepts_train_checkpoint(
    tmp_path: Path, tmp_imagefolder_with_val: Path
) -> None:
    """SC-2b: eval/tsne_vis.py accepts a checkpoint produced by train.py."""
    raise NotImplementedError("Wave 0 stub — filled in plan 10.1-05")


@pytest.mark.slow
def test_umap_accepts_train_checkpoint(
    tmp_path: Path, tmp_imagefolder_with_val: Path
) -> None:
    """SC-2c: eval/umap_vis.py accepts a checkpoint produced by train.py."""
    raise NotImplementedError("Wave 0 stub — filled in plan 10.1-05")


@pytest.mark.slow
def test_finetune_accepts_train_checkpoint(
    tmp_path: Path, tmp_imagefolder_with_val: Path
) -> None:
    """SC-2d: eval/finetune.py accepts a checkpoint produced by train.py."""
    raise NotImplementedError("Wave 0 stub — filled in plan 10.1-05")


@pytest.mark.slow
def test_cam_accepts_train_checkpoint(
    tmp_path: Path, tmp_imagefolder_with_val: Path
) -> None:
    """SC-2e: eval/cam_vis.py accepts a checkpoint produced by train.py."""
    raise NotImplementedError("Wave 0 stub — filled in plan 10.1-05")


# ---------------------------------------------------------------------------
# SC-3: full pipeline (train.py -> eval/linear_probe.py) completes
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_full_pipeline_train_then_probe(
    tmp_path: Path, tmp_imagefolder_with_val: Path
) -> None:
    """SC-3: full pipeline train.py -> eval/linear_probe.py completes successfully."""
    raise NotImplementedError("Wave 0 stub — filled in plan 10.1-05")
