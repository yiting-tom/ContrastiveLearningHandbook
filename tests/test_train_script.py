"""Smoke tests for train.py — the single-entry SSL training CLI.

These tests verify that train.py:
1. Exposes --config / --data-dir / --ckpt-path via argparse and prints help.
2. Loads a real YAML config, dispatches a method, builds the data module,
   and runs at least one training step on a toy ImageFolder fixture.
3. Surfaces config errors (ValidationError, FileNotFoundError) rather than
   silently failing.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = REPO_ROOT / "train.py"


# ---------------------------------------------------------------------------
# CLI surface tests
# ---------------------------------------------------------------------------

def test_train_py_help_exits_zero():
    """`python train.py --help` exits 0 and lists all three flags."""
    result = subprocess.run(
        [sys.executable, str(TRAIN_PY), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"--help exited {result.returncode}: {result.stderr}"
    assert "--config" in result.stdout
    assert "--data-dir" in result.stdout
    assert "--ckpt-path" in result.stdout


def test_train_py_missing_config_fails():
    """Without --config, argparse exits non-zero."""
    result = subprocess.run(
        [sys.executable, str(TRAIN_PY)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# End-to-end smoke: 1 epoch, 1 batch, toy ImageFolder
# ---------------------------------------------------------------------------

def test_train_py_runs_one_batch_on_toy_data(tmp_imagefolder, tmp_path, monkeypatch):
    """train.py runs end-to-end: load_config -> dispatcher -> SSLDataModule -> 1 step."""
    # Build a tiny override config from the canonical simclr_v1 quickstart
    src = REPO_ROOT / "configs" / "simclr_v1_resnet18.yaml"
    with open(src) as fh:
        raw = yaml.safe_load(fh)
    raw["max_epochs"] = 1
    raw["batch_size"] = 4
    raw["num_workers"] = 0
    raw["data_dir"] = str(tmp_imagefolder)

    cfg_path = tmp_path / "smoke.yaml"
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(raw, fh)

    # Patch sys.argv and call main() directly so we can monkeypatch Trainer
    import lightning as L

    original_init = L.Trainer.__init__

    def fast_init(self, *args, **kwargs):
        kwargs["max_epochs"] = 1
        kwargs["limit_train_batches"] = 1
        kwargs["accelerator"] = "cpu"
        kwargs["logger"] = False
        # Do NOT set enable_checkpointing=False: train.py now passes a
        # ModelCheckpoint callback (B3 fix), and Lightning raises
        # MisconfigurationException when enable_checkpointing=False conflicts
        # with an explicit ModelCheckpoint in the callbacks list.
        # Use default_root_dir so checkpoints are isolated to tmp_path.
        kwargs.setdefault("default_root_dir", str(tmp_path))
        kwargs["enable_progress_bar"] = False
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(L.Trainer, "__init__", fast_init)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(cfg_path)])

    # Import after monkeypatching argv
    import importlib
    if "train" in sys.modules:
        del sys.modules["train"]
    import train
    train.main()
    # If we reach this point without exception, the full pipeline ran one batch.


def test_train_py_invalid_config_raises(tmp_path, monkeypatch):
    """Pointing --config at a nonexistent file raises (not silently passes)."""
    bogus = tmp_path / "does_not_exist.yaml"
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(bogus)])
    if "train" in sys.modules:
        del sys.modules["train"]
    import train

    with pytest.raises((FileNotFoundError, OSError)):
        train.main()


# ---------------------------------------------------------------------------
# Regression-B3: train.py must configure ModelCheckpoint(save_last=True)
# ---------------------------------------------------------------------------

def test_train_py_imports_model_checkpoint():
    """B3 regression: train.py must import ModelCheckpoint from lightning.pytorch.callbacks."""
    source = TRAIN_PY.read_text()
    assert "from lightning.pytorch.callbacks import ModelCheckpoint" in source, (
        "train.py does not import ModelCheckpoint — `last.ckpt` will never be created"
    )


def test_train_py_has_save_last_checkpoint():
    """B3 regression: train.py must construct ModelCheckpoint(save_last=True, save_top_k=-1)."""
    source = TRAIN_PY.read_text()
    assert "ModelCheckpoint(save_last=True, save_top_k=-1)" in source, (
        "train.py does not pre-seed callbacks with ModelCheckpoint(save_last=True, save_top_k=-1)"
    )


def test_train_py_knn_callback_wiring_preserved():
    """Regression: the KNNCallback wiring must remain after B3 fix."""
    source = TRAIN_PY.read_text()
    assert "from eval.knn_callback import KNNCallback" in source, (
        "train.py lost the KNNCallback wiring — eval.knn integration is broken"
    )


# ---------------------------------------------------------------------------
# Regression-B5 doc: README CIFAR-10 prep snippet must create both train/+val/
# ---------------------------------------------------------------------------

README = REPO_ROOT / "README.md"


def test_readme_cifar10_prep_creates_both_splits():
    """B5 doc: README CIFAR-10 prep snippet must loop over train AND val splits."""
    source = README.read_text()
    assert 'for split, train_flag in [("train", True), ("val", False)]:' in source, (
        "README CIFAR-10 prep snippet only creates train/ split — val/ split missing"
    )


def test_readme_cifar10_prep_uses_root_variable():
    """B5 doc: README CIFAR-10 prep must use root = Path('data/cifar10_imagefolder')."""
    source = README.read_text()
    assert 'root = Path("data/cifar10_imagefolder")' in source, (
        "README CIFAR-10 prep does not define root variable for two-split loop"
    )


def test_readme_cifar10_prep_does_not_hardcode_train_only():
    """B5 doc: README must NOT have old hardcoded single-split path."""
    source = README.read_text()
    assert 'out = Path("data/cifar10_imagefolder/train")' not in source, (
        "README still has old hardcoded single-split path — val/ split missing"
    )


def test_readme_explanatory_line_points_at_parent_dir():
    """B5 doc: README explanatory line must point --data-dir at parent (not .../train)."""
    source = README.read_text()
    assert "--data-dir data/cifar10_imagefolder/train" not in source, (
        "README still instructs users to pass --data-dir data/cifar10_imagefolder/train"
    )
    assert "auto-detects the" in source and "train/" in source and "val/" in source, (
        "README explanatory text does not mention auto-detection of train/+val/ splits"
    )
