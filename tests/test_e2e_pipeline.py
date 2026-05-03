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
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = REPO_ROOT / "train.py"
EVAL_DIR = REPO_ROOT / "eval"


def _smoke_train(tmp_path: Path, data_dir: Path) -> Path:
    """Helper: run train.py once on a toy override of simclr_v1_resnet18.yaml.

    Args:
        tmp_path: per-test temp dir (from pytest fixture).
        data_dir: ImageFolder root with train/+val/ subdirs (from
            tmp_imagefolder_with_val fixture).

    Returns:
        Path to the produced `last.ckpt` file (proves plan 10.1-02 worked).
    """
    src = REPO_ROOT / "configs" / "simclr_v1_resnet18.yaml"
    raw = yaml.safe_load(src.read_text())
    raw.update({
        "max_epochs": 1,
        "batch_size": 4,
        "num_workers": 0,
        "n_views": 2,
        "data_dir": str(data_dir),
    })
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))

    train_log_dir = tmp_path / "train_logs"
    train_log_dir.mkdir()
    result = subprocess.run(
        [sys.executable, str(TRAIN_PY), "--config", str(cfg_path)],
        cwd=str(train_log_dir),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"train.py exited {result.returncode}\n"
        f"stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )
    # Plan 10.1-02 added ModelCheckpoint(save_last=True); last.ckpt MUST exist.
    last_ckpt = train_log_dir / "lightning_logs" / "version_0" / "checkpoints" / "last.ckpt"
    assert last_ckpt.exists(), (
        f"last.ckpt missing after train.py; got: "
        f"{list(train_log_dir.rglob('*.ckpt'))}"
    )
    return last_ckpt


def _run_eval(script: str, cfg_path: Path, ckpt: Path, extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    """Helper: run an eval script via subprocess and return the result for assertion."""
    cmd = [
        sys.executable, str(EVAL_DIR / script),
        str(cfg_path), "--ckpt", str(ckpt), "--device", "cpu",
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180,
    )


# ---------------------------------------------------------------------------
# Regression-B1: every eval script's --help exits 0 (sys.path bootstrap fix)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "script",
    ["linear_probe.py", "finetune.py", "tsne_vis.py", "umap_vis.py", "cam_vis.py"],
)
def test_each_eval_script_help_exits_zero(script: str) -> None:
    """Regression for B1: every eval script imports successfully via `python eval/X.py --help`."""
    script_path = EVAL_DIR / script
    assert script_path.exists(), f"{script_path} does not exist"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"eval/{script} --help exited {result.returncode}\n"
        f"stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )
    # B1 sanity: a clean argparse --help writes "usage:" to stdout
    assert "usage:" in result.stdout.lower(), (
        f"eval/{script} --help did not print usage; stdout=\n{result.stdout}"
    )


# ---------------------------------------------------------------------------
# Regression-B2: no SSLDataModule(cfg) misuse anywhere in eval/*.py
# ---------------------------------------------------------------------------

def test_no_ssldatamodule_cfg_misuse() -> None:
    """Regression for B2: greps eval/*.py source for `SSLDataModule(cfg)` mis-construction.

    Comment-aware: lines starting with `#` or `"` are skipped so docstrings and
    inline comments (like the explanatory ones added by plan 10.1-03 task 1)
    do not self-invalidate this gate.
    """
    eval_dir = REPO_ROOT / "eval"
    bad: list[str] = []
    for py in sorted(eval_dir.glob("*.py")):
        for i, line in enumerate(py.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"'):
                continue
            # bare bug pattern only — `SSLDataModule(data_dir=cfg.data_dir, ...)` is fine
            if stripped.startswith("dm = SSLDataModule(cfg)") or stripped == "SSLDataModule(cfg)":
                bad.append(f"{py.name}:{i}: {stripped}")
    assert not bad, "SSLDataModule(cfg) misuse found:\n" + "\n".join(bad)


# ---------------------------------------------------------------------------
# SC-1: train.py runs end-to-end on toy config + produces a checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_train_produces_checkpoint(tmp_path: Path, tmp_imagefolder_with_val: Path) -> None:
    """SC-1: train.py runs end-to-end on a toy CIFAR-10-shaped config and produces a checkpoint."""
    last_ckpt = _smoke_train(tmp_path, tmp_imagefolder_with_val)
    assert last_ckpt.exists()
    # Also assert at least one epoch=*-step=*.ckpt file (save_top_k=-1 saves every epoch)
    epoch_ckpts = list(last_ckpt.parent.glob("epoch=*-step=*.ckpt"))
    assert epoch_ckpts, f"no epoch=*.ckpt files in {last_ckpt.parent}"


# ---------------------------------------------------------------------------
# SC-2a-e: each eval script accepts a checkpoint produced by train.py
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_linear_probe_accepts_train_checkpoint(
    tmp_path: Path, tmp_imagefolder_with_val: Path
) -> None:
    """SC-2a: eval/linear_probe.py accepts a checkpoint produced by train.py."""
    last_ckpt = _smoke_train(tmp_path, tmp_imagefolder_with_val)
    cfg_path = tmp_path / "smoke.yaml"  # written by _smoke_train
    result = _run_eval("linear_probe.py", cfg_path, last_ckpt)
    assert result.returncode == 0, (
        f"linear_probe.py exited {result.returncode}\n"
        f"stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )


@pytest.mark.slow
def test_tsne_accepts_train_checkpoint(
    tmp_path: Path, tmp_imagefolder_with_val: Path
) -> None:
    """SC-2b: eval/tsne_vis.py accepts a checkpoint produced by train.py.

    Verifies plan 10.1-04 task 1: B4 perplexity clamp keeps tsne_vis.py from
    crashing on the 15-train-image fixture (default perplexities=[10,30,50]
    would otherwise exceed n_samples).

    Note: tsne_vis.py uses ImageFolder(root=cfg.data_dir) directly, so data_dir
    must point to the flat class-dir root (train/), not the train/+val/ parent.
    """
    last_ckpt = _smoke_train(tmp_path, tmp_imagefolder_with_val)
    # tsne_vis.py uses ImageFolder(root=data_dir) — needs flat class dirs, not train/+val/ parent
    src = REPO_ROOT / "configs" / "simclr_v1_resnet18.yaml"
    raw = yaml.safe_load(src.read_text())
    raw.update({
        "max_epochs": 1,
        "batch_size": 4,
        "num_workers": 0,
        "n_views": 2,
        "data_dir": str(tmp_imagefolder_with_val / "train"),
    })
    vis_cfg_path = tmp_path / "vis_smoke.yaml"
    vis_cfg_path.write_text(yaml.safe_dump(raw))
    result = _run_eval("tsne_vis.py", vis_cfg_path, last_ckpt)
    assert result.returncode == 0, (
        f"tsne_vis.py exited {result.returncode}\n"
        f"stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )


@pytest.mark.slow
def test_umap_accepts_train_checkpoint(
    tmp_path: Path, tmp_imagefolder_with_val: Path
) -> None:
    """SC-2c: eval/umap_vis.py accepts a checkpoint produced by train.py.

    Note: umap_vis.py uses ImageFolder(root=cfg.data_dir) directly, so data_dir
    must point to the flat class-dir root (train/), not the train/+val/ parent.
    """
    last_ckpt = _smoke_train(tmp_path, tmp_imagefolder_with_val)
    # umap_vis.py uses ImageFolder(root=data_dir) — needs flat class dirs, not train/+val/ parent
    src = REPO_ROOT / "configs" / "simclr_v1_resnet18.yaml"
    raw = yaml.safe_load(src.read_text())
    raw.update({
        "max_epochs": 1,
        "batch_size": 4,
        "num_workers": 0,
        "n_views": 2,
        "data_dir": str(tmp_imagefolder_with_val / "train"),
    })
    vis_cfg_path = tmp_path / "vis_smoke.yaml"
    vis_cfg_path.write_text(yaml.safe_dump(raw))
    result = _run_eval("umap_vis.py", vis_cfg_path, last_ckpt)
    assert result.returncode == 0, (
        f"umap_vis.py exited {result.returncode}\n"
        f"stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )


@pytest.mark.slow
def test_finetune_accepts_train_checkpoint(
    tmp_path: Path, tmp_imagefolder_with_val: Path
) -> None:
    """SC-2d: eval/finetune.py accepts a checkpoint produced by train.py."""
    last_ckpt = _smoke_train(tmp_path, tmp_imagefolder_with_val)
    cfg_path = tmp_path / "smoke.yaml"
    result = _run_eval("finetune.py", cfg_path, last_ckpt)
    assert result.returncode == 0, (
        f"finetune.py exited {result.returncode}\n"
        f"stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )


@pytest.mark.slow
def test_cam_accepts_train_checkpoint(
    tmp_path: Path, tmp_imagefolder_with_val: Path
) -> None:
    """SC-2e: eval/cam_vis.py accepts a checkpoint produced by train.py."""
    last_ckpt = _smoke_train(tmp_path, tmp_imagefolder_with_val)
    cfg_path = tmp_path / "smoke.yaml"
    result = _run_eval("cam_vis.py", cfg_path, last_ckpt)
    assert result.returncode == 0, (
        f"cam_vis.py exited {result.returncode}\n"
        f"stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )


# ---------------------------------------------------------------------------
# SC-3: full pipeline (train.py -> eval/linear_probe.py) completes
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_full_pipeline_train_then_probe(
    tmp_path: Path, tmp_imagefolder_with_val: Path
) -> None:
    """SC-3: full pipeline train.py -> eval/linear_probe.py completes successfully.

    Stronger assertion than SC-2a: also requires the canonical "Final val/acc:"
    end-of-run line in linear_probe.py stdout, proving the script ran the full
    feature-extract + linear-head training loop, not just argparse + setup.
    """
    last_ckpt = _smoke_train(tmp_path, tmp_imagefolder_with_val)
    cfg_path = tmp_path / "smoke.yaml"
    result = _run_eval("linear_probe.py", cfg_path, last_ckpt)
    assert result.returncode == 0, (
        f"linear_probe.py exited {result.returncode}\n"
        f"stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )
    assert "Final val/acc:" in result.stdout, (
        f"linear_probe.py did not print final val/acc; full stdout:\n{result.stdout}"
    )
