"""Integration test for the full evaluation pipeline.

Uses a synthetic checkpoint + synthetic ImageFolder (D-05):
1. Create synthetic ImageFolder (3 classes, 30 images, 32x32 PNG via PIL)
2. Initialize SimCLRv1Module with resnet18 backbone (random weights)
3. Save to .ckpt via torch.save in Lightning checkpoint format
4. Run full eval pipeline: k-NN + linear_probe + t-SNE + UMAP + finetune + CAM
5. Assert all output files exist and no exceptions raised

No network access needed -- purely synthetic data and random-weight checkpoint.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data(tmp_path):
    """Create synthetic ImageFolder with train/ and val/ splits.

    Train: 3 classes, 10 images each (30 total), 32x32 RGB PNGs.
    Val:   3 classes, 5 images each (15 total), 32x32 RGB PNGs.

    Returns:
        tmp_path (Path): Root directory containing train/ and val/ subdirs.
    """
    n_classes = 3
    for split, n_images in [("train", 10), ("val", 5)]:
        split_dir = tmp_path / split
        split_dir.mkdir()
        for cls_idx in range(n_classes):
            cls_dir = split_dir / f"class_{cls_idx}"
            cls_dir.mkdir()
            for img_idx in range(n_images):
                arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
                img.save(cls_dir / f"img_{img_idx:02d}.png")
    return tmp_path


@pytest.fixture
def synthetic_checkpoint(tmp_path, synthetic_data):
    """Create a valid Lightning checkpoint from SimCLRv1 with random weights.

    Steps:
    1. Import methods to register all dispatchers.
    2. Build a SimCLRv1Module with resnet18 backbone.
    3. Save checkpoint manually in Lightning format (state_dict + hyper_parameters).

    Returns:
        Tuple of (ckpt_path: Path, cfg: TrainConfig, data_dir: Path).
    """
    import methods  # noqa: F401 -- triggers register_method() calls
    from core.config import TrainConfig
    from core.dispatcher import method_dispatcher

    config_dict = {
        "method": "simclr_v1",
        "backbone": "resnet18",
        "pretrained": False,
        "max_epochs": 1,
        "warmup_epochs": 0,
        "batch_size": 4,
        "data_dir": str(synthetic_data),
        "n_views": 2,
    }
    cfg = TrainConfig.model_validate(config_dict)
    model = method_dispatcher(cfg)

    # Save in Lightning checkpoint format (state_dict + hyper_parameters)
    # Must include 'pytorch-lightning_version' key for load_from_checkpoint to work
    import lightning as L
    ckpt_path = tmp_path / "test_checkpoint.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": {"cfg": cfg.model_dump()},
            "pytorch-lightning_version": L.__version__,
        },
        ckpt_path,
    )

    return ckpt_path, cfg, synthetic_data


# ---------------------------------------------------------------------------
# Helper: load backbone from checkpoint
# ---------------------------------------------------------------------------

def _load_model_from_checkpoint(ckpt_path: Path, cfg):
    """Load the SimCLRv1Module from a manually-saved checkpoint."""
    from core.dispatcher import get_method

    MethodClass = get_method(cfg.method)
    # Use load_from_checkpoint for proper Lightning loading
    model = MethodClass.load_from_checkpoint(str(ckpt_path), cfg=cfg)
    model.eval()
    return model


def _make_imagenet_transform():
    """Return a standard ImageNet-normalized transform for 32x32 images."""
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _build_dataloader(data_dir: Path, split: str, batch_size: int = 4):
    """Build a simple DataLoader from a synthetic ImageFolder split."""
    ds = ImageFolder(
        root=str(data_dir / split),
        transform=_make_imagenet_transform(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def _extract_features_simple(backbone, dataloader, device="cpu"):
    """Extract L2-normalized features from backbone using a dataloader."""
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in dataloader:
            feats = backbone(imgs.to(device))
            feats = F.normalize(feats, dim=1)
            all_feats.append(feats.cpu())
            all_labels.append(labels.cpu())
    features = torch.cat(all_feats)
    labels = torch.cat(all_labels)
    return features, labels


# ---------------------------------------------------------------------------
# Test: FOUND-08 — EvalConfig schema exists and is operational
# ---------------------------------------------------------------------------

def test_eval_config_schema_exists():
    """Verify FOUND-08: EvalConfig and all 6 sub-configs are importable and constructable."""
    from core.config import (
        EvalConfig,
        LinearProbeConfig,
        KNNConfig,
        TSNEConfig,
        UMAPConfig,
        FinetuneConfig,
        CAMConfig,
    )

    # All 6 sub-configs should be importable
    assert EvalConfig is not None
    assert LinearProbeConfig is not None
    assert KNNConfig is not None
    assert TSNEConfig is not None
    assert UMAPConfig is not None
    assert FinetuneConfig is not None
    assert CAMConfig is not None

    # EvalConfig should have the 6 optional fields
    cfg = EvalConfig(
        knn=KNNConfig(),
        linear_probe=LinearProbeConfig(),
        tsne=TSNEConfig(),
        umap=UMAPConfig(),
        finetune=FinetuneConfig(),
        cam=CAMConfig(),
    )
    assert cfg.knn is not None
    assert cfg.linear_probe is not None
    assert cfg.tsne is not None
    assert cfg.umap is not None
    assert cfg.finetune is not None
    assert cfg.cam is not None


# ---------------------------------------------------------------------------
# Test: k-NN evaluation on synthetic checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_knn_on_synthetic(synthetic_checkpoint):
    """Test k-NN evaluation runs without crash and returns a valid accuracy."""
    from eval.knn_callback import knn_predict

    ckpt_path, cfg, data_dir = synthetic_checkpoint
    model = _load_model_from_checkpoint(ckpt_path, cfg)

    train_loader = _build_dataloader(data_dir, "train", batch_size=5)
    val_loader = _build_dataloader(data_dir, "val", batch_size=5)

    train_feats, train_labels = _extract_features_simple(model.backbone, train_loader)
    val_feats, val_labels = _extract_features_simple(model.backbone, val_loader)

    num_classes = int(train_labels.max().item()) + 1
    k = min(5, len(train_labels))

    acc = knn_predict(
        train_feats,
        train_labels,
        val_feats,
        val_labels,
        k=k,
        temperature=0.07,
        num_classes=num_classes,
    )

    # D-05: relaxed threshold — just check it runs without crash
    assert acc >= 0.0


# ---------------------------------------------------------------------------
# Test: Linear probe on synthetic checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_linear_probe_on_synthetic(synthetic_checkpoint, tmp_path):
    """Test linear probe training completes and cache files use checkpoint-keyed names."""
    import lightning as L
    from torch.utils.data import TensorDataset
    from eval.linear_probe import extract_and_cache, LinearProbeModule
    from core.config import LinearProbeConfig

    ckpt_path, cfg, data_dir = synthetic_checkpoint
    model = _load_model_from_checkpoint(ckpt_path, cfg)
    model.backbone.requires_grad_(False)

    cache_dir = tmp_path / "probe_cache"
    train_loader = _build_dataloader(data_dir, "train", batch_size=5)

    # Extract and cache features
    train_feats, train_labels = extract_and_cache(
        model.backbone,
        train_loader,
        cache_dir,
        "train",
        "cpu",
        str(ckpt_path),
    )

    # D-04: Verify checkpoint-keyed cache filenames
    ckpt_stem = ckpt_path.stem
    assert (cache_dir / f"{ckpt_stem}_features_train.pt").exists(), (
        f"Expected cache file {ckpt_stem}_features_train.pt not found in {cache_dir}"
    )
    assert (cache_dir / f"{ckpt_stem}_labels_train.pt").exists(), (
        f"Expected cache file {ckpt_stem}_labels_train.pt not found in {cache_dir}"
    )

    # Train linear probe for 2 epochs to check it runs
    feat_dim = train_feats.shape[1]
    num_classes = int(train_labels.max().item()) + 1
    lp_cfg = LinearProbeConfig(max_epochs=2, lr=0.1, milestones=[1])

    lp_module = LinearProbeModule(feat_dim, num_classes, lp_cfg)
    train_ds = TensorDataset(train_feats, train_labels)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)

    trainer = L.Trainer(
        max_epochs=2,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    trainer.fit(lp_module, train_dl)


# ---------------------------------------------------------------------------
# Test: t-SNE visualization on synthetic checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_tsne_on_synthetic(synthetic_checkpoint, tmp_path):
    """Test t-SNE visualization produces PNGs with perplexity in filename."""
    from eval.tsne_vis import run_tsne

    ckpt_path, cfg, data_dir = synthetic_checkpoint
    model = _load_model_from_checkpoint(ckpt_path, cfg)

    train_loader = _build_dataloader(data_dir, "train", batch_size=5)
    features, labels = _extract_features_simple(model.backbone, train_loader)
    features_np = features.numpy()
    labels_np = labels.numpy()

    output_dir = tmp_path / "tsne_out"
    # Use only 1 perplexity and small n_samples to minimize runtime (T-09-12)
    paths = run_tsne(features_np, labels_np, perplexities=[5], output_dir=output_dir)

    assert len(paths) == 1, f"Expected 1 PNG file, got {len(paths)}"
    assert paths[0].exists(), f"t-SNE output file does not exist: {paths[0]}"
    assert "perp" in paths[0].name, f"Expected 'perp' in filename: {paths[0].name}"
    assert paths[0].stat().st_size > 0, "t-SNE output file is empty"


# ---------------------------------------------------------------------------
# Test: UMAP visualization on synthetic checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_umap_on_synthetic(synthetic_checkpoint, tmp_path):
    """Test UMAP visualization produces a PNG and returns reducer."""
    from eval.umap_vis import run_umap

    ckpt_path, cfg, data_dir = synthetic_checkpoint
    model = _load_model_from_checkpoint(ckpt_path, cfg)

    train_loader = _build_dataloader(data_dir, "train", batch_size=5)
    features, labels = _extract_features_simple(model.backbone, train_loader)
    features_np = features.numpy()
    labels_np = labels.numpy()

    output_dir = tmp_path / "umap_out"
    path, reducer = run_umap(features_np, labels_np, "cosine", output_dir)

    assert path.exists(), f"UMAP output file does not exist: {path}"
    assert path.stat().st_size > 0, "UMAP output file is empty"
    assert reducer is not None, "UMAP reducer should be returned"


# ---------------------------------------------------------------------------
# Test: Fine-tuning on synthetic checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_finetune_on_synthetic(synthetic_checkpoint, tmp_path):
    """Test fine-tuning training completes without errors on synthetic data."""
    import lightning as L
    from eval.finetune import FinetuneModule
    from core.config import FinetuneConfig

    ckpt_path, cfg, data_dir = synthetic_checkpoint
    model = _load_model_from_checkpoint(ckpt_path, cfg)
    backbone = model.backbone
    feat_dim = backbone.num_features
    num_classes = 3

    ft_cfg = FinetuneConfig(backbone_lr=1e-4, head_lr=1e-3, freeze_bn=True)
    ft_module = FinetuneModule(
        backbone=backbone,
        feat_dim=feat_dim,
        num_classes=num_classes,
        ft_cfg=ft_cfg,
        max_epochs=1,
        warmup_epochs=0,
    )

    # Use a standard ImageFolder train split for fine-tuning
    train_loader = _build_dataloader(data_dir, "train", batch_size=5)

    trainer = L.Trainer(
        max_epochs=1,
        max_steps=2,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    trainer.fit(ft_module, train_loader)


# ---------------------------------------------------------------------------
# Test: CAM visualization on synthetic checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cam_on_synthetic(synthetic_checkpoint, tmp_path):
    """Test CAM visualization produces PNG overlay files."""
    from eval.cam_vis import run_cam
    from core.config import CAMConfig

    ckpt_path, cfg, data_dir = synthetic_checkpoint
    model = _load_model_from_checkpoint(ckpt_path, cfg)

    # Load 4 reference images as numpy arrays from train split
    images = []
    train_dir = data_dir / "train"
    for class_dir in sorted(train_dir.iterdir()):
        if class_dir.is_dir():
            for img_path in sorted(class_dir.iterdir()):
                arr = np.array(Image.open(img_path).convert("RGB"))
                images.append(arr)
                if len(images) >= 4:
                    break
        if len(images) >= 4:
            break

    output_dir = tmp_path / "cam_out"
    cam_cfg = CAMConfig(n_images=4, method="eigencam")
    paths = run_cam(model, "resnet18", images, cam_cfg, output_dir)

    assert len(paths) > 0, "CAM should produce at least one overlay PNG"
    for p in paths:
        assert p.exists(), f"CAM output file does not exist: {p}"
        assert p.stat().st_size > 0, f"CAM output file is empty: {p}"


# ---------------------------------------------------------------------------
# Test: Full pipeline — all 6 components in sequence
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_full_pipeline(synthetic_checkpoint, tmp_path):
    """Run all 6 eval components in sequence on the same synthetic checkpoint.

    Verifies end-to-end that kNN, linear probe, t-SNE, UMAP, finetune, and CAM
    all complete without exceptions and produce expected output files.
    """
    import lightning as L
    from torch.utils.data import TensorDataset
    from eval.knn_callback import knn_predict
    from eval.linear_probe import extract_and_cache, LinearProbeModule
    from eval.tsne_vis import run_tsne
    from eval.umap_vis import run_umap
    from eval.finetune import FinetuneModule
    from eval.cam_vis import run_cam
    from core.config import (
        KNNConfig, LinearProbeConfig, TSNEConfig,
        UMAPConfig, FinetuneConfig, CAMConfig,
    )

    ckpt_path, cfg, data_dir = synthetic_checkpoint
    model = _load_model_from_checkpoint(ckpt_path, cfg)
    backbone = model.backbone
    device = "cpu"

    train_loader = _build_dataloader(data_dir, "train", batch_size=5)
    val_loader = _build_dataloader(data_dir, "val", batch_size=5)

    # -------------------------------------------------------------------------
    # 1. k-NN evaluation
    # -------------------------------------------------------------------------
    train_feats, train_labels = _extract_features_simple(backbone, train_loader)
    val_feats, val_labels = _extract_features_simple(backbone, val_loader)
    num_classes = int(train_labels.max().item()) + 1
    k = min(5, len(train_labels))

    acc = knn_predict(
        train_feats, train_labels, val_feats, val_labels,
        k=k, temperature=0.07, num_classes=num_classes,
    )
    assert acc >= 0.0  # D-05: relaxed threshold

    # -------------------------------------------------------------------------
    # 2. Linear probe — verify checkpoint-keyed cache files (D-04)
    # -------------------------------------------------------------------------
    backbone.requires_grad_(False)
    cache_dir = tmp_path / "pipeline_cache"
    probe_train_feats, probe_train_labels = extract_and_cache(
        backbone, train_loader, cache_dir, "train", device, str(ckpt_path),
    )
    probe_val_feats, probe_val_labels = extract_and_cache(
        backbone, val_loader, cache_dir, "val", device, str(ckpt_path),
    )

    # Verify checkpoint-keyed filenames
    ckpt_stem = ckpt_path.stem
    assert (cache_dir / f"{ckpt_stem}_features_train.pt").exists()
    assert (cache_dir / f"{ckpt_stem}_labels_train.pt").exists()
    assert (cache_dir / f"{ckpt_stem}_features_val.pt").exists()
    assert (cache_dir / f"{ckpt_stem}_labels_val.pt").exists()

    feat_dim = probe_train_feats.shape[1]
    lp_cfg = LinearProbeConfig(max_epochs=1, lr=0.1, milestones=[])
    lp_module = LinearProbeModule(feat_dim, num_classes, lp_cfg)
    train_ds = TensorDataset(probe_train_feats, probe_train_labels)
    val_ds = TensorDataset(probe_val_feats, probe_val_labels)
    train_dl = DataLoader(train_ds, batch_size=8)
    val_dl = DataLoader(val_ds, batch_size=8)
    probe_trainer = L.Trainer(
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    probe_trainer.fit(lp_module, train_dl, val_dl)

    # -------------------------------------------------------------------------
    # 3. t-SNE visualization
    # -------------------------------------------------------------------------
    tsne_dir = tmp_path / "pipeline_tsne"
    tsne_paths = run_tsne(
        train_feats.numpy(), train_labels.numpy(),
        perplexities=[5],
        output_dir=tsne_dir,
    )
    assert len(tsne_paths) == 1
    assert tsne_paths[0].exists()

    # -------------------------------------------------------------------------
    # 4. UMAP visualization
    # -------------------------------------------------------------------------
    umap_dir = tmp_path / "pipeline_umap"
    umap_path, reducer = run_umap(
        train_feats.numpy(), train_labels.numpy(),
        metric="cosine",
        output_dir=umap_dir,
    )
    assert umap_path.exists()
    assert reducer is not None

    # -------------------------------------------------------------------------
    # 5. Fine-tuning
    # -------------------------------------------------------------------------
    backbone.requires_grad_(True)
    ft_cfg = FinetuneConfig(backbone_lr=1e-4, head_lr=1e-3, freeze_bn=True)
    ft_module = FinetuneModule(
        backbone=backbone,
        feat_dim=feat_dim,
        num_classes=num_classes,
        ft_cfg=ft_cfg,
        max_epochs=1,
        warmup_epochs=0,
    )
    ft_trainer = L.Trainer(
        max_epochs=1,
        max_steps=2,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    ft_trainer.fit(ft_module, train_loader)

    # -------------------------------------------------------------------------
    # 6. CAM visualization
    # -------------------------------------------------------------------------
    images = []
    train_dir = data_dir / "train"
    for class_dir in sorted(train_dir.iterdir()):
        if class_dir.is_dir():
            for img_path in sorted(class_dir.iterdir()):
                arr = np.array(Image.open(img_path).convert("RGB"))
                images.append(arr)
                if len(images) >= 4:
                    break
        if len(images) >= 4:
            break

    cam_dir = tmp_path / "pipeline_cam"
    cam_cfg = CAMConfig(n_images=4, method="eigencam")
    cam_paths = run_cam(model, "resnet18", images, cam_cfg, cam_dir)
    assert len(cam_paths) > 0
    for p in cam_paths:
        assert p.exists()

    # -------------------------------------------------------------------------
    # Final assertion: all output directories contain expected files
    # -------------------------------------------------------------------------
    assert tsne_dir.exists() and any(tsne_dir.iterdir())
    assert umap_dir.exists() and any(umap_dir.iterdir())
    assert cam_dir.exists() and any(cam_dir.iterdir())
