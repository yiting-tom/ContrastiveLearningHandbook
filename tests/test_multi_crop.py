"""Tests for MultiCropDataset and ssl_collate_multi_crop."""
import numpy as np
import pytest
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_imagefolder_dataset(tmp_path, n_classes=2, n_images=5, img_size=64):
    """Create a temp ImageFolder directory and return a torchvision ImageFolder
    (without any transform) suitable for wrapping with MultiCropDataset."""
    from torchvision.datasets import ImageFolder

    for cls_idx in range(n_classes):
        cls_dir = tmp_path / f"class_{cls_idx}"
        cls_dir.mkdir()
        for img_idx in range(n_images):
            arr = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            img.save(cls_dir / f"img_{img_idx:02d}.jpg")


    # No transform — MultiCropDataset applies its own augmentation
    return ImageFolder(str(tmp_path))


# ---------------------------------------------------------------------------
# MultiCropDataset tests
# ---------------------------------------------------------------------------

class TestMultiCropDataset:
    def test_returns_8_crops_and_label(self, tmp_path):
        """MultiCropDataset with n_large_crops=2, n_small_crops=6 returns (crops_list, label) with 8 crops."""
        from core.data import MultiCropDataset
        base_ds = make_imagefolder_dataset(tmp_path)
        ds = MultiCropDataset(
            base_ds,
            n_large_crops=2,
            large_size=224,
            n_small_crops=6,
            small_size=96,
        )
        crops, label = ds[0]
        assert isinstance(crops, list), "crops should be a list"
        assert len(crops) == 8, f"Expected 8 crops, got {len(crops)}"
        assert isinstance(label, (int, torch.Tensor)), "label should be int or tensor"

    def test_large_crops_are_224_small_crops_are_96(self, tmp_path):
        """First 2 crops are [C, 224, 224], last 6 are [C, 96, 96]."""
        from core.data import MultiCropDataset
        base_ds = make_imagefolder_dataset(tmp_path)
        ds = MultiCropDataset(
            base_ds,
            n_large_crops=2,
            large_size=224,
            n_small_crops=6,
            small_size=96,
        )
        crops, _ = ds[0]
        for i, crop in enumerate(crops[:2]):
            assert crop.shape == (3, 224, 224), f"Large crop {i} shape {crop.shape} != (3, 224, 224)"
        for i, crop in enumerate(crops[2:]):
            assert crop.shape == (3, 96, 96), f"Small crop {i} shape {crop.shape} != (3, 96, 96)"

    def test_ssl_collate_multi_crop_shapes(self, tmp_path):
        """ssl_collate_multi_crop returns list of 8 stacked tensors + labels."""
        from core.data import MultiCropDataset, ssl_collate_multi_crop
        base_ds = make_imagefolder_dataset(tmp_path)
        ds = MultiCropDataset(
            base_ds,
            n_large_crops=2,
            large_size=224,
            n_small_crops=6,
            small_size=96,
        )
        batch_size = 4
        batch = [ds[i % len(ds)] for i in range(batch_size)]
        crops_list, labels = ssl_collate_multi_crop(batch)
        assert isinstance(crops_list, list), "crops_list should be a list"
        assert len(crops_list) == 8, f"Expected 8 elements, got {len(crops_list)}"
        # First 2: large crops [B, C, 224, 224]
        for i in range(2):
            assert crops_list[i].shape == (batch_size, 3, 224, 224), (
                f"Large crop tensor {i} shape {crops_list[i].shape} != ({batch_size}, 3, 224, 224)"
            )
        # Last 6: small crops [B, C, 96, 96]
        for i in range(2, 8):
            assert crops_list[i].shape == (batch_size, 3, 96, 96), (
                f"Small crop tensor {i} shape {crops_list[i].shape} != ({batch_size}, 3, 96, 96)"
            )
        # Labels
        assert labels.shape == (batch_size,), f"Labels shape {labels.shape} != ({batch_size},)"
        assert labels.dtype == torch.long

    def test_edge_case_zero_small_crops(self, tmp_path):
        """MultiCropDataset with n_small_crops=0 returns exactly 2 crops."""
        from core.data import MultiCropDataset
        base_ds = make_imagefolder_dataset(tmp_path)
        ds = MultiCropDataset(
            base_ds,
            n_large_crops=2,
            large_size=224,
            n_small_crops=0,
            small_size=96,
        )
        crops, label = ds[0]
        assert len(crops) == 2, f"Expected 2 crops, got {len(crops)}"
        for crop in crops:
            assert crop.shape == (3, 224, 224)

    def test_preserves_labels_from_base_dataset(self, tmp_path):
        """MultiCropDataset label matches base dataset label."""
        from core.data import MultiCropDataset
        base_ds = make_imagefolder_dataset(tmp_path, n_classes=3)
        ds = MultiCropDataset(
            base_ds,
            n_large_crops=2,
            large_size=32,
            n_small_crops=2,
            small_size=16,
        )
        for idx in range(min(len(ds), 6)):
            _, label_mc = ds[idx]
            _, label_base = base_ds[idx]
            assert label_mc == label_base, (
                f"Label mismatch at idx {idx}: multi-crop={label_mc}, base={label_base}"
            )

    def test_ssldatamodule_uses_multi_crop_collate(self, tmp_path):
        """SSLDataModule with a pre-wrapped MultiCropDataset uses ssl_collate_multi_crop."""
        from core.data import MultiCropDataset, SSLDataModule, ssl_collate_multi_crop
        from torch.utils.data import DataLoader

        base_ds = make_imagefolder_dataset(tmp_path)
        mc_ds = MultiCropDataset(
            base_ds,
            n_large_crops=2,
            large_size=32,   # small sizes to keep test fast
            n_small_crops=2,
            small_size=16,
        )
        dm = SSLDataModule(
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
            dataset=mc_ds,
        )
        dm.setup()
        loader = dm.train_dataloader()
        # The collate_fn should be ssl_collate_multi_crop
        assert loader.collate_fn is ssl_collate_multi_crop, (
            "SSLDataModule should use ssl_collate_multi_crop when dataset is MultiCropDataset"
        )
        # And the batch should have the right structure
        crops_list, labels = next(iter(loader))
        assert isinstance(crops_list, list)
        assert len(crops_list) == 4  # 2 large + 2 small
