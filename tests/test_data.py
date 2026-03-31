"""Tests for ContrastiveAugmentation, MultiViewTransform, and SSLDataModule."""
import numpy as np
import pytest
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pil_image(h: int = 64, w: int = 64) -> Image.Image:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# ContrastiveAugmentation tests
# ---------------------------------------------------------------------------

class TestContrastiveAugmentation:
    def test_strong_output_shape(self):
        """Strong augmentation on a PIL image returns tensor [3, 224, 224]."""
        from core.data import ContrastiveAugmentation
        aug = ContrastiveAugmentation(size=224, strong=True)
        img = make_pil_image(64, 64)
        out = aug(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 224, 224)

    def test_strong_contains_expected_transforms(self):
        """Strong path includes ColorJitter, GaussianBlur, RandomGrayscale, RandomHorizontalFlip."""
        from core.data import ContrastiveAugmentation
        from torchvision.transforms import v2
        aug = ContrastiveAugmentation(size=224, strong=True)
        transform_types = [type(t) for t in aug.transform.transforms]
        # Unwrap RandomApply containers to check inner transforms
        all_types = []
        for t in aug.transform.transforms:
            if isinstance(t, v2.RandomApply):
                for inner in t.transforms:
                    all_types.append(type(inner))
            else:
                all_types.append(type(t))
        assert v2.ColorJitter in all_types, "Strong path must include ColorJitter"
        assert v2.GaussianBlur in all_types, "Strong path must include GaussianBlur"
        assert v2.RandomGrayscale in transform_types or v2.RandomGrayscale in all_types, (
            "Strong path must include RandomGrayscale"
        )
        assert v2.RandomHorizontalFlip in transform_types or v2.RandomHorizontalFlip in all_types, (
            "Strong path must include RandomHorizontalFlip"
        )

    def test_weak_no_gaussianblur(self):
        """Weak path does NOT include GaussianBlur or strong color jitter."""
        from core.data import ContrastiveAugmentation
        from torchvision.transforms import v2
        aug = ContrastiveAugmentation(size=224, strong=False)
        all_types = []
        for t in aug.transform.transforms:
            if isinstance(t, v2.RandomApply):
                for inner in t.transforms:
                    all_types.append(type(inner))
            else:
                all_types.append(type(t))
        assert v2.GaussianBlur not in all_types, "Weak path must NOT include GaussianBlur"

    def test_weak_output_shape(self):
        """Weak augmentation on a PIL image returns tensor [3, 224, 224]."""
        from core.data import ContrastiveAugmentation
        aug = ContrastiveAugmentation(size=224, strong=False)
        img = make_pil_image(64, 64)
        out = aug(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 224, 224)


# ---------------------------------------------------------------------------
# MultiViewTransform tests
# ---------------------------------------------------------------------------

class TestMultiViewTransform:
    def test_returns_two_views(self):
        """MultiViewTransform with n_views=2 returns a list of 2 tensors."""
        from core.data import ContrastiveAugmentation, MultiViewTransform
        aug = ContrastiveAugmentation(size=32, strong=True)
        mvt = MultiViewTransform(aug, n_views=2)
        img = make_pil_image(64, 64)
        views = mvt(img)
        assert isinstance(views, list)
        assert len(views) == 2
        for v in views:
            assert isinstance(v, torch.Tensor)
            assert v.shape == (3, 32, 32)

    def test_returns_eight_views(self):
        """MultiViewTransform with n_views=8 returns a list of 8 tensors."""
        from core.data import ContrastiveAugmentation, MultiViewTransform
        aug = ContrastiveAugmentation(size=32, strong=True)
        mvt = MultiViewTransform(aug, n_views=8)
        img = make_pil_image(64, 64)
        views = mvt(img)
        assert isinstance(views, list)
        assert len(views) == 8
        for v in views:
            assert v.shape == (3, 32, 32)

    def test_views_are_different(self):
        """Views from MultiViewTransform should not all be identical (randomness test)."""
        from core.data import ContrastiveAugmentation, MultiViewTransform
        aug = ContrastiveAugmentation(size=32, strong=True)
        mvt = MultiViewTransform(aug, n_views=4)
        img = make_pil_image(64, 64)
        views = mvt(img)
        # Check that not all views are the same (randomness with high probability)
        all_same = all(torch.allclose(views[0], views[i]) for i in range(1, 4))
        assert not all_same, "Views should differ due to random augmentation"


# ---------------------------------------------------------------------------
# SSLDataModule tests
# ---------------------------------------------------------------------------

class TestSSLDataModule:
    def test_n_views_2_batch_shape(self, tmp_imagefolder):
        """SSLDataModule with n_views=2, batch_size=4 yields views of shape [2, 4, 3, 32, 32]."""
        from core.data import SSLDataModule
        dm = SSLDataModule(
            data_dir=str(tmp_imagefolder),
            n_views=2,
            batch_size=4,
            num_workers=0,
            size=32,
            strong=True,
        )
        dm.setup()
        loader = dm.train_dataloader()
        views, labels = next(iter(loader))
        assert views.shape == (2, 4, 3, 32, 32), f"Expected [2,4,3,32,32] got {views.shape}"

    def test_n_views_8_batch_shape(self, tmp_imagefolder):
        """SSLDataModule with n_views=8, batch_size=4 yields views of shape [8, 4, 3, 32, 32]."""
        from core.data import SSLDataModule
        dm = SSLDataModule(
            data_dir=str(tmp_imagefolder),
            n_views=8,
            batch_size=4,
            num_workers=0,
            size=32,
            strong=True,
        )
        dm.setup()
        loader = dm.train_dataloader()
        views, labels = next(iter(loader))
        assert views.shape == (8, 4, 3, 32, 32), f"Expected [8,4,3,32,32] got {views.shape}"

    def test_labels_shape_and_dtype(self, tmp_imagefolder):
        """Labels tensor has shape [4] and dtype torch.long."""
        from core.data import SSLDataModule
        dm = SSLDataModule(
            data_dir=str(tmp_imagefolder),
            n_views=2,
            batch_size=4,
            num_workers=0,
            size=32,
        )
        dm.setup()
        loader = dm.train_dataloader()
        views, labels = next(iter(loader))
        assert labels.shape == (4,), f"Expected labels shape [4], got {labels.shape}"
        assert labels.dtype == torch.long, f"Expected torch.long, got {labels.dtype}"

    def test_datamodule_finds_classes(self, tmp_imagefolder):
        """SSLDataModule setup correctly finds ImageFolder classes."""
        from core.data import SSLDataModule
        dm = SSLDataModule(
            data_dir=str(tmp_imagefolder),
            n_views=2,
            batch_size=4,
            num_workers=0,
            size=32,
        )
        dm.setup()
        assert hasattr(dm, "train_dataset")
        assert len(dm.train_dataset.classes) == 3, (
            f"Expected 3 classes, found {len(dm.train_dataset.classes)}"
        )
