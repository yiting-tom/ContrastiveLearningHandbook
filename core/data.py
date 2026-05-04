"""SSL augmentation pipeline and data module.

Provides:
    - ContrastiveAugmentation: strong/weak single-view augmentation callable.
    - MultiViewTransform: wraps an augmentation to produce n_views copies.
    - ssl_collate_fn: custom collate for multi-view batches.
    - MultiCropDataset: dataset wrapper for SwAV/DINO-style multi-crop (variable sizes).
    - ssl_collate_multi_crop: collate for multi-crop batches with mixed spatial dims.
    - SSLDataModule: Lightning DataModule wrapping ImageFolder with multi-view support.
"""

import os
import random
from collections import defaultdict
from typing import Iterator

import torch
import lightning as L
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ContrastiveAugmentation:
    """SSL augmentation pipeline.

    Strong path (SimCLR, MoCo v2, BYOL, etc.):
        RandomResizedCrop -> ColorJitter(0.8s, 0.8s, 0.8s, 0.2s) with s=1.0, p=0.8
        -> RandomGrayscale(p=0.2) -> GaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=0.5)
        -> RandomHorizontalFlip -> ToImage -> ToDtype(float32) -> Normalize(imagenet_mean, imagenet_std)

    Weak path (Instance Discrimination, era-1):
        RandomResizedCrop -> ColorJitter(0.4, 0.4, 0.4, 0.1) p=0.8
        -> RandomGrayscale(p=0.2) -> RandomHorizontalFlip -> ToImage -> ToDtype(float32) -> Normalize

    Args:
        size: Crop size (default 224).
        strong: If True, use SimCLR-strength augmentation (s=1.0). If False, use era-1 weak (s=0.4).
    """

    def __init__(self, size: int = 224, strong: bool = True):
        s = 1.0 if strong else 0.4
        transforms_list = [
            v2.RandomResizedCrop(size, scale=(0.2, 1.0)),
            v2.RandomApply([v2.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
            v2.RandomGrayscale(p=0.2),
        ]
        if strong:
            transforms_list.append(
                v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5)
            )
        transforms_list.extend([
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.transform = v2.Compose(transforms_list)

    def __call__(self, img):
        return self.transform(img)


class MultiViewTransform:
    """Apply a base transform n_views times to produce multiple augmented views.

    Args:
        base_transform: Callable that takes a PIL Image and returns a tensor.
        n_views: Number of augmented views to produce per image.
    """

    def __init__(self, base_transform, n_views: int = 2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, img):
        return [self.base_transform(img) for _ in range(self.n_views)]


def ssl_collate_fn(batch):
    """Collate function for SSL multi-view datasets.

    Input: list of (views_list, label) tuples where views_list has n_views tensors.
    Output: (views_tensor, labels_tensor)
        views_tensor shape: [n_views, B, C, H, W]
        labels_tensor shape: [B]
    """
    views, labels = zip(*batch)
    n_views = len(views[0])
    stacked = [torch.stack([v[i] for v in views]) for i in range(n_views)]
    return torch.stack(stacked), torch.tensor(labels, dtype=torch.long)


class IndexedDataset(torch.utils.data.Dataset):
    """Wraps a dataset to also return the sample index.

    Used by Instance Discrimination (and future CMC) to look up and update
    memory bank entries by sample index.

    Args:
        dataset: Base dataset returning (views_list, label) tuples.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return (*data, idx)


def ssl_collate_with_index(batch):
    """Collate function for SSL datasets that also return sample indices.

    Input: list of (views_list, label, index) tuples.
    Output: (views_tensor, labels_tensor, indices_tensor)
        views_tensor shape: [n_views, B, C, H, W]
        labels_tensor shape: [B]
        indices_tensor shape: [B]
    """
    views, labels, indices = zip(*batch)
    n_views = len(views[0])
    stacked = [torch.stack([v[i] for v in views]) for i in range(n_views)]
    return (
        torch.stack(stacked),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(indices, dtype=torch.long),
    )


class MultiCropDataset(torch.utils.data.Dataset):
    """Dataset wrapper for multi-crop augmentation (SwAV/DINO style).

    Applies two sets of augmentations to each image: n_large_crops at large_size
    and n_small_crops at small_size, returning a flat list of crops plus the label.

    The base dataset must return (PIL_Image, label) -- it must NOT apply its own
    transform, as MultiCropDataset applies ContrastiveAugmentation internally.

    This is a reusable INFRA-04 component shared by SwAV (Phase 5) and DINO (Phase 7).

    Args:
        dataset: Base dataset returning (PIL_Image, label) tuples.
        n_large_crops: Number of large-resolution crops to produce.
        large_size: Spatial size for large crops (e.g. 224).
        n_small_crops: Number of small-resolution crops to produce.
        small_size: Spatial size for small crops (e.g. 96).
        strong: Use strong (SimCLR-style) augmentation if True (default).

    Returns:
        (crops, label) where crops is a list of n_large_crops + n_small_crops tensors.
    """

    def __init__(
        self,
        dataset,
        n_large_crops: int,
        large_size: int,
        n_small_crops: int,
        small_size: int,
        strong: bool = True,
    ):
        self.dataset = dataset
        self.n_large_crops = n_large_crops
        self.n_small_crops = n_small_crops
        self.large_aug = ContrastiveAugmentation(size=large_size, strong=strong)
        self.small_aug = ContrastiveAugmentation(size=small_size, strong=strong)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        crops = []
        for _ in range(self.n_large_crops):
            crops.append(self.large_aug(img))
        for _ in range(self.n_small_crops):
            crops.append(self.small_aug(img))
        return crops, label


def ssl_collate_multi_crop(batch):
    """Collate function for multi-crop SSL batches (SwAV/DINO style).

    Unlike ssl_collate_fn, this returns a LIST of tensors (not a single stacked
    tensor) because large and small crops have different spatial dimensions and
    cannot be stacked together.

    Input: list of (crops_list, label) tuples where each crops_list has n_crops tensors.
    Output: (crops_list, labels_tensor)
        crops_list: list of n_crops tensors, each shape [B, C, H, W]
        labels_tensor shape: [B]
    """
    all_crops, labels = zip(*batch)
    n_crops = len(all_crops[0])
    crops_list = [torch.stack([sample[i] for sample in all_crops]) for i in range(n_crops)]
    return crops_list, torch.tensor(labels, dtype=torch.long)


class ClassBalancedSampler(torch.utils.data.Sampler):
    """Batch sampler that guarantees at least ``n_samples_per_class`` instances
    per class in every batch.

    Produces batches of size ``n_classes_per_batch * n_samples_per_class`` by
    sampling ``n_classes_per_batch`` classes uniformly at random (without
    replacement within a batch), then drawing ``n_samples_per_class`` indices
    from each chosen class (with replacement if the class has fewer images).

    Requires the dataset to expose a ``.targets`` attribute (list[int]), which
    is available on ``torchvision.datasets.ImageFolder`` by default.

    Args:
        dataset: An ImageFolder-style dataset with a ``.targets`` attribute.
        n_classes_per_batch: Number of distinct classes per batch.
        n_samples_per_class: Number of samples per class per batch.

    Note:
        Pass this sampler to ``DataLoader(sampler=..., shuffle=False)``.
        ``shuffle=True`` is mutually exclusive with a custom sampler and will
        raise ``ValueError`` at DataLoader construction time.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_classes_per_batch: int,
        n_samples_per_class: int,
    ) -> None:
        super().__init__()
        self.n_classes_per_batch = n_classes_per_batch
        self.n_samples_per_class = n_samples_per_class

        # Build class -> [dataset_indices] map from dataset.targets
        targets = dataset.targets  # list[int] from ImageFolder
        self.class_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(targets):
            self.class_indices[label].append(idx)
        self.classes: list[int] = list(self.class_indices.keys())

        if len(self.classes) < n_classes_per_batch:
            raise ValueError(
                f"Dataset has {len(self.classes)} classes but "
                f"n_classes_per_batch={n_classes_per_batch}. "
                "Reduce n_classes_per_batch to fit the dataset."
            )

        batch_size = n_classes_per_batch * n_samples_per_class
        n_batches = max(1, len(targets) // batch_size)
        self._length = n_batches * batch_size

    def __iter__(self) -> Iterator[int]:
        n_batches = self._length // (self.n_classes_per_batch * self.n_samples_per_class)
        indices: list[int] = []
        for _ in range(n_batches):
            chosen_classes = random.sample(self.classes, self.n_classes_per_batch)
            for cls in chosen_classes:
                cls_idxs = self.class_indices[cls]
                # random.choices samples with replacement — safe for small classes
                chosen = random.choices(cls_idxs, k=self.n_samples_per_class)
                indices.extend(chosen)
        return iter(indices)

    def __len__(self) -> int:
        return self._length


class SSLDataModule(L.LightningDataModule):
    """Lightning DataModule for SSL pretraining.

    Wraps ImageFolder with multi-view augmentation.
    Each batch yields (views, labels) where views shape is [n_views, B, C, H, W].

    When a pre-wrapped MultiCropDataset is passed via the dataset parameter,
    the DataModule skips ImageFolder creation and uses ssl_collate_multi_crop
    automatically.

    Args:
        data_dir: Path to ImageFolder-style directory. If a 'train/' subdirectory
            exists it is used; otherwise data_dir itself is treated as the root.
        n_views: Number of augmented views per image (2 for SimCLR/MoCo, 8+ for SwAV/DINO).
        batch_size: Batch size per GPU.
        num_workers: DataLoader workers.
        size: Crop size for augmentations.
        strong: Whether to use strong (SimCLR-style) or weak (era-1) augmentation.
        dataset: Optional pre-wrapped dataset (e.g. MultiCropDataset). When provided,
            ImageFolder creation and transform wrapping are skipped.
    """

    def __init__(
        self,
        data_dir: str,
        n_views: int = 2,
        batch_size: int = 256,
        num_workers: int = 4,
        size: int = 224,
        strong: bool = True,
        dataset: torch.utils.data.Dataset | None = None,
        sampler_type: str | None = None,
        n_classes_per_batch: int | None = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.n_views = n_views
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size
        self.strong = strong
        self.dataset = dataset
        self.sampler_type = sampler_type
        self.n_classes_per_batch = n_classes_per_batch

    def setup(self, stage=None):
        if self.dataset is not None:
            # Use the pre-wrapped dataset as-is (e.g. MultiCropDataset)
            self.train_dataset = self.dataset
        else:
            augmentation = ContrastiveAugmentation(size=self.size, strong=self.strong)
            transform = MultiViewTransform(augmentation, n_views=self.n_views)

            # Try train/ subdirectory first, fall back to data_dir itself
            train_dir = os.path.join(self.data_dir, "train")
            if not os.path.isdir(train_dir):
                train_dir = self.data_dir
            self.train_dataset = ImageFolder(train_dir, transform=transform)

        val_dir = os.path.join(self.data_dir, "val")
        if os.path.isdir(val_dir):
            # Val uses single-view center crop for evaluation
            val_transform = v2.Compose([
                v2.Resize(self.size + 32),
                v2.CenterCrop(self.size),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
            self.val_dataset = ImageFolder(val_dir, transform=val_transform)
        else:
            self.val_dataset = None

    def train_dataloader(self):
        if isinstance(self.train_dataset, IndexedDataset):
            collate = ssl_collate_with_index
        elif isinstance(self.train_dataset, MultiCropDataset):
            collate = ssl_collate_multi_crop
        else:
            collate = ssl_collate_fn

        if self.sampler_type == "class_balanced":
            if self.n_classes_per_batch is None:
                raise ValueError(
                    "n_classes_per_batch must be set when sampler_type='class_balanced'"
                )
            # n_samples_per_class = batch_size // n_classes_per_batch
            n_samples = max(1, self.batch_size // self.n_classes_per_batch)
            sampler = ClassBalancedSampler(
                self.train_dataset,
                n_classes_per_batch=self.n_classes_per_batch,
                n_samples_per_class=n_samples,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
