"""SSL augmentation pipeline and data module.

Provides:
    - ContrastiveAugmentation: strong/weak single-view augmentation callable.
    - MultiViewTransform: wraps an augmentation to produce n_views copies.
    - ssl_collate_fn: custom collate for multi-view batches.
    - SSLDataModule: Lightning DataModule wrapping ImageFolder with multi-view support.
"""

import os

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


class SSLDataModule(L.LightningDataModule):
    """Lightning DataModule for SSL pretraining.

    Wraps ImageFolder with multi-view augmentation.
    Each batch yields (views, labels) where views shape is [n_views, B, C, H, W].

    Args:
        data_dir: Path to ImageFolder-style directory. If a 'train/' subdirectory
            exists it is used; otherwise data_dir itself is treated as the root.
        n_views: Number of augmented views per image (2 for SimCLR/MoCo, 8+ for SwAV/DINO).
        batch_size: Batch size per GPU.
        num_workers: DataLoader workers.
        size: Crop size for augmentations.
        strong: Whether to use strong (SimCLR-style) or weak (era-1) augmentation.
    """

    def __init__(
        self,
        data_dir: str,
        n_views: int = 2,
        batch_size: int = 256,
        num_workers: int = 4,
        size: int = 224,
        strong: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.n_views = n_views
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size
        self.strong = strong

    def setup(self, stage=None):
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ssl_collate_fn,
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
