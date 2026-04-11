"""Shared pytest fixtures for the contrastive learning tutorial repo."""
# macOS: FAISS ships its own OpenMP runtime which can conflict with PyTorch's.
# Setting this before any imports prevents a segfault when both are loaded.
# Safe for tutorial use on macOS; no effect on Linux.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def random_tensor():
    """Return a factory that creates random tensors of the given shape."""
    def _make(*shape):
        return torch.randn(*shape)
    return _make


@pytest.fixture
def tmp_imagefolder(tmp_path):
    """Create a temporary ImageFolder with 3 classes, 5 images each (32x32 RGB JPEGs).

    Directory structure:
        tmp_path/class_0/img_00.jpg
        tmp_path/class_0/img_01.jpg
        ...
        tmp_path/class_2/img_04.jpg

    Returns:
        tmp_path (Path): Root of the ImageFolder directory.
    """
    n_classes = 3
    n_images = 5
    for cls_idx in range(n_classes):
        cls_dir = tmp_path / f"class_{cls_idx}"
        cls_dir.mkdir()
        for img_idx in range(n_images):
            arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            img.save(cls_dir / f"img_{img_idx:02d}.jpg")
    return tmp_path


@pytest.fixture
def toy_config_dict():
    """Return a minimal valid TrainConfig dict for testing."""
    return {
        "method": "simclr_v1",
        "backbone": "resnet18",
        "pretrained": False,
        "max_epochs": 10,
        "warmup_epochs": 1,
        "batch_size": 32,
        "lr": 0.3,
        "weight_decay": 1e-6,
        "optimizer": "adamw",
    }
