"""DINOv2 Feature Extraction and Evaluation Tutorial.

Demonstrates using pretrained DINOv2 (Oquab et al., 2023) features for:
1. Zero-shot k-NN classification
2. Linear probing

DINOv2 is loaded as a feature extractor via timm -- no from-scratch training
(requires hundreds of GPU-days and the LVD-142M dataset).

Note: Register tokens (added Oct 2023) change the model API. Use the checkpoint
version that matches your timm installation. The correct lineage is:
DINO -> DINOv2 -> DINOv2 + Registers. "DINOv3" does not exist.

Usage:
    python eval/dinov2_demo.py --dataset cifar10
    python eval/dinov2_demo.py --dataset stl10
    python eval/dinov2_demo.py --dataset imagefolder --data-dir /path/to/data
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import timm
from torchvision.datasets import CIFAR10, STL10, ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for the DINOv2 demo script.

    Returns:
        Parsed argument namespace with dataset, data_dir, batch_size, k, and device.
    """
    parser = argparse.ArgumentParser(
        description="DINOv2 Feature Extraction and Evaluation Tutorial"
    )
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "stl10", "imagefolder"],
        default="cifar10",
        help="Dataset to evaluate on (default: cifar10)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root directory for dataset downloads or ImageFolder (default: data)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for feature extraction (default: 256)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="k for k-NN classification (default: 20)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (default: cuda if available, else cpu)",
    )
    return parser.parse_args()


def build_transform() -> Compose:
    """Build ImageNet-style validation transform for DINOv2 input.

    Returns:
        Composed transform: Resize(224) -> CenterCrop(224) -> ToTensor -> Normalize.
    """
    return Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_datasets(args: argparse.Namespace):
    """Load train and test datasets according to --dataset flag.

    Args:
        args: Parsed arguments with dataset, data_dir.

    Returns:
        Tuple of (train_dataset, test_dataset).

    Raises:
        ValueError: If --dataset is 'imagefolder' but --data-dir doesn't exist.
    """
    transform = build_transform()

    if args.dataset == "cifar10":
        train_ds = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
        test_ds = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    elif args.dataset == "stl10":
        train_ds = STL10(root=args.data_dir, split="train", download=True, transform=transform)
        test_ds = STL10(root=args.data_dir, split="test", download=True, transform=transform)
    elif args.dataset == "imagefolder":
        import os
        train_path = os.path.join(args.data_dir, "train")
        test_path = os.path.join(args.data_dir, "val")
        if not os.path.isdir(train_path):
            train_path = args.data_dir
        if not os.path.isdir(test_path):
            test_path = args.data_dir
        train_ds = ImageFolder(root=train_path, transform=transform)
        test_ds = ImageFolder(root=test_path, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset!r}")

    return train_ds, test_ds


def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
):
    """Extract features from a pretrained model over a dataloader.

    Args:
        model: Feature extractor (output is pooled feature vectors, not logits).
        dataloader: DataLoader yielding (images, labels) batches.
        device: Device string ('cpu', 'cuda', etc.).

    Returns:
        Tuple of (features, labels) as numpy arrays.
        features shape: [N, D] where D is the model's feature dimension.
        labels shape: [N]
    """
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            feats = model(imgs.to(device))
            features.append(feats.cpu())
            labels.append(lbls)
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


def run_knn(
    train_features,
    train_labels,
    test_features,
    test_labels,
    k: int,
) -> float:
    """Run k-NN classification on extracted features.

    Args:
        train_features: Training features [N_train, D].
        train_labels: Training labels [N_train].
        test_features: Test features [N_test, D].
        test_labels: Test labels [N_test].
        k: Number of nearest neighbors.

    Returns:
        k-NN accuracy as a float in [0, 1].
    """
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(train_features, train_labels)
    return knn.score(test_features, test_labels)


def run_linear_probe(
    train_features,
    train_labels,
    test_features,
    test_labels,
) -> float:
    """Run linear probing on extracted features using sklearn SGDClassifier.

    Trains a linear classifier on top of frozen DINOv2 features.

    Args:
        train_features: Training features [N_train, D].
        train_labels: Training labels [N_train].
        test_features: Test features [N_test, D].
        test_labels: Test labels [N_test].

    Returns:
        Linear probe accuracy as a float in [0, 1].
    """
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_test = scaler.transform(test_features)

    clf = SGDClassifier(
        loss="hinge",
        max_iter=1000,
        tol=1e-3,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, train_labels)
    return clf.score(X_test, test_labels)


def main() -> None:
    """Run DINOv2 feature extraction and evaluation pipeline.

    Loads DINOv2 ViT-Small via timm, extracts features from the chosen dataset,
    and reports both k-NN and linear probe accuracy.
    """
    args = get_args()
    device = args.device

    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Data dir: {args.data_dir}")

    # Load DINOv2 ViT-Small via timm (D-03)
    # num_classes=0 returns pooled feature vectors instead of logits
    print("Loading DINOv2 vit_small_patch14_dinov2.lvd142m via timm ...")
    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=True,
        num_classes=0,
    )
    model = model.eval().to(device)
    print(f"DINOv2 loaded. Feature dim: {model.num_features}")

    # Load datasets
    print("Loading datasets ...")
    train_ds, test_ds = load_datasets(args)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device != "cpu"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device != "cpu"),
    )

    # Extract features
    print("Extracting training features ...")
    train_features, train_labels = extract_features(model, train_loader, device)
    print(f"Training features: {train_features.shape}")

    print("Extracting test features ...")
    test_features, test_labels = extract_features(model, test_loader, device)
    print(f"Test features: {test_features.shape}")

    # k-NN evaluation
    print(f"Running k-NN (k={args.k}) ...")
    knn_acc = run_knn(train_features, train_labels, test_features, test_labels, k=args.k)
    print(f"k-NN accuracy (k={args.k}): {knn_acc:.4f}")

    # Linear probing
    print("Running linear probe ...")
    probe_acc = run_linear_probe(train_features, train_labels, test_features, test_labels)
    print(f"Linear probe accuracy: {probe_acc:.4f}")


if __name__ == "__main__":
    main()
