"""Generate before/after UMAP demo figures for the evolution slide deck.

Pedagogical illustration for slide 16 ("Live Demo"):
  - BEFORE: random-init ResNet-18 backbone  -> features are ~random, one blob
  - AFTER : ImageNet-pretrained ResNet-18    -> features cluster by class

This is a *proxy* for "untrained vs trained SSL backbone" so the deck has a
clear visual without hours of from-scratch SSL training. Swap in your own
trained checkpoint later via eval/umap_vis.py.

Run:  python tools/make_demo_umap.py
Out:  demo_assets/umap_before.png , demo_assets/umap_after.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

N_SAMPLES = 1500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("demo_assets")
OUT.mkdir(exist_ok=True)

# deck palette
NAVY2 = "#182740"
PALETTE = ["#22D3EE", "#F59E0B", "#F43F5E", "#8B5CF6", "#34D399",
           "#FB7185", "#60A5FA", "#FBBF24", "#A78BFA", "#2DD4BF"]
CMAP = matplotlib.colors.ListedColormap(PALETTE)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_loader() -> DataLoader:
    ds = CIFAR10(root="data/cifar10_raw", train=False, download=True, transform=transform)
    # balanced-ish subset: first N_SAMPLES after a fixed shuffle
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(ds))[:N_SAMPLES]
    return DataLoader(Subset(ds, idx.tolist()), batch_size=128, shuffle=False, num_workers=4)


@torch.no_grad()
def extract(backbone: torch.nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    feats, labs = [], []
    backbone.eval().to(DEVICE)
    for i, (imgs, labels) in enumerate(loader):
        f = backbone(imgs.to(DEVICE))
        f = F.normalize(f, dim=1)
        feats.append(f.cpu())
        labs.append(labels)
        print(f"  batch {i + 1}: {sum(x.shape[0] for x in feats)} samples", flush=True)
    return torch.cat(feats).numpy(), torch.cat(labs).numpy()


def plot(emb: np.ndarray, labels: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(NAVY2)
    ax.set_facecolor(NAVY2)
    ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap=CMAP, s=12, alpha=0.85, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.04)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15, facecolor=NAVY2)
    plt.close(fig)
    print(f"saved {path}", flush=True)


def main() -> None:
    print("loading CIFAR-10 (test split) ...", flush=True)
    loader = get_loader()

    print("extracting BEFORE features (random-init resnet18) ...", flush=True)
    before_bb = timm.create_model("resnet18", pretrained=False, num_classes=0)
    fb, lb = extract(before_bb, loader)

    print("extracting AFTER features (ImageNet-pretrained resnet18) ...", flush=True)
    after_bb = timm.create_model("resnet18", pretrained=True, num_classes=0)
    fa, la = extract(after_bb, loader)

    print("running UMAP (before) ...", flush=True)
    emb_b = umap.UMAP(metric="cosine", random_state=42, n_components=2).fit_transform(fb)
    print("running UMAP (after) ...", flush=True)
    emb_a = umap.UMAP(metric="cosine", random_state=42, n_components=2).fit_transform(fa)

    plot(emb_b, lb, OUT / "umap_before.png")
    plot(emb_a, la, OUT / "umap_after.png")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
