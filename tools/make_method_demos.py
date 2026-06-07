"""Per-method live-demo UMAPs for the slide deck.

Quickly trains EACH of the 13 trainable SSL methods on a small CIFAR-10 subset
(CPU, few epochs, reduced crop size) using the repo's own method_dispatcher +
SSLDataModule, then extracts backbone features and renders a UMAP per method.
DINOv2 (no training module) uses a pretrained model for feature extraction.

These are deliberately UNDER-trained (the deck labels them as such): the point
is to show a real per-method run, not a benchmark. Swap in fully-trained
checkpoints later for production figures.

Run:  python tools/make_method_demos.py
Out:  demo_assets/methods/<method>.png
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from PIL import Image
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

import lightning as L

from core.config import load_config
from core.data import SSLDataModule, IndexedDataset, MultiCropDataset, MultiViewTransform, ContrastiveAugmentation
from core.dispatcher import method_dispatcher
import methods  # noqa: F401 -- registers all methods

torch.manual_seed(42)
DEVICE = "cpu"
DATA = Path("data/cifar_demo")
OUT = Path("demo_assets/methods")
OUT.mkdir(parents=True, exist_ok=True)
N_PER_CLASS_TRAIN = 60
N_PER_CLASS_VAL = 50

NAVY2 = "#182740"
PALETTE = ["#22D3EE", "#F59E0B", "#F43F5E", "#8B5CF6", "#34D399",
           "#FB7185", "#60A5FA", "#FBBF24", "#A78BFA", "#2DD4BF"]
CMAP = matplotlib.colors.ListedColormap(PALETTE)

CONFIG = {
    "instance_discrimination": "configs/instance_discrimination_resnet18.yaml",
    "invariant_spread": "configs/invariant_spread_resnet18.yaml",
    "moco_v1": "configs/moco_v1_resnet18.yaml",
    "moco_v2": "configs/moco_v2_resnet18.yaml",
    "simclr_v1": "configs/simclr_v1_resnet18.yaml",
    "simclr_v2": "configs/simclr_v2_resnet18.yaml",
    "swav": "configs/swav_resnet18.yaml",
    "infomin": "configs/infomin_resnet18.yaml",
    "byol": "configs/byol_resnet18.yaml",
    "simsiam": "configs/simsiam_resnet18.yaml",
    "barlow_twins": "configs/barlow_twins_resnet18.yaml",
    "moco_v3": "configs/moco_v3_vit_small.yaml",
    "dino": "configs/dino_vit_small.yaml",
}
# per-method budget: size, batch, epochs, limit_train_batches, optional multicrop (nL,szL,nS,szS)
S = {
    "instance_discrimination": dict(size=96, batch=64, epochs=12, limit=1.0),
    "invariant_spread": dict(size=96, batch=64, epochs=18, limit=1.0),
    "moco_v1": dict(size=96, batch=64, epochs=18, limit=1.0),
    "moco_v2": dict(size=96, batch=64, epochs=18, limit=1.0),
    "simclr_v1": dict(size=96, batch=96, epochs=20, limit=1.0),
    "simclr_v2": dict(size=96, batch=96, epochs=20, limit=1.0),
    "infomin": dict(size=96, batch=96, epochs=20, limit=1.0),
    "byol": dict(size=96, batch=64, epochs=18, limit=1.0),
    "simsiam": dict(size=96, batch=64, epochs=18, limit=1.0),
    "barlow_twins": dict(size=96, batch=64, epochs=18, limit=1.0),
    "swav": dict(size=96, batch=48, epochs=12, limit=1.0, multicrop=(2, 96, 4, 48)),
    "moco_v3": dict(size=224, batch=16, epochs=2, limit=6),
    "dino": dict(size=224, batch=6, epochs=1, limit=4, multicrop=(2, 224, 2, 224)),
}


def prepare_cifar_subset() -> None:
    if (DATA / "val").is_dir() and any((DATA / "val").iterdir()):
        print("CIFAR subset already prepared.", flush=True)
        return
    print("preparing CIFAR-10 ImageFolder subset ...", flush=True)
    for split, train_flag, n in [("train", True, N_PER_CLASS_TRAIN), ("val", False, N_PER_CLASS_VAL)]:
        ds = CIFAR10(root="data/cifar10_raw", train=train_flag, download=True)
        counts = {c: 0 for c in range(10)}
        for img, y in ds:
            if counts[y] >= n:
                continue
            d = DATA / split / ds.classes[y]
            d.mkdir(parents=True, exist_ok=True)
            img.save(d / f"{counts[y]:04d}.png")
            counts[y] += 1
            if all(v >= n for v in counts.values()):
                break
    print("subset ready.", flush=True)


def build_loader_and_model(method: str):
    cfg = load_config(CONFIG[method])
    s = S[method]
    cfg = cfg.model_copy(update={
        "data_dir": str(DATA), "batch_size": s["batch"],
        "max_epochs": s["epochs"], "num_workers": 0,
    })
    model = method_dispatcher(cfg)

    wrapped = None
    train_dir = DATA / "train"
    if method == "instance_discrimination":
        aug = ContrastiveAugmentation(size=s["size"], strong=False)
        tf = MultiViewTransform(aug, n_views=cfg.n_views)
        wrapped = IndexedDataset(ImageFolder(str(train_dir), transform=tf))
    elif "multicrop" in s:
        nL, szL, nS, szS = s["multicrop"]
        wrapped = MultiCropDataset(ImageFolder(str(train_dir)), n_large_crops=nL,
                                   large_size=szL, n_small_crops=nS, small_size=szS)
    dm = SSLDataModule(data_dir=str(DATA), n_views=cfg.n_views, batch_size=s["batch"],
                       num_workers=0, dataset=wrapped, size=s["size"])
    return cfg, model, dm


@torch.no_grad()
def extract(backbone, eval_size: int):
    tf = transforms.Compose([
        transforms.Resize(eval_size + 16),
        transforms.CenterCrop(eval_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = ImageFolder(str(DATA / "val"), transform=tf)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    backbone.eval().to(DEVICE)
    feats, labs = [], []
    for imgs, labels in loader:
        f = backbone(imgs.to(DEVICE))
        if f.ndim > 2:
            f = f.flatten(1)
        feats.append(F.normalize(f, dim=1).cpu())
        labs.append(labels)
    return torch.cat(feats).numpy(), torch.cat(labs).numpy()


def plot(feats, labels, path: Path) -> None:
    emb = umap.UMAP(metric="cosine", random_state=42, n_components=2).fit_transform(feats)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    fig.patch.set_facecolor(NAVY2)
    ax.set_facecolor(NAVY2)
    ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap=CMAP, s=14, alpha=0.85, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([]); ax.margins(0.04)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.12, facecolor=NAVY2)
    plt.close(fig)
    print(f"  saved {path}", flush=True)


def run_method(method: str) -> None:
    print(f"\n=== {method} ===", flush=True)
    cfg, model, dm = build_loader_and_model(method)
    trainer = L.Trainer(
        max_epochs=S[method]["epochs"], limit_train_batches=S[method]["limit"],
        accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, gradient_clip_val=cfg.gradient_clip_val,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, dm)
    feats, labs = extract(model.backbone, S[method]["size"])
    print(f"  features {feats.shape}", flush=True)
    plot(feats, labs, OUT / f"{method}.png")


def run_pretrained(method: str) -> None:
    """Extract features from the method's backbone using ImageNet-pretrained
    weights (NO SSL training). Same-backbone methods therefore share a figure —
    the point is a fair 'fully-trained backbone' baseline, not per-method weights."""
    import timm
    print(f"\n=== {method} (pretrained backbone) ===", flush=True)
    cfg = load_config(CONFIG[method])
    bb = timm.create_model(cfg.backbone, pretrained=True, num_classes=0)
    feats, labs = extract(bb, 224)
    print(f"  backbone={cfg.backbone}  features {feats.shape}", flush=True)
    plot(feats, labs, OUT / f"{method}.png")


import torch.nn as nn
import torchvision
torch.hub.set_dir("data/torchhub")

# fallback (no clean official checkpoint): pretrained ImageNet init + short SSL fine-tune
FINETUNE = {
    "instance_discrimination": dict(size=160, batch=48, epochs=10),
    "invariant_spread": dict(size=160, batch=48, epochs=12),
    "simclr_v1": dict(size=160, batch=48, epochs=12),
    "simclr_v2": dict(size=160, batch=48, epochs=12),
    "infomin": dict(size=160, batch=48, epochs=12),
    "byol": dict(size=160, batch=48, epochs=10),
    "simsiam": dict(size=160, batch=48, epochs=10),
}


def _strip_fc(m):
    if hasattr(m, "fc"):
        m.fc = nn.Identity()
    return m


def _moco_official(url, prefix):
    sd = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=False)["state_dict"]
    r = torchvision.models.resnet50(weights=None)
    r.fc = nn.Identity()
    new = {k[len(prefix):]: v for k, v in sd.items()
           if k.startswith(prefix) and not k[len(prefix):].startswith("fc")}
    r.load_state_dict(new, strict=False)
    return r


OFFICIAL = {
    "dino": lambda: torch.hub.load("facebookresearch/dino:main", "dino_resnet50", verbose=False),
    "swav": lambda: _strip_fc(torch.hub.load("facebookresearch/swav:main", "resnet50", verbose=False)),
    "barlow_twins": lambda: _strip_fc(torch.hub.load("facebookresearch/barlowtwins:main", "resnet50", verbose=False)),
    "moco_v1": lambda: _moco_official("https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar", "module.encoder_q."),
    "moco_v2": lambda: _moco_official("https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar", "module.encoder_q."),
    "moco_v3": lambda: _moco_official("https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar", "module.base_encoder."),
}


def run_official(method: str) -> None:
    print(f"\n=== {method} (OFFICIAL checkpoint) ===", flush=True)
    bb = OFFICIAL[method]()
    feats, labs = extract(bb, 224)
    print(f"  features {feats.shape}", flush=True)
    plot(feats, labs, OUT / f"{method}.png")


def run_finetune(method: str) -> None:
    """ImageNet-pretrained backbone init, then short SSL fine-tune with the
    method's own loss so each method's features genuinely differ."""
    import timm
    print(f"\n=== {method} (pretrained init + SSL fine-tune) ===", flush=True)
    s = FINETUNE[method]
    S[method] = dict(s, limit=1.0)  # reuse build_loader_and_model budget
    cfg, model, dm = build_loader_and_model(method)
    # overwrite backbone weights with ImageNet-pretrained resnet18
    pre = timm.create_model("resnet18", pretrained=True, num_classes=0)
    missing, unexpected = model.backbone.load_state_dict(pre.state_dict(), strict=False)
    print(f"  loaded pretrained backbone (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    trainer = L.Trainer(
        max_epochs=s["epochs"], limit_train_batches=1.0, accelerator="cpu", devices=1,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        gradient_clip_val=cfg.gradient_clip_val, num_sanity_val_steps=0,
    )
    trainer.fit(model, dm)
    feats, labs = extract(model.backbone, s["size"])
    print(f"  features {feats.shape}", flush=True)
    plot(feats, labs, OUT / f"{method}.png")


def run_hybrid() -> None:
    prepare_cifar_subset()
    order = ["instance_discrimination", "invariant_spread", "moco_v1", "moco_v2",
             "simclr_v1", "simclr_v2", "swav", "infomin", "byol", "simsiam",
             "barlow_twins", "moco_v3", "dino"]
    for m in order:
        try:
            if m in OFFICIAL:
                run_official(m)
            else:
                run_finetune(m)
        except Exception:
            print(f"  !! {m} FAILED", flush=True)
            traceback.print_exc()
    try:
        run_dinov2()
    except Exception:
        print("  !! dinov2 FAILED", flush=True); traceback.print_exc()
    print("\nHYBRID DONE", flush=True)


def run_dinov2() -> None:
    print("\n=== dinov2 (pretrained feature extraction) ===", flush=True)
    import timm
    try:
        bb = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                               num_classes=0, img_size=224)
        size = 224
    except Exception as e:
        print("  dinov2 model load failed, fallback resnet18 pretrained:", repr(e)[:100], flush=True)
        bb = timm.create_model("resnet18", pretrained=True, num_classes=0)
        size = 224
    feats, labs = extract(bb, size)
    print(f"  features {feats.shape}", flush=True)
    plot(feats, labs, OUT / "dinov2.png")


def run_dinov3() -> None:
    """DINOv3 (Meta 2025) official gated weights via HuggingFace transformers.
    Requires accepting the DINOv3 license + `huggingface-cli login`."""
    print("\n=== dinov3 (OFFICIAL gated weights via HF) ===", flush=True)
    from transformers import AutoModel
    from huggingface_hub import get_token
    repo = "facebook/dinov3-vits16-pretrain-lvd1689m"
    model = AutoModel.from_pretrained(repo, token=get_token()).eval().to(DEVICE)

    class _Pooled(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x).pooler_output  # (B, 384) CLS-based feature

    feats, labs = extract(_Pooled(model), 224)
    print(f"  features {feats.shape}", flush=True)
    plot(feats, labs, OUT / "dinov3.png")


def main() -> None:
    prepare_cifar_subset()
    for m in CONFIG:
        try:
            run_method(m)
        except Exception:
            print(f"  !! {m} FAILED:", flush=True)
            traceback.print_exc()
    try:
        run_dinov2()
    except Exception:
        print("  !! dinov2 FAILED:", flush=True)
        traceback.print_exc()
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
