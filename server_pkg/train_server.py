"""Server-side trainer: trains SSL methods on GPU and snapshots backbone
features at several epochs so we can show how clustering emerges over training.

Runs inside the vllm container (torch+timm present; lightning installed offline).
UMAP/plotting is done back on the Mac from the saved .npz feature snapshots.

Usage (pin GPU via env):
  CUDA_VISIBLE_DEVICES=0 python train_server.py --methods simclr_v1,byol,simsiam \
      --data-dir data/cifar10_full --out snaps --epochs 200 --snap 0,5,15,40,100,200
"""
from __future__ import annotations
import argparse, os, sys, traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from core.config import load_config
from core.data import (SSLDataModule, IndexedDataset, MultiViewTransform,
                       ContrastiveAugmentation, MultiCropDataset)
from core.dispatcher import method_dispatcher
import methods  # noqa: F401

CONFIG = {
    "instance_discrimination": "configs/instance_discrimination_resnet18.yaml",
    "invariant_spread": "configs/invariant_spread_resnet18.yaml",
    "simclr_v1": "configs/simclr_v1_resnet18.yaml",
    "simclr_v2": "configs/simclr_v2_resnet18.yaml",
    "infomin": "configs/infomin_resnet18.yaml",
    "byol": "configs/byol_resnet18.yaml",
    "simsiam": "configs/simsiam_resnet18.yaml",
    "moco_v1": "configs/moco_v1_resnet18.yaml",
    "moco_v2": "configs/moco_v2_resnet18.yaml",
    "swav": "configs/swav_resnet18.yaml",
    "barlow_twins": "configs/barlow_twins_resnet18.yaml",
    "moco_v3": "configs/moco_v3_vit_small.yaml",
    "dino": "configs/dino_vit_small.yaml",
}
N_EVAL = 2000  # fixed eval subset for UMAP


def make_eval_loader(data_dir: str, size: int, workers: int):
    tf = transforms.Compose([
        transforms.Resize(size + 16), transforms.CenterCrop(size), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = ImageFolder(str(Path(data_dir) / "val"), transform=tf)
    g = np.random.default_rng(42)
    idx = g.permutation(len(ds))[:N_EVAL]
    return DataLoader(Subset(ds, idx.tolist()), batch_size=256, shuffle=False, num_workers=workers)


class FeatureSnapshot(L.Callback):
    def __init__(self, method, eval_loader, snap_epochs, out: Path):
        self.method, self.loader, self.snaps, self.out = method, eval_loader, set(snap_epochs), out

    @torch.no_grad()
    def _dump(self, pl_module, epoch):
        was = pl_module.training
        pl_module.eval()
        dev = pl_module.device
        feats, labs = [], []
        for imgs, y in self.loader:
            f = pl_module.backbone(imgs.to(dev))
            if f.ndim > 2:
                f = f.flatten(1)
            feats.append(F.normalize(f, dim=1).cpu())
            labs.append(y)
        np.savez(self.out / f"{self.method}_ep{epoch:03d}.npz",
                 feats=torch.cat(feats).numpy(), labels=torch.cat(labs).numpy())
        print(f"  [snap] {self.method} epoch {epoch}", flush=True)
        if was:
            pl_module.train()

    def on_train_start(self, trainer, pl_module):
        if 0 in self.snaps:
            self._dump(pl_module, 0)

    def on_train_epoch_end(self, trainer, pl_module):
        e = trainer.current_epoch + 1  # epoch just finished (1-indexed)
        if e in self.snaps or e == trainer.max_epochs:
            self._dump(pl_module, e)


def build(method, data_dir, epochs, size, workers, grad_clip=None):
    upd = {"data_dir": data_dir, "max_epochs": epochs, "num_workers": workers, "pretrained": False}
    if grad_clip and grad_clip > 0:
        upd["gradient_clip_val"] = grad_clip
    cfg = load_config(CONFIG[method]).model_copy(update=upd)
    model = method_dispatcher(cfg)
    wrapped = None
    td = Path(data_dir) / "train"
    if method == "instance_discrimination":
        aug = ContrastiveAugmentation(size=size, strong=False)
        wrapped = IndexedDataset(ImageFolder(str(td), transform=MultiViewTransform(aug, n_views=cfg.n_views)))
    elif method == "swav":
        wrapped = MultiCropDataset(ImageFolder(str(td)), n_large_crops=2, large_size=size,
                                   n_small_crops=4, small_size=max(48, size // 2))
    elif method == "dino":
        # ViT needs the same fixed resolution for every crop
        wrapped = MultiCropDataset(ImageFolder(str(td)), n_large_crops=2, large_size=size,
                                   n_small_crops=2, small_size=size)
    dm = SSLDataModule(data_dir=data_dir, n_views=cfg.n_views, batch_size=cfg.batch_size,
                       num_workers=workers, dataset=wrapped, size=size)
    return cfg, model, dm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", required=True)
    ap.add_argument("--data-dir", default="data/cifar10_full")
    ap.add_argument("--out", default="snaps")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--snap", default="0,5,15,40,100,200")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--grad-clip", type=float, default=0.0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    snaps = [int(x) for x in a.snap.split(",")]
    eval_loader = make_eval_loader(a.data_dir, a.size, a.workers)
    for m in a.methods.split(","):
        m = m.strip()
        if not m:
            continue
        print(f"\n===== TRAIN {m} ({a.epochs} ep, {a.size}px) on {torch.cuda.get_device_name(0)} =====", flush=True)
        try:
            cfg, model, dm = build(m, a.data_dir, a.epochs, a.size, a.workers, a.grad_clip)
            cb = FeatureSnapshot(m, eval_loader, snaps, out)
            trainer = L.Trainer(max_epochs=a.epochs, accelerator="gpu", devices=1,
                                precision="16-mixed", logger=False, enable_checkpointing=False,
                                enable_progress_bar=False, gradient_clip_val=cfg.gradient_clip_val,
                                num_sanity_val_steps=0, callbacks=[cb])
            trainer.fit(model, dm)
            print(f"  DONE {m}", flush=True)
        except Exception:
            print(f"  !! {m} FAILED", flush=True); traceback.print_exc()
    print("\nGROUP DONE", flush=True)


if __name__ == "__main__":
    main()
