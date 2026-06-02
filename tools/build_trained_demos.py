"""Build per-method UMAP figures from the GPU-trained feature snapshots.

Input : server_pkg/snaps_pulled/<method>_ep<NNN>.npz  (feats, labels) — scp'd back
Output: demo_assets/methods/<method>.png        (final epoch — embedded on slide)
        demo_assets/progression/<method>.png     (epoch strip — how clusters emerge)

Run:  python tools/build_trained_demos.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

SNAPS = Path("server_pkg/snaps_pulled")
OUT = Path("demo_assets/methods"); OUT.mkdir(parents=True, exist_ok=True)
PROG = Path("demo_assets/progression"); PROG.mkdir(parents=True, exist_ok=True)
EPOCHS = [0, 5, 15, 30, 50, 80, 120, 160, 200]  # denser readable subset for static strips
METHODS = ["instance_discrimination", "invariant_spread", "simclr_v1", "simclr_v2",
           "infomin", "byol", "simsiam"]

NAVY2 = "#182740"
PALETTE = ["#22D3EE", "#F59E0B", "#F43F5E", "#8B5CF6", "#34D399",
           "#FB7185", "#60A5FA", "#FBBF24", "#A78BFA", "#2DD4BF"]
CMAP = matplotlib.colors.ListedColormap(PALETTE)


def emb(feats):
    return umap.UMAP(metric="cosine", random_state=42, n_components=2).fit_transform(feats)


def plot_single(e, labels, path):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    fig.patch.set_facecolor(NAVY2); ax.set_facecolor(NAVY2)
    ax.scatter(e[:, 0], e[:, 1], c=labels, cmap=CMAP, s=14, alpha=0.85, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([]); ax.margins(0.04)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.12, facecolor=NAVY2)
    plt.close(fig)
    print("  saved", path, flush=True)


def plot_strip(items, path):  # items = [(epoch, emb, labels), ...]
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.25))
    fig.patch.set_facecolor(NAVY2)
    if n == 1:
        axes = [axes]
    for ax, (ep, e, labels) in zip(axes, items):
        ax.set_facecolor(NAVY2)
        ax.scatter(e[:, 0], e[:, 1], c=labels, cmap=CMAP, s=5, alpha=0.85, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([]); ax.margins(0.04)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"epoch {ep}", color="#C6D6F2", fontsize=11, pad=5)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.15, facecolor=NAVY2)
    plt.close(fig)
    print("  saved", path, flush=True)


def main():
    for m in METHODS:
        print(f"=== {m} ===", flush=True)
        fin = load_clean(SNAPS / f"{m}_ep200.npz")
        if fin is not None:
            plot_single(emb(fin[0]), fin[1], OUT / f"{m}.png")
        else:
            print("  !! final snapshot missing/all-NaN — skipping", m, flush=True)
        items = []
        for ep in EPOCHS:
            c = load_clean(SNAPS / f"{m}_ep{ep:03d}.npz")
            if c is not None:
                items.append((ep, emb(c[0]), c[1]))
        if items:
            plot_strip(items, PROG / f"{m}.png")
    print("DONE", flush=True)


def load_clean(path):
    if not path.exists():
        return None
    d = np.load(path)
    f, l = d["feats"], d["labels"]
    mask = ~np.isnan(f).any(axis=1)
    f, l = f[mask], l[mask]
    return (f, l) if len(f) else None


if __name__ == "__main__":
    main()
