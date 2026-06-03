"""Build per-method animated UMAP GIFs from every-epoch feature snapshots.

For smoothness we fit ONE UMAP reducer on the final epoch (ep200) and
`transform` every epoch's features into that fixed coordinate frame — so the
axes never rotate/jump; points just migrate from scattered → tight clusters as
training proceeds. Frame 0 is the final epoch (so PowerPoint's static poster
shows the finished clusters; non-365 viewers see the result, 365 animates).

Input : server_pkg/snaps_pulled/<method>_ep<NNN>.npz  (every epoch 0..200)
Output: demo_assets/gifs/<method>.gif

Run:  python tools/build_progression_gifs.py
"""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from PIL import Image

SNAPS = Path("server_pkg/snaps_pulled")
OUT = Path("demo_assets/gifs"); OUT.mkdir(parents=True, exist_ok=True)
METHODS = ["instance_discrimination", "invariant_spread", "simclr_v1", "simclr_v2",
           "infomin", "byol", "simsiam", "moco_v1", "moco_v2", "swav", "barlow_twins", "moco_v3", "dino"]
NAVY2 = "#182740"
PALETTE = ["#22D3EE", "#F59E0B", "#F43F5E", "#8B5CF6", "#34D399",
           "#FB7185", "#60A5FA", "#FBBF24", "#A78BFA", "#2DD4BF"]
CMAP = matplotlib.colors.ListedColormap(PALETTE)
FRAME_MS = 70          # per-frame duration
HOLD_FINAL_MS = 1400   # pause on the finished clusters


def epochs_for(method):
    eps = []
    for f in SNAPS.glob(f"{method}_ep*.npz"):
        m = re.search(r"_ep(\d+)\.npz$", f.name)
        if m:
            eps.append(int(m.group(1)))
    return sorted(eps)


def render(emb, labels, ep, xlim, ylim):
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    fig.patch.set_facecolor(NAVY2); ax.set_facecolor(NAVY2)
    ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap=CMAP, s=9, alpha=0.85, linewidths=0)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.04, 0.95, f"epoch {ep}", transform=ax.transAxes, color="#E8EEFB",
            fontsize=13, fontweight="bold", va="top")
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = Image.fromarray(buf[:, :, :3].copy())
    plt.close(fig)
    return img


def main():
    for m in METHODS:
        eps = epochs_for(m)
        if not eps:
            print(f"!! {m}: no snapshots", flush=True); continue
        print(f"=== {m}: {len(eps)} epochs ===", flush=True)
        feats = {e: np.load(SNAPS / f"{m}_ep{e:03d}.npz") for e in eps}
        # drop epochs whose features are NaN (e.g. a diverged run); need a clean final to anchor
        eps = [e for e in eps if not np.isnan(feats[e]["feats"]).any()]
        if not eps:
            print(f"  !! {m}: all epochs NaN — skipping GIF", flush=True); continue
        final = eps[-1]
        reducer = umap.UMAP(metric="cosine", random_state=42, n_components=2).fit(feats[final]["feats"])
        emb_final = reducer.embedding_
        labels = feats[final]["labels"]
        pad = 0.08 * (emb_final.max(0) - emb_final.min(0))
        xlim = (emb_final[:, 0].min() - pad[0], emb_final[:, 0].max() + pad[0])
        ylim = (emb_final[:, 1].min() - pad[1], emb_final[:, 1].max() + pad[1])
        frames, durs = [], []
        # poster = final clusters
        frames.append(render(emb_final, labels, final, xlim, ylim)); durs.append(HOLD_FINAL_MS)
        for e in eps:
            emb = emb_final if e == final else reducer.transform(feats[e]["feats"])
            frames.append(render(emb, feats[e]["labels"], e, xlim, ylim))
            durs.append(HOLD_FINAL_MS if e == final else FRAME_MS)
        out = OUT / f"{m}.gif"
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=durs,
                       loop=0, optimize=True, disposal=2)
        print(f"  saved {out}  ({len(frames)} frames)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
