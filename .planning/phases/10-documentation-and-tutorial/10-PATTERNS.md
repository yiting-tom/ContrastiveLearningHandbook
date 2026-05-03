# Phase 10: Documentation and Tutorial — Pattern Mapping

**Mapped:** 2026-05-03
**Phase directory:** `.planning/phases/10-documentation-and-tutorial/`

---

## File 1: `README.md` (new)

**Role:** Project overview, installation, quickstart, config explanation, method table, evaluation instructions (DOC-01).

**Closest analog:** None — no README exists. The closest structural reference is the module-level docstring of `core/dispatcher.py` (brief usage block with code) and the comment block at the top of `configs/simclr_v1_resnet18.yaml`.

**Config comment pattern to replicate:**

From `configs/simclr_v1_resnet18.yaml` lines 1–10:
```yaml
# SimCLR v1 (Chen et al., ICML 2020)
# A Simple Framework for Contrastive Learning of Visual Representations
#
# Note: SimCLR performance degrades sharply below batch_size=256.
# Effective negatives = 2*(batch_size-1). For best results, use
# batch_size=256+ with AdamW, or batch_size=1024+ with LARS.
#
# Uses strong augmentation (s=1.0 color jitter, Gaussian blur).
# Loss computed on projection output z; evaluation on backbone output h.

method: simclr_v1
backbone: resnet18
pretrained: false
```

**Dispatcher usage pattern to quote in README:**

From `core/dispatcher.py` module docstring:
```python
from core.dispatcher import method_dispatcher, register_method

# In a method module (e.g., methods/simclr_v1.py):
register_method("simclr_v1", SimCLRv1)

# In the training script:
model = method_dispatcher(cfg)  # returns SimCLRv1(cfg)
```

**Quickstart commands (all CLI must be verbatim-accurate):**
```bash
pip install -r requirements.txt
python train.py --config configs/simclr_v1_resnet18.yaml --data-dir data/
python eval/linear_probe.py configs/simclr_v1_resnet18.yaml --ckpt checkpoints/epoch=99.ckpt
python eval/umap_vis.py configs/simclr_v1_resnet18.yaml --ckpt checkpoints/epoch=99.ckpt
```

**Data preparation note (required — SSLDataModule is ImageFolder-based):**

`core/data.py` uses `torchvision.datasets.ImageFolder`. CIFAR-10 via `torchvision.datasets.CIFAR10(download=True)` does NOT produce an ImageFolder layout. The README must include one of:
- A `--dataset cifar10` shortcut in `train.py`, OR
- A 2-command data preparation snippet showing how to create an ImageFolder tree

**Method table must cover exactly 14 methods** (not 15 — `supcon_finetune` is stage-2 fine-tuning, not a standalone method):

| Method | Dispatcher Key | Era | Venue | Year |
|--------|----------------|-----|-------|------|
| Instance Discrimination | `instance_discrimination` | Era 1 | CVPR 2018 | 2018 |
| Invariant Spread | `invariant_spread` | Era 1 | CVPR 2019 | 2019 |
| MoCo v1 | `moco_v1` | Era 2 | CVPR 2020 | 2020 |
| MoCo v2 | `moco_v2` | Era 2 | arXiv 2020 | 2020 |
| SimCLR v1 | `simclr_v1` | Era 2 | ICML 2020 | 2020 |
| SimCLR v2 | `simclr_v2` | Era 2 | NeurIPS 2020 | 2020 |
| SwAV | `swav` | Era 2 | NeurIPS 2020 | 2020 |
| InfoMin | `infomin` | Era 2 | NeurIPS 2020 | 2020 |
| BYOL | `byol` | Era 3 | NeurIPS 2020 | 2020 |
| SimSiam | `simsiam` | Era 3 | CVPR 2021 | 2021 |
| Barlow Twins | `barlow_twins` | Era 3 | ICML 2021 | 2021 |
| MoCo v3 | `moco_v3` | Era 4 | ICCV 2021 | 2021 |
| DINO | `dino` | Era 4 | ICCV 2021 | 2021 |
| DINOv2* | `eval/dinov2_demo.py` | Era 4 | TMLR 2024 | 2023 |

*DINOv2: feature-extraction only via `eval/dinov2_demo.py` — not a registered `LightningModule`.

**Eval CLI pattern (verbatim from research):**
```bash
python eval/linear_probe.py <config.yaml> --ckpt <path>
python eval/tsne_vis.py <config.yaml> --ckpt <path>
python eval/umap_vis.py <config.yaml> --ckpt <path>
python eval/finetune.py <config.yaml> --ckpt <path>
python eval/cam_vis.py <config.yaml> --ckpt <path> [--classifier <path>]
python eval/dinov2_demo.py --dataset cifar10
```

---

## File 2: `train.py` (new)

**Role:** Thin Lightning orchestration entry point — the "single-command SimCLR training invocation" (DOC-01 success criterion 4).

**Closest analog:** `tests/test_simclr.py` lines 336–373 (`test_yaml_config_loads_and_trains`) — the only existing code that does the full load_config → method_dispatcher → SSLDataModule → Trainer.fit() chain.

**Full pattern to replicate from `tests/test_simclr.py`:**
```python
# From tests/test_simclr.py lines 340-373 (test_yaml_config_loads_and_trains)
import yaml as _yaml
from methods.simclr.module import SimCLRv1Module
from core.dispatcher import method_dispatcher, register_method, available_methods

with open("configs/simclr_v1_resnet18.yaml") as fh:
    raw = _yaml.safe_load(fh)

cfg = TrainConfig.model_validate(raw)
model = method_dispatcher(cfg)
dm = SSLDataModule(
    data_dir=cfg.data_dir,
    n_views=cfg.n_views,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    size=32,
    strong=True,
)
trainer = L.Trainer(
    max_epochs=1,
    accelerator="cpu",
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=False,
)
trainer.fit(model, dm)
```

**`train.py` must use `load_config()` not raw yaml.safe_load.** Reference from `tests/test_smoke_transformer.py` lines 101–102:
```python
from core.config import load_config
cfg = load_config("configs/moco_v3_vit_small.yaml")
```

**`import methods` triggers registration.** From `methods/__init__.py`:
```python
import methods.instance_discrimination  # noqa: F401
import methods.invariant_spread         # noqa: F401
import methods.simclr                   # noqa: F401
import methods.moco                     # noqa: F401
import methods.infomin                  # noqa: F401
import methods.swav                     # noqa: F401
import methods.byol                     # noqa: F401
import methods.simsiam                  # noqa: F401
import methods.barlow_twins             # noqa: F401
import methods.moco_v3                  # noqa: F401
import methods.dino                     # noqa: F401
import methods.supcon                   # noqa: F401
```
`train.py` must do `import methods` (not individual imports) to trigger all registrations.

**KNNCallback optional pattern.** From `eval/knn_callback.py` module docstring:
```python
from core.config import KNNConfig
from eval.knn_callback import KNNCallback

knn_cb = KNNCallback(KNNConfig(k=200, temperature=0.07, every_n_epochs=5))
trainer = L.Trainer(callbacks=[knn_cb], ...)
```

**`cfg.model_copy(update=...)` for data-dir override.** `TrainConfig` is a Pydantic v2 model; the correct immutable override pattern is `cfg.model_copy(update={"data_dir": args.data_dir})`.

**Trainer kwargs that must come from cfg:**
- `max_epochs=cfg.max_epochs`
- `gradient_clip_val=cfg.gradient_clip_val` (Optional[float] — may be None; pass as-is, Lightning ignores None)

**Complete `train.py` skeleton (executor must implement exactly this):**
```python
"""train.py — single-entry training script for all SSL methods."""
import argparse
import lightning as L
from core.config import load_config
from core.data import SSLDataModule
from core.dispatcher import method_dispatcher
import methods  # triggers register_method() for all 14+1 methods


def main():
    parser = argparse.ArgumentParser(description="Train an SSL method.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--data-dir", default=None, help="Override data_dir in config")
    parser.add_argument("--ckpt-path", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_dir:
        cfg = cfg.model_copy(update={"data_dir": args.data_dir})

    model = method_dispatcher(cfg)
    dm = SSLDataModule(
        data_dir=cfg.data_dir,
        n_views=cfg.n_views,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    callbacks = []
    if cfg.eval and cfg.eval.knn:
        from eval.knn_callback import KNNCallback
        callbacks.append(KNNCallback(cfg.eval.knn))

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        gradient_clip_val=cfg.gradient_clip_val,
        callbacks=callbacks,
    )
    trainer.fit(model, dm, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
```

---

## File 3: Docstrings in 8 method files (modify)

**Role:** Merge module-level DOC-02 content into class docstrings for 8 non-compliant classes (DOC-02).

**Closest analog (gold standard):** `SimCLRv1Module` in `methods/simclr/module.py` lines 33–67 — the only fully-compliant class docstring with all DOC-02 fields at the class level.

**Gold standard DOC-02 class docstring to replicate structure from:**
```python
class SimCLRv1Module(BaseSSLModule):
    """SimCLR v1 (Chen et al., ICML 2020).

    A Simple Framework for Contrastive Learning of Visual Representations.

    Two augmented views of each image are encoded by a shared backbone and
    projected through a 2-layer MLP (2048->2048->128). The symmetric NT-Xent
    loss (implemented via InfoNCELoss with queue=None) brings views of the same
    image together while pushing views of different images apart within the
    batch. Loss is computed on the projection output z; downstream evaluation
    should use the backbone representation h.

    Paper: "A Simple Framework for Contrastive Learning of Visual Representations"
    Authors: Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    Venue: ICML 2020
    arXiv: https://arxiv.org/abs/2002.05709

    Algorithm:
    1. Augment each image twice with strong augmentation (s=1.0 color jitter,
       Gaussian blur, random grayscale, random crop + resize).
    2. Encode both views: h_i = backbone(x_i), h_j = backbone(x_j).
    3. Project: z_i = projector(h_i), z_j = projector(h_j).
    4. Compute symmetric NT-Xent loss on (z_i, z_j).

    Gotchas:
    - Color jitter strength must be s=1.0 (not torchvision default ~0.4).
      ContrastiveAugmentation(strong=True) handles this.
    - Performance degrades sharply below batch_size=256 because effective
      negatives = 2*(batch_size-1). Use LARS optimizer for batch sizes >1024.
    - Loss is computed on z (projection), but evaluation must use h (backbone).
      Do not evaluate downstream tasks on z.
    - InfoNCELoss internally L2-normalizes inputs; do not pre-normalize.

    Reference implementation: https://github.com/google-research/simclr
    """
```

**Required DOC-02 fields (all must appear in the class docstring):**
1. First line: `<MethodName> (<Authors-Short>, <Venue> <Year>).`
2. Paper full title on its own line
3. 1–2 sentence algorithm description (prose)
4. `Paper: "<Full Title>"`
5. `Authors: <list>`
6. `Venue: <Conference/Journal> <Year>`
7. `arXiv: https://arxiv.org/abs/<id>`
8. `Algorithm:` numbered list
9. `Gotchas:` bulleted list (minimum 1 item)
10. `Reference implementation: https://github.com/<repo>`

**The 8 files requiring class docstring merge:**

| File | Class | Current class docstring | Source of DOC-02 content |
|------|-------|------------------------|--------------------------|
| `methods/swav/module.py` | `SwAVModule` | One-liner placeholder: `"""SwAVModule (see module-level docstring for full DOC-02 documentation)."""` | Module docstring lines 1–30 of `methods/swav/module.py` |
| `methods/infomin/module.py` | `InfoMinModule` | Partial (no Paper/Authors/Venue/arXiv/Gotchas) | Module docstring lines 1–32 of `methods/infomin/module.py` |
| `methods/byol/module.py` | `BYOLModule` | Partial (architecture description only, no Paper/arXiv/Gotchas) | Module docstring lines 1–28 of `methods/byol/module.py` |
| `methods/simsiam/module.py` | `SimSiamModule` | Partial (delegates to module doc) | Module docstring of `methods/simsiam/module.py` |
| `methods/barlow_twins/module.py` | `BarlowTwinsModule` | Partial (delegates to module doc) | Module docstring of `methods/barlow_twins/module.py` |
| `methods/dino/module.py` | `DINOModule` | Partial (architecture only, no Paper/arXiv/Gotchas block) | Module docstring lines 1–34 of `methods/dino/module.py` |
| `methods/supcon/module.py` | `SupConModule` | Partial — has Algorithm block but missing Paper/arXiv/Gotchas | Module docstring lines 1–29 of `methods/supcon/module.py` + Gotchas content below |
| `methods/supcon/module.py` | `SupConFinetuneModule` | Partial — has workflow description but no Paper/arXiv/Gotchas block | Same module docstring + Gotchas content below |

**SwAV current class docstring (must be completely replaced):**
```python
# methods/swav/module.py line 48 — current (non-compliant):
class SwAVModule(BaseSSLModule):
    """SwAVModule (see module-level docstring for full DOC-02 documentation)."""
```
Source material already written in module docstring lines 1–30 of `methods/swav/module.py`.

**BYOL current class docstring (must be extended with Paper/arXiv/Gotchas):**
```python
# methods/byol/module.py lines 44–65 — current class docstring:
class BYOLModule(BaseSSLModule):
    """BYOL (Grill et al., NeurIPS 2020).

    Bootstrap Your Own Latent — no-negative self-supervised learning.

    Architecture:
    - Online network: backbone -> projector -> predictor
    - Target network: backbone_ema -> projector_ema (NO predictor)

    The MSE loss (2 - 2*cosine_similarity) between online predictions and
    target projections, combined with EMA updates and the predictor asymmetry,
    prevents representational collapse without negatives.
    ...
    """
```
Missing fields (from module docstring lines 3–28 of `methods/byol/module.py`):
- `Paper: "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning"`
- `Authors: Jean-Bastien Grill, Florian Strub, ...`
- `Venue: NeurIPS 2020`
- `arXiv: https://arxiv.org/abs/2006.07733`
- `Algorithm:` numbered list
- `Gotchas:` (already in module docstring: predictor asymmetry, EMA schedule, requires_grad, L2-normalize, stop-gradient)
- `Reference implementation: https://github.com/deepmind/deepmind-research/tree/master/byol`

**DINO current class docstring (must be extended):**
```python
# methods/dino/module.py lines 50–59 — current class docstring:
class DINOModule(BaseSSLModule):
    """DINO: Self-distillation with student-teacher networks and centering (Caron et al., ICCV 2021).

    Architecture:
    - Student (online) network: backbone -> projector (3-layer MLP, 256-dim) -> prototype_layer (256->65536)
    - Teacher (EMA) network: backbone_ema -> projector_ema -> prototype_layer_ema (same dims, no grad)
    ...
    """
```
Missing: `Paper:`, `Authors:`, `Venue:`, `arXiv:`, `Algorithm:` numbered list, `Gotchas:`, `Reference implementation:` — all present in module docstring lines 1–34 of `methods/dino/module.py`.

**SupCon Gotchas content (not in module docstring — must be written fresh):**
```
Gotchas:
- Do NOT add a classifier during stage-1 pretraining — adding cross-entropy
  alongside SupCon loss collapses the contrastive representation.
- ClassBalancedSampler is required; without it, singleton classes per batch
  have no positives and the supervised contrastive term degrades to SimCLR.
- Use the sum-outside formulation (Eq. 2 of the paper), not sum-inside.
- SupConLoss normalizes features internally; do not pre-normalize inputs.
```

**SupConFinetuneModule Gotchas (not in module docstring — must be written fresh):**
```
Gotchas:
- Use SGD with weight_decay=0.0; any nonzero weight decay suppresses
  linear probe accuracy by regularizing the classifier away from signal.
- Freeze backbone AFTER loading the stage-1 checkpoint, not before.
- BN layers in the backbone must remain in eval() mode — freeze_backbone()
  handles this.
- Use only one view (views[0]) during stage-2 fine-tuning; no multi-view needed.
```

**Test that will validate DOC-02 compliance** (`tests/test_docstrings.py` — to be created):

Pattern from `tests/test_simclr.py` lines 493–514:
```python
def test_simclr_v1_docstring_has_doc02():
    """SimCLRv1Module docstring meets DOC-02 standard."""
    from methods.simclr.module import SimCLRv1Module

    doc = SimCLRv1Module.__doc__
    assert doc is not None, "SimCLRv1Module must have a docstring"
    assert "arXiv" in doc, "Docstring must contain arXiv link"
    assert "Gotcha" in doc or "gotcha" in doc.lower(), "Docstring must contain gotchas"
    assert "Reference implementation" in doc, "Docstring must contain reference URL"
    assert "ICML 2020" in doc, "Docstring must contain venue and year"
```

---

## File 4: `docs/tutorial.md` (new)

**Role:** Guide covering (a) add a new method, (b) run experiment end-to-end, (c) compare two methods (DOC-03). Markdown format — no Jupyter dependency.

**Closest analog:** No existing tutorial file. Reference patterns are module docstrings with `Usage::` blocks (e.g., `core/dispatcher.py`, `eval/knn_callback.py`) and the `BaseSSLModule` example in the `core/base.py` docstring.

**BaseSSLModule subclass example (from research — sourced from `core/base.py` docstring):**
```python
class MyMethod(BaseSSLModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg)
        self.backbone, self.feat_dim = build_backbone(cfg.backbone, cfg.pretrained)
        self.projector = self.build_projector()
        self.loss_fn = InfoNCELoss(temperature=0.5)

    def build_projector(self) -> nn.Module:
        return ProjectionHead(self.feat_dim, 2048, 128, num_layers=2)

    def training_step(self, batch, batch_idx):
        views, _ = batch
        z_i = self.projector(self.backbone(views[0]))
        z_j = self.projector(self.backbone(views[1]))
        loss = self.loss_fn(z_i, z_j)
        self.log_train_metrics(loss)
        return loss
```

**Dispatcher registration pattern (from `methods/__init__.py` and `core/dispatcher.py`):**
```python
# In methods/<newmethod>/__init__.py:
from core.dispatcher import register_method
from methods.<newmethod>.module import <NewMethodClass>
register_method("<key>", <NewMethodClass>)

# In methods/__init__.py (top-level) — add ONE line:
import methods.<newmethod>  # noqa: F401
```

**Exact imports the tutorial must use (verified from actual files):**
```python
from core.config import TrainConfig, load_config
from core.data import SSLDataModule
from core.dispatcher import method_dispatcher, register_method, available_methods
from core.base import BaseSSLModule
from core.backbone import build_backbone
from core.projection import ProjectionHead
from core.losses import InfoNCELoss
from eval.knn_callback import KNNCallback, KNNConfig
```

**Tutorial section (b) — end-to-end experiment using real CLI:**
```bash
# 1. Install
pip install -r requirements.txt

# 2. Train SimCLR v1 (200 epochs, ResNet-18, AdamW)
python train.py --config configs/simclr_v1_resnet18.yaml --data-dir data/

# 3. Linear probe evaluation
python eval/linear_probe.py configs/simclr_v1_resnet18.yaml \
    --ckpt checkpoints/epoch=199.ckpt

# 4. Visualize feature space
python eval/umap_vis.py configs/simclr_v1_resnet18.yaml \
    --ckpt checkpoints/epoch=199.ckpt
```

**Tutorial section (c) — compare two methods using real CLI:**
```bash
python eval/linear_probe.py configs/simclr_v1_resnet18.yaml --ckpt ckpt_simclr.ckpt
python eval/linear_probe.py configs/moco_v2_resnet18.yaml --ckpt ckpt_moco.ckpt

python eval/tsne_vis.py configs/simclr_v1_resnet18.yaml --ckpt ckpt_simclr.ckpt
python eval/tsne_vis.py configs/moco_v2_resnet18.yaml --ckpt ckpt_moco.ckpt
```

**Available configs for tutorial examples (all verified to exist):**
- `configs/simclr_v1_resnet18.yaml` — primary quickstart config
- `configs/moco_v1_resnet18.yaml`, `configs/moco_v2_resnet18.yaml`
- `configs/byol_resnet18.yaml`, `configs/simsiam_resnet18.yaml`
- `configs/barlow_twins_resnet18.yaml`
- `configs/moco_v3_vit_small.yaml`, `configs/dino_vit_small.yaml`

**Format requirement:** `docs/tutorial.md` — Markdown, NOT `notebooks/walkthrough.ipynb`. No Jupyter dependency.

---

## File 5: `tests/test_docstrings.py` (new)

**Role:** Automated DOC-02 compliance check — verifies each LightningModule subclass has all required fields in its class docstring.

**Closest analog:** `tests/test_smoke_transformer.py` lines 217–276 — existing `test_doc02_moco_v3` and `test_doc02_dino` functions. These check module-level docstrings; `test_docstrings.py` must check **class** docstrings only.

**Pattern from `tests/test_smoke_transformer.py` (check both module + class doc):**
```python
def test_doc02_moco_v3():
    import methods.moco_v3.module as mod
    from methods.moco_v3.module import MoCoV3Module

    docs = []
    if mod.__doc__:
        docs.append(mod.__doc__)
    if MoCoV3Module.__doc__:
        docs.append(MoCoV3Module.__doc__)
    full_doc = "\n".join(docs)

    assert "Chen" in full_doc
    assert "ICCV 2021" in full_doc
    assert "arxiv" in full_doc.lower()
    assert "patch" in full_doc.lower()
```

**Pattern from `tests/test_simclr.py` (class-docstring-only check — the correct model for test_docstrings.py):**
```python
def test_simclr_v1_docstring_has_doc02():
    from methods.simclr.module import SimCLRv1Module

    doc = SimCLRv1Module.__doc__
    assert doc is not None, "SimCLRv1Module must have a docstring"
    assert "arXiv" in doc, "Docstring must contain arXiv link"
    assert "Gotcha" in doc or "gotcha" in doc.lower(), "Docstring must contain gotchas"
    assert "Reference implementation" in doc, "Docstring must contain reference URL"
    assert "ICML 2020" in doc, "Docstring must contain venue and year"
```

**`test_docstrings.py` must check class docstrings (not module docstrings).** The distinction is critical: checking `SimCLRv1Module.__doc__` (class) is correct; checking `methods.simclr.module.__doc__` (module) is NOT sufficient for DOC-02 compliance.

**File structure to follow (from `tests/test_smoke_no_negative.py`):**
```python
"""<module docstring describing what is verified>"""
from __future__ import annotations

import pytest

# Imports via direct module path (never `import methods` in test_docstrings.py —
# use explicit per-module imports so each test is independently importable)


# ---------------------------------------------------------------------------
# DOC-02 field checker helper
# ---------------------------------------------------------------------------

def _check_doc02(cls, venue_year: str, arxiv_fragment: str) -> None:
    """Assert class docstring contains all DOC-02 required fields."""
    doc = cls.__doc__
    assert doc is not None, f"{cls.__name__} must have a class docstring"
    assert "Paper:" in doc, f"{cls.__name__}: missing 'Paper:' field"
    assert "Authors:" in doc, f"{cls.__name__}: missing 'Authors:' field"
    assert "Venue:" in doc, f"{cls.__name__}: missing 'Venue:' field"
    assert "arXiv:" in doc, f"{cls.__name__}: missing 'arXiv:' field"
    assert venue_year in doc, f"{cls.__name__}: missing venue/year '{venue_year}'"
    assert arxiv_fragment in doc, f"{cls.__name__}: missing arXiv fragment '{arxiv_fragment}'"
    assert "Gotcha" in doc or "gotcha" in doc.lower(), f"{cls.__name__}: missing 'Gotchas:' section"
    assert "Reference implementation" in doc, f"{cls.__name__}: missing 'Reference implementation:'"


# ---------------------------------------------------------------------------
# Per-class tests
# ---------------------------------------------------------------------------

def test_doc02_simclr_v1():
    from methods.simclr.module import SimCLRv1Module
    _check_doc02(SimCLRv1Module, "ICML 2020", "2002.05709")

# ... one function per LightningModule subclass ...
```

**Full list of classes and expected venue_year + arXiv fragment to assert:**

| Test function | Class | Import path | venue_year | arXiv fragment |
|---------------|-------|-------------|------------|----------------|
| `test_doc02_instance_discrimination` | `InstanceDiscriminationModule` | `methods.instance_discrimination.module` | `CVPR 2018` | `1805.01978` |
| `test_doc02_invariant_spread` | `InvariantSpreadModule` | `methods.invariant_spread.module` | `CVPR 2019` | (check arXiv present) |
| `test_doc02_simclr_v1` | `SimCLRv1Module` | `methods.simclr.module` | `ICML 2020` | `2002.05709` |
| `test_doc02_simclr_v2` | `SimCLRv2Module` | `methods.simclr.module` | `NeurIPS 2020` | `2006.10029` |
| `test_doc02_moco_v1` | `MoCoV1Module` | `methods.moco.module` | `CVPR 2020` | `1911.05722` |
| `test_doc02_moco_v2` | `MoCoV2Module` | `methods.moco.module` | `2020` | `2003.04297` |
| `test_doc02_swav` | `SwAVModule` | `methods.swav.module` | `NeurIPS 2020` | `2006.09882` |
| `test_doc02_infomin` | `InfoMinModule` | `methods.infomin.module` | `NeurIPS 2020` | `2005.10243` |
| `test_doc02_byol` | `BYOLModule` | `methods.byol.module` | `NeurIPS 2020` | `2006.07733` |
| `test_doc02_simsiam` | `SimSiamModule` | `methods.simsiam.module` | `CVPR 2021` | (check arXiv present) |
| `test_doc02_barlow_twins` | `BarlowTwinsModule` | `methods.barlow_twins.module` | `ICML 2021` | (check arXiv present) |
| `test_doc02_moco_v3` | `MoCoV3Module` | `methods.moco_v3.module` | `ICCV 2021` | `2104.02057` |
| `test_doc02_dino` | `DINOModule` | `methods.dino.module` | `ICCV 2021` | `2104.14294` |
| `test_doc02_supcon` | `SupConModule` | `methods.supcon.module` | `NeurIPS 2020` | `2004.11362` |
| `test_doc02_supcon_finetune` | `SupConFinetuneModule` | `methods.supcon.module` | `NeurIPS 2020` | `2004.11362` |

**Test runner command (from `pyproject.toml`):**
```bash
pytest tests/test_docstrings.py -x -q
```

**Do NOT use `import methods` at module level in `test_docstrings.py`.** Each test imports its class directly via explicit path (e.g., `from methods.swav.module import SwAVModule`) to avoid registry side effects and keep tests independent.

**Registry fixture is NOT needed in `test_docstrings.py`** — unlike dispatcher tests, docstring tests do not modify `_METHOD_REGISTRY`. No `clean_registry` fixture required.

---

## Cross-File Patterns

### `from __future__ import annotations`
All existing test files and method modules use this as the first non-docstring import. New files must include it.

### Module docstring format
All existing Python files in this repo start with a `"""<brief description>."""` module docstring followed by a blank line, then imports. Follow this pattern in `train.py` and `tests/test_docstrings.py`.

### Section separator comment style
All test files use this exact pattern for section headers:
```python
# ---------------------------------------------------------------------------
# Section Name
# ---------------------------------------------------------------------------
```

### `tests/conftest.py` fixtures available to all tests
- `random_tensor` — factory for random tensors
- `tmp_imagefolder` — 3-class ImageFolder with 5 images each (32x32 RGB JPEGs)
- `toy_config_dict` — minimal valid `TrainConfig` dict for `simclr_v1`

`test_docstrings.py` does not need any conftest fixtures — pure import + assertion tests.

### Assertion message format (from test_simclr.py)
```python
assert condition, f"<ClassName>: <what was expected>"
# Example:
assert "arXiv" in doc, "Docstring must contain arXiv link"
assert cfg.method == "simclr_v1", f"Expected method='simclr_v1', got {cfg.method!r}"
```

---

*Pattern mapping: 2026-05-03*
