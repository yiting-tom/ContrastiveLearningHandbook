# Phase 10: Documentation and Tutorial - Research

**Researched:** 2026-05-03
**Domain:** Technical writing, Python documentation conventions, Jupyter/Markdown tutorial design
**Confidence:** HIGH

---

## Summary

Phase 10 is primarily a writing and documentation phase, not an implementation phase. The codebase is fully built — all 14 methods exist as working LightningModule subclasses, 15 YAML configs exist in `configs/`, the full eval suite lives in `eval/`, and 326 tests pass. The code infrastructure is solid. What is missing is the presentation layer: no `README.md`, no `train.py` entry point, no `notebooks/` or `docs/` directory, and several class-level docstrings that need DOC-02 fields merged from module-level docstrings.

The key planning insight is that Phase 10 has two distinct subtasks: (1) documentation writing (README, docstring upgrades, tutorial text) and (2) one small code creation task — writing `train.py`, which does not yet exist but is referenced by every config file and docstring. The train.py is a thin Lightning orchestration script that loads a YAML, calls `method_dispatcher`, builds `SSLDataModule`, optionally adds `KNNCallback`, and calls `trainer.fit()`. It is the "single-command SimCLR training invocation" required by success criterion 4.

The docstring audit reveals a split-docstring pattern: for 8 of 14 methods (SwAV, InfoMin, BYOL, SimSiam, BarlowTwins, DINOModule, SupConModule, SupConFinetuneModule), the DOC-02 fields (Paper, Authors, Venue, arXiv, Gotchas, Reference implementation) live in the module-level docstring rather than the class docstring. DOC-02 requires these fields "in each `LightningModule` subclass" docstring. Plan 10-02 must merge module-level DOC-02 content into class docstrings for these 8 classes.

**Primary recommendation:** Write `train.py` as part of Plan 10-01 (README), since the README quickstart depends on it. All six tutorial sections can be written as a single `docs/tutorial.md` Markdown file (simpler than a Jupyter notebook for a code-only repo with no interactive outputs).

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DOC-01 | `README.md` covering project overview, installation, quickstart, config explanation, method table, evaluation instructions | `configs/simclr_v1_resnet18.yaml` is the quickstart config; `train.py` must be created; method table data documented in this research |
| DOC-02 | Per-method docstring in each LightningModule subclass: paper title, authors, venue, year, arXiv/DOI, 2-sentence algorithm, gotchas, reference implementation URL | Audit complete — 6 classes fully compliant, 8 classes have DOC-02 fields only in module docstring (need merge into class docstring) |
| DOC-03 | Tutorial notebook or guide covering: (a) add a new method, (b) run experiment end-to-end, (c) compare two methods | `method_dispatcher` pattern documented; eval CLI interfaces documented; all needed code infrastructure exists |
</phase_requirements>

---

## Codebase State Inventory

### What EXISTS (verified by file inspection)

**Method modules** — `[VERIFIED: codebase grep]`
| Dispatcher Key | Module Class | File |
|----------------|-------------|------|
| `instance_discrimination` | `InstanceDiscriminationModule` | `methods/instance_discrimination/module.py` |
| `invariant_spread` | `InvariantSpreadModule` | `methods/invariant_spread/module.py` |
| `simclr_v1` | `SimCLRv1Module` | `methods/simclr/module.py` |
| `simclr_v2` | `SimCLRv2Module` | `methods/simclr/module.py` |
| `moco_v1` | `MoCoV1Module` | `methods/moco/module.py` |
| `moco_v2` | `MoCoV2Module` | `methods/moco/module.py` |
| `swav` | `SwAVModule` | `methods/swav/module.py` |
| `infomin` | `InfoMinModule` | `methods/infomin/module.py` |
| `byol` | `BYOLModule` | `methods/byol/module.py` |
| `simsiam` | `SimSiamModule` | `methods/simsiam/module.py` |
| `barlow_twins` | `BarlowTwinsModule` | `methods/barlow_twins/module.py` |
| `moco_v3` | `MoCoV3Module` | `methods/moco_v3/module.py` |
| `dino` | `DINOModule` | `methods/dino/module.py` |
| `supcon` | `SupConModule` | `methods/supcon/module.py` |
| `supcon_finetune` | `SupConFinetuneModule` | `methods/supcon/module.py` |

Note: `supcon_finetune` is registered as a 15th key but is not a standalone contrastive SSL method. The DOC-01 method table covers 14 v1 methods (not `supcon_finetune`). `[VERIFIED: dispatcher.available_methods()]`

**YAML configs** — all exist in `configs/` `[VERIFIED: ls configs/]`
- `simclr_v1_resnet18.yaml`, `simclr_v1_resnet50_lars.yaml`, `simclr_v2_resnet18.yaml`
- `moco_v1_resnet18.yaml`, `moco_v2_resnet18.yaml`
- `instance_discrimination_resnet18.yaml`, `invariant_spread_resnet18.yaml`
- `swav_resnet18.yaml`, `infomin_resnet18.yaml`
- `byol_resnet18.yaml`, `simsiam_resnet18.yaml`, `barlow_twins_resnet18.yaml`
- `moco_v3_vit_small.yaml`, `dino_vit_small.yaml`
- `supcon_stage1_resnet18.yaml`, `supcon_stage2_resnet18.yaml`
- `example.yaml`

No `simclr_resnet50.yaml` with AdamW exists — the ResNet-50 config is LARS-only (`simclr_v1_resnet50_lars.yaml`). The standard tutorial quickstart config will use `configs/simclr_v1_resnet18.yaml`.

**Eval scripts** — all exist in `eval/` `[VERIFIED: ls eval/]`
- `eval/knn_callback.py` — `KNNCallback(KNNConfig)` Lightning callback
- `eval/linear_probe.py` — CLI: `python eval/linear_probe.py <config.yaml> --ckpt <path>`
- `eval/tsne_vis.py` — CLI: `python eval/tsne_vis.py <config.yaml> --ckpt <path>`
- `eval/umap_vis.py` — CLI: `python eval/umap_vis.py <config.yaml> --ckpt <path>`
- `eval/finetune.py` — CLI: `python eval/finetune.py <config.yaml> --ckpt <path>`
- `eval/cam_vis.py` — CLI: `python eval/cam_vis.py <config.yaml> --ckpt <path> [--classifier <path>]`
- `eval/dinov2_demo.py` — CLI: `python eval/dinov2_demo.py --dataset cifar10`

**Core modules** — `[VERIFIED: codebase inspection]`
- `core/config.py` — `TrainConfig`, `EvalConfig`, `load_config(path) -> TrainConfig`
- `core/dispatcher.py` — `method_dispatcher(cfg)`, `register_method(name, cls)`, `available_methods()`, `get_method(name)`
- `core/base.py` — `BaseSSLModule(L.LightningModule)` with `build_projector()` (abstract), `training_step()` (abstract), `configure_optimizers()`, `on_train_batch_end()`, `log_train_metrics()`
- `core/data.py` — `SSLDataModule`, `ContrastiveAugmentation`, `MultiCropDataset`

**Tools** — `[VERIFIED: ls tools/]`
- `tools/visualize_augmentations.py`
- `tools/compare_augmentations.py`

### What DOES NOT EXIST (verified by absence)

- `README.md` — does not exist in repo root `[VERIFIED: ls /]`
- `train.py` — does not exist anywhere in the repo `[VERIFIED: find -name train.py]`
- `notebooks/` directory — does not exist `[VERIFIED: ls /]`
- `docs/` directory — does not exist `[VERIFIED: ls /]`
- Any test files for documentation/tutorial correctness — no `tests/test_doc*.py` files

---

## DOC-02 Docstring Audit

`[VERIFIED: ast.get_docstring() on each class]`

DOC-02 requires each LightningModule subclass to have in its **class docstring**: paper title, authors, venue, year, arXiv/DOI link, 2-sentence algorithm description, gotcha list, reference implementation URL.

| Class | Class Docstring Status | Module Docstring Status | Gap |
|-------|----------------------|-------------------------|-----|
| `InstanceDiscriminationModule` | COMPLIANT | n/a | None |
| `InvariantSpreadModule` | COMPLIANT | n/a | None |
| `SimCLRv1Module` | COMPLIANT | n/a | None |
| `SimCLRv2Module` | COMPLIANT | n/a | None |
| `MoCoV1Module` | COMPLIANT | n/a | None |
| `MoCoV2Module` | COMPLIANT | n/a | None |
| `MoCoV3Module` | COMPLIANT | n/a | None |
| `SwAVModule` | MISSING (delegates to module doc) | COMPLIANT | Merge module → class |
| `InfoMinModule` | MISSING (delegates to module doc) | COMPLIANT | Merge module → class |
| `BYOLModule` | MISSING (delegates to module doc) | COMPLIANT | Merge module → class |
| `SimSiamModule` | MISSING (delegates to module doc) | COMPLIANT | Merge module → class |
| `BarlowTwinsModule` | MISSING (delegates to module doc) | COMPLIANT | Merge module → class |
| `DINOModule` | MISSING (delegates to module doc) | COMPLIANT | Merge module → class |
| `SupConModule` | MISSING (delegates to module doc) | PARTIAL (no Gotchas field) | Merge + add Gotchas |
| `SupConFinetuneModule` | MISSING (delegates to module doc) | PARTIAL (no Gotchas field) | Merge + add Gotchas |

**Key finding:** The DOC-02 content is largely written — it just lives in module docstrings rather than class docstrings. Plan 10-02 is a merge operation, not a rewrite from scratch. The exception is `SupConModule`/`SupConFinetuneModule` where the module docstring also lacks Gotchas and needs them added.

**SwAVModule special case:** The class docstring is `"""SwAVModule (see module-level docstring for full DOC-02 documentation)."""` — an explicit placeholder. The module docstring is fully DOC-02 compliant. Plan 10-02 merges module → class docstring for SwAV.

**SupCon Gotchas content** (needs to be added):
- Do NOT add a classifier during stage-1 pretraining — it collapses the contrastive representation
- ClassBalancedSampler is required; without it, singleton classes have no positives
- Use sum-outside formulation (Eq. 2), not sum-inside
- `SupConFinetuneModule`: SGD with weight_decay=0.0; any weight decay suppresses accuracy

---

## train.py Design

`train.py` does not exist but is referenced by 10+ files. It must be created in Plan 10-01 (the README depends on it for the quickstart). `[VERIFIED: grep for train.py references]`

**Required interface** (inferred from config patterns and supcon module docstrings):
```bash
python train.py --config configs/simclr_v1_resnet18.yaml [--data-dir /path/to/data] [--ckpt-path /path/to/ckpt]
```

**Minimal implementation pattern** (based on test smoke patterns and Lightning conventions):
```python
# Source: inferred from test_smoke_no_negative.py + core/config.py + core/dispatcher.py patterns
import argparse
import yaml
import lightning as L
from core.config import load_config, TrainConfig
from core.data import SSLDataModule
from core.dispatcher import method_dispatcher
import methods  # trigger registration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--ckpt-path", default=None)
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
```

**CIFAR-10 note:** `SSLDataModule` wraps `ImageFolder`, which requires a directory structure with class subfolders. For the README quickstart to work on CIFAR-10, either: (a) the quickstart assumes user has an ImageFolder-format dataset, or (b) a small dataset preparation note is included. CIFAR-10 must be arranged as an ImageFolder layout (`torchvision.datasets.CIFAR10` does not produce this automatically). The README should document this or use STL10/ImageNet-style data. `[ASSUMED]` — the exact quickstart data preparation path needs one decision.

---

## Method Table (DOC-01)

All 14 v1 methods with era, venue, and primary contribution. `[VERIFIED: REQUIREMENTS.md + Traceability table + arXiv cross-reference from module docstrings]`

| Method | Dispatcher Key | Era | Venue | Year | Primary Contribution |
|--------|----------------|-----|-------|------|---------------------|
| Instance Discrimination | `instance_discrimination` | Era 1: Proxy Tasks | CVPR 2018 | 2018 | Non-parametric memory bank; each image as its own class |
| Invariant Spread | `invariant_spread` | Era 1: Proxy Tasks | CVPR 2019 | 2019 | In-batch softmax contrastive; direct ancestor of SimCLR |
| MoCo v1 | `moco_v1` | Era 2: Queue-Based | CVPR 2020 | 2020 | Momentum encoder + FIFO queue for large negative set |
| MoCo v2 | `moco_v2` | Era 2: Queue-Based | arXiv 2020 | 2020 | MoCo + SimCLR architecture improvements (MLP head, blur, cosine LR) |
| SimCLR v1 | `simclr_v1` | Era 2: In-Batch | ICML 2020 | 2020 | Strong augmentation + in-batch symmetric NT-Xent loss |
| SimCLR v2 | `simclr_v2` | Era 2: In-Batch | NeurIPS 2020 | 2020 | Deeper 3-layer projection head; semi-supervised distillation (v1 scope: pretraining only) |
| SwAV | `swav` | Era 2: Prototype | NeurIPS 2020 | 2020 | Online clustering via Sinkhorn-Knopp OT; multi-crop |
| InfoMin | `infomin` | Era 2: Augmentation | NeurIPS 2020 | 2020 | Minimal-MI view design; augmentation-policy principle |
| BYOL | `byol` | Era 3: No-Negative | NeurIPS 2020 | 2020 | Bootstrap without negatives via predictor asymmetry + EMA |
| SimSiam | `simsiam` | Era 3: No-Negative | CVPR 2021 | 2021 | Stop-gradient as the only collapse prevention; no EMA |
| Barlow Twins | `barlow_twins` | Era 3: No-Negative | ICML 2021 | 2021 | Redundancy reduction via cross-correlation matrix toward identity |
| MoCo v3 | `moco_v3` | Era 4: Transformer | ICCV 2021 | 2021 | MoCo for ViTs; patch-projection freeze for training stability |
| DINO | `dino` | Era 4: Transformer | ICCV 2021 | 2021 | Student-teacher with centering + sharpening; no contrastive negatives |
| DINOv2 | `eval/dinov2_demo.py` | Era 4: Transformer | TMLR 2024 | 2023 | Large-scale pretraining with curated data; tutorial = feature extraction only |

**DINOv2 special handling:** DINOv2 is implemented as a standalone demo script (`eval/dinov2_demo.py`), not as a dispatcher-registered `LightningModule`. The method table should include it with a note that it is feature-extraction-only.

---

## Architecture Patterns

### Dispatcher Registration Pattern

```python
# Source: methods/__init__.py + core/dispatcher.py (VERIFIED)
# In methods/<method>/__init__.py:
from core.dispatcher import register_method
from methods.<method>.module import <MethodClass>
register_method("<method_key>", <MethodClass>)

# In methods/__init__.py (top-level):
import methods.<method>  # triggers register_method() via __init__.py side effect
```

### Adding a New Method (Tutorial Section a)

The complete workflow for adding a new SSL method: `[VERIFIED: codebase inspection]`

1. Create `methods/<newmethod>/` directory with `__init__.py` and `module.py`
2. In `module.py`: Subclass `BaseSSLModule`, implement `build_projector()` and `training_step()`
3. In `__init__.py`: Call `register_method("<key>", <NewMethodClass>)`
4. In `methods/__init__.py`: Add `import methods.<newmethod>` line
5. Create `configs/<newmethod>_resnet18.yaml` with `method: <key>`
6. Train: `python train.py --config configs/<newmethod>_resnet18.yaml --data-dir <data>`

**The minimal BaseSSLModule subclass:**
```python
# Source: core/base.py docstring example (VERIFIED)
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

### Config Schema Pattern

```python
# Source: core/config.py (VERIFIED)
from core.config import load_config
cfg = load_config("configs/simclr_v1_resnet18.yaml")
# cfg.method == "simclr_v1"
# cfg.simclr.temperature == 0.5
# TrainConfig has extra='forbid' — unknown keys raise ValidationError immediately
```

### End-to-End Experiment Workflow (Tutorial Section b)

```bash
# Source: configs/*.yaml + eval/*.py CLI interfaces (VERIFIED)

# 1. Install
pip install -r requirements.txt

# 2. Train
python train.py --config configs/simclr_v1_resnet18.yaml --data-dir data/cifar10

# 3. k-NN evaluation (runs in-training if KNNConfig set in YAML eval.knn block)
# OR post-training:
python eval/linear_probe.py configs/simclr_v1_resnet18.yaml --ckpt checkpoints/epoch=99.ckpt

# 4. Visualize
python eval/umap_vis.py configs/simclr_v1_resnet18.yaml --ckpt checkpoints/epoch=99.ckpt
```

### Comparing Two Methods (Tutorial Section c)

```bash
# Source: eval/ script CLI interfaces (VERIFIED)
# Run linear probe on two checkpoints, produce comparison table

python eval/linear_probe.py configs/simclr_v1_resnet18.yaml --ckpt ckpt_simclr.ckpt
python eval/linear_probe.py configs/moco_v2_resnet18.yaml --ckpt ckpt_moco.ckpt

python eval/tsne_vis.py configs/simclr_v1_resnet18.yaml --ckpt ckpt_simclr.ckpt
python eval/tsne_vis.py configs/moco_v2_resnet18.yaml --ckpt ckpt_moco.ckpt
```

---

## DOC-02 Compliant Docstring Template

Based on the SimCLRv1Module and MoCoV1Module patterns which are already compliant: `[VERIFIED: reading class docstrings]`

```python
class <MethodName>Module(BaseSSLModule):
    """<MethodName> (<Authors-Short>, <Venue> <Year>).

    <Full paper title>.

    <Sentence 1: what the method does architecturally.>
    <Sentence 2: what loss/mechanism drives learning.>

    Paper: "<Full Paper Title>"
    Authors: <Author 1>, <Author 2>, ...
    Venue: <Conference/Journal> <Year>
    arXiv: https://arxiv.org/abs/<id>

    Algorithm:
    1. <Step 1>
    2. <Step 2>
    ...

    Gotchas:
    - <Gotcha 1>
    - <Gotcha 2>
    ...

    Reference implementation: https://github.com/<repo>
    """
```

---

## Common Pitfalls

### Pitfall 1: README quickstart fails because train.py doesn't exist
**What goes wrong:** User runs `python train.py --config ...` and gets `No such file or directory`.
**Why it happens:** `train.py` is referenced everywhere but never implemented. Multiple config comments and module docstrings assume it exists.
**How to avoid:** Create `train.py` in Plan 10-01 before writing any README quickstart. The README must reference a script that actually exists.
**Warning signs:** Any reference to `python train.py` in config comments points to the missing script.

### Pitfall 2: README quickstart uses SSLDataModule with CIFAR-10 via wrong path
**What goes wrong:** User runs training on CIFAR-10 but `SSLDataModule` uses `ImageFolder` which requires a pre-structured directory tree. `torchvision.datasets.CIFAR10(download=True)` does not produce an ImageFolder layout.
**Why it happens:** `SSLDataModule` is an `ImageFolder`-based data module; CIFAR-10 can be obtained via `torchvision.datasets.CIFAR10` but that produces a different format.
**How to avoid:** Either (a) add a data-prep note to README showing how to convert CIFAR-10 to ImageFolder format, or (b) use STL10-unlabeled (which IS available as an ImageFolder) for the quickstart, or (c) have `train.py` support torchvision named datasets. Decision needed. `[ASSUMED]` — see Assumptions Log.
**Warning signs:** Step 4 success criterion says "train SimCLR on CIFAR-10 in under 5 commands" — must explicitly address how the user prepares CIFAR-10 data.

### Pitfall 3: Class docstrings vs module docstrings for DOC-02
**What goes wrong:** Plan 10-02 auditor checks module docstrings and marks them compliant, but REQUIREMENTS.md says "each `LightningModule` subclass has a docstring containing" the DOC-02 fields — meaning the class docstring, not the module docstring.
**Why it happens:** SwAV, InfoMin, BYOL, SimSiam, BarlowTwins, DINO, SupConModule, SupConFinetuneModule delegated DOC-02 content to module-level docstrings during Phases 5-8.
**How to avoid:** Plan 10-02 must explicitly update class docstrings, not just verify module docstrings.
**Warning signs:** Class docstring that says "see module-level docstring" is not DOC-02 compliant.

### Pitfall 4: SwAVModule class docstring is a placeholder
**What goes wrong:** `SwAVModule` class docstring is literally: `"""SwAVModule (see module-level docstring for full DOC-02 documentation)."""` — a one-liner placeholder that fails DOC-02 immediately.
**Why it happens:** Phase 5 plan explicitly used this pattern for SwAV.
**How to avoid:** Plan 10-02 must rewrite the SwAV class docstring completely, using the module-level docstring as source material.

### Pitfall 5: Tutorial notebook vs Markdown — Jupyter dependency
**What goes wrong:** If the tutorial is written as `notebooks/walkthrough.ipynb`, it requires Jupyter to be installed and creates an ongoing maintenance burden (cell outputs, kernel state).
**Why it happens:** ROADMAP mentions `notebooks/walkthrough.ipynb` as an option.
**How to avoid:** Use `docs/tutorial.md` Markdown format instead. A pure code-and-text tutorial in Markdown is renderable on GitHub without Jupyter. The REQUIREMENTS.md says "tutorial notebook **or guide**" — the guide format is simpler.

### Pitfall 6: Method table counts 15 items when only 14 are v1 methods
**What goes wrong:** Planner includes `supcon_finetune` as a 15th row in the method table, failing success criterion 5 ("all 14 v1 methods").
**Why it happens:** `available_methods()` returns 15 keys including `supcon_finetune`.
**How to avoid:** The method table covers 14 independently-motivated SSL methods. `supcon_finetune` is stage-2 fine-tuning of SupCon, not a separate method.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Model instantiation | Custom factory code in train.py | `method_dispatcher(cfg)` from `core.dispatcher` | Already exists with error messages |
| Config loading | `yaml.safe_load()` + manual dict access | `load_config(path)` from `core.config` | Pydantic validation, type safety |
| Dataset | Write a CIFAR-10 loader from scratch | `SSLDataModule` (+ ImageFolder data prep) | Handles multi-view, multi-crop, class-balanced |
| KNN evaluation | Build k-NN from scratch for tutorial | `KNNCallback` from `eval.knn_callback` | Already supports FAISS + brute-force |
| Documentation docstring format | Invent a new pattern | Mirror `SimCLRv1Module` class docstring exactly | Establishes consistency with already-compliant classes |

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.4.1 |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/` |

### Current Test Count
326 tests collected. All passing as of Phase 9 completion. `[VERIFIED: pytest --collect-only]`

### Phase 10 Test Mapping
Phase 10 is a documentation phase. Most success criteria are verified by human review, not automated tests. However, one smoke test is appropriate:

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DOC-01 | README exists and contains required sections | Manual review | `python -c "open('README.md')"` | No |
| DOC-02 | All class docstrings have required fields | Automated check | `pytest tests/test_docstrings.py -x` | No — Wave 0 gap |
| DOC-03 | Tutorial guide exists and covers 3 sections | Manual review | — | No |
| DOC-01 | `train.py` runs on toy data end-to-end | Smoke test | `pytest tests/test_train_script.py -x` | No — Wave 0 gap |

### Wave 0 Gaps
- [ ] `tests/test_docstrings.py` — verifies each LightningModule subclass docstring has Paper/Authors/Venue/arXiv/Gotchas/RefImpl fields (DOC-02)
- [ ] `tests/test_train_script.py` — smoke-tests `train.py --config configs/simclr_v1_resnet18.yaml` on toy data

*(Optional: DOC-01/DOC-03 success criteria are best verified by human review, not automated tests. The docstring and train.py tests are the automatable subset.)*

---

## Environment Availability

No new external dependencies are introduced in Phase 10. All required packages are in `requirements.txt`. `[VERIFIED: requirements.txt inspection]`

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| pytest | tests/test_docstrings.py, tests/test_train_script.py | Yes | 8.4.1 | — |
| lightning | train.py | Yes | 2.6.1 | — |
| pydantic | core/config.py | Yes | 2.12.5 | — |
| jupyter (optional) | docs/tutorial.md (if notebook format chosen) | Not verified | — | Use Markdown format instead |

**Recommendation:** Use `docs/tutorial.md` (Markdown) for the tutorial to avoid the Jupyter dependency.

---

## Open Questions

1. **CIFAR-10 data preparation for README quickstart**
   - What we know: `SSLDataModule` uses `ImageFolder`; CIFAR-10 via torchvision does not produce ImageFolder layout automatically.
   - What's unclear: Should `train.py` support `--dataset cifar10` (downloading automatically) or should the README document manual ImageFolder setup?
   - Recommendation: Add a `--dataset cifar10` shortcut to `train.py` using `torchvision.datasets.CIFAR10(download=True)` wrapped in a simple adapter, OR document the 2-command setup (`torchvision2imagefolder` or manual script). Either approach satisfies "under 5 commands."

2. **`simclr_resnet50.yaml` config**
   - What we know: Success criterion 1 says "single-command SimCLR training invocation" using CIFAR-10. The existing ResNet-50 config is LARS-only (`simclr_v1_resnet50_lars.yaml`). A standard AdamW ResNet-18 config exists.
   - What's unclear: Should Plan 10-01 create a `simclr_resnet50.yaml` (AdamW, ResNet-50) as the canonical quickstart config? ROADMAP plan 10-04 references `configs/simclr_resnet50.yaml` specifically.
   - Recommendation: Create `configs/simclr_resnet50.yaml` (AdamW, 200 epochs) as part of Plan 10-01. Use ResNet-18 for the CIFAR-10 quickstart (faster to train) and note ResNet-50 as the recommended config for larger datasets.

3. **Tutorial format: Jupyter vs Markdown**
   - What we know: REQUIREMENTS.md says "`notebooks/walkthrough.ipynb` or `docs/tutorial.md`". ROADMAP says "Assemble `notebooks/walkthrough.ipynb` or `docs/tutorial.md`".
   - What's unclear: User preference not specified in CONTEXT.md (no discussion phase).
   - Recommendation: Use `docs/tutorial.md` (Markdown). No Jupyter dependency, renderable on GitHub, easier to keep up-to-date.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | CIFAR-10 quickstart requires a data-preparation step or `train.py` shortcut for auto-download | train.py Design, Pitfall 2 | If SSLDataModule gains CIFAR-10 support, the quickstart path changes |
| A2 | `docs/tutorial.md` is preferred over `notebooks/walkthrough.ipynb` | Open Questions | If user strongly prefers Jupyter, create notebook instead — same content |
| A3 | DINOv2 row in method table notes "feature extraction only" — no `LightningModule` DOC-02 docstring needed | Codebase State Inventory | If DOC-02 is interpreted to include `eval/dinov2_demo.py`, add docstring there |

---

## Sources

### Primary (HIGH confidence)
- `/Users/yi-tingli/Documents/Projects/ml_topic_contrastive_learning/methods/` — all module source files read directly
- `/Users/yi-tingli/Documents/Projects/ml_topic_contrastive_learning/core/` — dispatcher, config, base class source
- `/Users/yi-tingli/Documents/Projects/ml_topic_contrastive_learning/configs/` — all 17 YAML configs verified
- `/Users/yi-tingli/Documents/Projects/ml_topic_contrastive_learning/eval/` — all 7 eval scripts verified
- `.planning/REQUIREMENTS.md` — DOC-01, DOC-02, DOC-03 definitions
- `.planning/ROADMAP.md` — plan descriptions and success criteria

### Secondary (MEDIUM confidence)
- `tests/` — smoke test patterns informed train.py design
- `methods/supcon/__init__.py` — two-stage workflow `train.py` command references

---

## Metadata

**Confidence breakdown:**
- Codebase state (what exists/missing): HIGH — direct file inspection
- DOC-02 docstring audit: HIGH — programmatic `ast.get_docstring()` check
- train.py design: HIGH — inferred from existing test patterns and dispatcher API
- Method table content: HIGH — from REQUIREMENTS.md + module docstrings (arXiv links already in code)
- Tutorial structure: MEDIUM — format choice (notebook vs Markdown) is an open question

**Research date:** 2026-05-03
**Valid until:** Phase 10 execution — no external dependencies, stable codebase
