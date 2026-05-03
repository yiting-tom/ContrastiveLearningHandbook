# Phase 9: Evaluation Suite - Context

**Gathered:** 2026-04-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a complete evaluation toolkit that can measure and visualize representation quality for any trained SSL method without modifying method code. Delivers: `KNNCallback` (in-training), and five offline eval scripts (`linear_probe.py`, `tsne_vis.py`, `umap_vis.py`, `finetune.py`, `cam_vis.py`) plus an integration test. No new training methods; no changes to existing method modules.

</domain>

<decisions>
## Implementation Decisions

### Eval Script Invocation
- **D-01:** All five offline eval scripts (`linear_probe.py`, `tsne_vis.py`, `umap_vis.py`, `finetune.py`, `cam_vis.py`) use the same invocation pattern:
  ```
  python eval/<script>.py configs/simclr.yaml --ckpt outputs/run1/checkpoints/epoch-99.ckpt
  ```
  The YAML config supplies all eval settings (via the existing `eval.*` sub-configs). The checkpoint path is passed separately via `--ckpt`. No unified entry point — each script is standalone.

### KNNCallback Data Source
- **D-02:** `KNNCallback` accesses labeled val data via `trainer.datamodule.val_dataloader()`. The existing `SSLDataModule.val_dataloader()` already returns a non-augmented ImageFolder split when a `val/` subdirectory exists under `data_dir`. No new DataModule interface required. Implication: tutorials that use `KNNCallback` must structure data with a `val/` split containing class subfolders.

### Dependencies
- **D-03:** All three eval-specific libraries go into `requirements.txt` (not a separate extras file):
  - `faiss-cpu>=1.7` (k-NN at scale)
  - `umap-learn>=0.5` (UMAP visualization)
  - `pytorch-grad-cam>=1.4` (CAM visualization)
  This keeps the tutorial install simple: `pip install -r requirements.txt` and everything works.

### Feature Cache Location (Linear Probe)
- **D-04:** Linear probe caches pre-extracted features in a sibling `cache/` directory next to the checkpoint file — auto-located, no config needed:
  ```
  outputs/simclr_run/checkpoints/epoch-99.ckpt
  outputs/simclr_run/cache/
    features_train.pt
    labels_train.pt
    features_val.pt
    labels_val.pt
  ```
  Cache is keyed to the checkpoint filename so different checkpoints don't collide.

### Integration Test Strategy
- **D-05:** Plan 09-07 integration test uses a **synthetic checkpoint + synthetic ImageFolder** — no network access, no training. Concretely:
  1. Create synthetic ImageFolder (3 classes, 30 images, 32×32 PNG via `torchvision`)
  2. Initialize `SimCLRModule` with `resnet18` backbone (tiny, fast)
  3. Save to `.ckpt` via `trainer.save_checkpoint()` (random weights, but valid Lightning checkpoint)
  4. Run full eval pipeline: KNN + linear_probe + tsne + umap + cam
  5. Assert all output files exist and no exceptions raised
  - The `eval/knn_acc > 0.1` threshold from the roadmap is relaxed to `eval/knn_acc >= 0.0` (just checks it runs without crash). The knn_acc > 0.1 threshold is meaningful only with real representations.

### Claude's Discretion
- ArgumentParser argument names for `--ckpt` and any secondary flags (e.g., `--output-dir`, `--device`)
- Whether `eval/` scripts are importable modules or pure `__main__` scripts (prefer importable for testability)
- Exact PNG filename conventions for t-SNE (`tsne_perp10.png`, `tsne_p10.png`, etc.) — just keep perplexity in filename
- FAISS brute-force fallback threshold (>100K is specified; implementation detail of how to detect dataset size)
- EigenCAM target layer selection logic implementation details (ResNet vs ViT detection)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Evaluation requirements (spec-level detail)
- `.planning/REQUIREMENTS.md` §Evaluation Suite — EVAL-01 through EVAL-06: full per-script specs including gotchas, default hyperparameters, FAISS threshold, target layer conventions, and the integration test requirement (EVAL-01 is KNNCallback, EVAL-06 is CAM)
- `.planning/REQUIREMENTS.md` §Shared Infrastructure — INFRA-04, INFRA-05 (MultiCropDataset, PredictorHead) — not eval-specific but check status before planning

### Existing eval config schemas
- `core/config.py` lines 153–215 — `LinearProbeConfig`, `KNNConfig`, `TSNEConfig`, `UMAPConfig`, `FinetuneConfig`, `CAMConfig`, and `EvalConfig` are already implemented. Do NOT redefine them. All eval scripts load via `TrainConfig.model_validate(yaml.safe_load(...))` and access `cfg.eval.*`.

### Existing DataModule (KNNCallback integration point)
- `core/data.py` lines 333–385 — `SSLDataModule.val_dataloader()` already handles the labeled val split. `val_dataset` is `None` if no `val/` directory exists. KNNCallback must guard against `None` gracefully.

### Phase 9 roadmap detail
- `.planning/ROADMAP.md` §Phase 9 — Pre-specified plan outlines (09-01 through 09-07) and 6 success criteria. Plan outlines are authoritative on what each plan must deliver.

No external ADRs — requirements are fully captured in the files above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `core/config.py` — `EvalConfig` + all 6 sub-configs already defined; eval scripts load the same `TrainConfig` YAML as training scripts
- `core/data.py` — `SSLDataModule.val_dataloader()` returns labeled ImageFolder val split (non-augmented); used directly by `KNNCallback`
- `core/backbone.py` — `build_backbone()` returns `(backbone, feat_dim)`; eval scripts call `pl_module.backbone` after loading checkpoint
- `core/base.py` — `BaseSSLModule` is the Lightning checkpoint base; all method checkpoints load via `BaseSSLModule.load_from_checkpoint()` (or subclass equivalent)
- `eval/__init__.py` — directory already exists; `eval/dinov2_demo.py` exists as a reference for how a standalone eval script can be structured

### Established Patterns
- All YAML configs load via `TrainConfig.model_validate(yaml.safe_load(open(path)))` — eval scripts must follow the same pattern (no argparse-only configs)
- `self.log(...)` / `self.log_dict(...)` for metrics — KNNCallback logs `eval/knn_acc` this way via `pl_module.log(...)`
- Test fixtures use synthetic `ImageFolder` created in a tmp directory (established in Phase 1 tests) — integration test reuses this pattern
- `extra='forbid'` on all Pydantic configs — adding new YAML keys without adding to the schema will raise `ValidationError`

### Integration Points
- `KNNCallback` integrates with Lightning's `Trainer` via `callbacks=[KNNCallback(...)]` in the training script; reads `trainer.datamodule` directly
- Offline eval scripts (`linear_probe.py`, etc.) are independent of the training loop — they only need a checkpoint path and a data directory
- Feature cache sits at `{checkpoint_path.parent.parent}/cache/` — eval scripts derive this path automatically from `--ckpt`
- `pytorch-grad-cam` needs a target layer and optional reshape transform — architecture-aware selection lives in `cam_vis.py` (not in `core/`)

</code_context>

<specifics>
## Specific Ideas

No specific reference implementations or "I want it like X" moments came up — open to standard approaches for script structure and output formatting.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 09-evaluation-suite*
*Context gathered: 2026-04-10*
