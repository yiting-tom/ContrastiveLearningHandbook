---
phase: 09-evaluation-suite
plan: 02
subsystem: eval
tags: [linear-probe, feature-caching, lightning, sgd, multisteplr]
dependency_graph:
  requires:
    - core/config.py (LinearProbeConfig, TrainConfig)
    - core/dispatcher.py (get_method)
    - core/base.py (BaseSSLModule)
  provides:
    - eval/linear_probe.py (LinearProbeModule, extract_and_cache, main)
  affects:
    - tests/test_eval_linear_probe.py
tech_stack:
  added: []
  patterns:
    - TDD (RED → GREEN)
    - Lightning LightningModule for linear classifier
    - Checkpoint-keyed feature caching to disk
    - L2-normalized features via F.normalize
    - SGD + MultiStepLR for linear probe training
key_files:
  created:
    - eval/linear_probe.py
    - tests/test_eval_linear_probe.py
  modified: []
decisions:
  - "SGD weight_decay=0.0 asserted at optimizer construction time to catch config drift"
  - "Cache key derived from Path(ckpt_path).stem for collision-free multi-checkpoint caching (D-04)"
  - "Multi-view SSL batch handled by slicing first view (imgs[:, 0]) when batch[0].ndim==5"
  - "torch.load(weights_only=True) used for cache files to mitigate T-09-04 pickle tampering risk"
metrics:
  duration_seconds: ~180
  completed_date: "2026-04-11"
  tasks_completed: 1
  files_created: 2
---

# Phase 09 Plan 02: Linear Probe Evaluation Summary

**One-liner:** LinearProbeModule with SGD wd=0.0 and MultiStepLR [60,80] milestone schedule, plus checkpoint-keyed feature caching to disk via extract_and_cache.

## What Was Built

`eval/linear_probe.py` — importable module + standalone `__main__` script that:

1. **`extract_and_cache(backbone, dataloader, cache_dir, split, device, ckpt_path)`**
   - Derives cache key from `Path(ckpt_path).stem` (D-04 checkpoint-keyed caching)
   - Cache filenames: `{ckpt_stem}_features_{split}.pt` and `{ckpt_stem}_labels_{split}.pt`
   - Skips extraction on cache hit (loads and returns cached tensors)
   - L2-normalizes features via `F.normalize(..., dim=1)` before saving
   - Handles SSL multi-view batches (`batch[0].ndim == 5` → takes first view)
   - Uses `torch.load(weights_only=True)` for safe deserialization (T-09-04)

2. **`class LinearProbeModule(L.LightningModule)`**
   - `__init__(feat_dim, num_classes, lp_cfg)`: creates `nn.Linear(feat_dim, num_classes)` + `nn.CrossEntropyLoss()`
   - `training_step`: cross-entropy loss + logs `train/loss` and `train/acc`
   - `validation_step`: logs `val/loss` and `val/acc`
   - `configure_optimizers`: SGD with `weight_decay=0.0` (asserted), `MultiStepLR(milestones=lp_cfg.milestones)`

3. **`main()`** — CLI entry point following D-01 invocation pattern:
   - Loads config via `TrainConfig.model_validate(yaml.safe_load(...))`
   - Imports `methods` package to trigger dispatcher registration
   - Loads checkpoint via `get_method(cfg.method).load_from_checkpoint(ckpt_path, cfg=cfg)`
   - Resolves cache dir: `{ckpt_path.parent.parent}/cache/`
   - Builds train/val loaders from `SSLDataModule`, extracts cached features
   - Trains `LinearProbeModule` and prints final `val/acc`

## Commits

| Task | Commit | Type | Description |
|------|--------|------|-------------|
| TDD RED | c472826 | test | Failing tests for LinearProbeModule + extract_and_cache (9 tests) |
| TDD GREEN | 36f0be0 | feat | Implement LinearProbeModule and feature caching |

## Test Results

- **9/9 tests pass** (`pytest tests/test_eval_linear_probe.py -x`)
- Tests cover: init, SGD wd=0.0, MultiStepLR milestones, train/val logging, cache file creation, cache loading (cache hit), L2 normalization, multi-view batch handling

## Deviations from Plan

None — plan executed exactly as written.

The implementation follows all plan requirements:
- SGD weight_decay=0.0 with assertion
- MultiStepLR with milestones from `lp_cfg.milestones`
- Checkpoint-keyed cache filenames (D-04)
- `torch.load(weights_only=True)` for T-09-04 mitigation
- D-01 argparse invocation pattern with `--ckpt`

## Known Stubs

None — all functionality is fully wired. The `main()` function is not exercised by unit tests (integration tested in 09-07), but all components it calls are individually tested.

## Threat Flags

None — all new surface matches the plan's threat model (T-09-04 `weights_only=True` applied, T-09-05 accepted, T-09-06 accepted).

## Self-Check: PASSED

- `eval/linear_probe.py` exists: FOUND
- `tests/test_eval_linear_probe.py` exists: FOUND
- Commit c472826 exists: FOUND
- Commit 36f0be0 exists: FOUND
- All 9 tests pass: CONFIRMED
