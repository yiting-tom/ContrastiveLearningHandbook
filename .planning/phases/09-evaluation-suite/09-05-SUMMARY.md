---
phase: 09-evaluation-suite
plan: "05"
subsystem: testing
tags: [integration-test, knn, linear-probe, tsne, umap, finetune, cam, synthetic-data]

dependency_graph:
  requires:
    - phase: 09-evaluation-suite plan 01
      provides: eval/knn_callback.py (knn_predict)
    - phase: 09-evaluation-suite plan 02
      provides: eval/linear_probe.py (LinearProbeModule, extract_and_cache)
    - phase: 09-evaluation-suite plan 03
      provides: eval/tsne_vis.py (run_tsne), eval/umap_vis.py (run_umap)
    - phase: 09-evaluation-suite plan 04
      provides: eval/finetune.py (FinetuneModule), eval/cam_vis.py (run_cam, get_target_layer)
    - phase: 01-foundation
      provides: core/config.py (EvalConfig + all 6 sub-configs)
  provides:
    - tests/test_eval_integration.py -- 8 integration tests for full eval pipeline
  affects:
    - CI test suite (adds ~33s integration test suite)

tech-stack:
  added: []
  patterns:
    - Synthetic ImageFolder fixture (train/val splits, 3 classes, 32x32 PNGs)
    - Lightning checkpoint saved with pytorch-lightning_version key for load_from_checkpoint
    - Direct torch.save() checkpoint format (no training loop needed for checkpoint creation)
    - pytest.mark.slow marker for integration test filtering
    - Relaxed accuracy threshold (acc >= 0.0) for kNN on random-weight checkpoint

key-files:
  created:
    - tests/test_eval_integration.py
  modified:
    - eval/tsne_vis.py (PCA n_components capped to min(50, n_samples-1) for small datasets)
    - pyproject.toml (added pytest.mark.slow registration)

key-decisions:
  - "Checkpoint saved manually via torch.save with pytorch-lightning_version key -- no training loop required"
  - "PCA in tsne_vis.py auto-capped to min(50, n_samples-1, n_features) to handle small synthetic datasets (bug fix)"
  - "pytest.mark.slow registered in pyproject.toml markers section to suppress PytestUnknownMarkWarning"
  - "perplexity=5 used in integration test instead of default [10,30,50] to keep runtime under 120s"
  - "max_steps=2 for finetune Trainer to minimize runtime while still exercising the training loop"

requirements-completed: [FOUND-08]

metrics:
  duration_seconds: ~600
  completed_date: "2026-04-11"
  tasks_completed: 1
  files_created: 1
  files_modified: 2
---

# Phase 09 Plan 05: Full Evaluation Pipeline Integration Test Summary

**Integration test exercising all 6 eval components (kNN, linear probe, t-SNE, UMAP, finetune, CAM) on a synthetic checkpoint with random weights -- no network access, no training loop required**

## What Was Built

`tests/test_eval_integration.py` -- 8 integration tests that validate the full evaluation pipeline end-to-end:

### Fixtures

1. **`synthetic_data(tmp_path)`**: Creates a synthetic ImageFolder with train/ (3 classes x 10 images) and val/ (3 classes x 5 images) splits, 32x32 RGB PNGs.

2. **`synthetic_checkpoint(tmp_path, synthetic_data)`**: Builds `SimCLRv1Module` with resnet18 (random weights), saves as valid Lightning checkpoint with `pytorch-lightning_version` key. Returns `(ckpt_path, cfg, data_dir)`.

### Tests

| Test | Component | Key Verification |
|------|-----------|-----------------|
| `test_eval_config_schema_exists` | FOUND-08 | All 6 sub-configs importable + EvalConfig constructable |
| `test_knn_on_synthetic` | kNN | `knn_predict` returns `acc >= 0.0` (D-05 relaxed threshold) |
| `test_linear_probe_on_synthetic` | Linear probe | D-04: cache files use `{ckpt_stem}_features_train.pt` naming |
| `test_tsne_on_synthetic` | t-SNE | 1 PNG with "perp" in filename, non-empty |
| `test_umap_on_synthetic` | UMAP | PNG exists + reducer returned |
| `test_finetune_on_synthetic` | Fine-tuning | Trainer.fit completes without error |
| `test_cam_on_synthetic` | CAM | PNG overlay files exist and are non-empty |
| `test_full_pipeline` | All 6 | Runs all components in sequence on same checkpoint |

## Commits

| Task | Commit | Type | Description |
|------|--------|------|-------------|
| Task 1 | f1623ff | feat | Integration test + bug fixes |

## Test Results

**8/8 tests pass** in 33.4s (well within 120s timeout)

```
tests/test_eval_integration.py ........  [100%]
8 passed, 14 warnings in 33.40s
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Missing pytorch-lightning_version key in synthetic checkpoint**
- **Found during:** Task 1 (test_knn_on_synthetic)
- **Issue:** `load_from_checkpoint` requires `checkpoint["pytorch-lightning_version"]` key; manually saved checkpoint lacked it, causing `KeyError`
- **Fix:** Added `"pytorch-lightning_version": L.__version__` to `torch.save()` dict in `synthetic_checkpoint` fixture
- **Files modified:** tests/test_eval_integration.py
- **Commit:** f1623ff

**2. [Rule 1 - Bug] PCA n_components exceeds min(n_samples, n_features) for small datasets**
- **Found during:** Task 1 (test_tsne_on_synthetic)
- **Issue:** `PCA(n_components=50)` fails when dataset has <50 samples (our synthetic train set has 30). `ValueError: n_components=50 must be between 0 and min(n_samples, n_features)=30`
- **Fix:** Capped n_components to `min(50, n_samples - 1, n_features)` in `eval/tsne_vis.py`
- **Files modified:** eval/tsne_vis.py
- **Commit:** f1623ff
- **Verification:** All 7 original t-SNE unit tests still pass

**3. [Rule 2 - Missing critical functionality] pytest.mark.slow not registered**
- **Found during:** Task 1 (all @pytest.mark.slow tests)
- **Issue:** `PytestUnknownMarkWarning` for every test using `@pytest.mark.slow`; unregistered marks generate warnings and can silently break mark-based filtering
- **Fix:** Added `markers = ["slow: ..."]` to `[tool.pytest.ini_options]` in `pyproject.toml`
- **Files modified:** pyproject.toml
- **Commit:** f1623ff

---

**Total deviations:** 3 auto-fixed (2 Rule 1 bugs, 1 Rule 2 missing config)

## Known Stubs

None -- all 6 eval components are fully wired to real implementations from plans 01-04.

## Threat Surface Scan

No new network endpoints, auth paths, or trust boundaries. All tests operate on local tmp_path fixtures only. T-09-12 (slow integration test) mitigated: 8 tests complete in 33s, well within 120s.

## Self-Check

Files verified:
- `tests/test_eval_integration.py`: FOUND
- `eval/tsne_vis.py`: FOUND (modified, PCA cap fix)
- `pyproject.toml`: FOUND (modified, slow marker registered)

Commits verified:
- `f1623ff`: feat(09-05): implement full evaluation pipeline integration test

Tests: 8/8 passing

## Self-Check: PASSED

---
*Phase: 09-evaluation-suite*
*Completed: 2026-04-11*
