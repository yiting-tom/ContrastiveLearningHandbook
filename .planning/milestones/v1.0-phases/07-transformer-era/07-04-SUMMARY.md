---
phase: 07-transformer-era
plan: "04"
subsystem: eval-scripts, configs, tests
tags: [dinov2, moco_v3, dino, yaml-config, smoke-test, doc02, tutorial]
dependency_graph:
  requires: [07-02, 07-03]
  provides: [ERA4-03, ERA4-01, ERA4-02]
  affects: [eval/, configs/, tests/]
tech_stack:
  added: [timm (DINOv2 demo), sklearn.neighbors.KNeighborsClassifier, sklearn.linear_model.SGDClassifier]
  patterns: [standalone-eval-script, yaml-config-pattern, smoke-test-pattern, doc02-docstring-pattern]
key_files:
  created:
    - eval/__init__.py
    - eval/dinov2_demo.py
    - configs/moco_v3_vit_small.yaml
    - configs/dino_vit_small.yaml
    - tests/test_dinov2_demo.py
    - tests/test_smoke_transformer.py
  modified: []
decisions:
  - "DINOv2 demo uses get_args() helper for testable argparse (not inline parse_args in main)"
  - "eval/ created as Python package (eval/__init__.py) to enable module-style import in tests"
  - "dino_vit_small.yaml includes swav block (documents multi-crop settings); SSLDataModule triggers multi-crop via pre-wrapped MultiCropDataset, not YAML config"
  - "test_smoke_dino_train uses n_prototypes=128 and 2-view mode for CPU test speed"
  - "DOC-02 tests check both module-level and class-level docstrings (joined) to be robust"
metrics:
  duration_seconds: ~300
  completed_date: "2026-04-10T05:44:40Z"
  tasks_completed: 2
  files_created: 6
  tests_added: 11
  tests_total: 244
---

# Phase 07 Plan 04: DINOv2 Demo, YAML Configs, DOC-02, and Smoke Tests Summary

DINOv2 feature extraction demo script (timm vit_small_patch14_dinov2.lvd142m), YAML training configs for MoCo v3 and DINO ViT-Small, DOC-02 docstring verification, and 3-epoch CPU smoke tests for both transformer-era methods.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create DINOv2 demo script | c73d0b2 | eval/__init__.py, eval/dinov2_demo.py, tests/test_dinov2_demo.py |
| 2 | YAML configs, DOC-02 verification, and smoke tests | f57f2f9 | configs/moco_v3_vit_small.yaml, configs/dino_vit_small.yaml, tests/test_smoke_transformer.py |

## What Was Built

### Task 1: DINOv2 Demo Script

`eval/dinov2_demo.py` — standalone tutorial script for DINOv2 feature extraction:

- Loads `vit_small_patch14_dinov2.lvd142m` via `timm.create_model(..., num_classes=0, pretrained=True)`
- CLI: `--dataset cifar10|stl10|imagefolder`, `--data-dir`, `--batch-size`, `--k`, `--device`
- `extract_features()` extracts pooled features from any DL with no_grad
- `run_knn()` uses `KNeighborsClassifier(n_neighbors=k, metric='cosine')` for zero-shot eval
- `run_linear_probe()` uses `SGDClassifier` on standardized features for linear probing
- Module docstring documents: register tokens (Oct 2023), `DINO -> DINOv2 -> DINOv2 + Registers`, "DINOv3" does not exist

### Task 2: YAML Configs, DOC-02, and Smoke Tests

**configs/moco_v3_vit_small.yaml:**
- `method: moco_v3`, `backbone: vit_small_patch16_224`, `optimizer: adamw`, `gradient_clip_val: 1.0`
- `moco_v3.temperature: 0.2`, `moco_v3.momentum: 0.99`, `moco_v3.predictor_hidden_dim: 4096`
- Comments document the 3 key differences from MoCo v1/v2

**configs/dino_vit_small.yaml:**
- `method: dino`, `backbone: vit_small_patch16_224`, `optimizer: adamw`, `gradient_clip_val: 3.0`
- `dino.n_prototypes: 65536`, teacher temp warmup, `n_views: 8`
- `swav` block documents multi-crop settings (not functionally required for config loading)

**tests/test_smoke_transformer.py (6 tests):**
- `test_moco_v3_yaml_valid` — load_config() + field assertions
- `test_dino_yaml_valid` — load_config() + field assertions
- `test_smoke_moco_v3_train` — 3-epoch CPU training, finite loss check
- `test_smoke_dino_train` — 3-epoch CPU training (2-view mode), finite loss check
- `test_doc02_moco_v3` — Chen, ICCV 2021, arxiv, patch in docstring
- `test_doc02_dino` — Caron, ICCV 2021, arxiv, centering in docstring

## Verification Results

```
python -m pytest tests/test_dinov2_demo.py tests/test_smoke_transformer.py -x -q
11 passed

python -c "from core.config import load_config; c = load_config('configs/moco_v3_vit_small.yaml'); print(c.method, c.moco_v3.temperature)"
moco_v3 0.2

python -c "from core.config import load_config; c = load_config('configs/dino_vit_small.yaml'); print(c.method, c.dino.n_prototypes)"
dino 65536

python -m pytest tests/ -q
244 passed, 141 warnings in 217.13s
```

## Deviations from Plan

None - plan executed exactly as written.

The task note about checking SSLDataModule for DINO multi-crop config was followed: `SSLDataModule` triggers multi-crop only when a pre-wrapped `MultiCropDataset` is passed via the `dataset` parameter, NOT from the YAML `swav` block. The `swav` block in `dino_vit_small.yaml` is present as documentation for the tutorial user, as it passes Pydantic validation via `SwAVConfig` and is ignored by the training infrastructure unless explicitly used.

## Decisions Made

1. `get_args()` is a named helper function (not inline) so `test_dinov2_demo_argparse_defaults` can call it directly without subprocess.
2. `eval/__init__.py` created as Python package to enable `import eval.dinov2_demo` in tests.
3. DOC-02 tests scan both module-level (`mod.__doc__`) and class-level (`Class.__doc__`) docstrings joined together for robustness.
4. `test_smoke_dino_train` uses `n_prototypes=128` and 2-view mode (not 8-view multi-crop) for fast CPU execution (~5s vs ~60s).

## Known Stubs

None. All implemented functionality is wired to real code paths.

## Threat Flags

None. `eval/dinov2_demo.py` CLI uses argparse with `choices=` constraint for `--dataset` and accepts `--data-dir` as a user filesystem path (local use only). Threat T-07-01 is accepted per plan threat model.

## Self-Check: PASSED

Files exist:
- FOUND: eval/__init__.py
- FOUND: eval/dinov2_demo.py
- FOUND: configs/moco_v3_vit_small.yaml
- FOUND: configs/dino_vit_small.yaml
- FOUND: tests/test_dinov2_demo.py
- FOUND: tests/test_smoke_transformer.py

Commits exist:
- c73d0b2 (Task 1: DINOv2 demo script)
- f57f2f9 (Task 2: YAML configs, smoke tests)
