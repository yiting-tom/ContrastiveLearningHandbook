---
phase: 05-swav-and-infomin
plan: "06"
subsystem: methods/infomin
tags: [infomin, simclr, augmentation, contrastive-learning]
dependency_graph:
  requires: [core/config.py, core/data.py, methods/simclr/module.py]
  provides: [methods/infomin/module.py, methods/infomin/__init__.py, tools/compare_augmentations.py]
  affects: [methods/__init__.py, methods/simclr/module.py, core/config.py]
tech_stack:
  added: [InfoMinConfig, InfoMinModule, build_augmentation() hook, compare_augmentations.py]
  patterns: [classmethod hook for subclass augmentation override, setup() production wiring]
key_files:
  created:
    - methods/infomin/module.py
    - methods/infomin/__init__.py
    - tools/compare_augmentations.py
    - tests/test_infomin.py
  modified:
    - methods/simclr/module.py
    - methods/__init__.py
    - core/config.py
decisions:
  - "build_augmentation() is a classmethod on SimCLRv1Module so InfoMinModule.build_augmentation() can be called without an instance (used in both setup() wiring and standalone test data creation)"
  - "InfoMinModule.setup() reads cfg.data_dir to avoid setup() failure when trainer.fit() calls it regardless of external dataloader -- tests must pass data_dir in cfg"
  - "SwAV test_sinkhorn_row_sums_uniform fails when run after test_simclr.py -- confirmed pre-existing on base branch, not caused by this plan"
metrics:
  duration_seconds: 1468
  completed_date: "2026-04-08"
  tasks_completed: 2
  files_modified: 7
requirements: [ERA2-06]
---

# Phase 05 Plan 06: InfoMinModule and Augmentation Comparison Summary

InfoMinModule (Tian et al., NeurIPS 2020) implemented as a SimCLRv1Module subclass with aggressive augmentation (s=1.5, grayscale p=0.4, no Gaussian blur) via build_augmentation() classmethod hook, dispatcher registration, production setup() wiring, and side-by-side comparison visualization script.

## What Was Built

### SimCLRv1Module.build_augmentation() Hook
Added `@classmethod build_augmentation(cls, size=224) -> ContrastiveAugmentation` to `SimCLRv1Module`. This is a backward-compatible hook -- existing training paths are unaffected. Subclasses override this method to customize augmentation policy. Requires importing `ContrastiveAugmentation` from `core.data`.

### InfoMinModule
`methods/infomin/module.py` -- `InfoMinModule(SimCLRv1Module)` with:
- `build_augmentation()` override: `s=1.5` color jitter (brightness upper=2.2), `grayscale_prob=0.4`, no GaussianBlur by default (`use_blur=False`)
- `__init__()` reads `InfoMinConfig` from `cfg.infomin` (or defaults)
- `setup()` constructs `ImageFolder` with `MultiViewTransform(InfoMinModule.build_augmentation())` for production `method: infomin` YAML usage

### InfoMinConfig
Added `InfoMinConfig` to `core/config.py` with `color_strength=1.5`, `grayscale_prob=0.4`, `use_blur=False`. Added `infomin: Optional[InfoMinConfig] = None` to `TrainConfig`.

### Dispatcher Registration
`methods/infomin/__init__.py` registers `"infomin"` via `register_method()`. `methods/__init__.py` auto-imports `methods.infomin` to trigger registration on startup.

### tools/compare_augmentations.py
CLI script with `--image` (required), `--output`, `--n-samples`, `--size`. Uses `matplotlib` Agg backend (headless). Calls `SimCLRv1Module.build_augmentation()` and `InfoMinModule.build_augmentation()` hooks. Produces 2-row grid (SimCLR top, InfoMin bottom) with original image in column 0 and augmented views in columns 1..n_samples. Row labels identify each method's policy.

### tests/test_infomin.py
9 tests: classmethod existence check, subclass relationship, no-GaussianBlur inspection, s=1.5 color jitter (brightness upper=2.2), grayscale p=0.4, dispatcher registration (×2), 5-epoch training via production data path (build_augmentation() -> MultiViewTransform -> DataLoader).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Config] InfoMinConfig not in core/config.py**
- **Found during:** Task 1 implementation
- **Issue:** Plan assumes `InfoMinConfig` exists in `core/config.py` (Plan 04 was expected to add it) but it wasn't there
- **Fix:** Added `InfoMinConfig` class and `infomin: Optional[InfoMinConfig] = None` field to `TrainConfig`
- **Files modified:** core/config.py
- **Commit:** b4989de

**2. [Rule 1 - Bug] Dispatcher registration fails after clean_registry fixture**
- **Found during:** GREEN phase test run
- **Issue:** `clean_registry` fixture saves registry snapshot before first `import methods.infomin`. After test, it restores to empty. Next test's `import methods.infomin` is a no-op (Python cached). Registry stays empty, test fails.
- **Fix:** Changed dispatcher registration tests to use `if "infomin" not in available_methods(): register_method(...)` pattern (consistent with existing test_simclr.py pattern)
- **Files modified:** tests/test_infomin.py
- **Commit:** b4989de

**3. [Rule 1 - Bug] InfoMinModule.setup() called by Lightning trainer even when train_dataloaders passed directly**
- **Found during:** Test 7 training test
- **Issue:** `InfoMinModule.setup()` tried `ImageFolder("data")` (default cfg.data_dir) when test passed `train_dataloaders=train_loader` directly to `trainer.fit()`
- **Fix:** Pass `data_dir=str(large_imagefolder)` in test config so `setup()` can find the data
- **Files modified:** tests/test_infomin.py
- **Commit:** b4989de

## Test Results

- `python -m pytest tests/test_infomin.py -v`: 9/9 passed
- `python -m pytest tests/ --ignore=tests/test_swav.py`: 169/169 passed
- Pre-existing `test_swav.py::test_sinkhorn_row_sums_uniform` fails when run after `test_simclr.py` -- confirmed present on base branch, not introduced by this plan
- `grep -n "build_augmentation" methods/simclr/module.py`: hook confirmed at line 87
- `python -c "import ast; ast.parse(open('tools/compare_augmentations.py').read()); print('Syntax OK')"`: Syntax OK

## Known Stubs

None. All acceptance criteria satisfied with real implementation.

## Threat Flags

None. `compare_augmentations.py` reads a local file path from CLI args (T-05-07: accept disposition -- local tool, no network input).

## Self-Check

Files exist:
- methods/infomin/module.py: FOUND
- methods/infomin/__init__.py: FOUND
- tools/compare_augmentations.py: FOUND
- tests/test_infomin.py: FOUND

Commits:
- 5af9c4f: test(05-06) RED phase failing tests
- b4989de: feat(05-06) GREEN phase implementation
- 415aeea: feat(05-06) comparison script
