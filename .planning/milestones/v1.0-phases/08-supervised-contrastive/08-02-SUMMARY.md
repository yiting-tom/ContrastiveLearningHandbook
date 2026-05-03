---
phase: 08
plan: 02
subsystem: data
tags: [class-balanced-sampler, ssl-data-module, supcon-config, data-pipeline]
dependency_graph:
  requires: [core/data.py, core/config.py]
  provides: [ClassBalancedSampler, SSLDataModule(sampler_type=class_balanced)]
  affects: [methods/supcon/, training pipeline]
tech_stack:
  added: []
  patterns: [class-balanced batch sampling, with-replacement sampling for small classes]
key_files:
  created: [tests/test_class_balanced_sampler.py]
  modified: [core/config.py, core/data.py]
decisions:
  - ClassBalancedSampler uses random.choices (with replacement) for small classes to avoid failures
  - SSLDataModule derives n_samples_per_class from batch_size // n_classes_per_batch dynamically
  - sampler_type is a string discriminator (not enum) to stay consistent with config conventions
metrics:
  duration_seconds: 124
  completed_date: "2026-04-10"
  tasks_completed: 3
  files_modified: 3
requirements: [SUP-01]
---

# Phase 8 Plan 2: ClassBalancedSampler + SSLDataModule Integration Summary

**One-liner:** ClassBalancedSampler guaranteeing n_samples_per_class per class per batch, wired into SSLDataModule via sampler_type="class_balanced" and SupConConfig extended with three new fields.

## What Was Built

### Task 2.1 — SupConConfig extension (`core/config.py`)

Added three fields to `SupConConfig`:
- `n_classes_per_batch: int = 8` — classes sampled per batch by ClassBalancedSampler
- `num_classes: int = 10` — number of output classes for the stage-2 classification head
- `projection_dim: int = 128` — projection head output dimension

### Task 2.2 — ClassBalancedSampler + SSLDataModule (`core/data.py`)

**ClassBalancedSampler** (`torch.utils.data.Sampler` subclass):
- Accepts a dataset with `.targets` attribute (ImageFolder-compatible)
- Guarantees exactly `n_samples_per_class` instances per chosen class per batch
- Uses `random.sample` (without replacement) to select `n_classes_per_batch` classes per batch
- Uses `random.choices` (with replacement) to sample indices per class — safe for small classes
- Raises `ValueError` if dataset has fewer classes than `n_classes_per_batch`
- Total length = `floor(len(dataset) / batch_size) * batch_size`

**SSLDataModule extensions:**
- New `__init__` params: `sampler_type: str | None = None`, `n_classes_per_batch: int | None = None`
- `train_dataloader()` branches on `sampler_type == "class_balanced"`:
  - Derives `n_samples_per_class = batch_size // n_classes_per_batch`
  - Constructs `ClassBalancedSampler`, sets `shuffle=False`
  - Passes `sampler=sampler` to DataLoader
- Default path unchanged: `sampler=None`, `shuffle=True`

### Task 2.3 — Unit tests (`tests/test_class_balanced_sampler.py`)

5 tests, all passing:

| Test | Coverage |
|------|----------|
| `test_min_class_count_per_batch` | SC-3: min class count per batch >= n_samples_per_class |
| `test_sampler_length` | `__len__` matches `__iter__` output, divisible by batch_size |
| `test_too_many_classes_raises` | ValueError when n_classes_per_batch > dataset classes |
| `test_dataloader_integration_no_error` | DataLoader works with sampler and shuffle=False |
| `test_shuffle_true_with_sampler_raises` | PyTorch incompatibility guard: shuffle=True raises ValueError |

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 2.1 | 64b86aa | feat(08-02): extend SupConConfig with n_classes_per_batch, num_classes, projection_dim |
| 2.2 | 970cd90 | feat(08-02): implement ClassBalancedSampler and extend SSLDataModule |
| 2.3 | 8991bfe | test(08-02): add ClassBalancedSampler unit tests — 5 tests covering SC-3 |

## Deviations from Plan

None — plan executed exactly as written.

## Success Criteria Coverage

- **SC-3**: Class-balanced sampler guarantees at least 2 instances per class per batch — verified in `test_min_class_count_per_batch` and the plan smoke test.
- **SC-5** (partial): `supcon` config path has `n_classes_per_batch` and `n_samples_per_class` wired into `SSLDataModule`.

## Known Stubs

None.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundary changes introduced.

## Self-Check: PASSED

- [x] `core/config.py` modified — SupConConfig has 3 new fields
- [x] `core/data.py` modified — ClassBalancedSampler class added, SSLDataModule extended
- [x] `tests/test_class_balanced_sampler.py` created — 5 tests, all passing
- [x] Commit 64b86aa exists
- [x] Commit 970cd90 exists
- [x] Commit 8991bfe exists
