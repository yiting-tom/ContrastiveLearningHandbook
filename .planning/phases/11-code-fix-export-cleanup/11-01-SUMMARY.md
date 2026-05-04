---
phase: "11"
plan: "11-01"
subsystem: "data-pipeline"
tags: [wire, instance-discrimination, indexed-dataset, collate, memory-bank]
dependency_graph:
  requires: ["11-03"]
  provides: ["WIRE-01"]
  affects: ["train.py", "core/data.py"]
tech_stack:
  added: []
  patterns: ["isinstance dispatch for collate selection", "pre-wrapped dataset injection via SSLDataModule.dataset param"]
key_files:
  created: []
  modified:
    - core/data.py
    - train.py
decisions:
  - "D-01: IndexedDataset wrapping stays in train.py, not inside SSLDataModule.setup()"
  - "D-02: SSLDataModule remains self-contained — collate selection via isinstance, no collate_fn param"
metrics:
  duration: "~8 minutes"
  completed: "2026-05-04"
  tasks_completed: 2
  files_modified: 2
---

# Phase 11 Plan 01: IndexedDataset Wiring for Instance Discrimination Summary

**One-liner:** Wire InstanceDiscrimination to IndexedDataset by adding an isinstance branch in SSLDataModule.train_dataloader() and a pre-wrapping block in train.py main() that passes the wrapped dataset via the existing dataset= parameter.

## What Was Built

- `core/data.py`: Added `if isinstance(self.train_dataset, IndexedDataset): collate = ssl_collate_with_index` branch in `SSLDataModule.train_dataloader()`, placed before the existing `MultiCropDataset` branch (lines 348-349).
- `train.py`: Extended `from core.data import SSLDataModule` to also import `IndexedDataset`. Inserted WIRE-01 wrapping block between config override and supcon_finetune if/else: when `cfg.method == "instance_discrimination"`, builds `ImageFolder` from the train directory with weak augmentation, wraps it in `IndexedDataset`, and assigns it to `_wrapped_dataset`. The `SSLDataModule` constructor call was updated to accept `dataset=_wrapped_dataset` (None for all other methods).

## Tasks Completed

| Task | Description | Commit | Files Modified |
|------|-------------|--------|----------------|
| 1 | Add IndexedDataset isinstance branch in SSLDataModule.train_dataloader() | a1c798e | core/data.py |
| 2 | Wrap ImageFolder in IndexedDataset for instance_discrimination in train.py | e7e12dc | train.py |

## Verification Results

All 6 plan checks passed:
1. `train.py` parses without syntax error
2. `core/data.py` parses without syntax error
3. `isinstance(self.train_dataset, IndexedDataset)` branch present in core/data.py
4. `ssl_collate_with_index` selected immediately after IndexedDataset branch
5. `IndexedDataset` imported and used in train.py
6. `instance_discrimination` guard condition and `dataset=_wrapped_dataset` in train.py

## Deviations from Plan

None - plan executed exactly as written. The supcon_finetune if/else (added by 11-03 in Wave 1) was already in place; the WIRE-01 block was inserted before it as specified.

## Threat Flags

None. No new network endpoints, auth paths, or external surface added. Path traversal is limited to local filesystem as noted in the plan's threat register (T-11-01-1 accepted).

## Known Stubs

None. The wiring is complete end-to-end: IndexedDataset wrapping in train.py connects to ssl_collate_with_index selection in SSLDataModule.train_dataloader().

## Self-Check: PASSED

- [x] core/data.py modified with IndexedDataset branch — FOUND
- [x] train.py modified with import, wrapping block, dataset= kwarg — FOUND
- [x] Commit a1c798e exists — FOUND (feat(11-01): add IndexedDataset isinstance branch)
- [x] Commit e7e12dc exists — FOUND (feat(11-01): wrap ImageFolder in IndexedDataset)
