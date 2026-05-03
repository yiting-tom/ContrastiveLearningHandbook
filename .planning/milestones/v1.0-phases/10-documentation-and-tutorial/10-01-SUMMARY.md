---
phase: 10
plan: 01
subsystem: documentation
tags: [readme, train-script, smoke-test, doc-01, cli, entry-point]
dependency_graph:
  requires: []
  provides: [train.py, tests/test_train_script.py, README.md]
  affects: [all 14 SSL methods via dispatcher, all 17 YAML configs, eval/ scripts]
tech_stack:
  added: []
  patterns: [argparse CLI, Pydantic model_copy override, monkeypatch Trainer.__init__]
key_files:
  created:
    - train.py
    - tests/test_train_script.py
    - README.md
  modified: []
decisions:
  - "train.py uses load_config() not raw yaml.safe_load — consistent with production smoke tests"
  - "KNNCallback import is lazy (inside if-block) to keep --help startup fast"
  - "Smoke test monkeypatches L.Trainer.__init__ to force CPU+1 batch without modifying train.py signature"
  - "README uses exact section headings from plan spec to support future grep-based link checks"
metrics:
  duration_minutes: 20
  completed_date: "2026-05-03"
  tasks_completed: 3
  files_changed: 3
---

# Phase 10 Plan 01: Train Script and README Summary

Single-line: Lightning CLI entry point (`train.py`) + smoke test + full DOC-01 README with 14-method table.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create train.py — CLI entry point for all SSL methods | cfd9663 | train.py |
| 2 | Create tests/test_train_script.py — smoke test for train.py | d55f0f5 | tests/test_train_script.py |
| 3 | Write README.md — DOC-01 (overview, install, quickstart, method table, eval) | 844d0a7 | README.md |

## What Was Built

**train.py** is a 73-line argparse CLI that accepts `--config`, `--data-dir`, and `--ckpt-path`.
It calls `load_config()`, `method_dispatcher()`, builds `SSLDataModule`, conditionally attaches
`KNNCallback`, and runs `Trainer.fit()`. Top-level `import methods` triggers `register_method()`
for all 14 SSL methods. Data-dir override uses `cfg.model_copy(update={"data_dir": ...})`.

**tests/test_train_script.py** provides 4 smoke tests:
1. `test_train_py_help_exits_zero` — subprocess `--help` exits 0, all 3 flags present
2. `test_train_py_missing_config_fails` — missing `--config` exits non-zero
3. `test_train_py_runs_one_batch_on_toy_data` — full load_config → dispatcher → SSLDataModule → fit() pipeline on toy ImageFolder
4. `test_train_py_invalid_config_raises` — nonexistent config raises FileNotFoundError

**README.md** (8.3KB, 183 lines) contains all DOC-01 required sections:
- Installation, Quickstart (5 commands), CIFAR-10 ImageFolder prep snippet
- Config system explanation with TrainConfig field table
- Method table for all 14 v1 methods with era/venue/year/primary contribution
- Evaluation section with all 6 eval tool CLI patterns
- Tutorial reference link, Project Layout, Citation

## Verification Results

```
pytest tests/test_train_script.py: 4 passed
python train.py --help: exits 0, --config/--data-dir/--ckpt-path present
README.md section checks: all 7 section headings present
README.md method table: all 14 dispatcher keys, 14 era markers, all 8 venue strings
README.md size: 8343 bytes (>= 5000)
```

Pre-existing failure (not caused by this plan):
- `tests/test_collapse_monitoring.py::test_corr_diag_mean_in_valid_range` — floating point
  precision issue (`1.0000014 <= 1.0`), existed before this plan. Logged as out-of-scope.

## Deviations from Plan

None - plan executed exactly as written. The exact `train.py` skeleton from the plan was
implemented verbatim. The test file structure matches the plan specification. README section
headings match the plan's required headers exactly.

## Known Stubs

None — README references `docs/tutorial.md` which does not yet exist (created in Plan 10-03),
but the reference is intentional and documented as a future plan dependency. The README's
eval CLI patterns are accurate stub commands using `<method>` placeholder — these are
intentional template patterns for user reference, not implementation stubs.

## Threat Flags

No new security-relevant surface introduced. train.py passes user-supplied `--config` path to
`load_config()` which uses `yaml.safe_load` (mitigates T-10-01 per plan threat model).
Path traversal via `--data-dir` is accepted per T-10-02 (solo-developer tutorial repo).

## Self-Check: PASSED

- train.py: FOUND
- tests/test_train_script.py: FOUND
- README.md: FOUND
- commit cfd9663: FOUND
- commit d55f0f5: FOUND
- commit 844d0a7: FOUND
