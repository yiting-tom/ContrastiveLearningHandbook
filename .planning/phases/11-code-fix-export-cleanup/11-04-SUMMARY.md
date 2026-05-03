---
phase: "11"
plan: "11-04"
subsystem: core
tags: [exports, public-api, cleanup, verification]
dependency_graph:
  requires: []
  provides: [core.__all__ with 4 new symbols, EXPORT-01, CLEAN-01]
  affects: [core/__init__.py]
tech_stack:
  added: []
  patterns: [try/except ImportError re-export pattern]
key_files:
  created: []
  modified:
    - core/__init__.py
decisions:
  - "D-08: Follow existing try/except ImportError pattern for all new re-exports"
  - "D-09: CLEAN-01 is verify-only — single InfoMinConfig definition at line 137 confirmed; no duplicate to remove"
metrics:
  duration: "~5 minutes"
  completed: "2026-05-04"
  tasks_completed: 2
  tasks_total: 2
---

# Phase 11 Plan 04: Code Fix & Export Cleanup (EXPORT-01 + CLEAN-01) Summary

Added 4 missing public exports to core/__init__.py following the existing try/except ImportError pattern, and verified that core/config.py contains exactly one InfoMinConfig definition.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add 4 missing exports to core/__init__.py | 268fd3e | core/__init__.py |
| 2 | Verify CLEAN-01 — single InfoMinConfig in config.py | (verify-only) | core/config.py (no change) |

## What Was Done

### Task 1: EXPORT-01 — Add 4 missing exports

Modified `core/__init__.py` to:

1. Extended `from core.data import ...` block to include `MultiCropDataset`
2. Extended `from core.losses import ...` block to include `SupConLoss`
3. Extended `from core.projection import ...` block to include `PredictorHead`
4. Added new `try/except` block for `method_dispatcher` from `core.dispatcher`
5. Added all four names to `__all__`: `MultiCropDataset`, `SupConLoss`, `PredictorHead`, `method_dispatcher`

All additions follow the pre-existing `try/except ImportError: pass` pattern for parallel plan safety.

### Task 2: CLEAN-01 — Verify single InfoMinConfig

Ran `grep -n "class InfoMinConfig" core/config.py` — result:
```
137:class InfoMinConfig(_StrictBase):
```

Exactly one definition found. No code change required.

**Note:** REQUIREMENTS.md referenced "dead code at lines 72-83" as a duplicate InfoMinConfig. Lines 72-83 actually contain `BarlowTwinsConfig`, not a duplicate InfoMinConfig. The single InfoMinConfig at line 137 is the only and correct definition. CLEAN-01 requirement is satisfied without any code modification.

## Verification Results

```
EXPORT-01: all 4 imports ok
__all__ ok
1       (InfoMinConfig count — correct)
existing exports ok
```

- `from core import PredictorHead, SupConLoss, MultiCropDataset, method_dispatcher` — exit 0
- All 4 new names present in `core.__all__`
- `grep -c "class InfoMinConfig" core/config.py` returns 1
- Existing exports (ProjectionHead, InfoNCELoss, SSLDataModule, BaseSSLModule) still importable

## Deviations from Plan

None — plan executed exactly as written. Task 2 was verify-only as expected; no duplicate InfoMinConfig was found, confirming D-09.

## Known Stubs

None.

## Threat Flags

No new security-relevant surface introduced. The `__all__` expansion is intentionally public API; no sensitive internals were added.

## Self-Check: PASSED

- core/__init__.py: modified and committed at 268fd3e
- core/config.py: unchanged (verify-only task, no modification needed)
- All 4 required imports work without error
- Existing exports regression check passes
