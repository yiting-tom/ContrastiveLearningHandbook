---
phase: 01-foundation
plan: "01"
subsystem: infra
tags: [pydantic, yaml, pytest, torch, lightning, timm, config]

# Dependency graph
requires: []
provides:
  - requirements.txt with pinned dependencies (torch==2.11.0, lightning==2.6.1, timm==1.0.26, pydantic==2.12.5, etc.)
  - pyproject.toml with pytest configuration
  - core/__init__.py, methods/__init__.py, tests/__init__.py package inits
  - tests/conftest.py with random_tensor, tmp_imagefolder, toy_config_dict fixtures
  - core/config.py: TrainConfig, EvalConfig, 8 method sub-configs, 6 eval sub-configs, load_config helper
  - configs/example.yaml: SimCLR v1 example configuration

affects: [all subsequent plans that import from core/ or run tests]

# Tech tracking
tech-stack:
  added: [pydantic==2.12.5, pyyaml==6.0.3, pytest==8.4.1, torch==2.11.0, torchvision==0.26.0, lightning==2.6.1, timm==1.0.26]
  patterns:
    - "_StrictBase(BaseModel) with extra='forbid' — all config classes inherit this"
    - "TDD — write failing tests (RED), then implement (GREEN)"
    - "load_config helper validates YAML via TrainConfig.model_validate()"

key-files:
  created:
    - requirements.txt
    - pyproject.toml
    - core/__init__.py
    - core/config.py
    - methods/__init__.py
    - configs/example.yaml
    - tests/__init__.py
    - tests/conftest.py
    - tests/test_config.py
  modified: []

key-decisions:
  - "extra='forbid' on all Pydantic sub-configs (D-08) — unknown YAML keys raise ValidationError immediately, catches tutorial copy-paste typos"
  - "All per-method sub-configs (SimCLR, MoCo, BYOL, SwAV, BarlowTwins, SimSiam, DINO, SupCon) are Optional and default to None"
  - "load_config uses yaml.safe_load + TrainConfig.model_validate — no Hydra/OmegaConf"

patterns-established:
  - "Pattern 1: _StrictBase — inherit from _StrictBase to get extra='forbid' automatically, never use BaseModel directly"
  - "Pattern 2: TDD cycle — test file committed RED, then implementation committed GREEN"
  - "Pattern 3: Shared pytest fixtures in conftest.py — random_tensor, tmp_imagefolder, toy_config_dict reused across all test files"

requirements-completed: [FOUND-02, FOUND-08]

# Metrics
duration: 15min
completed: 2026-03-31
---

# Phase 01 Plan 01: Project Scaffold and Pydantic Config Schema Summary

**Pydantic v2 TrainConfig/EvalConfig with extra='forbid' propagation, 8 method sub-configs, 6 eval sub-schemas, and load_config YAML helper — project scaffold with pinned deps and shared pytest fixtures**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-31T15:42:41Z
- **Completed:** 2026-03-31T15:57:00Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments

- Project scaffold in place: requirements.txt with pinned versions, pyproject.toml with pytest config, all package __init__.py files
- Pydantic v2 config schema: _StrictBase enforces extra='forbid' on all levels; TrainConfig with 8 optional method sub-configs; EvalConfig with 6 eval sub-schemas
- load_config() reads YAML via yaml.safe_load and validates via TrainConfig.model_validate()
- Shared pytest fixtures: random_tensor, tmp_imagefolder (3 classes x 5 32x32 RGB JPEGs), toy_config_dict
- All 7 config tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Project scaffold** - `4031ba8` (chore)
2. **Task 2 RED: Failing tests for config schema** - `27aefe6` (test)
3. **Task 2 GREEN: Implement core/config.py and configs/example.yaml** - `e21a5a0` (feat)

_Note: Task 2 used TDD — test commit (RED) followed by implementation commit (GREEN)_

## Files Created/Modified

- `requirements.txt` — Pinned dependencies: torch==2.11.0, torchvision==0.26.0, lightning==2.6.1, timm==1.0.26, pydantic==2.12.5, pyyaml==6.0.3, pytest==8.4.1
- `pyproject.toml` — pytest config with testpaths=["tests"], addopts="-x -q"
- `core/__init__.py` — Package init (empty, re-exports added as modules are created)
- `core/config.py` — TrainConfig, EvalConfig, 8 method sub-configs, 6 eval sub-configs, load_config
- `methods/__init__.py` — Package init
- `configs/example.yaml` — SimCLR v1 example with knn eval block
- `tests/__init__.py` — Package init
- `tests/conftest.py` — Shared fixtures: random_tensor, tmp_imagefolder, toy_config_dict
- `tests/test_config.py` — 7 tests covering all config validation behaviors

## Decisions Made

- Used `_StrictBase(BaseModel)` pattern so ALL sub-configs inherit `extra='forbid'` automatically — consistent with D-08, zero risk of forgetting on any new sub-config class
- Per-method sub-configs are all `Optional[MethodConfig] = None` so the base YAML doesn't need to include method-specific blocks; they're only validated when present
- `load_config` uses `open(path)` context manager rather than `yaml.safe_load(open(path))` (no context manager in the one-liner) for proper file handle cleanup

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All subsequent plans can now `from core.config import TrainConfig, load_config`
- Shared test fixtures available in conftest.py for all future test files
- Pattern established: new method sub-configs inherit _StrictBase and are added as Optional fields to TrainConfig

---
*Phase: 01-foundation*
*Completed: 2026-03-31*
