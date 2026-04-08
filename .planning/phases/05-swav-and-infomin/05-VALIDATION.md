---
phase: 5
slug: swav-and-infomin
status: ready
nyquist_compliant: true
wave_0_complete: false
created: 2026-04-08
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml (pytest section) |
| **Quick run command** | `python -m pytest tests/test_swav.py tests/test_infomin.py tests/test_multi_crop.py tests/test_swav_prototype.py -x -v` |
| **Full suite command** | `python -m pytest tests/ -x -v` |
| **Estimated runtime** | ~45 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_swav.py tests/test_infomin.py tests/test_multi_crop.py tests/test_swav_prototype.py -x -v`
- **After every plan wave:** Run `python -m pytest tests/ -x -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 45 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | INFRA-04 | T-05-01 | N/A | unit | `python -m pytest tests/test_multi_crop.py -x -v` | ❌ W0 | ⬜ pending |
| 05-02-01 | 02 | 1 | ERA2-05 | T-05-02 | N/A | unit | `python -m pytest tests/test_swav.py -x -v -k "sinkhorn or swav_loss"` | ❌ W0 | ⬜ pending |
| 05-03-01 | 03 | 1 | ERA2-05 | T-05-03 | N/A | unit | `python -m pytest tests/test_swav_prototype.py -x -v` | ❌ W0 | ⬜ pending |
| 05-04-01 | 04 | 1 | ERA2-05 | T-05-04 | Pydantic extra=forbid validates | unit | `python -m pytest tests/test_config.py -x -v` | ✅ | ⬜ pending |
| 05-05-01 | 05 | 2 | ERA2-05 | T-05-05 | N/A | integration | `python -c "from methods.swav.module import SwAVModule; print('OK')"` | ❌ W0 | ⬜ pending |
| 05-05-02 | 05 | 2 | ERA2-05, INFRA-04 | T-05-06 | N/A | integration | `python -m pytest tests/test_swav.py -x -v` | ❌ W0 | ⬜ pending |
| 05-06-01 | 06 | 1 | ERA2-06 | T-05-07 | N/A | unit | `python -m pytest tests/test_infomin.py -x -v` | ❌ W0 | ⬜ pending |
| 05-06-02 | 06 | 1 | ERA2-06 | — | N/A | syntax | `python -c "import ast; ast.parse(open('tools/compare_augmentations.py').read())"` | ❌ W0 | ⬜ pending |
| 05-07-01 | 07 | 3 | ERA2-05, ERA2-06 | T-05-08 | Pydantic validates YAML | integration | `python -c "from core.config import load_config; load_config('configs/swav_resnet18.yaml'); load_config('configs/infomin_resnet18.yaml'); print('OK')"` | ❌ W0 | ⬜ pending |
| 05-07-02 | 07 | 3 | ERA2-05, ERA2-06, INFRA-04 | — | N/A | smoke | `python -m pytest tests/test_swav.py tests/test_infomin.py -x -v -k "smoke or yaml_config"` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_multi_crop.py` — stubs for INFRA-04 (MultiCropDataset tests)
- [ ] `tests/test_swav.py` — stubs for ERA2-05 (Sinkhorn-Knopp, SwAV module tests)
- [ ] `tests/test_swav_prototype.py` — stubs for ERA2-05 (prototype layer tests)
- [ ] `tests/test_infomin.py` — stubs for ERA2-06 (InfoMin module tests)
- [ ] `tests/conftest.py` — existing fixtures sufficient (large_imagefolder, clean_registry patterns already established)

*Note: All TDD tasks in Plans 01-04, 06 create their test files as part of the task. Wave 0 is implicitly handled by each TDD task writing tests first (RED phase).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Augmentation comparison visual quality | ERA2-06 | Visual inspection of augmentation differences | Run `python tools/compare_augmentations.py --image <image>` and verify SimCLR vs InfoMin grid shows visible difference in color distortion and blur |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 45s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
