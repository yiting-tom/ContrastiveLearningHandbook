---
phase: 7
slug: transformer-era
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-10
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.1 |
| **Config file** | `tests/conftest.py` (shared fixtures) |
| **Quick run command** | `python -m pytest tests/test_moco_v3.py tests/test_dino.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_moco_v3.py tests/test_dino.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -x -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | INFRA-05 | — | N/A | unit | `pytest tests/test_predictor_head.py::test_predictor_docstring -x` | ❌ W0 | ⬜ pending |
| 07-02-01 | 02 | 1 | ERA4-01 | — | N/A | unit | `pytest tests/test_moco_v3.py::test_patch_projection_frozen -x` | ❌ W0 | ⬜ pending |
| 07-02-02 | 02 | 1 | ERA4-01 | — | N/A | smoke | `pytest tests/test_moco_v3.py::test_moco_v3_train_3_epochs -x` | ❌ W0 | ⬜ pending |
| 07-02-03 | 02 | 1 | ERA4-01 | — | N/A | unit | `pytest tests/test_moco_v3.py::test_moco_v3_uses_adamw -x` | ❌ W0 | ⬜ pending |
| 07-03-01 | 03 | 1 | ERA4-01 | — | N/A | unit | `pytest tests/test_moco_v3.py::test_patch_projection_frozen -x` | ❌ W0 | ⬜ pending |
| 07-04-01 | 04 | 2 | ERA4-02 | — | N/A | unit | `pytest tests/test_dino.py::test_teacher_global_crops_only -x` | ❌ W0 | ⬜ pending |
| 07-05-01 | 05 | 2 | ERA4-02 | — | N/A | unit | `pytest tests/test_dino.py::test_centering_update_before_loss -x` | ❌ W0 | ⬜ pending |
| 07-05-02 | 05 | 2 | ERA4-02 | — | N/A | smoke | `pytest tests/test_dino.py::test_dino_train_3_epochs -x` | ❌ W0 | ⬜ pending |
| 07-06-01 | 06 | 2 | ERA4-02 | — | N/A | unit | `pytest tests/test_dino.py::test_centering_update_before_loss -x` | ❌ W0 | ⬜ pending |
| 07-07-01 | 07 | 3 | ERA4-03 | — | N/A | smoke | `pytest tests/test_dinov2_demo.py -x` | ❌ W0 | ⬜ pending |
| 07-08-01 | 08 | 3 | ERA4-01,ERA4-02,ERA4-03 | — | N/A | smoke | `python -m pytest tests/ -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_moco_v3.py` — stubs for ERA4-01 (patch freeze, training, AdamW, dispatcher)
- [ ] `tests/test_dino.py` — stubs for ERA4-02 (centering, global crops, training, dispatcher)
- [ ] `tests/test_dinov2_demo.py` — stubs for ERA4-03 (model loading, feature extraction)
- [ ] `tests/test_predictor_head.py` — add docstring assertion for INFRA-05 (existing file needs update)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| DINOv2 model downloads and runs k-NN on real CIFAR-10 | ERA4-03 | Requires internet + download time | Run `python eval/dinov2_demo.py --dataset cifar10` and confirm accuracy number printed |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
