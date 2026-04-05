---
phase: 3
slug: simclr
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-05
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (configured in pyproject.toml) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `python -m pytest tests/test_simclr.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_simclr.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 3-01-01 | 03-01 | 1 | ERA2-03.3 | unit | `pytest tests/test_simclr.py::test_ntxent_symmetry -x` | ❌ W0 | ⬜ pending |
| 3-01-02 | 03-01 | 1 | ERA2-03.4 | unit | `pytest tests/test_simclr.py::test_identical_views_minimum -x` | ❌ W0 | ⬜ pending |
| 3-02-01 | 03-02 | 1 | ERA2-03.1 | integration | `pytest tests/test_simclr.py::test_simclr_v1_train_5_epochs -x` | ❌ W0 | ⬜ pending |
| 3-02-02 | 03-02 | 1 | SC-3 | integration | `pytest tests/test_simclr.py::test_dispatcher_registration -x` | ❌ W0 | ⬜ pending |
| 3-03-01 | 03-03 | 2 | ERA2-04.1 | unit | `pytest tests/test_simclr.py::test_simclr_v2_3layer_head -x` | ❌ W0 | ⬜ pending |
| 3-03-02 | 03-03 | 2 | ERA2-04.2 | unit | `pytest tests/test_simclr.py::test_v2_only_changes_projector -x` | ❌ W0 | ⬜ pending |
| 3-04-01 | 03-04 | 2 | ERA2-03.2 | unit | `pytest tests/test_simclr.py::test_strong_augmentation_s1 -x` | ❌ W0 | ⬜ pending |
| 3-05-01 | 03-05 | 2 | SC-4 | integration | `pytest tests/test_simclr.py::test_yaml_config_loads -x` | ❌ W0 | ⬜ pending |
| 3-05-02 | 03-05 | 2 | ERA2-03.5 | unit | `pytest tests/test_simclr.py::test_lars_optimizer_activates -x` | ❌ W0 | ⬜ pending |
| 3-05-03 | 03-05 | 2 | SC-5 | unit | `pytest tests/test_simclr.py::test_default_optimizer_is_adamw -x` | ❌ W0 | ⬜ pending |
| 3-06-01 | 03-06 | 3 | ERA2-03, ERA2-04 | integration | `python -m pytest tests/ -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_simclr.py` — stubs for all ERA2-03 and ERA2-04 test cases listed above
- Reuse `LossTracker` callback pattern from `tests/test_invariant_spread.py` for training tests
- Reuse `clean_registry` fixture from `tests/test_invariant_spread.py`
- Reuse existing `tmp_imagefolder`, `random_tensor`, `toy_config_dict` fixtures from `tests/conftest.py`

*No new framework install needed — pytest already configured.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Augmentation grid visually shows strong color jitter and Gaussian blur | ERA2-03 SC-1 | Visual inspection required | Run `python tools/visualize_augmentations.py`; inspect `tools/output/augmentation_grid.png` for vivid color distortion and blur |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
