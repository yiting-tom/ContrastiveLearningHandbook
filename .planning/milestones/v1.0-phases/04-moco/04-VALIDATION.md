---
phase: 4
slug: moco
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-05
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (via pyproject.toml) |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] |
| **Quick run command** | `python -m pytest tests/test_queue.py tests/test_moco.py -x -q` |
| **Full suite command** | `python -m pytest -x -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_queue.py tests/test_moco.py -x -q`
- **After every plan wave:** Run `python -m pytest -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | INFRA-03 | unit | `python -m pytest tests/test_queue.py -x -q` | ❌ W0 | ⬜ pending |
| 04-01-02 | 01 | 1 | INFRA-03 | unit | `python -m pytest tests/test_queue.py::test_queue_size_invariant -x` | ❌ W0 | ⬜ pending |
| 04-02-01 | 02 | 1 | ERA2-01 | unit | `python -m pytest tests/test_moco.py::test_ema_params_not_in_optimizer -x` | ❌ W0 | ⬜ pending |
| 04-03-01 | 03 | 1 | ERA2-01 | unit | `python -m pytest tests/test_moco.py::test_moco_v1_train_5_epochs -x` | ❌ W0 | ⬜ pending |
| 04-04-01 | 04 | 2 | ERA2-01 | unit (mock) | `python -m pytest tests/test_moco.py::test_ema_update_call_order -x` | ❌ W0 | ⬜ pending |
| 04-04-02 | 04 | 2 | ERA2-01 | integration | `python -m pytest tests/test_moco.py::test_moco_v1_train_5_epochs -x` | ❌ W0 | ⬜ pending |
| 04-05-01 | 05 | 2 | ERA2-02 | unit | `python -m pytest tests/test_moco.py::test_moco_v2_has_mlp_head -x` | ❌ W0 | ⬜ pending |
| 04-05-02 | 05 | 2 | ERA2-02 | integration | `python -m pytest tests/test_moco.py::test_moco_v2_train_5_epochs -x` | ❌ W0 | ⬜ pending |
| 04-06-01 | 06 | 2 | ERA2-01/02 | unit | `python -m pytest tests/test_moco.py::test_dispatcher_registration -x` | ❌ W0 | ⬜ pending |
| 04-06-02 | 06 | 2 | DOC-02 | unit | `python -m pytest tests/test_moco.py::test_docstring -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_queue.py` — stubs for INFRA-03 (FIFO, wrap-around, size invariant, get_negatives detached)
- [ ] `tests/test_moco.py` — stubs for ERA2-01, ERA2-02 (EMA exclusion, training smoke, v2 MLP head, dispatcher, docstrings)

*Existing infrastructure covers framework and fixtures — only new test files needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Shuffled-BN limitation documented | ERA2-01 | Docstring review — not automatable | Read `MoCoV1Module` class docstring and verify single-GPU caveat is documented |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
