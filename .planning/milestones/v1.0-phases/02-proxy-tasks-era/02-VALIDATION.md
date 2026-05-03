---
phase: 2
slug: proxy-tasks-era
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-01
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | none — standard pytest discovery |
| **Quick run command** | `python -m pytest tests/ -x -q` |
| **Full suite command** | `python -m pytest tests/ -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green (all 70 existing + new Phase 2 tests)
- **Max feedback latency:** ~30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 2-01-01 | 01 | 1 | INFRA-02 | unit | `python -m pytest tests/test_memory_bank.py -x` | ❌ W0 | ⬜ pending |
| 2-01-02 | 01 | 1 | INFRA-02 | unit | `python -m pytest tests/test_memory_bank.py::test_init_l2_normalized -x` | ❌ W0 | ⬜ pending |
| 2-02-01 | 02 | 1 | ERA1-01 | unit | `python -m pytest tests/test_nce_loss.py::test_z_fixed_after_first_call -x` | ❌ W0 | ⬜ pending |
| 2-02-02 | 02 | 1 | ERA1-01 | unit | `python -m pytest tests/test_nce_loss.py::test_z_is_register_buffer -x` | ❌ W0 | ⬜ pending |
| 2-03-01 | 03 | 2 | ERA1-01 | integration | `python -m pytest tests/test_instance_discrimination.py::test_train_5_epochs -x` | ❌ W0 | ⬜ pending |
| 2-03-02 | 03 | 2 | ERA1-01 | unit | `python -m pytest tests/test_instance_discrimination.py::test_dispatcher_registration -x` | ❌ W0 | ⬜ pending |
| 2-04-01 | 04 | 2 | ERA1-02 | integration | `python -m pytest tests/test_invariant_spread.py::test_train_5_epochs -x` | ❌ W0 | ⬜ pending |
| 2-04-02 | 04 | 2 | ERA1-02 | unit | `python -m pytest tests/test_invariant_spread.py::test_dispatcher_registration -x` | ❌ W0 | ⬜ pending |
| 2-05-01 | 05 | 3 | ERA1-01/02 | unit | `python -m pytest tests/test_instance_discrimination.py tests/test_invariant_spread.py -k dispatcher -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_memory_bank.py` — stubs for INFRA-02 (init, get, update, L2-norm, staleness docstring)
- [ ] `tests/test_nce_loss.py` — stubs for ERA1-01 (Z fixed after first call, Z is register_buffer, loss finite)
- [ ] `tests/test_instance_discrimination.py` — stubs for ERA1-01 module (5-epoch train, dispatcher registration)
- [ ] `tests/test_invariant_spread.py` — stubs for ERA1-02 module (5-epoch train, dispatcher registration, loss decreases in first 3 epochs)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| MemoryBank docstring contains "stale" and cross-reference to MoCo | INFRA-02 | String check in docstring | `grep -n "stale\|MoCo" core/memory_bank.py` — must mention staleness limitation and MoCo queue as solution |
| InvariantSpreadModule docstring mentions batch-size sensitivity | ERA1-02 | String check in docstring | `grep -n "batch.size\|batch size\|sensitive" methods/invariant_spread/module.py` |
| Z does not change after first mini-batch in real training run | ERA1-01 | Requires actual training loop | Add logging in InstanceDiscriminationModule to print Z after first step and epoch 2, confirm identical |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
