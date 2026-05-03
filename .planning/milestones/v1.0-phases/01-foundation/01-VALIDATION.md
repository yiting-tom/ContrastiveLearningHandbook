---
phase: 1
slug: foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-31
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.1 |
| **Config file** | none — Wave 0 installs |
| **Quick run command** | `pytest tests/ -x -q` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x -q`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | FOUND-01, FOUND-09 | integration | `pytest tests/test_base.py -x` | ❌ W0 | ⬜ pending |
| 1-02-01 | 02 | 1 | FOUND-02, FOUND-08 | unit | `pytest tests/test_config.py -x` | ❌ W0 | ⬜ pending |
| 1-03-01 | 03 | 1 | FOUND-03 | unit | `pytest tests/test_backbone.py -x` | ❌ W0 | ⬜ pending |
| 1-04-01 | 04 | 1 | FOUND-04 | unit | `pytest tests/test_projection.py -x` | ❌ W0 | ⬜ pending |
| 1-05-01 | 05 | 2 | FOUND-05, FOUND-06 | unit/integration | `pytest tests/test_data.py -x` | ❌ W0 | ⬜ pending |
| 1-06-01 | 06 | 2 | FOUND-07, FOUND-10 | unit | `pytest tests/test_ema.py tests/test_dispatcher.py -x` | ❌ W0 | ⬜ pending |
| 1-07-01 | 07 | 2 | INFRA-01, INFRA-06 | unit | `pytest tests/test_losses.py tests/test_optimizers.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/__init__.py` — package init
- [ ] `tests/conftest.py` — shared fixtures (synthetic ImageFolder, random tensor helpers, toy config dict)
- [ ] `pytest.ini` or `pyproject.toml [tool.pytest.ini_options]` — pytest config
- [ ] `requirements.txt` — pinned dependency versions
- [ ] `tests/test_base.py` — stubs for FOUND-01, FOUND-09
- [ ] `tests/test_config.py` — stubs for FOUND-02, FOUND-08
- [ ] `tests/test_backbone.py` — stubs for FOUND-03
- [ ] `tests/test_projection.py` — stubs for FOUND-04
- [ ] `tests/test_data.py` — stubs for FOUND-05, FOUND-06
- [ ] `tests/test_dispatcher.py` — stubs for FOUND-07
- [ ] `tests/test_ema.py` — stubs for FOUND-10
- [ ] `tests/test_losses.py` — stubs for INFRA-01
- [ ] `tests/test_optimizers.py` — stubs for INFRA-06

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| TensorBoard log files written to disk | FOUND-09 | Requires file system inspection | Run training, check `lightning_logs/` for events files |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
