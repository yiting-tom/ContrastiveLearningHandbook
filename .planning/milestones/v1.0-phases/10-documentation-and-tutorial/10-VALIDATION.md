---
phase: 10
slug: documentation-and-tutorial
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-03
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (existing) |
| **Config file** | `pyproject.toml` (pytest section) |
| **Quick run command** | `pytest tests/ -q --tb=short` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -q --tb=short`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 01 | 1 | DOC-01 | — | N/A | manual | `grep -c '##' README.md` | ❌ W0 | ⬜ pending |
| 10-01-02 | 01 | 1 | DOC-01 | — | N/A | manual | `python train.py --help` | ❌ W0 | ⬜ pending |
| 10-02-01 | 02 | 1 | DOC-02 | — | N/A | unit | `pytest tests/test_docstrings.py -q` | ❌ W0 | ⬜ pending |
| 10-03-01 | 03 | 2 | DOC-03 | — | N/A | manual | `grep -c 'BaseSSLModule' docs/tutorial.md` | ❌ W0 | ⬜ pending |
| 10-04-01 | 04 | 2 | DOC-03 | — | N/A | manual | `grep -c 'train.py' docs/tutorial.md` | ❌ W0 | ⬜ pending |
| 10-05-01 | 05 | 2 | DOC-03 | — | N/A | manual | `grep -c 'comparison' docs/tutorial.md` | ❌ W0 | ⬜ pending |
| 10-06-01 | 06 | 3 | DOC-01,DOC-03 | — | N/A | manual | `python -c "import nbformat; nbformat.read('notebooks/walkthrough.ipynb','rb')"` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_docstrings.py` — programmatic DOC-02 docstring audit for all LightningModule subclasses
- [ ] `train.py` — entrypoint required by quickstart; created in Plan 10-01

*Existing pytest infrastructure covers test execution; Wave 0 adds doc-specific test stubs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| README quickstart runs in ≤5 commands | DOC-01 | Requires live training environment | Follow README step-by-step on a clean env |
| Tutorial sections readable and accurate | DOC-03 | Subjective prose quality | Human review of docs/tutorial.md |
| Method table complete (14 methods) | DOC-01 | Visual table completeness | `grep -c '|' README.md` + manual count |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
