---
phase: 9
slug: evaluation-suite
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-10
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.1 |
| **Config file** | none — uses pytest defaults |
| **Quick run command** | `pytest tests/test_eval_knn.py tests/test_eval_linear_probe.py tests/test_eval_tsne.py tests/test_eval_umap.py tests/test_eval_finetune.py tests/test_eval_cam.py -x` |
| **Full suite command** | `pytest tests/ -x` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_eval_*.py -x --timeout=60`
- **After every plan wave:** Run `pytest tests/ -x`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 09-01-W0 | 01 | 0 | EVAL-01 | — | N/A | stub | `pytest tests/test_eval_knn.py -x` | ❌ W0 | ⬜ pending |
| 09-02-W0 | 02 | 0 | EVAL-02 | — | N/A | stub | `pytest tests/test_eval_linear_probe.py -x` | ❌ W0 | ⬜ pending |
| 09-03-W0 | 03 | 0 | EVAL-03 | — | N/A | stub | `pytest tests/test_eval_tsne.py -x` | ❌ W0 | ⬜ pending |
| 09-04-W0 | 04 | 0 | EVAL-04 | — | N/A | stub | `pytest tests/test_eval_umap.py -x` | ❌ W0 | ⬜ pending |
| 09-05-W0 | 05 | 0 | EVAL-05 | — | N/A | stub | `pytest tests/test_eval_finetune.py -x` | ❌ W0 | ⬜ pending |
| 09-06-W0 | 06 | 0 | EVAL-06 | — | N/A | stub | `pytest tests/test_eval_cam.py -x` | ❌ W0 | ⬜ pending |
| 09-07-W0 | 07 | 0 | FOUND-08 | — | N/A | stub | `pytest tests/test_eval_integration.py -x` | ❌ W0 | ⬜ pending |
| 09-01-01 | 01 | 1 | EVAL-01 | — | N/A | unit | `pytest tests/test_eval_knn.py -x` | ✅ W0 | ⬜ pending |
| 09-02-01 | 02 | 1 | EVAL-02 | — | N/A | unit | `pytest tests/test_eval_linear_probe.py -x` | ✅ W0 | ⬜ pending |
| 09-03-01 | 03 | 1 | EVAL-03 | — | N/A | unit | `pytest tests/test_eval_tsne.py -x` | ✅ W0 | ⬜ pending |
| 09-04-01 | 04 | 1 | EVAL-04 | — | N/A | unit | `pytest tests/test_eval_umap.py -x` | ✅ W0 | ⬜ pending |
| 09-05-01 | 05 | 2 | EVAL-05 | — | N/A | unit | `pytest tests/test_eval_finetune.py -x` | ✅ W0 | ⬜ pending |
| 09-06-01 | 06 | 2 | EVAL-06 | — | N/A | unit | `pytest tests/test_eval_cam.py -x` | ✅ W0 | ⬜ pending |
| 09-07-01 | 07 | 3 | FOUND-08 | — | N/A | integration | `pytest tests/test_eval_integration.py -x` | ✅ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_eval_knn.py` — stubs for EVAL-01
- [ ] `tests/test_eval_linear_probe.py` — stubs for EVAL-02
- [ ] `tests/test_eval_tsne.py` — stubs for EVAL-03
- [ ] `tests/test_eval_umap.py` — stubs for EVAL-04
- [ ] `tests/test_eval_finetune.py` — stubs for EVAL-05
- [ ] `tests/test_eval_cam.py` — stubs for EVAL-06
- [ ] `tests/test_eval_integration.py` — stubs for integration test (FOUND-08 / D-05)
- [ ] Install new dependencies: `pip install faiss-cpu umap-learn grad-cam` (add to `requirements.txt`)

*All test files are missing — Wave 0 must create them.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CAM overlay images are visually meaningful | EVAL-06 | Requires human visual inspection | Run `python eval/cam_vis.py configs/simclr.yaml --ckpt <ckpt>`, inspect saved overlay PNGs |
| t-SNE cluster separation is visible | EVAL-03 | Requires human visual inspection | Run `python eval/tsne_vis.py configs/simclr.yaml --ckpt <ckpt>`, inspect 3 PNG outputs |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
