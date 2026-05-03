# Roadmap: Contrastive Learning Tutorial Repo

## Milestones

- ✅ **v1.0 MVP** — Phases 1–10, 10.1 (shipped 2026-05-03)
- **v1.1 Debt Payoff** — Phases 11–12 (active)

## Phases

<details>
<summary>✅ v1.0 MVP (Phases 1–10, 10.1) — SHIPPED 2026-05-03</summary>

- [x] Phase 1: Foundation (7/7 plans) — completed 2026-03-31
- [x] Phase 2: Proxy Tasks Era (5/5 plans) — completed 2026-04-02
- [x] Phase 3: SimCLR (3/3 plans) — completed
- [x] Phase 4: MoCo (3/3 plans) — completed
- [x] Phase 5: SwAV and InfoMin (7/7 plans) — completed
- [x] Phase 6: No-Negative Methods (7/7 plans) — completed
- [x] Phase 7: Transformer Era (4/4 plans) — completed
- [x] Phase 8: Supervised Contrastive (5/5 plans) — completed
- [x] Phase 9: Evaluation Suite (5/5 plans) — completed
- [x] Phase 10: Documentation and Tutorial (6/6 plans) — completed 2026-05-03
- [x] Phase 10.1: Fix train.py + eval script integration bugs (5/5 plans, INSERTED) — completed 2026-05-03

Full phase details: `.planning/milestones/v1.0-ROADMAP.md`

</details>

### v1.1 Debt Payoff

- [ ] **Phase 11: Code Fix & Export Cleanup** — Patch train.py wiring for 3 niche methods, add 4 missing exports, remove dead code
- [ ] **Phase 12: Integration Test Suite** — Write and verify @pytest.mark.slow tests for BYOL, Barlow Twins, and README Quickstart

## Phase Details

### Phase 11: Code Fix & Export Cleanup
**Goal**: All known v1.0 code gaps are closed — train.py routes every method correctly, all public symbols are importable, and dead code is removed
**Depends on**: Nothing (debt-payoff edits are independent of each other)
**Requirements**: WIRE-01, WIRE-02, WIRE-03, EXPORT-01, CLEAN-01
**Success Criteria** (what must be TRUE):
  1. Running `python train.py --method instance_discrimination` with a real dataset does not raise a missing-collate or missing-IndexedDataset error
  2. Running `python train.py --method swav` (or dino) instantiates MultiCropDataset with 2×224 + 6×96 crops, not 8 uniform crops
  3. Running `python train.py --method supcon --stage 2 --stage1_ckpt <path>` calls `from_stage1_ckpt()` and loads the backbone from the checkpoint rather than training a random backbone
  4. `from core import PredictorHead, SupConLoss, MultiCropDataset, method_dispatcher` executes without ImportError
  5. `config.py` contains exactly one `InfoMinConfig` definition (lines 72–83 duplicate removed, no remaining dead block)
**Plans**: TBD

### Phase 12: Integration Test Suite
**Goal**: Automated slow tests assert that BYOL and Barlow Twins produce non-collapsed embeddings after real training, and that README Quickstart commands run without error
**Depends on**: Phase 11 (tests exercise corrected train.py wiring paths)
**Requirements**: TEST-01, TEST-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. `pytest -m slow tests/test_integration.py::test_byol_embedding_std` passes — asserts `embedding_std > 0.1` after 5 epochs on real CIFAR-10
  2. `pytest -m slow tests/test_integration.py::test_barlow_twins_corr_diag_mean` passes — asserts `corr_diag_mean > 0.5` after 5 epochs on real CIFAR-10
  3. `pytest -m slow tests/test_integration.py::test_readme_quickstart` passes — runs all README Quickstart CLI commands for 1 epoch on CPU without error or exception
  4. All three slow tests are marked `@pytest.mark.slow` and are excluded from the default `pytest` run (only execute when `-m slow` is passed)
**Plans**: TBD

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Foundation | v1.0 | 7/7 | Complete | 2026-03-31 |
| 2. Proxy Tasks Era | v1.0 | 5/5 | Complete | 2026-04-02 |
| 3. SimCLR | v1.0 | 3/3 | Complete | — |
| 4. MoCo | v1.0 | 3/3 | Complete | — |
| 5. SwAV and InfoMin | v1.0 | 7/7 | Complete | — |
| 6. No-Negative Methods | v1.0 | 7/7 | Complete | — |
| 7. Transformer Era | v1.0 | 4/4 | Complete | — |
| 8. Supervised Contrastive | v1.0 | 5/5 | Complete | — |
| 9. Evaluation Suite | v1.0 | 5/5 | Complete | — |
| 10. Documentation and Tutorial | v1.0 | 6/6 | Complete | 2026-05-03 |
| 10.1. Fix train.py + eval bugs (INSERTED) | v1.0 | 5/5 | Complete | 2026-05-03 |
| 11. Code Fix & Export Cleanup | v1.1 | 0/? | Not started | — |
| 12. Integration Test Suite | v1.1 | 0/? | Not started | — |
