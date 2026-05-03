# Retrospective — Contrastive Learning Tutorial Repo

---

## Milestone: v1.0 MVP

**Shipped:** 2026-05-03
**Phases:** 11 | **Plans:** 57 | **Timeline:** 34 days (2026-03-30 → 2026-05-03)
**Commits:** 274 | **LOC:** ~112,700 Python

### What Was Built

- Foundation infrastructure: BaseSSLModule, timm backbone factory, ProjectionHead, InfoNCELoss, LARS, ContrastiveAugmentation, SSLDataModule, EMAUpdater, method dispatcher
- Era 1 methods: Instance Discrimination (memory bank + fixed-Z NCE), Invariant Spread
- Era 2 methods: SimCLR v1/v2, MoCo v1/v2, SwAV (Sinkhorn-Knopp + multi-crop), InfoMin augmentation demo
- No-negative methods: BYOL, SimSiam, Barlow Twins — with collapse monitoring wired for all three
- Transformer era: MoCo v3 (ViT patch freeze), DINO (student-teacher + centering), DINOv2 feature extraction
- Supervised: SupCon with ClassBalancedSampler and two-stage pretraining
- Evaluation suite: KNNCallback, linear probe, t-SNE, UMAP, fine-tuning, CAM visualization
- Documentation: train.py CLI, README with 14-method table, DOC-02 paper-accurate gotcha docstrings, docs/tutorial.md era narrative
- Phase 10.1: Closed critical eval-script integration bugs; 13/13 e2e tests GREEN

### What Worked

- **Wave-based parallel plan execution** — plans with zero file overlap ran concurrently, reducing per-phase wall time significantly (e.g., Phase 5's 7 plans completed faster than sequential execution would allow)
- **Shared BaseSSLModule pattern** — after Phase 1 established the base, every subsequent method phase was scoped narrowly: implement the module, wire the dispatcher, write YAML configs and tests. No phase needed to re-invent the training loop.
- **Registry dispatcher with `register_method()`** — each phase called one function to register; no dispatcher internals needed to change. This kept Phase 1 as a clean foundation without forward-planning for specific methods.
- **Incremental test suite** — earlier-phase tests kept running in later phases; regressions caught immediately without re-running full suite manually.
- **Phase 10.1 insertion** — running a milestone audit before closing found integration bugs that would have been embarrassing in a published tutorial. The insert-and-fix pattern preserved the v1.0 tag while closing critical gaps.
- **LARS from scratch** — ~60 lines, tutorial-readable, no lightly/torchlars dep. Reduced dependency surface and serves as supplementary learning material.

### What Was Inefficient

- **ROADMAP.md progress table drift** — the bottom progress table was not updated during execution (showed 0/7 for phases that were complete). Had to rely on SUMMARY.md existence and audit cross-reference to reconstruct true state. A lighter "update this row on phase completion" protocol would prevent staleness.
- **REQUIREMENTS.md checkbox drift** — 20 requirements stayed unchecked through phases 5–10 because execution didn't loop back to update the source file. The milestone audit found this and flagged it as a tracking artifact. Updating checkboxes at phase completion would keep REQUIREMENTS.md as a live document.
- **Late integration testing** — blockers BLOCKER-1/4/5 (InstanceDiscrimination wiring, MultiCropDataset, SupCon stage-2) were only discovered at milestone audit time. If a single "train.py → each method" smoke test had been part of each method phase's acceptance criteria, these would have been caught earlier.
- **Human-verification items** — 3 items (BYOL embedding_std, Barlow Twins corr_diag_mean, README GPU quickstart) required real GPU runs that weren't feasible in CI. These sat as unverified items at v1.0 close. Should be flagged as "requires human sign-off" explicitly in each phase's success criteria.

### Patterns Established

- **`build_projector()` override pattern** — SimCLRv2 overrides only `build_projector()` on SimCLRv1; MoCoV2 overrides only `build_projector()` on MoCoV1. Minimal variant pattern avoids code duplication and makes diff between v1 and v2 immediately readable.
- **`SupConLoss(labels=None)` graceful degradation** — loss degenerates to SimCLR NT-Xent when labels are absent. Verified with a unit test asserting equality. Pattern worth reusing for any loss that generalizes between supervised and self-supervised settings.
- **EMA in `on_train_batch_end`** — consistent placement across MoCo v1/v2/v3, BYOL, DINO. Never in `training_step` or `on_before_optimizer_step`. Documented as a cross-cutting constraint in REQUIREMENTS.md.
- **`extra='forbid'` on all Pydantic sub-configs** — catches tutorial copy-paste YAML typos at config load time, before any model is instantiated. Applied via `_StrictBase` pattern.
- **Milestone audit before close** — `/gsd-audit-milestone` run before completing v1.0 surfaced 5 blockers. Resulted in Phase 10.1 insertion. Made closing the milestone higher-confidence. Establish as standard practice.

### Key Lessons

1. **Update ROADMAP.md progress table at each phase completion** — don't let it drift for 10 phases. One-row update costs 30 seconds; auditing 10 rows of drift costs 30 minutes.
2. **Add a `train.py → <method>` smoke integration test to each method phase** — this would have caught BLOCKER-1 (InstanceDiscrimination), BLOCKER-4 (SwAV/DINO multi-crop), and BLOCKER-5 (SupCon stage-2) in their respective phases rather than at milestone close.
3. **Mark human-verification items explicitly in success criteria** — "requires GPU run" or "requires human reading" should be a first-class gate in the phase plan, not an open item discovered at audit time.
4. **`core/__init__.py __all__` hygiene** — add new public symbols to `__all__` at implementation time, not as a cleanup task. `from core import X` silently failing is a bad tutorial user experience.

### Cost Observations

- Model mix: primarily Claude Sonnet (fast plan execution), Claude Opus (complex phases like Phase 6 no-negative methods and Phase 10 tutorial assembly)
- Sessions: multiple across 34 days with context handoffs via STATE.md
- Notable: Wave-based parallel execution reduced per-phase wall time by ~30–40% on phases with independent plans (Phase 5, Phase 6, Phase 10)

---

## Cross-Milestone Trends

| Metric | v1.0 |
|--------|------|
| Days | 34 |
| Phases | 11 |
| Plans | 57 |
| LOC | ~112,700 |
| Commits | 274 |
| Requirements satisfied | 34/40 (85%) |
| E2E tests GREEN | 13/13 |
| Blockers deferred | 5 |
