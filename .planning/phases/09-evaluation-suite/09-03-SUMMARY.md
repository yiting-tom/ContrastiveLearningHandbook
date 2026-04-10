---
phase: 09-evaluation-suite
plan: 03
subsystem: evaluation
tags: [tsne, umap, sklearn, umap-learn, matplotlib, visualization, ssl, representations]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: core/config.py TSNEConfig and UMAPConfig schemas
  - phase: 09-evaluation-suite
    provides: eval/ directory structure and EvalConfig schema

provides:
  - eval/tsne_vis.py -- t-SNE visualization with 3-perplexity sweep, PCA pre-reduction
  - eval/umap_vis.py -- UMAP visualization with cosine metric, reducer reuse
  - tests/test_eval_tsne.py -- 7 unit tests for t-SNE script
  - tests/test_eval_umap.py -- 8 unit tests for UMAP script

affects: [09-evaluation-suite, 09-07-integration-test]

# Tech tracking
tech-stack:
  added: [umap-learn==0.5.12, pynndescent==0.6.0]
  patterns:
    - TDD (RED/GREEN) for eval scripts
    - matplotlib Agg backend for headless PNG generation
    - Optional n_samples subsampling before dimensionality reduction (OOM safety)
    - run_X() function returns all outputs so tests can verify without running main()

key-files:
  created:
    - eval/tsne_vis.py
    - eval/umap_vis.py
    - tests/test_eval_tsne.py
    - tests/test_eval_umap.py

key-decisions:
  - "run_tsne accepts optional n_samples arg for direct-call subsampling (separate from extract_features path)"
  - "run_umap returns (path, reducer) tuple so reducer can map new samples post-hoc"
  - "PCA pre-reduction gate is dim > 50 (not >= 50) matching the spec and sklearn default behavior"
  - "umap-learn installed at execution time (was missing from environment per RESEARCH.md)"

patterns-established:
  - "Eval script pattern: importable module with run_X() + get_args() + main() + __main__ guard"
  - "Subsample-before-reduction: guard against OOM by capping samples in run_X() function, not just extract_features()"
  - "Test mocking: patch sklearn/umap constructors to capture kwargs and verify hyperparameter contract"

requirements-completed: [EVAL-03, EVAL-04]

# Metrics
duration: 15min
completed: 2026-04-11
---

# Phase 09 Plan 03: t-SNE and UMAP Visualization Summary

**t-SNE script with PCA pre-reduction + 3-perplexity sweep and UMAP script with cosine metric, reducer reuse, and torchdr suggestion for >50K samples**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-04-11T00:00:00Z
- **Completed:** 2026-04-11
- **Tasks:** 2 (both TDD)
- **Files created:** 4

## Accomplishments

- `eval/tsne_vis.py`: runs PCA to 50 dims (when dim > 50), then t-SNE sweep over perplexities [10, 30, 50] saving `tsne_perp{N}.png` per value; uses `init='pca'`, `metric='cosine'`, `learning_rate='auto'`
- `eval/umap_vis.py`: runs `umap.UMAP(metric='cosine', random_state=42)`, saves `umap.png`, prints torchdr note for >50K samples, and returns `(path, reducer)` tuple for new-sample mapping
- 15 unit tests passing across both scripts; both scripts importable as modules

## Task Commits

1. **Task 1 RED: Failing t-SNE tests** - `ff69056` (test)
2. **Task 1 GREEN: t-SNE implementation** - `cc2b076` (feat)
3. **Task 2 RED: Failing UMAP tests** - `188c67d` (test)
4. **Task 2 GREEN: UMAP implementation** - `139a3ad` (feat)

## Files Created/Modified

- `eval/tsne_vis.py` - t-SNE visualization script with PCA pre-reduction, 3-perplexity sweep, D-01 invocation pattern
- `eval/umap_vis.py` - UMAP visualization script with cosine metric, reducer return, torchdr suggestion
- `tests/test_eval_tsne.py` - 7 unit tests: 3 PNGs produced, filenames with perplexity, PCA applied/skipped by dim, TSNE kwargs verified, n_samples subsampling, non-empty PNG
- `tests/test_eval_umap.py` - 8 unit tests: PNG produced, filename, reducer returned, UMAP kwargs, torchdr print/no-print, non-empty PNG, n_samples subsampling

## Decisions Made

- `run_umap` returns `(path, reducer)` tuple (not just path) so callers can apply `reducer.transform(new_feats)` on new samples without re-fitting. This is mentioned in EVAL-04 spec.
- Both `run_tsne` and `run_umap` accept an optional `n_samples` parameter for direct subsampling when calling the functions standalone (separate from the `extract_features` path in `main()`). This supports the threat mitigation T-09-08 (large n_samples OOM).
- `umap-learn` was not installed in the environment. Added it via `pip install umap-learn` as part of D-03 decision (all eval libs in requirements.txt). No code deviation.

## Deviations from Plan

None - plan executed exactly as written. umap-learn installation was expected per RESEARCH.md (listed as "No -- must install").

## Issues Encountered

- `umap-learn` was not pre-installed. Installed it before writing tests (Rule 3 pre-condition). No impact on implementation.

## Known Stubs

None - both scripts are fully implemented. `main()` functions require a real checkpoint and data directory (as designed for offline eval tools), but `run_tsne()` and `run_umap()` are fully functional standalone.

## Threat Flags

None - no new network endpoints or trust boundaries introduced. Both scripts operate on local files only.

## Self-Check

Files verified:
- `eval/tsne_vis.py`: FOUND
- `eval/umap_vis.py`: FOUND
- `tests/test_eval_tsne.py`: FOUND
- `tests/test_eval_umap.py`: FOUND

Commits verified:
- `ff69056`: test(09-03): add failing tests for t-SNE visualization
- `cc2b076`: feat(09-03): implement t-SNE visualization script
- `188c67d`: test(09-03): add failing tests for UMAP visualization
- `139a3ad`: feat(09-03): implement UMAP visualization script

## Self-Check: PASSED

## Next Phase Readiness

- t-SNE and UMAP scripts ready for use in 09-07 integration test
- Both scripts follow D-01 invocation pattern and are importable as modules
- No blockers for remaining evaluation suite plans

---
*Phase: 09-evaluation-suite*
*Completed: 2026-04-11*
