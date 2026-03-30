# Phase 1: Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the discussion.

**Date:** 2026-03-30
**Phase:** 01-foundation
**Mode:** discuss
**Areas discussed:** Package layout, LARS sourcing, Test fixture strategy, Pydantic extra-fields policy

---

## Gray Areas Presented

| Area | Options offered |
|------|----------------|
| Package layout | flat-by-component, grouped-by-concern, single-module-file |
| LARS sourcing | from scratch, import from lightly |
| Test fixture strategy | random tensors only, random + synthetic ImageFolder, CIFAR-10 |
| Pydantic extra-fields policy | strict (forbid), permissive (ignore) |

---

## Decisions Made

### Package layout
- **Clarification from user:** "every method has a package" — each SSL method gets its own directory under `methods/`.
- **Chosen:** Top-level `core/` + `methods/` structure. Foundation code flat in `core/`; each method is a subdirectory package under `methods/`.

### LARS sourcing
- **Chosen:** Implement from scratch in `core/optimizers.py`. Rationale: tutorial repo — implementation should be readable alongside the paper.

### Test fixture strategy
- **Chosen:** Random tensors for unit tests + synthetic ImageFolder (temp dir) for data module tests.

### Pydantic extra-fields policy
- **Chosen:** Strict — `extra='forbid'` on all config models. Unknown keys raise `ValidationError`. Helps tutorial users catch typos early.

---

## No Corrections Applied

All selections were direct choices — no override of recommended defaults.

---

## Scope Notes

No deferred ideas emerged. Discussion stayed within Phase 1 boundary.
