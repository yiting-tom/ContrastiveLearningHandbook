# Phase 3: SimCLR - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the discussion.

**Date:** 2026-04-02
**Phase:** 03-simclr
**Mode:** discuss
**Areas discussed:** NT-Xent loss design, SimCLRv2 module design, Augmentation validation script

## Gray Areas Presented

### NT-Xent loss design
| Option | Notes |
|--------|-------|
| Use InfoNCELoss directly (selected) | No new class; plan 03-01 becomes unit tests |
| New NTXentLoss wrapper | Thin alias over InfoNCELoss |
| Standalone NTXentLoss (no reuse) | Self-contained, max tutorial readability |

### SimCLRv2 module design
| Option | Notes |
|--------|-------|
| Subclass SimCLRv1Module (selected) | build_projector() override, num_layers=3 |
| Config-only, single class | Single SimCLRModule reads num_layers from config |

### Augmentation validation script
| Option | Notes |
|--------|-------|
| tools/visualize_augmentations.py (selected) | Standalone CLI script |
| scripts/check_augmentations.py | Same, different convention |
| Claude decides | Leave to planning |

## Decisions Made

### NT-Xent Loss
- **Decision:** Use `InfoNCELoss` directly with `queue=None` — no new loss class
- **Rationale:** `InfoNCELoss._symmetric_loss` already IS NT-Xent; a new class would create "two classes that do the same thing" confusion

### SimCLRv2 Module Design
- **Decision:** `SimCLRv2Module(SimCLRv1Module)` subclass overriding `build_projector()` with `num_layers=3`
- **Rationale:** Clear OOP narrative (v2 = v1 + deeper head); provides a natural home for v2-specific docstring and gotchas

### Augmentation Validation Script
- **Decision:** `tools/visualize_augmentations.py` as a standalone CLI script
- **Rationale:** Persistent utility users can run on any image, not a one-off test

## Deferred Ideas

None.
