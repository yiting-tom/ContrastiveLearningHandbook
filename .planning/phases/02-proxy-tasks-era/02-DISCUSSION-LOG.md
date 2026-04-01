# Phase 2: Proxy Tasks Era - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the analysis.

**Date:** 2026-04-01
**Phase:** 02-proxy-tasks-era
**Mode:** assumptions (--auto)
**Areas analyzed:** Sub-Config Schema Extension, NCE Loss vs. InfoNCELoss Reuse, MemoryBank Device and Gradient Handling, Registration Pattern and Module File Layout

## Assumptions Presented

### Sub-Config Schema Extension
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Add InstanceDiscriminationConfig and InvariantSpreadConfig to core/config.py as Optional fields on TrainConfig | Confident | core/config.py lines 36–99 (existing sub-configs), lines 193–199 (TrainConfig Optional fields), extra='forbid' constraint, configs/example.yaml pattern |

### NCE Loss vs. InfoNCELoss Reuse
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| New standalone NCE loss in methods/instance_discrimination/ — not reusing InfoNCELoss | Confident | core/losses.py lines 55–57 (internal L2-norm, no Z slot), ROADMAP plan 02-02 (Z fixed after first batch, ε=1e-7), ROADMAP plan 02-04 explicitly says InvariantSpread "reuses InfoNCELoss" |

### MemoryBank Device and Gradient Handling
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| nn.Embedding with requires_grad=False, no optimizer params | Likely | core/base.py lines 87–99 (learnable_params defaults to self.parameters()), ROADMAP plan 02-01 (update-by-index interface), ROADMAP plan 02-02 (Z fixed) |

### Registration Pattern and Module File Layout
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| register_method() called from methods/instance_discrimination/__init__.py; methods/__init__.py imports sub-packages | Likely | core/dispatcher.py lines 32–55 (registry dict, register_method), line 13 docstring ("Phases 2–8 call register_method() in __init__.py"), Phase 1 D-01 (methods/ sub-packages) |

## Corrections Made

No corrections — all assumptions auto-confirmed (--auto mode, all Confident/Likely).

## Auto-Resolved

All assumptions were Confident or Likely — no Unclear items required auto-resolution.

## External Research

No external research required — all Phase 2 decisions determinable from existing codebase.
