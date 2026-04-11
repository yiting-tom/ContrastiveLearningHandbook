---
status: testing
phase: 04-moco
source: [04-01-SUMMARY.md, 04-02-SUMMARY.md, 04-03-SUMMARY.md]
started: 2026-04-07T00:00:00Z
updated: 2026-04-07T00:00:00Z
---

## Current Test

<!-- OVERWRITE each test - shows where we are -->

number: 5
name: MoCoV2Module Uses MLP Projector
expected: |
  Import MoCoV2Module and inspect its projector.
  `type(module.projector)` should NOT be `nn.Linear` — it should be a ProjectionHead with 2 layers (MLP + BN).
awaiting: user response

## Tests

### 1. Full Test Suite Passes
expected: Run `pytest` from the project root. All 141 tests should pass with no failures or errors. Summary line should show "141 passed".
result: pass

### 2. MomentumQueue FIFO and L2 Normalization
expected: Import MomentumQueue from core and run: `q = MomentumQueue(dim=4, queue_size=8); q.enqueue(torch.randn(2,4)); keys = q.get_negatives()`. Keys should have shape [4, 8], and each column should be L2-normalized (norm ≈ 1.0).
result: pass

### 3. MomentumQueue Checkpoint Persistence
expected: Create a queue, enqueue some keys, save with `torch.save(q.state_dict(), ...)`, reload into a fresh queue with `q2.load_state_dict(...)`. The loaded queue should have the same keys and pointer as the original.
result: pass

### 4. MoCoV1Module Uses Bare Linear Projector
expected: Import MoCoV1Module and inspect `module.projector` — it should be an `nn.Linear` layer (no hidden layer, no BN). Check `type(module.projector)` returns `torch.nn.Linear`.
result: pass

### 5. MoCoV2Module Uses MLP Projector
expected: Import MoCoV2Module and inspect `module.projector` — it should be a `ProjectionHead` with 2 layers (MLP with BN). Check that `type(module.projector)` is NOT `nn.Linear` and the module has more than one layer.
result: [pending]

### 6. Dispatcher Registration (moco_v1 and moco_v2)
expected: After `import methods`, calling the dispatcher with `"moco_v1"` and `"moco_v2"` should return the correct module classes without errors. No KeyError or ImportError.
result: [pending]

### 7. YAML Configs Validate Without Errors
expected: Load `configs/moco_v1_resnet18.yaml` and `configs/moco_v2_resnet18.yaml` via the project's TrainConfig (or equivalent config loader). Both should parse cleanly — no validation errors, missing fields, or wrong types.
result: [pending]

### 8. DOC-02 Docstrings Present on MoCo Modules
expected: Run the docstring validation tests (or inspect `MoCoV1Module.__doc__` and `MoCoV2Module.__doc__`). Both should contain DOC-02 required fields including: shuffled-BN limitation mention (v1), momentum sensitivity note, and Venue field (v2).
result: [pending]

## Summary

total: 8
passed: 4
issues: 0
pending: 4
skipped: 0
blocked: 0

## Gaps

[none yet]
