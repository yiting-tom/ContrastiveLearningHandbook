# Phase 11: Code Fix & Export Cleanup - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-03
**Phase:** 11-Code Fix & Export Cleanup
**Areas discussed:** DINO crop config source, IndexedDataset collate wiring, supcon_finetune guard, InfoMinConfig CLEAN-01 verification

---

## DINO crop config source

| Option | Description | Selected |
|--------|-------------|----------|
| Hardcode in train.py | Fixed 2×224 + 6×96 values in train.py for dino; no config schema change | ✓ |
| Add crop fields to DINOConfig | Add n_large_crops, large_size, n_small_crops, small_size to DINOConfig with SwAV defaults | |

**User's choice:** Hardcode in train.py

| Follow-up: Shared vs separate blocks | Description | Selected |
|--------------------------------------|-------------|----------|
| Shared helper: one block for both | cfg.method in {"swav", "dino"} — reads SwAVConfig if swav, uses defaults for dino | ✓ |
| Separate if-blocks per method | Independent if-blocks for swav and dino | |

**User's choice:** Shared helper block

**Notes:** Avoids DINOConfig schema churn. SwAVConfig crop field defaults (n_large_crops=2, large_size=224, n_small_crops=6, small_size=96) match DINO paper defaults — reuse as hardcoded values for DINO.

---

## IndexedDataset collate wiring

| Option | Description | Selected |
|--------|-------------|----------|
| isinstance check inside SSLDataModule | Add elif isinstance(IndexedDataset) in train_dataloader() | ✓ |
| Add collate_fn param to SSLDataModule | API change: SSLDataModule.__init__ accepts collate_fn= | |

**User's choice:** isinstance check inside SSLDataModule

| Follow-up: Where wrapping happens | Description | Selected |
|-----------------------------------|-------------|----------|
| train.py wraps (Recommended) | train.py creates IndexedDataset, passes via dataset= param | ✓ |
| SSLDataModule wraps internally | DataModule detects method name and wraps itself | |

**User's choice:** train.py wraps the dataset

**Notes:** Keeps method-specific wiring decisions in train.py where the other method-specific fixes live. SSLDataModule stays generic.

---

## supcon_finetune guard

| Option | Description | Selected |
|--------|-------------|----------|
| Raise clear error | sys.exit(1) with "supcon_finetune requires --ckpt-path" message | ✓ |
| Fall back silently | Call method_dispatcher, train from scratch with random backbone | |

**User's choice:** Raise clear error

| Follow-up: Pass ckpt_path to trainer.fit()? | Description | Selected |
|---------------------------------------------|-------------|----------|
| No — don't pass to trainer.fit() | from_stage1_ckpt() loads backbone; trainer.fit() with no ckpt_path starts fresh | ✓ |
| Yes — pass to trainer.fit() too | Would cause Lightning to resume stage-1 training state (wrong behavior) | |

**User's choice:** Do NOT pass ckpt_path to trainer.fit() for supcon_finetune

**Notes:** from_stage1_ckpt() is a classmethod that extracts the backbone from the Lightning checkpoint — it's not the same as resuming training state.

---

## InfoMinConfig CLEAN-01 verification

| Option | Description | Selected |
|--------|-------------|----------|
| Verify then skip if already clean | Confirm one InfoMinConfig, mark done with 'no duplicate found' | ✓ |
| Full config.py dead-code audit | Scan all of config.py for any dead blocks beyond InfoMinConfig | |

**User's choice:** Verify then skip if already clean

**Notes:** Current grep shows only one InfoMinConfig definition at line 137. ROADMAP mentions "lines 72–83 duplicate" but those lines are BarlowTwinsConfig/SimSiamConfig in current code. Likely already removed in a prior phase.

---

## Claude's Discretion

- Organization of WIRE-01/02/03 if-blocks within train.py main() — planner may extract into a helper function or keep inline.

## Deferred Ideas

None — discussion stayed within phase scope.
