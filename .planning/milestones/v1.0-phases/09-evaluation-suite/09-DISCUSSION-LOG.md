> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the discussion.

**Date:** 2026-04-10
**Phase:** 09-evaluation-suite
**Mode:** discuss
**Areas discussed:** Eval script invocation, KNNCallback data source, Optional vs. required deps, Feature cache location, Integration test approach

## Gray Areas Presented

| Gray Area | Options Offered | User Choice |
|-----------|----------------|-------------|
| Eval script invocation | Config + checkpoint flag / Checkpoint path inside YAML / Unified entry point | Config + checkpoint flag |
| KNNCallback data source | DataModule val_dataloader / Separate eval dataset config | DataModule val_dataloader |
| Optional vs. required deps | All in requirements.txt / Optional with graceful fallback | All in requirements.txt |
| Feature cache location | Checkpoint sibling dir / Configurable cache_dir | Checkpoint sibling dir |
| Integration test approach | Synthetic checkpoint + synthetic data / Train 5 epochs in test | Synthetic checkpoint + synthetic data |

## Discussion Summary

All recommended options were selected. No corrections or scope creep to record.

- **Invocation:** `python eval/<script>.py configs/exp.yaml --ckpt path/to/ckpt` — config supplies eval settings, checkpoint passed separately
- **KNN data:** `trainer.datamodule.val_dataloader()` — reuses existing SSLDataModule val split; requires `val/` subfolder in data_dir
- **Deps:** `faiss-cpu`, `umap-learn`, `pytorch-grad-cam` all go in `requirements.txt`
- **Cache:** `{ckpt_parent.parent}/cache/` — auto-located, no config needed
- **Integration test:** Synthetic ImageFolder + random-weight SimCLR checkpoint; knn_acc threshold relaxed to >= 0.0

## Deferred Ideas

None.
