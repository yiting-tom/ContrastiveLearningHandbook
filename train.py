"""train.py — single-entry training script for all SSL methods.

Usage::

    python train.py --config configs/simclr_v1_resnet18.yaml
    python train.py --config configs/byol_resnet18.yaml --data-dir data/imagenet100
    python train.py --config configs/supcon_stage2_resnet18.yaml --ckpt-path logs/.../last.ckpt

The script loads a YAML config via core.config.load_config, dispatches the
method via core.dispatcher.method_dispatcher, builds an SSLDataModule, and
runs Lightning's Trainer.fit(). Importing `methods` triggers register_method()
for all 14 SSL methods at startup.
"""
from __future__ import annotations

import argparse
import sys

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from core.config import load_config
from core.data import SSLDataModule
from core.dispatcher import method_dispatcher
import methods  # noqa: F401  triggers register_method() for all 14 methods


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a self-supervised contrastive learning method.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g., configs/simclr_v1_resnet18.yaml)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override data_dir from config (ImageFolder root)",
    )
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Resume training from a Lightning checkpoint",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_dir is not None:
        cfg = cfg.model_copy(update={"data_dir": args.data_dir})

    # WIRE-03 (D-05, D-06, D-07): SupCon stage-2 must load the stage-1 backbone.
    # The default dispatcher for "supcon_finetune" would instantiate a random backbone.
    # from_stage1_ckpt() loads backbone.* weights and freezes them before returning.
    # ckpt_path must NOT be passed to trainer.fit() — that would resume Lightning
    # training state (optimizer state, epoch count) from a stage-1 run, not just
    # load the backbone.
    if cfg.method == "supcon_finetune":
        if args.ckpt_path is None:
            print(
                "Error: supcon_finetune requires --ckpt-path pointing to a stage-1 checkpoint",
                file=sys.stderr,
            )
            sys.exit(1)
        from methods.supcon.module import SupConFinetuneModule
        model = SupConFinetuneModule.from_stage1_ckpt(args.ckpt_path, cfg)
    else:
        model = method_dispatcher(cfg)
    dm = SSLDataModule(
        data_dir=cfg.data_dir,
        n_views=cfg.n_views,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # Always save `last.ckpt` so that documented eval commands
    # (e.g., `python eval/linear_probe.py ... --ckpt .../last.ckpt`) work
    # without requiring users to discover the epoch=*-step=*.ckpt filename.
    # save_top_k=-1 saves every epoch; pair with save_last=True for the
    # stable `last.ckpt` alias. See:
    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
    callbacks: list[L.Callback] = [
        ModelCheckpoint(save_last=True, save_top_k=-1),
    ]
    if cfg.eval is not None and cfg.eval.knn is not None:
        from eval.knn_callback import KNNCallback
        callbacks.append(KNNCallback(cfg.eval.knn))

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        gradient_clip_val=cfg.gradient_clip_val,
        callbacks=callbacks,
    )
    # D-07: Do not pass ckpt_path to trainer.fit() for supcon_finetune.
    # from_stage1_ckpt() already loaded the backbone; passing ckpt_path here
    # would incorrectly resume Lightning training state from the stage-1 run.
    _fit_ckpt = None if cfg.method == "supcon_finetune" else args.ckpt_path
    trainer.fit(model, dm, ckpt_path=_fit_ckpt)


if __name__ == "__main__":
    main()
