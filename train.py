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

import lightning as L

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

    model = method_dispatcher(cfg)
    dm = SSLDataModule(
        data_dir=cfg.data_dir,
        n_views=cfg.n_views,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    callbacks: list[L.Callback] = []
    if cfg.eval is not None and cfg.eval.knn is not None:
        from eval.knn_callback import KNNCallback
        callbacks.append(KNNCallback(cfg.eval.knn))

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        gradient_clip_val=cfg.gradient_clip_val,
        callbacks=callbacks,
    )
    trainer.fit(model, dm, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
