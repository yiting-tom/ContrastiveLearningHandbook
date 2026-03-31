"""Abstract base class for all self-supervised learning methods.

Every SSL method in this tutorial repo subclasses ``BaseSSLModule`` and
overrides only ``build_projector`` and ``training_step``.  All shared
infrastructure — optimizer dispatch, warmup-cosine LR scheduling, EMA
updates, and TensorBoard logging — is implemented here once.

Usage::

    class MyMethod(BaseSSLModule):
        def __init__(self, cfg: TrainConfig):
            super().__init__(cfg)
            self.projector = self.build_projector()

        def build_projector(self):
            return nn.Sequential(...)

        def training_step(self, batch, batch_idx):
            loss = ...
            self.log_train_metrics(loss)
            return loss
"""
from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn
import lightning as L
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from core.config import TrainConfig
from core.optimizers import LARS


class BaseSSLModule(L.LightningModule):
    """Abstract base class for all SSL methods.

    Subclasses MUST implement:
      - ``build_projector() -> nn.Module``: returns the projection head.
      - ``training_step(batch, batch_idx) -> loss``: returns the training loss.

    Subclasses MAY override:
      - ``learnable_params``: property returning parameters for the optimizer.
        Default returns ``self.parameters()`` — override to exclude EMA target
        network params (those MUST have ``requires_grad=False``).

    Provides:
      - ``configure_optimizers()``: AdamW / SGD / LARS dispatch with
        warmup-cosine LR scheduler wired at step granularity.
      - ``on_train_batch_end()``: EMA update hook (fires if
        ``self.ema_updater`` is set).
      - ``log_train_metrics(loss, **kwargs)``: TensorBoard logging helper.

    Args:
        cfg: Fully-validated ``TrainConfig`` instance.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters({"cfg": cfg.model_dump()})

        # EMA support — subclasses set these if using a momentum encoder.
        # CRITICAL: target network params MUST have requires_grad=False.
        self.ema_updater = None        # Optional[EMAUpdater]
        self._online_params = None     # Optional[Iterable[nn.Parameter]]
        self._target_params = None     # Optional[Iterable[nn.Parameter]]

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def build_projector(self) -> nn.Module:
        """Build and return the projection head.

        Called by the subclass ``__init__`` after ``super().__init__(cfg)``.
        The returned module should be stored as ``self.projector``.
        """
        ...

    # ------------------------------------------------------------------
    # Learnable parameters
    # ------------------------------------------------------------------

    @property
    def learnable_params(self):
        """Parameters passed to the optimizer.

        Override in subclasses that use EMA to exclude target network params.
        Default: all module parameters.

        CRITICAL: Momentum encoder parameters MUST NOT appear here.
        Call ``target_network.requires_grad_(False)`` immediately after
        ``deepcopy`` so they are excluded from the default as well, but
        overriding this property makes the intent explicit.
        """
        return self.parameters()

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Build optimizer and warmup-cosine LR scheduler.

        Dispatches to AdamW, SGD, or LARS based on ``cfg.optimizer``.
        Scheduler: linear warmup for ``cfg.warmup_epochs`` then cosine
        annealing to zero.  Interval is *step*-based (not epoch-based) so
        that the LR updates smoothly regardless of dataset size.

        Returns:
            Dict with ``"optimizer"`` and ``"lr_scheduler"`` keys compatible
            with PyTorch Lightning's ``configure_optimizers`` API.

        Raises:
            ValueError: If ``cfg.optimizer`` is not one of ``adamw``, ``sgd``,
                ``lars``.
        """
        params = list(self.learnable_params)

        if self.cfg.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=self.cfg.lr,
                momentum=0.9,
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer == "lars":
            optimizer = LARS(
                params,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            )
        else:
            raise ValueError(
                f"Unknown optimizer: {self.cfg.optimizer!r}. "
                f"Supported values: 'adamw', 'sgd', 'lars'."
            )

        # Compute step counts —  estimated_stepping_batches accounts for
        # gradient accumulation and multi-device training automatically.
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(
            total_steps * self.cfg.warmup_epochs / max(self.cfg.max_epochs, 1)
        )

        # Linear warmup from a near-zero LR to cfg.lr over warmup_steps.
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=max(warmup_steps, 1),
        )
        # Cosine annealing from cfg.lr to 0 over the remaining steps.
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps - warmup_steps, 1),
        )
        # Transition from warmup to cosine at warmup_steps.
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[max(warmup_steps, 1)],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    # ------------------------------------------------------------------
    # EMA hook
    # ------------------------------------------------------------------

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """EMA update hook — runs after the optimizer step.

        Calls ``EMAUpdater.step`` to update the target (momentum) network
        toward the online network.  Only active if ``self.ema_updater`` and
        ``self._online_params`` have been set by the subclass.

        Args:
            outputs: Return value of ``training_step``.
            batch: The current batch.
            batch_idx: Index of the current batch.
        """
        if self.ema_updater is not None and self._online_params is not None:
            self.ema_updater.step(self._online_params, self._target_params)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_train_metrics(self, loss: torch.Tensor, **kwargs) -> None:
        """Log standard SSL training metrics to TensorBoard.

        Logs ``train/loss`` (on_step + on_epoch) and ``train/lr``
        (on_step only).  Extra keyword arguments are logged as
        ``train/{key}``.

        Args:
            loss: The scalar training loss tensor.
            **kwargs: Additional metrics to log under the ``train/`` prefix.
        """
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        # Log current LR from the first optimizer param group.
        opt = self.optimizers()
        if opt is not None:
            lr = opt.param_groups[0]["lr"]
            self.log("train/lr", lr, on_step=True, on_epoch=False)
        for key, value in kwargs.items():
            self.log(f"train/{key}", value, on_step=True, on_epoch=True)
