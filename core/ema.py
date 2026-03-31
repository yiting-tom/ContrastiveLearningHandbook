"""EMA updater for momentum-based self-supervised learning methods.

Used by MoCo v1/v2/v3, BYOL, DINO, and similar methods that maintain a
momentum (teacher/target) encoder alongside an online (student) encoder.
"""
import math
from typing import Iterable

import torch
import torch.nn as nn


class EMAUpdater:
    """Exponential Moving Average updater with cosine-scheduled momentum.

    Used by MoCo v1/v2/v3, BYOL, and DINO for momentum encoder updates.
    MUST be called in on_train_batch_end, NOT in training_step or
    on_before_optimizer_step.

    Momentum schedule: cosine ramp from base_momentum to end_momentum over
    total_steps. The schedule follows the BYOL paper formula:

        m(t) = end_momentum - (end_momentum - base_momentum)
               * (cos(pi * t / T) + 1) / 2

    where t is the current step and T is total_steps.

    At step 0: momentum = base_momentum
    At step total_steps: momentum = end_momentum

    Args:
        base_momentum: Starting momentum value (e.g., 0.996 for BYOL,
            0.999 for MoCo). Should be close to but less than end_momentum.
        end_momentum: Final momentum value (typically 1.0).
        total_steps: Total training steps for schedule computation.
            Usually global_step at the end of training.
    """

    def __init__(
        self,
        base_momentum: float,
        end_momentum: float,
        total_steps: int,
    ) -> None:
        self.base_momentum = base_momentum
        self.end_momentum = end_momentum
        self.total_steps = total_steps
        self._step: int = 0

    @property
    def current_momentum(self) -> float:
        """Compute current momentum using cosine schedule.

        Returns:
            Momentum value in [base_momentum, end_momentum].
        """
        t = self._step / max(self.total_steps, 1)
        return self.end_momentum - (self.end_momentum - self.base_momentum) * (
            math.cos(math.pi * t) + 1
        ) / 2

    @torch.no_grad()
    def step(
        self,
        online_params: Iterable[nn.Parameter],
        target_params: Iterable[nn.Parameter],
    ) -> None:
        """Update target parameters toward online parameters.

        Formula: target = m * target + (1 - m) * online

        Call this from Lightning's on_train_batch_end hook, not from
        training_step or on_before_optimizer_step. The target parameters
        MUST have requires_grad=False to prevent gradient flow through the
        momentum encoder.

        Args:
            online_params: Iterable of online (student) network parameters.
                These are updated by the optimizer; EMA does not touch them.
            target_params: Iterable of target (teacher/momentum) network
                parameters. These MUST have requires_grad=False.
        """
        m = self.current_momentum
        for p_online, p_target in zip(online_params, target_params):
            p_target.data.mul_(m).add_(p_online.data, alpha=1 - m)
        self._step += 1
