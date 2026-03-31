"""LARS (Layer-wise Adaptive Rate Scaling) optimizer for large-batch SSL training.

Implemented from scratch for tutorial readability per project decision D-04.
No dependency on lightly, torchlars, or other external optimizer libraries.
"""
import torch
from torch.optim import Optimizer


class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling optimizer.

    Reference: https://arxiv.org/abs/1708.03888

    LARS scales the learning rate per-layer based on the ratio of the
    parameter norm to the gradient norm (the "trust ratio"). This makes
    large-batch training more stable by preventing layers with small
    gradients from receiving excessively large effective learning rates.

    Bias parameters and batch-normalization parameters (1-D tensors) are
    excluded from trust ratio scaling by default, as their gradient
    dynamics differ from weight tensors (Pitfall 8 in RESEARCH.md).

    Args:
        params: Iterable of parameters to optimize or dict of param groups.
        lr: Base learning rate.
        momentum: Momentum coefficient. Default: 0.9.
        weight_decay: L2 weight decay coefficient. Default: 1e-6.
        eta: LARS trust coefficient. Controls how aggressively to scale
            the learning rate. Default: 0.001.
        exclude_bias_and_norm: If True, bias and normalization parameters
            (1-D tensors) use trust_ratio=1.0 (no LARS scaling). Default: True.

    Example::

        optimizer = LARS(model.parameters(), lr=0.3, weight_decay=1e-6)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 1e-6,
        eta: float = 0.001,
        exclude_bias_and_norm: bool = True,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eta=eta,
            exclude_bias_and_norm=exclude_bias_and_norm,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            loss: The loss value if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            eta = group["eta"]
            exclude_bias_and_norm = group["exclude_bias_and_norm"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay: g = g + weight_decay * p
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Compute LARS trust ratio
                # Exclude 1-D params (bias, BN weight/bias) when requested
                if exclude_bias_and_norm and p.ndim == 1:
                    trust_ratio = 1.0
                else:
                    p_norm = p.norm()
                    g_norm = grad.norm()
                    if p_norm > 0 and g_norm > 0:
                        trust_ratio = eta * p_norm / g_norm
                    else:
                        # Safety: if either norm is zero, don't scale
                        trust_ratio = 1.0

                scaled_lr = lr * trust_ratio

                # Apply momentum: buf = momentum * buf + grad
                if momentum != 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        # First step: initialize buffer from current grad
                        state["momentum_buffer"] = torch.clone(grad).detach()
                    else:
                        state["momentum_buffer"].mul_(momentum).add_(grad)
                    grad = state["momentum_buffer"]

                # Parameter update: p = p - scaled_lr * grad
                p.add_(grad, alpha=-scaled_lr)

        return loss
