"""SwAV loss functions: Sinkhorn-Knopp optimal transport and swapped-prediction loss.

Reference:
    Caron et al., "Unsupervised Learning of Visual Features by Contrasting
    Cluster Assignments" (NeurIPS 2020). https://arxiv.org/abs/2006.09882

Reference implementation:
    https://github.com/facebookresearch/swav
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def sinkhorn_knopp(
    scores: torch.Tensor,
    n_iters: int = 3,
    epsilon: float = 0.05,
) -> torch.Tensor:
    """Convert raw prototype scores into a doubly-stochastic assignment matrix Q.

    Applies the Sinkhorn-Knopp algorithm to produce soft cluster assignments
    where each row (sample) and each column (prototype) has approximately equal
    mass. This enforces equipartition of samples across prototypes, preventing
    mode collapse.

    Args:
        scores: Raw prototype similarity scores, shape [B, K].
            B = batch size, K = number of prototypes.
            Typically the dot product of L2-normalized features and prototypes.
        n_iters: Number of Sinkhorn normalisation iterations (default: 3).
            Use 10+ iterations in unit tests for accurate doubly-stochastic check;
            3 iterations is sufficient for training stability.
        epsilon: Temperature parameter controlling assignment sharpness (default: 0.05).
            Smaller epsilon produces sharper (more discrete) assignments.

    Returns:
        Q: Soft assignment matrix of shape [B, K].
            - Row sums are approximately 1.0 (each sample distributed over prototypes).
            - Column sums are approximately B/K (uniform prototype usage).
            - All values are non-negative.
            - Tensor has requires_grad=False (decorated with @torch.no_grad).

    Notes:
        Doubly-stochastic property: After convergence, Q approximates a doubly-
        stochastic matrix scaled so that row sums = 1 and col sums = B/K.
        The number of iterations required for convergence grows with B and K;
        3 iterations is empirically sufficient for typical SwAV training settings
        (B=256, K=3000).

    From the official facebookresearch/swav implementation (single-GPU version).
    """
    # Subtract per-column max for numerical stability (log-sum-exp trick).
    # Prevents exp overflow when scores/epsilon are large (e.g. scores > 4.4
    # with epsilon=0.05 overflows float32). Subtracting max does not change the
    # doubly-stochastic result because the subsequent row/column normalization
    # is scale-invariant.
    scaled = scores / epsilon  # [B, K]
    scaled = scaled - scaled.max(dim=0, keepdim=True).values  # per-prototype max subtraction
    Q = torch.exp(scaled).t()  # [K, B]
    B = Q.shape[1]
    K = Q.shape[0]

    # Normalize the joint distribution to sum to 1
    Q = Q / Q.sum()

    for _ in range(n_iters):
        # Normalize rows (each prototype's total assignment = 1/K).
        # Clamp denominators to avoid division by zero in degenerate cases.
        Q = Q / Q.sum(dim=1, keepdim=True).clamp(min=1e-10) / K
        # Normalize columns (each sample's total assignment = 1/B)
        Q = Q / Q.sum(dim=0, keepdim=True).clamp(min=1e-10) / B

    # Scale so row sums = 1 (each sample's assignment sums to 1)
    Q = Q * B
    return Q.t()  # [B, K]


def swav_loss(
    z_list: list,
    prototype_layer: nn.Module,
    temperature: float,
    n_large_crops: int,
    sinkhorn_fn: callable,
) -> torch.Tensor:
    """Compute the SwAV swapped-prediction loss over multiple crop views.

    For each large crop i, compute soft cluster assignments q_i via Sinkhorn-Knopp.
    Then for every other crop v, compute log-softmax prototype scores and take the
    cross-entropy between q_i (codes, target) and the scores of crop v (predictions).
    The loss is averaged over all large-crop / other-crop pairs.

    Args:
        z_list: List of feature tensors, one per crop view.
            Each tensor has shape [B, D] where B = batch size, D = feature dim.
            First n_large_crops entries correspond to large crops; remaining entries
            correspond to small crops.
        prototype_layer: nn.Module that maps [B, D] -> [B, K] prototype scores.
            Typically an nn.Linear with weight L2-normalized after each update step.
        temperature: Softmax temperature for the prediction distribution (default: 0.1).
            Lower temperature makes predictions sharper.
        n_large_crops: Number of large crop views used to compute codes.
            Codes are computed ONLY from large crops (indices 0..n_large_crops-1).
            All crops (large + small) predict each large crop's codes.
        sinkhorn_fn: Callable with signature sinkhorn_fn(scores) -> Q.
            Injected for testability; in production, pass sinkhorn_knopp.

    Returns:
        Scalar loss tensor. Averaged over n_large_crops * (n_crops - 1) terms.

    Notes:
        CRITICAL: Sinkhorn output is detached (.detach()) -- codes are training
        targets (constants), not predictions. Gradients flow only through the
        prediction side (log_softmax of prototype scores).

        CRITICAL: Codes are computed only from large crops. Small crops only act
        as predictors, never as code sources, following the original SwAV paper.
    """
    n_crops = len(z_list)

    # Compute prototype scores for all crops after L2 normalization
    scores = [prototype_layer(F.normalize(z, dim=1)) for z in z_list]

    loss = torch.tensor(0.0, device=z_list[0].device)

    for i in range(n_large_crops):
        # Compute codes from large crop i; detach so codes are constants
        q = sinkhorn_fn(scores[i].detach())

        for v in range(n_crops):
            if v == i:
                continue
            # Prediction: log-softmax of prototype scores for crop v
            p = F.log_softmax(scores[v] / temperature, dim=1)
            # Cross-entropy: -sum_k q_k * log p_k, averaged over batch
            loss -= torch.mean(torch.sum(q * p, dim=1))

    # Average over all (large_crop, other_crop) pairs
    loss /= n_large_crops * (n_crops - 1)
    return loss
