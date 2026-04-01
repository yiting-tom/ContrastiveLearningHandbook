"""NCE loss with fixed normalization constant Z for Instance Discrimination.

This is a standalone loss (does NOT subclass the core InfoNCE loss) because
Z-normalization semantics are incompatible with the core loss forward() which
L2-normalizes inputs internally. See planning decision D-02.

Reference:
    Wu et al., "Unsupervised Feature Learning via Non-Parametric Instance
    Discrimination" (CVPR 2018). https://arxiv.org/abs/1805.01978
"""
import torch
import torch.nn as nn


class NCELossWithFixedZ(nn.Module):
    """(m+1)-way NCE loss with fixed normalization constant Z.

    Per Wu et al. (CVPR 2018), Z is estimated from the first mini-batch
    and fixed thereafter. Recomputing Z each step destabilizes training.

    Args:
        temperature: Softmax temperature (default: 0.07).
        n_negatives: Number of negatives sampled from bank (default: 4096).
        eps: Epsilon for numerical stability in denominator (default: 1e-7).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        n_negatives: int = 4096,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.n_negatives = n_negatives
        self.eps = eps
        self.register_buffer("Z", torch.tensor(-1.0))
        self.register_buffer("z_initialized", torch.tensor(False))

    def forward(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """Compute (m+1)-way NCE loss.

        Args:
            query: [B, D] L2-normalized encoder output.
            positive: [B, D] L2-normalized bank features for same indices.
            negatives: [B, m, D] L2-normalized sampled negative features.

        Returns:
            Scalar loss tensor.
        """
        # Positive logits: [B, 1]
        pos_logit = (query * positive).sum(dim=1, keepdim=True) / self.temperature
        # Negative logits: [B, m]
        neg_logits = torch.bmm(negatives, query.unsqueeze(2)).squeeze(2) / self.temperature

        # Estimate Z on first forward pass only
        if not self.z_initialized:
            with torch.no_grad():
                all_logits = torch.cat([pos_logit, neg_logits], dim=1)
                self.Z.fill_(all_logits.exp().mean().item() * (self.n_negatives + 1))
                self.z_initialized.fill_(True)

        # NCE probability
        pos_prob = pos_logit.exp() / (self.Z + self.eps)
        neg_prob = neg_logits.exp() / (self.Z + self.eps)

        # NCE loss: -log P(positive) - sum(log(1 - P(negative)))
        loss = (
            -pos_prob.clamp(min=self.eps).log().mean()
            - (1 - neg_prob + self.eps).clamp(min=self.eps).log().sum(dim=1).mean()
        )
        return loss
