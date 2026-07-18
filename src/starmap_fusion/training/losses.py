"""Loss functions for center heatmap supervision."""

from __future__ import annotations

import torch
from torch import Tensor


def center_focal_loss(
    logits: Tensor,
    targets: Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> Tensor:
    """Compute a CenterNet-style focal loss for Gaussian heatmap targets."""

    probabilities = logits.sigmoid().clamp(min=1e-6, max=1.0 - 1e-6)
    positive_mask = targets.eq(1.0).to(logits.dtype)
    negative_mask = targets.lt(1.0).to(logits.dtype)
    negative_weights = (1.0 - targets).pow(beta)

    positive_loss = (
        torch.log(probabilities) * (1.0 - probabilities).pow(alpha) * positive_mask
    )
    negative_loss = (
        torch.log(1.0 - probabilities)
        * probabilities.pow(alpha)
        * negative_weights
        * negative_mask
    )
    positive_count = positive_mask.sum()
    total_loss = -(positive_loss.sum() + negative_loss.sum())
    return total_loss / positive_count.clamp(min=1.0)

