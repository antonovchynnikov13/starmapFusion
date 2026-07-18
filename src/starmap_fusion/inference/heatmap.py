"""Heatmap peak extraction for star-center predictions."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as functional


def decode_heatmap(
    logits: Tensor,
    confidence_threshold: float = 0.3,
    kernel_size: int = 3,
    max_detections: int = 1000,
) -> list[Tensor]:
    """Decode local maxima into ``(x, y, confidence)`` detections per image."""

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    probabilities = logits.sigmoid()
    pooled = functional.max_pool2d(
        probabilities,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    peak_mask = probabilities.eq(pooled) & probabilities.ge(confidence_threshold)
    decoded: list[Tensor] = []
    for image_probabilities, image_peaks in zip(probabilities, peak_mask, strict=True):
        peak_indices = image_peaks[0].nonzero(as_tuple=False)
        if peak_indices.numel() == 0:
            decoded.append(torch.empty((0, 3), device=logits.device))
            continue
        scores = image_probabilities[0, peak_indices[:, 0], peak_indices[:, 1]]
        keep_count = min(max_detections, scores.numel())
        top_scores, top_indices = scores.topk(keep_count)
        selected = peak_indices[top_indices]
        decoded.append(
            torch.stack(
                (selected[:, 1].to(logits.dtype), selected[:, 0].to(logits.dtype), top_scores),
                dim=1,
            )
        )
    return decoded

