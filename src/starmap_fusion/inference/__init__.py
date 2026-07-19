"""Inference and post-processing interfaces."""

from .heatmap import decode_heatmap
from .predictor import StarDetection, StarPredictor, load_inference_model

__all__ = [
    "StarDetection",
    "StarPredictor",
    "decode_heatmap",
    "load_inference_model",
]
