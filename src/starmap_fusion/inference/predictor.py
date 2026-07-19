"""High-level tiled inference for star-center coordinate extraction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor
from torchvision.transforms import functional as transform_functional

from starmap_fusion.inference.heatmap import decode_heatmap
from starmap_fusion.models.star_heatmap import StarHeatmapDetector


@dataclass(frozen=True, slots=True)
class StarDetection:
    """One star-center prediction in original-image coordinates."""

    x: float
    y: float
    confidence: float


@dataclass(frozen=True, slots=True)
class InferenceModelConfig:
    """Runtime parameters stored with an inference checkpoint."""

    input_size: tuple[int, int]
    output_stride: int
    confidence_threshold: float
    pyramid_channels: int
    head_channels: int
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]


def load_inference_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[StarHeatmapDetector, InferenceModelConfig, dict[str, Any]]:
    """Load an inference or training checkpoint and reconstruct the model."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    config = InferenceModelConfig(
        input_size=tuple(checkpoint.get("input_size", (1024, 1024))),
        output_stride=int(checkpoint.get("output_stride", 4)),
        confidence_threshold=float(checkpoint.get("confidence_threshold", 0.3)),
        pyramid_channels=int(checkpoint.get("pyramid_channels", 128)),
        head_channels=int(checkpoint.get("head_channels", 128)),
        normalization_mean=tuple(
            checkpoint.get("normalization", {}).get("mean", (0.485, 0.456, 0.406))
        ),
        normalization_std=tuple(
            checkpoint.get("normalization", {}).get("std", (0.229, 0.224, 0.225))
        ),
    )
    model = StarHeatmapDetector(
        pretrained_backbone=False,
        pyramid_channels=config.pyramid_channels,
        head_channels=config.head_channels,
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, config, checkpoint


def _tile_starts(length: int, tile_size: int, overlap: int) -> tuple[int, ...]:
    """Return tile origins that cover a dimension including its far edge."""

    if length <= tile_size:
        return (0,)
    step = tile_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than tile_size")
    starts = list(range(0, length - tile_size + 1, step))
    last_start = length - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return tuple(starts)


def merge_detections(
    detections: Sequence[StarDetection],
    merge_radius: float,
) -> list[StarDetection]:
    """Greedily merge overlapping-tile detections by confidence."""

    kept: list[StarDetection] = []
    radius_squared = merge_radius**2
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        is_duplicate = any(
            (candidate.x - existing.x) ** 2 + (candidate.y - existing.y) ** 2
            <= radius_squared
            for existing in kept
        )
        if not is_duplicate:
            kept.append(candidate)
    return sorted(kept, key=lambda item: (item.y, item.x))


def detect_with_sep(image: Image.Image, threshold_sigma: float = 3.0) -> list[tuple[float, float]]:
    """Detect classical point-source candidates with SEP background subtraction."""

    try:
        import sep
    except ImportError as error:
        raise RuntimeError("SEP filtering requires `pip install sep`") from error
    grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    background = sep.Background(grayscale)
    objects = sep.extract(
        grayscale - background.back(),
        threshold_sigma,
        err=background.globalrms,
    )
    return [(float(item["x"]), float(item["y"])) for item in objects]


def fuse_with_sep(
    ml_detections: Sequence[StarDetection],
    sep_coordinates: Sequence[tuple[float, float]],
    match_radius: float = 5.0,
    include_sep_only: bool = False,
    sep_only_confidence: float = 0.4,
) -> list[StarDetection]:
    """Boost SEP-confirmed ML points and optionally retain SEP-only candidates."""

    radius_squared = match_radius**2
    matched_sep: set[int] = set()
    fused: list[StarDetection] = []
    for detection in ml_detections:
        closest_index: int | None = None
        closest_distance = float("inf")
        for index, (sep_x, sep_y) in enumerate(sep_coordinates):
            distance = (detection.x - sep_x) ** 2 + (detection.y - sep_y) ** 2
            if distance <= radius_squared and distance < closest_distance:
                closest_index = index
                closest_distance = distance
        if closest_index is None:
            fused.append(detection)
            continue
        matched_sep.add(closest_index)
        sep_x, sep_y = sep_coordinates[closest_index]
        fused.append(
            StarDetection(
                x=(detection.x + sep_x) / 2.0,
                y=(detection.y + sep_y) / 2.0,
                confidence=1.0 - (1.0 - detection.confidence) * 0.5,
            )
        )
    if include_sep_only:
        for index, (sep_x, sep_y) in enumerate(sep_coordinates):
            if index not in matched_sep:
                fused.append(StarDetection(sep_x, sep_y, sep_only_confidence))
    return merge_detections(fused, merge_radius=match_radius)


class StarPredictor:
    """Predict star coordinates on arbitrary images using overlapping tiles."""

    def __init__(
        self,
        model: StarHeatmapDetector,
        config: InferenceModelConfig,
        device: torch.device,
    ) -> None:
        """Store the loaded model and preprocessing parameters."""

        self.model = model
        self.config = config
        self.device = device

    def _prepare_tile(self, tile: Image.Image) -> tuple[Tensor, int, int]:
        """Pad one RGB tile to model input size and normalize it."""

        input_height, input_width = self.config.input_size
        tile_width, tile_height = tile.size
        canvas = Image.new("RGB", (input_width, input_height), color=(0, 0, 0))
        canvas.paste(tile, (0, 0))
        tensor = transform_functional.to_tensor(canvas)
        tensor = transform_functional.normalize(
            tensor,
            mean=self.config.normalization_mean,
            std=self.config.normalization_std,
        )
        return tensor.unsqueeze(0), tile_width, tile_height

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        confidence_threshold: float | None = None,
        overlap: int = 256,
        merge_radius: float = 5.0,
        mask_left_pixels: int = 0,
    ) -> list[StarDetection]:
        """Return filtered star centers in original-image pixel coordinates."""

        rgb_image = image.convert("RGB")
        image_width, image_height = rgb_image.size
        tile_height, tile_width = self.config.input_size
        threshold = (
            self.config.confidence_threshold
            if confidence_threshold is None
            else confidence_threshold
        )
        candidates: list[StarDetection] = []
        for top in _tile_starts(image_height, tile_height, overlap):
            for left in _tile_starts(image_width, tile_width, overlap):
                right = min(image_width, left + tile_width)
                bottom = min(image_height, top + tile_height)
                tile = rgb_image.crop((left, top, right, bottom))
                tile_tensor, valid_width, valid_height = self._prepare_tile(tile)
                logits = self.model(tile_tensor.to(self.device, non_blocking=True))
                decoded = decode_heatmap(
                    logits,
                    confidence_threshold=threshold,
                )[0].cpu()
                for heatmap_x, heatmap_y, confidence in decoded.tolist():
                    local_x = heatmap_x * self.config.output_stride
                    local_y = heatmap_y * self.config.output_stride
                    if local_x >= valid_width or local_y >= valid_height:
                        continue
                    global_x = local_x + left
                    global_y = local_y + top
                    if global_x < mask_left_pixels:
                        continue
                    candidates.append(
                        StarDetection(global_x, global_y, float(confidence))
                    )
        return merge_detections(candidates, merge_radius=merge_radius)


def save_predictions_json(
    output_path: Path,
    image_path: Path,
    image_size: tuple[int, int],
    detections: Sequence[StarDetection],
    confidence_threshold: float,
) -> None:
    """Save coordinates and confidence values in a stable JSON schema."""

    payload = {
        "image": image_path.name,
        "width": image_size[0],
        "height": image_size[1],
        "confidence_threshold": confidence_threshold,
        "count": len(detections),
        "stars": [asdict(detection) for detection in detections],
        "coordinates": [[detection.x, detection.y] for detection in detections],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def save_visualization(
    output_path: Path,
    image: Image.Image,
    detections: Sequence[StarDetection],
    radius: int = 8,
) -> None:
    """Draw detected star centers and confidence values on an image."""

    visualization = image.convert("RGB")
    drawer = ImageDraw.Draw(visualization)
    for detection in detections:
        box = (
            detection.x - radius,
            detection.y - radius,
            detection.x + radius,
            detection.y + radius,
        )
        drawer.ellipse(box, outline=(255, 0, 0), width=2)
        drawer.ellipse(
            (detection.x - 1, detection.y - 1, detection.x + 1, detection.y + 1),
            fill=(255, 0, 0),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visualization.save(output_path)
