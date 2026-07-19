"""PyTorch dataset and Gaussian target generation for star centers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as transform_functional


@dataclass(frozen=True, slots=True)
class HeatmapSample:
    """A loaded image and its heatmap training targets."""

    image: Tensor
    heatmap: Tensor
    centers: Tensor
    sample_id: str
    original_size: tuple[int, int]


def draw_gaussian(
    heatmap: Tensor,
    center_x: float,
    center_y: float,
    sigma: float,
) -> None:
    """Draw one truncated 2D Gaussian onto a heatmap in place."""

    height, width = heatmap.shape[-2:]
    radius = max(1, int(3.0 * sigma))
    left = max(0, int(center_x) - radius)
    right = min(width, int(center_x) + radius + 1)
    top = max(0, int(center_y) - radius)
    bottom = min(height, int(center_y) + radius + 1)
    if left >= right or top >= bottom:
        return

    x_coordinates = torch.arange(left, right, dtype=heatmap.dtype)
    y_coordinates = torch.arange(top, bottom, dtype=heatmap.dtype)
    y_grid, x_grid = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
    gaussian = torch.exp(
        -((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / (2.0 * sigma**2)
    )
    heatmap[..., top:bottom, left:right] = torch.maximum(
        heatmap[..., top:bottom, left:right], gaussian
    )
    peak_x = min(width - 1, max(0, round(center_x)))
    peak_y = min(height - 1, max(0, round(center_y)))
    heatmap[..., peak_y, peak_x] = 1.0


class LamostHeatmapDataset(Dataset[dict[str, Any]]):
    """Load LAMOST images and create stride-aligned Gaussian center heatmaps."""

    def __init__(
        self,
        manifest_path: Path,
        dataset_root: Path,
        image_size: tuple[int, int] = (512, 512),
        output_stride: int = 4,
        minimum_sigma: float = 1.0,
        maximum_sigma: float = 4.0,
    ) -> None:
        """Initialize the dataset from a prepared JSONL manifest."""

        self.manifest_path = manifest_path
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.output_stride = output_stride
        self.minimum_sigma = minimum_sigma
        self.maximum_sigma = maximum_sigma
        self.records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.records:
            raise ValueError(f"Manifest contains no records: {manifest_path}")
        if image_size[0] % output_stride or image_size[1] % output_stride:
            raise ValueError("Both image dimensions must be divisible by output_stride")

    def __len__(self) -> int:
        """Return the number of indexed images."""

        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one image and create its scaled center heatmap."""

        record = self.records[index]
        image_path = self.dataset_root / record["image_path"]
        with Image.open(image_path) as source_image:
            rgb_image = source_image.convert("RGB")
            original_width, original_height = rgb_image.size
            resized_image = rgb_image.resize(
                (self.image_size[1], self.image_size[0]),
                resample=Image.Resampling.BILINEAR,
            )

        image_tensor = transform_functional.to_tensor(resized_image)
        image_tensor = transform_functional.normalize(
            image_tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        output_height = self.image_size[0] // self.output_stride
        output_width = self.image_size[1] // self.output_stride
        heatmap = torch.zeros((1, output_height, output_width), dtype=torch.float32)
        scaled_centers: list[tuple[float, float]] = []
        scale_x = self.image_size[1] / original_width / self.output_stride
        scale_y = self.image_size[0] / original_height / self.output_stride

        for ellipse in record["ellipses"]:
            center_x = float(ellipse["center_x"]) * scale_x
            center_y = float(ellipse["center_y"]) * scale_y
            if not (0.0 <= center_x < output_width and 0.0 <= center_y < output_height):
                continue
            source_radius = max(float(ellipse["semi_major"]), float(ellipse["semi_minor"]))
            sigma = source_radius * (scale_x + scale_y) * 0.25
            sigma = min(self.maximum_sigma, max(self.minimum_sigma, sigma))
            draw_gaussian(heatmap, center_x, center_y, sigma)
            scaled_centers.append((center_x, center_y))

        centers_tensor = torch.tensor(scaled_centers, dtype=torch.float32).reshape(-1, 2)
        return {
            "image": image_tensor,
            "heatmap": heatmap,
            "centers": centers_tensor,
            "sample_id": str(record["sample_id"]),
            "original_size": (original_height, original_width),
        }


def collate_heatmap_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack fixed-size tensors while retaining variable-length center lists."""

    return {
        "images": torch.stack([sample["image"] for sample in samples]),
        "heatmaps": torch.stack([sample["heatmap"] for sample in samples]),
        "centers": [sample["centers"] for sample in samples],
        "sample_ids": [sample["sample_id"] for sample in samples],
        "original_sizes": [sample["original_size"] for sample in samples],
    }
