#!/usr/bin/env python3
"""Train the ResNet-50 FPN star heatmap detector on CUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from starmap_fusion.data.heatmap_dataset import (  # noqa: E402
    LamostHeatmapDataset,
    collate_heatmap_batch,
)
from starmap_fusion.models.star_heatmap import (  # noqa: E402
    StarHeatmapDetector,
    count_trainable_parameters,
)
from starmap_fusion.training.engine import TrainingConfig, fit  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the training command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--match-radius", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    return parser


def main() -> int:
    """Validate CUDA, build data loaders, and start training."""

    arguments = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. In Colab select Runtime > Change runtime type > GPU.")
    device = torch.device("cuda")
    train_dataset = LamostHeatmapDataset(
        manifest_path=arguments.manifest_dir / "train.jsonl",
        dataset_root=arguments.dataset_root,
        image_size=(arguments.image_size, arguments.image_size),
    )
    validation_dataset = LamostHeatmapDataset(
        manifest_path=arguments.manifest_dir / "validation.jsonl",
        dataset_root=arguments.dataset_root,
        image_size=(arguments.image_size, arguments.image_size),
    )
    common_loader_arguments = {
        "batch_size": arguments.batch_size,
        "num_workers": arguments.workers,
        "pin_memory": True,
        "persistent_workers": arguments.workers > 0,
        "collate_fn": collate_heatmap_batch,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_arguments)
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **common_loader_arguments
    )
    model = StarHeatmapDetector(
        pretrained_backbone=not arguments.no_pretrained_backbone
    )
    config = TrainingConfig(
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        confidence_threshold=arguments.confidence_threshold,
        match_radius=arguments.match_radius,
        seed=arguments.seed,
    )
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    fit(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        output_dir=arguments.output_dir,
        device=device,
        config=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
