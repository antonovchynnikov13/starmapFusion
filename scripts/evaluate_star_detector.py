#!/usr/bin/env python3
"""Evaluate a trained star detector on a prepared LAMOST split."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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
from starmap_fusion.inference.predictor import load_inference_model  # noqa: E402
from starmap_fusion.training.engine import evaluate  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the evaluation command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--match-radius", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> int:
    """Run evaluation and save metrics as JSON."""

    arguments = build_parser().parse_args()
    if arguments.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(arguments.device)
    model, _, checkpoint = load_inference_model(arguments.checkpoint, device)
    dataset = LamostHeatmapDataset(
        manifest_path=arguments.manifest,
        dataset_root=arguments.dataset_root,
        image_size=(arguments.image_size, arguments.image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=arguments.workers > 0,
        collate_fn=collate_heatmap_batch,
    )
    metrics = evaluate(
        model=model,
        data_loader=loader,
        device=device,
        confidence_threshold=arguments.confidence_threshold,
        match_radius=arguments.match_radius,
    )
    payload = {
        **asdict(metrics),
        "checkpoint": arguments.checkpoint.name,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "manifest": arguments.manifest.name,
        "samples": len(dataset),
        "confidence_threshold": arguments.confidence_threshold,
        "match_radius_heatmap_pixels": arguments.match_radius,
        "match_radius_input_pixels": arguments.match_radius * 4,
        "device": str(device),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

