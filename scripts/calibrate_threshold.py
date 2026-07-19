#!/usr/bin/env python3
"""Calibrate the star confidence threshold on the validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from starmap_fusion.data.heatmap_dataset import (  # noqa: E402
    LamostHeatmapDataset,
    collate_heatmap_batch,
)
from starmap_fusion.inference.heatmap import decode_heatmap  # noqa: E402
from starmap_fusion.inference.predictor import load_inference_model  # noqa: E402
from starmap_fusion.training.engine import _match_points  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the threshold calibration command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-stop", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--minimum-precision", type=float, default=0.80)
    parser.add_argument("--match-radius", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def _thresholds(start: float, stop: float, step: float) -> list[float]:
    """Build an inclusive stable floating-point threshold sequence."""

    count = int(round((stop - start) / step))
    return [round(start + index * step, 6) for index in range(count + 1)]


def main() -> int:
    """Run inference once, sweep thresholds, and save the selected operating point."""

    arguments = build_parser().parse_args()
    if arguments.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
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
    cached: list[tuple[torch.Tensor, list[torch.Tensor]]] = []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Caching validation logits"):
            logits = model(batch["images"].to(device, non_blocking=True))
            cached.append((logits.cpu().to(torch.float16), batch["centers"]))

    sweep: list[dict[str, Any]] = []
    for threshold in _thresholds(
        arguments.threshold_start,
        arguments.threshold_stop,
        arguments.threshold_step,
    ):
        true_positives = false_positives = false_negatives = 0
        for logits, centers_batch in cached:
            predictions_batch = decode_heatmap(logits, confidence_threshold=threshold)
            for predictions, centers in zip(predictions_batch, centers_batch, strict=True):
                matched = _match_points(predictions, centers, arguments.match_radius)
                true_positives += matched[0]
                false_positives += matched[1]
                false_negatives += matched[2]
        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, true_positives + false_negatives)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        sweep.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            }
        )

    eligible = [row for row in sweep if row["precision"] >= arguments.minimum_precision]
    selected_for_recall = max(eligible, key=lambda row: row["recall"]) if eligible else None
    selected_for_f1 = max(sweep, key=lambda row: row["f1"])
    payload = {
        "checkpoint": arguments.checkpoint.name,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "samples": len(dataset),
        "minimum_precision": arguments.minimum_precision,
        "selected_for_recall": selected_for_recall,
        "selected_for_f1": selected_for_f1,
        "sweep": sweep,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "sweep"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
