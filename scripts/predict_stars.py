#!/usr/bin/env python3
"""Extract star-center coordinates from one or more images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from starmap_fusion.inference.predictor import (  # noqa: E402
    StarPredictor,
    detect_with_sep,
    fuse_with_sep,
    load_inference_model,
    save_predictions_json,
    save_visualization,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the inference command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--merge-radius", type=float, default=5.0)
    parser.add_argument("--mask-left-pixels", type=int, default=0)
    parser.add_argument("--use-sep", action="store_true")
    parser.add_argument(
        "--include-sep-only",
        action="store_true",
        help="Include unmatched SEP sources; disabled by default to suppress noise.",
    )
    parser.add_argument("--sep-threshold-sigma", type=float, default=3.0)
    parser.add_argument("--sep-match-radius", type=float, default=5.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def resolve_device(requested_device: str) -> torch.device:
    """Resolve automatic device selection and validate CUDA requests."""

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested_device)


def main() -> int:
    """Load the model, predict each image, and save JSON plus visualization."""

    arguments = build_parser().parse_args()
    device = resolve_device(arguments.device)
    model, config, _ = load_inference_model(arguments.checkpoint, device)
    predictor = StarPredictor(model, config, device)
    threshold = (
        config.confidence_threshold
        if arguments.confidence_threshold is None
        else arguments.confidence_threshold
    )
    for image_path in arguments.input:
        with Image.open(image_path) as image:
            detections = predictor.predict(
                image,
                confidence_threshold=threshold,
                overlap=arguments.overlap,
                merge_radius=arguments.merge_radius,
                mask_left_pixels=arguments.mask_left_pixels,
            )
            if arguments.use_sep:
                sep_coordinates = detect_with_sep(
                    image,
                    threshold_sigma=arguments.sep_threshold_sigma,
                )
                detections = fuse_with_sep(
                    detections,
                    sep_coordinates,
                    match_radius=arguments.sep_match_radius,
                    include_sep_only=arguments.include_sep_only,
                )
                detections = [
                    detection
                    for detection in detections
                    if detection.x >= arguments.mask_left_pixels
                ]
            stem = image_path.stem
            save_predictions_json(
                arguments.output_dir / f"{stem}.json",
                image_path,
                image.size,
                detections,
                threshold,
            )
            save_visualization(
                arguments.output_dir / f"{stem}_detected.png",
                image,
                detections,
            )
        print(f"{image_path.name}: {len(detections)} stars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
