#!/usr/bin/env python3
"""Build tracked model-card artifacts from training outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Sequence


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _polyline_points(
    values: Sequence[float],
    width: int,
    height: int,
    padding: int,
    minimum: float,
    maximum: float,
) -> str:
    """Map values to SVG polyline coordinates."""

    value_range = max(1e-12, maximum - minimum)
    plot_width = width - 2 * padding
    plot_height = height - 2 * padding
    points: list[str] = []
    for index, value in enumerate(values):
        x_coordinate = padding + index * plot_width / max(1, len(values) - 1)
        y_coordinate = padding + (maximum - value) * plot_height / value_range
        points.append(f"{x_coordinate:.2f},{y_coordinate:.2f}")
    return " ".join(points)


def write_svg_chart(
    output_path: Path,
    title: str,
    series: dict[str, Sequence[float]],
    colors: dict[str, str],
) -> None:
    """Write a dependency-free multi-series SVG line chart."""

    width, height, padding = 900, 520, 70
    all_values = [value for values in series.values() for value in values]
    minimum, maximum = min(all_values), max(all_values)
    margin = max(1e-6, (maximum - minimum) * 0.08)
    minimum -= margin
    maximum += margin
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" '
        f'font-family="sans-serif" font-size="22">{title}</text>',
        f'<line x1="{padding}" y1="{height-padding}" x2="{width-padding}" '
        f'y2="{height-padding}" stroke="#333"/>',
        f'<line x1="{padding}" y1="{padding}" x2="{padding}" '
        f'y2="{height-padding}" stroke="#333"/>',
        f'<text x="{width / 2}" y="{height-18}" text-anchor="middle" '
        'font-family="sans-serif">Epoch</text>',
        f'<text x="12" y="{height / 2}" font-family="sans-serif" '
        f'transform="rotate(-90 12 {height / 2})">Value</text>',
    ]
    for tick_index in range(6):
        ratio = tick_index / 5
        y_coordinate = height - padding - ratio * (height - 2 * padding)
        value = minimum + ratio * (maximum - minimum)
        elements.extend(
            [
                f'<line x1="{padding}" y1="{y_coordinate:.2f}" '
                f'x2="{width-padding}" y2="{y_coordinate:.2f}" stroke="#ddd"/>',
                f'<text x="{padding-8}" y="{y_coordinate+4:.2f}" text-anchor="end" '
                f'font-family="monospace" font-size="12">{value:.4f}</text>',
            ]
        )
    for index, (name, values) in enumerate(series.items()):
        points = _polyline_points(values, width, height, padding, minimum, maximum)
        color = colors[name]
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        legend_y = 55 + index * 22
        elements.append(
            f'<text x="{width-220}" y="{legend_y}" font-family="sans-serif" '
            f'font-size="14" fill="{color}">{name}</text>'
        )
    elements.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _format_test_metrics(metrics: dict[str, object] | None) -> str:
    """Format independent test metrics for the generated model card."""

    if metrics is None:
        return ""
    return f"""
## Independent LAMOST test

- Samples: {int(metrics['samples'])}
- Precision: {float(metrics['precision']):.4f}
- Recall: {float(metrics['recall']):.4f}
- F1: {float(metrics['f1']):.4f}
- Confidence threshold: {float(metrics['confidence_threshold']):.2f}
"""


def build_parser() -> argparse.ArgumentParser:
    """Create the model-release builder interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--inference-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-metrics", type=Path)
    parser.add_argument("--release-tag", default="star-detector-v1.0.0")
    return parser


def main() -> int:
    """Generate tracked metadata, plots, checksums, and colleague instructions."""

    arguments = build_parser().parse_args()
    history = json.loads(arguments.history.read_text(encoding="utf-8"))
    test_metrics = (
        json.loads(arguments.test_metrics.read_text(encoding="utf-8"))
        if arguments.test_metrics is not None
        else None
    )
    best_epoch = max(history, key=lambda row: row["f1"])
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(arguments.history, arguments.output_dir / "training_history.json")
    model_config = {
        "format_version": 1,
        "architecture": "StarHeatmapDetector",
        "backbone": "ResNet-50",
        "feature_pyramid": "FPN P2-P5",
        "head": "Gaussian center heatmap",
        "class_names": ["star"],
        "input_size": [1024, 1024],
        "output_stride": 4,
        "pyramid_channels": 128,
        "head_channels": 128,
        "confidence_threshold": 0.3,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "best_validation_epoch": int(best_epoch["epoch"]),
        "best_validation_metrics": best_epoch,
        "test_metrics": test_metrics,
        "weights_file": arguments.inference_model.name,
        "weights_sha256": sha256(arguments.inference_model),
        "release_tag": arguments.release_tag,
    }
    (arguments.output_dir / "model_config.json").write_text(
        json.dumps(model_config, indent=2) + "\n",
        encoding="utf-8",
    )
    write_svg_chart(
        arguments.output_dir / "plots" / "losses.svg",
        "Training and validation loss",
        {
            "train_loss": [row["train_loss"] for row in history],
            "validation_loss": [row["validation_loss"] for row in history],
        },
        {"train_loss": "#2563eb", "validation_loss": "#dc2626"},
    )
    write_svg_chart(
        arguments.output_dir / "plots" / "metrics.svg",
        "Validation metrics",
        {
            "precision": [row["precision"] for row in history],
            "recall": [row["recall"] for row in history],
            "f1": [row["f1"] for row in history],
        },
        {"precision": "#7c3aed", "recall": "#059669", "f1": "#ea580c"},
    )
    write_svg_chart(
        arguments.output_dir / "plots" / "learning_rate.svg",
        "Cosine learning-rate schedule",
        {"learning_rate": [row["learning_rate"] for row in history]},
        {"learning_rate": "#0891b2"},
    )
    readme = f"""# Star detector v1.0.0

ResNet-50 + FPN heatmap model that returns star-center pixel coordinates.

## Validation

- Best epoch: {int(best_epoch['epoch'])}
- Precision: {best_epoch['precision']:.4f}
- Recall: {best_epoch['recall']:.4f}
- F1: {best_epoch['f1']:.4f}
- Matching tolerance: 2 heatmap pixels (8 input pixels)

The model was selected by validation F1. Test metrics are stored in
`test_metrics.json` after the independent test evaluation.
{_format_test_metrics(test_metrics)}

## Weights

The inference-ready weights are tracked at
`artifacts/star_detector_resnet50_fpn_v1/weights/{arguments.inference_model.name}`.
They use FP16 storage and are loaded into the runtime model automatically.

SHA-256: `{model_config['weights_sha256']}`

## Install

```bash
python3 -m pip install -r requirements-inference.txt
```

## Predict coordinates

```bash
python3 scripts/predict_stars.py \\
  --checkpoint artifacts/star_detector_resnet50_fpn_v1/weights/{arguments.inference_model.name} \\
  --input photo.jpg \\
  --output-dir predictions
```

The command creates `photo.json` with `x`, `y`, and confidence values, plus
`photo_detected.png`. Use the `coordinates` field when only coordinate pairs are
required.

## Known limitation

The model was trained only on LAMOST-DET. Performance on cloudy, green-tinted,
or structurally occluded camera imagery must be evaluated separately.
"""
    (arguments.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Release metadata written to: {arguments.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
