#!/usr/bin/env python3
"""Create LAMOST train/validation/test manifests with point annotations."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from starmap_fusion.data.lamost import build_dataset_manifests  # noqa: E402


def positive_integer(value: str) -> int:
    """Parse a positive command-line integer."""

    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed_value


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    default_root = os.environ.get("LAMOST_ROOT", str(PROJECT_ROOT / "data" / "lamost"))
    parser = argparse.ArgumentParser(
        description="Index official LAMOST splits and convert ellipse annotations to centers."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path(default_root))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "lamost",
    )
    parser.add_argument("--max-train", type=positive_integer)
    parser.add_argument("--max-validation", type=positive_integer)
    parser.add_argument("--max-test", type=positive_integer)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    """Run dataset preparation and print the resulting split sizes."""

    arguments = build_parser().parse_args()
    limits = {
        "train": arguments.max_train,
        "dev": arguments.max_validation,
        "test": arguments.max_test,
    }
    counts = build_dataset_manifests(
        dataset_root=arguments.dataset_root,
        output_dir=arguments.output_dir,
        limits=limits,
        seed=arguments.seed,
    )
    for split_name, count in counts.items():
        print(f"{split_name}: {count} samples")
    print(f"Manifests written to: {arguments.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

