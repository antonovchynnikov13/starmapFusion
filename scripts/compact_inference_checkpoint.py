#!/usr/bin/env python3
"""Store floating checkpoint tensors in FP16 to reduce distribution size."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def build_parser() -> argparse.ArgumentParser:
    """Create the checkpoint compaction command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _compact(value: Any) -> Any:
    """Recursively convert floating tensors to half-precision storage."""

    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value.to(torch.float16)
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_compact(item) for item in value)
    return value


def main() -> int:
    """Load, compact, and save an inference checkpoint."""

    arguments = build_parser().parse_args()
    checkpoint = torch.load(arguments.input, map_location="cpu", weights_only=False)
    compact_checkpoint = _compact(checkpoint)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact_checkpoint, arguments.output)
    print(f"Saved compact checkpoint: {arguments.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
