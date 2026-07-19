#!/usr/bin/env python3
"""Export only manifest-referenced LAMOST images for Colab training."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Iterator, TypeVar


ItemType = TypeVar("ItemType")


def iter_with_progress(
    items: Iterable[ItemType],
    description: str,
    total: int,
) -> Iterator[ItemType]:
    """Yield items and print dependency-free periodic progress updates."""

    report_interval = max(1, total // 20)
    for index, item in enumerate(items, start=1):
        if index == 1 or index % report_interval == 0 or index == total:
            print(f"{description}: {index}/{total}")
        yield item


def copy_manifest_images(
    manifest_path: Path,
    dataset_root: Path,
    output_root: Path,
) -> int:
    """Copy images referenced by one manifest while preserving relative paths."""

    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in iter_with_progress(records, manifest_path.stem, len(records)):
        relative_image_path = Path(record["image_path"])
        source_path = dataset_root / relative_image_path
        destination_path = output_root / relative_image_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source image: {source_path}")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
    return len(records)


def build_parser() -> argparse.ArgumentParser:
    """Create the subset export command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    """Copy selected images and manifests into a portable directory."""

    arguments = build_parser().parse_args()
    output_manifest_dir = arguments.output_root / "manifests"
    output_manifest_dir.mkdir(parents=True, exist_ok=True)
    total_images = 0
    for manifest_name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        source_manifest = arguments.manifest_dir / manifest_name
        total_images += copy_manifest_images(
            manifest_path=source_manifest,
            dataset_root=arguments.dataset_root,
            output_root=arguments.output_root,
        )
        shutil.copy2(source_manifest, output_manifest_dir / manifest_name)
    metadata_path = arguments.manifest_dir / "metadata.json"
    if metadata_path.is_file():
        shutil.copy2(metadata_path, output_manifest_dir / "metadata.json")
    print(f"Exported {total_images} images to: {arguments.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
