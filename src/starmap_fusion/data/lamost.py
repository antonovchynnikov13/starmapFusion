"""LAMOST-DET indexing and ellipse-to-center conversion utilities."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


OFFICIAL_SPLITS: tuple[str, ...] = ("train", "dev", "test")
IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


@dataclass(frozen=True, slots=True)
class LamostEllipse:
    """One normalized LAMOST ellipse annotation."""

    center_x: float
    center_y: float
    semi_major: float
    semi_minor: float
    angle: float

    def center(self) -> tuple[float, float]:
        """Return the ellipse center used by the point detector."""

        return self.center_x, self.center_y


@dataclass(frozen=True, slots=True)
class LamostManifestRecord:
    """One image and its point/ellipse annotations in a JSONL manifest."""

    sample_id: str
    split: str
    image_path: str
    annotation_path: str
    centers: tuple[tuple[float, float], ...]
    ellipses: tuple[LamostEllipse, ...]

    def to_json(self) -> str:
        """Serialize the record as one compact JSON line."""

        payload = asdict(self)
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def parse_gt_norm(annotation_path: Path) -> tuple[LamostEllipse, ...]:
    """Parse a LAMOST ``gt_norm`` file and validate its declared object count."""

    lines = [line.strip() for line in annotation_path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty annotation file: {annotation_path}")

    try:
        expected_count = int(lines[0])
    except ValueError as error:
        raise ValueError(f"Invalid object count in {annotation_path}: {lines[0]!r}") from error

    ellipses: list[LamostEllipse] = []
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"Expected 5 ellipse values in {annotation_path}:{line_number}, "
                f"found {len(fields)}"
            )
        try:
            values = tuple(float(field) for field in fields)
        except ValueError as error:
            raise ValueError(
                f"Non-numeric ellipse value in {annotation_path}:{line_number}"
            ) from error
        ellipses.append(LamostEllipse(*values))

    if len(ellipses) != expected_count:
        raise ValueError(
            f"Declared {expected_count} objects in {annotation_path}, "
            f"parsed {len(ellipses)}"
        )
    return tuple(ellipses)


def discover_samples(dataset_root: Path, split: str) -> tuple[tuple[Path, Path], ...]:
    """Match images with ``gt_norm`` annotations for one official split."""

    if split not in OFFICIAL_SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {OFFICIAL_SPLITS}")

    image_dir = dataset_root / split / "images"
    annotation_dir = dataset_root / split / "gt_norm"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory does not exist: {annotation_dir}")

    images = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    annotations = {path.stem: path for path in annotation_dir.glob("*.txt")}
    missing_annotations = sorted(images.keys() - annotations.keys())
    missing_images = sorted(annotations.keys() - images.keys())
    if missing_annotations or missing_images:
        details = (
            f"missing annotations={len(missing_annotations)}, "
            f"missing images={len(missing_images)}"
        )
        raise ValueError(f"Unmatched files in split {split!r}: {details}")

    return tuple((images[sample_id], annotations[sample_id]) for sample_id in sorted(images))


def select_samples(
    samples: Sequence[tuple[Path, Path]],
    max_samples: int | None,
    seed: int,
) -> tuple[tuple[Path, Path], ...]:
    """Select a deterministic random subset without copying source files."""

    if max_samples is None or max_samples >= len(samples):
        return tuple(samples)
    if max_samples <= 0:
        raise ValueError("max_samples must be a positive integer or None")

    random_generator = random.Random(seed)
    selected_indices = sorted(random_generator.sample(range(len(samples)), max_samples))
    return tuple(samples[index] for index in selected_indices)


def iter_manifest_records(
    dataset_root: Path,
    split: str,
    max_samples: int | None = None,
    seed: int = 42,
) -> Iterator[LamostManifestRecord]:
    """Yield validated records for an official LAMOST split."""

    samples = select_samples(discover_samples(dataset_root, split), max_samples, seed)
    for image_path, annotation_path in samples:
        ellipses = parse_gt_norm(annotation_path)
        yield LamostManifestRecord(
            sample_id=image_path.stem,
            split=split,
            image_path=str(image_path.relative_to(dataset_root)),
            annotation_path=str(annotation_path.relative_to(dataset_root)),
            centers=tuple(ellipse.center() for ellipse in ellipses),
            ellipses=ellipses,
        )


def write_manifest(records: Iterable[LamostManifestRecord], output_path: Path) -> int:
    """Write records to JSONL and return the number of written samples."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(record.to_json())
            output_file.write("\n")
            count += 1
    return count


def build_dataset_manifests(
    dataset_root: Path,
    output_dir: Path,
    limits: dict[str, int | None] | None = None,
    seed: int = 42,
) -> dict[str, int]:
    """Build train/validation/test manifests from the official dataset splits.

    LAMOST calls its validation split ``dev``. The output manifest is named
    ``validation.jsonl`` while each record keeps ``split='dev'`` so the source
    provenance remains explicit.
    """

    resolved_root = dataset_root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"LAMOST dataset root does not exist: {resolved_root}")

    split_names = {"train": "train", "dev": "validation", "test": "test"}
    requested_limits = limits or {}
    counts: dict[str, int] = {}
    for split, manifest_name in split_names.items():
        records = iter_manifest_records(
            dataset_root=resolved_root,
            split=split,
            max_samples=requested_limits.get(split),
            seed=seed,
        )
        counts[manifest_name] = write_manifest(records, output_dir / f"{manifest_name}.jsonl")

    metadata = {
        "dataset_root": str(resolved_root),
        "seed": seed,
        "limits": requested_limits,
        "counts": counts,
        "split_mapping": split_names,
        "annotation_format": "center_x center_y semi_major semi_minor angle",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return counts

