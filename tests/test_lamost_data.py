"""Tests for LAMOST indexing and center conversion."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from starmap_fusion.data.lamost import (  # noqa: E402
    build_dataset_manifests,
    parse_gt_norm,
)


class LamostDataTest(unittest.TestCase):
    """Validate parsing, splitting, and deterministic subsetting."""

    def test_parse_gt_norm_converts_ellipses_to_centers(self) -> None:
        """The parser should retain ellipse fields and expose center coordinates."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            annotation_path = Path(temporary_directory) / "sample.txt"
            annotation_path.write_text(
                "2\n10.5 20.25 3.0 2.0 -0.5\n30.0 40.0 4.0 1.5 1.2\n",
                encoding="utf-8",
            )
            ellipses = parse_gt_norm(annotation_path)

        self.assertEqual(len(ellipses), 2)
        self.assertEqual(ellipses[0].center(), (10.5, 20.25))
        self.assertEqual(ellipses[1].center(), (30.0, 40.0))

    def test_parse_gt_norm_rejects_incorrect_count(self) -> None:
        """A mismatched header count should fail before training starts."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            annotation_path = Path(temporary_directory) / "sample.txt"
            annotation_path.write_text("2\n10 20 3 2 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Declared 2 objects"):
                parse_gt_norm(annotation_path)

    def test_build_manifests_uses_official_splits_and_limits(self) -> None:
        """Preparation should map dev to validation and avoid copying images."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "LAMOST"
            output_dir = root / "processed"
            for split in ("train", "dev", "test"):
                (dataset_root / split / "images").mkdir(parents=True)
                (dataset_root / split / "gt_norm").mkdir(parents=True)
                for index in range(3):
                    sample_id = f"{split}_{index}"
                    (dataset_root / split / "images" / f"{sample_id}.png").write_bytes(b"png")
                    (dataset_root / split / "gt_norm" / f"{sample_id}.txt").write_text(
                        "1\n10 20 3 2 0\n",
                        encoding="utf-8",
                    )

            counts = build_dataset_manifests(
                dataset_root=dataset_root,
                output_dir=output_dir,
                limits={"train": 2, "dev": 1, "test": 1},
                seed=7,
            )
            validation_record = json.loads(
                (output_dir / "validation.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )

        self.assertEqual(counts, {"train": 2, "validation": 1, "test": 1})
        self.assertEqual(validation_record["split"], "dev")
        self.assertEqual(validation_record["centers"], [[10.0, 20.0]])


if __name__ == "__main__":
    unittest.main()

