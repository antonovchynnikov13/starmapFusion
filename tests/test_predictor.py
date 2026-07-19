"""Tests for inference post-processing helpers."""

import unittest

from starmap_fusion.inference.predictor import (
    StarDetection,
    _tile_starts,
    fuse_with_sep,
    merge_detections,
)


class PredictorTest(unittest.TestCase):
    """Verify tiling and coordinate post-processing behavior."""

    def test_tile_starts_cover_far_edge(self) -> None:
        """The final tile must be aligned with the far image edge."""

        self.assertEqual(_tile_starts(2500, 1024, 256), (0, 768, 1476))

    def test_merge_detections_keeps_highest_confidence(self) -> None:
        """Nearby tile duplicates must collapse to the strongest point."""

        detections = [
            StarDetection(10.0, 10.0, 0.7),
            StarDetection(11.0, 11.0, 0.9),
            StarDetection(30.0, 30.0, 0.8),
        ]
        self.assertEqual(
            merge_detections(detections, merge_radius=3.0),
            [
                StarDetection(11.0, 11.0, 0.9),
                StarDetection(30.0, 30.0, 0.8),
            ],
        )

    def test_sep_only_candidates_are_opt_in(self) -> None:
        """SEP must not add noisy candidates unless explicitly requested."""

        ml_detections = [StarDetection(10.0, 10.0, 0.6)]
        sep_coordinates = [(11.0, 9.0), (100.0, 100.0)]
        confirmed = fuse_with_sep(ml_detections, sep_coordinates, match_radius=3.0)
        expanded = fuse_with_sep(
            ml_detections,
            sep_coordinates,
            match_radius=3.0,
            include_sep_only=True,
        )
        self.assertEqual(confirmed, [StarDetection(10.5, 9.5, 0.8)])
        self.assertEqual(len(expanded), 2)


if __name__ == "__main__":
    unittest.main()
