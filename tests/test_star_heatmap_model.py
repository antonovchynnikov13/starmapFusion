"""Shape and decoding tests for the heatmap detector."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import torch

    from starmap_fusion.inference.heatmap import decode_heatmap
    from starmap_fusion.models.star_heatmap import StarHeatmapDetector
except ImportError:
    torch = None
    decode_heatmap = None
    StarHeatmapDetector = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the local environment")
class StarHeatmapModelTest(unittest.TestCase):
    """Validate the model output stride and peak decoder."""

    def test_model_returns_stride_four_heatmap(self) -> None:
        """A 128-pixel input should produce a 32-pixel heatmap."""

        assert torch is not None
        assert StarHeatmapDetector is not None
        model = StarHeatmapDetector(pretrained_backbone=False).eval()
        with torch.inference_mode():
            output = model(torch.zeros((1, 3, 128, 128)))
        self.assertEqual(tuple(output.shape), (1, 1, 32, 32))

    def test_decoder_returns_xy_score_coordinates(self) -> None:
        """The decoder should retain local maxima above the threshold."""

        assert torch is not None
        assert decode_heatmap is not None
        logits = torch.full((1, 1, 8, 8), -10.0)
        logits[0, 0, 3, 5] = 10.0
        detections = decode_heatmap(logits, confidence_threshold=0.5)
        self.assertEqual(detections[0].shape, (1, 3))
        self.assertEqual(detections[0][0, :2].tolist(), [5.0, 3.0])


if __name__ == "__main__":
    unittest.main()
