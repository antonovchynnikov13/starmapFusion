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
    from starmap_fusion.training.engine import _match_points
    from starmap_fusion.training.losses import center_focal_loss
except ImportError:
    torch = None
    decode_heatmap = None
    StarHeatmapDetector = None
    _match_points = None
    center_focal_loss = None


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

    def test_point_matching_accepts_mixed_precision_predictions(self) -> None:
        """Validation should compare AMP predictions with float32 targets safely."""

        assert torch is not None
        assert _match_points is not None
        predictions = torch.tensor([[5.0, 3.0, 0.9]], dtype=torch.float16)
        targets = torch.tensor([[5.0, 3.0]], dtype=torch.float32)
        matched = _match_points(predictions, targets, radius=2.0)
        self.assertEqual(matched, (1, 0, 0))

    def test_focal_loss_is_finite_for_float16_extreme_logits(self) -> None:
        """AMP logits close to zero or one should not produce NaN loss."""

        assert torch is not None
        assert center_focal_loss is not None
        logits = torch.tensor([[[[20.0, -20.0]]]], dtype=torch.float16)
        targets = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
        loss = center_focal_loss(logits, targets)
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == "__main__":
    unittest.main()
