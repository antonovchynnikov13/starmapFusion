"""ResNet-50, FPN, and heatmap head architecture."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import Tensor, nn
from torch.nn import functional as functional
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork


class HeatmapHead(nn.Module):
    """Fuse all FPN levels and predict one star-center heatmap."""

    def __init__(self, pyramid_channels: int, hidden_channels: int) -> None:
        """Initialize the multi-scale fusion and prediction layers."""

        super().__init__()
        fused_channels = pyramid_channels * 4
        self.layers = nn.Sequential(
            nn.Conv2d(fused_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        nn.init.constant_(self.layers[-1].bias, -4.6)

    def forward(self, pyramid: OrderedDict[str, Tensor]) -> Tensor:
        """Upsample P3-P5 to P2, fuse them, and return heatmap logits."""

        target_size = pyramid["p2"].shape[-2:]
        aligned_features = [pyramid["p2"]]
        for level_name in ("p3", "p4", "p5"):
            aligned_features.append(
                functional.interpolate(
                    pyramid[level_name],
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return self.layers(torch.cat(aligned_features, dim=1))


class StarHeatmapDetector(nn.Module):
    """Detect star centers with a ResNet-50 backbone and FPN feature fusion."""

    output_stride: int = 4

    def __init__(
        self,
        pretrained_backbone: bool = True,
        pyramid_channels: int = 128,
        head_channels: int = 128,
    ) -> None:
        """Initialize the backbone, feature pyramid, and one-class heatmap head."""

        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet50(weights=weights)
        self.backbone = create_feature_extractor(
            backbone,
            return_nodes={
                "layer1": "c2",
                "layer2": "c3",
                "layer3": "c4",
                "layer4": "c5",
            },
        )
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=pyramid_channels,
        )
        self.head = HeatmapHead(pyramid_channels, head_channels)

    def forward(self, images: Tensor) -> Tensor:
        """Return stride-4 star-center heatmap logits for an image batch."""

        backbone_features = self.backbone(images)
        fpn_input = OrderedDict(
            (level_name, backbone_features[level_name])
            for level_name in ("c2", "c3", "c4", "c5")
        )
        raw_pyramid = self.fpn(fpn_input)
        pyramid = OrderedDict(
            (output_name, raw_pyramid[input_name])
            for input_name, output_name in zip(
                ("c2", "c3", "c4", "c5"),
                ("p2", "p3", "p4", "p5"),
                strict=True,
            )
        )
        return self.head(pyramid)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of parameters updated during training."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

