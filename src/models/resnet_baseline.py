"""
resnet_baseline.py — ResNet/EfficientNet-based binary hallucination-risk classifier.

Wraps a timm backbone (default: resnet50) with a single-logit classification
head for binary hallucination-risk classification of Hebrew PDF page images.
"""

import torch
import torch.nn as nn
import timm


class ResNetClassifier(nn.Module):
    """
    ResNet (or EfficientNet) backbone with a single-logit output for binary
    classification.

    Input:  FloatTensor[B, C, H, W]  (C=1 or 3, H=W=224)
    Output: logits FloatTensor[B, 1]

    Grayscale input (C=1) is handled by repeating the channel 3×.
    The backbone is loaded via timm with num_classes=0 to get a feature
    extractor, and a linear head is attached on top.
    """

    num_classes: int = 1

    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        num_classes: int = 1,
    ) -> None:
        """
        Args:
            model_name: timm model identifier, e.g. 'resnet50' or
                        'efficientnet_b0'.
            pretrained: If True, load ImageNet-pretrained weights.
            num_classes: Output logit count (always 1 for binary BCE training).
        """
        super().__init__()
        self.num_classes = num_classes

        # num_classes=0 returns a feature extractor (no classification head)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        feature_dim: int = self.backbone.num_features
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor[B, C, H, W] — C may be 1 (grayscale) or 3 (RGB).

        Returns:
            logits: FloatTensor[B, 1]
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        features = self.backbone(x)   # [B, feature_dim]
        logits = self.head(features)  # [B, num_classes]
        return logits
