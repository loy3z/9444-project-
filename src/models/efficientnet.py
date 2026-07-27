"""EuroSAT EfficientNet model."""

from __future__ import annotations

from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def create_efficientnet(
    num_classes: int = 10,
    *,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build EfficientNet-B0 and change the last layer for EuroSAT."""

    if num_classes <= 1:
        raise ValueError(
            f"num_classes must be greater than 1, got {num_classes}."
        )

    # baseline 用 ImageNet pretrained weights
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    # EfficientNet 的分类头是 classifier (Sequential)，包含 Dropout 和 Linear
    # 替换 classifier[1] 可以保留前面的 Dropout 层
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, num_classes)

    return model