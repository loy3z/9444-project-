"""EuroSAT ResNet50 model."""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def create_resnet50(
    num_classes: int = 10,
    *,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build ResNet50 and change the last layer for EuroSAT."""

    if num_classes <= 1:
        raise ValueError(
            f"num_classes must be greater than 1, got {num_classes}."
        )

    # baseline 用 ImageNet pretrained weights
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    # EuroSAT 有 10 类，所以换掉原来的 ImageNet classifier
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)

    return model
