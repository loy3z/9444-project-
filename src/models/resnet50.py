"""ResNet50 model definition for EuroSAT land-cover classification."""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def create_resnet50(
    num_classes: int = 10,
    *,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Create a ResNet50 classifier with a task-specific output layer.

    Args:
        num_classes: Number of EuroSAT classes.
        pretrained: Load the default ImageNet-1K weights when ``True``.
        freeze_backbone: Freeze all layers except the final classifier.
            The project baseline leaves the full network trainable.
    """

    if num_classes <= 1:
        raise ValueError(
            f"num_classes must be greater than 1, got {num_classes}."
        )

    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)

    return model
