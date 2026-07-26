from __future__ import annotations

from torchvision import transforms
from torchvision.transforms import Compose


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_cnn_transforms() -> tuple[Compose, Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    return train_transform, evaluation_transform


def get_pretrained_transforms(
    image_size: int = 224,
) -> tuple[Compose, Compose]:
    if image_size <= 0:
        raise ValueError(
            "image_size must be greater than 0"
        )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=image_size,
                scale=(0.90, 1.0),
                ratio=(0.90, 1.10),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(
                brightness=0.20,
                contrast=0.20,
                saturation=0.15,
                hue=0.03,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    return train_transform, evaluation_transform