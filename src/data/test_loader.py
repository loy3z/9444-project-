from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import EuroSATDataset
from src.data.transforms import get_pretrained_transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "EuroSAT_RGB"
)

SPLIT_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "eurosat_split_seed42.csv"
)


def main() -> None:
    train_transform, evaluation_transform = (
        get_pretrained_transforms(image_size=224)
    )

    train_dataset = EuroSATDataset(
        data_root=DATA_ROOT,
        split_csv=SPLIT_CSV,
        split="train",
        transform=train_transform,
    )

    validation_dataset = EuroSATDataset(
        data_root=DATA_ROOT,
        split_csv=SPLIT_CSV,
        split="val",
        transform=evaluation_transform,
    )

    test_dataset = EuroSATDataset(
        data_root=DATA_ROOT,
        split_csv=SPLIT_CSV,
        split="test",
        transform=evaluation_transform,
    )

    assert (
        train_dataset.class_to_idx
        == validation_dataset.class_to_idx
        == test_dataset.class_to_idx
    ), "Class mappings do not match across splits."

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    images, labels = next(iter(train_loader))

    assert images.ndim == 4
    assert images.shape[1:] == (3, 224, 224)
    assert labels.ndim == 1
    assert images.shape[0] == labels.shape[0]

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(validation_loader)}")
    print(f"Test batches: {len(test_loader)}")

    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Label dtype: {labels.dtype}")
    print(
        f"Label range: "
        f"{labels.min().item()} to "
        f"{labels.max().item()}"
    )

    print(f"Classes: {train_dataset.classes}")
    print(
        f"Class mapping: "
        f"{train_dataset.class_to_idx}"
    )


if __name__ == "__main__":
    main()