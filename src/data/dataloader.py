from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import EuroSATDataset
from src.data.transforms import (
    get_cnn_transforms,
    get_pretrained_transforms,
)


# ============================================================
# Default paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "EuroSAT_RGB"
)

DEFAULT_SPLIT_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "eurosat_split_seed42.csv"
)


ModelType = Literal["cnn", "pretrained"]


# ============================================================
# Return structure
# ============================================================

@dataclass
class DataLoaderBundle:
    """
    Container for EuroSAT datasets, DataLoaders and class metadata.
    """

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    train_dataset: EuroSATDataset
    val_dataset: EuroSATDataset
    test_dataset: EuroSATDataset

    classes: list[str]
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]


# ============================================================
# Reproducibility utilities
# ============================================================

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy and PyTorch.

    Note:
        Full deterministic training may reduce performance.
        This function mainly ensures consistent data loading
        and random augmentation behaviour.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative.")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """
    Set a different but reproducible random seed for each
    DataLoader worker.

    PyTorch provides a worker-specific initial seed through
    torch.initial_seed().
    """

    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# DataLoader factory
# ============================================================

def create_dataloaders(
    batch_size: int = 32,
    image_size: int = 224,
    model_type: ModelType = "pretrained",
    num_workers: int = 0,
    seed: int = 42,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    split_csv: str | Path = DEFAULT_SPLIT_CSV,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    drop_last: bool = False,
) -> DataLoaderBundle:
    """
    Create reproducible EuroSAT train, validation and test DataLoaders.

    Args:
        batch_size:
            Number of images in each batch.

        image_size:
            Image size used by pretrained models.
            For example, 224 produces images of shape
            [batch_size, 3, 224, 224].

            This parameter is ignored when model_type="cnn"
            if the CNN transform keeps the original 64x64 size.

        model_type:
            "cnn":
                Use transforms for the custom CNN.

            "pretrained":
                Use transforms for pretrained models such as
                ResNet50, ViT and EfficientNet.

        num_workers:
            Number of subprocesses used to load data.

            Recommended:
                Windows initial testing: 0
                Linux or stable Windows environment: 2-4

        seed:
            Random seed used for DataLoader shuffling and workers.

        data_root:
            Path to the extracted EuroSAT_RGB directory.

        split_csv:
            Path to the fixed train/val/test split CSV.

        pin_memory:
            Whether to use pinned CPU memory.

            If None, it is automatically enabled when CUDA
            is available.

        persistent_workers:
            Whether DataLoader workers remain alive between epochs.

            If None, it is enabled when num_workers > 0.

        drop_last:
            Whether to discard the final incomplete training batch.

            Normally False for this project.

    Returns:
        DataLoaderBundle containing:
            - train_loader
            - val_loader
            - test_loader
            - datasets
            - class metadata
    """

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if batch_size <= 0:
        raise ValueError(
            f"batch_size must be greater than 0, got {batch_size}."
        )

    if image_size <= 0:
        raise ValueError(
            f"image_size must be greater than 0, got {image_size}."
        )

    if num_workers < 0:
        raise ValueError(
            f"num_workers cannot be negative, got {num_workers}."
        )

    if seed < 0:
        raise ValueError(
            f"seed must be non-negative, got {seed}."
        )

    if model_type not in {"cnn", "pretrained"}:
        raise ValueError(
            "model_type must be either 'cnn' or 'pretrained', "
            f"got {model_type!r}."
        )

    data_root = Path(data_root)
    split_csv = Path(split_csv)

    if not data_root.exists():
        raise FileNotFoundError(
            f"EuroSAT data directory does not exist:\n{data_root}"
        )

    if not data_root.is_dir():
        raise NotADirectoryError(
            f"data_root is not a directory:\n{data_root}"
        )

    if not split_csv.exists():
        raise FileNotFoundError(
            f"Split CSV does not exist:\n{split_csv}"
        )

    if not split_csv.is_file():
        raise FileNotFoundError(
            f"split_csv is not a file:\n{split_csv}"
        )

    # --------------------------------------------------------
    # Set random seed
    # --------------------------------------------------------

    set_seed(seed)

    # Generator controls train DataLoader shuffling.
    generator = torch.Generator()
    generator.manual_seed(seed)

    # --------------------------------------------------------
    # Select transforms
    # --------------------------------------------------------

    if model_type == "cnn":
        train_transform, evaluation_transform = (
            get_cnn_transforms()
        )
    else:
        train_transform, evaluation_transform = (
            get_pretrained_transforms(
                image_size=image_size
            )
        )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_dataset = EuroSATDataset(
        data_root=data_root,
        split_csv=split_csv,
        split="train",
        transform=train_transform,
    )

    val_dataset = EuroSATDataset(
        data_root=data_root,
        split_csv=split_csv,
        split="val",
        transform=evaluation_transform,
    )

    test_dataset = EuroSATDataset(
        data_root=data_root,
        split_csv=split_csv,
        split="test",
        transform=evaluation_transform,
    )

    # --------------------------------------------------------
    # Validate datasets
    # --------------------------------------------------------

    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty.")

    if len(test_dataset) == 0:
        raise RuntimeError("Test dataset is empty.")

    if not (
        train_dataset.class_to_idx
        == val_dataset.class_to_idx
        == test_dataset.class_to_idx
    ):
        raise RuntimeError(
            "Class-to-index mappings are inconsistent across "
            "train, validation and test datasets."
        )

    if not (
        train_dataset.classes
        == val_dataset.classes
        == test_dataset.classes
    ):
        raise RuntimeError(
            "Class name ordering is inconsistent across "
            "train, validation and test datasets."
        )

    # --------------------------------------------------------
    # Resolve DataLoader settings
    # --------------------------------------------------------

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    # persistent_workers cannot be True when num_workers == 0.
    if num_workers == 0:
        persistent_workers = False

    common_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": persistent_workers,
    }

    # --------------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        drop_last=drop_last,
        generator=generator,
        **common_loader_kwargs,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )

    classes = list(train_dataset.classes)
    class_to_idx = dict(train_dataset.class_to_idx)

    if hasattr(train_dataset, "idx_to_class"):
        idx_to_class = dict(train_dataset.idx_to_class)
    else:
        idx_to_class = {
            class_index: class_name
            for class_name, class_index in class_to_idx.items()
        }

    return DataLoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        classes=classes,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
    )


# ============================================================
# Test entry point
# ============================================================

def main() -> None:
    """
    Run a basic DataLoader verification test.

    Run from the project root:

        python -m src.data.dataloader
    """

    data = create_dataloaders(
        batch_size=32,
        image_size=224,
        model_type="pretrained",
        num_workers=0,
        seed=42,
    )

    images, labels = next(iter(data.train_loader))

    print("=" * 60)
    print("EuroSAT DataLoader test")
    print("=" * 60)

    print(f"Train samples:      {len(data.train_dataset)}")
    print(f"Validation samples: {len(data.val_dataset)}")
    print(f"Test samples:       {len(data.test_dataset)}")

    print()

    print(f"Train batches:      {len(data.train_loader)}")
    print(f"Validation batches: {len(data.val_loader)}")
    print(f"Test batches:       {len(data.test_loader)}")

    print()

    print(f"Image batch shape:  {tuple(images.shape)}")
    print(f"Label batch shape:  {tuple(labels.shape)}")
    print(f"Image dtype:        {images.dtype}")
    print(f"Label dtype:        {labels.dtype}")

    print()

    print(f"Minimum label:      {labels.min().item()}")
    print(f"Maximum label:      {labels.max().item()}")
    print(f"Number of classes:  {len(data.classes)}")

    print()

    print("Class mapping:")

    for class_index, class_name in data.idx_to_class.items():
        print(f"  {class_index}: {class_name}")

    # Basic assertions
    assert images.ndim == 4
    assert labels.ndim == 1
    assert images.shape[0] == labels.shape[0]
    assert images.shape[1] == 3
    assert images.shape[2] == 224
    assert images.shape[3] == 224
    assert labels.dtype == torch.int64
    assert labels.min().item() >= 0
    assert labels.max().item() < len(data.classes)

    print()
    print("DataLoader verification passed.")


if __name__ == "__main__":
    main()