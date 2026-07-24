from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


VALID_SPLITS = {"train", "val", "test"}
REQUIRED_COLUMNS = {
    "filepath",
    "label",
    "class_id",
    "split",
}


class EuroSATDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split_csv: str | Path,
        split: str,
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. "
                f"Expected one of {sorted(VALID_SPLITS)}."
            )

        self.data_root = Path(data_root)
        self.split_csv = Path(split_csv)
        self.split = split
        self.transform = transform

        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.data_root}"
            )

        if not self.data_root.is_dir():
            raise NotADirectoryError(
                f"data_root is not a directory: {self.data_root}"
            )

        if not self.split_csv.exists():
            raise FileNotFoundError(
                f"Split CSV not found: {self.split_csv}"
            )

        dataframe = pd.read_csv(self.split_csv)

        missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

        if missing_columns:
            raise ValueError(
                "Split CSV is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if dataframe["filepath"].isna().any():
            raise ValueError(
                "Split CSV contains missing filepath values."
            )

        if dataframe["label"].isna().any():
            raise ValueError(
                "Split CSV contains missing label values."
            )

        if dataframe["class_id"].isna().any():
            raise ValueError(
                "Split CSV contains missing class_id values."
            )

        invalid_splits = (
            set(dataframe["split"].dropna().unique())
            - VALID_SPLITS
        )

        if invalid_splits:
            raise ValueError(
                "Split CSV contains invalid split values: "
                f"{sorted(invalid_splits)}"
            )

        duplicated_paths = dataframe["filepath"].duplicated()

        if duplicated_paths.any():
            duplicate_examples = (
                dataframe.loc[duplicated_paths, "filepath"]
                .head(5)
                .tolist()
            )

            raise ValueError(
                "Split CSV contains duplicated filepaths. "
                f"Examples: {duplicate_examples}"
            )

        dataframe["class_id"] = dataframe["class_id"].astype(int)

        self.samples = (
            dataframe[dataframe["split"] == split]
            .copy()
            .reset_index(drop=True)
        )

        if self.samples.empty:
            raise RuntimeError(
                f"No samples found for split '{split}'."
            )

        class_mapping = (
            dataframe[["label", "class_id"]]
            .drop_duplicates()
            .sort_values("class_id")
        )

        duplicate_labels = class_mapping["label"].duplicated()
        duplicate_ids = class_mapping["class_id"].duplicated()

        if duplicate_labels.any() or duplicate_ids.any():
            raise ValueError(
                "Inconsistent label and class_id mapping "
                "found in the split CSV."
            )

        self.class_to_idx = {
            str(row.label): int(row.class_id)
            for row in class_mapping.itertuples(index=False)
        }

        self.idx_to_class = {
            class_id: class_name
            for class_name, class_id
            in self.class_to_idx.items()
        }

        self.classes = [
            self.idx_to_class[class_id]
            for class_id in sorted(self.idx_to_class)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Any, int]:
        if index < 0:
            index += len(self.samples)

        if index < 0 or index >= len(self.samples):
            raise IndexError(
                f"Dataset index out of range: {index}"
            )

        row = self.samples.iloc[index]

        image_path = (
            self.data_root / str(row["filepath"])
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image file not found: {image_path}"
            )

        try:
            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB").copy()
        except (
            OSError,
            UnidentifiedImageError,
        ) as error:
            raise RuntimeError(
                f"Unable to read image: {image_path}"
            ) from error

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["class_id"])

        return image, label