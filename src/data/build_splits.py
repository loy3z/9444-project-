from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "EuroSAT_RGB"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "eurosat_split_seed42.csv"
)

VALID_EXTENSIONS = {".jpg"}

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def collect_images() -> pd.DataFrame:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_ROOT}"
        )

    records: list[dict[str, object]] = []

    class_directories = sorted(
        path for path in DATA_ROOT.iterdir()
        if path.is_dir()
    )

    class_to_index = {
        class_directory.name: index
        for index, class_directory
        in enumerate(class_directories)
    }

    for class_directory in class_directories:
        class_name = class_directory.name
        class_id = class_to_index[class_name]

        for image_path in sorted(class_directory.iterdir()):
            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            relative_path = image_path.relative_to(DATA_ROOT)

            records.append(
                {
                    "filepath": relative_path.as_posix(),
                    "label": class_name,
                    "class_id": class_id,
                }
            )

    if not records:
        raise RuntimeError(
            f"No images found under: {DATA_ROOT}"
        )

    return pd.DataFrame(records)


def build_splits(dataframe: pd.DataFrame) -> pd.DataFrame:
    train_dataframe, temporary_dataframe = train_test_split(
        dataframe,
        test_size=VALIDATION_RATIO + TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=dataframe["label"],
    )

    temporary_test_ratio = (
        TEST_RATIO / (VALIDATION_RATIO + TEST_RATIO)
    )

    validation_dataframe, test_dataframe = train_test_split(
        temporary_dataframe,
        test_size=temporary_test_ratio,
        random_state=RANDOM_SEED,
        stratify=temporary_dataframe["label"],
    )

    train_dataframe = train_dataframe.copy()
    validation_dataframe = validation_dataframe.copy()
    test_dataframe = test_dataframe.copy()

    train_dataframe["split"] = "train"
    validation_dataframe["split"] = "val"
    test_dataframe["split"] = "test"

    split_dataframe = pd.concat(
        [
            train_dataframe,
            validation_dataframe,
            test_dataframe,
        ],
        ignore_index=True,
    )

    return split_dataframe.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)


def validate_splits(dataframe: pd.DataFrame) -> None:
    duplicated_paths = dataframe["filepath"].duplicated().sum()

    if duplicated_paths:
        raise RuntimeError(
            f"Found {duplicated_paths} duplicated file paths."
        )

    expected_splits = {"train", "val", "test"}
    actual_splits = set(dataframe["split"].unique())

    if actual_splits != expected_splits:
        raise RuntimeError(
            f"Unexpected splits: {actual_splits}"
        )

    print("\nSplit totals:")
    print(dataframe["split"].value_counts())

    print("\nClass distribution by split:")
    print(
        dataframe.groupby(["split", "label"])
        .size()
        .unstack(fill_value=0)
    )


def main() -> None:
    dataframe = collect_images()
    split_dataframe = build_splits(dataframe)

    validate_splits(split_dataframe)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"\nSaved split CSV to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()