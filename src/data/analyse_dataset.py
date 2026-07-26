from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "EuroSAT_RGB"

VALID_EXTENSIONS = {".jpg"}


def analyse_dataset() -> None:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"EuroSAT dataset not found at: {DATA_ROOT}"
        )

    class_directories = sorted(
        path for path in DATA_ROOT.iterdir()
        if path.is_dir()
    )

    if not class_directories:
        raise RuntimeError(
            f"No class folders found inside: {DATA_ROOT}"
        )

    class_counts: Counter[str] = Counter()
    image_sizes: Counter[tuple[int, int]] = Counter()
    image_modes: Counter[str] = Counter()
    bad_files: list[tuple[str, str]] = []

    for class_directory in class_directories:
        image_paths = sorted(
            path for path in class_directory.iterdir()
            if path.suffix.lower() in VALID_EXTENSIONS
        )

        class_counts[class_directory.name] = len(image_paths)

        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    image_sizes[image.size] += 1
                    image_modes[image.mode] += 1
                    image.verify()
            except (UnidentifiedImageError, OSError) as error:
                bad_files.append(
                    (str(image_path), str(error))
                )

    total_images = sum(class_counts.values())

    print("=" * 60)
    print("EuroSAT RGB Dataset Report")
    print("=" * 60)

    print(f"\nDataset root: {DATA_ROOT}")
    print(f"Number of classes: {len(class_counts)}")
    print(f"Total images: {total_images}")

    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        percentage = count / total_images * 100
        print(
            f"{class_name:25s} "
            f"{count:5d} "
            f"({percentage:5.2f}%)"
        )

    print("\nImage sizes:")
    for size, count in image_sizes.items():
        print(f"{size}: {count}")

    print("\nImage modes:")
    for mode, count in image_modes.items():
        print(f"{mode}: {count}")

    print(f"\nBad files: {len(bad_files)}")
    for file_path, error in bad_files:
        print(f"{file_path}: {error}")

    if total_images != 27000:
        print(
            "\nWarning: expected approximately 27,000 images, "
            f"but found {total_images}."
        )

    if len(class_counts) != 10:
        print(
            "\nWarning: expected 10 classes, "
            f"but found {len(class_counts)}."
        )


if __name__ == "__main__":
    analyse_dataset()