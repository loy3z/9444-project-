"""EfficientNet-B0 training for EuroSAT."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.dataloader import (
    DEFAULT_DATA_ROOT,
    DEFAULT_SPLIT_CSV,
    create_dataloaders,
    set_seed,
)
from src.models.efficientnet import create_efficientnet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NUM_CLASSES = 10


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune and evaluate EfficientNet-B0 on EuroSAT RGB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=positive_int, default=20)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--image-size", type=positive_int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=non_negative_int, default=0)
    parser.add_argument("--seed", type=non_negative_int, default=42)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing the ten EuroSAT_RGB class folders.",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=DEFAULT_SPLIT_CSV,
        help="Fixed train/validation/test split CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Result directory. A sensible project-local default is used.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load ImageNet weights.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the final classifier instead of full fine-tuning.",
    )
    parser.add_argument(
        "--class-weighted",
        action="store_true",
        help="Use inverse-frequency class-weighted cross-entropy.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use automatic mixed precision on CUDA.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request deterministic PyTorch operations when available.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run one epoch with two train/validation batches, no pretrained "
            "download, and no test evaluation."
        ),
    )
    parser.add_argument(
        "--max-train-batches",
        type=positive_int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-val-batches",
        type=positive_int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-test-batches",
        type=positive_int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--log-interval",
        type=positive_int,
        default=50,
        help="Print progress after this many batches.",
    )

    args = parser.parse_args()

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than 0")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")

    if args.smoke_test:
        args.epochs = 1
        args.max_train_batches = 2
        args.max_val_batches = 2
        args.max_test_batches = None
        args.no_pretrained = True
        args.skip_test = True
        args.amp = False

    if args.output_dir is None:
        if args.smoke_test:
            experiment_name = "efficientnet_smoke"
        elif args.class_weighted:
            experiment_name = "efficientnet_class_weighted"
        else:
            experiment_name = "efficientnet"
        args.output_dir = PROJECT_ROOT / "outputs" / experiment_name

    return args


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = torch.cuda.is_available()
        torch.backends.cudnn.deterministic = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access CUDA.")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but it is not available.")

    return torch.device(requested)


def make_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def confusion_from_predictions(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    indices = targets.to(torch.int64) * num_classes + predictions.to(torch.int64)
    return torch.bincount(
        indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, Any]:
    confusion_float = confusion.to(torch.float64)
    support = confusion_float.sum(dim=1)
    predicted = confusion_float.sum(dim=0)
    true_positive = confusion_float.diag()

    precision = torch.where(
        predicted > 0,
        true_positive / predicted,
        torch.zeros_like(true_positive),
    )
    recall = torch.where(
        support > 0,
        true_positive / support,
        torch.zeros_like(true_positive),
    )
    denominator = precision + recall
    f1 = torch.where(
        denominator > 0,
        2 * precision * recall / denominator,
        torch.zeros_like(denominator),
    )

    total = confusion_float.sum()
    accuracy = (
        float(true_positive.sum() / total)
        if total.item() > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": float(f1.mean()),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.to(torch.int64).tolist(),
        "confusion_matrix": confusion.to(torch.int64).tolist(),
    }


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    phase: str,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    use_amp: bool = False,
    max_batches: int | None = None,
    log_interval: int = 50,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_loss = 0.0
    total_examples = 0
    optimizer_steps = 0
    started = time.perf_counter()

    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with autocast_context(use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                    optimizer_steps += 1
                else:
                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= scale_before_step:
                        optimizer_steps += 1

        predictions = logits.argmax(dim=1)
        batch_size = targets.size(0)
        total_loss += float(loss.detach()) * batch_size
        total_examples += batch_size
        confusion += confusion_from_predictions(
            targets.detach().cpu(),
            predictions.detach().cpu(),
            num_classes,
        )

        completed_batches = batch_index + 1
        effective_total = min(len(loader), max_batches or len(loader))
        if (
            completed_batches % log_interval == 0
            or completed_batches == effective_total
        ):
            print(
                f"  {phase}: batch {completed_batches}/{effective_total} "
                f"loss={total_loss / total_examples:.4f}",
                flush=True,
            )

    if total_examples == 0:
        raise RuntimeError(f"No samples were processed during the {phase} phase.")

    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / total_examples
    metrics["samples"] = total_examples
    metrics["seconds"] = time.perf_counter() - started
    metrics["optimizer_steps"] = optimizer_steps
    return metrics


def calculate_class_weights(
    train_dataset: Any,
    num_classes: int,
) -> tuple[torch.Tensor, list[int]]:
    if not hasattr(train_dataset, "samples"):
        raise TypeError("The training dataset does not expose split metadata.")

    class_ids = torch.as_tensor(
        train_dataset.samples["class_id"].to_numpy(),
        dtype=torch.int64,
    )
    counts = torch.bincount(class_ids, minlength=num_classes)
    if (counts == 0).any():
        raise ValueError("Every class must have at least one training sample.")

    weights = counts.sum() / (num_classes * counts.to(torch.float32))
    return weights, counts.tolist()


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def json_ready_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def save_history(history: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "train_seconds",
        "val_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_per_class_metrics(
    metrics: dict[str, Any],
    classes: list[str],
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class_id", "class_name", "precision", "recall", "f1", "support"])
        for class_id, class_name in enumerate(classes):
            writer.writerow(
                [
                    class_id,
                    class_name,
                    metrics["precision_per_class"][class_id],
                    metrics["recall_per_class"][class_id],
                    metrics["f1_per_class"][class_id],
                    metrics["support_per_class"][class_id],
                ]
            )


def save_confusion_csv(
    confusion: list[list[int]],
    classes: list[str],
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *classes])
        for class_name, row in zip(classes, confusion):
            writer.writerow([class_name, *row])


def save_confusion_figure(
    confusion: list[list[int]],
    classes: list[str],
    path: Path,
    *,
    normalized: bool,
) -> None:
    matrix = np.asarray(confusion, dtype=np.float64)
    if normalized:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix),
            where=row_sums != 0,
        )

    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        title="Normalized confusion matrix" if normalized else "Confusion matrix",
        xlabel="Predicted class",
        ylabel="Actual class",
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")

    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            label = f"{value:.2f}" if normalized else str(int(value))
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_training_curves(history: list[dict[str, Any]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="Train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))

    axes[2].plot(epochs, [row["train_macro_f1"] for row in history], label="Train")
    axes[2].plot(epochs, [row["val_macro_f1"] for row in history], label="Validation")
    axes[2].set(title="Macro-F1", xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1))

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()

    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return str(device)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configure_reproducibility(args.seed, args.deterministic)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    print("=" * 72)
    print("COMP9444 Project 54 - EfficientNet")
    print("=" * 72)
    print(f"Mode:        {'smoke test' if args.smoke_test else 'full experiment'}")
    print(f"Device:      {describe_device(device)}")
    print(f"AMP:         {use_amp}")
    print(f"Output:      {args.output_dir.resolve()}")
    print(f"Data root:   {args.data_root.resolve()}")

    data = create_dataloaders(
        batch_size=args.batch_size,
        image_size=args.image_size,
        model_type="pretrained",
        num_workers=args.num_workers,
        seed=args.seed,
        data_root=args.data_root,
        split_csv=args.split_csv,
    )
    if len(data.classes) != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} classes, found {len(data.classes)}."
        )

    model = create_efficientnet(
        num_classes=len(data.classes),
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    total_parameters, trainable_parameters = count_parameters(model)

    class_weights = None
    train_class_counts = None
    if args.class_weighted:
        class_weights, train_class_counts = calculate_class_weights(
            data.train_dataset,
            len(data.classes),
        )
        class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01,
    )
    scaler = make_grad_scaler(use_amp)

    config = {
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": json_ready_args(args),
        "selection_metric": "validation_macro_f1",
        "loss": "class_weighted_cross_entropy" if args.class_weighted else "cross_entropy",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "pretrained": not args.no_pretrained,
        "full_fine_tuning": not args.freeze_backbone,
        "amp_enabled": use_amp,
        "device": describe_device(device),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "classes": data.classes,
        "class_to_idx": data.class_to_idx,
        "dataset_sizes": {
            "train": len(data.train_dataset),
            "validation": len(data.val_dataset),
            "test": len(data.test_dataset),
        },
        "train_class_counts": train_class_counts,
        "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }
    save_json(config, args.output_dir / "config.json")

    print(f"Classes:     {data.classes}")
    print(f"Parameters:  {total_parameters:,} total; {trainable_parameters:,} trainable")
    print(f"Loss:        {config['loss']}")

    history: list[dict[str, Any]] = []
    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_model_path = args.output_dir / "best_model.pt"
    training_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{args.epochs} - lr={learning_rate:.6g}")

        train_metrics = run_epoch(
            model=model,
            loader=data.train_loader,
            criterion=criterion,
            device=device,
            num_classes=len(data.classes),
            phase="train",
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            max_batches=args.max_train_batches,
            log_interval=args.log_interval,
        )
        val_metrics = run_epoch(
            model=model,
            loader=data.val_loader,
            criterion=criterion,
            device=device,
            num_classes=len(data.classes),
            phase="validation",
            use_amp=use_amp,
            max_batches=args.max_val_batches,
            log_interval=args.log_interval,
        )

        history_row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "train_seconds": train_metrics["seconds"],
            "val_seconds": val_metrics["seconds"],
        }
        history.append(history_row)
        save_history(history, args.output_dir / "history.csv")

        print(
            "  summary: "
            f"train loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"macro-F1={train_metrics['macro_f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"macro-F1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            save_json(
                {
                    "best_epoch": best_epoch,
                    "validation_metrics": val_metrics,
                    "selection_metric": "validation_macro_f1",
                },
                args.output_dir / "best_validation_metrics.json",
            )
            save_per_class_metrics(
                val_metrics,
                data.classes,
                args.output_dir / "best_validation_per_class_metrics.csv",
            )
            print(f"  saved new best model (validation macro-F1={best_val_macro_f1:.4f})")

        if train_metrics["optimizer_steps"] > 0:
            scheduler.step()
        else:
            print(
                "  warning: no optimizer step completed in this epoch; "
                "learning-rate scheduler was not advanced."
            )

    training_seconds = time.perf_counter() - training_started
    save_training_curves(history, args.output_dir / "training_curves.png")

    if not best_model_path.exists():
        raise RuntimeError("Training finished without producing a best checkpoint.")

    summary: dict[str, Any] = {
        "status": "smoke_test_complete" if args.smoke_test else "training_complete",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_macro_f1,
        "training_seconds": training_seconds,
        "checkpoint_size_bytes": best_model_path.stat().st_size,
        "test_evaluated": False,
    }

    if not args.skip_test:
        print("\nLoading the best validation checkpoint for one final test evaluation...")
        state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict)

        test_metrics = run_epoch(
            model=model,
            loader=data.test_loader,
            criterion=criterion,
            device=device,
            num_classes=len(data.classes),
            phase="test",
            use_amp=use_amp,
            max_batches=args.max_test_batches,
            log_interval=args.log_interval,
        )
        save_json(test_metrics, args.output_dir / "test_metrics.json")
        save_per_class_metrics(
            test_metrics,
            data.classes,
            args.output_dir / "test_per_class_metrics.csv",
        )
        save_confusion_csv(
            test_metrics["confusion_matrix"],
            data.classes,
            args.output_dir / "confusion_matrix.csv",
        )
        save_confusion_figure(
            test_metrics["confusion_matrix"],
            data.classes,
            args.output_dir / "confusion_matrix.png",
            normalized=False,
        )
        save_confusion_figure(
            test_metrics["confusion_matrix"],
            data.classes,
            args.output_dir / "confusion_matrix_normalized.png",
            normalized=True,
        )

        summary["test_evaluated"] = True
        summary["test_accuracy"] = test_metrics["accuracy"]
        summary["test_macro_f1"] = test_metrics["macro_f1"]

        print(
            f"Test accuracy={test_metrics['accuracy']:.4f}; "
            f"macro-F1={test_metrics['macro_f1']:.4f}"
        )
    else:
        print("\nTest evaluation skipped (expected for a smoke test).")

    save_json(summary, args.output_dir / "summary.json")
    print(f"Results saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()