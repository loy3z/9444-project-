"""Sample-efficiency sweep: pretrained models vs training-set size, with seed repeats.

Helber et al. (2019) chart accuracy against training-set fraction only for *randomly
initialised* networks (their Table II); their fine-tuned results exist at the 80/20 split
alone (Table III). This script fills that gap for pretrained models, and because every point
is repeated over several seeds it simultaneously produces the error bars the single-run
notebook comparison lacks.

Protocol is identical to the notebook (AdamW 1e-4 / wd 1e-4, cosine to 1e-6, 20 epochs,
batch 32, AMP, checkpoint chosen by best validation macro-F1). Only the training subset and
the seed vary. Validation and test sets are always the full, untouched splits.

Results append to results.csv after every run, so the sweep is resumable and partial output
is usable. Re-running skips completed (model, fraction, seed) combinations.

    python -m src.experiments.sample_efficiency
    python -m src.experiments.sample_efficiency --smoke
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from src.data.dataloader import create_dataloaders

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sample_efficiency"
RESULTS_CSV = OUTPUT_DIR / "results.csv"

NUM_CLASSES = 10
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

FRACTIONS = (0.10, 0.25, 0.50, 1.00)
SEEDS = (1, 2, 3)
MODEL_NAMES = ("resnet50", "efficientnet_b0", "deit_tiny")


# ------------------------------------------------------------------ models
def build_model(name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    if name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if name == "deit_tiny":
        import timm

        return timm.create_model(
            "deit_tiny_patch16_224", pretrained=True, num_classes=num_classes
        )

    raise ValueError(f"Unknown model: {name}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------ metrics
def confusion_from_predictions(targets, predictions, num_classes):
    indices = targets.to(torch.int64) * num_classes + predictions.to(torch.int64)
    return torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)


def metrics_from_confusion(confusion):
    matrix = confusion.to(torch.float64)
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    true_positive = matrix.diag()
    precision = torch.where(predicted > 0, true_positive / predicted, 0.0)
    recall = torch.where(support > 0, true_positive / support, 0.0)
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {
        "accuracy": float(true_positive.sum() / matrix.sum()),
        "macro_f1": float(f1.mean()),
        "f1_per_class": f1.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        predictions = logits.argmax(dim=1)
        total_loss += float(loss.detach()) * targets.size(0)
        total_examples += targets.size(0)
        confusion += confusion_from_predictions(
            targets.detach().cpu(), predictions.detach().cpu(), NUM_CLASSES
        )

    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / total_examples
    return metrics


# ------------------------------------------------------------------ subsetting
def stratified_subset_indices(dataset, fraction: float, seed: int) -> list[int]:
    """Class-proportional subsample of a training split, reproducible from `seed`."""
    if fraction >= 1.0:
        return list(range(len(dataset)))

    class_ids = dataset.samples["class_id"].to_numpy()
    generator = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in np.unique(class_ids):
        pool = np.flatnonzero(class_ids == class_id)
        keep = max(1, int(round(len(pool) * fraction)))
        selected.extend(generator.choice(pool, size=keep, replace=False).tolist())
    selected.sort()
    return selected


def build_train_loader(dataset, indices: list[int], seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
        generator=generator,
        drop_last=False,
    )


# ------------------------------------------------------------------ one run
def train_one(model_name, fraction, seed, bundle, device, epochs):
    set_seed(seed)
    model = build_model(model_name).to(device)

    indices = stratified_subset_indices(bundle.train_dataset, fraction, seed)
    train_loader = build_train_loader(bundle.train_dataset, indices, seed)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LEARNING_RATE * 0.01)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_state = None
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        run_epoch(model, train_loader, criterion, device,
                  optimizer=optimizer, scaler=scaler, use_amp=use_amp)
        val_metrics = run_epoch(model, bundle.val_loader, criterion, device, use_amp=use_amp)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
        scheduler.step()

    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    test_metrics = run_epoch(model, bundle.test_loader, criterion, device, use_amp=use_amp)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "fraction": fraction,
        "seed": seed,
        "train_images": len(indices),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "val_macro_f1": best_val_macro_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "training_seconds": round(training_seconds, 1),
        "f1_per_class": json.dumps([round(v, 6) for v in test_metrics["f1_per_class"]]),
    }


# ------------------------------------------------------------------ sweep
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="1 epoch, tiny fraction, single seed and model")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke:
        fractions, seeds, models, epochs = (0.02,), (1,), ("deit_tiny",), 1
        results_csv = OUTPUT_DIR / "smoke_results.csv"
    else:
        fractions, seeds, models, epochs = FRACTIONS, SEEDS, MODEL_NAMES, EPOCHS
        results_csv = RESULTS_CSV

    done = set()
    if results_csv.is_file():
        previous = pd.read_csv(results_csv)
        done = {(r.model, round(r.fraction, 4), int(r.seed)) for r in previous.itertuples()}
        print(f"Resuming: {len(done)} runs already complete")

    bundle = create_dataloaders(
        batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, model_type="pretrained",
        num_workers=NUM_WORKERS, seed=42,
    )
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else device}")
    print(f"Full train / val / test: {len(bundle.train_dataset)} / "
          f"{len(bundle.val_dataset)} / {len(bundle.test_dataset)}")

    # Seed-major, then cheapest fractions first, so one complete curve lands early.
    schedule = [(seed, fraction, name)
                for seed in seeds for fraction in fractions for name in models]
    todo = [job for job in schedule if (job[2], round(job[1], 4), job[0]) not in done]
    print(f"Runs: {len(todo)} to go of {len(schedule)}\n")

    for index, (seed, fraction, name) in enumerate(todo, start=1):
        print(f"[{index}/{len(todo)}] {name}  fraction={fraction:.2f}  seed={seed}", flush=True)
        started = time.perf_counter()
        row = train_one(name, fraction, seed, bundle, device, epochs)
        row["wall_seconds"] = round(time.perf_counter() - started, 1)

        frame = pd.DataFrame([row])
        frame.to_csv(results_csv, mode="a", header=not results_csv.is_file(), index=False)
        print(f"    train_images={row['train_images']:,}  "
              f"test_acc={row['test_accuracy']*100:.2f}%  "
              f"test_macroF1={row['test_macro_f1']*100:.2f}%  "
              f"({row['wall_seconds']:.0f}s)\n", flush=True)

    print(f"Sweep complete -> {results_csv}")


if __name__ == "__main__":
    main()
