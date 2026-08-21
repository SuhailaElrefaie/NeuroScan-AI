from __future__ import annotations

import csv
import json
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_3d import BraTSWindowDataset, discover_volumes, split_volume_ids
from models.unet3d import UNet3D


@dataclass(frozen=True)
class Config:
    h5_dir: str = "archive/BraTS2021_h5/content/data"
    runs_dir: str = "runs_3d"
    best_dir: str = "best_model_3d"

    seed: int = 42
    val_fraction: float = 0.20

    depth: int = 32
    image_height: int = 160
    image_width: int = 160

    train_windows_per_patient: int = 4
    tumour_window_probability: float = 0.75
    val_stride: int = 16

    batch_size: int = 4
    epochs: int = 80
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    num_workers: int = 4
    patience: int = 12
    min_delta: float = 0.001
    gradient_clip: float = 1.0

    bce_weight: float = 0.40
    dice_weight: float = 0.60
    positive_weight: float = 5.0

    default_threshold: float = 0.50


CFG = Config()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. On the GPU server, first check `nvidia-smi` "
            "and your CUDA-enabled PyTorch installation."
        )
    return torch.device("cuda:0")


class DiceBCELoss(nn.Module):
    def __init__(
        self,
        pos_weight: float,
        bce_weight: float,
        dice_weight: float,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
        )

        probabilities = torch.sigmoid(logits)
        dims = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * targets).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
        dice_loss = 1.0 - (
            (2.0 * intersection + self.smooth)
            / (denominator + self.smooth)
        ).mean()

        return self.bce_weight * bce + self.dice_weight * dice_loss


def confusion_counts(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> Tuple[float, float, float]:
    predictions = probabilities >= threshold
    truth = targets >= 0.5

    tp = torch.logical_and(predictions, truth).sum().item()
    fp = torch.logical_and(predictions, ~truth).sum().item()
    fn = torch.logical_and(~predictions, truth).sum().item()
    return tp, fp, fn


def metrics_from_counts(tp: float, fp: float, fn: float) -> Dict[str, float]:
    eps = 1e-7
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "Dice coefficient": float(dice),
        "Mean IoU": float(iou),
        "Precision": float(precision),
        "Recall / Sensitivity": float(recall),
    }


def make_loaders(config: Config) -> Tuple[DataLoader, DataLoader, list[int], list[int]]:
    volumes = discover_volumes(config.h5_dir)
    train_ids, val_ids = split_volume_ids(
        sorted(volumes),
        val_fraction=config.val_fraction,
        seed=config.seed,
    )

    image_size = (config.image_height, config.image_width)
    train_dataset = BraTSWindowDataset(
        h5_dir=config.h5_dir,
        volume_ids=train_ids,
        depth=config.depth,
        image_size=image_size,
        training=True,
        samples_per_volume=config.train_windows_per_patient,
        tumour_probability=config.tumour_window_probability,
        val_stride=config.val_stride,
        augment=True,
        seed=config.seed,
    )
    val_dataset = BraTSWindowDataset(
        h5_dir=config.h5_dir,
        volume_ids=val_ids,
        depth=config.depth,
        image_size=image_size,
        training=False,
        samples_per_volume=1,
        tumour_probability=0.0,
        val_stride=config.val_stride,
        augment=False,
        seed=config.seed,
    )

    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": True,
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, train_ids, val_ids


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    gradient_clip: float,
) -> float:
    model.train()
    total_loss = 0.0

    for batch_index, (images, masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=True):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if batch_index % 20 == 0 or batch_index == len(loader):
            print(
                f"  train batch {batch_index:4d}/{len(loader)} "
                f"loss={loss.item():.4f}",
                flush=True,
            )

    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    tp = fp = fn = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=True):
            logits = model(images)
            loss = criterion(logits, masks)

        total_loss += loss.item()
        probabilities = torch.sigmoid(logits)
        batch_tp, batch_fp, batch_fn = confusion_counts(
            probabilities,
            masks,
            threshold,
        )
        tp += batch_tp
        fp += batch_fp
        fn += batch_fn

    return total_loss / max(1, len(loader)), metrics_from_counts(tp, fp, fn)


@torch.no_grad()
def tune_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: Iterable[float],
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    thresholds = list(thresholds)
    counts = {value: [0.0, 0.0, 0.0] for value in thresholds}

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=True):
            probabilities = torch.sigmoid(model(images))

        for threshold in thresholds:
            tp, fp, fn = confusion_counts(probabilities, masks, threshold)
            counts[threshold][0] += tp
            counts[threshold][1] += fp
            counts[threshold][2] += fn

    results = {
        threshold: metrics_from_counts(*values)
        for threshold, values in counts.items()
    }
    best_threshold = max(
        results,
        key=lambda value: results[value]["Dice coefficient"],
    )
    return float(best_threshold), results[best_threshold]


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    set_reproducibility(CFG.seed)
    device = get_device()

    print("=" * 72)
    print("Proper 3D BraTS training")
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))
    print("PyTorch sees:", torch.cuda.get_device_name(0))
    print("Device:", device)
    print("=" * 72)

    train_loader, val_loader, train_ids, val_ids = make_loaders(CFG)
    print(f"Train patients: {len(train_ids)}")
    print(f"Validation patients: {len(val_ids)}")
    print(f"Train windows/epoch: {len(train_loader.dataset)}")
    print(f"Validation windows: {len(val_loader.dataset)}")

    model = UNet3D(in_channels=4, out_channels=1).to(device)
    criterion = DiceBCELoss(
        pos_weight=CFG.positive_weight,
        bce_weight=CFG.bce_weight,
        dice_weight=CFG.dice_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_dir = Path(CFG.runs_dir) / f"run_3d_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_dir = Path(CFG.best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)

    config_payload = asdict(CFG)
    config_payload["train_patient_ids"] = train_ids
    config_payload["validation_patient_ids"] = val_ids
    save_json(run_dir / "config.json", config_payload)

    history_path = run_dir / "history_3d.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_dice_at_0.5",
                "val_iou_at_0.5",
                "precision_at_0.5",
                "recall_at_0.5",
                "learning_rate",
                "seconds",
            ]
        )

    best_dice = -1.0
    epochs_without_improvement = 0
    history_rows = []
    best_checkpoint = run_dir / "best_checkpoint.pth"

    for epoch in range(1, CFG.epochs + 1):
        start_time = time.time()
        print(f"\nEpoch {epoch}/{CFG.epochs}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            CFG.gradient_clip,
        )
        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            CFG.default_threshold,
        )
        val_dice = val_metrics["Dice coefficient"]
        scheduler.step(val_dice)

        seconds = time.time() - start_time
        learning_rate = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.4f} | "
            f"val={val_loss:.4f} | "
            f"dice={val_dice:.4f} | "
            f"iou={val_metrics['Mean IoU']:.4f} | "
            f"precision={val_metrics['Precision']:.4f} | "
            f"recall={val_metrics['Recall / Sensitivity']:.4f} | "
            f"lr={learning_rate:.2e} | "
            f"{seconds / 60:.1f} min",
            flush=True,
        )

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice_at_0.5": val_dice,
            "val_iou_at_0.5": val_metrics["Mean IoU"],
            "precision_at_0.5": val_metrics["Precision"],
            "recall_at_0.5": val_metrics["Recall / Sensitivity"],
            "learning_rate": learning_rate,
            "seconds": seconds,
        }
        history_rows.append(history_row)

        with history_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(list(history_row.values()))

        improved = val_dice > best_dice + CFG.min_delta
        if improved:
            best_dice = val_dice
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_dice_at_0.5": val_dice,
                    "config": asdict(CFG),
                },
                best_checkpoint,
            )
            print(f"  Saved new best checkpoint (Dice {best_dice:.4f}).")
        else:
            epochs_without_improvement += 1
            print(
                f"  No improvement for "
                f"{epochs_without_improvement}/{CFG.patience} epochs."
            )

        if epochs_without_improvement >= CFG.patience:
            print("Early stopping.")
            break

    if not best_checkpoint.exists():
        raise RuntimeError("Training ended without producing a checkpoint.")

    # This checkpoint was created by this training process and is trusted.
    try:
        checkpoint = torch.load(
            best_checkpoint,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nTuning the probability threshold on the validation set...")
    threshold_values = np.arange(0.20, 0.81, 0.05).round(2).tolist()
    best_threshold, tuned_metrics = tune_threshold(
        model,
        val_loader,
        device,
        threshold_values,
    )

    final_metrics = {
        **tuned_metrics,
        "Threshold": best_threshold,
        "Epoch": int(checkpoint["epoch"]),
        "Validation loss": float(
            next(
                row["val_loss"]
                for row in history_rows
                if row["epoch"] == int(checkpoint["epoch"])
            )
        ),
        "Depth": CFG.depth,
        "Image size": [CFG.image_height, CFG.image_width],
        "Train patients": len(train_ids),
        "Validation patients": len(val_ids),
        "Normalization": "per-modality non-zero z-score",
        "Validation": "patient-level split with full-depth overlapping windows",
        "Run folder": str(run_dir),
    }

    # Save plain state_dict so the existing app can load it directly.
    run_model_path = run_dir / "model_3d.pth"
    torch.save(model.state_dict(), run_model_path)
    save_json(run_dir / "metrics_3d.json", final_metrics)

    shutil.copy2(run_model_path, best_dir / "best_unet3d.pth")
    shutil.copy2(history_path, best_dir / "best_history_3d.csv")
    shutil.copy2(
        run_dir / "metrics_3d.json",
        best_dir / "best_metrics_3d.json",
    )

    print("\n" + "=" * 72)
    print("Training complete.")
    print(f"Best validation threshold: {best_threshold:.2f}")
    print(f"Best tuned Dice: {tuned_metrics['Dice coefficient']:.4f}")
    print(f"Model: {best_dir / 'best_unet3d.pth'}")
    print(f"Metrics: {best_dir / 'best_metrics_3d.json'}")
    print(f"History: {best_dir / 'best_history_3d.csv'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
