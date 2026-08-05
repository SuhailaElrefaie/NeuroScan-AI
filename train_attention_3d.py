"""Train the deeper attention 3D U-Net on the full converted BraTS dataset.

The script does not overwrite the currently deployed baseline.  It produces a
candidate checkpoint named best_model_3d/best_attention_unet3d.pth.  Promote it
only after its validation metrics are satisfactory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_attention_3d import AttentionTrainingDataset, VolumeStore
from models.attention_unet3d import AttentionUNet3D


@dataclass
class TrainConfig:
    h5_dir: str
    runs_dir: str = "runs_3d_attention"
    best_dir: str = "best_model_3d"
    seed: int = 42
    val_fraction: float = 0.20
    depth: int = 32
    image_height: int = 160
    image_width: int = 160
    samples_per_volume: int = 2
    tumour_probability: float = 0.75
    val_stride: int = 16
    batch_size: int = 2
    gradient_accumulation: int = 2
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_workers: int = 4
    patience: int = 10
    min_delta: float = 0.001
    gradient_clip: float = 1.0
    bce_weight: float = 0.40
    dice_weight: float = 0.60
    positive_weight: float = 5.0
    base_channels: int = 16
    num_heads: int = 8
    rebuild_index: bool = False


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-dir", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples-per-volume", type=int, default=2)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=8)
    args = parser.parse_args()

    h5_dir = args.h5_dir or os.environ.get("NEUROSCAN_H5_DIR") or detect_h5_dir()
    return TrainConfig(
        h5_dir=h5_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        num_workers=args.workers,
        samples_per_volume=args.samples_per_volume,
        rebuild_index=args.rebuild_index,
        base_channels=args.base_channels,
        num_heads=args.num_heads,
    )


def detect_h5_dir() -> str:
    candidates = [
        Path("archive/BraTS2021_h5/content/data"),
        Path("archive.nobackup/BraTS2021_h5/content/data"),
        Path("/projects/elrefais/NeuroScan-AI/archive.nobackup/BraTS2021_h5/content/data"),
        Path("/projects/elrefais/NeuroScan-AI/archive.nobackup/content/data"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "Could not auto-detect the converted BraTS H5 folder. Pass it explicitly with "
        "--h5-dir. To locate it, run: find /projects/elrefais /home/elrefais "
        "-type f -name 'volume_*_slice_*.h5' -print -quit"
    )


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)

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


def metrics_from_counts(tp: float, fp: float, fn: float) -> Dict[str, float]:
    eps = 1e-7
    return {
        "Dice coefficient": float((2 * tp) / (2 * tp + fp + fn + eps)),
        "Mean IoU": float(tp / (tp + fp + fn + eps)),
        "Precision": float(tp / (tp + fp + eps)),
        "Recall / Sensitivity": float(tp / (tp + fn + eps)),
    }


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return nullcontext()


def make_loader(
    store: VolumeStore,
    train_ids: Sequence[int],
    config: TrainConfig,
) -> DataLoader:
    dataset = AttentionTrainingDataset(
        store,
        train_ids,
        samples_per_volume=config.samples_per_volume,
        tumour_probability=config.tumour_probability,
        augment=True,
        seed=config.seed,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: TrainConfig,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0

    for batch_index, (images, masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast_context(device):
            logits = model(images)
            raw_loss = criterion(logits, masks)
            loss = raw_loss / config.gradient_accumulation

        scaler.scale(loss).backward()
        should_step = (
            batch_index % config.gradient_accumulation == 0
            or batch_index == len(loader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(raw_loss.item())
        if batch_index % 25 == 0 or batch_index == len(loader):
            print(
                f"  train batch {batch_index:4d}/{len(loader)} "
                f"loss={raw_loss.item():.4f}",
                flush=True,
            )

    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate_full_volumes(
    model: nn.Module,
    store: VolumeStore,
    volume_ids: Sequence[int],
    criterion: nn.Module,
    device: torch.device,
    thresholds: Iterable[float],
    stride: int,
) -> Tuple[float, Dict[float, Dict[str, float]]]:
    """Merge overlapping windows per patient before calculating validation metrics."""
    model.eval()
    thresholds = [float(value) for value in thresholds]
    counts = {value: [0.0, 0.0, 0.0] for value in thresholds}
    total_loss = 0.0
    window_count = 0
    height, width = store.image_size

    for patient_number, volume_id in enumerate(volume_ids, start=1):
        full_depth = store.volume_depth(volume_id)
        probability_sum = np.zeros((full_depth, height, width), dtype=np.float32)
        probability_count = np.zeros_like(probability_sum)
        truth = np.zeros((full_depth, height, width), dtype=bool)

        for start in store.window_starts(volume_id, stride=stride):
            image, mask, valid_depth = store.load_window(volume_id, start)
            image_batch = image.unsqueeze(0).to(device, non_blocking=True)
            mask_batch = mask.unsqueeze(0).to(device, non_blocking=True)

            with autocast_context(device):
                logits = model(image_batch)
                loss = criterion(logits, mask_batch)

            probability = torch.sigmoid(logits)[0, 0, :valid_depth].float().cpu().numpy()
            truth_window = mask[0, :valid_depth].numpy() >= 0.5
            end = start + valid_depth
            probability_sum[start:end] += probability
            probability_count[start:end] += 1.0
            truth[start:end] = truth_window
            total_loss += float(loss.item())
            window_count += 1

        probability_count[probability_count == 0] = 1.0
        merged = probability_sum / probability_count

        for threshold in thresholds:
            prediction = merged >= threshold
            tp = float(np.logical_and(prediction, truth).sum())
            fp = float(np.logical_and(prediction, ~truth).sum())
            fn = float(np.logical_and(~prediction, truth).sum())
            counts[threshold][0] += tp
            counts[threshold][1] += fp
            counts[threshold][2] += fn

        if patient_number % 25 == 0 or patient_number == len(volume_ids):
            print(
                f"  validated {patient_number}/{len(volume_ids)} patients",
                flush=True,
            )

    metrics = {
        threshold: metrics_from_counts(*values)
        for threshold, values in counts.items()
    }
    return total_loss / max(1, window_count), metrics


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    config = parse_args()
    set_reproducibility(config.seed)
    device = get_device()

    print("=" * 78)
    print("NeuroScan AI: deeper 3D U-Net + multi-head attention")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))
    print("H5 dataset:", config.h5_dir)
    print("=" * 78)

    store = VolumeStore(
        config.h5_dir,
        depth=config.depth,
        image_size=(config.image_height, config.image_width),
        rebuild_index=config.rebuild_index,
    )
    train_ids, val_ids = store.split_ids(config.val_fraction, config.seed)
    train_loader = make_loader(store, train_ids, config)

    print(f"Total patients: {len(store.volume_ids)}")
    print(f"Training patients: {len(train_ids)}")
    print(f"Validation patients: {len(val_ids)}")
    print(f"Training windows per epoch: {len(train_loader.dataset)}")

    model = AttentionUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=config.base_channels,
        num_heads=config.num_heads,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {parameter_count:,}")

    criterion = DiceBCELoss(
        pos_weight=config.positive_weight,
        bce_weight=config.bce_weight,
        dice_weight=config.dice_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_dir = Path(config.runs_dir) / f"run_attention_3d_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_dir = Path(config.best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)

    config_payload = asdict(config)
    config_payload["train_patient_ids"] = train_ids
    config_payload["validation_patient_ids"] = val_ids
    config_payload["parameter_count"] = parameter_count
    save_json(run_dir / "config.json", config_payload)

    history_path = run_dir / "history_attention_3d.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
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
    best_checkpoint = run_dir / "best_attention_checkpoint.pth"
    history_rows = []

    for epoch in range(1, config.epochs + 1):
        started = time.time()
        print(f"\nEpoch {epoch}/{config.epochs}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            config,
        )
        val_loss, validation = validate_full_volumes(
            model,
            store,
            val_ids,
            criterion,
            device,
            thresholds=[0.50],
            stride=config.val_stride,
        )
        metrics = validation[0.50]
        val_dice = metrics["Dice coefficient"]
        scheduler.step(val_dice)
        seconds = time.time() - started
        learning_rate = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f} | "
            f"dice={val_dice:.4f} | iou={metrics['Mean IoU']:.4f} | "
            f"precision={metrics['Precision']:.4f} | "
            f"recall={metrics['Recall / Sensitivity']:.4f} | "
            f"lr={learning_rate:.2e} | {seconds / 60:.1f} min",
            flush=True,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice_at_0.5": val_dice,
            "val_iou_at_0.5": metrics["Mean IoU"],
            "precision_at_0.5": metrics["Precision"],
            "recall_at_0.5": metrics["Recall / Sensitivity"],
            "learning_rate": learning_rate,
            "seconds": seconds,
        }
        history_rows.append(row)
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(list(row.values()))

        if val_dice > best_dice + config.min_delta:
            best_dice = val_dice
            epochs_without_improvement = 0
            torch.save(
                {
                    "architecture": "attention_unet3d",
                    "model_config": {
                        "in_channels": 4,
                        "out_channels": 1,
                        "base_channels": config.base_channels,
                        "num_heads": config.num_heads,
                    },
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_dice_at_0.5": val_dice,
                    "training_config": asdict(config),
                },
                best_checkpoint,
            )
            print(f"  Saved new best attention checkpoint (Dice {best_dice:.4f}).")
        else:
            epochs_without_improvement += 1
            print(
                f"  No improvement for {epochs_without_improvement}/{config.patience} epochs."
            )

        if epochs_without_improvement >= config.patience:
            print("Early stopping.")
            break

    if not best_checkpoint.exists():
        raise RuntimeError("Training ended without a checkpoint.")

    try:
        checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nTuning the final probability threshold on merged validation volumes...")
    threshold_values = np.arange(0.20, 0.81, 0.05).round(2).tolist()
    tuned_val_loss, threshold_results = validate_full_volumes(
        model,
        store,
        val_ids,
        criterion,
        device,
        thresholds=threshold_values,
        stride=config.val_stride,
    )
    best_threshold = max(
        threshold_results,
        key=lambda threshold: threshold_results[threshold]["Dice coefficient"],
    )
    tuned_metrics = threshold_results[best_threshold]

    final_metrics = {
        **tuned_metrics,
        "Threshold": float(best_threshold),
        "Epoch": int(checkpoint["epoch"]),
        "Validation loss": float(tuned_val_loss),
        "Depth": config.depth,
        "Image size": [config.image_height, config.image_width],
        "Train patients": len(train_ids),
        "Validation patients": len(val_ids),
        "Total patients": len(store.volume_ids),
        "Architecture": "Deeper residual 3D U-Net with bottleneck multi-head attention",
        "Attention heads": config.num_heads,
        "Base channels": config.base_channels,
        "Parameters": parameter_count,
        "Normalization": "per-modality non-zero z-score",
        "Validation": "patient-level split with per-patient overlapping-window merge",
        "Run folder": str(run_dir),
    }

    candidate_model = best_dir / "best_attention_unet3d.pth"
    candidate_metrics = best_dir / "best_attention_metrics_3d.json"
    candidate_history = best_dir / "best_attention_history_3d.csv"
    shutil.copy2(best_checkpoint, candidate_model)
    save_json(candidate_metrics, final_metrics)
    shutil.copy2(history_path, candidate_history)
    save_json(run_dir / "metrics_attention_3d.json", final_metrics)

    print("\n" + "=" * 78)
    print("Attention training complete. The current deployed baseline was NOT overwritten.")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best tuned Dice: {tuned_metrics['Dice coefficient']:.4f}")
    print(f"Candidate model: {candidate_model}")
    print(f"Candidate metrics: {candidate_metrics}")
    print("After comparing it with the baseline, promote it with:")
    print("  python3 promote_attention_model.py")
    print("=" * 78)


if __name__ == "__main__":
    main()
