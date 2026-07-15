import csv
import json
import os
import shutil
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset_3d import BraTS3DVolumeDataset
from models.unet3d import UNet3D


H5_DIR = "archive/BraTS2020_training_data/content/data"

RUNS_DIR = "runs_3d"
BEST_DIR = "best_model_3d"

BEST_MODEL_PATH = os.path.join(BEST_DIR, "best_unet3d.pth")
BEST_METRICS_PATH = os.path.join(BEST_DIR, "best_metrics_3d.json")
BEST_HISTORY_PATH = os.path.join(BEST_DIR, "best_history_3d.csv")

BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
SEED = 42
THRESHOLD = 0.5

DEPTH = 16
IMAGE_SIZE = (96,96)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def dice_coefficient(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > THRESHOLD).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()

    return (2.0 * intersection + smooth) / (
        preds.sum() + targets.sum() + smooth
    )


def dice_loss(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)

    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        probs.sum(dim=1) + targets.sum(dim=1) + smooth
    )

    return 1.0 - dice.mean()


def combined_loss(logits, targets, device):
    pos_weight = torch.tensor([10.0], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, targets)
    d_loss = dice_loss(logits, targets)

    return bce + d_loss


def get_confusion_counts(logits, masks):
    probs = torch.sigmoid(logits)
    preds = (probs > THRESHOLD).float()
    masks = (masks > 0.5).float()

    tp = (preds * masks).sum().item()
    fp = (preds * (1 - masks)).sum().item()
    fn = ((1 - preds) * masks).sum().item()
    tn = ((1 - preds) * (1 - masks)).sum().item()

    return tp, fp, fn, tn


def compute_metrics_from_counts(tp, fp, fn, tn, eps=1e-7):
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    return {
        "Dice coefficient": dice,
        "Mean IoU": iou,
        "Precision": precision,
        "Recall / Sensitivity": recall
    }


def save_json(path, data):
    with open(path, "w") as file:
        json.dump(data, file, indent=4)


def make_run_folder():
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(BEST_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, f"run_3d_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    return run_dir


def load_previous_best_dice():
    if not os.path.exists(BEST_METRICS_PATH):
        return 0.0

    with open(BEST_METRICS_PATH, "r") as file:
        metrics = json.load(file)

    return float(metrics.get("Dice coefficient", 0.0))


def main():
    torch.manual_seed(SEED)

    device = get_device()
    print("Using device:", device)

    run_dir = make_run_folder()

    run_model_path = os.path.join(run_dir, "model_3d.pth")
    run_history_path = os.path.join(run_dir, "history_3d.csv")
    run_metrics_path = os.path.join(run_dir, "metrics_3d.json")

    print("Loading 3D H5 dataset...")

    train_full_dataset = BraTS3DVolumeDataset(
        h5_dir=H5_DIR,
        depth=DEPTH,
        image_size=IMAGE_SIZE,
        augment=True,
        only_tumor_windows=True
    )

    val_full_dataset = BraTS3DVolumeDataset(
        h5_dir=H5_DIR,
        depth=DEPTH,
        image_size=IMAGE_SIZE,
        augment=False,
        only_tumor_windows=True
    )

    all_indices = torch.randperm(
        len(train_full_dataset),
        generator=torch.Generator().manual_seed(SEED)
    ).tolist()

    val_size = int(len(all_indices) * VAL_SPLIT)
    train_size = len(all_indices) - val_size

    if train_size == 0 or val_size == 0:
        raise ValueError(
            "Not enough 3D volumes for train/validation split. "
            "Use more volume files."
        )

    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]

    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(val_full_dataset, val_indices)

    print("Train volumes:", len(train_dataset))
    print("Validation volumes:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    sample_image, _ = train_full_dataset[0]
    in_channels = sample_image.shape[0]

    print("3D input channels:", in_channels)
    print("3D input shape:", sample_image.shape)

    model = UNet3D(in_channels=in_channels, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_run_dice = 0.0
    best_run_metrics = None

    with open(run_history_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice"])

    for epoch in range(EPOCHS):
        model.train()
        train_loss_total = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = combined_loss(outputs, masks, device)

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

            print(
                f"Epoch {epoch + 1}, "
                f"Batch {batch_idx + 1}/{len(train_loader)}, "
                f"Loss: {loss.item():.4f}"
            )

        model.eval()
        val_loss_total = 0.0
        val_dice_total = 0.0

        total_tp = 0.0
        total_fp = 0.0
        total_fn = 0.0
        total_tn = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = combined_loss(outputs, masks, device)

                val_loss_total += loss.item()
                val_dice_total += dice_coefficient(outputs, masks).item()

                tp, fp, fn, tn = get_confusion_counts(outputs, masks)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

        avg_train_loss = train_loss_total / len(train_loader)
        avg_val_loss = val_loss_total / len(val_loader)
        avg_val_dice = val_dice_total / len(val_loader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Dice: {avg_val_dice:.4f}"
        )

        with open(run_history_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1,
                avg_train_loss,
                avg_val_loss,
                avg_val_dice
            ])

        if avg_val_dice > best_run_dice:
            best_run_dice = avg_val_dice
            torch.save(model.state_dict(), run_model_path)

            extra_metrics = compute_metrics_from_counts(
                total_tp,
                total_fp,
                total_fn,
                total_tn
            )

            best_run_metrics = {
                "Validation loss": avg_val_loss,
                "Dice coefficient": avg_val_dice,
                "Mean IoU": extra_metrics["Mean IoU"],
                "Precision": extra_metrics["Precision"],
                "Recall / Sensitivity": extra_metrics["Recall / Sensitivity"],
                "Epoch": epoch + 1,
                "Threshold": THRESHOLD,
                "Depth": DEPTH,
                "Image size": IMAGE_SIZE,
                "Run folder": run_dir
            }

            save_json(run_metrics_path, best_run_metrics)

            print(f"Saved best 3D model for this run to {run_model_path}")
            print(f"Saved 3D run metrics to {run_metrics_path}")

    previous_best_dice = load_previous_best_dice()

    print("3D training finished.")
    print(f"Best Dice in this 3D run: {best_run_dice:.4f}")
    print(f"Previous best 3D Dice: {previous_best_dice:.4f}")

    if best_run_metrics is not None and best_run_dice > previous_best_dice:
        shutil.copy(run_model_path, BEST_MODEL_PATH)
        shutil.copy(run_metrics_path, BEST_METRICS_PATH)
        shutil.copy(run_history_path, BEST_HISTORY_PATH)

        print("New best 3D model found.")
        print(f"Updated {BEST_MODEL_PATH}")
        print(f"Updated {BEST_METRICS_PATH}")
        print(f"Updated {BEST_HISTORY_PATH}")
    else:
        print("This 3D run did not beat the current best 3D model.")


if __name__ == "__main__":
    main()