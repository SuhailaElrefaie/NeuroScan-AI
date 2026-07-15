import csv
import json
import os
import shutil
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import BrainTumorSegmentationDataset
from models.unet import UNet


# Dataset paths
IMAGE_DIR = "Data/segmentation_task/train/images"
MASK_DIR = "Data/segmentation_task/train/masks"

# Output folders
RUNS_DIR = "runs"
BEST_DIR = "best_model"

# Best model files
BEST_MODEL_PATH = os.path.join(BEST_DIR, "best_unet.pth")
BEST_METRICS_PATH = os.path.join(BEST_DIR, "best_metrics.json")
BEST_HISTORY_PATH = os.path.join(BEST_DIR, "best_history.csv")

# Safe backup of your 0.8397 model
BACKUP_MODEL_PATH = os.path.join(BEST_DIR, "backup_08397.pth")

# Training settings
BATCH_SIZE = 2
EPOCHS = 80
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
SUBSET_SIZE = None
SEED = 42
THRESHOLD = 0.3


def dice_coefficient(preds, targets, smooth=1e-6): # Calculate Dice coefficient between predicted and true masks.
    preds = torch.sigmoid(preds)
    preds = (preds > THRESHOLD).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()

    return (2.0 * intersection + smooth) / (
        preds.sum() + targets.sum() + smooth
    )


def dice_loss(preds, targets, smooth=1e-6): # Calculate Dice loss for segmentation training.
    preds = torch.sigmoid(preds)

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        preds.sum(dim=1) + targets.sum(dim=1) + smooth
    )

    return 1.0 - dice.mean()


def combined_loss(preds, targets, device): # Combined loss: weighted BCE + Dice loss.
    pos_weight = torch.tensor([10.0], device=device)

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(preds, targets)
    d_loss = dice_loss(preds, targets)

    return bce + d_loss


def get_confusion_counts(logits, masks, threshold=THRESHOLD): # Calculate TP, FP, FN, TN for binary segmentation.
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    masks = (masks > 0.5).float()

    tp = (preds * masks).sum().item()
    fp = (preds * (1 - masks)).sum().item()
    fn = ((1 - preds) * masks).sum().item()
    tn = ((1 - preds) * (1 - masks)).sum().item()

    return tp, fp, fn, tn


def compute_metrics_from_counts(tp, fp, fn, tn, eps=1e-7): # Convert TP, FP, FN, TN into useful segmentation metrics.
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


def get_device(): # Select the best available device.
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def make_run_folder(): # Create a unique folder for this training run.
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(BEST_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    return run_dir


def load_previous_best_dice(): #  Load the previous best Dice score if it exists.
    if not os.path.exists(BEST_METRICS_PATH):
        return 0.0

    with open(BEST_METRICS_PATH, "r") as file:
        metrics = json.load(file)

    return float(metrics.get("Dice coefficient", 0.0))


def save_json(path, data): # Save dictionary as JSON.
    with open(path, "w") as file:
        json.dump(data, file, indent=4)


def main(): # Train the U-Net segmentation model.
    torch.manual_seed(SEED)

    device = get_device()
    print("Using device:", device)

    run_dir = make_run_folder()

    run_model_path = os.path.join(run_dir, "model.pth")
    run_history_path = os.path.join(run_dir, "history.csv")
    run_metrics_path = os.path.join(run_dir, "metrics.json")

    print("Run folder:", run_dir)
    print("Loading dataset...")

    train_full_dataset = BrainTumorSegmentationDataset(
        IMAGE_DIR,
        MASK_DIR,
        augment=True
    )

    val_full_dataset = BrainTumorSegmentationDataset(
        IMAGE_DIR,
        MASK_DIR,
        augment=False
    )

    print("Original dataset size:", len(train_full_dataset))

    if SUBSET_SIZE is None:
        all_indices = torch.randperm(len(train_full_dataset)).tolist()
    else:
        subset_size = min(SUBSET_SIZE, len(train_full_dataset))
        all_indices = torch.randperm(len(train_full_dataset))[:subset_size].tolist()

    print("Using dataset size:", len(all_indices))

    val_size = int(len(all_indices) * VAL_SPLIT)
    train_size = len(all_indices) - val_size

    if train_size == 0 or val_size == 0:
        raise ValueError("Dataset split is invalid. Check dataset size and VAL_SPLIT.")

    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]

    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(val_full_dataset, val_indices)

    print("Train size:", len(train_dataset))
    print("Validation size:", len(val_dataset))
    print("Starting training...")

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

    model = UNet().to(device)
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

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch + 1}, "
                    f"Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        model.eval()
        val_loss_total = 0.0
        val_dice_total = 0.0

        saved_outputs = []
        saved_masks = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = combined_loss(outputs, masks, device)

                val_loss_total += loss.item()
                val_dice_total += dice_coefficient(outputs, masks).item()

                saved_outputs.append(outputs.detach().cpu())
                saved_masks.append(masks.detach().cpu())

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

            total_tp = 0.0
            total_fp = 0.0
            total_fn = 0.0
            total_tn = 0.0

            for outputs_cpu, masks_cpu in zip(saved_outputs, saved_masks):
                tp, fp, fn, tn = get_confusion_counts(
                    outputs_cpu,
                    masks_cpu,
                    threshold=THRESHOLD
                )

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

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
                "Run folder": run_dir
            }

            save_json(run_metrics_path, best_run_metrics)

            print(f"Saved best model for this run to {run_model_path}")
            print(f"Saved run metrics to {run_metrics_path}")

    previous_best_dice = load_previous_best_dice()

    print("Training finished.")
    print(f"Best Dice in this run: {best_run_dice:.4f}")
    print(f"Previous best Dice: {previous_best_dice:.4f}")

    if best_run_metrics is not None and best_run_dice > previous_best_dice:
        shutil.copy(run_model_path, BEST_MODEL_PATH)
        shutil.copy(run_metrics_path, BEST_METRICS_PATH)
        shutil.copy(run_history_path, BEST_HISTORY_PATH)

        print("New best model found.")
        print(f"Updated {BEST_MODEL_PATH}")
        print(f"Updated {BEST_METRICS_PATH}")
        print(f"Updated {BEST_HISTORY_PATH}")
    else:
        print("This run did not beat the current best model.")
        print("Best model was not changed.")


if __name__ == "__main__":
    main()