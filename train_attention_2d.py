import csv
import json
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import BrainTumorSegmentationDataset
from models.attention_unet2d import AttentionUNet2D


IMAGE_DIR = "Data/segmentation_task/train/images"
MASK_DIR = "Data/segmentation_task/train/masks"

RUNS_DIR = "runs_attention_2d"
BEST_DIR = "best_model"

BEST_MODEL_PATH = os.path.join(
    BEST_DIR,
    "best_attention_unet2d.pth"
)

BEST_METRICS_PATH = os.path.join(
    BEST_DIR,
    "best_attention_metrics_2d.json"
)

BEST_HISTORY_PATH = os.path.join(
    BEST_DIR,
    "best_attention_history_2d.csv"
)


BATCH_SIZE = 2
EPOCHS = 80
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.20
SEED = 42
THRESHOLD = 0.30
PATIENCE = 12


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def dice_loss(logits, targets, smooth=1e-6):

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.view(
        probabilities.size(0), -1
    )

    targets = targets.view(
        targets.size(0), -1
    )

    intersection = (
        probabilities * targets
    ).sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        probabilities.sum(dim=1)
        + targets.sum(dim=1)
        + smooth
    )

    return 1.0 - dice.mean()


def combined_loss(logits, targets, device):

    pos_weight = torch.tensor(
        [10.0],
        device=device
    )

    bce = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )(logits, targets)

    return bce + dice_loss(
        logits,
        targets
    )


def confusion_counts(
    logits,
    masks,
    threshold=THRESHOLD
):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities > threshold
    ).float()

    masks = (masks > 0.5).float()

    tp = (
        predictions * masks
    ).sum().item()

    fp = (
        predictions * (1 - masks)
    ).sum().item()

    fn = (
        (1 - predictions) * masks
    ).sum().item()

    return tp, fp, fn


def metrics_from_counts(tp, fp, fn):

    eps = 1e-7

    dice = (
        2 * tp
    ) / (
        2 * tp + fp + fn + eps
    )

    iou = tp / (
        tp + fp + fn + eps
    )

    precision = tp / (
        tp + fp + eps
    )

    recall = tp / (
        tp + fn + eps
    )

    return {
        "Dice coefficient": dice,
        "Mean IoU": iou,
        "Precision": precision,
        "Recall / Sensitivity": recall
    }


@torch.no_grad()
def validate(
    model,
    loader,
    device
):

    model.eval()

    total_loss = 0.0

    tp = 0.0
    fp = 0.0
    fn = 0.0

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        loss = combined_loss(
            logits,
            masks,
            device
        )

        total_loss += loss.item()

        batch_tp, batch_fp, batch_fn = (
            confusion_counts(
                logits,
                masks
            )
        )

        tp += batch_tp
        fp += batch_fp
        fn += batch_fn

    metrics = metrics_from_counts(
        tp,
        fp,
        fn
    )

    return (
        total_loss / max(len(loader), 1),
        metrics
    )


def main():

    torch.manual_seed(SEED)

    device = get_device()

    print("=" * 70)
    print("2D Multi-Head Attention U-Net")
    print("Device:", device)
    print("=" * 70)

    os.makedirs(
        RUNS_DIR,
        exist_ok=True
    )

    os.makedirs(
        BEST_DIR,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y_%m_%d_%H%M%S"
    )

    run_dir = os.path.join(
        RUNS_DIR,
        f"run_attention_2d_{timestamp}"
    )

    os.makedirs(
        run_dir,
        exist_ok=True
    )

    train_full = (
        BrainTumorSegmentationDataset(
            IMAGE_DIR,
            MASK_DIR,
            augment=True
        )
    )

    val_full = (
        BrainTumorSegmentationDataset(
            IMAGE_DIR,
            MASK_DIR,
            augment=False
        )
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    indices = torch.randperm(
        len(train_full),
        generator=generator
    ).tolist()

    val_size = int(
        len(indices) * VAL_SPLIT
    )

    train_indices = indices[:-val_size]
    val_indices = indices[-val_size:]

    train_dataset = Subset(
        train_full,
        train_indices
    )

    val_dataset = Subset(
        val_full,
        val_indices
    )

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

    print(
        "Training images:",
        len(train_dataset)
    )

    print(
        "Validation images:",
        len(val_dataset)
    )

    model = AttentionUNet2D(
        in_channels=1,
        out_channels=1,
        num_heads=8
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=1e-6
        )
    )

    history = []

    best_dice = -1.0
    no_improvement = 0

    run_best_path = os.path.join(
        run_dir,
        "best_attention_unet2d.pth"
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        for images, masks in train_loader:

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            logits = model(images)

            loss = combined_loss(
                logits,
                masks,
                device
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            running_loss += loss.item()

        train_loss = (
            running_loss
            / max(len(train_loader), 1)
        )

        val_loss, metrics = validate(
            model,
            val_loader,
            device
        )

        dice = metrics[
            "Dice coefficient"
        ]

        scheduler.step(dice)

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train={train_loss:.4f} | "
            f"val={val_loss:.4f} | "
            f"dice={dice:.4f} | "
            f"iou={metrics['Mean IoU']:.4f} | "
            f"precision={metrics['Precision']:.4f} | "
            f"recall={metrics['Recall / Sensitivity']:.4f} | "
            f"lr={current_lr:.2e}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": dice,
            "val_iou": metrics[
                "Mean IoU"
            ],
            "precision": metrics[
                "Precision"
            ],
            "recall": metrics[
                "Recall / Sensitivity"
            ],
            "learning_rate": current_lr
        })

        if dice > best_dice:

            best_dice = dice
            no_improvement = 0

            torch.save(
                model.state_dict(),
                run_best_path
            )

            print(
                f"Saved new best model: "
                f"Dice={best_dice:.4f}"
            )

        else:

            no_improvement += 1

        if no_improvement >= PATIENCE:

            print("Early stopping.")
            break

    history_path = os.path.join(
        run_dir,
        "history.csv"
    )

    with open(
        history_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=history[0].keys()
        )

        writer.writeheader()
        writer.writerows(history)

    model.load_state_dict(
        torch.load(
            run_best_path,
            map_location=device
        )
    )

    _, final_metrics = validate(
        model,
        val_loader,
        device
    )

    final_metrics[
        "Threshold"
    ] = THRESHOLD

    final_metrics[
        "Architecture"
    ] = "2D U-Net + Multi-Head Attention"

    final_metrics[
        "Attention heads"
    ] = 8

    final_metrics[
        "Training images"
    ] = len(train_dataset)

    final_metrics[
        "Validation images"
    ] = len(val_dataset)

    final_metrics[
        "Best Dice"
    ] = best_dice

    metrics_path = os.path.join(
        run_dir,
        "metrics.json"
    )

    with open(
        metrics_path,
        "w"
    ) as f:

        json.dump(
            final_metrics,
            f,
            indent=4
        )

    import shutil

    shutil.copy2(
        run_best_path,
        BEST_MODEL_PATH
    )

    shutil.copy2(
        history_path,
        BEST_HISTORY_PATH
    )

    shutil.copy2(
        metrics_path,
        BEST_METRICS_PATH
    )

    print()
    print("=" * 70)
    print("Attention training complete")
    print("Best Dice:", best_dice)
    print("Model:", BEST_MODEL_PATH)
    print("Metrics:", BEST_METRICS_PATH)
    print("History:", BEST_HISTORY_PATH)
    print("=" * 70)


if __name__ == "__main__":
    main()
