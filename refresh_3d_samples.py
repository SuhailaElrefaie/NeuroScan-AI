"""Preserve current Sample 5 as Sample 1 and select four better 3D samples.

The new samples are selected using the deployed v2 model's actual full-volume
predictions. The script evaluates a diverse candidate pool, calculates Dice,
precision, recall and IoU against the BraTS ground truth, then chooses four
high-quality patients with different tumour sizes.

This script only changes sample_data/3d. It never touches sample_data/2d.
"""
from __future__ import annotations

import csv
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from full_volume_3d import (
    MODEL_DEPTH,
    MODEL_HEIGHT,
    MODEL_WIDTH,
    WINDOW_STRIDE,
    _normalise_like_training,
    _restore_probability,
    _window_starts,
)
from models.unet3d import UNet3D


H5_DIR = Path("archive/BraTS2021_h5/content/data")
SAMPLE_DIR = Path("sample_data/3d")
MODEL_PATH = Path("best_model_3d/best_unet3d.pth")
REPORT_PATH = SAMPLE_DIR / "sample_selection_report.csv"

THRESHOLD = 0.55
CANDIDATE_COUNT = 48
RANDOM_SEED = 42

PATTERN = re.compile(r"volume_(\d+)_slice_(\d+)\.h5$")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_subjects() -> Dict[int, List[Tuple[int, Path]]]:
    subjects: Dict[int, List[Tuple[int, Path]]] = defaultdict(list)

    for path in H5_DIR.glob("volume_*_slice_*.h5"):
        match = PATTERN.match(path.name)
        if match:
            volume_id = int(match.group(1))
            slice_id = int(match.group(2))
            subjects[volume_id].append((slice_id, path))

    for volume_id in subjects:
        subjects[volume_id].sort(key=lambda item: item[0])

    return dict(subjects)


def load_subject(files: List[Tuple[int, Path]]) -> Tuple[np.ndarray, np.ndarray]:
    images = []
    masks = []

    for _, path in files:
        try:
            with h5py.File(path, "r") as handle:
                image = np.asarray(handle["image"][()], dtype=np.float32)
                mask = np.asarray(handle["mask"][()])
        except (OSError, KeyError) as error:
            print(f"Skipping unreadable file {path.name}: {error}")
            continue

        if image.ndim != 3 or image.shape[-1] != 4:
            raise ValueError(f"Unexpected image shape in {path}: {image.shape}")

        if mask.ndim == 3:
            mask = np.any(mask > 0, axis=-1)
        else:
            mask = mask > 0

        images.append(np.moveaxis(image, -1, 0))
        masks.append(mask.astype(np.uint8))

    if not images:
        raise RuntimeError("No readable slices found for this subject.")

    return np.stack(images, axis=1), np.stack(masks, axis=0)


def scan_tumour_sizes(
    subjects: Dict[int, List[Tuple[int, Path]]]
) -> List[Tuple[int, int, int]]:
    """Return (tumour_voxels, tumour_slices, volume_id)."""
    ranked = []

    for number, (volume_id, files) in enumerate(sorted(subjects.items()), start=1):
        tumour_voxels = 0
        tumour_slices = 0

        for _, path in files:
            try:
                with h5py.File(path, "r") as handle:
                    mask = np.asarray(handle["mask"][()])
            except (OSError, KeyError):
                continue

            binary = mask > 0
            count = int(binary.sum())
            tumour_voxels += count
            tumour_slices += int(count > 0)

        if tumour_voxels > 0:
            ranked.append((tumour_voxels, tumour_slices, volume_id))

        if number % 100 == 0:
            print(f"Scanned {number}/{len(subjects)} patients...")

    ranked.sort()
    return ranked


def load_model(device: torch.device) -> UNet3D:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}")

    model = UNet3D(in_channels=4, out_channels=1).to(device)

    try:
        state_dict = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(MODEL_PATH, map_location=device)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def prepare_window(window: np.ndarray) -> torch.Tensor:
    original_depth = window.shape[1]

    if original_depth < MODEL_DEPTH:
        pad = MODEL_DEPTH - original_depth
        window = np.pad(
            window,
            ((0, 0), (0, pad), (0, 0), (0, 0)),
            mode="edge",
        )

    window = _normalise_like_training(window)
    tensor = torch.from_numpy(window).unsqueeze(0)

    tensor = F.interpolate(
        tensor,
        size=(MODEL_DEPTH, MODEL_HEIGHT, MODEL_WIDTH),
        mode="trilinear",
        align_corners=False,
    )
    return tensor


@torch.no_grad()
def predict_subject(
    model: UNet3D,
    image: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    _, depth, height, width = image.shape
    probability_sum = np.zeros((depth, height, width), dtype=np.float32)
    probability_count = np.zeros((depth, height, width), dtype=np.float32)

    for start in _window_starts(depth, MODEL_DEPTH, WINDOW_STRIDE):
        end = min(start + MODEL_DEPTH, depth)
        valid_depth = end - start

        network_input = prepare_window(image[:, start:end]).to(device)
        logits = model(network_input)
        probability = torch.sigmoid(logits).squeeze(0).squeeze(0)
        restored = _restore_probability(probability, height, width)[:valid_depth]

        probability_sum[start:end] += restored
        probability_count[start:end] += 1.0

    probability_count[probability_count == 0] = 1.0
    return probability_sum / probability_count


def calculate_metrics(
    probability: np.ndarray,
    truth: np.ndarray,
) -> Dict[str, float]:
    prediction = probability >= THRESHOLD
    truth = truth > 0

    tp = int(np.logical_and(prediction, truth).sum())
    fp = int(np.logical_and(prediction, ~truth).sum())
    fn = int(np.logical_and(~prediction, truth).sum())

    eps = 1e-7
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    predicted_slices = int(np.any(prediction, axis=(1, 2)).sum())
    true_slices = int(np.any(truth, axis=(1, 2)).sum())

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "predicted_voxels": int(prediction.sum()),
        "true_voxels": int(truth.sum()),
        "predicted_slices": predicted_slices,
        "true_slices": true_slices,
    }


def current_sample_five() -> Path:
    files = sorted(SAMPLE_DIR.glob("*.npz"))
    if len(files) < 5:
        raise RuntimeError(
            f"Expected at least five existing NPZ samples in {SAMPLE_DIR.resolve()}."
        )
    return files[4]


def npz_volume_id(path: Path) -> int | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            if "volume_id" in data.files:
                return int(data["volume_id"].item())
    except (OSError, ValueError, KeyError):
        pass
    return None


def choose_candidate_ids(
    ranked: List[Tuple[int, int, int]],
    excluded_ids: set[int],
) -> List[int]:
    eligible = [
        row for row in ranked
        if row[2] not in excluded_ids and 6 <= row[1] <= 80
    ]

    if len(eligible) <= CANDIDATE_COUNT:
        return [row[2] for row in eligible]

    # Evenly cover small through large tumours.
    positions = np.linspace(
        0,
        len(eligible) - 1,
        num=CANDIDATE_COUNT,
        dtype=int,
    )
    return [eligible[position][2] for position in positions]


def select_diverse_best(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Pick one strong result from each tumour-size quartile."""
    acceptable = [
        result for result in results
        if result["dice"] >= 0.82
        and result["precision"] >= 0.78
        and result["recall"] >= 0.78
    ]

    if len(acceptable) < 4:
        acceptable = sorted(results, key=lambda row: row["dice"], reverse=True)[:16]

    acceptable.sort(key=lambda row: row["true_voxels"])
    bins = np.array_split(np.asarray(acceptable, dtype=object), 4)

    chosen = []
    used_ids = set()

    for bin_rows in bins:
        rows = list(bin_rows)
        rows.sort(
            key=lambda row: (
                row["dice"],
                min(row["precision"], row["recall"]),
            ),
            reverse=True,
        )

        for row in rows:
            if row["volume_id"] not in used_ids:
                chosen.append(row)
                used_ids.add(row["volume_id"])
                break

    if len(chosen) < 4:
        for row in sorted(results, key=lambda value: value["dice"], reverse=True):
            if row["volume_id"] not in used_ids:
                chosen.append(row)
                used_ids.add(row["volume_id"])
            if len(chosen) == 4:
                break

    return chosen[:4]


def save_sample(
    destination: Path,
    image: np.ndarray,
    mask: np.ndarray,
    volume_id: int,
) -> None:
    np.savez_compressed(
        destination,
        image=image,
        mask=mask,
        volume_id=np.asarray(volume_id),
        modality_names=np.asarray(["FLAIR", "T1", "T1CE", "T2"]),
    )


def main() -> None:
    if not H5_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {H5_DIR.resolve()}")

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    preserved_source = current_sample_five()
    preserved_volume_id = npz_volume_id(preserved_source)

    print(f"Preserving current Sample 5: {preserved_source.name}")
    if preserved_volume_id is not None:
        print(f"Preserved patient volume ID: {preserved_volume_id}")

    with tempfile.TemporaryDirectory() as temporary_directory:
        preserved_copy = Path(temporary_directory) / "preserved_sample_5.npz"
        shutil.copy2(preserved_source, preserved_copy)

        subjects = collect_subjects()
        print(f"Found {len(subjects)} patient volumes.")

        ranked = scan_tumour_sizes(subjects)
        excluded = {preserved_volume_id} if preserved_volume_id is not None else set()
        candidate_ids = choose_candidate_ids(ranked, excluded)

        print(f"Evaluating {len(candidate_ids)} diverse candidate patients...")
        device = get_device()
        print(f"Using device: {device}")
        model = load_model(device)

        evaluated = []

        for index, volume_id in enumerate(candidate_ids, start=1):
            image, mask = load_subject(subjects[volume_id])
            probability = predict_subject(model, image, device)
            metrics = calculate_metrics(probability, mask)
            metrics["volume_id"] = volume_id
            metrics["depth"] = image.shape[1]
            metrics["image"] = image
            metrics["mask"] = mask
            evaluated.append(metrics)

            print(
                f"[{index:02d}/{len(candidate_ids)}] volume={volume_id} "
                f"dice={metrics['dice']:.4f} "
                f"precision={metrics['precision']:.4f} "
                f"recall={metrics['recall']:.4f} "
                f"tumour_slices={metrics['true_slices']}"
            )

        chosen = select_diverse_best(evaluated)
        if len(chosen) != 4:
            raise RuntimeError("Could not select four replacement samples.")

        # Clear only NPZ samples after all selections have safely completed.
        for old_file in SAMPLE_DIR.glob("*.npz"):
            old_file.unlink()

        sample_one = SAMPLE_DIR / "sample_3d_full_01.npz"
        shutil.copy2(preserved_copy, sample_one)
        print(f"Saved preserved Sample 5 as new Sample 1: {sample_one}")

        report_rows = [{
            "sample_number": 1,
            "volume_id": preserved_volume_id if preserved_volume_id is not None else "preserved",
            "dice": "preserved existing sample",
            "iou": "",
            "precision": "",
            "recall": "",
            "true_voxels": "",
            "true_slices": "",
        }]

        for sample_number, result in enumerate(chosen, start=2):
            destination = SAMPLE_DIR / f"sample_3d_full_{sample_number:02d}.npz"
            save_sample(
                destination,
                result.pop("image"),
                result.pop("mask"),
                int(result["volume_id"]),
            )

            report_rows.append({
                "sample_number": sample_number,
                "volume_id": result["volume_id"],
                "dice": round(result["dice"], 4),
                "iou": round(result["iou"], 4),
                "precision": round(result["precision"], 4),
                "recall": round(result["recall"], 4),
                "true_voxels": result["true_voxels"],
                "true_slices": result["true_slices"],
            })

            print(
                f"Saved Sample {sample_number}: volume={result['volume_id']} "
                f"dice={result['dice']:.4f}"
            )

        with REPORT_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=report_rows[0].keys())
            writer.writeheader()
            writer.writerows(report_rows)

    print("\nFinished.")
    print("Sample 1 is the old Sample 5.")
    print("Samples 2–5 are newly selected high-quality, diverse patients.")
    print(f"Selection report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
