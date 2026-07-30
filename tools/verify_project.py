"""Read-only integrity checks for the NeuroScan AI repository."""
from __future__ import annotations

import csv
import json
import py_compile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path.relative_to(ROOT)}")
    print(f"OK  {path.relative_to(ROOT)}")


def check_python_files() -> None:
    print("\nPython syntax")
    for relative in [
        "ui.py",
        "predict.py",
        "full_volume_3d.py",
        "train.py",
        "train_3d.py",
        "dataset.py",
        "dataset_3d.py",
        "gradcam.py",
        "models/unet.py",
        "models/unet3d.py",
    ]:
        path = ROOT / relative
        require(path)
        py_compile.compile(str(path), doraise=True)


def check_saved_results() -> None:
    print("\nSaved results")
    metrics_path = ROOT / "best_model_3d/best_metrics_3d.json"
    history_path = ROOT / "best_model_3d/best_history_3d.csv"
    model_path = ROOT / "best_model_3d/best_unet3d.pth"
    for path in [metrics_path, history_path, model_path]:
        require(path)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    required_metric_keys = {
        "Dice coefficient",
        "Mean IoU",
        "Precision",
        "Recall / Sensitivity",
        "Threshold",
        "Epoch",
    }
    missing = required_metric_keys.difference(metrics)
    if missing:
        raise KeyError(f"3D metrics missing keys: {sorted(missing)}")

    with history_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = set(reader.fieldnames or [])

    if not rows:
        raise ValueError("3D history CSV is empty.")
    if "epoch" not in columns or "val_loss" not in columns:
        raise KeyError(f"Unexpected 3D history columns: {sorted(columns)}")
    if not ({"val_dice", "val_dice_at_0.5"} & columns):
        raise KeyError("3D history has no validation Dice column.")


def check_samples() -> None:
    print("\nPublic samples (read-only)")
    two_d = sorted((ROOT / "sample_data/2d").glob("*"))
    three_d = sorted((ROOT / "sample_data/3d").glob("*.npz"))

    if len(two_d) < 5:
        raise ValueError(f"Expected at least 5 public 2D samples; found {len(two_d)}")
    if len(three_d) != 5:
        raise ValueError(f"Expected exactly 5 public 3D NPZ samples; found {len(three_d)}")

    print(f"OK  sample_data/2d ({len(two_d)} files)")

    for path in three_d:
        with np.load(path, allow_pickle=False) as data:
            if "image" not in data.files:
                raise KeyError(f"{path.name} does not contain an image array")

            image = np.asarray(data["image"])
            if image.ndim != 4 or 4 not in (image.shape[0], image.shape[-1]):
                raise ValueError(f"Unexpected MRI shape in {path.name}: {image.shape}")

            if image.shape[0] == 4:
                depth_shape = image.shape[1:]
            else:
                depth_shape = image.shape[:3]

            if "mask" in data.files:
                mask = np.squeeze(np.asarray(data["mask"]))
                if mask.shape != depth_shape:
                    raise ValueError(
                        f"Mask/image mismatch in {path.name}: {mask.shape} vs {depth_shape}"
                    )

        print(f"OK  sample_data/3d/{path.name} {tuple(image.shape)}")


def main() -> None:
    check_python_files()
    check_saved_results()
    check_samples()
    print("\nAll checks passed. No files were modified.")


if __name__ == "__main__":
    main()
