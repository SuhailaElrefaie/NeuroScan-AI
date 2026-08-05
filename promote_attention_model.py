"""Promote a validated attention checkpoint to the paths used by ui.py."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import torch

from models.attention_unet3d import AttentionUNet3D


BEST_DIR = Path("best_model_3d")
CANDIDATE_MODEL = BEST_DIR / "best_attention_unet3d.pth"
CANDIDATE_METRICS = BEST_DIR / "best_attention_metrics_3d.json"
CANDIDATE_HISTORY = BEST_DIR / "best_attention_history_3d.csv"
DEPLOYED_MODEL = BEST_DIR / "best_unet3d.pth"
DEPLOYED_METRICS = BEST_DIR / "best_metrics_3d.json"
DEPLOYED_HISTORY = BEST_DIR / "best_history_3d.csv"


def main() -> None:
    required = [CANDIDATE_MODEL, CANDIDATE_METRICS, CANDIDATE_HISTORY]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing attention training outputs: " + ", ".join(missing))

    try:
        checkpoint = torch.load(CANDIDATE_MODEL, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(CANDIDATE_MODEL, map_location="cpu")

    if checkpoint.get("architecture") != "attention_unet3d":
        raise ValueError("Candidate checkpoint is not marked as attention_unet3d.")

    model_config = checkpoint.get("model_config", {})
    model = AttentionUNet3D(
        in_channels=int(model_config.get("in_channels", 4)),
        out_channels=int(model_config.get("out_channels", 1)),
        base_channels=int(model_config.get("base_channels", 16)),
        num_heads=int(model_config.get("num_heads", 8)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metrics = json.loads(CANDIDATE_METRICS.read_text(encoding="utf-8"))
    dice = float(metrics.get("Dice coefficient", 0.0))
    print(f"Candidate validation Dice: {dice:.4f}")
    confirmation = input("Type DEPLOY to replace the current 3D model: ").strip()
    if confirmation != "DEPLOY":
        print("Cancelled. No files were changed.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BEST_DIR / f"baseline_backup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for path in (DEPLOYED_MODEL, DEPLOYED_METRICS, DEPLOYED_HISTORY):
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)

    shutil.copy2(CANDIDATE_MODEL, DEPLOYED_MODEL)
    shutil.copy2(CANDIDATE_METRICS, DEPLOYED_METRICS)
    shutil.copy2(CANDIDATE_HISTORY, DEPLOYED_HISTORY)

    print("Attention model deployed.")
    print(f"Previous deployed files backed up to: {backup_dir}")
    print("Now run: python3 tools/verify_project.py")
    print("Then run: streamlit run ui.py")


if __name__ == "__main__":
    main()
