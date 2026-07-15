from pathlib import Path
import re
from collections import defaultdict

ROOT = Path(".")

def image_files(path):
    path = ROOT / path
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

def count_pairs(img_dir, mask_dir):
    imgs = image_files(img_dir)
    masks = image_files(mask_dir)

    img_keys = {p.stem for p in imgs}
    mask_keys = {p.stem for p in masks}
    matched = img_keys & mask_keys

    return {
        "image_dir": img_dir,
        "mask_dir": mask_dir,
        "raw_images": len(imgs),
        "raw_masks": len(masks),
        "matched_pairs_used_by_code": len(matched),
        "images_without_mask": len(img_keys - mask_keys),
        "masks_without_image": len(mask_keys - img_keys),
    }

print("\n===== 2D DATASET COUNTS =====")
for split in ["train", "test"]:
    result = count_pairs(
        f"Data/segmentation_task/{split}/images",
        f"Data/segmentation_task/{split}/masks"
    )
    print(f"\n{split.upper()}")
    for k, v in result.items():
        print(f"{k}: {v}")

print("\n===== 3D DATASET COUNTS =====")
h5_dir = ROOT / "archive/BraTS2020_training_data/content/data"
if not h5_dir.exists():
    print("3D H5 folder not found:", h5_dir)
else:
    h5_files = sorted(h5_dir.glob("*.h5"))
    volumes = defaultdict(list)

    pattern = re.compile(r"volume_(\d+)_slice_(\d+)\.h5$")
    for file in h5_files:
        match = pattern.match(file.name)
        if match:
            volume_id = int(match.group(1))
            slice_id = int(match.group(2))
            volumes[volume_id].append(slice_id)

    print("Raw .h5 slice files:", len(h5_files))
    print("Unique 3D volumes used by code:", len(volumes))

    if volumes:
        slice_counts = [len(v) for v in volumes.values()]
        print("Min slices per volume:", min(slice_counts))
        print("Max slices per volume:", max(slice_counts))
        print("First 10 volume IDs:", sorted(volumes.keys())[:10])

print("\n===== MODEL / RUN FILES =====")
for folder in ["best_model", "best_model_3d", "runs", "runs_3d"]:
    p = ROOT / folder
    if p.exists():
        print(folder, "exists with", len(list(p.rglob("*"))), "items")
    else:
        print(folder, "NOT FOUND")
