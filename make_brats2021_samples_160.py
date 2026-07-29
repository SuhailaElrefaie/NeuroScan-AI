from pathlib import Path
import re
import h5py
import numpy as np
from PIL import Image

DATA_DIR = Path("archive/BraTS2021_h5/content/data")
OUT_DIR = Path("sample_data/3d_brats2021_160")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEPTH = 32
SIZE = 160

pattern = re.compile(r"volume_(\d+)_slice_(\d+)\.h5$")
groups = {}

for path in DATA_DIR.glob("volume_*_slice_*.h5"):
    m = pattern.match(path.name)
    if m:
        vol = int(m.group(1))
        sl = int(m.group(2))
        groups.setdefault(vol, []).append((sl, path))

print("Volumes found:", len(groups))

scores = []
for vol, files in groups.items():
    tumor_pixels = 0
    for _, p in files:
        with h5py.File(p, "r") as f:
            mask = f["mask"][()]
            tumor_pixels += int((mask > 0).sum())
    scores.append((tumor_pixels, vol))

scores = sorted(scores, reverse=True)
selected = scores[:5]

def resize_float(arr, size):
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img).astype(np.float32)

def resize_mask(arr, size):
    img = Image.fromarray((arr > 0).astype(np.uint8) * 255)
    img = img.resize((size, size), Image.Resampling.NEAREST)
    return (np.asarray(img) > 0).astype(np.uint8)

def choose_centered(files):
    files = sorted(files, key=lambda x: x[0])
    areas = []
    for sl, p in files:
        with h5py.File(p, "r") as f:
            mask = f["mask"][()]
        areas.append((int((mask > 0).sum()), sl, p))

    best_i = max(range(len(areas)), key=lambda i: areas[i][0])
    start = max(0, best_i - DEPTH // 2)
    start = min(start, max(0, len(areas) - DEPTH))
    chosen = areas[start:start + DEPTH]

    while len(chosen) < DEPTH:
        chosen.append(chosen[-1])

    return [p for _, _, p in chosen]

for i, (score, vol) in enumerate(selected, start=1):
    chosen = choose_centered(groups[vol])

    image_slices = []
    mask_slices = []

    for p in chosen:
        with h5py.File(p, "r") as f:
            image = f["image"][()]  # H,W,4
            mask = f["mask"][()]    # H,W

        channels = []
        for c in range(4):
            channels.append(resize_float(image[:, :, c], SIZE))

        image_slices.append(np.stack(channels, axis=0))  # 4,H,W
        mask_slices.append(resize_mask(mask, SIZE))

    image_volume = np.stack(image_slices, axis=1).astype(np.float32) # 4,D,H,W
    mask_volume = np.stack(mask_slices, axis=0).astype(np.uint8)     # D,H,W

    out = OUT_DIR / f"brats2021_sample_{i}_volume_{vol}.npz"
    np.savez_compressed(out, image=image_volume, mask=mask_volume, volume_id=vol, source="BraTS2021")

    print("Saved:", out, "image", image_volume.shape, "mask", mask_volume.shape, "tumor voxels", int(mask_volume.sum()))
