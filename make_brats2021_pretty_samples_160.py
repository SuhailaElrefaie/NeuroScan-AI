from pathlib import Path
import re
import h5py
import numpy as np
from PIL import Image

DATA_DIR = Path("archive/BraTS2021_h5/content/data")
OUT_DIR = Path("sample_data/3d_brats2021_pretty_160")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEPTH = 32
SIZE = 160
pattern = re.compile(r"volume_(\d+)_slice_(\d+)\.h5$")

groups = {}
for p in DATA_DIR.glob("volume_*_slice_*.h5"):
    m = pattern.match(p.name)
    if m:
        vol = int(m.group(1))
        sl = int(m.group(2))
        groups.setdefault(vol, []).append((sl, p))

def resize_float(arr):
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    img = img.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    return np.asarray(img).astype(np.float32)

def resize_mask(arr):
    img = Image.fromarray((arr > 0).astype(np.uint8) * 255)
    img = img.resize((SIZE, SIZE), Image.Resampling.NEAREST)
    return (np.asarray(img) > 0).astype(np.uint8)

candidates = []

for vol, files in groups.items():
    files = sorted(files)
    total = 0
    max_slice_area = 0
    tumor_slices = 0

    for _, p in files:
        with h5py.File(p, "r") as f:
            mask = f["mask"][()]
        area = int((mask > 0).sum())
        total += area
        max_slice_area = max(max_slice_area, area)
        if area > 100:
            tumor_slices += 1

    # Avoid tiny tumors and avoid giant ugly tumors
    if 3000 <= total <= 90000 and 5 <= tumor_slices <= 60 and 300 <= max_slice_area <= 7000:
        candidates.append((total, max_slice_area, tumor_slices, vol))

print("Candidate volumes:", len(candidates))

# Pick spread-out medium cases
candidates = sorted(candidates, key=lambda x: x[0])
if len(candidates) < 5:
    selected = candidates[:5]
else:
    idxs = np.linspace(0, len(candidates) - 1, 5).astype(int)
    selected = [candidates[i] for i in idxs]

print("Selected:")
for item in selected:
    print(item)

def choose_window(files):
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

for i, (_, _, _, vol) in enumerate(selected, start=1):
    chosen = choose_window(groups[vol])
    image_slices = []
    mask_slices = []

    for p in chosen:
        with h5py.File(p, "r") as f:
            image = f["image"][()]
            mask = f["mask"][()]

        chans = [resize_float(image[:, :, c]) for c in range(4)]
        image_slices.append(np.stack(chans, axis=0))
        mask_slices.append(resize_mask(mask))

    image_volume = np.stack(image_slices, axis=1).astype(np.float32)
    mask_volume = np.stack(mask_slices, axis=0).astype(np.uint8)

    out = OUT_DIR / f"brats2021_pretty_sample_{i}_volume_{vol}.npz"
    np.savez_compressed(out, image=image_volume, mask=mask_volume, volume_id=vol, source="BraTS2021")
    print("Saved:", out, image_volume.shape, "tumor voxels:", int(mask_volume.sum()))
