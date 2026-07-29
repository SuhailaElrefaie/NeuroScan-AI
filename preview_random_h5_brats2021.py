from pathlib import Path
import re
import h5py
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/Users/suhailaelrefaei/Desktop/ISP/FirstDraft/archive/BraTS2021_h5")

# Auto-detect the actual H5 data folder
if (ROOT / "content" / "data").exists():
    DATA_DIR = ROOT / "content" / "data"
elif (ROOT / "data").exists():
    DATA_DIR = ROOT / "data"
else:
    DATA_DIR = ROOT

OUT = Path("sample_data/h5_brats2021_best_slice_preview.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

pattern = re.compile(r"volume_(\d+)_slice_(\d+)\.h5$")
groups = {}

for p in DATA_DIR.glob("volume_*_slice_*.h5"):
    m = pattern.match(p.name)
    if m:
        vol = int(m.group(1))
        sl = int(m.group(2))
        groups.setdefault(vol, []).append((sl, p))

print("Data folder:", DATA_DIR)
print("Volumes found:", len(groups))

if not groups:
    raise RuntimeError("No H5 volume slice files found.")

# Pick a good medium tumor case, not tiny and not insane huge
candidates = []

for vol, files in groups.items():
    total_area = 0
    max_slice_area = 0
    best_slice = None
    best_path = None

    for sl, p in files:
        with h5py.File(p, "r") as f:
            mask = f["mask"][()]
        area = int((mask > 0).sum())
        total_area += area

        if area > max_slice_area:
            max_slice_area = area
            best_slice = sl
            best_path = p

    # Medium visible tumor, avoids awful giant cases
    if 1000 <= max_slice_area <= 9000 and 5000 <= total_area <= 120000:
        candidates.append((max_slice_area, total_area, vol, best_slice, best_path))

if not candidates:
    # fallback: just pick strongest tumor volume
    print("No medium candidates found, using strongest available tumor case.")
    for vol, files in groups.items():
        max_slice_area = 0
        best_slice = None
        best_path = None

        for sl, p in files:
            with h5py.File(p, "r") as f:
                mask = f["mask"][()]
            area = int((mask > 0).sum())

            if area > max_slice_area:
                max_slice_area = area
                best_slice = sl
                best_path = p

        candidates.append((max_slice_area, max_slice_area, vol, best_slice, best_path))

# Pick a decent one from the middle of the candidate list
candidates = sorted(candidates, key=lambda x: x[0])
chosen = candidates[len(candidates) // 2]

max_area, total_area, vol, slice_idx, h5_path = chosen

print("Chosen volume:", vol)
print("Chosen slice:", slice_idx)
print("Slice tumor area:", max_area)
print("Total volume tumor area:", total_area)
print("File:", h5_path)

with h5py.File(h5_path, "r") as f:
    image = f["image"][()]  # H,W,4 = flair,t1,t1ce,t2
    mask = f["mask"][()]    # H,W

def normalize_mri(arr):
    arr = arr.astype(np.float32)
    nonzero = arr[arr > 0]

    if nonzero.size > 0:
        low, high = np.percentile(nonzero, [1, 99])
    else:
        low, high = np.percentile(arr, [1, 99])

    arr = np.clip(arr, low, high)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return (arr * 255).astype(np.uint8)

def crop_to_brain(img, mask=None):
    brain = img > 8
    ys, xs = np.where(brain)

    if len(xs) == 0 or len(ys) == 0:
        return img, mask

    pad = 12
    x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
    y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])

    img_crop = img[y1:y2, x1:x2]
    mask_crop = None if mask is None else mask[y1:y2, x1:x2]
    return img_crop, mask_crop

def make_panel(channel_index, title, overlay=False):
    img = normalize_mri(image[:, :, channel_index])
    m = mask > 0

    img, m = crop_to_brain(img, m)

    rgb = np.stack([img, img, img], axis=-1)

    if overlay:
        red = np.zeros_like(rgb)
        red[:, :, 0] = 255

        # transparent red fill
        rgb[m] = (0.72 * rgb[m] + 0.28 * red[m]).astype(np.uint8)

        # red outline
        edge = np.zeros_like(m, dtype=bool)
        edge[1:, :] |= m[1:, :] != m[:-1, :]
        edge[:-1, :] |= m[:-1, :] != m[1:, :]
        edge[:, 1:] |= m[:, 1:] != m[:, :-1]
        edge[:, :-1] |= m[:, :-1] != m[:, 1:]
        rgb[edge] = [255, 0, 0]

    pil = Image.fromarray(rgb).resize((420, 420), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (420, 465), (15, 15, 15))
    canvas.paste(pil, (0, 45))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 420, 45), fill=(0, 0, 0))
    draw.text((14, 14), title, fill=(255, 255, 255))

    return canvas

# Channel order from your converter: flair, t1, t1ce, t2
panels = [
    make_panel(0, "FLAIR MRI"),
    make_panel(1, "T1 MRI"),
    make_panel(2, "T1ce MRI"),
    make_panel(2, "T1ce + Ground Truth Tumor", overlay=True),
]

sheet = Image.new("RGB", (840, 930), (30, 30, 30))
sheet.paste(panels[0], (0, 0))
sheet.paste(panels[1], (420, 0))
sheet.paste(panels[2], (0, 465))
sheet.paste(panels[3], (420, 465))

sheet.save(OUT)

print("Saved preview:", OUT)
