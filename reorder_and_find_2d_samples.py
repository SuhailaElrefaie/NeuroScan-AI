from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import shutil

IMAGE_DIR = Path("Data/segmentation_task/train/images")
MASK_DIR = Path("Data/segmentation_task/train/masks")
SAMPLE_DIR = Path("sample_data/2d")
PREVIEW_OUT = Path("sample_data/final_2d_samples_preview.png")
BACKUP_DIR = Path("sample_data/2d_backup_before_final_mix")

exts = [".png", ".jpg", ".jpeg"]

# Backup current samples
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
for p in SAMPLE_DIR.glob("*"):
    if p.is_file():
        shutil.copy2(p, BACKUP_DIR / p.name)

# Save current wanted samples before overwriting
old_4 = Image.open(SAMPLE_DIR / "sample_4.png").convert("L")
old_3 = Image.open(SAMPLE_DIR / "sample_3.png").convert("L")
old_5 = Image.open(SAMPLE_DIR / "sample_5.png").convert("L")

# Used images should not be repeated as new picks
used_arrays = [
    np.asarray(old_4.resize((128, 128))),
    np.asarray(old_3.resize((128, 128))),
    np.asarray(old_5.resize((128, 128))),
]

def is_too_similar(img):
    arr = np.asarray(img.resize((128, 128)))
    for u in used_arrays:
        diff = np.mean(np.abs(arr.astype(float) - u.astype(float)))
        if diff < 8:
            return True
    return False

def find_mask_for_image(img_path):
    stem = img_path.stem
    for ext in exts:
        possible = MASK_DIR / f"{stem}{ext}"
        if possible.exists():
            return possible
    return None

# Find 2 new clean-looking candidates
images = []
for ext in exts:
    images.extend(IMAGE_DIR.glob(f"*{ext}"))

candidates = []

for img_path in sorted(images):
    mask_path = find_mask_for_image(img_path)
    if mask_path is None:
        continue

    try:
        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if is_too_similar(img):
            continue

        img_small = img.resize((256, 256))
        mask_small = mask.resize((256, 256), Image.Resampling.NEAREST)

        img_arr = np.asarray(img_small)
        mask_arr = np.asarray(mask_small)

        tumor_area = int((mask_arr > 20).sum())
        contrast = float(img_arr.std())
        brightness = float(img_arr.mean())

        # Keep visible but not insane-looking tumors
        if 700 <= tumor_area <= 9000 and contrast > 22 and 25 < brightness < 170:
            # Score: clear contrast + medium tumor size
            ideal_area = 3500
            area_score = -abs(tumor_area - ideal_area) / ideal_area
            score = contrast + (area_score * 10)
            candidates.append((score, contrast, tumor_area, img_path))

    except Exception:
        pass

candidates = sorted(candidates, reverse=True)

if len(candidates) < 2:
    raise RuntimeError("Could not find 2 new good samples. Try relaxing the filters.")

new_1 = Image.open(candidates[0][3]).convert("L")
new_2 = Image.open(candidates[1][3]).convert("L")

# Clear sample folder
for p in SAMPLE_DIR.glob("*"):
    if p.is_file():
        p.unlink()

# New order
final_samples = [
    ("sample_1.png", old_4, "old 4th"),
    ("sample_2.png", old_3, "old 3rd"),
    ("sample_3.png", old_5, "old 5th"),
    ("sample_4.png", new_1, f"new: {candidates[0][3].name}"),
    ("sample_5.png", new_2, f"new: {candidates[1][3].name}"),
]

preview_tiles = []

for filename, img, label in final_samples:
    img.save(SAMPLE_DIR / filename)

    tile = img.resize((240, 240)).convert("RGB")
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, 240, 36), fill=(0, 0, 0))
    draw.text((8, 8), filename, fill=(255, 255, 255))
    draw.text((8, 22), label[:28], fill=(255, 255, 255))
    preview_tiles.append(tile)

    print(f"Saved {filename} from {label}")

sheet = Image.new("RGB", (240 * 5, 240), "white")
for i, tile in enumerate(preview_tiles):
    sheet.paste(tile, (i * 240, 0))

sheet.save(PREVIEW_OUT)

print("Preview saved:", PREVIEW_OUT)
print("Backup saved in:", BACKUP_DIR)
print("\nNew auto-picked files:")
print("sample_4:", candidates[0][3].name, "| area:", candidates[0][2], "| contrast:", round(candidates[0][1], 1))
print("sample_5:", candidates[1][3].name, "| area:", candidates[1][2], "| contrast:", round(candidates[1][1], 1))
