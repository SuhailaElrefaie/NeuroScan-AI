from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import math

IMAGE_DIR = Path("Data/segmentation_task/train/images")
MASK_DIR = Path("Data/segmentation_task/train/masks")
OUT_DIR = Path("sample_data/mri_like_candidates")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for p in OUT_DIR.glob("*"):
    if p.is_file():
        p.unlink()

exts = [".png", ".jpg", ".jpeg"]

reference = Image.open("sample_data/2d/sample_1.png").convert("L").resize((128, 128))
ref_arr = np.asarray(reference).astype(float)

images = []
for ext in exts:
    images.extend(IMAGE_DIR.glob(f"*{ext}"))

def find_mask(img_path):
    stem = img_path.stem
    for ext in exts:
        p = MASK_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None

candidates = []

for img_path in sorted(images):
    mask_path = find_mask(img_path)
    if mask_path is None:
        continue

    try:
        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        small = img.resize((128, 128))
        arr = np.asarray(small).astype(float)

        score_img = img.resize((256, 256))
        score_mask = mask.resize((256, 256), Image.Resampling.NEAREST)

        img_arr = np.asarray(score_img)
        mask_arr = np.asarray(score_mask)

        tumor_area = int((mask_arr > 20).sum())
        contrast = float(img_arr.std())
        brightness = float(img_arr.mean())

        # CT-looking images usually have very bright skull/bone ring.
        very_bright_ratio = float((img_arr > 230).mean())

        # Similarity to sample_1
        similarity_error = float(np.mean(np.abs(arr - ref_arr)))

        # Keep visible tumor, but reject CT-looking bright-bone images
        if 500 <= tumor_area <= 12000 and contrast > 15 and 15 < brightness < 180:
            if very_bright_ratio < 0.035:  # rejects bright skull CT-like scans
                candidates.append((similarity_error, contrast, tumor_area, brightness, very_bright_ratio, img_path))

    except Exception:
        pass

candidates = sorted(candidates, key=lambda x: x[0])
selected = candidates[:30]

tiles = []
names = []

for i, (sim, contrast, area, bright, bright_ratio, img_path) in enumerate(selected, start=1):
    img = Image.open(img_path).convert("L")

    tile = img.resize((240, 240)).convert("RGB")
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, 240, 48), fill=(0, 0, 0))
    draw.text((8, 6), f"{i}: {img_path.name[:20]}", fill=(255, 255, 255))
    draw.text((8, 22), f"area {area} | sim {sim:.1f}", fill=(255, 255, 255))
    draw.text((8, 36), f"bright {bright_ratio:.3f}", fill=(255, 255, 255))

    tiles.append(tile)
    names.append((i, img_path.name))

cols = 5
rows = math.ceil(len(tiles) / cols)
sheet = Image.new("RGB", (cols * 240, rows * 240), "white")

for i, tile in enumerate(tiles):
    sheet.paste(tile, ((i % cols) * 240, (i // cols) * 240))

sheet_path = OUT_DIR / "mri_like_sheet.png"
sheet.save(sheet_path)

with open(OUT_DIR / "mri_like_names.txt", "w") as f:
    for i, name in names:
        f.write(f"{i}: {name}\n")

print("Saved:", sheet_path)
print("Saved:", OUT_DIR / "mri_like_names.txt")
