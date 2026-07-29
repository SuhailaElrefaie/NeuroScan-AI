from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import numpy as np

IMAGE_DIR = Path("Data/segmentation_task/train/images")
MASK_DIR = Path("Data/segmentation_task/train/masks")
OUT_DIR = Path("sample_data/2d_candidate_previews")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for p in OUT_DIR.glob("*"):
    if p.is_file():
        p.unlink()

exts = [".png", ".jpg", ".jpeg"]
images = []
for ext in exts:
    images.extend(IMAGE_DIR.glob(f"*{ext}"))

candidates = []

for img_path in sorted(images):
    stem = img_path.stem

    mask_path = None
    for ext in exts:
        possible = MASK_DIR / f"{stem}{ext}"
        if possible.exists():
            mask_path = possible
            break

    if mask_path is None:
        continue

    try:
        original = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Resize only for scoring, not for final saving
        img_small = original.resize((256, 256))
        mask_small = mask.resize((256, 256), Image.Resampling.NEAREST)

        img_arr = np.asarray(img_small)
        mask_arr = np.asarray(mask_small)

        tumor_area = int((mask_arr > 20).sum())
        contrast = float(img_arr.std())
        brightness = float(img_arr.mean())

        w, h = original.size

        # Avoid bad cases
        if 500 <= tumor_area <= 12000 and contrast > 18 and 20 < brightness < 190:
            candidates.append((contrast, tumor_area, w * h, img_path, mask_path))

    except Exception:
        pass

# Prefer higher contrast and bigger original resolution
candidates = sorted(candidates, key=lambda x: (x[0], x[2]), reverse=True)

selected = candidates[:30]

tiles = []
names = []

for idx, (contrast, area, pixels, img_path, mask_path) in enumerate(selected, start=1):
    img = Image.open(img_path).convert("L")
    mask = Image.open(mask_path).convert("L")

    img_preview = img.resize((220, 220))
    mask_preview = mask.resize((220, 220), Image.Resampling.NEAREST)

    rgb = ImageOps.colorize(img_preview, black="black", white="white").convert("RGB")
    arr = np.asarray(rgb).copy()
    m = np.asarray(mask_preview) > 20

    red = np.zeros_like(arr)
    red[:, :, 0] = 255
    arr[m] = (0.65 * arr[m] + 0.35 * red[m]).astype(np.uint8)

    tile = Image.fromarray(arr)
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, 220, 36), fill=(0, 0, 0))
    draw.text((6, 6), f"{idx}: {img_path.name[:18]}", fill=(255, 255, 255))
    draw.text((6, 20), f"area {area} | contrast {contrast:.1f}", fill=(255, 255, 255))

    tiles.append(tile)
    names.append((idx, img_path.name, mask_path.name))

cols = 5
rows = int(np.ceil(len(tiles) / cols))
sheet = Image.new("RGB", (cols * 220, rows * 220), "white")

for i, tile in enumerate(tiles):
    x = (i % cols) * 220
    y = (i // cols) * 220
    sheet.paste(tile, (x, y))

sheet_path = OUT_DIR / "candidate_sheet.png"
sheet.save(sheet_path)

with open(OUT_DIR / "candidate_names.txt", "w") as f:
    for idx, img_name, mask_name in names:
        f.write(f"{idx}: {img_name} | mask: {mask_name}\n")

print("Saved preview sheet:", sheet_path)
print("Saved names:", OUT_DIR / "candidate_names.txt")
