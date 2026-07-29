from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import shutil

IMAGE_DIR = Path("Data/segmentation_task/train/images")
MASK_DIR = Path("Data/segmentation_task/train/masks")
OUT_DIR = Path("sample_data/2d")
PREVIEW_PATH = Path("sample_data/2d_preview_contact_sheet.png")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Clear old 2D public samples
for p in OUT_DIR.glob("*"):
    if p.is_file():
        p.unlink()

image_exts = [".png", ".jpg", ".jpeg"]

images = []
for ext in image_exts:
    images.extend(IMAGE_DIR.glob(f"*{ext}"))

candidates = []

for img_path in sorted(images):
    stem = img_path.stem

    mask_path = None
    for ext in image_exts:
        possible = MASK_DIR / f"{stem}{ext}"
        if possible.exists():
            mask_path = possible
            break

    if mask_path is None:
        continue

    try:
        img = Image.open(img_path).convert("L").resize((256, 256))
        mask = Image.open(mask_path).convert("L").resize((256, 256), Image.Resampling.NEAREST)

        img_arr = np.asarray(img)
        mask_arr = np.asarray(mask)

        tumor_area = int((mask_arr > 20).sum())

        # Avoid tiny invisible tumors and huge messy masks
        if 700 <= tumor_area <= 9000:
            contrast = float(img_arr.std())
            brightness = float(img_arr.mean())

            # Prefer clear-looking scans
            if contrast > 20 and 20 < brightness < 180:
                candidates.append((tumor_area, contrast, img_path, mask_path))

    except Exception:
        pass

print(f"Candidates found: {len(candidates)}")

if len(candidates) < 5:
    raise RuntimeError("Not enough good candidates found. Try relaxing the area limits.")

# Sort by tumor size and pick varied examples, not all similar
candidates = sorted(candidates, key=lambda x: x[0])
indices = np.linspace(0, len(candidates) - 1, 5).astype(int)
selected = [candidates[i] for i in indices]

preview_tiles = []

for i, (area, contrast, img_path, mask_path) in enumerate(selected, start=1):
    img = Image.open(img_path).convert("L").resize((256, 256))
    mask = Image.open(mask_path).convert("L").resize((256, 256), Image.Resampling.NEAREST)

    out_path = OUT_DIR / f"sample_{i}.png"
    img.save(out_path)

    # Make preview overlay just for you
    img_rgb = ImageOps.colorize(img, black="black", white="white").convert("RGB")
    mask_arr = np.asarray(mask)
    overlay = np.asarray(img_rgb).copy()

    red = np.zeros_like(overlay)
    red[:, :, 0] = 255

    mask_bool = mask_arr > 20
    overlay[mask_bool] = (0.65 * overlay[mask_bool] + 0.35 * red[mask_bool]).astype(np.uint8)

    tile = Image.fromarray(overlay)
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, 256, 32), fill=(0, 0, 0))
    draw.text((8, 8), f"Sample {i} | area {area}", fill=(255, 255, 255))
    preview_tiles.append(tile)

    print(f"Saved {out_path} from {img_path.name} | tumor area={area} | contrast={contrast:.1f}")

sheet = Image.new("RGB", (256 * 5, 256), "white")
for i, tile in enumerate(preview_tiles):
    sheet.paste(tile, (i * 256, 0))

sheet.save(PREVIEW_PATH)
print(f"Preview saved to: {PREVIEW_PATH}")
