from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import numpy as np

SELECTED = [7, 15, 24, 17, 27]

IMAGE_DIR = Path("Data/segmentation_task/train/images")
MASK_DIR = Path("Data/segmentation_task/train/masks")
NAMES_FILE = Path("sample_data/2d_candidate_previews/candidate_names.txt")
OUT_DIR = Path("sample_data/2d")
PREVIEW_OUT = Path("sample_data/selected_2d_samples_preview.png")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Read candidate list
mapping = {}

with open(NAMES_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        idx_part, rest = line.split(":", 1)
        idx = int(idx_part.strip())

        img_name = rest.split("|")[0].strip()
        mask_name = rest.split("mask:")[1].strip()

        mapping[idx] = (img_name, mask_name)

# Clear old public 2D samples only
for p in OUT_DIR.glob("*"):
    if p.is_file():
        p.unlink()

preview_tiles = []

for out_i, selected_idx in enumerate(SELECTED, start=1):
    img_name, mask_name = mapping[selected_idx]

    img_path = IMAGE_DIR / img_name
    mask_path = MASK_DIR / mask_name

    if not img_path.exists():
        raise FileNotFoundError(f"Missing image: {img_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask: {mask_path}")

    # Save image as PNG without resizing, so quality/dimensions stay as original as possible
    img = Image.open(img_path).convert("L")
    out_path = OUT_DIR / f"sample_{out_i}.png"
    img.save(out_path)

    # Preview overlay only for checking
    mask = Image.open(mask_path).convert("L").resize(img.size, Image.Resampling.NEAREST)

    preview_img = img.resize((240, 240))
    preview_mask = mask.resize((240, 240), Image.Resampling.NEAREST)

    rgb = ImageOps.colorize(preview_img, black="black", white="white").convert("RGB")
    arr = np.asarray(rgb).copy()
    m = np.asarray(preview_mask) > 20

    red = np.zeros_like(arr)
    red[:, :, 0] = 255
    arr[m] = (0.65 * arr[m] + 0.35 * red[m]).astype(np.uint8)

    tile = Image.fromarray(arr)
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, 240, 34), fill=(0, 0, 0))
    draw.text((8, 8), f"Sample {out_i} from candidate {selected_idx}", fill=(255, 255, 255))
    preview_tiles.append(tile)

    print(f"Saved sample_{out_i}.png from candidate {selected_idx}: {img_name}")

sheet = Image.new("RGB", (240 * 5, 240), "white")
for i, tile in enumerate(preview_tiles):
    sheet.paste(tile, (i * 240, 0))

sheet.save(PREVIEW_OUT)
print("Preview saved:", PREVIEW_OUT)
