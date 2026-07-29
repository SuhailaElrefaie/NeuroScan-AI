from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import subprocess
import math

SELECTED_CANDIDATES = [7, 15, 24, 27]

IMAGE_DIR = Path("Data/segmentation_task/train/images")
MASK_DIR = Path("Data/segmentation_task/train/masks")
NAMES_FILE = Path("sample_data/2d_candidate_previews/candidate_names.txt")
OUT_DIR = Path("sample_data/2d")
PREVIEW_OUT = Path("sample_data/mixed_2d_samples_preview.png")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- load candidate mapping ----------
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

# ---------- get first OG sample from git history ----------
commits = subprocess.check_output(
    ["git", "log", "--reverse", "--format=%H", "--", "sample_data/2d"],
    text=True
).strip().splitlines()

if not commits:
    raise RuntimeError("No git history found for sample_data/2d")

first_commit = commits[0]

files = subprocess.check_output(
    ["git", "ls-tree", "-r", "--name-only", first_commit, "sample_data/2d"],
    text=True
).strip().splitlines()

image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
if not image_files:
    raise RuntimeError("No original sample images found in first commit")

og_first_file = sorted(image_files)[0]
og_bytes = subprocess.check_output(["git", "show", f"{first_commit}:{og_first_file}"])

# ---------- clear old public samples ----------
for p in OUT_DIR.glob("*"):
    if p.is_file():
        p.unlink()

preview_tiles = []

def make_preview_tile(img, title):
    preview_img = img.resize((240, 240))
    tile = preview_img.convert("RGB")
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, 240, 34), fill=(0, 0, 0))
    draw.text((8, 8), title, fill=(255, 255, 255))
    return tile

# ---------- sample 1-3 from chosen candidates ----------
for out_i, selected_idx in enumerate(SELECTED_CANDIDATES[:3], start=1):
    img_name, mask_name = mapping[selected_idx]
    img_path = IMAGE_DIR / img_name
    mask_path = MASK_DIR / mask_name

    img = Image.open(img_path).convert("L")
    img.save(OUT_DIR / f"sample_{out_i}.png")

    preview_tiles.append(make_preview_tile(img, f"Sample {out_i} | candidate {selected_idx}"))
    print(f"Saved sample_{out_i}.png from candidate {selected_idx}: {img_name}")

# ---------- sample 4 from first OG sample ----------
og_temp = OUT_DIR / "_og_temp_image"
og_temp.write_bytes(og_bytes)
og_img = Image.open(og_temp).convert("L")
og_img.save(OUT_DIR / "sample_4.png")
og_temp.unlink(missing_ok=True)

preview_tiles.append(make_preview_tile(og_img, "Sample 4 | first OG sample"))
print(f"Saved sample_4.png from first OG sample: {og_first_file}")

# ---------- sample 5 from last chosen candidate ----------
last_idx = SELECTED_CANDIDATES[3]
img_name, mask_name = mapping[last_idx]
img_path = IMAGE_DIR / img_name
img = Image.open(img_path).convert("L")
img.save(OUT_DIR / "sample_5.png")

preview_tiles.append(make_preview_tile(img, f"Sample 5 | candidate {last_idx}"))
print(f"Saved sample_5.png from candidate {last_idx}: {img_name}")

# ---------- save preview sheet ----------
sheet = Image.new("RGB", (240 * 5, 240), "white")
for i, tile in enumerate(preview_tiles):
    sheet.paste(tile, (i * 240, 0))

sheet.save(PREVIEW_OUT)
print("Preview saved:", PREVIEW_OUT)
