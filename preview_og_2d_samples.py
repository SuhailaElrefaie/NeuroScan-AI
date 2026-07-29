from pathlib import Path
from PIL import Image, ImageDraw
import subprocess
import math

OUT_DIR = Path("sample_data/og_2d_samples_preview")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Clear old preview only
for p in OUT_DIR.glob("*"):
    if p.is_file():
        p.unlink()

# First commit that touched sample_data/2d
commits = subprocess.check_output(
    ["git", "log", "--reverse", "--format=%H", "--", "sample_data/2d"],
    text=True
).strip().splitlines()

if not commits:
    raise RuntimeError("No Git history found for sample_data/2d")

first_commit = commits[0]
print("Using original 2D sample commit:", first_commit)

files = subprocess.check_output(
    ["git", "ls-tree", "-r", "--name-only", first_commit, "sample_data/2d"],
    text=True
).strip().splitlines()

image_files = [
    f for f in files
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

if not image_files:
    raise RuntimeError("No original images found in sample_data/2d")

tiles = []

for i, file_path in enumerate(image_files, start=1):
    data = subprocess.check_output(["git", "show", f"{first_commit}:{file_path}"])

    out_file = OUT_DIR / Path(file_path).name
    out_file.write_bytes(data)

    img = Image.open(out_file).convert("L")
    preview = img.resize((240, 240))

    tile = preview.convert("RGB")
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, 240, 34), fill=(0, 0, 0))
    draw.text((8, 8), f"{i}: {out_file.name}", fill=(255, 255, 255))
    tiles.append(tile)

    print(f"{i}: {out_file.name}")

cols = 5
rows = math.ceil(len(tiles) / cols)

sheet = Image.new("RGB", (cols * 240, rows * 240), "white")

for i, tile in enumerate(tiles):
    x = (i % cols) * 240
    y = (i // cols) * 240
    sheet.paste(tile, (x, y))

sheet_path = OUT_DIR / "og_2d_samples_sheet.png"
sheet.save(sheet_path)

print("Saved preview sheet:", sheet_path)
