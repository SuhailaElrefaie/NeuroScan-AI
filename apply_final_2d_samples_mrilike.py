from pathlib import Path
from PIL import Image, ImageDraw
import shutil

IMAGE_DIR = Path("Data/segmentation_task/train/images")
SAMPLE_DIR = Path("sample_data/2d")
NAMES_FILE = Path("sample_data/mri_like_candidates/mri_like_names.txt")
PREVIEW_OUT = Path("sample_data/final_2d_samples_preview.png")
BACKUP_DIR = Path("sample_data/2d_backup_before_mrilike_final")

# chosen numbers from MRI-like sheet
CHOSEN = [26, 19]

# back up current sample folder
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
for p in SAMPLE_DIR.glob("*"):
    if p.is_file():
        shutil.copy2(p, BACKUP_DIR / p.name)

# keep current 4th, 3rd, 5th
img1 = Image.open(SAMPLE_DIR / "sample_4.png").convert("L")
img2 = Image.open(SAMPLE_DIR / "sample_3.png").convert("L")
img3 = Image.open(SAMPLE_DIR / "sample_5.png").convert("L")

# load MRI-like candidate mapping
mapping = {}
with open(NAMES_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        idx_part, name = line.split(":", 1)
        mapping[int(idx_part.strip())] = name.strip()

img4_name = mapping[CHOSEN[0]]
img5_name = mapping[CHOSEN[1]]

img4 = Image.open(IMAGE_DIR / img4_name).convert("L")
img5 = Image.open(IMAGE_DIR / img5_name).convert("L")

# clear current sample folder
for p in SAMPLE_DIR.glob("*"):
    if p.is_file():
        p.unlink()

final_samples = [
    ("sample_1.png", img1, "old 4th"),
    ("sample_2.png", img2, "old 3rd"),
    ("sample_3.png", img3, "old 5th"),
    ("sample_4.png", img4, f"candidate {CHOSEN[0]}"),
    ("sample_5.png", img5, f"candidate {CHOSEN[1]}"),
]

tiles = []

for filename, img, label in final_samples:
    img.save(SAMPLE_DIR / filename)

    tile = img.resize((240, 240)).convert("RGB")
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, 240, 36), fill=(0, 0, 0))
    draw.text((8, 8), filename, fill=(255, 255, 255))
    draw.text((8, 22), label, fill=(255, 255, 255))
    tiles.append(tile)

    print(f"Saved {filename} from {label}")

sheet = Image.new("RGB", (240 * 5, 240), "white")
for i, tile in enumerate(tiles):
    sheet.paste(tile, (i * 240, 0))

sheet.save(PREVIEW_OUT)
print("Preview saved:", PREVIEW_OUT)
print("Backup saved:", BACKUP_DIR)
print("sample_4 source:", img4_name)
print("sample_5 source:", img5_name)
