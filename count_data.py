from pathlib import Path
import re
from collections import defaultdict

def count_2d(split):
    img_dir = Path(f"Data/segmentation_task/{split}/images")
    mask_dir = Path(f"Data/segmentation_task/{split}/masks")

    imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    masks = [p for p in mask_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]

    img_keys = {p.stem for p in imgs}
    mask_keys = {p.stem for p in masks}
    matched = img_keys & mask_keys

    print(f"\n{split.upper()} 2D")
    print("Raw images:", len(imgs))
    print("Raw masks:", len(masks))
    print("Matched image-mask pairs used by code:", len(matched))
    print("Images without mask:", len(img_keys - mask_keys))
    print("Masks without image:", len(mask_keys - img_keys))

count_2d("train")
count_2d("test")

print("\n3D H5")
h5_dir = Path("archive/BraTS2020_training_data/content/data")
pattern = re.compile(r"volume_(\d+)_slice_(\d+)\.h5$")
volumes = defaultdict(list)

for f in h5_dir.glob("*.h5"):
    m = pattern.match(f.name)
    if m:
        volumes[int(m.group(1))].append(int(m.group(2)))

print("Raw H5 slice files:", len(list(h5_dir.glob('*.h5'))))
print("Unique 3D volumes used by code:", len(volumes))

if volumes:
    counts = [len(v) for v in volumes.values()]
    print("Min slices per volume:", min(counts))
    print("Max slices per volume:", max(counts))
    print("First 10 volume IDs:", sorted(volumes.keys())[:10])
