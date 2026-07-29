from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw

SUBJECT_DIR = Path("PASTE_FOLDER_PATH_HERE")

print("Using subject:", SUBJECT_DIR)

def find_file(keyword):
    files = list(SUBJECT_DIR.glob(f"*{keyword}*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"Missing {keyword} file in {SUBJECT_DIR}")
    return files[0]

flair_path = find_file("flair")
t1ce_path = find_file("t1ce")
t2_path = find_file("t2")
seg_path = find_file("seg")

flair = nib.load(str(flair_path)).get_fdata()
t1ce = nib.load(str(t1ce_path)).get_fdata()
t2 = nib.load(str(t2_path)).get_fdata()
seg = nib.load(str(seg_path)).get_fdata()

areas = (seg > 0).sum(axis=(0, 1))
slice_idx = int(np.argmax(areas))

print("Best tumor slice:", slice_idx)
print("Tumor pixels on slice:", int(areas[slice_idx]))

def normalize_mri(slice_2d):
    arr = slice_2d.astype(np.float32)
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

def make_panel(volume, title, mask=None, overlay=False):
    img = normalize_mri(volume[:, :, slice_idx])
    mask_slice = None if mask is None else (mask[:, :, slice_idx] > 0)

    img, mask_slice = crop_to_brain(img, mask_slice)
    rgb = np.stack([img, img, img], axis=-1)

    if overlay and mask_slice is not None:
        red = np.zeros_like(rgb)
        red[:, :, 0] = 255

        rgb[mask_slice] = (0.72 * rgb[mask_slice] + 0.28 * red[mask_slice]).astype(np.uint8)

        m = mask_slice.astype(np.uint8)
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

panels = [
    make_panel(t1ce, "T1ce MRI"),
    make_panel(flair, "FLAIR MRI"),
    make_panel(t2, "T2 MRI"),
    make_panel(t1ce, "T1ce + Tumor Overlay", mask=seg, overlay=True),
]

sheet = Image.new("RGB", (840, 930), (30, 30, 30))
sheet.paste(panels[0], (0, 0))
sheet.paste(panels[1], (420, 0))
sheet.paste(panels[2], (0, 465))
sheet.paste(panels[3], (420, 465))

out = Path("sample_data/brats2021_best_slice_preview.png")
out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(out)

print("Saved preview:", out)
