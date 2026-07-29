from pathlib import Path
import h5py
import nibabel as nib
import numpy as np


INPUT_ROOT = Path("/Users/suhailaelrefaei/Desktop/ISP/Datasets/BraTS2021/full_extracted")
OUTPUT_ROOT = Path("archive/BraTS2021_h5/content/data")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def normalize_volume(vol):
    vol = vol.astype(np.float32)
    nonzero = vol[vol > 0]

    if nonzero.size == 0:
        return np.zeros_like(vol, dtype=np.float32)

    mean = nonzero.mean()
    std = nonzero.std()

    if std < 1e-8:
        return np.zeros_like(vol, dtype=np.float32)

    return (vol - mean) / std


def find_subject_dirs(root):
    subject_dirs = []

    folders_to_check = [root] + [p for p in root.rglob("*") if p.is_dir()]

    for folder in folders_to_check:
        if not folder.is_dir():
            continue

        nii_files = list(folder.glob("*.nii")) + list(folder.glob("*.nii.gz"))

        names = [p.name.lower() for p in nii_files]

        has_flair = any("flair" in n for n in names)
        has_t1 = any("_t1." in n or "_t1.nii" in n for n in names)
        has_t1ce = any("t1ce" in n or "t1gd" in n for n in names)
        has_t2 = any("_t2." in n or "_t2.nii" in n for n in names)
        has_seg = any("seg" in n for n in names)

        if has_flair and has_t1 and has_t1ce and has_t2 and has_seg:
            subject_dirs.append(folder)

    return sorted(subject_dirs)


def pick_file(folder, keyword):
    files = list(folder.glob("*.nii")) + list(folder.glob("*.nii.gz"))

    for f in files:
        name = f.name.lower()
        if keyword == "t1":
            if "_t1." in name or "_t1.nii" in name:
                return f
        elif keyword == "t2":
            if "_t2." in name or "_t2.nii" in name:
                return f
        elif keyword in name:
            return f

    raise FileNotFoundError(f"Could not find {keyword} file in {folder}")


subjects = find_subject_dirs(INPUT_ROOT)

print(f"Found subjects: {len(subjects)}")

if len(subjects) == 0:
    raise RuntimeError("No valid BraTS2021 subject folders found. Check extracted folder structure.")

volume_counter = 0

for subject_dir in subjects:
    print(f"Converting: {subject_dir.name}")

    flair_path = pick_file(subject_dir, "flair")
    t1_path = pick_file(subject_dir, "t1")
    t1ce_path = pick_file(subject_dir, "t1ce")
    t2_path = pick_file(subject_dir, "t2")
    seg_path = pick_file(subject_dir, "seg")

    flair = normalize_volume(nib.load(str(flair_path)).get_fdata())
    t1 = normalize_volume(nib.load(str(t1_path)).get_fdata())
    t1ce = normalize_volume(nib.load(str(t1ce_path)).get_fdata())
    t2 = normalize_volume(nib.load(str(t2_path)).get_fdata())

    seg = nib.load(str(seg_path)).get_fdata()
    mask = (seg > 0).astype(np.uint8)

    # BraTS NIfTI is usually [H, W, D].
    # Current project H5 slice format uses:
    # image: [H, W, 4]
    # mask: [H, W]
    depth = mask.shape[2]

    for slice_idx in range(depth):
        image_slice = np.stack(
            [
                flair[:, :, slice_idx],
                t1[:, :, slice_idx],
                t1ce[:, :, slice_idx],
                t2[:, :, slice_idx],
            ],
            axis=-1
        ).astype(np.float32)

        mask_slice = mask[:, :, slice_idx].astype(np.uint8)

        out_path = OUTPUT_ROOT / f"volume_{volume_counter}_slice_{slice_idx}.h5"

        with h5py.File(out_path, "w") as f:
            f.create_dataset("image", data=image_slice, compression="gzip")
            f.create_dataset("mask", data=mask_slice, compression="gzip")

    volume_counter += 1

print("Done.")
print(f"Converted volumes: {volume_counter}")
print(f"Output folder: {OUTPUT_ROOT}")
