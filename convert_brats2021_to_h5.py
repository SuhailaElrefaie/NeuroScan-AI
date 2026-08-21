from __future__ import annotations

from pathlib import Path

import h5py
import nibabel as nib
import numpy as np


INPUT_ROOT = Path("archive/BraTS2021_Training_Data")
OUTPUT_ROOT = Path("archive/BraTS2021_h5/content/data")


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Z-score non-zero brain voxels while keeping background exactly zero."""
    volume = np.nan_to_num(np.asarray(volume, dtype=np.float32))
    brain = volume != 0
    result = np.zeros_like(volume, dtype=np.float32)

    if not np.any(brain):
        return result

    values = volume[brain]
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-8:
        std = 1.0

    result[brain] = (values - mean) / std
    return result


def find_subject_dirs(root: Path) -> list[Path]:
    subject_dirs = []
    folders_to_check = [root, *(path for path in root.rglob("*") if path.is_dir())]

    for folder in folders_to_check:
        nii_files = [*folder.glob("*.nii"), *folder.glob("*.nii.gz")]
        names = [path.name.lower() for path in nii_files]

        has_flair = any("flair" in name for name in names)
        has_t1 = any("_t1." in name or "_t1.nii" in name for name in names)
        has_t1ce = any("t1ce" in name or "t1gd" in name for name in names)
        has_t2 = any("_t2." in name or "_t2.nii" in name for name in names)
        has_seg = any("seg" in name for name in names)

        if has_flair and has_t1 and has_t1ce and has_t2 and has_seg:
            subject_dirs.append(folder)

    return sorted(set(subject_dirs))


def pick_file(folder: Path, keyword: str) -> Path:
    files = [*folder.glob("*.nii"), *folder.glob("*.nii.gz")]

    for path in files:
        name = path.name.lower()
        if keyword == "t1" and ("_t1." in name or "_t1.nii" in name):
            return path
        if keyword == "t2" and ("_t2." in name or "_t2.nii" in name):
            return path
        if keyword not in {"t1", "t2"} and keyword in name:
            return path

    raise FileNotFoundError(f"Could not find {keyword} file in {folder}")


def main() -> None:
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"BraTS input folder not found: {INPUT_ROOT.resolve()}\n"
            "Update INPUT_ROOT near the top of this script."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    subjects = find_subject_dirs(INPUT_ROOT)
    print(f"Found subjects: {len(subjects)}")

    if not subjects:
        raise RuntimeError(
            "No valid BraTS 2021 subject folders found. Check INPUT_ROOT and "
            "the extracted folder structure."
        )

    for volume_id, subject_dir in enumerate(subjects):
        print(f"[{volume_id + 1}/{len(subjects)}] Converting {subject_dir.name}")

        flair = normalize_volume(nib.load(str(pick_file(subject_dir, "flair"))).get_fdata())
        t1 = normalize_volume(nib.load(str(pick_file(subject_dir, "t1"))).get_fdata())
        t1ce = normalize_volume(nib.load(str(pick_file(subject_dir, "t1ce"))).get_fdata())
        t2 = normalize_volume(nib.load(str(pick_file(subject_dir, "t2"))).get_fdata())
        segmentation = nib.load(str(pick_file(subject_dir, "seg"))).get_fdata()
        mask = (segmentation > 0).astype(np.uint8)

        # Channel order used throughout the project: FLAIR, T1, T1CE, T2.
        for slice_index in range(mask.shape[2]):
            image_slice = np.stack(
                [
                    flair[:, :, slice_index],
                    t1[:, :, slice_index],
                    t1ce[:, :, slice_index],
                    t2[:, :, slice_index],
                ],
                axis=-1,
            ).astype(np.float32)

            output_path = OUTPUT_ROOT / f"volume_{volume_id}_slice_{slice_index}.h5"
            with h5py.File(output_path, "w") as handle:
                handle.create_dataset("image", data=image_slice, compression="gzip")
                handle.create_dataset(
                    "mask",
                    data=mask[:, :, slice_index].astype(np.uint8),
                    compression="gzip",
                )

    print("Done.")
    print(f"Converted volumes: {len(subjects)}")
    print(f"Output folder: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
