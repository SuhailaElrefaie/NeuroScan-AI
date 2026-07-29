"""Create complete-patient NPZ samples without touching the 2D sample folder."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

H5_DIR = Path("archive/BraTS2021_h5/content/data")
OUTPUT_DIR = Path("sample_data/3d")
NUMBER_OF_SAMPLES = 5

PATTERN = re.compile(r"volume_(\d+)_slice_(\d+)\.h5$")


def collect_subjects() -> dict[int, list[tuple[int, Path]]]:
    subjects: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for path in H5_DIR.glob("volume_*_slice_*.h5"):
        match = PATTERN.match(path.name)
        if match:
            subjects[int(match.group(1))].append((int(match.group(2)), path))
    for subject_id in subjects:
        subjects[subject_id].sort(key=lambda item: item[0])
    return dict(subjects)


def load_subject(files: list[tuple[int, Path]]) -> tuple[np.ndarray, np.ndarray]:
    images = []
    masks = []
    for _, path in files:
        try:
            with h5py.File(path, "r") as handle:
                image = np.asarray(handle["image"][()], dtype=np.float32)
                mask = np.asarray(handle["mask"][()])
        except OSError as error:
            print(f"Skipping unreadable file {path.name}: {error}")
            continue

        if image.ndim != 3 or image.shape[-1] != 4:
            raise ValueError(f"Unexpected image shape in {path}: {image.shape}")
        images.append(np.moveaxis(image, -1, 0))  # [4,H,W]
        masks.append((mask > 0).astype(np.uint8))

    if not images:
        raise RuntimeError("No readable slices were found for this subject.")

    return np.stack(images, axis=1), np.stack(masks, axis=0)


def main() -> None:
    if not H5_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {H5_DIR.resolve()}")

    subjects = collect_subjects()
    if not subjects:
        raise RuntimeError(f"No volume_X_slice_Y.h5 files found in {H5_DIR.resolve()}")

    ranked = []
    for subject_id, files in subjects.items():
        tumour_slices = 0
        for _, path in files:
            try:
                with h5py.File(path, "r") as handle:
                    tumour_slices += int(np.any(handle["mask"][()] > 0))
            except OSError:
                continue
        ranked.append((tumour_slices, len(files), subject_id))

    ranked.sort(reverse=True)
    chosen = [subject_id for tumour_count, depth, subject_id in ranked if tumour_count > 0][:NUMBER_OF_SAMPLES]
    if len(chosen) < NUMBER_OF_SAMPLES:
        chosen.extend([sid for _, _, sid in ranked if sid not in chosen][: NUMBER_OF_SAMPLES - len(chosen)])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Only remove existing 3D NPZ samples. The 2D folder is never touched.
    for old_file in OUTPUT_DIR.glob("*.npz"):
        old_file.unlink()

    for sample_number, subject_id in enumerate(chosen, start=1):
        image, mask = load_subject(subjects[subject_id])
        output_path = OUTPUT_DIR / f"sample_3d_full_{sample_number:02d}.npz"
        np.savez_compressed(
            output_path,
            image=image,
            mask=mask,
            volume_id=np.asarray(subject_id),
            modality_names=np.asarray(["FLAIR", "T1", "T1CE", "T2"]),
        )
        print(
            f"Saved {output_path} | image={image.shape} | mask={mask.shape} | "
            f"tumour_slices={int(np.any(mask > 0, axis=(1, 2)).sum())}"
        )


if __name__ == "__main__":
    main()
