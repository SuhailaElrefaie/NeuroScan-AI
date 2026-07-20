"""
Create small public sample files for the deployed Streamlit app.

This script copies 5 2D MRI images and creates 5 small 3D .npz volumes
that users can download from the website and re-upload to test the app.

Run from the project root:
    python make_public_samples.py
"""

import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


SAMPLE_2D_DIR = Path("sample_data/2d")
SAMPLE_3D_DIR = Path("sample_data/3d")

TWO_D_CANDIDATE_DIRS = [
    Path("Data/segmentation_task/test/images"),
    Path("Data/segmentation_task/train/images"),
]

H5_DIR = "archive/BraTS2020_training_data/content/data"


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def make_dirs():
    SAMPLE_2D_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_3D_DIR.mkdir(parents=True, exist_ok=True)


def clear_old_samples():
    for folder in [SAMPLE_2D_DIR, SAMPLE_3D_DIR]:
        for file in folder.glob("*"):
            if file.is_file():
                file.unlink()


def create_2d_samples(max_samples=5):
    source_files = []

    for folder in TWO_D_CANDIDATE_DIRS:
        if not folder.exists():
            continue

        files = [
            file for file in sorted(folder.iterdir())
            if file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]

        source_files.extend(files)

        if len(source_files) >= max_samples:
            break

    source_files = source_files[:max_samples]

    if len(source_files) == 0:
        print("No 2D sample images found. Skipping 2D samples.")
        return

    for index, source in enumerate(source_files, start=1):
        # Convert to PNG so the web app has consistent downloadable samples.
        output_path = SAMPLE_2D_DIR / f"sample_2d_{index:02d}.png"
        image = Image.open(source).convert("L")
        image.save(output_path)
        print(f"Created 2D sample: {output_path}")


def create_3d_samples(max_samples=5):
    if not Path(H5_DIR).exists():
        print(f"3D H5 folder not found: {H5_DIR}. Skipping 3D samples.")
        return

    try:
        from dataset_3d import BraTS3DVolumeDataset
    except Exception as error:
        print("Could not import BraTS3DVolumeDataset. Skipping 3D samples.")
        print(error)
        return

    dataset = BraTS3DVolumeDataset(
        h5_dir=H5_DIR,
        depth=16,
        image_size=(96, 96),
        augment=False,
        only_tumor_windows=True,
    )

    number_to_save = min(max_samples, len(dataset))

    for index in range(number_to_save):
        image_tensor, _ = dataset[index]

        # Save only the image because the public app predicts the mask itself.
        # Shape: [4, 16, 96, 96]
        image = image_tensor.cpu().numpy().astype(np.float32)

        output_path = SAMPLE_3D_DIR / f"sample_3d_{index + 1:02d}.npz"
        np.savez_compressed(output_path, image=image)
        print(f"Created 3D sample: {output_path}")


def main():
    make_dirs()
    clear_old_samples()
    create_2d_samples(max_samples=5)
    create_3d_samples(max_samples=5)

    print("\nDone. Commit and push these folders:")
    print("  sample_data/2d")
    print("  sample_data/3d")


if __name__ == "__main__":
    main()
