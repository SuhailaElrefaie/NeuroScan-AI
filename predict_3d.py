import os

import numpy as np
import torch
from PIL import Image

from dataset_3d import BraTS3DVolumeDataset
from models.unet3d import UNet3D


H5_DIR = "archive/BraTS2020_training_data/content/data"

# Use your latest quick 3D run first.
# Later, when best_model_3d updates properly, you can switch this to:
# MODEL_PATH = "best_model_3d/best_unet3d.pth"
MODEL_PATH = "runs_3d/run_3d_2026_05_14_191516/model_3d.pth"

DEPTH = 16
IMAGE_SIZE = (96, 96)
THRESHOLD = 0.5


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_3d_model():
    device = get_device()

    model = UNet3D(in_channels=4, out_channels=1).to(device)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"3D model not found: {MODEL_PATH}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    return model, device


def load_3d_dataset():
    dataset = BraTS3DVolumeDataset(
        h5_dir=H5_DIR,
        depth=DEPTH,
        image_size=IMAGE_SIZE,
        augment=False,
        only_tumor_windows=True
    )

    return dataset


def normalize_slice_for_display(slice_2d):
    slice_2d = np.asarray(slice_2d, dtype=np.float32)

    min_val = slice_2d.min()
    max_val = slice_2d.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(slice_2d, dtype=np.uint8)

    slice_2d = (slice_2d - min_val) / (max_val - min_val)
    slice_2d = (slice_2d * 255).astype(np.uint8)

    return slice_2d


def create_3d_overlay(image_slice, mask_slice, overlay_alpha=0.35):
    image_uint8 = normalize_slice_for_display(image_slice)
    image_rgb = np.stack([image_uint8] * 3, axis=-1)

    red_layer = np.zeros_like(image_rgb)
    red_layer[:, :, 0] = 255

    mask_3d = np.stack([mask_slice] * 3, axis=-1)

    overlay = np.where(
        mask_3d == 1,
        (1 - overlay_alpha) * image_rgb + overlay_alpha * red_layer,
        image_rgb
    )

    overlay = overlay.astype(np.uint8)

    return Image.fromarray(overlay)


def predict_3d_volume(volume_index=0, threshold=THRESHOLD, overlay_alpha=0.35):
    dataset = load_3d_dataset()

    if volume_index < 0 or volume_index >= len(dataset):
        raise IndexError(
            f"Volume index {volume_index} is invalid. "
            f"Dataset has {len(dataset)} volumes."
        )

    image, true_mask = dataset[volume_index]

    model, device = load_3d_model()

    image_batch = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_batch)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    pred_mask = (probs > threshold).astype(np.uint8)

    image_np = image.cpu().numpy()
    true_mask_np = true_mask.squeeze().cpu().numpy().astype(np.uint8)

    return {
        "image": image_np,
        "true_mask": true_mask_np,
        "probability": probs,
        "pred_mask": pred_mask,
        "num_volumes": len(dataset),
        "depth": image_np.shape[1],
        "volume_id": dataset.volume_ids[volume_index]
    }


def get_display_slices(result, slice_index, modality_index=0, overlay_alpha=0.35):
    image_volume = result["image"]
    true_mask = result["true_mask"]
    pred_mask = result["pred_mask"]
    probability = result["probability"]

    image_slice = image_volume[modality_index, slice_index, :, :]
    true_mask_slice = true_mask[slice_index, :, :]
    pred_mask_slice = pred_mask[slice_index, :, :]
    probability_slice = probability[slice_index, :, :]

    input_img = Image.fromarray(normalize_slice_for_display(image_slice))
    true_mask_img = Image.fromarray((true_mask_slice * 255).astype(np.uint8))
    pred_mask_img = Image.fromarray((pred_mask_slice * 255).astype(np.uint8))
    prob_img = Image.fromarray(normalize_slice_for_display(probability_slice))

    overlay_img = create_3d_overlay(
        image_slice,
        pred_mask_slice,
        overlay_alpha=overlay_alpha
    )

    return input_img, pred_mask_img, true_mask_img, overlay_img, prob_img