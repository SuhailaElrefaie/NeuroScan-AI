import cv2
import json
import os

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from gradcam import GradCAM, create_gradcam_overlay
from models.attention_unet2d import AttentionUNet2D


MODEL_PATH = "best_model/best_attention_unet2d.pth"
METRICS_PATH = "best_model/best_attention_metrics_2d.json"
IMAGE_SIZE = (256, 256)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model():
    device = get_device()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Attention model not found: {MODEL_PATH}. "
            "Copy best_attention_unet2d.pth into best_model/ first."
        )

    model = AttentionUNet2D(
        in_channels=1,
        out_channels=1,
        num_heads=8,
    ).to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, device


def get_image_transform():
    return T.Compose([
        T.Resize(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])


def postprocess_mask(mask: np.ndarray, min_area: int = 80) -> np.ndarray:
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.medianBlur(mask, 3)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255

    return (cleaned > 0).astype(np.uint8)


def get_best_threshold(default=0.30):
    if not os.path.exists(METRICS_PATH):
        return default

    try:
        with open(METRICS_PATH, "r") as file:
            metrics = json.load(file)

        return float(
            metrics.get(
                "Threshold",
                metrics.get(
                    "threshold",
                    metrics.get("best_threshold", default),
                ),
            )
        )
    except Exception:
        return default


def predict_mask(image_path, threshold=None, min_area=80):
    if threshold is None:
        threshold = get_best_threshold()

    model, device = load_model()
    transform = get_image_transform()

    original = Image.open(image_path).convert("L")
    resized = original.resize(IMAGE_SIZE)

    image_tensor = transform(original).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output).squeeze().cpu().numpy()

    candidate = (probability > threshold).astype(np.uint8)
    final_mask = postprocess_mask(candidate, min_area=min_area)

    return resized, final_mask


def create_overlay(image_pil, mask, overlay_alpha=0.35):
    image_rgb = np.array(image_pil.convert("RGB"))

    red_layer = np.zeros_like(image_rgb)
    red_layer[:, :, 0] = 255

    mask_3d = np.stack([mask] * 3, axis=-1)

    overlay = np.where(
        mask_3d == 1,
        (1 - overlay_alpha) * image_rgb + overlay_alpha * red_layer,
        image_rgb,
    )

    overlay = overlay.astype(np.uint8)

    return (
        Image.fromarray(overlay),
        Image.fromarray((mask * 255).astype(np.uint8)),
    )


def predict_with_gradcam(
    image_path,
    threshold=None,
    min_area=80,
    overlay_alpha=0.35,
    gradcam_alpha=0.40,
):
    if threshold is None:
        threshold = get_best_threshold()

    model, device = load_model()
    transform = get_image_transform()

    original = Image.open(image_path).convert("L")
    resized = original.resize(IMAGE_SIZE)

    image_tensor = transform(original).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output).squeeze().cpu().numpy()

    final_mask = (probability > threshold).astype(np.uint8)
    final_mask = postprocess_mask(final_mask, min_area=min_area)

    target_layer = model.dec2.layers[3]

    gradcam = GradCAM(model, target_layer)

    try:
        cam = gradcam.generate(image_tensor)
    finally:
        gradcam.remove_hooks()

    overlay, mask_only = create_overlay(
        resized,
        final_mask,
        overlay_alpha=overlay_alpha,
    )

    gradcam_overlay, heatmap_only = create_gradcam_overlay(
        resized,
        cam,
        heatmap_alpha=gradcam_alpha,
    )

    return (
        resized,
        final_mask,
        overlay,
        mask_only,
        gradcam_overlay,
        heatmap_only,
    )
