import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from gradcam import GradCAM, create_gradcam_overlay
from models.unet import UNet


MODEL_PATH = "best_model/best_unet.pth"
IMAGE_SIZE = (256, 256)


def get_device(): # Select the best available device for inference.
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model(): # Load the trained U-Net model from disk.
    device = get_device()

    model = UNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    return model, device


def get_image_transform(): # Create the preprocessing pipeline used before prediction.
    return T.Compose([ # Torchvision transform pipeline for resizing, converting to tensor, and normalizing grayscale MRI images.
        T.Resize(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])


def postprocess_mask(mask: np.ndarray, min_area: int = 80) -> np.ndarray: # Clean a predicted binary mask after model inference.
    """
    The function:
    - converts the mask to image scale
    - applies median blurring to reduce small noise
    - removes connected regions smaller than a chosen minimum area
    """

    mask = (mask * 255).astype(np.uint8)

    # Smooth small isolated noisy pixels
    mask = cv2.medianBlur(mask, 3)

    # Find separate connected regions in the mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    cleaned = np.zeros_like(mask)

    # Keep only regions large enough to be considered useful detections
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == label] = 255

    return (cleaned > 0).astype(np.uint8)


def predict_mask(image_path, threshold=0.35, min_area=80): # Predict a binary tumor mask from one MRI image.
    """
    Returns:
        A tuple containing:
        - resized MRI image as a PIL image
        - cleaned binary tumor mask as a NumPy array
    """

    model, device = load_model()
    transform = get_image_transform()

    original = Image.open(image_path).convert("L")
    resized = original.resize(IMAGE_SIZE)

    image_tensor = transform(original).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).squeeze().cpu().numpy()

    print("Prediction min/max:", float(prob.min()), float(prob.max()))

    candidate = (prob > threshold).astype(np.uint8)
    final_mask = postprocess_mask(candidate, min_area=min_area)

    print("Using threshold:", threshold)
    print("Mask pixels:", int(final_mask.sum()))

    return resized, final_mask


def create_overlay(image_pil, mask, overlay_alpha=0.35): # Create a red tumor overlay on top of an MRI image.
    """
    Returns:
        A tuple containing:
        - MRI image with red tumor overlay
        - black-and-white tumor mask image
    """

    image_rgb = np.array(image_pil.convert("RGB"))
    overlay = image_rgb.copy()

    # Create a red image layer used only where the mask is positive
    red_layer = np.zeros_like(image_rgb)
    red_layer[:, :, 0] = 255

    mask_3d = np.stack([mask] * 3, axis=-1)

    overlay = np.where(
        mask_3d == 1,
        (1 - overlay_alpha) * overlay + overlay_alpha * red_layer,
        overlay
    )

    overlay = overlay.astype(np.uint8)

    return (
        Image.fromarray(overlay),
        Image.fromarray((mask * 255).astype(np.uint8))
    )


def predict_with_gradcam( # Predict tumor segmentation results and generate a Grad-CAM explanation.
    image_path,
    threshold=0.35,
    min_area=80,
    overlay_alpha=0.35,
    gradcam_alpha=0.40
):
    """
    Returns:
        A tuple containing:
        - resized MRI image
        - binary tumor mask array
        - MRI image with segmentation overlay
        - black-and-white mask image
        - MRI image with Grad-CAM overlay
        - Grad-CAM heatmap image
    """

    model, device = load_model()
    transform = get_image_transform()

    original = Image.open(image_path).convert("L")
    resized = original.resize(IMAGE_SIZE)

    image_tensor = transform(original).unsqueeze(0).to(device)

    # Generate segmentation prediction
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).squeeze().cpu().numpy()

    final_mask = (prob > threshold).astype(np.uint8)
    final_mask = postprocess_mask(final_mask, min_area=min_area)

    # Use the second decoder block for Grad-CAM because it keeps more spatial detail than the lowest-resolution bottleneck layer
    target_layer = model.dec2.layers[3]

    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(image_tensor)
    gradcam.remove_hooks()

    overlay, mask_only = create_overlay(
        resized,
        final_mask,
        overlay_alpha=overlay_alpha
    )

    gradcam_overlay, heatmap_only = create_gradcam_overlay(
        resized,
        cam,
        heatmap_alpha=gradcam_alpha
    )

    return (
        resized,
        final_mask,
        overlay,
        mask_only,
        gradcam_overlay,
        heatmap_only
    )