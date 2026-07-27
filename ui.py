import io
import os
import json
import glob

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

import plotly.graph_objects as go

from predict import predict_with_gradcam


# =========================
# Page setup
# =========================

st.set_page_config(
    page_title="NeuroScan AI | Tumor Segmentation",
    layout="wide"
)


# =========================
# File paths
# =========================

BEST_METRICS_PATH = "best_model/best_metrics.json"
BEST_HISTORY_PATH = "best_model/best_history.csv"

BEST_METRICS_3D_PATH = "best_model_3d/best_metrics_3d.json"
BEST_HISTORY_3D_PATH = "best_model_3d/best_history_3d.csv"
BEST_MODEL_3D_PATH = "best_model_3d/best_unet3d.pth"

LATEST_3D_RUN_METRICS_PATH = "runs_3d/run_3d_2026_05_14_191516/metrics_3d.json"

DEPTH_3D = 32
IMAGE_SIZE_3D = (160, 160)

SAMPLE_2D_DIR = "sample_data/2d"
SAMPLE_3D_DIR = "sample_data/3d"


# =========================
# Helper functions
# =========================

def load_json(path):
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return json.load(f)


def load_csv(path):
    if not os.path.exists(path):
        return None

    return pd.read_csv(path)


def find_latest_file(pattern):
    files = glob.glob(pattern)

    if len(files) == 0:
        return None

    return max(files, key=os.path.getmtime)


def get_sample_files(folder, extensions):
    if not os.path.exists(folder):
        return []

    sample_files = []

    for extension in extensions:
        sample_files.extend(glob.glob(os.path.join(folder, f"*.{extension}")))

    return sorted(sample_files)


def read_file_bytes(path):
    with open(path, "rb") as file:
        return file.read()


def render_sample_downloads(folder, extensions, title, help_text, mime_type):
    sample_files = get_sample_files(folder, extensions)

    with st.expander(title, expanded=False):
        st.caption(help_text)

        if len(sample_files) == 0:
            st.warning(
                f"No sample files found in `{folder}`. Run `python make_public_samples.py` locally, "
                "then commit and push the generated `sample_data` folder."
            )
            return

        columns = st.columns(min(5, len(sample_files)))

        for index, path in enumerate(sample_files[:5]):
            file_name = os.path.basename(path)
            extension = os.path.splitext(path)[1].lower()

            with columns[index % len(columns)]:
                st.download_button(
                    label=f"⬇️ Download {index + 1}",
                    data=read_file_bytes(path),
                    file_name=file_name,
                    mime=mime_type,
                    use_container_width=True
                )


def render_sample_folder_sidebar(folder, extensions, title, help_text, mime_type):
    """Small sample-download folder shown in the analysis sidebar."""
    sample_files = get_sample_files(folder, extensions)

    with st.sidebar.expander(title, expanded=False):
        st.caption(help_text)

        if len(sample_files) == 0:
            st.warning(
                f"No sample files found in `{folder}`. Run `python make_public_samples.py`, "
                "then commit and push the generated `sample_data` folder."
            )
            return

        for index, path in enumerate(sample_files[:5]):
            file_name = os.path.basename(path)

            st.download_button(
                label=f"⬇️ Download {index + 1}",
                data=read_file_bytes(path),
                file_name=file_name,
                mime=mime_type,
                use_container_width=True,
                key=f"download_{title}_{index}_{file_name}"
            )


def render_export_2d_sidebar():
    """Export buttons for the latest 2D prediction. Rendered after prediction so it updates immediately."""
    with st.sidebar.expander("Export 2D Results", expanded=False):
        if "export_2d" not in st.session_state:
            st.caption("Upload a 2D image first to export results.")
            return

        export_2d = st.session_state["export_2d"]

        st.download_button(
            label="Segmentation Overlay",
            data=export_2d["overlay"],
            file_name="2d_segmentation_overlay.png",
            mime="image/png",
            use_container_width=True,
            key="export_2d_overlay"
        )

        st.download_button(
            label="Tumor Mask",
            data=export_2d["mask_only"],
            file_name="2d_tumor_mask.png",
            mime="image/png",
            use_container_width=True,
            key="export_2d_mask"
        )

        st.download_button(
            label="Grad-CAM Overlay",
            data=export_2d["gradcam_overlay"],
            file_name="2d_gradcam_overlay.png",
            mime="image/png",
            use_container_width=True,
            key="export_2d_gradcam"
        )


def render_export_3d_sidebar():
    """Export buttons for the latest selected 3D slice. Rendered after prediction so it updates immediately."""
    with st.sidebar.expander("Export 3D Slice Results", expanded=False):
        if "export_3d" not in st.session_state:
            st.caption("Upload a 3D volume first to export results.")
            return

        export_3d = st.session_state["export_3d"]

        st.download_button(
            label="MRI Slice",
            data=export_3d["input_img"],
            file_name=export_3d["input_name"],
            mime="image/png",
            use_container_width=True,
            key="export_3d_input"
        )

        st.download_button(
            label="Prediction Mask",
            data=export_3d["pred_mask_img"],
            file_name=export_3d["pred_name"],
            mime="image/png",
            use_container_width=True,
            key="export_3d_prediction"
        )

        st.download_button(
            label="Overlay",
            data=export_3d["overlay_img"],
            file_name=export_3d["overlay_name"],
            mime="image/png",
            use_container_width=True,
            key="export_3d_overlay"
        )

        if export_3d["true_mask_img"] is not None:
            st.download_button(
                label="Ground Truth Mask",
                data=export_3d["true_mask_img"],
                file_name=export_3d["true_name"],
                mime="image/png",
                use_container_width=True,
                key="export_3d_true"
            )

        st.download_button(
            label="Probability Map",
            data=export_3d["prob_img"],
            file_name=export_3d["prob_name"],
            mime="image/png",
            use_container_width=True,
            key="export_3d_probability"
        )


def image_to_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def normalize_slice_for_display(slice_2d):
    """Contrast-stretch MRI slices for display only.

    This makes uploaded MRI views look cleaner in the web app. It does not
    change the tensor sent into the model and does not affect Dice/metrics.
    """
    slice_2d = np.asarray(slice_2d, dtype=np.float32)
    slice_2d = np.nan_to_num(slice_2d)

    low, high = np.percentile(slice_2d, (1, 99))

    if high - low < 1e-8:
        low, high = float(slice_2d.min()), float(slice_2d.max())

    if high - low < 1e-8:
        return np.zeros_like(slice_2d, dtype=np.uint8)

    slice_2d = np.clip(slice_2d, low, high)
    slice_2d = (slice_2d - low) / (high - low + 1e-8)
    slice_2d = (slice_2d * 255).astype(np.uint8)

    return slice_2d


def resize_for_display(image, width=384, resample=None):
    """Upscale small 3D model inputs so Streamlit does not stretch them badly."""
    if image is None:
        return None

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    if resample is None:
        resample = Image.Resampling.LANCZOS

    w, h = image.size
    if w >= width:
        return image

    new_h = max(1, int(h * (width / w)))
    return image.resize((width, new_h), resample=resample)


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


def get_3d_display_slices(result, slice_index, modality_index=0, overlay_alpha=0.35):
    image_volume = result["image"]
    true_mask = result.get("true_mask", None)
    pred_mask = result["pred_mask"]
    probability = result["probability"]

    image_slice = image_volume[modality_index, slice_index, :, :]
    pred_mask_slice = pred_mask[slice_index, :, :]
    probability_slice = probability[slice_index, :, :]

    input_img = Image.fromarray(normalize_slice_for_display(image_slice))
    pred_mask_img = Image.fromarray((pred_mask_slice * 255).astype(np.uint8))
    prob_img = Image.fromarray(normalize_slice_for_display(probability_slice))

    overlay_img = create_3d_overlay(
        image_slice,
        pred_mask_slice,
        overlay_alpha=overlay_alpha
    )

    if true_mask is not None:
        true_mask_slice = true_mask[slice_index, :, :]
        true_mask_img = Image.fromarray((true_mask_slice * 255).astype(np.uint8))
    else:
        true_mask_img = None

    return input_img, pred_mask_img, true_mask_img, overlay_img, prob_img



def normalize_slice_for_viewer(slice_2d, window_mode="Auto contrast", window_center=0.0, window_width=4.0):
    """Medical-viewer style brightness/contrast for display only."""
    arr = np.asarray(slice_2d, dtype=np.float32)
    arr = np.nan_to_num(arr)

    if window_mode == "Manual":
        low = float(window_center) - float(window_width) / 2.0
        high = float(window_center) + float(window_width) / 2.0
    else:
        low, high = np.percentile(arr, (1, 99))

    if high - low < 1e-8:
        low, high = float(arr.min()), float(arr.max())

    if high - low < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)

    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low + 1e-8)
    return (arr * 255).astype(np.uint8)


def get_mask_boundary(mask_2d):
    """Return the contour/boundary pixels of a binary mask without extra dependencies."""
    mask = np.asarray(mask_2d).astype(bool)

    if mask.sum() == 0:
        return mask

    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    neighbors_all = (
        padded[:-2, 1:-1] &
        padded[2:, 1:-1] &
        padded[1:-1, :-2] &
        padded[1:-1, 2:] &
        padded[:-2, :-2] &
        padded[:-2, 2:] &
        padded[2:, :-2] &
        padded[2:, 2:]
    )
    return center & (~neighbors_all)


def create_real_viewer_image(
    image_slice,
    mask_slice,
    probability_slice=None,
    mode="Contour + Overlay",
    overlay_alpha=0.35,
    window_mode="Auto contrast",
    window_center=0.0,
    window_width=4.0,
    display_width=720
):
    """Create a PACS-like 2D viewer image with contour/overlay/windowing."""
    base = normalize_slice_for_viewer(
        image_slice,
        window_mode=window_mode,
        window_center=window_center,
        window_width=window_width
    )

    mask = np.asarray(mask_slice).astype(bool)
    boundary = get_mask_boundary(mask)
    rgb = np.stack([base] * 3, axis=-1).astype(np.float32)

    if mode == "MRI only":
        out = rgb

    elif mode == "Prediction mask":
        out = np.zeros_like(rgb)
        out[mask] = [255, 45, 45]
        out[boundary] = [255, 230, 80]

    elif mode == "Probability map" and probability_slice is not None:
        prob = np.asarray(probability_slice, dtype=np.float32)
        prob = np.nan_to_num(prob)
        prob = np.clip(prob, 0, 1)
        heat = np.zeros_like(rgb)
        heat[:, :, 0] = 255 * prob
        heat[:, :, 1] = 180 * np.maximum(prob - 0.35, 0) / 0.65
        out = (0.55 * rgb + 0.75 * heat)
        out = np.clip(out, 0, 255)

    else:
        out = rgb.copy()

        if mode in ["Overlay", "Contour + Overlay"]:
            red = np.zeros_like(rgb)
            red[:, :, 0] = 255
            out[mask] = (1 - overlay_alpha) * out[mask] + overlay_alpha * red[mask]

        if mode in ["Contour only", "Contour + Overlay"]:
            out[boundary] = [255, 35, 35]

            # Add a tiny yellow highlight where the contour is strongest/visible.
            if mask.sum() > 0:
                ys, xs = np.where(mask)
                cy, cx = int(np.mean(ys)), int(np.mean(xs))
                y0, y1 = max(0, cy - 2), min(out.shape[0], cy + 3)
                x0, x1 = max(0, cx - 2), min(out.shape[1], cx + 3)
                out[y0:y1, x0:x1] = [255, 230, 80]

    out = np.clip(out, 0, 255).astype(np.uint8)
    image = Image.fromarray(out)
    return resize_for_display(image, width=display_width)


def create_viewer_filmstrip(
    result,
    center_slice,
    modality_index=0,
    overlay_alpha=0.35,
    window_mode="Auto contrast",
    window_center=0.0,
    window_width=4.0,
    count=9,
    thumb_size=92
):
    """Create a small slice filmstrip like a lightweight medical viewer."""
    depth = result["depth"]
    half = count // 2
    indices = [min(max(0, center_slice + i), depth - 1) for i in range(-half, half + 1)]

    thumbs = []
    for idx in indices:
        image_slice = result["image"][modality_index, idx]
        mask_slice = result["pred_mask"][idx]
        prob_slice = result["probability"][idx]

        thumb = create_real_viewer_image(
            image_slice,
            mask_slice,
            prob_slice,
            mode="Contour + Overlay",
            overlay_alpha=overlay_alpha,
            window_mode=window_mode,
            window_center=window_center,
            window_width=window_width,
            display_width=thumb_size
        ).resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        thumbs.append((idx, thumb))

    strip_h = thumb_size + 24
    strip_w = len(thumbs) * (thumb_size + 8) - 8
    strip = Image.new("RGB", (strip_w, strip_h), color=(7, 11, 16))
    draw = ImageDraw.Draw(strip)

    x = 0
    for idx, thumb in thumbs:
        strip.paste(thumb, (x, 0))
        outline = (0, 123, 255) if idx == center_slice else (55, 65, 81)
        draw.rectangle([x, 0, x + thumb_size - 1, thumb_size - 1], outline=outline, width=2)
        draw.text((x + 6, thumb_size + 5), f"S{idx}", fill=(210, 210, 210))
        x += thumb_size + 8

    return strip


def preprocess_uploaded_3d_mask(mask_array):
    import torch
    import torch.nn.functional as F

    mask = np.asarray(mask_array)
    mask = np.nan_to_num(mask)

    if mask.ndim == 4:
        mask = np.squeeze(mask)

    if mask.ndim != 3:
        return None

    mask_tensor = torch.from_numpy((mask > 0).astype(np.float32))
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)

    mask_tensor = F.interpolate(
        mask_tensor,
        size=(DEPTH_3D, IMAGE_SIZE_3D[0], IMAGE_SIZE_3D[1]),
        mode="nearest"
    )

    return (mask_tensor.squeeze().numpy() > 0.5).astype(np.uint8)


def get_representative_3d_indices(mask_3d):
    """Find useful axial/coronal/sagittal indices from the prediction mask."""
    mask_3d = np.asarray(mask_3d)
    depth, height, width = mask_3d.shape

    coords = np.argwhere(mask_3d > 0)

    if coords.size == 0:
        return depth // 2, height // 2, width // 2

    slice_areas = mask_3d.reshape(depth, -1).sum(axis=1)
    z_index = int(np.argmax(slice_areas))
    y_index = int(np.median(coords[:, 1]))
    x_index = int(np.median(coords[:, 2]))

    return z_index, y_index, x_index


def get_orthogonal_3d_views(result, modality_index=0, overlay_alpha=0.35):
    """Create medical-style axial/coronal/sagittal overlays."""
    image_volume = result["image"]
    pred_mask = result["pred_mask"]

    z_index, y_index, x_index = get_representative_3d_indices(pred_mask)

    axial_img = image_volume[modality_index, z_index, :, :]
    axial_mask = pred_mask[z_index, :, :]

    coronal_img = image_volume[modality_index, :, y_index, :]
    coronal_mask = pred_mask[:, y_index, :]

    sagittal_img = image_volume[modality_index, :, :, x_index]
    sagittal_mask = pred_mask[:, :, x_index]

    views = {
        "Axial": {
            "index": z_index,
            "image": resize_for_display(Image.fromarray(normalize_slice_for_display(axial_img))),
            "overlay": resize_for_display(create_3d_overlay(axial_img, axial_mask, overlay_alpha)),
        },
        "Coronal": {
            "index": y_index,
            "image": resize_for_display(Image.fromarray(normalize_slice_for_display(coronal_img))),
            "overlay": resize_for_display(create_3d_overlay(coronal_img, coronal_mask, overlay_alpha)),
        },
        "Sagittal": {
            "index": x_index,
            "image": resize_for_display(Image.fromarray(normalize_slice_for_display(sagittal_img))),
            "overlay": resize_for_display(create_3d_overlay(sagittal_img, sagittal_mask, overlay_alpha)),
        },
    }

    return views, z_index


def predict_3d_volume(volume_index=0, threshold=0.5):
    from predict_3d import predict_3d_volume as run_predict_3d

    result = run_predict_3d(
        volume_index=int(volume_index),
        threshold=threshold
    )

    result["source_type"] = "demo"
    return result


def get_device_3d():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_3d_model_path():
    latest_run_model = find_latest_file("runs_3d/run_3d_*/model_3d.pth")

    if os.path.exists(BEST_MODEL_3D_PATH):
        return BEST_MODEL_3D_PATH

    if latest_run_model is not None:
        return latest_run_model

    return None


def preprocess_uploaded_3d_array(array):
    import torch
    import torch.nn.functional as F

    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array)

    # Accepted shapes:
    # [4, D, H, W]
    # [D, H, W, 4]
    # [D, H, W] -> repeated into 4 channels
    if array.ndim == 4:
        if array.shape[0] == 4:
            image = array
        elif array.shape[-1] == 4:
            image = np.moveaxis(array, -1, 0)
        else:
            raise ValueError(
                "Uploaded 4D volume must have 4 channels as either "
                "[4, D, H, W] or [D, H, W, 4]."
            )

    elif array.ndim == 3:
        # If the upload has only one MRI channel, repeat it to 4 channels
        # so it can enter the 3D model.
        image = np.stack([array] * 4, axis=0)

    else:
        raise ValueError(
            "Uploaded volume must be 3D or 4D. Expected [4,D,H,W], [D,H,W,4], or [D,H,W]."
        )

    mean = image.mean()
    std = image.std()

    if std < 1e-8:
        image = image * 0.0
    else:
        image = (image - mean) / std

    image_tensor = torch.from_numpy(image).float()

    # Add batch dimension: [1, 4, D, H, W]
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = F.interpolate(
        image_tensor,
        size=(DEPTH_3D, IMAGE_SIZE_3D[0], IMAGE_SIZE_3D[1]),
        mode="trilinear",
        align_corners=False
    )

    # Remove batch dimension: [4, DEPTH_3D, H, W]
    image_tensor = image_tensor.squeeze(0)

    return image_tensor


def predict_uploaded_3d_npz(uploaded_file, threshold=0.5):
    import torch
    from models.unet3d import UNet3D

    data = np.load(uploaded_file)

    if "image" in data.files:
        array = data["image"]
    else:
        first_key = data.files[0]
        array = data[first_key]

    true_mask = None
    if "mask" in data.files:
        true_mask = preprocess_uploaded_3d_mask(data["mask"])

    image_tensor = preprocess_uploaded_3d_array(array)

    model_path = get_3d_model_path()

    if model_path is None:
        raise FileNotFoundError(
            "No 3D model found. Expected best_model_3d/best_unet3d.pth "
            "or a model_3d.pth inside runs_3d."
        )

    device = get_device_3d()

    model = UNet3D(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image_batch = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_batch)
        probability = torch.sigmoid(logits).squeeze().cpu().numpy()

    pred_mask = (probability > threshold).astype(np.uint8)

    return {
        "image": image_tensor.cpu().numpy(),
        "true_mask": true_mask,
        "probability": probability,
        "pred_mask": pred_mask,
        "num_volumes": 1,
        "depth": image_tensor.shape[1],
        "volume_id": "uploaded",
        "source_type": "upload",
        "model_path": model_path
    }


def create_3d_tumor_volume_render(mask_3d):
    """
    Clean 3D visualization for the predicted tumor only.

    This replaces the old noisy MRI point-cloud plot. It crops around the
    predicted tumor so the 3D view is lighter, cleaner, and easier to read.
    """
    mask = np.asarray(mask_3d, dtype=np.float32)

    if mask.sum() == 0:
        return None

    coords = np.argwhere(mask > 0)

    if coords.size == 0:
        return None

    depth, height, width = mask.shape
    pad = 3

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    z_min = max(0, z_min - pad)
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)

    z_max = min(depth - 1, z_max + pad)
    y_max = min(height - 1, y_max + pad)
    x_max = min(width - 1, x_max + pad)

    cropped = mask[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]

    z, y, x = np.mgrid[
        z_min:z_max + 1,
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=cropped.flatten(),
            isomin=0.5,
            isomax=1.0,
            opacity=0.28,
            surface_count=5,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorscale="Reds",
            name="Predicted Tumor Volume"
        )
    )

    zc, yc, xc = coords.mean(axis=0)

    fig.add_trace(
        go.Scatter3d(
            x=[xc],
            y=[yc],
            z=[zc],
            mode="markers",
            marker=dict(
                size=7,
                color="gold",
                symbol="diamond"
            ),
            name="Tumor Center"
        )
    )

    fig.update_layout(
        title="3D Predicted Tumor Volume",
        height=620,
        margin=dict(l=0, r=0, b=0, t=45),
        scene=dict(
            xaxis_title="Width",
            yaxis_title="Height",
            zaxis_title="Slice",
            aspectmode="data",
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        legend=dict(x=0, y=1)
    )

    return fig


def create_tumor_area_by_slice_chart(mask_3d):
    """
    Shows how the predicted tumor area changes across MRI slices.
    This is clearer and more report-friendly than a noisy 3D point cloud.
    """
    mask = np.asarray(mask_3d, dtype=np.uint8)

    slice_areas = mask.sum(axis=(1, 2))

    if slice_areas.max() == 0:
        return None

    slices = np.arange(len(slice_areas))
    largest_slice = int(slice_areas.argmax())

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=slices,
            y=slice_areas,
            mode="lines+markers",
            name="Predicted tumor area",
            line=dict(width=3),
            marker=dict(size=6)
        )
    )

    fig.add_vline(
        x=largest_slice,
        line_width=2,
        line_dash="dash",
        annotation_text=f"Largest slice: {largest_slice}",
        annotation_position="top"
    )

    fig.update_layout(
        title="Predicted Tumor Area Across MRI Slices",
        xaxis_title="MRI Slice",
        yaxis_title="Predicted Tumor Pixels",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


# =========================
# Styling
# =========================

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7f9;
    }

    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: 600;
    }

    .summary-card {
        background-color: #171b22;
        border: 1px solid #303642;
        border-radius: 14px;
        padding: 22px 24px;
        min-height: 150px;
    }

    .summary-title {
        font-size: 1rem;
        font-weight: 600;
        color: #c9d1d9;
        margin-bottom: 12px;
    }

    .summary-value {
        font-size: 2.3rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.1;
        margin-bottom: 10px;
    }

    .summary-note {
        font-size: 0.9rem;
        color: #8b949e;
        line-height: 1.3;
    }

    .home-brain-wrap {
        display: flex;
        justify-content: center;
        margin-top: 10px;
        margin-bottom: 8px;
    }

    .home-brain-icon {
        width: 118px;
        max-width: 28vw;
        filter: drop-shadow(0 0 12px rgba(0, 123, 255, 0.25));
    }


    .guide-card {
        background-color: #171b22;
        border: 1px solid #303642;
        border-radius: 14px;
        padding: 20px 22px;
        min-height: 155px;
        margin-bottom: 10px;
    }

    .guide-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background-color: #007bff;
        color: white;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .guide-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .guide-text {
        font-size: 0.92rem;
        color: #aeb6c2;
        line-height: 1.45;
    }

    .route-card {
        background-color: #11161d;
        border: 1px solid #2d3440;
        border-radius: 14px;
        padding: 18px 20px;
        min-height: 120px;
    }

    .route-title {
        color: #ffffff;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 6px;
    }

    .route-text {
        color: #aeb6c2;
        font-size: 0.9rem;
        line-height: 1.4;
    }

    .view-note {
        background-color: #eef6ff;
        border-left: 4px solid #007bff;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 14px;
        color: #1f2937;
        font-size: 0.95rem;
    }


    .viewer-panel {
        background-color: #070b10;
        border: 1px solid #1f2937;
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.20);
    }

    .viewer-title {
        color: #e5e7eb;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 4px;
    }

    .viewer-subtitle {
        color: #9ca3af;
        font-size: 0.88rem;
        margin-bottom: 10px;
    }

    .viewer-badge {
        display: inline-block;
        background-color: #111827;
        border: 1px solid #374151;
        color: #d1d5db;
        border-radius: 999px;
        padding: 6px 10px;
        margin-right: 6px;
        margin-bottom: 6px;
        font-size: 0.82rem;
    }

    .viewer-small-note {
        color: #6b7280;
        font-size: 0.82rem;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# Initial page state
# =========================

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Home"

if "current_page" not in st.session_state:
    st.session_state["current_page"] = st.session_state["active_page"]

if "upload_reset_counter" not in st.session_state:
    st.session_state["upload_reset_counter"] = 0

page = st.session_state["active_page"]


# =========================
# Header
# =========================

brain_icon = ""

if page == "Home":
    brain_icon = """
    <div class='home-brain-wrap'>
        <svg class='home-brain-icon' viewBox='0 0 220 170' fill='none' xmlns='http://www.w3.org/2000/svg'>
            <path d='M78 130C48 130 28 109 28 80C28 54 46 35 70 35C76 19 91 10 109 14C120 5 139 7 150 20C173 20 192 40 192 66C205 76 207 101 191 116C182 132 164 138 146 133C136 145 116 149 102 137C94 142 84 139 78 130Z' stroke='#007bff' stroke-width='6' stroke-linecap='round' stroke-linejoin='round'/>
            <path d='M72 35C65 50 68 63 82 70' stroke='#007bff' stroke-width='4' stroke-linecap='round'/>
            <path d='M110 14C100 29 101 45 116 55' stroke='#007bff' stroke-width='4' stroke-linecap='round'/>
            <path d='M150 20C143 34 146 48 160 58' stroke='#007bff' stroke-width='4' stroke-linecap='round'/>
            <path d='M82 70C67 76 62 93 72 107' stroke='#007bff' stroke-width='4' stroke-linecap='round'/>
            <path d='M116 55C104 68 108 85 123 92' stroke='#007bff' stroke-width='4' stroke-linecap='round'/>
            <path d='M160 58C147 68 148 88 163 97' stroke='#007bff' stroke-width='4' stroke-linecap='round'/>
            <path d='M72 107C86 103 96 108 102 137' stroke='#007bff' stroke-width='4' stroke-linecap='round'/>
            <path d='M123 92C112 105 119 123 136 130' stroke='#007bff' stroke-width='4' stroke-linecap='round'/>
            <path d='M163 97C152 105 148 118 146 133' stroke='#007bff' stroke-width='4' stroke-linecap='round'/>
            <path d='M92 86C102 78 115 78 126 86' stroke='#007bff' stroke-width='3.5' stroke-linecap='round'/>
            <path d='M132 72C141 67 153 69 161 78' stroke='#007bff' stroke-width='3.5' stroke-linecap='round'/>
            <path d='M59 84C70 83 78 88 83 98' stroke='#007bff' stroke-width='3.5' stroke-linecap='round'/>
        </svg>
    </div>
    """

st.markdown(
    f"""
    {brain_icon}
    <h1 style='text-align: center;'>AI-Assisted Brain Tumor MRI Segmentation</h1>
    <p style='text-align: center; font-size: 1.15rem; color: #666666;'>
        2D and 3D U-Net segmentation with visual explanation
    </p>
    """,
    unsafe_allow_html=True
)

# =========================
# Main-page navigation
# =========================

# Pages are now controlled from the main screen instead of two separate
# sidebar radio groups. The rest of the UI is kept the same.

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Home"

if "current_page" not in st.session_state:
    st.session_state["current_page"] = st.session_state["active_page"]

if "upload_reset_counter" not in st.session_state:
    st.session_state["upload_reset_counter"] = 0


def go_to_page(page_name):
    st.session_state["active_page"] = page_name


def render_workflow_buttons(workflow):
    """Show Info / Analysis / Training buttons for one workflow."""
    if workflow == "2D":
        info_page = "2D Info"
        analysis_page = "2D MRI Analysis"
        training_page = "2D Training Progress"
    else:
        info_page = "3D Info"
        analysis_page = "3D MRI Analysis"
        training_page = "3D Training Progress"

    c1, c2, c3 = st.columns(3)

    with c1:
        st.button(f"{workflow} Info", on_click=go_to_page, args=(info_page,))

    with c2:
        st.button(f"{workflow} Analysis", on_click=go_to_page, args=(analysis_page,))

    with c3:
        st.button(f"{workflow} Training", on_click=go_to_page, args=(training_page,))


def render_home_button_bottom():
    """Show a small Home button at the bottom-right of non-home pages."""
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    left_space, home_col = st.columns([6, 1])

    with home_col:
        st.button(
            "← Home",
            on_click=go_to_page,
            args=("Home",),
            key=f"home_bottom_{st.session_state['active_page'].replace(' ', '_')}"
        )


page = st.session_state["active_page"]

# Hide the sidebar everywhere except the two Analysis pages.
# The sidebar is only used for analysis controls/export buttons.
if page not in ["2D MRI Analysis", "3D MRI Analysis"]:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        [data-testid="collapsedControl"] {
            display: none;
        }
        section[data-testid="stSidebar"] + div {
            margin-left: 0rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Reset uploaded files and stored results when changing pages
if page != st.session_state["current_page"]:
    keys_to_clear = [
        "result_3d",
        "input_mode_3d_previous",
        "export_2d",
        "export_3d",
        "last_3d_upload_signature"
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state["upload_reset_counter"] += 1
    st.session_state["current_page"] = page

# =========================
# Sidebar controls
# =========================

if page == "2D MRI Analysis":
    render_sample_folder_sidebar(
        folder=SAMPLE_2D_DIR,
        extensions=["png", "jpg", "jpeg"],
        title="📁 Test 2D samples",
        help_text="Download a sample MRI image, then upload it in the 2D analysis page.",
        mime_type="image/png"
    )

    with st.sidebar.expander("2D Controls", expanded=False):
        threshold = st.slider(
            "Segmentation Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.40,
            step=0.05,
            help="Lower values include more predicted tumor pixels. Higher values make the mask stricter."
        )

        min_area = st.slider(
            "Minimum Region Size",
            min_value=0,
            max_value=500,
            value=80,
            step=10,
            help="Removes detected regions smaller than this number of pixels."
        )

        overlay_alpha = st.slider(
            "Segmentation Overlay Opacity",
            min_value=0.10,
            max_value=0.90,
            value=0.35,
            step=0.05,
            help="Controls how strongly the red tumor overlay appears."
        )

        gradcam_alpha = st.slider(
            "Grad-CAM Opacity",
            min_value=0.10,
            max_value=0.90,
            value=0.40,
            step=0.05,
            help="Controls how strongly the Grad-CAM heatmap appears."
        )


elif page == "3D MRI Analysis":
    render_sample_folder_sidebar(
        folder=SAMPLE_3D_DIR,
        extensions=["npz"],
        title="📁 Test 3D samples",
        help_text="Download a sample .npz volume, then upload it in the 3D analysis page.",
        mime_type="application/octet-stream"
    )

    with st.sidebar.expander("3D Controls", expanded=False):
        threshold_3d = st.slider(
            "3D Segmentation Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05
        )

        overlay_alpha_3d = st.slider(
            "3D Overlay Opacity",
            min_value=0.10,
            max_value=0.90,
            value=0.35,
            step=0.05
        )

        modality_index = st.selectbox(
            "MRI Modality Channel",
            options=[0, 1, 2, 3],
            format_func=lambda x: f"Channel {x}"
        )

        viewer_mode_3d = st.selectbox(
            "Viewer Layer",
            options=["Contour + Overlay", "Contour only", "Overlay", "MRI only", "Prediction mask", "Probability map"],
            index=0,
            help="Controls what the main medical-style viewer displays."
        )

        window_mode_3d = st.selectbox(
            "Window / Level",
            options=["Auto contrast", "Manual"],
            index=0,
            help="Auto contrast usually looks best for normalized MRI samples. Manual lets you adjust brightness/contrast."
        )

        window_center_3d = st.slider(
            "Window Center",
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.1,
            disabled=(window_mode_3d == "Auto contrast")
        )

        window_width_3d = st.slider(
            "Window Width",
            min_value=0.5,
            max_value=8.0,
            value=4.0,
            step=0.1,
            disabled=(window_mode_3d == "Auto contrast")
        )


# =========================
# Model metric summary values
# =========================

best_metrics = load_json(BEST_METRICS_PATH)
best_metrics_3d = load_json(BEST_METRICS_3D_PATH)

latest_3d_metrics_path = find_latest_file("runs_3d/run_3d_*/metrics_3d.json")

if best_metrics_3d is None and latest_3d_metrics_path is not None:
    best_metrics_3d = load_json(latest_3d_metrics_path)

if best_metrics is not None:
    best_dice = best_metrics.get("Dice coefficient", 0)
    dice_text = f"{best_dice:.4f}"
else:
    dice_text = "No model yet"

if best_metrics_3d is not None:
    best_dice_3d = best_metrics_3d.get("Dice coefficient", 0)
    dice_3d_text = f"{best_dice_3d:.4f}"
else:
    dice_3d_text = "No best model yet"

# =========================
# Home / workflow pages
# =========================

if page == "Home":
    st.subheader("How to use this website")

    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown(
            """
            <div class='guide-card'>
                <div class='guide-number'>1</div>
                <div class='guide-title'>Choose 2D or 3D</div>
                <div class='guide-text'>
                    Use <b>2D</b> for one MRI image such as PNG or JPG. Use <b>3D</b> for a volume file in NPZ format.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with g2:
        st.markdown(
            """
            <div class='guide-card'>
                <div class='guide-number'>2</div>
                <div class='guide-title'>Open Analysis</div>
                <div class='guide-text'>
                    Go to the Analysis page, download one of the sample files from the sidebar, then upload it to test the model.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with g3:
        st.markdown(
            """
            <div class='guide-card'>
                <div class='guide-number'>3</div>
                <div class='guide-title'>View the result</div>
                <div class='guide-text'>
                    The app shows the MRI input, predicted tumor mask, overlay, and model confidence/visual explanation.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("Choose MRI Workflow")

    col1, col2 = st.columns(2)

    with col1:
        st.button("2D MRI Workflow", on_click=go_to_page, args=("2D Info",))
        st.caption("Best for testing a single MRI slice image. The prediction runs automatically after upload.")

    with col2:
        st.button("3D MRI Workflow", on_click=go_to_page, args=("3D Info",))
        st.caption("Best for testing a small 3D MRI volume in .npz format. Sample files are available in the sidebar.")

    st.markdown("---")
    st.subheader("What each section means")

    n1, n2, n3 = st.columns(3)

    with n1:
        st.markdown(
            """
            <div class='route-card'>
                <div class='route-title'>Info</div>
                <div class='route-text'>Explains the selected model, input format, and saved performance score.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with n2:
        st.markdown(
            """
            <div class='route-card'>
                <div class='route-title'>Analysis</div>
                <div class='route-text'>Main testing page. Upload an MRI file and view the segmentation output.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with n3:
        st.markdown(
            """
            <div class='route-card'>
                <div class='route-title'>Training</div>
                <div class='route-text'>Shows the model training curves and validation metrics used for evaluation.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    st.info("Tip: If you do not have MRI files, open an Analysis page and use the sample downloads in the sidebar.")


elif page == "2D Info":
    render_workflow_buttons("2D")
    st.markdown("---")
    st.subheader("2D MRI Workflow")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Architecture", "U-Net")
    with c2:
        st.metric("Input", "2D grayscale MRI")
    with c3:
        st.metric("Best Dice", dice_text)

    st.markdown("""
    This section is for single-slice MRI tumor segmentation.

    - **Analysis:** upload a PNG/JPG MRI slice and generate prediction outputs
    - **Training:** view the saved 2D model performance and training curves
    """)


elif page == "3D Info":
    render_workflow_buttons("3D")
    st.markdown("---")
    st.subheader("3D MRI Workflow")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Architecture", "3D U-Net")
    with c2:
        st.metric("Input", "4-channel MRI volume")
    with c3:
        st.metric("Best Dice", dice_3d_text)

    st.markdown("""
    This section is for volume-based MRI tumor segmentation.

    - **Analysis:** download a sample `.npz` file or upload your own 3D MRI volume
    - **Training:** view the saved 3D model performance and training curves
    """)


# =========================
# Page 1: 2D MRI Analysis
# =========================

elif page == "2D MRI Analysis":
    render_workflow_buttons("2D")
    st.markdown("---")
    st.subheader("2D MRI Slice Analysis")


    uploaded_file = st.file_uploader(
        "Upload MRI Scan Image (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        key=f"upload_2d_{st.session_state['upload_reset_counter']}"
    )

    if uploaded_file:
        with open("temp_upload.png", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Analyzing MRI scan..."):
            (
                resized_img,
                mask,
                overlay,
                mask_only,
                gradcam_overlay,
                heatmap_only
            ) = predict_with_gradcam(
                "temp_upload.png",
                threshold=threshold,
                min_area=min_area,
                overlay_alpha=overlay_alpha,
                gradcam_alpha=gradcam_alpha
            )
            
            st.session_state["export_2d"] = {
                "overlay": image_to_png_bytes(overlay),
                "mask_only": image_to_png_bytes(mask_only),
                "gradcam_overlay": image_to_png_bytes(gradcam_overlay)
            }

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("### Input Scan")
            st.image(
                resized_img,
                use_container_width=True,
                caption="Pre-processed Grayscale MRI"
            )

        with col2:
            st.markdown("### Segmentation")
            st.image(
                overlay,
                use_container_width=True,
                caption="Red: Predicted Tumor Region"
            )

        with col3:
            st.markdown("### Tumor Mask")
            st.image(
                mask_only,
                use_container_width=True,
                caption="White: Predicted Tumor Pixels"
            )

        with col4:
            st.markdown("### Grad-CAM")
            st.image(
                gradcam_overlay,
                use_container_width=True,
                caption="Model Attention Heatmap"
            )

        st.markdown("---")

        m_col1, m_col2 = st.columns(2)

        tumor_pixels = int(mask.sum())

        if tumor_pixels > 0:
            detection_status = "Detected"
            detection_note = "Tumor region predicted by the model"
        else:
            detection_status = "Not detected"
            detection_note = "No tumor region predicted by the model"

        with m_col1:
            st.metric("Prediction Result", detection_status)
            st.caption(detection_note)

        with m_col2:
            st.metric("Predicted Mask Area", f"{tumor_pixels} px")
            st.caption("Number of pixels included in the predicted mask")


    else:
        st.info(
            "Upload a PNG or JPG MRI slice to generate a predicted tumor overlay, "
            "binary tumor mask, and Grad-CAM explanation."
        )


# =========================
# Page 2: 3D MRI Analysis
# =========================

elif page == "3D MRI Analysis":
    render_workflow_buttons("3D")
    st.markdown("---")
    st.subheader("3D MRI Volume Analysis")

    uploaded_3d_file = st.file_uploader(
        "Upload 3D MRI volume (.NPZ)",
        type=["npz"],
        help="Expected array shape: [4, D, H, W], [D, H, W, 4], or [D, H, W]. If the file contains arrays named 'image' and 'mask', both will be used.",
        key=f"upload_3d_{st.session_state['upload_reset_counter']}"
    )

    if uploaded_3d_file:
        upload_signature = (
            f"{uploaded_3d_file.name}_{uploaded_3d_file.size}_"
            f"{threshold_3d}_{overlay_alpha_3d}_{modality_index}_{viewer_mode_3d}_"
            f"{window_mode_3d}_{window_center_3d}_{window_width_3d}"
        )

        if st.session_state.get("last_3d_upload_signature") != upload_signature:
            with st.spinner("Analyzing 3D MRI volume..."):
                result = predict_uploaded_3d_npz(
                    uploaded_file=uploaded_3d_file,
                    threshold=threshold_3d
                )

            st.session_state["result_3d"] = result
            st.session_state["last_3d_upload_signature"] = upload_signature

    else:
        if "result_3d" in st.session_state:
            del st.session_state["result_3d"]
        if "export_3d" in st.session_state:
            del st.session_state["export_3d"]
        if "last_3d_upload_signature" in st.session_state:
            del st.session_state["last_3d_upload_signature"]

        st.info(
            "Upload a .NPZ 3D MRI volume to open the clinical-style viewer with slice scrolling, "
            "tumor contour overlay, prediction mask, probability map, and tumor-area chart."
        )

    if "result_3d" in st.session_state:
        result = st.session_state["result_3d"]

        suggested_slice, _, _ = get_representative_3d_indices(result["pred_mask"])
        tumor_voxels = int(result["pred_mask"].sum())
        true_voxels = int(result["true_mask"].sum()) if result.get("true_mask") is not None else None

        st.markdown("### Clinical-Style MRI Viewer")
        st.markdown(
            f"""
            <div class='viewer-panel'>
                <div class='viewer-title'>3D MRI segmentation viewer</div>
                <div class='viewer-subtitle'>Slice-based inspection with window/level display, tumor contour overlay, and model probability review.</div>
                <span class='viewer-badge'>Model input: {DEPTH_3D} × {IMAGE_SIZE_3D[0]} × {IMAGE_SIZE_3D[1]}</span>
                <span class='viewer-badge'>Modality channel: {modality_index}</span>
                <span class='viewer-badge'>Threshold: {threshold_3d:.2f}</span>
                <span class='viewer-badge'>Predicted voxels: {tumor_voxels}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        slice_index = st.slider(
            "Scroll through MRI slices",
            min_value=0,
            max_value=result["depth"] - 1,
            value=suggested_slice,
            step=1,
            help="This behaves like scrolling through a lightweight medical viewer. The default slice is where the predicted tumor area is largest."
        )

        image_slice = result["image"][modality_index, slice_index]
        pred_mask_slice = result["pred_mask"][slice_index]
        probability_slice = result["probability"][slice_index]
        true_mask_slice = result["true_mask"][slice_index] if result.get("true_mask") is not None else None

        viewer_img_display = create_real_viewer_image(
            image_slice=image_slice,
            mask_slice=pred_mask_slice,
            probability_slice=probability_slice,
            mode=viewer_mode_3d,
            overlay_alpha=overlay_alpha_3d,
            window_mode=window_mode_3d,
            window_center=window_center_3d,
            window_width=window_width_3d,
            display_width=720
        )

        input_img, pred_mask_img, true_mask_img, overlay_img, prob_img = get_3d_display_slices(
            result,
            slice_index=slice_index,
            modality_index=modality_index,
            overlay_alpha=overlay_alpha_3d
        )

        pred_mask_img_display = resize_for_display(pred_mask_img, width=320, resample=Image.Resampling.NEAREST)
        prob_img_display = resize_for_display(prob_img, width=320)
        overlay_img_display = resize_for_display(overlay_img, width=320)
        true_mask_img_display = resize_for_display(true_mask_img, width=320, resample=Image.Resampling.NEAREST) if true_mask_img is not None else None

        st.session_state["export_3d"] = {
            "input_img": image_to_png_bytes(viewer_img_display),
            "pred_mask_img": image_to_png_bytes(pred_mask_img_display),
            "overlay_img": image_to_png_bytes(overlay_img_display),
            "true_mask_img": image_to_png_bytes(true_mask_img_display) if true_mask_img_display is not None else None,
            "prob_img": image_to_png_bytes(prob_img_display),
            "input_name": f"3d_volume_{result['volume_id']}_slice_{slice_index}_viewer.png",
            "pred_name": f"3d_volume_{result['volume_id']}_slice_{slice_index}_prediction_mask.png",
            "overlay_name": f"3d_volume_{result['volume_id']}_slice_{slice_index}_overlay.png",
            "true_name": f"3d_volume_{result['volume_id']}_slice_{slice_index}_ground_truth.png",
            "prob_name": f"3d_volume_{result['volume_id']}_slice_{slice_index}_probability.png"
        }

        main_view, side_view = st.columns([2.25, 1])

        with main_view:
            st.image(
                viewer_img_display,
                use_container_width=True,
                caption=f"Viewer mode: {viewer_mode_3d} | Slice {slice_index}"
            )

            filmstrip = create_viewer_filmstrip(
                result,
                center_slice=slice_index,
                modality_index=modality_index,
                overlay_alpha=overlay_alpha_3d,
                window_mode=window_mode_3d,
                window_center=window_center_3d,
                window_width=window_width_3d
            )
            st.image(filmstrip, use_container_width=True, caption="Slice filmstrip around the selected slice")

        with side_view:
            st.markdown("#### Segmentation panels")
            st.image(pred_mask_img_display, use_container_width=True, caption="Prediction mask")
            st.image(prob_img_display, use_container_width=True, caption="Probability map")

            if true_mask_img_display is not None:
                st.image(true_mask_img_display, use_container_width=True, caption="Ground truth mask from sample file")

        st.markdown("---")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Prediction Result", "Detected" if tumor_voxels > 0 else "Not detected")
        with m2:
            st.metric("Predicted Tumor Voxels", tumor_voxels)
        with m3:
            if true_voxels is not None:
                st.metric("Ground Truth Tumor Voxels", true_voxels)
            else:
                st.metric("Viewer Slice", slice_index)

        st.markdown("### Tumor Area Across Slices")
        area_chart = create_tumor_area_by_slice_chart(result["pred_mask"])
        if area_chart is not None:
            st.plotly_chart(area_chart, use_container_width=True)
            st.caption("The dashed line marks the slice with the largest predicted tumor area.")
        else:
            st.info("No predicted tumor area found across slices.")

        with st.expander("Advanced: experimental 3D tumor-only render", expanded=False):
            volume_fig = create_3d_tumor_volume_render(result["pred_mask"])
            if volume_fig is not None:
                st.plotly_chart(volume_fig, use_container_width=True)
                st.caption(
                    "This is an optional tumor-only render. The main viewer above is the recommended clinical-style view."
                )
            else:
                st.info("No 3D tumor volume could be displayed.")

# =========================
# Page 3: 2D Training Progress
# =========================

elif page == "2D Training Progress":
    render_workflow_buttons("2D")
    st.markdown("---")
    st.subheader("2D Training Progress")

    history = load_csv(BEST_HISTORY_PATH)
    best_metrics = load_json(BEST_METRICS_PATH)

    if history is not None:
        history_display = history.copy()
        history_display["train_loss"] = history_display["train_loss"].round(4)
        history_display["val_loss"] = history_display["val_loss"].round(4)
        history_display["val_dice"] = history_display["val_dice"].round(4)

        best_dice = history["val_dice"].max()
        best_epoch = history.loc[history["val_dice"].idxmax(), "epoch"]

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Best Validation Dice", f"{best_dice:.4f}")

        with c2:
            st.metric("Best Epoch", int(best_epoch))

        with c3:
            st.metric("Final Training Loss", f"{history['train_loss'].iloc[-1]:.4f}")

        st.markdown("---")
        st.markdown("### Best 2D Model Metrics")

        if best_metrics is not None:
            metrics_table = pd.DataFrame(
                [
                    {
                        "Metric": "Validation Loss",
                        "Value": round(best_metrics.get("Validation loss", 0), 4)
                    },
                    {
                        "Metric": "Dice Coefficient",
                        "Value": round(best_metrics.get("Dice coefficient", 0), 4)
                    },
                    {
                        "Metric": "Mean IoU",
                        "Value": round(best_metrics.get("Mean IoU", 0), 4)
                    },
                    {
                        "Metric": "Precision",
                        "Value": round(best_metrics.get("Precision", 0), 4)
                    },
                    {
                        "Metric": "Recall / Sensitivity",
                        "Value": round(best_metrics.get("Recall / Sensitivity", 0), 4)
                    },
                    {
                        "Metric": "Best Epoch",
                        "Value": int(best_metrics.get("Epoch", best_epoch))
                    },
                    {
                        "Metric": "Threshold",
                        "Value": best_metrics.get("Threshold", "N/A")
                    }
                ]
            )

            st.dataframe(
                metrics_table,
                use_container_width=True,
                hide_index=True
            )

            run_folder = best_metrics.get("Run folder", None)
            if run_folder is not None:
                st.caption(f"Best model source: {run_folder}")

        else:
            st.warning("Best metrics file not found: best_model/best_metrics.json")

        st.markdown("---")

        st.markdown("### 2D Loss During Training")
        loss_chart = history.set_index("epoch")[["train_loss", "val_loss"]]
        st.line_chart(loss_chart)

        st.markdown("### 2D Validation Dice Score")
        dice_chart = history.set_index("epoch")[["val_dice"]]
        st.line_chart(dice_chart)

    else:
        st.info(
            "No best 2D training history found yet. Expected file: "
            "best_model/best_history.csv"
        )


# =========================
# Page 4: 3D Training Progress
# =========================

elif page == "3D Training Progress":
    render_workflow_buttons("3D")
    st.markdown("---")
    st.subheader("3D Training Progress")

    latest_3d_metrics_path = find_latest_file("runs_3d/run_3d_*/metrics_3d.json")
    latest_3d_history_path = find_latest_file("runs_3d/run_3d_*/history_3d.csv")

    metrics_3d = load_json(BEST_METRICS_3D_PATH)
    metrics_source = BEST_METRICS_3D_PATH

    if metrics_3d is None and latest_3d_metrics_path is not None:
        metrics_3d = load_json(latest_3d_metrics_path)
        metrics_source = latest_3d_metrics_path

    history_3d = load_csv(BEST_HISTORY_3D_PATH)
    history_source = BEST_HISTORY_3D_PATH

    if history_3d is None and latest_3d_history_path is not None:
        history_3d = load_csv(latest_3d_history_path)
        history_source = latest_3d_history_path

    if metrics_3d is not None:
        st.caption(f"3D metrics source: {metrics_source}")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Best 3D Dice",
                f"{metrics_3d.get('Dice coefficient', 0):.4f}"
            )

        with c2:
            st.metric(
                "Best Epoch",
                int(metrics_3d.get("Epoch", 0))
            )

        with c3:
            st.metric(
                "Validation Loss",
                f"{metrics_3d.get('Validation loss', 0):.4f}"
            )

        st.markdown("---")
        st.markdown("### 3D Model Metrics")

        metrics_table_3d = pd.DataFrame(
            [
                {
                    "Metric": "Validation Loss",
                    "Value": round(metrics_3d.get("Validation loss", 0), 4)
                },
                {
                    "Metric": "Dice Coefficient",
                    "Value": round(metrics_3d.get("Dice coefficient", 0), 4)
                },
                {
                    "Metric": "Mean IoU",
                    "Value": round(metrics_3d.get("Mean IoU", 0), 4)
                },
                {
                    "Metric": "Precision",
                    "Value": round(metrics_3d.get("Precision", 0), 4)
                },
                {
                    "Metric": "Recall / Sensitivity",
                    "Value": round(metrics_3d.get("Recall / Sensitivity", 0), 4)
                },
                {
                    "Metric": "Threshold",
                    "Value": metrics_3d.get("Threshold", "N/A")
                },
                {
                    "Metric": "Depth",
                    "Value": metrics_3d.get("Depth", "N/A")
                },
                {
                    "Metric": "Image Size",
                    "Value": str(metrics_3d.get("Image size", "N/A"))
                },
            ]
        )

        st.dataframe(
            metrics_table_3d,
            use_container_width=True,
            hide_index=True
        )

        run_folder = metrics_3d.get("Run folder", None)
        if run_folder is not None:
            st.caption(f"Best model source: {run_folder}")

    else:
        st.warning(
            "No 3D metrics found yet. Expected either "
            "best_model_3d/best_metrics_3d.json or latest run metrics."
        )

    if history_3d is not None:
        st.markdown("---")
        st.caption(f"3D history source: {history_source}")

        st.markdown("### 3D Loss During Training")
        st.line_chart(history_3d.set_index("epoch")[["train_loss", "val_loss"]])

        st.markdown("### 3D Validation Dice")
        st.line_chart(history_3d.set_index("epoch")[["val_dice"]])

    else:
        st.info(
            "No best 3D history file found yet. This is okay if your quick run "
            "did not update best_model_3d."
        )

# Sidebar export buttons are rendered after page content so they update immediately
# after a new upload/prediction and never duplicate.
if page == "2D MRI Analysis":
    st.sidebar.markdown("---")
    render_export_2d_sidebar()
elif page == "3D MRI Analysis":
    st.sidebar.markdown("---")
    render_export_3d_sidebar()

# Bottom navigation
if page != "Home":
    render_home_button_bottom()