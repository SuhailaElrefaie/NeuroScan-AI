"""Full-volume 3D inference and visualisation helpers for NeuroScan AI.

This version keeps the complete-volume sliding-window inference used by the
project, supports the newer attention U-Net checkpoint, and uses a hybrid 3D
viewer:
  * the MRI anatomy is shown as a translucent grey volume;
  * the predicted tumour is extracted as a smooth triangular surface with
    marching cubes.

The renderer is intentionally implemented inside the existing Streamlit /
Plotly pipeline. It does not copy the linked neuro-voxel implementation.
"""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Dict, Optional, Tuple, Union

import numpy as np


def _display_normalise_slice(slice_2d):
    """Normalize one MRI slice to uint8 for display."""
    array = np.nan_to_num(
        np.asarray(slice_2d, dtype=np.float32)
    )

    low, high = np.percentile(
        array,
        (1.0, 99.0)
    )

    if high <= low + 1e-8:
        low = float(array.min())
        high = float(array.max())

    if high <= low + 1e-8:
        return np.zeros_like(
            array,
            dtype=np.uint8
        )

    array = np.clip(
        array,
        low,
        high
    )

    array = (
        array - low
    ) / (
        high - low + 1e-8
    )

    return (
        array * 255
    ).astype(np.uint8)

import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from skimage.measure import marching_cubes
from PIL import Image
from plotly.subplots import make_subplots
from models.unet3d import UNet3D
from models.attention_unet3d import AttentionUNet3D
from gradcam import GradCAM, create_gradcam_overlay
from gradcam import GradCAM

MODEL_DEPTH = 32
MODEL_HEIGHT = 160
MODEL_WIDTH = 160
WINDOW_STRIDE = 16

MODALITY_NAMES = {
    0: "FLAIR",
    1: "T1",
    2: "T1CE",
    3: "T2",
}

def generate_3d_gradcam(
    result,
    model_path,
    device,
    selected_slice,
    modality_index=0,
    heatmap_alpha=0.42,
):
    """
    Generate colored Grad-CAM for the 3D model around the selected MRI slice.
    """

    image = _canonicalise_image(
        np.asarray(result["image"])
    )

    _, depth, height, width = image.shape

    selected_slice = int(
        np.clip(
            selected_slice,
            0,
            depth - 1
        )
    )

    start = selected_slice - MODEL_DEPTH // 2

    start = max(
        0,
        min(
            start,
            max(
                depth - MODEL_DEPTH,
                0
            )
        )
    )

    end = min(
        start + MODEL_DEPTH,
        depth
    )

    valid_depth = end - start

    window = image[:, start:end]

    network_input = _prepare_window(
        window
    ).to(device)

    model = _load_model(
        model_path,
        device
    )

    if isinstance(model, AttentionUNet3D):
        target_layer = model.enc4

    else:
        target_layer = model.enc3

    gradcam = GradCAM(
        model,
        target_layer
    )

    try:
        cam = gradcam.generate(
            network_input
        )

    finally:
        gradcam.remove_hooks()

    cam_tensor = (
        torch
        .from_numpy(cam)
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
    )

    restored = F.interpolate(
        cam_tensor,
        size=(
            MODEL_DEPTH,
            height,
            width
        ),
        mode="trilinear",
        align_corners=False
    )

    restored = (
        restored
        .squeeze(0)
        .squeeze(0)
        .numpy()
        .astype(np.float32)
    )

    restored = restored[:valid_depth]

    local_slice = int(
        np.clip(
            selected_slice - start,
            0,
            restored.shape[0] - 1
        )
    )

    cam_slice = restored[
        local_slice
    ]

    mri_slice = _display_normalise_slice(
        image[
            int(modality_index),
            selected_slice
        ]
    )

    overlay_pil, heatmap_pil = (
        create_gradcam_overlay(
            Image.fromarray(mri_slice),
            cam_slice,
            heatmap_alpha=heatmap_alpha
        )
    )

    return {
        "cam_volume": restored,
        "cam_slice": cam_slice,
        "overlay_rgb": np.asarray(
            overlay_pil
        ),
        "heatmap_rgb": np.asarray(
            heatmap_pil
        ),
        "window_start": int(start),
        "window_end": int(end),
        "architecture": type(model).__name__,
    }

def create_3d_gradcam_overlay(
    mri_slice,
    cam_slice,
    alpha=0.40
):
    import cv2

    mri = _display_normalise_slice(
        mri_slice
    )

    mri_rgb = np.stack(
        [mri, mri, mri],
        axis=-1
    )

    mri_rgb = (
        mri_rgb * 255
    ).astype(np.uint8)

    heatmap = (
        np.clip(
            cam_slice,
            0,
            1
        ) * 255
    ).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        mri_rgb,
        1.0 - alpha,
        heatmap,
        alpha,
        0
    )

    return overlay, heatmap

def build_orthogonal_mri_figure(
    result,
    modality_index=0,
    axial_index=None,
    coronal_index=None,
    sagittal_index=None,
    show_prediction=True,
):
    """
    MRI-viewer style axial/coronal/sagittal display.

    Red pixels show the predicted tumour.
    """

    image = np.asarray(
        result["image"],
        dtype=np.float32
    )[int(modality_index)]

    prediction = np.asarray(
        result["pred_mask"],
        dtype=np.uint8
    )

    depth, height, width = image.shape

    if axial_index is None:
        axial_index = depth // 2

    if coronal_index is None:
        coronal_index = height // 2

    if sagittal_index is None:
        sagittal_index = width // 2

    axial_index = int(
        np.clip(
            axial_index,
            0,
            depth - 1
        )
    )

    coronal_index = int(
        np.clip(
            coronal_index,
            0,
            height - 1
        )
    )

    sagittal_index = int(
        np.clip(
            sagittal_index,
            0,
            width - 1
        )
    )

    axial = _display_normalise_slice(
        image[axial_index]
    )

    coronal = _display_normalise_slice(
        image[:, coronal_index, :]
    )

    sagittal = _display_normalise_slice(
        image[:, :, sagittal_index]
    )

    axial_mask = prediction[
        axial_index
    ]

    coronal_mask = prediction[
        :,
        coronal_index,
        :
    ]

    sagittal_mask = prediction[
        :,
        :,
        sagittal_index
    ]

    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            f"Axial — slice {axial_index}",
            f"Coronal — slice {coronal_index}",
            f"Sagittal — slice {sagittal_index}",
        ]
    )

    slices = [
        (axial, axial_mask),
        (coronal, coronal_mask),
        (sagittal, sagittal_mask),
    ]

    for column, (
        mri_slice,
        mask_slice
    ) in enumerate(
        slices,
        start=1
    ):

        figure.add_trace(
            go.Heatmap(
                z=mri_slice,
                colorscale="gray",
                showscale=False,
                hoverinfo="skip"
            ),
            row=1,
            col=column
        )

        if (
            show_prediction
            and np.any(mask_slice)
        ):

            overlay = np.where(
                mask_slice > 0,
                1.0,
                np.nan
            )

            figure.add_trace(
                go.Heatmap(
                    z=overlay,
                    colorscale=[
                        [0, "rgba(255,0,0,0.65)"],
                        [1, "rgba(255,0,0,0.65)"]
                    ],
                    zmin=0,
                    zmax=1,
                    showscale=False,
                    hoverinfo="skip"
                ),
                row=1,
                col=column
            )

    figure.update_yaxes(
        autorange="reversed",
        scaleanchor="x"
    )

    figure.update_layout(
        height=420,
        margin=dict(
            l=10,
            r=10,
            t=55,
            b=10
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="black"
    )

    return figure

def _canonicalise_image(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array)

    if array.ndim == 4 and array.shape[0] == 4:
        image = array
    elif array.ndim == 4 and array.shape[-1] == 4:
        image = np.moveaxis(array, -1, 0)
    elif array.ndim == 3:
        image = np.stack([array] * 4, axis=0)
    else:
        raise ValueError(
            "Expected [4,D,H,W], [D,H,W,4], or [D,H,W] for the uploaded MRI volume."
        )

    if image.shape[1] < 2:
        raise ValueError("The uploaded file does not contain a usable 3D depth.")

    return image.astype(np.float32, copy=False)


def _canonicalise_mask(
    mask: Optional[np.ndarray],
    target_shape: Tuple[int, int, int],
) -> Optional[np.ndarray]:
    if mask is None:
        return None

    mask = np.asarray(mask)
    mask = np.squeeze(mask)
    if mask.shape != target_shape:
        return None
    return (mask > 0).astype(np.uint8)


def _window_starts(
    depth: int,
    window: int = MODEL_DEPTH,
    stride: int = WINDOW_STRIDE,
) -> list[int]:
    if depth <= window:
        return [0]
    starts = list(range(0, depth - window + 1, stride))
    last = depth - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def _normalise_like_training(window: np.ndarray) -> np.ndarray:
    """Match training: per-modality z-score over non-zero voxels."""
    window = np.nan_to_num(np.asarray(window, dtype=np.float32), copy=False)
    result = np.zeros_like(window, dtype=np.float32)

    for channel in range(window.shape[0]):
        modality = window[channel]
        brain = modality != 0
        if not np.any(brain):
            continue

        values = modality[brain]
        mean = float(values.mean())
        std = float(values.std())
        if std < 1e-6:
            std = 1.0

        result[channel, brain] = (values - mean) / std

    return result


def _prepare_window(window: np.ndarray) -> torch.Tensor:
    """Convert [4,D,H,W] to [1,4,32,160,160]."""
    original_depth = window.shape[1]
    if original_depth < MODEL_DEPTH:
        pad = MODEL_DEPTH - original_depth
        window = np.pad(
            window,
            ((0, 0), (0, pad), (0, 0), (0, 0)),
            mode="edge",
        )

    window = _normalise_like_training(window)
    tensor = torch.from_numpy(window).unsqueeze(0)
    return F.interpolate(
        tensor,
        size=(MODEL_DEPTH, MODEL_HEIGHT, MODEL_WIDTH),
        mode="trilinear",
        align_corners=False,
    )


def _restore_probability(
    probability: torch.Tensor,
    height: int,
    width: int,
) -> np.ndarray:
    probability = probability.unsqueeze(0).unsqueeze(0)
    restored = F.interpolate(
        probability,
        size=(MODEL_DEPTH, height, width),
        mode="trilinear",
        align_corners=False,
    )
    return restored.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def _load_model(model_path: Union[str, Path], device: torch.device) -> torch.nn.Module:
    """Load either a legacy raw state_dict or the newer wrapped attention checkpoint.

    Attention training saves metadata plus the actual weights under
    ``model_state_dict``. Older project checkpoints may be a raw state_dict.
    This loader supports both formats and builds the architecture from checkpoint
    metadata when it is available.
    """
    try:
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    is_wrapped_checkpoint = (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    )

    if is_wrapped_checkpoint:
        state_dict = checkpoint["model_state_dict"]
        architecture = str(checkpoint.get("architecture", "")).lower()
        model_config = checkpoint.get("model_config", {}) or {}
    else:
        state_dict = checkpoint
        architecture = ""
        model_config = {}

    name = Path(model_path).name.lower()
    use_attention = architecture == "attention_unet3d" or "attention" in name

    if use_attention:
        model = AttentionUNet3D(
            in_channels=int(model_config.get("in_channels", 4)),
            out_channels=int(model_config.get("out_channels", 1)),
            base_channels=int(model_config.get("base_channels", 16)),
            num_heads=int(model_config.get("num_heads", 8)),
        )
    else:
        model = UNet3D(in_channels=4, out_channels=1)

    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_full_volume_npz(
    uploaded_file: Union[str, Path, BinaryIO],
    model_path: str,
    device: torch.device,
    threshold: float = 0.55,
) -> Dict[str, object]:
    """Predict a complete MRI volume using overlapping 32-slice windows."""
    with np.load(uploaded_file, allow_pickle=False) as data:
        image_key = "image" if "image" in data.files else data.files[0]
        image = _canonicalise_image(data[image_key])
        raw_mask = data["mask"] if "mask" in data.files else None
        volume_id = (
            str(data["volume_id"].item())
            if "volume_id" in data.files
            else "uploaded"
        )

    _, depth, height, width = image.shape
    true_mask = _canonicalise_mask(raw_mask, (depth, height, width))

    model = _load_model(model_path, device)

    probability_sum = np.zeros((depth, height, width), dtype=np.float32)
    probability_count = np.zeros((depth, height, width), dtype=np.float32)

    starts = _window_starts(depth)
    with torch.no_grad():
        for start in starts:
            end = min(start + MODEL_DEPTH, depth)
            valid_depth = end - start
            window = image[:, start:end]
            network_input = _prepare_window(window).to(device)
            logits = model(network_input)
            probability = torch.sigmoid(logits).squeeze(0).squeeze(0)
            restored = _restore_probability(probability, height, width)[:valid_depth]
            probability_sum[start:end] += restored
            probability_count[start:end] += 1.0

    probability_count[probability_count == 0] = 1.0
    probability = probability_sum / probability_count
    pred_mask = (probability >= float(threshold)).astype(np.uint8)

    return {
        "image": image,
        "true_mask": true_mask,
        "probability": probability,
        "pred_mask": pred_mask,
        "threshold": float(threshold),
        "num_volumes": 1,
        "depth": depth,
        "volume_id": volume_id,
        "source_type": "full_volume_upload" if depth > MODEL_DEPTH else "upload",
        "model_path": model_path,
        "window_count": len(starts),
        "original_shape": tuple(image.shape),
        "architecture": type(model).__name__,
    }



from scipy import ndimage as ndi


def _estimate_background(modality: np.ndarray) -> float:
    """Estimate the scanner/background value from the outer shell.

    BraTS sample NPZ files in this project are already normalized and their
    background is a constant negative value, not zero. Using ``volume != 0``
    therefore incorrectly marks the whole cuboid as anatomy.
    """
    v = np.asarray(modality, dtype=np.float32)
    n = 3
    shell = np.concatenate([
        v[:n].ravel(), v[-n:].ravel(),
        v[:, :n].ravel(), v[:, -n:].ravel(),
        v[:, :, :n].ravel(), v[:, :, -n:].ravel(),
    ])
    return float(np.median(shell))


def _brain_mask_from_mri(image_4d: np.ndarray) -> np.ndarray:
    """Recover the real BraTS brain support from all four modalities.

    A voxel is anatomy when it differs from the modality-specific background.
    All four modalities in the exported project samples share the same support,
    so a majority vote is used as a guard against a noisy channel.
    """
    image = np.asarray(image_4d, dtype=np.float32)
    votes = np.zeros(image.shape[1:], dtype=np.uint8)

    for c in range(image.shape[0]):
        m = image[c]
        bg = _estimate_background(m)
        dynamic = max(float(m.max() - m.min()), 1.0)
        tolerance = max(1e-5, dynamic * 1e-5)
        votes += (np.abs(m - bg) > tolerance).astype(np.uint8)

    required = max(1, int(np.ceil(image.shape[0] / 2)))
    mask = votes >= required

    # Keep the main 3-D anatomical object, close small cracks and fill cavities
    # so marching cubes extracts the external brain boundary rather than noise.
    labels, count = ndi.label(mask)
    if count > 1:
        sizes = ndi.sum(mask, labels, index=np.arange(1, count + 1))
        keep = int(np.argmax(sizes)) + 1
        mask = labels == keep

    structure = ndi.generate_binary_structure(3, 1)
    mask = ndi.binary_closing(mask, structure=structure, iterations=1)
    mask = ndi.binary_fill_holes(mask)
    return mask.astype(bool)


def _brain_bbox(mask: np.ndarray, pad: int = 5) -> Tuple[slice, slice, slice]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError('No brain anatomy could be extracted from the MRI volume.')
    lo = np.maximum(coords.min(axis=0) - pad, 0)
    hi = np.minimum(coords.max(axis=0) + pad + 1, np.array(mask.shape))
    return tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))


def _resize_display_volume(volume: np.ndarray, max_shape=(105, 128, 128), order: int = 1) -> np.ndarray:
    """Resize a cropped display volume while preserving its aspect ratio."""
    shape = np.array(volume.shape, dtype=float)
    target = np.array(max_shape, dtype=float)
    scale = min(1.0, float(np.min(target / shape)))
    if scale >= 0.999:
        return np.asarray(volume)
    return ndi.zoom(np.asarray(volume), zoom=(scale, scale, scale), order=order)


def _normalise_mri_for_surface(modality: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """Map real in-brain MRI intensities to 0..1 for grey vertex shading."""
    m = np.asarray(modality, dtype=np.float32)
    vals = m[brain_mask]
    if vals.size == 0:
        return np.zeros_like(m)
    lo, hi = np.percentile(vals, (3.0, 99.0))
    if hi <= lo + 1e-8:
        return np.zeros_like(m)
    out = np.clip((m - lo) / (hi - lo), 0.0, 1.0)
    out[~brain_mask] = 0.0
    return out.astype(np.float32)


def _surface_from_binary(mask: np.ndarray, sigma: float = 1.15):
    """Create a smooth closed triangular surface from a binary 3-D mask."""
    field = ndi.gaussian_filter(mask.astype(np.float32), sigma=sigma)
    # Zero padding guarantees a closed surface even when anatomy touches the
    # first/last acquired MRI slice.
    field = np.pad(field, 2, mode='constant', constant_values=0.0)
    if not (field.min() < 0.5 < field.max()):
        raise ValueError('The extracted anatomy does not contain a usable surface.')
    vertices, faces, normals, values = marching_cubes(
        field, level=0.5, allow_degenerate=False
    )
    vertices -= 2.0
    return vertices, faces


def _sample_vertex_intensity(intensity: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    z = np.clip(np.rint(vertices[:, 0]).astype(int), 0, intensity.shape[0] - 1)
    y = np.clip(np.rint(vertices[:, 1]).astype(int), 0, intensity.shape[1] - 1)
    x = np.clip(np.rint(vertices[:, 2]).astype(int), 0, intensity.shape[2] - 1)
    return intensity[z, y, x]


def _add_brain_surface(
    figure: go.Figure,
    image_4d: np.ndarray,
    modality_index: int,
):
    brain_full = _brain_mask_from_mri(image_4d)
    bbox = _brain_bbox(brain_full, pad=4)

    brain = brain_full[bbox]
    modality = np.asarray(image_4d[int(modality_index)], dtype=np.float32)[bbox]
    intensity = _normalise_mri_for_surface(modality, brain)

    brain_small = _resize_display_volume(brain.astype(np.float32), order=1)
    brain_small = brain_small > 0.45
    intensity_small = _resize_display_volume(intensity, order=1)

    vertices, faces = _surface_from_binary(brain_small, sigma=1.05)
    vertex_values = _sample_vertex_intensity(intensity_small, vertices)

    figure.add_trace(go.Mesh3d(
        x=vertices[:, 2], y=vertices[:, 1], z=vertices[:, 0],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=vertex_values,
        intensitymode='vertex',
        colorscale=[
            [0.0, 'rgb(72,72,76)'],
            [0.35, 'rgb(112,112,118)'],
            [0.7, 'rgb(165,165,170)'],
            [1.0, 'rgb(215,215,218)'],
        ],
        cmin=0.0, cmax=1.0,
        showscale=False,
        opacity=0.24,
        name=f'MRI brain surface ({MODALITY_NAMES.get(int(modality_index), "MRI")})',
        flatshading=False,
        lighting=dict(ambient=0.42, diffuse=0.72, specular=0.18, roughness=0.68, fresnel=0.08),
        lightposition=dict(x=180, y=220, z=260),
        hoverinfo='skip',
    ))

    return bbox, brain.shape, brain_small.shape


def _crop_and_resize_probability(
    probability: np.ndarray,
    bbox: Tuple[slice, slice, slice],
    final_shape: Tuple[int, int, int],
) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float32)[bbox]
    zoom = np.array(final_shape, dtype=float) / np.array(p.shape, dtype=float)
    return ndi.zoom(p, zoom=zoom, order=1)


def _add_tumour_surface(
    figure: go.Figure,
    probability: np.ndarray,
    bbox: Tuple[slice, slice, slice],
    target_shape: Tuple[int, int, int],
    threshold: float,
) -> None:
    p = _crop_and_resize_probability(probability, bbox, target_shape)
    if not (float(p.min()) < threshold < float(p.max())):
        return

    # A small amount of display-only smoothing removes stair-step voxel edges.
    p = ndi.gaussian_filter(p, sigma=0.55)
    if not (float(p.min()) < threshold < float(p.max())):
        return

    padded = np.pad(p, 2, mode='constant', constant_values=0.0)
    vertices, faces, _, _ = marching_cubes(
        padded, level=float(threshold), allow_degenerate=False
    )
    vertices -= 2.0

    figure.add_trace(go.Mesh3d(
        x=vertices[:, 2], y=vertices[:, 1], z=vertices[:, 0],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='rgb(245,45,58)',
        opacity=0.97,
        name='Predicted tumour surface',
        flatshading=False,
        lighting=dict(ambient=0.35, diffuse=0.78, specular=0.30, roughness=0.48, fresnel=0.10),
        lightposition=dict(x=180, y=220, z=260),
        hoverinfo='skip',
    ))


def build_tumor_figure(
    result: Dict[str, object],
    modality_index: int = 0,
) -> go.Figure:
    """True MRI-derived brain surface + predicted tumour surface.

    No point cloud, no Plotly Volume cuboid and no slice-plane cross. The brain
    boundary is recovered from the actual BraTS background value stored in the
    uploaded NPZ and rendered as a smooth triangular mesh.
    """
    image_4d = _canonicalise_image(np.asarray(result['image']))
    probability = np.asarray(result['probability'], dtype=np.float32)
    threshold = float(result.get('threshold', 0.55))

    figure = go.Figure()
    bbox, _crop_shape, display_shape = _add_brain_surface(
        figure, image_4d, int(modality_index)
    )
    _add_tumour_surface(
        figure, probability, bbox, display_shape, threshold
    )

    figure.update_layout(
        height=680,
        margin=dict(l=0, r=0, t=46, b=0),
        title='MRI-derived brain surface with predicted tumour',
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            aspectmode='data',
            camera=dict(eye=dict(x=1.50, y=1.35, z=1.05)),
            bgcolor='rgb(13,15,20)',
        ),
        paper_bgcolor='rgb(13,15,20)',
        plot_bgcolor='rgb(13,15,20)',
        font=dict(color='white'),
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
    )
    return figure
