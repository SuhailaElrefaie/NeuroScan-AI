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
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from skimage.measure import marching_cubes

from models.unet3d import UNet3D
from models.attention_unet3d import AttentionUNet3D

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


def _downsample(
    volume: np.ndarray,
    max_depth: int = 52,
    max_side: int = 64,
) -> np.ndarray:
    """Fast display-only downsampling. Does not affect model inference."""
    d, h, w = volume.shape
    sd = max(1, int(np.ceil(d / max_depth)))
    sh = max(1, int(np.ceil(h / max_side)))
    sw = max(1, int(np.ceil(w / max_side)))
    return volume[::sd, ::sh, ::sw]


def _display_normalise(volume: np.ndarray) -> np.ndarray:
    """Contrast-normalise MRI anatomy while keeping true background at zero."""
    volume = np.nan_to_num(np.asarray(volume, dtype=np.float32))
    nonzero = volume[np.abs(volume) > 1e-8]
    if nonzero.size == 0:
        return np.zeros_like(volume)

    low, high = np.percentile(nonzero, (3.0, 99.5))
    if high - low < 1e-8:
        return np.zeros_like(volume)

    normalised = np.clip((volume - low) / (high - low), 0.0, 1.0)
    normalised[np.abs(volume) <= 1e-8] = 0.0
    return normalised.astype(np.float32)


def _resize_probability_for_mesh(
    probability: np.ndarray,
    target_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Trilinear display-only resize gives marching cubes a smoother field."""
    tensor = torch.from_numpy(
        np.asarray(probability, dtype=np.float32)
    ).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        tensor,
        size=target_shape,
        mode="trilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0).numpy()


def _add_tumour_surface(
    figure: go.Figure,
    probability: np.ndarray,
    target_shape: Tuple[int, int, int],
    threshold: float,
) -> None:
    """Extract the predicted tumour boundary as a triangular surface mesh."""
    probability_small = _resize_probability_for_mesh(probability, target_shape)

    minimum = float(probability_small.min())
    maximum = float(probability_small.max())
    level = float(threshold)
    if not (minimum < level < maximum):
        return

    vertices, faces, _, _ = marching_cubes(
        probability_small,
        level=level,
        allow_degenerate=False,
    )

    # skimage vertices are returned as (z, y, x).
    z = vertices[:, 0]
    y = vertices[:, 1]
    x = vertices[:, 2]

    figure.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="rgb(235,55,65)",
            opacity=0.95,
            name="Predicted tumour surface",
            flatshading=False,
            lighting=dict(
                ambient=0.35,
                diffuse=0.75,
                specular=0.25,
                roughness=0.55,
                fresnel=0.10,
            ),
            lightposition=dict(x=120, y=160, z=220),
            hoverinfo="skip",
        )
    )


def build_tumor_figure(
    result: Dict[str, object],
    modality_index: int = 0,
) -> go.Figure:
    """Hybrid view: grey MRI volume + smooth predicted tumour surface."""
    image = np.asarray(result["image"])[int(modality_index)]
    probability = np.asarray(result["probability"], dtype=np.float32)
    threshold = float(result.get("threshold", 0.55))

    image_small = _display_normalise(_downsample(image))
    d, h, w = image_small.shape
    z, y, x = np.mgrid[0:d, 0:h, 0:w]

    figure = go.Figure()

    # Grey MRI anatomy. Multiple translucent intensity surfaces preserve the
    # voxel/volume appearance while giving a recognisable full brain context.
    figure.add_trace(
        go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=image_small.flatten(),
            isomin=0.08,
            isomax=0.92,
            opacity=0.075,
            surface_count=12,
            colorscale="Greys",
            reversescale=False,
            showscale=False,
            name=f"MRI anatomy ({MODALITY_NAMES.get(int(modality_index), 'MRI')})",
            caps=dict(x_show=False, y_show=False, z_show=False),
            hoverinfo="skip",
        )
    )

    _add_tumour_surface(
        figure=figure,
        probability=probability,
        target_shape=(d, h, w),
        threshold=threshold,
    )

    figure.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=48, b=0),
        title="3D MRI volume with predicted tumour surface",
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            aspectmode="data",
            camera=dict(eye=dict(x=1.55, y=1.55, z=1.10)),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            y=1.02,
            x=0.5,
            xanchor="center",
        ),
    )
    return figure
