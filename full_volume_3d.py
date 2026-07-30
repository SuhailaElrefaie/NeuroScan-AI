"""Full-volume 3D inference and visualisation helpers for NeuroScan AI.

The trained network still receives 32x160x160 windows. A complete patient
volume can contain about 155 slices, so this module runs overlapping windows
and merges their predictions back into the original full depth and resolution.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F

from models.unet3d import UNet3D

MODEL_DEPTH = 32
MODEL_HEIGHT = 160
MODEL_WIDTH = 160
WINDOW_STRIDE = 16

# The AWSAF BraTS H5 dataset stores its four image channels in this order.
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


def _canonicalise_mask(mask: Optional[np.ndarray], target_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None

    mask = np.asarray(mask)
    mask = np.squeeze(mask)
    if mask.shape != target_shape:
        return None
    return (mask > 0).astype(np.uint8)


def _window_starts(depth: int, window: int = MODEL_DEPTH, stride: int = WINDOW_STRIDE) -> list[int]:
    if depth <= window:
        return [0]
    starts = list(range(0, depth - window + 1, stride))
    last = depth - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def _normalise_like_training(window: np.ndarray) -> np.ndarray:
    """Match the v2 training pipeline: per-modality z-score on non-zero voxels."""
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
    """Convert [4,D,H,W] to the network's [1,4,32,160,160] input."""
    original_depth = window.shape[1]
    if original_depth < MODEL_DEPTH:
        pad = MODEL_DEPTH - original_depth
        window = np.pad(window, ((0, 0), (0, pad), (0, 0), (0, 0)), mode="edge")

    window = _normalise_like_training(window)
    tensor = torch.from_numpy(window).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(MODEL_DEPTH, MODEL_HEIGHT, MODEL_WIDTH),
        mode="trilinear",
        align_corners=False,
    )
    return tensor


def _restore_probability(probability: torch.Tensor, height: int, width: int) -> np.ndarray:
    probability = probability.unsqueeze(0).unsqueeze(0)
    restored = F.interpolate(
        probability,
        size=(MODEL_DEPTH, height, width),
        mode="trilinear",
        align_corners=False,
    )
    return restored.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def predict_full_volume_npz(
    uploaded_file: Union[str, Path, BinaryIO],
    model_path: str,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Predict either an old 32-slice sample or a complete patient volume."""
    with np.load(uploaded_file, allow_pickle=False) as data:
        image_key = "image" if "image" in data.files else data.files[0]
        image = _canonicalise_image(data[image_key])
        raw_mask = data["mask"] if "mask" in data.files else None
        volume_id = str(data["volume_id"].item()) if "volume_id" in data.files else "uploaded"

    _, depth, height, width = image.shape
    true_mask = _canonicalise_mask(raw_mask, (depth, height, width))

    model = UNet3D(in_channels=4, out_channels=1).to(device)
    try:
        state_dict = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)
    model.eval()

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
        "num_volumes": 1,
        "depth": depth,
        "volume_id": volume_id,
        "source_type": "full_volume_upload" if depth > MODEL_DEPTH else "upload",
        "model_path": model_path,
        "window_count": len(starts),
        "original_shape": tuple(image.shape),
    }


def _downsample(volume: np.ndarray, max_depth: int = 64, max_side: int = 72) -> np.ndarray:
    d, h, w = volume.shape
    sd = max(1, int(np.ceil(d / max_depth)))
    sh = max(1, int(np.ceil(h / max_side)))
    sw = max(1, int(np.ceil(w / max_side)))
    return volume[::sd, ::sh, ::sw]


def _display_normalise(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.zeros_like(volume)
    low, high = np.percentile(finite, (5, 99))
    if high - low < 1e-8:
        return np.zeros_like(volume)
    return np.clip((volume - low) / (high - low), 0, 1)


def build_tumor_figure(result: Dict[str, object], modality_index: int = 0) -> go.Figure:
    """Create a lightweight interactive full-volume brain/tumour rendering."""
    image = np.asarray(result["image"])[int(modality_index)]
    mask = np.asarray(result["pred_mask"], dtype=np.float32)

    image_small = _display_normalise(_downsample(image))
    mask_small = _downsample(mask)
    d, h, w = image_small.shape
    z, y, x = np.mgrid[0:d, 0:h, 0:w]

    figure = go.Figure()
    figure.add_trace(
        go.Isosurface(
            x=x.flatten(), y=y.flatten(), z=z.flatten(),
            value=image_small.flatten(),
            isomin=0.20, isomax=0.72,
            opacity=0.10,
            surface_count=3,
            colorscale="Greys",
            showscale=False,
            name="MRI anatomy",
            caps=dict(x_show=False, y_show=False, z_show=False),
            hoverinfo="skip",
        )
    )

    if np.any(mask_small > 0):
        figure.add_trace(
            go.Isosurface(
                x=x.flatten(), y=y.flatten(), z=z.flatten(),
                value=mask_small.flatten(),
                isomin=0.5, isomax=1.0,
                opacity=0.82,
                surface_count=1,
                colorscale=[[0.0, "rgb(255,70,70)"], [1.0, "rgb(255,70,70)"]],
                showscale=False,
                name="Predicted tumour",
                caps=dict(x_show=False, y_show=False, z_show=False),
            )
        )

    figure.update_layout(
        height=620,
        margin=dict(l=0, r=0, t=45, b=0),
        title="Interactive full-volume MRI and predicted tumour",
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            aspectmode="data",
            camera=dict(eye=dict(x=1.55, y=1.55, z=1.15)),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    )
    return figure
