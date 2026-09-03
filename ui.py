import io
import os
import json
import glob

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
from PIL import Image


from predict import predict_with_gradcam


from full_volume_3d import (
    MODALITY_NAMES,
    build_tumor_figure,
    build_orthogonal_mri_figure,
    generate_3d_gradcam,
    predict_full_volume_npz,
    _canonicalise_image,
    _add_brain_surface,
    _add_tumour_surface,
    _crop_and_resize_probability,
)

st.set_page_config(
    page_title="NeuroScan AI | Tumor Segmentation",
    layout="wide"
)

BASELINE_2D_METRICS = "best_model/best_metrics.json"
BASELINE_2D_HISTORY = "best_model/best_history.csv"
BASELINE_2D_MODEL = "best_model/best_unet.pth"

ATTENTION_2D_METRICS = "best_model/best_attention_metrics_2d.json"
ATTENTION_2D_HISTORY = "best_model/best_attention_history_2d.csv"
ATTENTION_2D_MODEL = "best_model/best_attention_unet2d.pth"

BASELINE_3D_METRICS = "best_model_3d/best_metrics_3d.json"
BASELINE_3D_HISTORY = "best_model_3d/best_history_3d.csv"
BASELINE_3D_MODEL = "best_model_3d/best_unet3d.pth"

ATTENTION_3D_METRICS = "best_model_3d/best_attention_metrics_3d.json"
ATTENTION_3D_HISTORY = "best_model_3d/best_attention_history_3d.csv"
ATTENTION_3D_MODEL = "best_model_3d/best_attention_unet3d.pth"

BEST_METRICS_PATH = ATTENTION_2D_METRICS
BEST_HISTORY_PATH = ATTENTION_2D_HISTORY

BEST_METRICS_3D_PATH = ATTENTION_3D_METRICS
BEST_HISTORY_3D_PATH = ATTENTION_3D_HISTORY
BEST_MODEL_3D_PATH = ATTENTION_3D_MODEL



SAMPLE_2D_DIR = "sample_data/2d"
SAMPLE_3D_DIR = "sample_data/3d"


def load_json(path):
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return json.load(f)


def load_csv(path):
    if not os.path.exists(path):
        return None

    return pd.read_csv(path)


def metric_value(metrics, key):
    if not metrics:
        return None

    value = metrics.get(key)
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_metric(metrics, *keys):
    for key in keys:
        value = metric_value(metrics, key)
        if value is not None:
            return value
    return None


def render_experiment_card(title, architecture, metrics, model_path=None):
    st.markdown(f"#### {title}")
    st.caption(architecture)

    if not metrics:
        st.info("Experiment results are not available yet.")
        if model_path and os.path.exists(model_path):
            st.caption("Model checkpoint exists, but its metrics file is not available yet.")
        return

    dice = _first_metric(metrics, "Dice coefficient", "Best Dice", "best_dice", "Dice")
    iou = _first_metric(metrics, "Mean IoU", "IoU", "iou")
    precision = _first_metric(metrics, "Precision", "precision")
    recall = _first_metric(metrics, "Recall / Sensitivity", "Recall", "recall")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Dice", f"{dice:.4f}" if dice is not None else "—")
        st.metric("Precision", f"{precision:.4f}" if precision is not None else "—")

    with c2:
        st.metric("IoU", f"{iou:.4f}" if iou is not None else "—")
        st.metric("Recall", f"{recall:.4f}" if recall is not None else "—")

    threshold = metrics.get("Threshold", None)
    if threshold is not None:
        st.caption(f"Evaluation threshold: {threshold}")

    if model_path:
        if os.path.exists(model_path):
            st.caption(f"Checkpoint: `{model_path}`")
        else:
            st.caption(f"Expected checkpoint: `{model_path}`")


def render_experiment_comparison(workflow):
    st.markdown("---")
    st.markdown("### Model Experiments")
    st.caption(
        "Compare the original U-Net baseline with the multi-head-attention "
        "experiment using saved validation results."
    )

    if workflow == "2D":
        baseline_metrics = load_json(BASELINE_2D_METRICS)
        attention_metrics = load_json(ATTENTION_2D_METRICS)

        baseline_title = "Experiment A — Baseline"
        baseline_architecture = "2D U-Net"
        baseline_model = BASELINE_2D_MODEL

        attention_title = "Experiment B — Multi-Head Attention"
        attention_architecture = "2D U-Net + multi-head self-attention at the bottleneck"
        attention_model = ATTENTION_2D_MODEL

    elif workflow == "3D":
        baseline_metrics = load_json(BASELINE_3D_METRICS)
        attention_metrics = load_json(ATTENTION_3D_METRICS)

        baseline_title = "Experiment A — Baseline"
        baseline_architecture = "3D U-Net"
        baseline_model = BASELINE_3D_MODEL

        attention_title = "Experiment B — Multi-Head Attention"
        attention_architecture = "3D U-Net + multi-head self-attention"
        attention_model = ATTENTION_3D_MODEL

    else:
        raise ValueError(f"Unknown workflow: {workflow}")

    left, right = st.columns(2)

    with left:
        render_experiment_card(
            baseline_title,
            baseline_architecture,
            baseline_metrics,
            model_path=baseline_model
        )

    with right:
        render_experiment_card(
            attention_title,
            attention_architecture,
            attention_metrics,
            model_path=attention_model
        )

    baseline_dice = (
        _first_metric(baseline_metrics, "Dice coefficient", "Best Dice", "best_dice", "Dice")
        if baseline_metrics else None
    )
    attention_dice = (
        _first_metric(attention_metrics, "Dice coefficient", "Best Dice", "best_dice", "Dice")
        if attention_metrics else None
    )

    if baseline_dice is not None and attention_dice is not None:
        difference = attention_dice - baseline_dice
        st.markdown("##### Experiment conclusion")

        if difference > 0:
            st.success(
                f"Multi-head attention improved validation Dice by "
                f"{difference:+.4f} ({baseline_dice:.4f} → {attention_dice:.4f})."
            )
        elif difference < 0:
            st.warning(
                f"Multi-head attention scored {abs(difference):.4f} lower than the baseline "
                f"({baseline_dice:.4f} → {attention_dice:.4f})."
            )
        else:
            st.info(
                f"Both experiments achieved the same saved Dice score: "
                f"{baseline_dice:.4f}."
            )

    elif workflow == "2D" and attention_metrics is None:
        st.info(
            "The 2D attention experiment is not available here yet. "
            "After training finishes, copy `best_attention_metrics_2d.json` "
            "and `best_attention_unet2d.pth` into `best_model/`; "
            "this comparison updates automatically."
        )



def _ui_metric(metrics, *keys):
    if not metrics:
        return None
    for key in keys:
        value = metrics.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _dice_column(history):
    if history is None or history.empty:
        return None
    for name in ("val_dice", "val_dice_at_0.5"):
        if name in history.columns:
            return name
    return None


def _best_epoch_from_history(history):
    column = _dice_column(history)
    if history is None or history.empty or column is None:
        return None

    numeric = pd.to_numeric(history[column], errors="coerce")
    if numeric.dropna().empty:
        return None

    index = numeric.idxmax()
    try:
        return int(history.loc[index, "epoch"])
    except Exception:
        return None


def _render_current_metrics(metrics, history, dimension):
    dice = _ui_metric(metrics, "Dice coefficient", "Best Dice", "best_dice", "Dice")
    iou = _ui_metric(metrics, "Mean IoU", "IoU", "iou")
    precision = _ui_metric(metrics, "Precision", "precision")
    recall = _ui_metric(metrics, "Recall / Sensitivity", "Recall", "recall")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Dice", f"{dice:.4f}" if dice is not None else "—")
    with c2:
        st.metric("IoU", f"{iou:.4f}" if iou is not None else "—")
    with c3:
        st.metric("Precision", f"{precision:.4f}" if precision is not None else "—")
    with c4:
        st.metric("Recall", f"{recall:.4f}" if recall is not None else "—")

    best_epoch = None
    if metrics:
        try:
            best_epoch = int(metrics.get("Epoch"))
        except Exception:
            best_epoch = None

    if best_epoch is None:
        best_epoch = _best_epoch_from_history(history)

    if best_epoch is not None:
        st.caption(f"Best epoch: {best_epoch}")

    if history is None or history.empty:
        st.info(f"No {dimension} training history is available yet.")
        return

    loss_columns = [
        column for column in ("train_loss", "val_loss")
        if column in history.columns
    ]

    if loss_columns:
        st.markdown("#### Loss")
        loss_data = history[["epoch"] + loss_columns].copy()
        loss_data["epoch"] = pd.to_numeric(loss_data["epoch"], errors="coerce")
        for column in loss_columns:
            loss_data[column] = pd.to_numeric(loss_data[column], errors="coerce")
        loss_data = loss_data.dropna(subset=["epoch"]).set_index("epoch")
        st.line_chart(loss_data[loss_columns])

    dice_column = _dice_column(history)
    if dice_column:
        st.markdown("#### Validation Dice")
        dice_data = history[["epoch", dice_column]].copy()
        dice_data["epoch"] = pd.to_numeric(dice_data["epoch"], errors="coerce")
        dice_data[dice_column] = pd.to_numeric(
            dice_data[dice_column], errors="coerce"
        )
        dice_data = (
            dice_data
            .dropna(subset=["epoch", dice_column])
            .rename(columns={dice_column: "Validation Dice"})
            .set_index("epoch")
        )
        if not dice_data.empty:
            st.line_chart(dice_data)


def _render_experiment_comparison(
    baseline_metrics,
    attention_metrics,
    baseline_history,
    attention_history,
    dimension,
):
    specs = [
        ("Dice", ("Dice coefficient", "Best Dice", "best_dice", "Dice")),
        ("IoU", ("Mean IoU", "IoU", "iou")),
        ("Precision", ("Precision", "precision")),
        ("Recall", ("Recall / Sensitivity", "Recall", "recall")),
    ]

    rows = []
    for label, keys in specs:
        rows.append(
            {
                "Metric": label,
                "Baseline": _ui_metric(baseline_metrics, *keys),
                "Attention": _ui_metric(attention_metrics, *keys),
            }
        )

    comparison = pd.DataFrame(rows).set_index("Metric")

    left, right = st.columns(2)

    with left:
        st.markdown(f"#### Experiment 1: Baseline {dimension} U-Net")

        b_dice = comparison.loc["Dice", "Baseline"]
        b_iou = comparison.loc["IoU", "Baseline"]
        b_precision = comparison.loc["Precision", "Baseline"]
        b_recall = comparison.loc["Recall", "Baseline"]

        st.metric("Dice", f"{b_dice:.4f}" if pd.notna(b_dice) else "—")
        st.metric("IoU", f"{b_iou:.4f}" if pd.notna(b_iou) else "—")
        st.metric(
            "Precision",
            f"{b_precision:.4f}" if pd.notna(b_precision) else "—",
        )
        st.metric(
            "Recall",
            f"{b_recall:.4f}" if pd.notna(b_recall) else "—",
        )

    with right:
        st.markdown(
            f"#### Experiment 2: {dimension} U-Net + Multi-Head Attention"
        )

        a_dice = comparison.loc["Dice", "Attention"]
        a_iou = comparison.loc["IoU", "Attention"]
        a_precision = comparison.loc["Precision", "Attention"]
        a_recall = comparison.loc["Recall", "Attention"]

        delta = None
        if pd.notna(a_dice) and pd.notna(b_dice):
            delta = f"{a_dice - b_dice:+.4f}"

        st.metric(
            "Dice",
            f"{a_dice:.4f}" if pd.notna(a_dice) else "—",
            delta=delta,
        )
        st.metric("IoU", f"{a_iou:.4f}" if pd.notna(a_iou) else "—")
        st.metric(
            "Precision",
            f"{a_precision:.4f}" if pd.notna(a_precision) else "—",
        )
        st.metric(
            "Recall",
            f"{a_recall:.4f}" if pd.notna(a_recall) else "—",
        )

    valid = comparison.dropna(how="all")
    if not valid.empty:
        st.markdown("#### Validation Metric Comparison")
        st.bar_chart(valid)

    b_col = _dice_column(baseline_history)
    a_col = _dice_column(attention_history)

    if b_col and a_col:
        b = baseline_history[["epoch", b_col]].copy()
        a = attention_history[["epoch", a_col]].copy()

        b["epoch"] = pd.to_numeric(b["epoch"], errors="coerce")
        b["Baseline"] = pd.to_numeric(b[b_col], errors="coerce")

        a["epoch"] = pd.to_numeric(a["epoch"], errors="coerce")
        a["Attention"] = pd.to_numeric(a[a_col], errors="coerce")

        merged = pd.merge(
            b[["epoch", "Baseline"]],
            a[["epoch", "Attention"]],
            on="epoch",
            how="outer",
        ).sort_values("epoch")

        merged = merged.dropna(subset=["epoch"]).set_index("epoch")

        if not merged.empty:
            st.markdown("#### Validation Dice Across Epochs")
            st.line_chart(merged)

    if pd.notna(b_dice) and pd.notna(a_dice):
        difference = a_dice - b_dice

        if difference > 0:
            st.success(
                f"Experiment 2 improved Dice by {difference:+.4f} "
                f"and is used as the deployed {dimension} model."
            )
        elif difference < 0:
            st.warning(
                f"Experiment 2 scored {abs(difference):.4f} below the baseline."
            )
        else:
            st.info("Both experiments achieved the same Dice score.")



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


def render_sample_folder_sidebar(folder, extensions, title, help_text, mime_type):
    sample_files = get_sample_files(folder, extensions)

    with st.sidebar.expander(title, expanded=False):
        st.caption(help_text)

        if len(sample_files) == 0:
            st.warning(
                f"No sample files found in `{folder}`. Restore the tracked files inside "
                "`sample_data/2d` or `sample_data/3d`."
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

        if export_3d.get("gradcam_overlay") is not None:
            st.download_button(
                label="Grad-CAM Overlay",
                data=export_3d["gradcam_overlay"],
                file_name=export_3d["gradcam_name"],
                mime="image/png",
                use_container_width=True,
                key="export_3d_gradcam"
            )


def image_to_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def normalize_slice_for_display(slice_2d):
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


def get_representative_3d_indices(mask_3d):
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


def get_device_3d():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_3d_model_path():
    if os.path.exists(BEST_MODEL_3D_PATH):
        return BEST_MODEL_3D_PATH
    return None


def predict_uploaded_3d_npz(uploaded_file, threshold=0.55):
    model_path = get_3d_model_path()
    if model_path is None:
        raise FileNotFoundError(
            "No deployed 3D model found. Expected "
            "best_model_3d/best_attention_unet3d.pth."
        )

    return predict_full_volume_npz(
        uploaded_file=uploaded_file,
        model_path=model_path,
        device=get_device_3d(),
        threshold=threshold,
    )

def build_full_gradcam_volume(result, gradcam_result):
    image = np.asarray(result["image"])

    if image.ndim == 4 and image.shape[0] == 4:
        depth, height, width = image.shape[1:]
    elif image.ndim == 4 and image.shape[-1] == 4:
        depth, height, width = image.shape[:3]
    elif image.ndim == 3:
        depth, height, width = image.shape
    else:
        raise ValueError("Unsupported MRI shape for 3D Grad-CAM rendering.")

    full_cam = np.zeros(
        (depth, height, width),
        dtype=np.float32,
    )

    cam = np.asarray(
        gradcam_result["cam_volume"],
        dtype=np.float32,
    )

    start = int(
        gradcam_result.get(
            "window_start",
            0,
        )
    )

    end = int(
        gradcam_result.get(
            "window_end",
            start + cam.shape[0],
        )
    )

    end = min(
        end,
        depth,
        start + cam.shape[0],
    )

    valid_depth = max(
        0,
        end - start,
    )

    if valid_depth > 0:
        full_cam[start:end] = cam[:valid_depth]

    return full_cam


def _add_gradcam_isosurface(
    figure,
    volume,
    level,
    color,
    opacity,
    name,
):
    volume = np.asarray(volume, dtype=np.float32)
    volume = ndi.gaussian_filter(volume, sigma=0.65)

    if not (
        float(volume.min())
        < float(level)
        < float(volume.max())
    ):
        return False

    padded = np.pad(
        volume,
        2,
        mode="constant",
        constant_values=0.0,
    )

    vertices, faces, _, _ = marching_cubes(
        padded,
        level=float(level),
        allow_degenerate=False,
    )

    vertices -= 2.0

    figure.add_trace(
        go.Mesh3d(
            x=vertices[:, 2],
            y=vertices[:, 1],
            z=vertices[:, 0],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=opacity,
            name=name,
            flatshading=False,
            lighting=dict(
                ambient=0.40,
                diffuse=0.76,
                specular=0.18,
                roughness=0.58,
                fresnel=0.08,
            ),
            lightposition=dict(
                x=180,
                y=220,
                z=260,
            ),
            hoverinfo="skip",
        )
    )

    return True


def build_combined_tumor_gradcam_figure(
    result,
    gradcam_result,
    modality_index=0,
    show_tumour=True,
    show_gradcam=True,
):
    image_4d = _canonicalise_image(
        np.asarray(result["image"])
    )

    probability = np.asarray(
        result["probability"],
        dtype=np.float32,
    )

    threshold = float(
        result.get(
            "threshold",
            0.55,
        )
    )

    figure = go.Figure()

    bbox, _crop_shape, display_shape = _add_brain_surface(
        figure,
        image_4d,
        int(modality_index),
    )

    if show_tumour:
        _add_tumour_surface(
            figure,
            probability,
            bbox,
            display_shape,
            threshold,
        )

    if show_gradcam and gradcam_result is not None:
        full_cam = build_full_gradcam_volume(
            result,
            gradcam_result,
        )

        cam_small = _crop_and_resize_probability(
            full_cam,
            bbox,
            display_shape,
        )

        cam_small = np.maximum(
            np.asarray(cam_small, dtype=np.float32),
            0.0,
        )

        positive = cam_small[
            cam_small > 0
        ]

        if positive.size > 12:
            high = float(
                np.percentile(
                    positive,
                    99.0,
                )
            )

            if high > 1e-8:
                cam_small = np.clip(
                    cam_small / high,
                    0.0,
                    1.0,
                )

                positive = cam_small[
                    cam_small > 0.02
                ]

                if positive.size > 12:
                    low_level = float(
                        np.clip(
                            np.percentile(
                                positive,
                                52.0,
                            ),
                            0.18,
                            0.48,
                        )
                    )

                    mid_level = float(
                        np.clip(
                            np.percentile(
                                positive,
                                70.0,
                            ),
                            low_level + 0.04,
                            0.68,
                        )
                    )

                    high_level = float(
                        np.clip(
                            np.percentile(
                                positive,
                                86.0,
                            ),
                            mid_level + 0.04,
                            0.84,
                        )
                    )

                    _add_gradcam_isosurface(
                        figure,
                        cam_small,
                        low_level,
                        "rgb(255,190,70)",
                        0.18,
                        "Grad-CAM — broader attention",
                    )

                    _add_gradcam_isosurface(
                        figure,
                        cam_small,
                        mid_level,
                        "rgb(255,142,32)",
                        0.34,
                        "Grad-CAM — medium attention",
                    )

                    _add_gradcam_isosurface(
                        figure,
                        cam_small,
                        high_level,
                        "rgb(255,86,24)",
                        0.62,
                        "Grad-CAM — strongest attention",
                    )

    figure.update_layout(
        height=680,
        margin=dict(
            l=0,
            r=0,
            t=46,
            b=0,
        ),
        title="MRI Brain — Tumor and 3D Grad-CAM",
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            aspectmode="data",
            camera=dict(
                eye=dict(
                    x=1.50,
                    y=1.35,
                    z=1.05,
                )
            ),
            bgcolor="rgb(13,15,20)",
        ),
        paper_bgcolor="rgb(13,15,20)",
        plot_bgcolor="rgb(13,15,20)",
        font=dict(
            color="white",
        ),
        legend=dict(
            orientation="h",
            y=1.02,
            x=0.5,
            xanchor="center",
        ),
    )

    return figure


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
    </style>
    """,
    unsafe_allow_html=True
)


if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Home"

if "current_page" not in st.session_state:
    st.session_state["current_page"] = st.session_state["active_page"]

if "upload_reset_counter" not in st.session_state:
    st.session_state["upload_reset_counter"] = 0

page = st.session_state["active_page"]


if page != "Home":
    top_home_col, top_space = st.columns([1, 7])

    with top_home_col:
        if st.button(
            "← Home",
            key=f"home_top_{page.replace(' ', '_')}",
            use_container_width=True,
        ):
            st.session_state["active_page"] = "Home"
            st.rerun()

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

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Home"

if "current_page" not in st.session_state:
    st.session_state["current_page"] = st.session_state["active_page"]

if "upload_reset_counter" not in st.session_state:
    st.session_state["upload_reset_counter"] = 0


def go_to_page(page_name):
    st.session_state["active_page"] = page_name


def render_workflow_buttons(workflow):
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

if page != st.session_state["current_page"]:
    keys_to_clear = [
        "result_3d",
        "input_mode_3d_previous",
        "export_2d",
        "export_3d",
        "last_3d_upload_signature",
        "gradcam_3d",
        "gradcam_3d_key"
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state["upload_reset_counter"] += 1
    st.session_state["current_page"] = page

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
            value=0.55,
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
            "MRI Modality",
            options=[0, 1, 2, 3],
            index=0,
            format_func=lambda x: MODALITY_NAMES[x],
            help="The model uses all four modalities. This control only changes the MRI shown in the viewer."
        )



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

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.info("Tip: If you do not have MRI files, open an Analysis page and use the sample downloads in the sidebar.")


elif page == "2D Info":
    render_workflow_buttons("2D")
    st.markdown("---")
    st.subheader("2D MRI Workflow")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Architecture", "Attention U-Net")
    with c2:
        st.metric("Input", "2D grayscale MRI")
    with c3:
        st.metric("Best Dice", dice_text)

    st.markdown("""
    This section uses the selected 2D attention U-Net for single-slice MRI tumor segmentation.

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
    - **Ground truth:** shown only when the uploaded NPZ already contains a compatible `mask` array
    - **Training:** view the saved 3D model performance and training curves
    """)


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

        st.markdown("### Grad-CAM")
        grad_left, grad_center, grad_right = st.columns([1, 2, 1])

        with grad_center:
            st.image(
                gradcam_overlay,
                use_container_width=True,
                caption="Model Attention Heatmap"
            )

        st.markdown("### Segmentation Views")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Input Scan")
            st.image(
                resized_img,
                use_container_width=True,
                caption="Pre-processed Grayscale MRI"
            )

        with col2:
            st.markdown("#### Segmentation")
            st.image(
                overlay,
                use_container_width=True,
                caption="Red: Predicted Tumor Region"
            )

        with col3:
            st.markdown("#### Tumor Mask")
            st.image(
                mask_only,
                use_container_width=True,
                caption="White: Predicted Tumor Pixels"
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


elif page == "3D MRI Analysis":
    render_workflow_buttons("3D")
    st.markdown("---")
    st.subheader("3D MRI Volume Analysis")

    uploaded_3d_file = st.file_uploader(
        "Upload 3D MRI volume (.NPZ)",
        type=["npz"],
        help=(
            "Supports full patient volumes such as [4,155,240,240], plus "
            "older 32-slice samples. The model processes full volumes in "
            "overlapping 32-slice windows."
        ),
        key=f"upload_3d_{st.session_state['upload_reset_counter']}"
    )

    if uploaded_3d_file:
        upload_signature = (
            f"{uploaded_3d_file.name}_{uploaded_3d_file.size}_{threshold_3d}"
        )

        if (
            st.session_state.get("last_3d_upload_signature")
            != upload_signature
        ):
            with st.spinner("Analyzing 3D MRI volume..."):
                result = predict_uploaded_3d_npz(
                    uploaded_file=uploaded_3d_file,
                    threshold=threshold_3d,
                )

            st.session_state["result_3d"] = result
            st.session_state["last_3d_upload_signature"] = upload_signature

            for cache_key in (
                "gradcam_3d",
                "gradcam_3d_key",
            ):
                if cache_key in st.session_state:
                    del st.session_state[cache_key]

    else:
        for key in (
            "result_3d",
            "export_3d",
            "last_3d_upload_signature",
            "gradcam_3d",
            "gradcam_3d_key",
        ):
            if key in st.session_state:
                del st.session_state[key]

        st.info(
            "Upload a .NPZ 3D MRI volume to generate the slice views, "
            "automatic Grad-CAM explanation, and interactive 3D rendering."
        )

    if "result_3d" in st.session_state:
        result = st.session_state["result_3d"]

        suggested_slice, _, _ = get_representative_3d_indices(
            result["pred_mask"]
        )

        st.markdown("### 3D Slice Explorer")
        st.caption(
            "Use the slider to move through the MRI volume. "
            "Grad-CAM updates automatically for the selected slice."
        )

        slice_index = st.slider(
            "Slice",
            min_value=0,
            max_value=result["depth"] - 1,
            value=suggested_slice,
            step=1,
            help="Move through the 3D MRI volume slice by slice.",
        )

        (
            input_img,
            pred_mask_img,
            true_mask_img,
            overlay_img,
            prob_img,
        ) = get_3d_display_slices(
            result,
            slice_index=slice_index,
            modality_index=modality_index,
            overlay_alpha=overlay_alpha_3d,
        )

        input_img_display = resize_for_display(
            input_img,
            resample=Image.Resampling.LANCZOS,
        )

        pred_mask_img_display = resize_for_display(
            pred_mask_img,
            resample=Image.Resampling.NEAREST,
        )

        overlay_img_display = resize_for_display(
            overlay_img,
            resample=Image.Resampling.NEAREST,
        )

        prob_img_display = resize_for_display(
            prob_img,
            resample=Image.Resampling.LANCZOS,
        )

        gradcam_key = (
            f"{result['volume_id']}_{slice_index}_{modality_index}_"
            f"{BEST_MODEL_3D_PATH}"
        )

        if (
            "gradcam_3d" not in st.session_state
            or st.session_state.get("gradcam_3d_key") != gradcam_key
        ):
            with st.spinner("Generating Grad-CAM for this slice..."):
                gradcam_result = generate_3d_gradcam(
                    result=result,
                    model_path=BEST_MODEL_3D_PATH,
                    device=get_device_3d(),
                    selected_slice=slice_index,
                    modality_index=modality_index,
                    heatmap_alpha=0.42,
                )

            st.session_state["gradcam_3d"] = gradcam_result
            st.session_state["gradcam_3d_key"] = gradcam_key

        else:
            gradcam_result = st.session_state["gradcam_3d"]

        gradcam_overlay_display = resize_for_display(
            Image.fromarray(
                gradcam_result["overlay_rgb"]
            ),
            resample=Image.Resampling.LANCZOS,
        )

        st.session_state["export_3d"] = {
            "input_img": image_to_png_bytes(input_img_display),
            "pred_mask_img": image_to_png_bytes(pred_mask_img_display),
            "overlay_img": image_to_png_bytes(overlay_img_display),
            "true_mask_img": None,
            "prob_img": image_to_png_bytes(prob_img_display),
            "gradcam_overlay": image_to_png_bytes(gradcam_overlay_display),
            "input_name": (
                f"3d_volume_{result['volume_id']}_slice_"
                f"{slice_index}_mri.png"
            ),
            "pred_name": (
                f"3d_volume_{result['volume_id']}_slice_"
                f"{slice_index}_prediction_mask.png"
            ),
            "overlay_name": (
                f"3d_volume_{result['volume_id']}_slice_"
                f"{slice_index}_overlay.png"
            ),
            "true_name": "",
            "prob_name": (
                f"3d_volume_{result['volume_id']}_slice_"
                f"{slice_index}_probability.png"
            ),
            "gradcam_name": (
                f"3d_volume_{result['volume_id']}_slice_"
                f"{slice_index}_gradcam_overlay.png"
            ),
        }

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown("#### MRI Slice")
            st.image(
                input_img_display,
                use_container_width=True,
                caption=f"Slice {slice_index}",
            )

        with col2:
            st.markdown("#### Prediction")
            st.image(
                pred_mask_img_display,
                use_container_width=True,
                caption="Predicted tumor mask",
            )

        with col3:
            st.markdown("#### Overlay")
            st.image(
                overlay_img_display,
                use_container_width=True,
                caption="Prediction over MRI",
            )

        with col4:
            st.markdown("#### Grad-CAM")
            st.image(
                gradcam_overlay_display,
                use_container_width=True,
                caption="Model attention for this slice",
            )

        with col5:
            st.markdown("#### Probability")
            st.image(
                prob_img_display,
                use_container_width=True,
                caption="Model probability map",
            )

        st.markdown("---")

        tumor_voxels = int(
            result["pred_mask"].sum()
        )

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(
                "Prediction Result",
                (
                    "Detected"
                    if tumor_voxels > 0
                    else "Not detected"
                ),
            )

        with metric_col2:
            st.metric(
                "Predicted Tumor Voxels",
                tumor_voxels,
            )

        st.markdown("---")
        st.markdown("### Interactive 3D View")

        render_col1, render_col2 = st.columns(2)

        with render_col1:
            show_tumour_render = st.checkbox(
                "Show Tumor Rendering",
                value=True,
                key="show_tumour_render_3d",
            )

        with render_col2:
            show_gradcam_render = st.checkbox(
                "Show 3D Grad-CAM",
                value=False,
                key="show_gradcam_render_3d",
            )

        st.caption(
            "Red = predicted tumor. "
            "Orange/yellow = Grad-CAM attention; stronger attention is "
            "shown with a brighter, more opaque inner surface."
        )

        if not show_tumour_render and not show_gradcam_render:
            st.info(
                "Select Tumor Rendering, 3D Grad-CAM, or both."
            )

        else:
            figure_3d = build_combined_tumor_gradcam_figure(
                result=result,
                gradcam_result=gradcam_result,
                modality_index=modality_index,
                show_tumour=show_tumour_render,
                show_gradcam=show_gradcam_render,
            )

            st.plotly_chart(
                figure_3d,
                use_container_width=True,
            )

        if result.get("window_count", 1) > 1:
            st.caption(
                f"Full depth: {result['depth']} slices. "
                f"The 32-slice model was run over "
                f"{result['window_count']} overlapping windows and "
                "merged into one prediction."
            )



elif page == "2D Training Progress":
    render_workflow_buttons("2D")
    st.markdown("---")
    st.subheader("2D Training")

    current_tab, experiments_tab = st.tabs(
        ["Current Training", "Experiments"]
    )

    attention_metrics_2d = load_json(ATTENTION_2D_METRICS)
    attention_history_2d = load_csv(ATTENTION_2D_HISTORY)
    baseline_metrics_2d = load_json(BASELINE_2D_METRICS)
    baseline_history_2d = load_csv(BASELINE_2D_HISTORY)

    with current_tab:
        st.markdown("### Deployed 2D Model")
        st.caption(
            "Experiment 2 · 2D U-Net + Multi-Head Attention. "
            "This model is used by the 2D Analysis page."
        )

        if attention_metrics_2d is not None:
            _render_current_metrics(
                attention_metrics_2d,
                attention_history_2d,
                "2D",
            )
        else:
            st.warning(
                "Attention metrics are not available yet: "
                "`best_model/best_attention_metrics_2d.json`"
            )

    with experiments_tab:
        _render_experiment_comparison(
            baseline_metrics_2d,
            attention_metrics_2d,
            baseline_history_2d,
            attention_history_2d,
            "2D",
        )

elif page == "3D Training Progress":
    render_workflow_buttons("3D")
    st.markdown("---")
    st.subheader("3D Training")

    current_tab, experiments_tab = st.tabs(
        ["Current Training", "Experiments"]
    )

    attention_metrics_3d = load_json(ATTENTION_3D_METRICS)
    attention_history_3d = load_csv(ATTENTION_3D_HISTORY)
    baseline_metrics_3d = load_json(BASELINE_3D_METRICS)
    baseline_history_3d = load_csv(BASELINE_3D_HISTORY)

    with current_tab:
        st.markdown("### Deployed 3D Model")
        st.caption(
            "Experiment 2 · 3D U-Net + Multi-Head Attention. "
            "This model is used by the 3D Analysis page."
        )

        if attention_metrics_3d is not None:
            _render_current_metrics(
                attention_metrics_3d,
                attention_history_3d,
                "3D",
            )
        else:
            st.warning(
                "Attention metrics are not available yet: "
                "`best_model_3d/best_attention_metrics_3d.json`"
            )

    with experiments_tab:
        _render_experiment_comparison(
            baseline_metrics_3d,
            attention_metrics_3d,
            baseline_history_3d,
            attention_history_3d,
            "3D",
        )

if page == "2D MRI Analysis":
    st.sidebar.markdown("---")
    render_export_2d_sidebar()
elif page == "3D MRI Analysis":
    st.sidebar.markdown("---")
    render_export_3d_sidebar()
