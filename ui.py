"""
Streamlit user interface for the brain tumor MRI segmentation system.

The interface allows the user to:
- upload an MRI image
- view the input scan
- view the segmentation overlay
- view the binary tumor mask
- view the Grad-CAM heatmap
- download the generated segmentation result and tumor mask
- view saved training progress from training_history.csv
"""

import io
import json
import os

import pandas as pd
import streamlit as st
from predict import predict_with_gradcam, get_best_threshold


SUMMARY_JSON_PATH = "training_summary.json"
HISTORY_CSV_PATH = "training_history.csv"


st.set_page_config(
    page_title="NeuroScan AI | Tumor Segmentation",
    layout="wide"
)


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

@st.cache_data
def load_training_summary():
    """
    Loads best Dice and best threshold information saved by train.py.
    """

    if not os.path.exists(SUMMARY_JSON_PATH):
        return None

    try:
        with open(SUMMARY_JSON_PATH, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def rounded_slider_value(value):
    """
    Rounds a threshold value so Streamlit can use it in a 0.05-step slider.
    """

    value = round(float(value) / 0.05) * 0.05
    return min(max(value, 0.10), 0.90)


summary = load_training_summary()

if summary is not None:
    default_threshold = rounded_slider_value(summary.get("best_threshold", get_best_threshold()))
    best_dice_text = f"{summary.get('best_dice_with_threshold_tuning', 0):.4f}"
    best_epoch_text = str(summary.get("best_epoch", "N/A"))
else:
    default_threshold = rounded_slider_value(get_best_threshold())
    best_dice_text = "Run train.py first"
    best_epoch_text = "N/A"


# -----------------------------
# CUSTOM PAGE STYLING
# -----------------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7f9;
    }

    .stButton>button, .stDownloadButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] {
        font-size: 1.05rem;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        font-size: 1.02rem;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] label p {
        font-size: 1.12rem;
        font-weight: 600;
    }

    .metric-container {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# PAGE HEADER
# -----------------------------

st.markdown(
    """
    <h1 style='text-align: center;'>AI-Assisted Brain Tumor MRI Segmentation</h1>
    <p style='text-align: center; font-size: 1.15rem; color: #666666;'>
        U-Net segmentation with Grad-CAM visual explanation
    </p>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# SIDEBAR
# -----------------------------

page = st.sidebar.radio(
    "Navigation",
    ["MRI Analysis", "Training Progress"]
)

st.sidebar.markdown("---")

with st.sidebar.expander("Controls", expanded=False):
    threshold = st.slider(
        "Segmentation Threshold",
        min_value=0.10,
        max_value=0.90,
        value=default_threshold,
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

st.sidebar.markdown("---")

st.sidebar.markdown(
    f"""
    ### Current Model
    - **Architecture:** U-Net
    - **Input:** 2D grayscale MRI
    - **Best Validation Dice:** {best_dice_text}
    - **Best Epoch:** {best_epoch_text}
    - **Default Threshold:** {default_threshold:.2f}
    """
)


# -----------------------------
# MRI ANALYSIS PAGE
# -----------------------------

if page == "MRI Analysis":
    uploaded_file = st.file_uploader(
        "Upload MRI Scan Image (PNG/JPG)",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        with open("temp_upload.png", "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
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

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)

            tumor_pixels = int(mask.sum())
            detection_status = "Positive" if tumor_pixels > 0 else "Negative"

            with m_col1:
                st.metric("Detection Status", detection_status)

            with m_col2:
                st.metric("Estimated Tumor Area", f"{tumor_pixels} px")

            with m_col3:
                st.metric("Threshold Used", f"{threshold:.2f}")

            with m_col4:
                st.metric("Validation Dice", best_dice_text)

            st.markdown("---")
            st.markdown("### Export Results")

            overlay_buf = io.BytesIO()
            overlay.save(overlay_buf, format="PNG")
            overlay_bytes = overlay_buf.getvalue()

            st.download_button(
                label="📥 Download Segmentation Overlay",
                data=overlay_bytes,
                file_name="segmentation_overlay.png",
                mime="image/png"
            )

            mask_buf = io.BytesIO()
            mask_only.save(mask_buf, format="PNG")
            mask_bytes = mask_buf.getvalue()

            st.download_button(
                label="📥 Download Tumor Mask",
                data=mask_bytes,
                file_name="tumor_mask.png",
                mime="image/png"
            )

            gradcam_buf = io.BytesIO()
            gradcam_overlay.save(gradcam_buf, format="PNG")
            gradcam_bytes = gradcam_buf.getvalue()

            st.download_button(
                label="📥 Download Grad-CAM Overlay",
                data=gradcam_bytes,
                file_name="gradcam_overlay.png",
                mime="image/png"
            )

        except FileNotFoundError as error:
            st.error(str(error))
            st.info("Run python3 train.py first so the trained model file is created.")

    else:
        st.info(
            "Upload a PNG or JPG MRI slice to generate a predicted tumor overlay, "
            "binary tumor mask, and Grad-CAM explanation."
        )


# -----------------------------
# TRAINING PROGRESS PAGE
# -----------------------------

elif page == "Training Progress":
    st.subheader("Training Progress")

    if os.path.exists(HISTORY_CSV_PATH):
        history = pd.read_csv(HISTORY_CSV_PATH)

        st.markdown("### Loss During Training")
        st.line_chart(
            history,
            x="epoch",
            y=["train_loss", "val_loss"],
            x_label="Epoch",
            y_label="Loss"
        )

        st.markdown("### Validation Dice Score")
        st.line_chart(
            history,
            x="epoch",
            y="val_dice",
            x_label="Epoch",
            y_label="Dice Score"
        )

        best_dice = history["val_dice"].max()
        best_epoch = history.loc[history["val_dice"].idxmax(), "epoch"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Best Validation Dice at 0.50", f"{best_dice:.4f}")

        with col2:
            st.metric("Best Epoch", int(best_epoch))

        with col3:
            st.metric("Best Threshold", f"{default_threshold:.2f}")

        if summary is not None:
            st.markdown("### Saved Training Summary")
            st.json(summary)

    else:
        st.info(
            "No training history file found yet. Run train.py once to generate "
            "training_history.csv and training_summary.json."
        )
