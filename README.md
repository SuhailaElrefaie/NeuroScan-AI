# NeuroScan AI

**AI-assisted brain tumour MRI segmentation using 2D and 3D deep learning**

NeuroScan AI is an Individual Software Project developed at Charles University under the supervision of **Kassem Anis**. The system segments brain tumours from MRI scans using PyTorch models and presents the results through an interactive Streamlit application.

**Live demo:** https://neuroscan-ai-isp.streamlit.app

---

## Project overview

NeuroScan AI contains two segmentation pipelines:

- **2D U-Net:** processes individual grayscale MRI slices and predicts a binary tumour mask.
- **3D Attention U-Net:** processes multi-modal MRI volumes and predicts a binary whole-tumour mask across the full patient volume.

The model predicts the **tumour mask only**. The MRI shown in the application always comes from the uploaded/original scan. The predicted mask is then overlaid on the MRI for visualisation.

---

## Main features

### 2D workflow

- Upload a `.png`, `.jpg`, or `.jpeg` brain MRI.
- Resize and normalise the image to the model input format.
- Segment tumour pixels using a 2D U-Net.
- Display:
  - original MRI,
  - predicted binary mask,
  - red tumour overlay,
  - Grad-CAM explanation heatmap.
- Export generated images.

### 3D workflow

- Upload a multi-modal MRI volume as `.npz`.
- Supports:
  - `[4, D, H, W]`
  - `[D, H, W, 4]`
  - `[D, H, W]` for compatibility by repeating the single volume to four channels.
- The four MRI modalities are:
  - `0 = FLAIR`
  - `1 = T1`
  - `2 = T1CE`
  - `3 = T2`
- Perform full-volume segmentation with overlapping 32-slice inference windows.
- Explore every axial slice.
- Display:
  - selected MRI modality,
  - predicted tumour mask,
  - red overlay,
  - probability map,
  - optional ground-truth mask when included in the uploaded sample,
  - interactive 3D rendering.
- The 3D viewer derives the anatomical surface from the **original MRI** and renders the predicted tumour as a separate surface mesh using **Marching Cubes**.

---

## Model architecture

### 2D U-Net

Input:

```text
[1, 256, 256]
```

Architecture:

```text
1
↓
32
↓
64
↓
128
↓
256 bottleneck
↓
128
↓
64
↓
32
↓
1 output channel
```

The encoder uses convolution blocks and max pooling. The decoder uses transposed convolutions and U-Net skip connections to restore spatial detail.

The model returns **raw logits**. Sigmoid and thresholding are applied during evaluation/inference to create the binary mask.

### 3D Attention U-Net

Model input:

```text
[4, 32, 160, 160]
```

The deployed 3D model uses:

- 3D residual convolution blocks,
- Group Normalisation,
- SiLU activation,
- four encoder levels,
- multi-head self-attention at the bottleneck,
- U-Net skip connections,
- 3D transposed-convolution decoder,
- one binary tumour output channel.

Attention is applied at the compact bottleneck so the model can capture longer-range relationships without the very high cost of full-resolution attention.

---

## Full-volume 3D inference

The network processes fixed windows of:

```text
4 modalities × 32 slices × 160 × 160
```

A complete MRI volume may be much larger, for example around:

```text
4 × 155 × 240 × 240
```

`full_volume_3d.py` therefore:

1. loads the complete volume,
2. creates overlapping 32-slice windows,
3. applies the same per-modality normalisation used for training,
4. runs the deployed 3D model,
5. resizes each probability window back to the original in-plane resolution,
6. averages predictions where windows overlap,
7. applies the selected threshold,
8. returns one full-depth tumour mask.

This means the model sees 32 slices at a time, but the final result covers the complete uploaded patient volume.

---

## 3D surface rendering

3D rendering is a **visualisation stage**, not part of model training.

```text
Original MRI
    ↓
MRI-derived anatomical surface
    ↓
Marching Cubes
    ↓
grey brain mesh

Predicted probability
    ↓
threshold
    ↓
Marching Cubes
    ↓
red tumour mesh
```

The U-Net does **not** generate the grey MRI or brain anatomy. It only predicts tumour segmentation logits.

---

## Datasets

### BRISC 2025 — 2D

Used for single-slice segmentation.

https://www.kaggle.com/datasets/briscdataset/brisc2025/

Expected local structure:

```text
Data/segmentation_task/train/images
Data/segmentation_task/train/masks
Data/segmentation_task/test/images
Data/segmentation_task/test/masks
```

### BraTS — 3D

Used for multi-modal whole-tumour MRI segmentation.

Converted H5 slices use the format:

```text
image  # [H, W, 4]
mask   # binary whole-tumour target
```

All non-zero tumour labels are merged into one binary whole-tumour class.

The complete datasets are intentionally excluded from GitHub because of their size. Only small demonstration files are included.

---

## Saved validation results

### 2D model

| Metric | Result |
|---|---:|
| Dice | **0.8473** |
| Mean IoU | 0.7378 |
| Precision | 0.8378 |
| Recall | 0.8607 |
| Threshold | 0.30 |

### 3D model

| Metric | Result |
|---|---:|
| Dice | **0.9326** |
| Mean IoU | 0.8737 |
| Precision | 0.9368 |
| Recall | 0.9284 |
| Threshold | 0.55 |
| Best checkpoint epoch | 35 |

The 3D score is an internal validation result using a patient-level split and full-depth overlapping validation windows. It is not an official hidden BraTS challenge score.

---

## Repository structure

```text
NeuroScan-AI/
│
├── ui.py
├── predict.py
├── full_volume_3d.py
│
├── dataset.py
├── dataset_3d.py
├── dataset_attention_3d.py
│
├── train.py
├── train_3d.py
├── train_attention_3d.py
│
├── convert_brats2021_to_h5.py
├── gradcam.py
├── requirements.txt
├── run_app.command
│
├── models/
│   ├── unet.py
│   ├── unet3d.py
│   └── attention_unet3d.py
│
├── best_model/
│   ├── best_unet.pth
│   ├── best_metrics.json
│   └── best_history.csv
│
├── best_model_3d/
│   ├── best_attention_unet3d.pth
│   ├── best_attention_metrics_3d.json
│   └── best_attention_history_3d.csv
│
├── sample_data/
│   ├── 2d/
│   └── 3d/
│
└── tools/
    ├── verify_project.py
    └── check_gpu.py
```

### Important files

| File | Purpose |
|---|---|
| `dataset.py` | Loads and prepares 2D MRI/mask pairs |
| `models/unet.py` | 2D U-Net architecture |
| `train.py` | 2D training and validation |
| `predict.py` | 2D inference, mask post-processing and Grad-CAM path |
| `gradcam.py` | Grad-CAM implementation |
| `dataset_3d.py` | Baseline patient-level 3D window loader |
| `dataset_attention_3d.py` | Dataset/indexing pipeline used for attention-model training |
| `models/unet3d.py` | Baseline 3D U-Net |
| `models/attention_unet3d.py` | Deployed residual attention 3D U-Net |
| `train_3d.py` | Baseline/final 3D training pipeline |
| `train_attention_3d.py` | Attention U-Net training pipeline |
| `full_volume_3d.py` | Full-volume 3D inference and interactive surface rendering |
| `ui.py` | Streamlit application |

---

## Installation

```bash
git clone https://github.com/SuhailaElrefaie/NeuroScan-AI.git
cd NeuroScan-AI

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run ui.py
```

---

## Deployed model files

2D:

```text
best_model/best_unet.pth
```

3D:

```text
best_model_3d/best_attention_unet3d.pth
```

The Streamlit application loads the attention checkpoint directly.

---

## Training

### 2D

```bash
python3 train.py
```

### Baseline 3D

```bash
python3 train_3d.py
```

### Attention 3D

Example:

```bash
python3 train_attention_3d.py --h5-dir /path/to/BraTS2021_h5/content/data
```

The 3D training pipelines use patient-level train/validation separation to avoid slices from the same patient leaking between training and validation.

---

## Verify the project

```bash
python3 tools/verify_project.py
```

This checks important project paths, model files, metrics and demonstration data without modifying the repository.

---

## Technology stack

- Python
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Pandas
- Pillow
- OpenCV
- h5py
- nibabel
- Plotly
- scikit-image
- SciPy
- Git / GitHub
- Streamlit Community Cloud

---

## Limitations

- NeuroScan AI is an **academic prototype**, not a medical device.
- It must not be used for diagnosis or clinical decision-making.
- The 3D task is binary whole-tumour segmentation rather than separate BraTS tumour subregions.
- The validation metrics are internal project results and have not been externally validated on an independent clinical cohort.
- 3D surface quality depends on the spatial resolution and preprocessing of the uploaded MRI.

---

## Author

Developed for the **Individual Software Project, Charles University**.

Supervisor: **Kassem Anis**
