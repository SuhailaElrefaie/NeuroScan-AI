# NeuroScan AI

**AI-assisted brain tumour MRI segmentation using 2D and 3D deep learning**

NeuroScan AI is an Individual Software Project developed at Charles University under the supervision of **Kassem Anis**. The project focuses on automatic brain-tumour segmentation from MRI scans using PyTorch.

**Live demo:** https://neuroscan-ai-isp.streamlit.app

---

## Project goal

The goal of NeuroScan AI is to identify and segment tumour regions in brain MRI scans.

The project contains two main model pipelines:

- **2D segmentation** for individual MRI slices.
- **3D segmentation** for complete multi-modal MRI volumes.

Both pipelines perform **binary segmentation**:

```text
0 = background
1 = tumour
```

---

# 2D Pipeline

The 2D model works with individual grayscale MRI slices.

### Dataset

**BRISC 2025**

https://www.kaggle.com/datasets/briscdataset/brisc2025/

Expected local structure:

```text
Data/segmentation_task/train/images
Data/segmentation_task/train/masks
Data/segmentation_task/test/images
Data/segmentation_task/test/masks
```

### Input

```text
[1, 256, 256]
```

### Main training settings

```text
Image size:       256 × 256
Batch size:       2
Epochs:           80
Learning rate:    1e-4
Validation split: 20%
Threshold:        0.30
```


### Saved 2D results

| Metric | Result |
|---|---:|
| Dice | **0.8473** |
| Mean IoU | 0.7378 |
| Precision | 0.8378 |
| Recall | 0.8607 |
| Threshold | 0.30 |

### 2D files

```text
dataset.py
models/unet.py
train.py
predict.py
gradcam.py

best_model/
├── best_unet.pth
├── best_metrics.json
└── best_history.csv
```

---

# 3D Pipeline

The 3D model works with multi-modal BraTS MRI volumes.

### Dataset

**BraTS 2021 Task 1**

https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1

The NIfTI dataset can be converted into H5 slices using:

```text
convert_brats2021_to_h5.py
```

Converted files are stored locally under:

```text
archive/BraTS2021_h5/content/data
```

Each H5 slice contains:

```text
image  # [H, W, 4]
mask   # binary whole-tumour target
```

MRI modality order:

```text
0 = FLAIR
1 = T1
2 = T1CE
3 = T2
```

All non-zero BraTS tumour labels are merged into one binary whole-tumour target.

The full dataset is excluded from GitHub because of its size.

### Model input

Training uses fixed 3D windows:

```text
[4, 32, 160, 160]
```

where:

```text
4   = MRI modalities
32  = consecutive slices
160 = height
160 = width
```

### Main training settings

```text
Depth:                    32
Image size:               160 × 160
Validation split:         20% by patient
Training windows/patient: 4
Tumour-window probability: 0.75
Validation stride:        16
Maximum epochs:           80
Learning rate:            2e-4
```


### Saved 3D results

| Metric | Result |
|---|---:|
| Dice | **0.9326** |
| Mean IoU | 0.8737 |
| Precision | 0.9368 |
| Recall | 0.9284 |
| Threshold | 0.55 |
| Best checkpoint epoch | 35 |

The 3D result is an internal validation result using a patient-level split and full-depth overlapping validation windows. It is not an official hidden BraTS challenge score.

### 3D files

```text
dataset_3d.py
dataset_attention_3d.py
models/unet3d.py
models/attention_unet3d.py
train_3d.py
train_attention_3d.py
convert_brats2021_to_h5.py
full_volume_3d.py

best_model_3d/
├── best_attention_unet3d.pth
├── best_attention_metrics_3d.json
└── best_attention_history_3d.csv
```

---

## Repository structure

```text
NeuroScan-AI/
│
├── README.md
├── requirements.txt
├── ui.py
│
├── 2D pipeline
│   ├── dataset.py
│   ├── train.py
│   ├── predict.py
│   ├── gradcam.py
│   └── models/unet.py
│
├── 3D pipeline
│   ├── dataset_3d.py
│   ├── dataset_attention_3d.py
│   ├── train_3d.py
│   ├── train_attention_3d.py
│   ├── full_volume_3d.py
│   ├── convert_brats2021_to_h5.py
│   ├── models/unet3d.py
│   └── models/attention_unet3d.py
│
├── best_model/
├── best_model_3d/
├── sample_data/
└── tools/
```

> The 2D and 3D labels above are logical groupings. The source files remain in their existing repository locations so current imports and deployment paths continue to work.

---

## Installation

```bash
git clone https://github.com/SuhailaElrefaie/NeuroScan-AI.git
cd NeuroScan-AI

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Training

### 2D

```bash
python3 train.py
```

### 3D

```bash
python3 train_3d.py
```

Attention-model training:

```bash
python3 train_attention_3d.py --h5-dir /path/to/BraTS2021_h5/content/data
```

---

## Verify the project

```bash
python3 tools/verify_project.py
```

---

## Technology stack

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Pillow
- OpenCV
- h5py
- nibabel
- scikit-image
- SciPy
- Git / GitHub

---

## Limitations

- NeuroScan AI is an academic prototype and is not a medical device.
- It must not be used for diagnosis or clinical decision-making.
- The 3D task performs binary whole-tumour segmentation rather than separate BraTS tumour subregions.
- Validation results are internal project results and have not been externally validated on an independent clinical cohort.

---

## Author

Developed for the **Individual Software Project, Charles University**.

Supervisor: **Kassem Anis**
