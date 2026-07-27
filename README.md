# NeuroScan AI

NeuroScan AI is an AI-assisted brain tumor MRI segmentation project built as an Individual Software Project at Charles University under the supervision of Kassem Anis.

The application uses a 2D U-Net for single MRI slice segmentation and a 3D U-Net for MRI volume segmentation. It is built with Python, PyTorch, and Streamlit.

## Live Demo

The app is deployed here:

https://neuroscan-ai-isp.streamlit.app

You can test the system directly in the browser by uploading a 2D MRI image or a 3D `.npz` MRI volume. Sample files are available inside the app sidebar.

## What the App Does

- Upload a 2D MRI image and generate a predicted tumor mask.
- Show a red overlay of the predicted tumor area.
- Show a Grad-CAM heatmap for the 2D model.
- Upload a 3D MRI volume as `.npz` and run 3D segmentation.
- Show 3D slice results including the MRI slice, predicted mask, overlay, and probability map.
- Show saved training metrics for the 2D and 3D models.

## Datasets Used

This project uses two public MRI datasets:

### 2D Dataset

**BRISC 2025** is used for 2D brain MRI image segmentation.

Dataset link:

https://www.kaggle.com/datasets/briscdataset/brisc2025/

### 3D Dataset

**BraTS 2021 Task 1** is used for 3D brain tumor MRI volume segmentation.

Dataset link:

https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1?select=BraTS2021_Training_Data.tar

The BraTS 2021 data was converted from NIfTI MRI volumes into `.h5` slice files for compatibility with the project 3D dataset loader.

The full datasets are not included in this repository because they are large. Only small sample files are included for testing the Streamlit website.

## Dataset Setup

Expected local 2D dataset path:

```text
Data/segmentation_task/train/images
Data/segmentation_task/train/masks
Data/segmentation_task/test/images
Data/segmentation_task/test/masks
```

Expected local 3D dataset path after conversion:

```text
archive/BraTS2021_h5/content/data
```

Each converted `.h5` slice contains:

```text
image  # MRI slice with 4 modality channels
mask   # binary tumor segmentation mask
```

## Model Overview

### 2D U-Net

The 2D model segments tumor regions from a single grayscale MRI slice.

- Input: 2D grayscale MRI image
- Output: binary tumor mask
- Visual outputs: segmentation overlay, mask, and Grad-CAM heatmap

### 3D U-Net

The 3D model segments tumor regions from a 4-channel MRI volume.

- Input: 4-channel MRI volume
- Output: 3D tumor mask
- Final training data: converted BraTS 2021 H5 dataset
- Final volume input size: 32 slices resized to 160 × 160

## Project Files

```text
ui.py                 # Streamlit web interface
predict.py            # 2D prediction code
predict_3d.py         # 3D prediction code
train.py              # 2D training script
train_3d.py           # 3D training script
dataset.py            # 2D dataset loader
dataset_3d.py         # 3D H5 dataset loader
gradcam.py            # Grad-CAM helper
models/               # U-Net model definitions
best_model/           # Saved best 2D model files
best_model_3d/        # Saved best 3D model files
sample_data/          # Small sample files for website testing
```

## Technologies Used

- Python
- PyTorch
- Streamlit
- NumPy
- Pandas
- Pillow
- OpenCV
- Plotly
- h5py
- nibabel

## Run Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run ui.py
```

## Training

Train the 2D model:

```bash
python train.py
```

Train the 3D model:

```bash
python train_3d.py
```

The app expects saved model files here:

```text
best_model/best_unet.pth
best_model_3d/best_unet3d.pth
```

## Notes

This project is an academic prototype. It is not a medical diagnosis tool and should not be used for clinical decision-making.
