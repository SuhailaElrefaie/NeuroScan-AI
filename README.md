# NeuroScan AI

A small student project for AI-assisted brain tumor MRI segmentation.

The app uses a 2D U-Net for MRI slice segmentation and a 3D U-Net for MRI volume segmentation. It is built with Python, PyTorch, and Streamlit.

## What the app does

- Upload a 2D MRI image and get a predicted tumor mask
- Show a red overlay of the predicted tumor area
- Show a Grad-CAM heatmap for the 2D model
- Upload a 3D MRI volume as `.npz` and run 3D prediction
- Show slice results and an interactive 3D pixel view
- Show simple training metrics for the saved models

## Project files

```text
ui.py              # Streamlit interface
predict.py         # 2D prediction code
predict_3d.py      # 3D prediction code
train.py           # 2D training script
train_3d.py        # 3D training script
dataset.py         # 2D dataset loader
dataset_3d.py      # 3D BraTS H5 dataset loader
gradcam.py         # Grad-CAM helper
models/            # U-Net model files
sample_data/       # Small sample files for testing the website
```

## Datasets

The full datasets are not included in GitHub because they are too large.

Expected local 2D dataset path:

```text
Data/segmentation_task/train/images
Data/segmentation_task/train/masks
Data/segmentation_task/test/images
Data/segmentation_task/test/masks
```

Expected local 3D dataset path:

```text
archive/BraTS2020_training_data/content/data
```

The website includes a few small sample files so the app can be tested without downloading the full datasets.

## Run locally

Install the packages:

```bash
pip install -r requirements.txt
```

Run the app:

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

This is a prototype for an academic project. It is not a medical diagnosis tool.
