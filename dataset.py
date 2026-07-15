"""
Dataset loader for brain tumor MRI segmentation.

This file defines a PyTorch Dataset class that:
- Finds matching MRI image and mask files
- Loads both as grayscale images
- Resizes them to a fixed image size
- Optionally applies simple data augmentation
- Converts tumor masks into binary masks
"""

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class BrainTumorSegmentationDataset(Dataset): # PyTorch dataset for paired MRI images and tumor segmentation masks.
    """
    Images and masks are matched using the filename before the extension.
    """

    def __init__(self, image_dir, mask_dir, image_size=(256, 256), augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment

        # Read supported image files from both folders
        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        mask_files = [
            f for f in os.listdir(mask_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # Match images and masks by the filename without the extension
        image_map = {os.path.splitext(f)[0]: f for f in image_files}
        mask_map = {os.path.splitext(f)[0]: f for f in mask_files}
        common_keys = sorted(set(image_map.keys()) & set(mask_map.keys()))

        self.pairs = [(image_map[key], mask_map[key]) for key in common_keys]

    def transform(self, image, mask): # Apply preprocessing to an MRI image and its mask.
        """
        The image is:
        - resized
        - optionally augmented
        - converted to a tensor
        - normalized to the range expected by the model

        The mask is:
        - resized using nearest-neighbor interpolation
        - augmented in exactly the same way as the image
        - converted to a binary tensor, where tumor pixels are 1 and background is 0

        Returns:
            A tuple containing:
            - processed image tensor
            - binary mask tensor
        """

        # Resize image and mask to the same fixed size
        image = TF.resize(image, self.image_size)
        mask = TF.resize(
            mask,
            self.image_size,
            interpolation=T.InterpolationMode.NEAREST
        )

        # Apply random augmentation only when enabled.
        # The same flip is applied to both image and mask so they remain aligned.
        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        # Convert the MRI image to a normalized tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.5], std=[0.5])

        # Convert the mask to a binary tensor
        mask = np.array(mask)
        mask = (mask >= 128).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

    def __len__(self): # Return the number of matched image-mask pairs in the dataset.
        return len(self.pairs)

    def __getitem__(self, idx): # Load one MRI image and its matching segmentation mask.
        """
        Returns:
            A tuple containing:
            - processed MRI image tensor
            - processed binary mask tensor
        """
        img_name, mask_name = self.pairs[idx]

        image = Image.open(
            os.path.join(self.image_dir, img_name)
        ).convert("L")

        mask = Image.open(
            os.path.join(self.mask_dir, mask_name)
        ).convert("L")

        return self.transform(image, mask)