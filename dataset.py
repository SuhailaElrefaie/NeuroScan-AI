import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class BrainTumorSegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir, image_size=(256, 256), augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment

        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        mask_files = [
            f for f in os.listdir(mask_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        image_map = {os.path.splitext(f)[0]: f for f in image_files}
        mask_map = {os.path.splitext(f)[0]: f for f in mask_files}
        common_keys = sorted(set(image_map.keys()) & set(mask_map.keys()))

        self.pairs = [(image_map[key], mask_map[key]) for key in common_keys]

    def transform(self, image, mask): 
        image = TF.resize(image, self.image_size)
        mask = TF.resize(
            mask,
            self.image_size,
            interpolation=T.InterpolationMode.NEAREST
        )

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.5], std=[0.5])

        mask = np.array(mask)
        mask = (mask >= 128).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]

        image = Image.open(
            os.path.join(self.image_dir, img_name)
        ).convert("L")

        mask = Image.open(
            os.path.join(self.mask_dir, mask_name)
        ).convert("L")

        return self.transform(image, mask)
