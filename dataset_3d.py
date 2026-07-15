import os
import re
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class BraTS3DVolumeDataset(Dataset):
    """
    Loads BraTS H5 slice files and groups them into 3D MRI volumes.

    Expected filenames:
        volume_1_slice_0.h5
        volume_1_slice_1.h5
        volume_1_slice_2.h5

    Each H5 file should contain:
        image
        mask

    Output:
        image tensor: [channels, depth, height, width]
        mask tensor:  [1, depth, height, width]
    """

    def __init__(
        self,
        h5_dir,
        depth=32,
        image_size=(128, 128),
        augment=False,
        only_tumor_windows=True
    ):
        self.h5_dir = Path(h5_dir)
        self.depth = depth
        self.image_size = image_size
        self.augment = augment
        self.only_tumor_windows = only_tumor_windows

        self.volumes = self._group_files_by_volume()

        if len(self.volumes) == 0:
            raise ValueError(f"No H5 volume files found in: {h5_dir}")

        self.volume_ids = sorted(self.volumes.keys())

        print("3D volumes found:", len(self.volume_ids))

    def _parse_filename(self, filename):
        match = re.match(r"volume_(\d+)_slice_(\d+)\.h5", filename)

        if match is None:
            return None

        volume_id = int(match.group(1))
        slice_id = int(match.group(2))

        return volume_id, slice_id

    def _group_files_by_volume(self):
        volumes = {}

        for file in os.listdir(self.h5_dir):
            if not file.endswith(".h5"):
                continue

            parsed = self._parse_filename(file)

            if parsed is None:
                continue

            volume_id, slice_id = parsed

            if volume_id not in volumes:
                volumes[volume_id] = []

            volumes[volume_id].append((slice_id, self.h5_dir / file))

        for volume_id in volumes:
            volumes[volume_id] = sorted(volumes[volume_id], key=lambda x: x[0])

        return volumes

    def _normalize_image(self, image):
        image = image.astype(np.float32)
        image = np.nan_to_num(image)

        mean = image.mean()
        std = image.std()

        if std < 1e-8:
            return image * 0.0

        return (image - mean) / std

    def _load_full_volume(self, volume_id):
        slice_files = self.volumes[volume_id]

        image_slices = []
        mask_slices = []

        for _, path in slice_files:
            with h5py.File(path, "r") as f:
                image = f["image"][()]
                mask = f["mask"][()]

            image = np.asarray(image)

            # BraTS H5 usually stores image as [H, W, C], commonly C=4.
            # Convert it to [C, H, W].
            if image.ndim == 2:
                image = image[None, :, :]
            elif image.ndim == 3:
                image = np.moveaxis(image, -1, 0)
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")

            mask = np.asarray(mask)

            # Some BraTS H5 masks are [H, W].
            # Some are [H, W, C] with multiple tumor channels/classes.
            # We collapse all mask channels into one binary tumor mask:
            # tumor = any non-zero value in any class/channel.
            if mask.ndim == 3:
                mask = np.any(mask > 0, axis=-1)
            elif mask.ndim == 2:
                mask = mask > 0
            else:
                raise ValueError(f"Unsupported mask shape: {mask.shape}")

            mask = mask.astype(np.float32)

            image_slices.append(image)
            mask_slices.append(mask)

        # image_slices: list of [C, H, W]
        # stack -> [D, C, H, W]
        # moveaxis -> [C, D, H, W]
        image_volume = np.stack(image_slices, axis=0)
        image_volume = np.moveaxis(image_volume, 1, 0)

        # mask_slices: list of [H, W]
        # stack -> [D, H, W]
        # add channel -> [1, D, H, W]
        mask_volume = np.stack(mask_slices, axis=0)
        mask_volume = mask_volume[None, :, :, :]

        image_volume = self._normalize_image(image_volume)

        image_tensor = torch.from_numpy(image_volume).float()
        mask_tensor = torch.from_numpy(mask_volume).float()

        return image_tensor, mask_tensor

    def _choose_depth_window(self, image, mask):
        total_depth = image.shape[1]

        if total_depth <= self.depth:
            pad_needed = self.depth - total_depth
            image = F.pad(image, (0, 0, 0, 0, 0, pad_needed))
            mask = F.pad(mask, (0, 0, 0, 0, 0, pad_needed))
            return image, mask

        if self.only_tumor_windows and mask.sum() > 0:
            tumor_depths = torch.where(mask[0].sum(dim=(1, 2)) > 0)[0]

            if len(tumor_depths) > 0:
                center = int(tumor_depths[len(tumor_depths) // 2])
                start = center - self.depth // 2
                start = max(0, min(start, total_depth - self.depth))
            else:
                start = random.randint(0, total_depth - self.depth)
        else:
            start = random.randint(0, total_depth - self.depth)

        end = start + self.depth

        image = image[:, start:end, :, :]
        mask = mask[:, start:end, :, :]

        return image, mask

    def _resize_volume(self, image, mask):
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        image = F.interpolate(
            image,
            size=(self.depth, self.image_size[0], self.image_size[1]),
            mode="trilinear",
            align_corners=False
        )

        mask = F.interpolate(
            mask,
            size=(self.depth, self.image_size[0], self.image_size[1]),
            mode="nearest"
        )

        return image.squeeze(0), mask.squeeze(0)

    def _augment(self, image, mask):
        if not self.augment:
            return image, mask

        if random.random() > 0.5:
            image = torch.flip(image, dims=[3])
            mask = torch.flip(mask, dims=[3])

        if random.random() > 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])

        return image, mask

    def __len__(self):
        return len(self.volume_ids)

    def __getitem__(self, idx):
        volume_id = self.volume_ids[idx]

        image, mask = self._load_full_volume(volume_id)
        image, mask = self._choose_depth_window(image, mask)
        image, mask = self._resize_volume(image, mask)
        image, mask = self._augment(image, mask)

        return image, mask