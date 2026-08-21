from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


_FILENAME_RE = re.compile(r"volume_(\d+)_slice_(\d+)\.h5$")


def discover_volumes(h5_dir: str | Path) -> Dict[int, List[Tuple[int, Path]]]:
    root = Path(h5_dir)
    if not root.exists():
        raise FileNotFoundError(f"3D dataset folder not found: {root.resolve()}")

    volumes: Dict[int, List[Tuple[int, Path]]] = {}
    for path in root.glob("volume_*_slice_*.h5"):
        match = _FILENAME_RE.match(path.name)
        if not match:
            continue
        volume_id = int(match.group(1))
        slice_id = int(match.group(2))
        volumes.setdefault(volume_id, []).append((slice_id, path))

    for volume_id in volumes:
        volumes[volume_id].sort(key=lambda item: item[0])

    if not volumes:
        raise RuntimeError(f"No matching H5 files found in: {root.resolve()}")
    return volumes


def split_volume_ids(
    volume_ids: Sequence[int],
    val_fraction: float = 0.20,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    ids = list(sorted(volume_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)

    val_count = max(1, int(round(len(ids) * val_fraction)))
    val_ids = sorted(ids[:val_count])
    train_ids = sorted(ids[val_count:])

    if not train_ids:
        raise ValueError("Training split is empty.")
    return train_ids, val_ids


def _read_mask(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as file:
        mask = np.asarray(file["mask"][()])
    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=-1)
    elif mask.ndim == 2:
        mask = mask > 0
    else:
        raise ValueError(f"Unsupported mask shape {mask.shape} in {path}")
    return mask.astype(np.uint8)


def _read_image_and_mask(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as file:
        image = np.asarray(file["image"][()])
        mask = np.asarray(file["mask"][()])

    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape {image.shape} in {path}")

    if image.shape[-1] <= 8:
        image = np.moveaxis(image, -1, 0)
    elif image.shape[0] > 8:
        raise ValueError(f"Cannot identify channel axis for {image.shape} in {path}")

    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=-1)
    elif mask.ndim == 2:
        mask = mask > 0
    else:
        raise ValueError(f"Unsupported mask shape {mask.shape} in {path}")

    return image.astype(np.float32), mask.astype(np.float32)


def normalize_modalities_nonzero(image: np.ndarray) -> np.ndarray:
    image = np.nan_to_num(image.astype(np.float32), copy=False)
    result = np.zeros_like(image, dtype=np.float32)

    for channel in range(image.shape[0]):
        modality = image[channel]
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


class BraTSWindowDataset(Dataset):
    """
    Efficient patch dataset for converted BraTS H5 slices.

    Training:
      - multiple windows per patient each epoch
      - mostly tumour-containing windows, plus random background/context windows

    Validation:
      - deterministic overlapping windows covering the complete patient depth
      - gives a much more honest validation score than tumour-only validation
    """

    def __init__(
        self,
        h5_dir: str | Path,
        volume_ids: Iterable[int],
        depth: int = 32,
        image_size: Tuple[int, int] = (160, 160),
        training: bool = True,
        samples_per_volume: int = 4,
        tumour_probability: float = 0.75,
        val_stride: int = 16,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.h5_dir = Path(h5_dir)
        self.depth = int(depth)
        self.image_size = tuple(image_size)
        self.training = bool(training)
        self.samples_per_volume = int(samples_per_volume)
        self.tumour_probability = float(tumour_probability)
        self.val_stride = int(val_stride)
        self.augment = bool(augment and training)
        self.seed = int(seed)

        all_volumes = discover_volumes(self.h5_dir)
        requested = set(int(v) for v in volume_ids)
        self.volumes = {k: v for k, v in all_volumes.items() if k in requested}

        missing = requested.difference(self.volumes)
        if missing:
            print(f"Warning: {len(missing)} requested volume IDs were not found.")

        self.volume_ids = sorted(self.volumes)
        if not self.volume_ids:
            raise RuntimeError("No volumes available for this dataset split.")

        print(
            f"Scanning masks for {'training' if training else 'validation'} split "
            f"({len(self.volume_ids)} patients)..."
        )
        self.tumour_positions: Dict[int, List[int]] = {}
        self.valid_files: Dict[int, List[Tuple[int, Path]]] = {}

        for volume_id in self.volume_ids:
            valid: List[Tuple[int, Path]] = []
            tumour_indices: List[int] = []

            for slice_id, path in self.volumes[volume_id]:
                try:
                    mask = _read_mask(path)
                except (OSError, KeyError, ValueError) as exc:
                    print(f"Skipping unreadable H5 file: {path} ({exc})")
                    continue

                valid.append((slice_id, path))
                if mask.any():
                    tumour_indices.append(len(valid) - 1)

            if valid:
                self.valid_files[volume_id] = valid
                self.tumour_positions[volume_id] = tumour_indices

        self.volume_ids = [v for v in self.volume_ids if v in self.valid_files]
        if not self.volume_ids:
            raise RuntimeError("All H5 volumes were unreadable.")

        self.val_windows: List[Tuple[int, int]] = []
        if not self.training:
            for volume_id in self.volume_ids:
                total = len(self.valid_files[volume_id])
                starts = self._covering_starts(total)
                self.val_windows.extend((volume_id, start) for start in starts)

        print(
            f"Prepared {'training' if training else 'validation'} dataset: "
            f"{len(self.volume_ids)} patients, {len(self)} windows"
        )

    def _covering_starts(self, total_depth: int) -> List[int]:
        if total_depth <= self.depth:
            return [0]

        last = total_depth - self.depth
        starts = list(range(0, last + 1, self.val_stride))
        if starts[-1] != last:
            starts.append(last)
        return starts

    def __len__(self) -> int:
        if self.training:
            return len(self.volume_ids) * self.samples_per_volume
        return len(self.val_windows)

    def _choose_training_start(self, volume_id: int, rng: random.Random) -> int:
        total = len(self.valid_files[volume_id])
        if total <= self.depth:
            return 0

        tumour = self.tumour_positions.get(volume_id, [])
        use_tumour = bool(tumour) and rng.random() < self.tumour_probability

        if use_tumour:
            centre = rng.choice(tumour)
            jitter = rng.randint(-self.depth // 4, self.depth // 4)
            start = centre - self.depth // 2 + jitter
        else:
            start = rng.randint(0, total - self.depth)

        return max(0, min(start, total - self.depth))

    def _load_window(self, volume_id: int, start: int) -> Tuple[np.ndarray, np.ndarray]:
        files = self.valid_files[volume_id]
        selected = files[start : start + self.depth]

        images: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        for _, path in selected:
            image, mask = _read_image_and_mask(path)
            images.append(image)
            masks.append(mask)

        image_volume = np.stack(images, axis=1)  # [C, D, H, W]
        mask_volume = np.stack(masks, axis=0)[None]  # [1, D, H, W]

        # Pad at end when a patient has fewer slices than requested depth.
        pad_depth = self.depth - image_volume.shape[1]
        if pad_depth > 0:
            image_volume = np.pad(
                image_volume,
                ((0, 0), (0, pad_depth), (0, 0), (0, 0)),
                mode="constant",
            )
            mask_volume = np.pad(
                mask_volume,
                ((0, 0), (0, pad_depth), (0, 0), (0, 0)),
                mode="constant",
            )

        image_volume = normalize_modalities_nonzero(image_volume)

        image_tensor = torch.from_numpy(image_volume).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_volume).float().unsqueeze(0)

        image_tensor = torch.nn.functional.interpolate(
            image_tensor,
            size=(self.depth, self.image_size[0], self.image_size[1]),
            mode="trilinear",
            align_corners=False,
        )
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor,
            size=(self.depth, self.image_size[0], self.image_size[1]),
            mode="nearest",
        )

        return image_tensor.squeeze(0), mask_tensor.squeeze(0)

    def _augment_pair(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        rng: random.Random,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Left-right flip is anatomically reasonable for this binary segmentation task.
        if rng.random() < 0.5:
            image = torch.flip(image, dims=[3])
            mask = torch.flip(mask, dims=[3])

        # Mild intensity augmentation; applied independently per channel.
        for channel in range(image.shape[0]):
            if rng.random() < 0.35:
                scale = rng.uniform(0.90, 1.10)
                shift = rng.uniform(-0.10, 0.10)
                image[channel] = image[channel] * scale + shift

        if rng.random() < 0.25:
            noise_std = rng.uniform(0.01, 0.04)
            image = image + torch.randn_like(image) * noise_std

        return image, mask

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Worker-aware changing seed so training windows vary across epochs/workers.
        worker = torch.utils.data.get_worker_info()
        worker_seed = worker.seed if worker is not None else torch.initial_seed()
        rng = random.Random(worker_seed + index * 1009)

        if self.training:
            volume_id = self.volume_ids[index % len(self.volume_ids)]
            start = self._choose_training_start(volume_id, rng)
        else:
            volume_id, start = self.val_windows[index]

        image, mask = self._load_window(volume_id, start)
        if self.augment:
            image, mask = self._augment_pair(image, mask, rng)

        return image, mask
