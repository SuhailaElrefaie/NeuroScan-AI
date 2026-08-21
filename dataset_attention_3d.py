from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


_FILENAME_RE = re.compile(r"volume_(\d+)_slice_(\d+)\.h5$")


@dataclass(frozen=True)
class VolumeRecord:
    volume_id: int
    files: Tuple[str, ...]
    tumour_positions: Tuple[int, ...]


class VolumeStore:
    def __init__(
        self,
        h5_dir: str | Path,
        *,
        depth: int = 32,
        image_size: Tuple[int, int] = (160, 160),
        cache_name: str = ".attention_volume_index.json",
        rebuild_index: bool = False,
    ) -> None:
        self.h5_dir = Path(h5_dir).expanduser().resolve()
        self.depth = int(depth)
        self.image_size = tuple(int(value) for value in image_size)
        self.cache_path = self.h5_dir / cache_name

        if not self.h5_dir.exists():
            raise FileNotFoundError(f"3D H5 dataset folder not found: {self.h5_dir}")

        self.records = self._load_or_build_index(rebuild_index)
        if not self.records:
            raise RuntimeError(f"No valid BraTS H5 volumes found in {self.h5_dir}")

        self.volume_ids = sorted(self.records)

    def _load_or_build_index(self, rebuild: bool) -> Dict[int, VolumeRecord]:
        if self.cache_path.exists() and not rebuild:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if payload.get("h5_dir") == str(self.h5_dir):
                records: Dict[int, VolumeRecord] = {}
                for item in payload.get("volumes", []):
                    volume_id = int(item["volume_id"])
                    records[volume_id] = VolumeRecord(
                        volume_id=volume_id,
                        files=tuple(item["files"]),
                        tumour_positions=tuple(int(v) for v in item["tumour_positions"]),
                    )
                if records:
                    print(f"Loaded cached volume index: {self.cache_path}")
                    return records

        grouped: Dict[int, List[Tuple[int, Path]]] = {}
        for path in self.h5_dir.glob("*.h5"):
            match = _FILENAME_RE.match(path.name)
            if match is None:
                continue
            volume_id = int(match.group(1))
            slice_id = int(match.group(2))
            grouped.setdefault(volume_id, []).append((slice_id, path))

        if not grouped:
            return {}

        print(f"Building patient index for {len(grouped)} volumes...")
        records: Dict[int, VolumeRecord] = {}
        bad_files: List[str] = []

        for number, volume_id in enumerate(sorted(grouped), start=1):
            ordered = sorted(grouped[volume_id], key=lambda pair: pair[0])
            valid_files: List[str] = []
            tumour_positions: List[int] = []

            for _, path in ordered:
                try:
                    with h5py.File(path, "r") as handle:
                        if "image" not in handle or "mask" not in handle:
                            raise KeyError("missing image or mask dataset")
                        mask = np.asarray(handle["mask"][()])
                    has_tumour = bool(np.any(mask > 0))
                except Exception as error:
                    bad_files.append(f"{path}: {error}")
                    continue

                position = len(valid_files)
                valid_files.append(path.name)
                if has_tumour:
                    tumour_positions.append(position)

            if valid_files:
                records[volume_id] = VolumeRecord(
                    volume_id=volume_id,
                    files=tuple(valid_files),
                    tumour_positions=tuple(tumour_positions),
                )

            if number % 50 == 0 or number == len(grouped):
                print(f"  indexed {number}/{len(grouped)} volumes", flush=True)

        payload = {
            "h5_dir": str(self.h5_dir),
            "volumes": [
                {
                    "volume_id": record.volume_id,
                    "files": list(record.files),
                    "tumour_positions": list(record.tumour_positions),
                }
                for record in records.values()
            ],
            "bad_files": bad_files,
        }
        self.cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if bad_files:
            bad_path = self.h5_dir / "attention_bad_h5_files.txt"
            bad_path.write_text("\n".join(bad_files), encoding="utf-8")
            print(f"Skipped {len(bad_files)} unreadable files. See {bad_path}")

        return records

    def split_ids(self, val_fraction: float = 0.20, seed: int = 42) -> Tuple[List[int], List[int]]:
        ids = self.volume_ids.copy()
        random.Random(seed).shuffle(ids)
        val_count = max(1, int(round(len(ids) * val_fraction)))
        return sorted(ids[val_count:]), sorted(ids[:val_count])

    def volume_depth(self, volume_id: int) -> int:
        return len(self.records[int(volume_id)].files)

    def window_starts(self, volume_id: int, stride: int = 16) -> List[int]:
        total_depth = self.volume_depth(volume_id)
        if total_depth <= self.depth:
            return [0]
        starts = list(range(0, total_depth - self.depth + 1, int(stride)))
        last = total_depth - self.depth
        if starts[-1] != last:
            starts.append(last)
        return starts

    def choose_training_start(
        self,
        volume_id: int,
        tumour_probability: float,
        rng: random.Random,
    ) -> int:
        record = self.records[int(volume_id)]
        total_depth = len(record.files)
        if total_depth <= self.depth:
            return 0

        max_start = total_depth - self.depth
        use_tumour = record.tumour_positions and rng.random() < tumour_probability
        if use_tumour:
            centre = rng.choice(record.tumour_positions)
            jitter = rng.randint(-self.depth // 4, self.depth // 4)
            return max(0, min(centre - self.depth // 2 + jitter, max_start))
        return rng.randint(0, max_start)

    @staticmethod
    def _read_slice(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(path, "r") as handle:
            image = np.asarray(handle["image"][()])
            mask = np.asarray(handle["mask"][()])

        if image.ndim == 2:
            image = image[..., None]
        if image.ndim != 3:
            raise ValueError(f"Unsupported image shape {image.shape} in {path}")

        if image.shape[-1] == 4:
            image = np.moveaxis(image, -1, 0)
        elif image.shape[0] != 4:
            if image.shape[-1] == 1:
                image = np.repeat(np.moveaxis(image, -1, 0), 4, axis=0)
            else:
                raise ValueError(f"Expected four MRI modalities in {path}, got {image.shape}")

        if mask.ndim == 3:
            mask = np.any(mask > 0, axis=-1)
        elif mask.ndim == 2:
            mask = mask > 0
        else:
            raise ValueError(f"Unsupported mask shape {mask.shape} in {path}")

        return image.astype(np.float32), mask.astype(np.float32)

    @staticmethod
    def _normalise_modalities(image: np.ndarray) -> np.ndarray:
        result = np.zeros_like(image, dtype=np.float32)
        for channel in range(image.shape[0]):
            values = image[channel]
            brain = values != 0
            if not np.any(brain):
                continue
            selected = values[brain]
            mean = float(selected.mean())
            std = float(selected.std())
            if std < 1e-6:
                std = 1.0
            result[channel, brain] = (selected - mean) / std
        return result

    def load_window(self, volume_id: int, start: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        record = self.records[int(volume_id)]
        end = min(int(start) + self.depth, len(record.files))
        selected = record.files[int(start):end]
        valid_depth = len(selected)
        if not selected:
            raise IndexError(f"Empty window for volume {volume_id} at start {start}")

        images: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        for filename in selected:
            image_slice, mask_slice = self._read_slice(self.h5_dir / filename)
            images.append(image_slice)
            masks.append(mask_slice)

        image = np.stack(images, axis=1)  # [C,D,H,W]
        mask = np.stack(masks, axis=0)[None, ...]  # [1,D,H,W]

        if valid_depth < self.depth:
            pad = self.depth - valid_depth
            image = np.pad(image, ((0, 0), (0, pad), (0, 0), (0, 0)), mode="edge")
            mask = np.pad(mask, ((0, 0), (0, pad), (0, 0), (0, 0)), mode="constant")

        image = self._normalise_modalities(image)
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        image_tensor = F.interpolate(
            image_tensor,
            size=(self.depth, self.image_size[0], self.image_size[1]),
            mode="trilinear",
            align_corners=False,
        )
        mask_tensor = F.interpolate(
            mask_tensor,
            size=(self.depth, self.image_size[0], self.image_size[1]),
            mode="nearest",
        )
        return image_tensor.squeeze(0), mask_tensor.squeeze(0), valid_depth


class AttentionTrainingDataset(Dataset):
    def __init__(
        self,
        store: VolumeStore,
        volume_ids: Sequence[int],
        *,
        samples_per_volume: int = 2,
        tumour_probability: float = 0.75,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        self.store = store
        self.volume_ids = list(int(value) for value in volume_ids)
        self.samples_per_volume = int(samples_per_volume)
        self.tumour_probability = float(tumour_probability)
        self.augment = bool(augment)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.volume_ids) * self.samples_per_volume

    def _augment(self, image: torch.Tensor, mask: torch.Tensor, rng: random.Random) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.augment:
            return image, mask

        if rng.random() < 0.5:
            image = torch.flip(image, dims=[3])
            mask = torch.flip(mask, dims=[3])
        if rng.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])

        if rng.random() < 0.35:
            scale = rng.uniform(0.90, 1.10)
            shift = rng.uniform(-0.10, 0.10)
            brain = image != 0
            image = torch.where(brain, image * scale + shift, image)

        if rng.random() < 0.20:
            image = image + torch.randn_like(image) * 0.025

        return image, mask

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        volume_id = self.volume_ids[index % len(self.volume_ids)]
        rng = random.Random(self.seed + index + random.randint(0, 1_000_000))
        start = self.store.choose_training_start(
            volume_id,
            self.tumour_probability,
            rng,
        )
        image, mask, _ = self.store.load_window(volume_id, start)
        return self._augment(image, mask, rng)
