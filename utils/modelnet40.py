"""Utilities for loading the ModelNet40 dataset from HDF5 files."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ModelNet40Stats:
    mean: torch.Tensor
    std: torch.Tensor


class ModelNet40(Dataset):
    """ModelNet40 dataset backed by the official HDF5 release.

    Parameters
    ----------
    root: str
        Directory containing the HDF5 files. The dataset expects the same
        folder structure as the official ``modelnet40_ply_hdf5_2048`` release.
    split: str
        ``"train"`` or ``"test"`` split.
    num_points: int
        Number of points to return for each shape. If the raw point cloud
        contains more points they will be randomly subsampled (without
        replacement); if it contains fewer points, points will be duplicated.
    normalize: bool
        If ``True`` shapes are zero-centred and scaled to lie inside the unit
        sphere.
    random_rotate: bool
        If ``True`` randomly permutes the order of the returned points. This is
        useful when the downstream model expects points to be in arbitrary
        order.
    seed: Optional[int]
        Random seed used for subsampling points.
    """

    classes: Sequence[str] = (
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl',
        'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser',
        'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop',
        'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant',
        'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table',
        'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
    )

    def __init__(
        self,
        root: str,
        split: str = 'test',
        num_points: int = 1024,
        normalize: bool = True,
        random_rotate: bool = False,
        seed: Optional[int] = 0,
    ) -> None:
        super().__init__()
        assert split in {'train', 'test'}
        self.root = root
        self.split = split
        self.num_points = num_points
        self.normalize = normalize
        self.random_rotate = random_rotate
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self._data, self._labels = self._load_split()
        self.dataset_name = "ModelNet40"

    # ------------------------------------------------------------------
    # Loading utilities
    # ------------------------------------------------------------------
    def _resolve_files(self) -> List[str]:
        pattern = os.path.join(self.root, f'ply_data_{self.split}*.h5')
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f'No HDF5 files matching pattern {pattern!r}. '
                'Please download the official ModelNet40 HDF5 release and '
                'place it inside the provided root directory.'
            )
        return files

    def _load_split(self) -> tuple[np.ndarray, np.ndarray]:
        data_parts: List[np.ndarray] = []
        label_parts: List[np.ndarray] = []
        for file_path in self._resolve_files():
            with h5py.File(file_path, 'r') as f:
                data_parts.append(f['data'][:])
                label_parts.append(f['label'][:])
        data = np.concatenate(data_parts, axis=0)
        labels = np.concatenate(label_parts, axis=0).squeeze().astype(np.int64)
        return data, labels

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._data.shape[0]

    def _sample_points(self, points: np.ndarray) -> np.ndarray:
        total_points = points.shape[0]
        if total_points == self.num_points:
            choice = np.arange(total_points)
        elif total_points > self.num_points:
            choice = self._rng.choice(total_points, self.num_points, replace=False)
        else:
            extra = self._rng.choice(total_points, self.num_points - total_points, replace=True)
            choice = np.concatenate([np.arange(total_points), extra], axis=0)
        sampled = points[choice]
        if self.random_rotate:
            self._rng.shuffle(sampled)
        return sampled

    @staticmethod
    def _normalise(points: np.ndarray) -> np.ndarray:
        points = points - np.mean(points, axis=0)
        scale = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if scale > 0:
            points = points / scale
        return points

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        points = self._data[idx]
        label = int(self._labels[idx])
        points = self._sample_points(points)
        if self.normalize:
            points = self._normalise(points)

        pointcloud = torch.from_numpy(points.astype(np.float32))
        cate = torch.tensor([label], dtype=torch.long)
        return {
            'pointcloud': pointcloud,
            'cate': cate,
        }


def compute_modelnet40_stats(dataset: ModelNet40) -> ModelNet40Stats:
    """Compute dataset-level mean and standard deviation."""
    stacked = torch.from_numpy(dataset._data.reshape(-1, 3).astype(np.float32))
    mean = stacked.mean(dim=0)
    std = stacked.std()
    return ModelNet40Stats(mean=mean, std=std)
