"""Utility modules for converting between point clouds and voxel grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class VoxelTransform:
    """Stores the affine parameters used during voxelisation.

    Attributes:
        mins: Per-sample minimum xyz coordinate of the original point cloud.
        ranges: Per-sample xyz range (max - min) of the original point cloud.
    """

    mins: torch.Tensor
    ranges: torch.Tensor


class PointCloudToVoxel(nn.Module):
    """Converts batched point clouds to dense voxel occupancy grids.

    The module normalises each point cloud to the unit cube before assigning
    points to voxels. The normalisation statistics are returned so that the
    voxel representation can later be mapped back to the original coordinate
    system.
    """

    def __init__(self, grid_size: Tuple[int, int, int] = (64, 64, 64)) -> None:
        super().__init__()
        if len(grid_size) != 3:
            raise ValueError("grid_size must contain exactly three integers")
        self.grid_size = tuple(int(v) for v in grid_size)

    def forward(
        self,
        points: torch.Tensor,
        transform: Optional[VoxelTransform] = None,
    ) -> Tuple[torch.Tensor, VoxelTransform]:
        """Voxelises the given batched point clouds.

        Args:
            points: Tensor of shape ``(B, N, 3)`` containing xyz coordinates.

        Returns:
            voxel_grid: Tensor of shape ``(B, 1, D, H, W)`` with binary
                occupancies.
            transform: :class:`VoxelTransform` storing the normalisation stats.
        """

        if points.dim() != 3 or points.size(-1) != 3:
            raise ValueError("points must be of shape (B, N, 3)")

        if transform is None:
            mins = points.amin(dim=1, keepdim=True)
            maxs = points.amax(dim=1, keepdim=True)
            ranges = (maxs - mins).clamp_min_(1e-6)
        else:
            mins = transform.mins
            ranges = transform.ranges

        points_norm = (points - mins) / ranges
        points_norm = points_norm.clamp(0.0, 1.0 - 1e-6)

        d, h, w = self.grid_size
        scale = points_norm.new_tensor([d, h, w]) - 1e-6
        idx = points_norm * scale
        idx = torch.clamp(idx, min=0.0)
        idx = torch.minimum(idx, scale)
        idx = idx.long()

        batch_size, num_points, _ = points.shape
        voxel_grid = torch.zeros((batch_size, 1, d, h, w), dtype=points.dtype, device=points.device)
        flat = voxel_grid.view(batch_size, -1)
        linear_idx = (
            idx[..., 0] * (h * w)
            + idx[..., 1] * w
            + idx[..., 2]
        )
        values = torch.ones_like(linear_idx, dtype=points.dtype)
        flat.scatter_add_(1, linear_idx.view(batch_size, -1), values.view(batch_size, -1))
        voxel_grid = flat.view(batch_size, 1, d, h, w)

        normaliser = voxel_grid.amax(dim=(-3, -2, -1), keepdim=True).clamp_min_(1.0)
        voxel_grid = voxel_grid / normaliser

        return voxel_grid, VoxelTransform(mins=mins, ranges=ranges)


class VoxelToPointCloud(nn.Module):
    """Converts voxel grids back to point cloud representations."""

    def __init__(self, grid_size: Tuple[int, int, int] = (64, 64, 64)) -> None:
        super().__init__()
        if len(grid_size) != 3:
            raise ValueError("grid_size must contain exactly three integers")
        self.grid_size = tuple(int(v) for v in grid_size)

    def forward(
        self,
        voxels: torch.Tensor,
        transform: VoxelTransform,
        num_points: Optional[int] = None,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        """Reconstructs point clouds from voxel occupancies.

        Args:
            voxels: Tensor of shape ``(B, 1, D, H, W)``.
            transform: :class:`VoxelTransform` returned by
                :class:`PointCloudToVoxel`.
            num_points: Desired number of output points per sample. Defaults to
                all voxels if ``None``.
            threshold: Minimum occupancy for a voxel to be considered active.

        Returns:
            Tensor of shape ``(B, num_points, 3)`` containing xyz coordinates.
        """

        if voxels.dim() != 5 or voxels.size(1) != 1:
            raise ValueError("voxels must be of shape (B, 1, D, H, W)")

        batch_size = voxels.size(0)
        device = voxels.device
        dtype = voxels.dtype
        d, h, w = self.grid_size
        total_voxels = d * h * w

        if num_points is None:
            num_points = total_voxels

        flat = voxels.view(batch_size, total_voxels)
        outputs = []

        dz = h * w
        dy = w

        grid_size = torch.tensor([d, h, w], dtype=dtype, device=device)
        mins = transform.mins
        ranges = transform.ranges

        for b in range(batch_size):
            weights = flat[b]
            active = torch.nonzero(weights > threshold, as_tuple=False).squeeze(-1)

            if active.numel() == 0:
                active = torch.arange(total_voxels, device=device)
                active_weights = torch.ones_like(active, dtype=dtype)
            else:
                active_weights = weights[active]

            if active.numel() >= num_points:
                topk = torch.topk(active_weights, num_points, largest=True, sorted=False)
                chosen = active[topk.indices]
            else:
                chosen = active
                deficit = num_points - active.numel()
                probs = active_weights / active_weights.sum().clamp_min(1e-12)
                extra = active[torch.multinomial(probs, deficit, replacement=True)]
                chosen = torch.cat([chosen, extra], dim=0)

            iz = chosen // dz
            iy = (chosen % dz) // dy
            ix = chosen % dy

            coords = torch.stack([iz, iy, ix], dim=-1).to(dtype)
            jitter = torch.rand_like(coords)
            coords = (coords + jitter) / grid_size

            points = coords * ranges[b] + mins[b]
            outputs.append(points)

        return torch.stack(outputs, dim=0)

