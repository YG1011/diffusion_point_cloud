"""Spherical harmonic utilities for diffusion guidance.

A self-contained implementation that mirrors the earlier inlined helpers while
keeping the original GFT utilities available in :mod:`utils.graph_frequency`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyshtools as pysh
import torch
from cv2 import getGaussianKernel
from torch import nn


def _convert_pc_to_grid(pc: np.ndarray, lmax: int, device: str) -> tuple:
    """Project a point cloud onto a spherical grid centred at its centroid."""

    torch_device = torch.device(device)
    pc_tensor = torch.from_numpy(pc).to(torch_device)

    grid = pysh.SHGrid.from_zeros(lmax, grid="DH")
    nlon, nlat = grid.nlon, grid.nlat
    ngrid = nlon * nlat

    grid_lon = torch.from_numpy(
        np.linspace(0, nlon * np.pi * 2 / (nlon - 1), num=nlon, endpoint=False)
    ).to(torch_device)
    grid_lat = torch.from_numpy(
        np.linspace(nlat // 2 * np.pi / -(nlat - 1), np.pi / 2, num=nlat, endpoint=True)
    ).to(torch_device)
    grid_lon = grid_lon.view(1, 1, nlon).expand(1, nlat, nlon)
    grid_lat = grid_lat.view(1, nlat, 1).expand(1, nlat, nlon)
    grid_lon = grid_lon.reshape(1, ngrid)
    grid_lat = grid_lat.reshape(1, ngrid)

    origin = torch.mean(pc_tensor, axis=0)
    centred_pc = pc_tensor - origin
    npc = centred_pc.size(0)

    pc_x, pc_y, pc_z = centred_pc[:, 0], centred_pc[:, 1], centred_pc[:, 2]
    pc_r = torch.sqrt(pc_x * pc_x + pc_y * pc_y + pc_z * pc_z)
    pc_lat = torch.arcsin(pc_z / pc_r)
    pc_lon = torch.atan2(pc_y, pc_x)

    pc_r = pc_r.view(npc, 1)
    pc_lat = pc_lat.view(npc, 1)
    pc_lon = pc_lon.view(npc, 1)

    dist = (
        -torch.cos(grid_lat) * torch.cos(pc_lat) * torch.cos(grid_lon - pc_lon)
        + torch.sin(grid_lat) * torch.sin(pc_lat)
    )

    argmin = torch.argmin(dist, axis=0)
    grid_r = pc_r[argmin].view(nlat, nlon)
    grid.data = grid_r.to("cpu").numpy()

    argmin = torch.argmin(dist, axis=1)
    flag = torch.zeros(ngrid, dtype=bool)
    flag[argmin] = True
    flag = flag.to("cpu").numpy()

    return grid, flag, origin.to("cpu").numpy()


def _convert_grid_to_pc(grid: pysh.SHGrid, flag: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Reconstruct a point cloud from its spherical grid representation."""

    nlon = grid.nlon
    nlat = grid.nlat
    lon = np.linspace(0, nlon * np.pi * 2 / (nlon - 1), num=nlon, endpoint=False)
    lat = np.linspace(nlat // 2 * np.pi / -(nlat - 1), np.pi / 2, num=nlat, endpoint=True)
    lon = np.broadcast_to(lon.reshape((1, nlon)), (nlat, nlon))
    lat = np.broadcast_to(lat.reshape((nlat, 1)), (nlat, nlon))
    r = grid.data

    z = np.sin(lat) * r
    t = np.cos(lat) * r
    x = t * np.cos(lon)
    y = t * np.sin(lon)

    pc = np.zeros(grid.data.shape + (3,))
    pc[:, :, 0] = x
    pc[:, :, 1] = y
    pc[:, :, 2] = -z
    pc = pc.reshape((-1, 3))
    pc = pc[flag, :]
    pc += origin

    return pc


def _gaussian_weights(degree_count: int, sigma: float) -> np.ndarray:
    """Return positive Gaussian weights without zeroing any frequencies."""

    weights = getGaussianKernel(degree_count * 2 - 1, sigma)[degree_count - 1 :]
    weights = weights.squeeze()
    weights = weights / weights.max()
    return weights


def _duplicate_randomly(pc: np.ndarray, size: int) -> np.ndarray:
    """Pad the point cloud by duplicating random points if it shrank."""

    loss_cnt = size - pc.shape[0]
    if loss_cnt <= 0:
        return pc
    rand_indices = np.random.randint(0, pc.shape[0], size=loss_cnt)
    dup = pc[rand_indices]
    return np.concatenate((pc, dup))


def spherical_harmonic_transform(
    pc: np.ndarray, lmax: int, device: str
) -> tuple[pysh.SHCoeffs, np.ndarray, np.ndarray]:
    """Convert a point cloud to spherical harmonic coefficients without loss."""

    grid, flag, origin = _convert_pc_to_grid(pc, lmax, device)
    coeffs = grid.expand()
    return coeffs, flag, origin


def inverse_spherical_harmonic(
    coeffs: pysh.SHCoeffs, flag: np.ndarray, origin: np.ndarray
) -> np.ndarray:
    """Reconstruct a point cloud from spherical harmonic coefficients."""

    grid = coeffs.expand()
    return _convert_grid_to_pc(grid, flag, origin)


def spherical_harmonic_exchange(
    predicted_pc: np.ndarray,
    reference_pc: np.ndarray,
    lmax: int,
    sigma: float,
    blend_weight: float,
    target_size: int,
    device: str,
) -> np.ndarray:
    """Blend spectra without removing frequencies, inspired by Alg. 1."""

    ref_coeffs, flag, origin = spherical_harmonic_transform(reference_pc, lmax, device)
    pred_coeffs, _, _ = spherical_harmonic_transform(predicted_pc, lmax, device)

    weights = _gaussian_weights(ref_coeffs.coeffs.shape[1], sigma)

    ref_filtered = ref_coeffs.copy()
    ref_filtered.coeffs = ref_filtered.coeffs * weights[None, :, None]

    blended = pred_coeffs.copy()
    blended.coeffs = (1.0 - blend_weight) * pred_coeffs.coeffs + blend_weight * ref_filtered.coeffs

    guided_pc = inverse_spherical_harmonic(blended, flag, origin)
    guided_pc = _duplicate_randomly(guided_pc, target_size)
    return guided_pc


class SphericalHarmonicGuidance(nn.Module):
    """Guides diffusion updates using spherical harmonic spectrum exchange."""

    def __init__(
        self,
        lmax: int = 32,
        sigma: float = 1.5,
        blend_weight: Optional[float] = None,
        target_size: Optional[int] = None,
        **_: object,
    ) -> None:
        super().__init__()
        if lmax <= 0:
            raise ValueError("lmax must be positive for spherical harmonics")
        if sigma <= 0:
            raise ValueError("sigma must be positive for the Gaussian kernel")
        if blend_weight is not None and blend_weight < 0:
            raise ValueError("blend_weight must be non-negative when provided")

        self.lmax = int(lmax)
        self.sigma = float(sigma)
        self.blend_weight = 0.5 if blend_weight is None else float(blend_weight)
        self.target_size = target_size

    def forward(
        self, predicted_points: torch.Tensor, reference_points: torch.Tensor
    ) -> torch.Tensor:
        """Blend spectra of reference and predicted shapes without dropping modes."""

        if predicted_points.shape != reference_points.shape:
            raise ValueError("predicted_points and reference_points must share shape")
        if predicted_points.dim() != 3 or predicted_points.size(-1) != 3:
            raise ValueError("point clouds must have shape (B, N, 3)")

        batch_size, num_points, _ = predicted_points.shape
        device = predicted_points.device
        dtype = predicted_points.dtype

        guided_points = []
        for b in range(batch_size):
            ref_np = reference_points[b].detach().cpu().numpy()
            pred_np = predicted_points[b].detach().cpu().numpy()
            guided = spherical_harmonic_exchange(
                predicted_pc=pred_np,
                reference_pc=ref_np,
                lmax=self.lmax,
                sigma=self.sigma,
                blend_weight=self.blend_weight,
                target_size=self.target_size or num_points,
                device=str(device),
            )
            guided_points.append(torch.from_numpy(guided))

        guided_points = torch.stack(guided_points, dim=0).to(device=device, dtype=dtype)
        return guided_points
