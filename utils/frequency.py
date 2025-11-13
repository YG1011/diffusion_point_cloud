"""Frequency-domain utilities for point cloud diffusion."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


def dft_3d(voxel_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the 3-D discrete Fourier transform of a voxel grid."""

    if voxel_grid.dim() < 3:
        raise ValueError("voxel_grid must have at least 3 dimensions")

    dims = (-3, -2, -1)
    frequency = torch.fft.fftn(voxel_grid, dim=dims, norm="ortho")
    frequency = torch.fft.fftshift(frequency, dim=dims)
    amplitude = torch.abs(frequency)
    phase = torch.angle(frequency)
    return amplitude, phase


def idft_3d(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Inverse transform that reconstructs a voxel grid from amplitude/phase."""

    complex_spectrum = torch.polar(amplitude, phase)
    dims = (-3, -2, -1)
    complex_spectrum = torch.fft.ifftshift(complex_spectrum, dim=dims)
    spatial = torch.fft.ifftn(complex_spectrum, dim=dims, norm="ortho")
    return spatial.real


class FrequencyMaskGenerator(nn.Module):
    """Generates low-pass masks used for amplitude/phase mixing."""

    def __init__(
        self,
        grid_size: Tuple[int, int, int] = (64, 64, 64),
        amplitude_radius: Optional[float] = None,
        phase_radius: Optional[float] = None,
    ) -> None:
        super().__init__()
        if len(grid_size) != 3:
            raise ValueError("grid_size must contain exactly three integers")
        self.grid_size = tuple(int(v) for v in grid_size)
        self.amplitude_radius = amplitude_radius
        self.phase_radius = amplitude_radius if phase_radius is None else phase_radius

        amplitude_mask = self._build_mask(self.grid_size, self.amplitude_radius)
        phase_mask = self._build_mask(self.grid_size, self.phase_radius)

        self.register_buffer("_amplitude_mask", amplitude_mask, persistent=False)
        self.register_buffer("_phase_mask", phase_mask, persistent=False)

        # Store the mask coverage so downstream modules can gauge how strong
        # the frequency guidance should be. Hard replacement of the masked
        # spectrum proved brittle when the voxel statistics were noisy, so we
        # expose the average mask weight as a lightweight heuristic for
        # blending in the spatial domain.
        self.register_buffer(
            "_amplitude_fraction",
            amplitude_mask.float().mean(),
            persistent=False,
        )

    @staticmethod
    def _radial_grid(grid_size: Tuple[int, int, int]) -> torch.Tensor:
        device = torch.device("cpu")
        dtype = torch.float32
        d, h, w = grid_size
        zs = torch.arange(-(d // 2), d - d // 2, device=device, dtype=dtype)
        ys = torch.arange(-(h // 2), h - h // 2, device=device, dtype=dtype)
        xs = torch.arange(-(w // 2), w - w // 2, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
        distance = torch.sqrt(zz**2 + yy**2 + xx**2)
        return distance

    @classmethod
    def _build_mask(
        cls,
        grid_size: Tuple[int, int, int],
        radius: Optional[float],
    ) -> torch.Tensor:
        distance = cls._radial_grid(grid_size)
        if radius is None or radius <= 0:
            mask = torch.zeros_like(distance)
        else:
            mask = (distance <= radius).to(distance.dtype)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns amplitude and phase masks."""

        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32

        h_a = self._amplitude_mask.to(device=device, dtype=dtype)
        h_p = self._phase_mask.to(device=device, dtype=dtype)
        return h_a, h_p

    @property
    def amplitude_fraction(self) -> float:
        """Returns the mean weight of the amplitude mask."""

        return float(self._amplitude_fraction)


def amplitude_exchange(
    predicted: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Blends amplitude spectra using a binary/soft mask."""

    return mask * reference + (1.0 - mask) * predicted


def _wrap_to_pi(phase: torch.Tensor) -> torch.Tensor:
    return torch.remainder(phase + torch.pi, 2 * torch.pi) - torch.pi


def phase_projection(
    predicted: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor,
    delta: Optional[float] = None,
) -> torch.Tensor:
    """Aligns the low-frequency phase of ``predicted`` with ``reference``."""

    delta_phase = _wrap_to_pi(reference - predicted)
    if delta is not None:
        delta_phase = torch.clamp(delta_phase, min=-delta, max=delta)
    projected = predicted + mask * delta_phase
    return _wrap_to_pi(projected)


class FrequencyGuidance(nn.Module):
    """Applies frequency-domain guidance during diffusion sampling."""

    def __init__(
        self,
        mask_generator: FrequencyMaskGenerator,
        phase_delta: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.mask_generator = mask_generator
        self.phase_delta = phase_delta

    def forward(
        self,
        predicted_voxel: torch.Tensor,
        reference_voxel: torch.Tensor,
    ) -> torch.Tensor:
        amplitude_pred, phase_pred = dft_3d(predicted_voxel)
        amplitude_ref, phase_ref = dft_3d(reference_voxel)
        h_a, h_p = self.mask_generator(
            device=predicted_voxel.device, dtype=predicted_voxel.dtype
        )
        amplitude_guided = amplitude_exchange(amplitude_pred, amplitude_ref, h_a)
        phase_guided = phase_projection(phase_pred, phase_ref, h_p, self.phase_delta)
        guided_voxel = idft_3d(amplitude_guided, phase_guided)

        # Project the reconstructed signal back to the non-negative range used
        # by the voxel occupancies. Negative activations create artefacts when
        # sampling points, so we drop them while keeping track of the overall
        # energy so that we can later restore it.
        guided_voxel = guided_voxel.clamp_min(0.0)

        # Preserve the global statistics of the model prediction. Matching the
        # total occupancy prevents the guidance from collapsing the voxel mass
        # to a tiny subset of cells, which previously made the devoxelisation
        # degrade into almost uniform noise.
        pred_sum = predicted_voxel.sum(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-6)
        guided_sum = guided_voxel.sum(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-6)
        guided_voxel = guided_voxel * pred_sum / guided_sum

        peak_pred = predicted_voxel.amax(dim=(-3, -2, -1), keepdim=True)
        peak_guided = guided_voxel.amax(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-6)
        guided_voxel = guided_voxel * peak_pred / peak_guided

        # Blend the guided reconstruction with the network prediction in the
        # spatial domain. Rather than fully replacing the low-frequency
        # content, we interpolate using the fraction of spectrum that is being
        # modified. This keeps the update gentle when only a small subset of
        # frequencies is constrained.
        blend = predicted_voxel.new_tensor(self.mask_generator.amplitude_fraction)
        guided_voxel = predicted_voxel + blend * (guided_voxel - predicted_voxel)

        # The previous rescaling can push a few voxels slightly above the
        # original dynamic range. Clamping brings the tensor back to a valid
        # occupancy field while keeping a small epsilon to avoid returning an
        # entirely empty grid.
        guided_voxel = guided_voxel.clamp(min=0.0)

        return guided_voxel

