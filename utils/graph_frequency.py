"""Graph Fourier transform utilities for point cloud diffusion guidance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class GraphSpectrum:
    """Container holding the eigendecomposition of a graph Laplacian."""

    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor


def _pairwise_distances(points: torch.Tensor) -> torch.Tensor:
    """Returns the pairwise Euclidean distance matrix for each batch sample."""

    if points.dim() != 3 or points.size(-1) != 3:
        raise ValueError("points must have shape (B, N, 3)")

    return torch.cdist(points, points, p=2)


def _build_knn_adjacency(
    points: torch.Tensor,
    k: int,
    bandwidth: Optional[float],
    eps: float,
) -> torch.Tensor:
    """Constructs a symmetric k-NN adjacency matrix with Gaussian weights."""

    batch_size, num_points, _ = points.shape
    if num_points <= 1 or k <= 0:
        return torch.zeros(
            batch_size,
            num_points,
            num_points,
            dtype=points.dtype,
            device=points.device,
        )

    distances = _pairwise_distances(points)
    k = min(k, num_points - 1)
    # torch.topk requires k > 0, hence the early return above.
    knn_dist, knn_idx = torch.topk(distances, k=k + 1, largest=False)
    knn_dist = knn_dist[:, :, 1:]
    knn_idx = knn_idx[:, :, 1:]

    if bandwidth is None:
        adaptive_bandwidth = knn_dist.mean(dim=(1, 2), keepdim=True)
    else:
        adaptive_bandwidth = points.new_full((batch_size, 1, 1), float(bandwidth))
    adaptive_bandwidth = adaptive_bandwidth.clamp_min(eps)

    weights = torch.exp(-(knn_dist**2) / (adaptive_bandwidth**2))

    adjacency = torch.zeros(
        batch_size,
        num_points,
        num_points,
        dtype=points.dtype,
        device=points.device,
    )
    adjacency.scatter_(2, knn_idx, weights)
    adjacency = 0.5 * (adjacency + adjacency.transpose(-1, -2))

    eye = torch.eye(num_points, dtype=points.dtype, device=points.device)
    adjacency = adjacency * (1 - eye.unsqueeze(0))

    return adjacency


def _laplacian_from_adjacency(
    adjacency: torch.Tensor,
    normalised: bool,
    eps: float,
) -> torch.Tensor:
    """Computes either the combinatorial or symmetric-normalised Laplacian."""

    degree = adjacency.sum(dim=-1)
    if normalised:
        inv_sqrt_degree = degree.clamp_min(eps).pow(-0.5)
        eye = torch.eye(
            adjacency.size(-1), device=adjacency.device, dtype=adjacency.dtype
        ).unsqueeze(0)
        laplacian = eye - (
            inv_sqrt_degree.unsqueeze(-1)
            * adjacency
            * inv_sqrt_degree.unsqueeze(-2)
        )
    else:
        laplacian = torch.diag_embed(degree) - adjacency
    return laplacian


def graph_spectrum(
    points: torch.Tensor,
    k: int = 16,
    bandwidth: Optional[float] = None,
    normalised: bool = True,
    eps: float = 1e-6,
) -> GraphSpectrum:
    """Computes the eigenvalues and eigenvectors of the graph Laplacian."""

    adjacency = _build_knn_adjacency(points, k=k, bandwidth=bandwidth, eps=eps)
    laplacian = _laplacian_from_adjacency(adjacency, normalised=normalised, eps=eps)
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    return GraphSpectrum(eigenvalues=eigenvalues, eigenvectors=eigenvectors)


def graph_fourier_transform(points: torch.Tensor, eigenvectors: torch.Tensor) -> torch.Tensor:
    """Projects 3D coordinates onto the graph Fourier basis."""

    return torch.matmul(eigenvectors.transpose(-1, -2), points)


def inverse_graph_fourier_transform(
    coefficients: torch.Tensor, eigenvectors: torch.Tensor
) -> torch.Tensor:
    """Reconstructs point coordinates from graph Fourier coefficients."""

    return torch.matmul(eigenvectors, coefficients)


class GraphFrequencyGuidance(nn.Module):
    """Guides diffusion updates using a graph Fourier decomposition."""

    def __init__(
        self,
        k: int = 16,
        lowpass_ratio: float = 0.125,
        bandwidth: Optional[float] = None,
        blend_weight: Optional[float] = None,
        normalised_laplacian: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if k < 0:
            raise ValueError("k must be non-negative")
        if lowpass_ratio <= 0:
            raise ValueError("lowpass_ratio must be positive")
        if blend_weight is not None and blend_weight < 0:
            raise ValueError("blend_weight must be non-negative when provided")

        self.k = int(k)
        self.lowpass_ratio = float(lowpass_ratio)
        self.bandwidth = bandwidth
        self.blend_weight = blend_weight
        self.normalised_laplacian = normalised_laplacian
        self.eps = float(eps)

    def forward(
        self,
        predicted_points: torch.Tensor,
        reference_points: torch.Tensor,
    ) -> torch.Tensor:
        """Replaces the low-frequency graph coefficients with the reference."""

        if predicted_points.shape != reference_points.shape:
            raise ValueError("predicted_points and reference_points must share shape")
        if predicted_points.dim() != 3 or predicted_points.size(-1) != 3:
            raise ValueError("point clouds must have shape (B, N, 3)")

        batch_size, num_points, _ = predicted_points.shape
        if num_points == 0:
            return predicted_points

        reference_center = reference_points.mean(dim=1, keepdim=True)
        predicted_center = predicted_points.mean(dim=1, keepdim=True)

        ref_centered = reference_points - reference_center
        pred_centered = predicted_points - predicted_center

        spectrum = graph_spectrum(
            ref_centered,
            k=self.k,
            bandwidth=self.bandwidth,
            normalised=self.normalised_laplacian,
            eps=self.eps,
        )
        eigenvectors = spectrum.eigenvectors

        ref_coeff = graph_fourier_transform(ref_centered, eigenvectors)
        pred_coeff = graph_fourier_transform(pred_centered, eigenvectors)

        if self.lowpass_ratio < 1.0:
            num_low = max(1, int(round(self.lowpass_ratio * num_points)))
        else:
            num_low = min(num_points, int(round(self.lowpass_ratio)))
        num_low = max(1, min(num_low, num_points))

        low_mask = (
            torch.arange(num_points, device=predicted_points.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            < num_low
        )
        coeff_guided = torch.where(low_mask, ref_coeff, pred_coeff)

        guided_centered = inverse_graph_fourier_transform(coeff_guided, eigenvectors)

        pred_energy = (
            pred_centered.pow(2).mean(dim=(1, 2), keepdim=True).clamp_min(self.eps)
        )
        guided_energy = (
            guided_centered.pow(2).mean(dim=(1, 2), keepdim=True).clamp_min(self.eps)
        )
        guided_centered = guided_centered * torch.sqrt(pred_energy / guided_energy)

        guided_points = guided_centered + predicted_center

        if self.blend_weight is not None:
            blend = float(self.blend_weight)
        else:
            blend = float(num_low) / float(num_points)
        blend_tensor = predicted_points.new_full((batch_size, 1, 1), blend)

        return predicted_points + blend_tensor * (guided_points - predicted_points)
