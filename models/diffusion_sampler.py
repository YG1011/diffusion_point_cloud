"""Diffusion sampler with frequency-domain guidance for point clouds."""

from __future__ import annotations

from typing import Optional

import torch

from .diffusion import VarianceSchedule
from utils.frequency import FrequencyGuidance
from utils.voxel import PointCloudToVoxel, VoxelToPointCloud, VoxelTransform


class DiffusionSampler:
    """Sampling controller implementing Algorithm 1 for 3D point clouds."""

    def __init__(
        self,
        model,
        var_sched: VarianceSchedule,
        pointcloud_to_voxel: PointCloudToVoxel,
        voxel_to_pointcloud: VoxelToPointCloud,
        frequency_guidance: FrequencyGuidance,
        forward_noise_steps: Optional[int] = None,
    ) -> None:
        self.model = model
        self.var_sched = var_sched
        self.pointcloud_to_voxel = pointcloud_to_voxel
        self.voxel_to_pointcloud = voxel_to_pointcloud
        self.frequency_guidance = frequency_guidance
        self.forward_noise_steps = forward_noise_steps

    def get_noised_x(
        self,
        x_adv: torch.Tensor,
        t: int,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies the forward diffusion process to ``x_adv`` at timestep ``t``."""

        if noise is None:
            noise = torch.randn_like(x_adv)

        batch_size = x_adv.size(0)
        alpha_bar = self.var_sched.alpha_bars[[t] * batch_size].to(
            device=x_adv.device, dtype=x_adv.dtype
        )
        c0 = torch.sqrt(alpha_bar).view(batch_size, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(batch_size, 1, 1)
        return c0 * x_adv + c1 * noise

    def _predict_x0(
        self, x_t: torch.Tensor, t: int, context: Optional[torch.Tensor]
    ) -> torch.Tensor:
        batch_size = x_t.size(0)
        beta = self.var_sched.betas[[t] * batch_size].to(
            device=x_t.device, dtype=x_t.dtype
        )
        eps_theta = self.model(x_t, beta=beta, context=context)
        alpha_bar = self.var_sched.alpha_bars[[t] * batch_size].to(
            device=x_t.device, dtype=x_t.dtype
        )
        c0 = 1.0 / torch.sqrt(alpha_bar).view(batch_size, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(batch_size, 1, 1)
        x0_t = c0 * (x_t - c1 * eps_theta)
        return x0_t

    @staticmethod
    def _predict_eps(x_t: torch.Tensor, x0: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
        batch_size = x_t.size(0)
        alpha_bar = alpha_bar.view(batch_size, 1, 1)
        return (x_t - torch.sqrt(alpha_bar) * x0) / torch.sqrt(1 - alpha_bar)

    def sample(
        self,
        x_adv: torch.Tensor,
        context: Optional[torch.Tensor],
        num_steps: Optional[int] = None,
        flexibility: float = 0.0,
        return_intermediates: bool = False,
    ):
        """Generates samples guided by the reference adversarial example."""

        device = x_adv.device
        batch_size, num_points, _ = x_adv.shape

        start_t = self.var_sched.num_steps if num_steps is None else int(num_steps)
        if start_t <= 0 or start_t > self.var_sched.num_steps:
            raise ValueError(
                "num_steps must be in the range [1, var_sched.num_steps]"
            )

        reference_voxel, reference_transform = self.pointcloud_to_voxel(x_adv)
        reference_voxel = reference_voxel.detach()
        reference_transform = VoxelTransform(
            mins=reference_transform.mins.detach(),
            ranges=reference_transform.ranges.detach(),
        )

        noise = torch.randn_like(x_adv)
        x_t = self.get_noised_x(x_adv, start_t, noise=noise)

        intermediates = {}
        for t in range(start_t, 0, -1):
            z = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
            alpha = self.var_sched.alphas[t].to(device=device, dtype=x_adv.dtype)
            alpha_bar_t = self.var_sched.alpha_bars[t].to(device=device, dtype=x_adv.dtype)
            sigma = self.var_sched.get_sigmas(t, flexibility)
            sigma = torch.as_tensor(sigma, device=device, dtype=x_adv.dtype).view(1, 1, 1)

            x0_pred = self._predict_x0(x_t, t, context)
            voxel_pred, _ = self.pointcloud_to_voxel(x0_pred, transform=reference_transform)
            guided_voxel = self.frequency_guidance(voxel_pred, reference_voxel)
            x0_guided = self.voxel_to_pointcloud(
                guided_voxel, reference_transform, num_points=num_points
            )

            eps = self._predict_eps(
                x_t,
                x0_guided,
                self.var_sched.alpha_bars[[t] * batch_size].to(device=device, dtype=x_adv.dtype),
            )
            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar_t)
            x_next = c0 * (x_t - c1 * eps) + sigma * z

            if return_intermediates:
                intermediates[t] = x_t.detach()

            x_t = x_next

        return (x_t, intermediates) if return_intermediates else x_t

