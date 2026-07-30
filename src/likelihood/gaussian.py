"""
Gaussian likelihood function.
Computes likelihood score for Gaussian measurement model.
"""
import math

import torch


class GaussianLikelihood:
    """
    Gaussian likelihood for inverse problems.
    Assumes measurement model: y = A(x) + noise, where noise ~ N(0, sigma^2 I).
    """
    
    def __init__(self, operator, sigma):
        """
        Initialize Gaussian likelihood.
        
        Args:
            operator: forward operator instance
            sigma: noise standard deviation
        """
        self.operator = operator
        self.device = operator.device
        if (
            isinstance(sigma, bool)
            or not isinstance(sigma, (int, float))
            or not math.isfinite(sigma)
            or sigma <= 0
        ):
            raise ValueError("Gaussian sigma must be positive and finite")
        self.sigma = float(sigma)

    def get_label(
        self,
        data,
        seed=42,
        noise_offset=0,
        noise_total=None,
    ):
        """
        Generate noisy measurement y = A(x) + noise.

        Args:
            data: clean data [B, C, H, W]
            seed: random seed for reproducibility
            noise_offset: global image offset of this batch partition
            noise_total: total images in the unpartitioned measurement batch

        Returns:
            Noisy measurement [B, C, H, W]
        """
        if data.ndim < 1 or data.shape[0] < 1:
            raise ValueError("Measurement data must have a nonempty batch axis")
        if type(noise_offset) is not int or noise_offset < 0:
            raise ValueError("noise_offset must be a non-negative integer")
        if noise_total is None:
            noise_total = noise_offset + data.shape[0]
        if (
            type(noise_total) is not int
            or noise_total < noise_offset + data.shape[0]
        ):
            raise ValueError(
                "noise_total must cover noise_offset plus the local batch"
            )
        if seed is not None and (
            type(seed) is not int or not 0 <= seed < 2**63
        ):
            raise ValueError("seed must be an integer in [0, 2**63) or None")

        data = data.to(self.device)
        operated = self.operator.forward(data)

        # Every partition generates the same full noise tensor and selects
        # its global slice. This preserves the old one-GPU sequence exactly
        # and makes measurement noise independent of GPU chunking.
        local_generator = torch.Generator(device=self.device)
        if seed is not None:
            local_generator.manual_seed(seed)
        full_noise = torch.randn(
            (noise_total, *operated.shape[1:]),
            generator=local_generator,
            device=self.device,
            dtype=operated.dtype,
        )
        noise = full_noise[
            noise_offset:noise_offset + operated.shape[0]
        ]

        return operated + self.sigma * noise

    def likelihood_fn(self, x0, y):
        """
        Compute likelihood score gradient.

        Args:
            x0: current state [B, C, H, W]
            y: measurement [B, C, H, W]

        Returns:
            Likelihood gradient [B, C, H, W]
        """
        x = x0.detach().to(self.device).requires_grad_(True)
        y = y.detach().to(self.device)

        # Compute data fidelity term
        Ax = self.operator.forward(x)
        diff = -torch.sum((y - Ax)**2, dim=tuple(range(1, len(x0.shape))))
        total_diff = diff.sum()
        gradient = torch.autograd.grad(
            total_diff,
            x,
            create_graph=False,
            retain_graph=False,
        )[0]

        return gradient / (2 * self.sigma**2)
