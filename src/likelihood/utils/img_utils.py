"""Convolution kernels used by the linear blur operators."""
import numpy as np
import scipy
import torch
from torch import nn

from .motionblur import Kernel


class DeterministicReflectionPad2d(nn.Module):
    """Reflection padding with a deterministic CUDA backward."""

    def __init__(self, padding):
        super().__init__()
        if type(padding) is not int or padding < 0:
            raise ValueError("padding must be a non-negative integer")
        self.padding = padding

    def forward(self, value):
        padding = self.padding
        if padding == 0:
            return value

        height, width = value.shape[-2:]
        if padding >= height or padding >= width:
            raise ValueError(
                f"Reflection padding {padding} must be smaller than "
                f"the input size {(height, width)}"
            )

        value = torch.cat(
            (
                value[..., 1:padding + 1].flip(-1),
                value,
                value[..., -padding - 1:-1].flip(-1),
            ),
            dim=-1,
        )
        return torch.cat(
            (
                value[..., 1:padding + 1, :].flip(-2),
                value,
                value[..., -padding - 1:-1, :].flip(-2),
            ),
            dim=-2,
        )


class DeterminismAwareReflectionPad2d(nn.Module):
    """Select padding at call time so strict mode cannot be bypassed."""

    def __init__(self, padding):
        super().__init__()
        self.native = nn.ReflectionPad2d(padding)
        self.deterministic = DeterministicReflectionPad2d(padding)

    def forward(self, value):
        if torch.are_deterministic_algorithms_enabled():
            return self.deterministic(value)
        return self.native(value)


class Blurkernel(nn.Module):
    """Depthwise RGB convolution with a fixed Gaussian or motion kernel."""

    def __init__(
        self,
        blur_type='gaussian',
        kernel_size=31,
        std=3.0,
        device=None,
    ):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            DeterminismAwareReflectionPad2d(self.kernel_size // 2),
            nn.Conv2d(
                3,
                3,
                self.kernel_size,
                stride=1,
                padding=0,
                bias=False,
                groups=3,
            ),
        )
        self.weights_init()

    def forward(self, value):
        return self.seq(value)

    def weights_init(self):
        if self.blur_type == 'gaussian':
            impulse = np.zeros((self.kernel_size, self.kernel_size))
            impulse[self.kernel_size // 2, self.kernel_size // 2] = 1
            kernel = scipy.ndimage.gaussian_filter(
                impulse,
                sigma=self.std,
            )
        elif self.blur_type == 'motion':
            kernel = Kernel(
                size=(self.kernel_size, self.kernel_size),
                intensity=self.std,
            ).kernelMatrix
        else:
            raise ValueError(f"Unknown blur type: {self.blur_type}")

        self.k = torch.from_numpy(kernel)
        for parameter in self.parameters():
            parameter.data.copy_(self.k)

    def update_weights(self, kernel):
        if not torch.is_tensor(kernel):
            kernel = torch.from_numpy(kernel).to(self.device)
        for parameter in self.parameters():
            parameter.data.copy_(kernel)

    def get_kernel(self):
        return self.k
