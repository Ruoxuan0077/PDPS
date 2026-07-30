"""
Forward operators for inverse problems.
Implements various degradation models: blur, noise, inpainting, etc.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils/bkse'))

import numpy as np
from torch.nn import functional as F
import yaml
import torch
from .utils.motionblur import Kernel
from .utils.img_utils import Blurkernel
from .utils.interpolation import deterministic_bilinear_resize
from .utils.bkse.models.kernel_encoding.kernel_wizard import KernelWizard


def _bilinear_resize(data, *, size=None, scale_factor=None):
    if torch.are_deterministic_algorithms_enabled():
        return deterministic_bilinear_resize(
            data,
            size=size,
            scale_factor=scale_factor,
        )
    return F.interpolate(
        data,
        size=size,
        scale_factor=scale_factor,
        mode='bilinear',
        align_corners=True,
    )


class BaseOperator:
    """Base class for forward operators."""

    def __init__(self, device):
        """
        Initialize operator.
        
        Args:
            device: torch device
        """
        self.device = device

    @staticmethod
    def _freeze_module(module):
        """Mark a physical forward model as fixed rather than trainable."""
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def forward(self, data, **kwargs):
        """
        Apply forward operator A(x).
        
        Args:
            data: input tensor
        
        Returns:
            Output tensor after applying operator
        """
        raise NotImplementedError("Subclass must implement forward()")


class MotionBlur(BaseOperator):
    """Motion blur operator."""
    
    def __init__(self, kernel_size, intensity, device, seed=42):
        super().__init__(device)
        self.kernel_size = kernel_size

        # The legacy Kernel implementation seeds NumPy globally. Preserve the
        # exact seeded kernel while restoring the caller's RNG state.
        numpy_state = np.random.get_state()
        try:
            self.conv = Blurkernel(
                blur_type='motion',
                kernel_size=kernel_size,
                std=intensity,
                device=device,
            ).to(device)
            self.kernel = Kernel(
                size=(kernel_size, kernel_size),
                intensity=intensity,
                seed=seed,
            )
            kernel = torch.tensor(
                self.kernel.kernelMatrix,
                dtype=torch.float32,
            )
        finally:
            np.random.set_state(numpy_state)

        self.conv.update_weights(kernel)
        self._freeze_module(self.conv)

    def forward(self, data, **kwargs):
        return self.conv(data)


class GaussianBlur(BaseOperator):
    """Gaussian blur operator."""
    
    def __init__(self, kernel_size, intensity, device):
        super().__init__(device)
        self.kernel_size = kernel_size
        self.conv = Blurkernel(
            blur_type='gaussian',
            kernel_size=kernel_size,
            std=intensity,
            device=device
        ).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))
        self._freeze_module(self.conv)

    def forward(self, data, **kwargs):
        return self.conv(data)


class NonlinearBlur(BaseOperator):
    """Nonlinear blur operator using kernel prediction network."""
    
    def __init__(
        self,
        opt_yml_path,
        device,
        kernel_scale=0.3,
        preserve_input_size=False,
    ):
        super().__init__(device)
        self.kernel_scale = kernel_scale
        self.preserve_input_size = preserve_input_size
        self.blur_model = self._prepare_model(opt_yml_path)
        self.random_kernel = self._get_random_kernel()
    
    def _prepare_model(self, opt_yml_path):
        """Load kernel prediction model."""
        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path))
        blur_model = blur_model.to(self.device)
        self._freeze_module(blur_model)
        return blur_model
    
    def _get_random_kernel(self, seed=7):
        """Generate random kernel."""
        local_generator = torch.Generator(device=self.device)
        if seed is not None:
            local_generator.manual_seed(seed)
        return torch.randn(1, 512, 1, 1, 
                          generator=local_generator, 
                          device=self.device) * self.kernel_scale
    
    def forward(self, data, **kwargs):
        input_size = data.shape[-2:]
        data = (data + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        data = _bilinear_resize(data, scale_factor=2)
        blurred = self.blur_model.adaptKernel(data, kernel=self.random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]
        if self.preserve_input_size:
            blurred = _bilinear_resize(blurred, size=input_size)
        return blurred


def get_operator(name, **kwargs):
    """
    Factory function to create operator.
    
    Args:
        name: operator name
        **kwargs: operator-specific parameters
    
    Returns:
        Operator instance
    """
    operators = {
        'motion_blur': MotionBlur,
        'gaussian_blur': GaussianBlur,
        'nonlinear_blur': NonlinearBlur,
    }
    
    if name not in operators:
        raise ValueError(f"Unknown operator: {name}")
    
    return operators[name](**kwargs)
