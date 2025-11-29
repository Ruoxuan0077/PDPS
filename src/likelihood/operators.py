"""
Forward operators for inverse problems.
Implements various degradation models: blur, noise, inpainting, etc.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils/bkse'))

from functools import partial
from torch.nn import functional as F
import yaml
import torch
from .utils.motionblur import Kernel
from .utils.resizer import Resizer
from .utils.img_utils import Blurkernel
from .utils.bkse.models.kernel_encoding.kernel_wizard import KernelWizard


class BaseOperator:
    """Base class for forward operators."""
    
    def __init__(self, device):
        """
        Initialize operator.
        
        Args:
            device: torch device
        """
        self.device = device
    
    def forward(self, data, **kwargs):
        """
        Apply forward operator A(x).
        
        Args:
            data: input tensor
        
        Returns:
            Output tensor after applying operator
        """
        raise NotImplementedError("Subclass must implement forward()")


class Denoise(BaseOperator):
    """Identity operator (no degradation)."""
    
    def forward(self, data):
        return data


class SuperResolution(BaseOperator):
    """Super-resolution operator (downsampling)."""
    
    def __init__(self, in_shape, scale_factor, device):
        super().__init__(device)
        self.in_shape = in_shape
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)
    
    def forward(self, data, **kwargs):
        return self.down_sample(data)


class Inpainting(BaseOperator):
    """Inpainting operator (masking)."""
    
    def __init__(self, img_size, mask_ratio, device, seed=42):
        super().__init__(device)
        self.mask_ratio = mask_ratio
        self.img_size = img_size
        self.seed = seed
    
    def forward(self, data, **kwargs):
        total = self.img_size ** 2
        mask_vec = torch.ones([1, total])
        generator = torch.Generator().manual_seed(self.seed)
        samples = torch.randperm(total, generator=generator)[:int(total * self.mask_ratio)]
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.img_size, self.img_size)
        mask_b = mask_b.repeat(3, 1, 1)
        mask = torch.ones_like(data, device=data.device)
        mask[:, ...] = mask_b
        return data * mask


class MotionBlur(BaseOperator):
    """Motion blur operator."""
    
    def __init__(self, kernel_size, intensity, device, seed=42):
        super().__init__(device)
        self.kernel_size = kernel_size
        self.conv = Blurkernel(
            blur_type='motion',
            kernel_size=kernel_size,
            std=intensity,
            device=device
        ).to(device)
        
        self.kernel = Kernel(size=(kernel_size, kernel_size), 
                           intensity=intensity, seed=seed)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
    
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
    
    def forward(self, data, **kwargs):
        return self.conv(data)


class NonlinearBlur(BaseOperator):
    """Nonlinear blur operator using kernel prediction network."""
    
    def __init__(self, opt_yml_path, device):
        super().__init__(device)
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
        return blur_model
    
    def _get_random_kernel(self, seed=7):
        """Generate random kernel."""
        local_generator = torch.Generator(device=self.device)
        if seed is not None:
            local_generator.manual_seed(seed)
        return torch.randn(1, 512, 1, 1, 
                          generator=local_generator, 
                          device=self.device) * 0.3
    
    def forward(self, data, **kwargs):
        data = (data + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        data = F.interpolate(data, scale_factor=2, mode='bilinear', align_corners=True)
        blurred = self.blur_model.adaptKernel(data, kernel=self.random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]
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
        'denoise': Denoise,
        'super_resolution': SuperResolution,
        'inpainting': Inpainting,
        'motion_blur': MotionBlur,
        'gaussian_blur': GaussianBlur,
        'nonlinear_blur': NonlinearBlur,
    }
    
    if name not in operators:
        raise ValueError(f"Unknown operator: {name}")
    
    return operators[name](**kwargs)

