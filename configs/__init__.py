"""
Configuration system for sampling methods.
Extreme minimal design with shared base class.
"""
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod


# Shared configurations
TASKS = {
    'motion_deblur':    {'operator': {'name': 'motion_blur', 'kernel_size': 15, 'intensity': 0.5}},
    'gaussian_deblur':  {'operator': {'name': 'gaussian_blur', 'kernel_size': 15, 'intensity': 2.0}},
    'nonlinear_deblur': {'operator': {'name': 'nonlinear_blur', 'opt_yml_path': 'src/likelihood/utils/bkse/options/generate_blur/default.yml'}},
    'gaussian_deblur_fft': {'operator': {'name': 'gaussian_blur_fft', 'sigma_blur': 1.5, 'kernel_size': 15}},
    # 'inpainting':     {'operator': {'name': 'inpainting', 'mask_ratio': 0.9, 'img_size': 64}},
}

DEFAULT_PRIOR = {'name': 'edm', 'model_path': 'data/nn/edm/edm-ffhq-64x64-uncond-ve.pkl', 'sigma_ve': 0.09, 'sigma_final': 0.03}
DEFAULT_LIKELIHOOD = {'name': 'gaussian', 'sigma': 0.05}


@dataclass
class BaseConfig(ABC):
    """Base configuration with shared functionality."""
    task: str
    dataset: str
    image_id: Optional[str] = None
    num_samples: int = 1
    _is_paper: bool = False
    
    @property
    def mode(self):
        return 'single' if self.image_id else 'batch'
    
    @property
    @abstractmethod
    def method_name(self) -> str:
        pass
    
    @property
    def operator(self):
        return TASKS[self.task]['operator']
    
    @property
    def likelihood(self):
        return DEFAULT_LIKELIHOOD
    
    @property
    def prior(self):
        return DEFAULT_PRIOR
    
    def get_label_path(self):
        base = f"data/label/{self.dataset}"
        return f"{base}/{self.image_id}.png" if self.image_id else base
    
    def get_output_dir(self):
        """Universal output directory (same for all methods)."""
        task_name = self.operator['name']
        base = 'paper' if self._is_paper else 'custom'
        loc = f"{self.dataset}_{self.image_id}" if self.image_id else self.dataset
        return f"fig/{self.method_name}/{base}/{self.mode}/{task_name}/{loc}"


# Factory functions
from .pdps import PDPSConfig
from .dps import DPSConfig
from .pnp_flow import PnPFlowConfig

METHODS = {'pdps': PDPSConfig, 'dps': DPSConfig, 'pnp_flow': PnPFlowConfig}


def from_paper(method, dataset, mode, task, image_id=None):
    """Create config from paper preset."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    return METHODS[method].from_paper(dataset, mode, task, image_id)


def from_custom(method, task, dataset, image_id=None, num_samples=1, **overrides):
    """Create custom config."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    return METHODS[method].from_custom(task, dataset, image_id, num_samples, **overrides)


__all__ = ['from_paper', 'from_custom', 'PDPSConfig', 'DPSConfig', 'PnPFlowConfig', 'TASKS', 'DEFAULT_PRIOR', 'DEFAULT_LIKELIHOOD']
