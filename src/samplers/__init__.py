"""
Sampler module for different sampling methods.
Provides unified interface for PDPS and DPS algorithms.
"""
from .base import BaseSampler
from .pdps import PDPSSampler
from .dps import DPSSampler
from .pnp_flow import PnPFlowSampler


def create_sampler(config, device):
    """
    Factory function to create appropriate sampler based on config.method_name.
    
    Args:
        config: Config instance
        device: torch device
    
    Returns:
        Sampler instance (PDPS, DPS, ...)
    """
    samplers = {
        'pdps': PDPSSampler,
        'dps': DPSSampler,
        'pnp_flow': PnPFlowSampler,
    }
    
    if config.method_name not in samplers:
        raise ValueError(f"Unknown method: {config.method_name}. Choose from {list(samplers.keys())}")
    
    return samplers[config.method_name](config, device)


__all__ = ['BaseSampler', 'PDPSSampler', 'DPSSampler', 'PnPFlowSampler', 'create_sampler']

