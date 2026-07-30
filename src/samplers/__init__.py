"""
Sampler module for different sampling methods.
Provides a unified interface for PDPS, DPS, and TV algorithms.
"""
from .base import BaseSampler
from .pdps import PDPSSampler
from .dps import DPSSampler
from .tv import TVSampler


SAMPLERS = {
    'pdps': PDPSSampler,
    'dps': DPSSampler,
    'tv': TVSampler,
}


def create_sampler(config, device):
    """
    Factory function to create appropriate sampler based on config.method_name.
    
    Args:
        config: Config instance
        device: torch device
    
    Returns:
        Sampler instance (PDPS, DPS, ...)
    """
    if config.method_name not in SAMPLERS:
        raise ValueError(
            f"Unknown method: {config.method_name}. "
            f"Choose from {list(SAMPLERS)}"
        )
    
    return SAMPLERS[config.method_name](config, device)


__all__ = [
    'BaseSampler',
    'PDPSSampler',
    'DPSSampler',
    'TVSampler',
    'SAMPLERS',
    'create_sampler',
]
