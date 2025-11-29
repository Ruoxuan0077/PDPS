"""
Base sampler interface.
All sampling methods must implement this interface.
"""
import torch


class BaseSampler:
    """
    Abstract base class for sampling algorithms.
    Defines the interface that all samplers must implement.
    """
    
    def __init__(self, config, device):
        """
        Initialize sampler with configuration.
        
        Args:
            config: Config instance
            device: torch device for computation
        """
        self.config = config
        self.device = device
    
    def sample(self, x, y, **kwargs):
        """
        Execute sampling algorithm and return final usable results.
        
        Args:
            x: initial samples [B, C, H, W]
            y: measurements [B, C, H, W]
            **kwargs: additional arguments (likelihood, etc.)
        
        Returns:
            Final samples [B, C, H, W] ready to use (fully processed)
        """
        raise NotImplementedError("Subclass must implement sample()")


