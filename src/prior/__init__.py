"""
Unified prior module.
Provides consistent interface for different sampling methods.
"""
from .edm import EDMPrior


def get_prior(name='edm', **kwargs):
    """
    Factory function to create prior model.
    
    Args:
        name: prior model name
        **kwargs: model-specific parameters
    
    Returns:
        Prior model instance
    """
    priors = {
        'edm': EDMPrior,
    }
    
    if name not in priors:
        raise ValueError(f"Unknown prior: {name}")
    
    return priors[name](**kwargs)


__all__ = ['EDMPrior', 'get_prior']


