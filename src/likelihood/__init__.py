from .operators import BaseOperator, get_operator
from .gaussian import GaussianLikelihood


def get_likelihood(name='gaussian', **kwargs):
    """
    Factory function to create likelihood.
    
    Args:
        name: likelihood name
        **kwargs: likelihood-specific parameters (must include 'operator')
    
    Returns:
        Likelihood instance
    """
    likelihoods = {
        'gaussian': GaussianLikelihood,
    }
    
    if name not in likelihoods:
        raise ValueError(f"Unknown likelihood: {name}")
    
    return likelihoods[name](**kwargs)


__all__ = [
    'BaseOperator',
    'get_operator',
    'get_likelihood',
    'GaussianLikelihood',
]
