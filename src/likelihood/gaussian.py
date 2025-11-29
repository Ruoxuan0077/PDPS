"""
Gaussian likelihood function.
Computes likelihood score for Gaussian measurement model.
"""
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
        self.sigma = sigma
    
    def get_label(self, data, seed=42):
        """
        Generate noisy measurement y = A(x) + noise.
        
        Args:
            data: clean data [B, C, H, W]
            seed: random seed for reproducibility
        
        Returns:
            Noisy measurement [B, C, H, W]
        """
        data = data.to(self.device)
        operated = self.operator.forward(data)
        
        # Generate Gaussian noise
        local_generator = torch.Generator(device=self.device)
        if seed is not None:
            local_generator.manual_seed(seed)
        noise = torch.randn(operated.size(), 
                           generator=local_generator, 
                           device=self.device)
        
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
        y = y.to(self.device)
        
        # Compute data fidelity term
        Ax = self.operator.forward(x)
        diff = -torch.sum((y - Ax)**2, dim=tuple(range(1, len(x0.shape))))
        total_diff = diff.sum()
        total_diff.backward()
        
        # Likelihood gradient
        gradient = x.grad.clone() / (2 * self.sigma**2)

        return gradient

