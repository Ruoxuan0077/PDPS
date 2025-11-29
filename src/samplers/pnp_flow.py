"""
PnP-Flow sampler implementation.
Wraps the PnP-Flow algorithm with unified sampler interface.
"""
from .base import BaseSampler
from ..prior import get_prior
import torch


class PnPFlowSampler(BaseSampler):
    """PnP-Flow (Posterior Diffusion Posterior PnPFlowSampling) algorithm wrapper."""
    
    def __init__(self, config, device):
        """
        Initialize PnP-Flow sampler with prior model.
        
        Args:
            config: Config instance with PnP-Flow parameters
            device: torch device for computation
        """
        super().__init__(config, device)
        self.prior = get_prior(device=device, **config.prior)

    def learning_rate_strat(self, lr, t):
        t = t.view(-1, 1, 1, 1)
        gamma_style = self.config.params['gamma_style']
        
        if gamma_style == '1_minus_t':
            return lr * (1 - t)
        elif gamma_style == 'sqrt_1_minus_t':
            return lr * torch.sqrt(1 - t)
        elif gamma_style == 'constant':
            return lr
        else:  # 'alpha_1_minus_t'
            alpha = self.config.params['alpha']
            return lr * (1 - t)**alpha

    def interpolation_step(self, x, t):
        return t * x + torch.randn_like(x) * (1 - t)

    def denoiser(self, x, t):
        """
        Linear interpolation denoiser.
        """
        t = torch.as_tensor(t, device=x.device)
        linear_interpolation_score = self.prior.score_fn(x / t, ((1 - t) / t).item()) / t
        linear_interpolation_noise = -(1 - t) * linear_interpolation_score
        return (x - (1 - t) * linear_interpolation_noise) / t

    def v(self, x, t):
        t = torch.as_tensor(t, device=x.device)
        return (self.denoiser(x, t.item()) - x) / (1 - t)

    def sample(self, x, y, likelihood=None, **kwargs):
        """
        Execute PnP-Flow sampling algorithm and return final denoised results.
        """
        if likelihood is None:
            raise ValueError("PnP-Flow requires likelihood")
        
        pnp_flow_cfg = self.config.params
        sigma_noise = likelihood.sigma
        sum_samples = pnp_flow_cfg['sum_samples']
        steps = pnp_flow_cfg['steps_pnp']
        eps = 0.01
        delta = (1.0 - eps) / steps
        lr = sigma_noise**2 * pnp_flow_cfg['lr_pnp']
        
        for iteration in range(steps):
            if iteration % 20 == 0:
                print(f"[PnP-Flow] Iteration {iteration}/{steps}")
            t = torch.tensor(eps + delta * iteration, device=self.device)
            lr_t = self.learning_rate_strat(lr, t)
            z = x + lr_t * likelihood.likelihood_fn(x, y)
            x_new = torch.zeros_like(x)
            for _ in range(sum_samples):  # Average over sum_samples
                z_tilde = self.interpolation_step(z, t.view(-1, 1, 1, 1))
                x_new += self.denoiser(z_tilde, t)
            x = x_new / sum_samples
        return x

    