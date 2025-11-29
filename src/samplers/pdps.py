"""
PDPS sampler implementation.
Complete PDPS algorithm integrated within the sampler class.
"""
from .base import BaseSampler
from ..prior import get_prior
import torch
import time
import math


class PDPSSampler(BaseSampler):
    """PDPS (Posterior Diffusion Posterior Sampling) algorithm."""
    
    def __init__(self, config, device):
        """
        Initialize PDPS sampler with prior model.
        
        Args:
            config: Config instance with PDPS parameters
            device: torch device for computation
        """
        super().__init__(config, device)
        
        # Initialize unified prior model (VE interface)
        self.prior = get_prior(device=device, **config.prior)
        self.prior_fn = self.prior.score_fn
        self.denoiser_fn = self.prior.denoiser
    
    def sample(self, x, y, likelihood=None, **kwargs):
        """
        Execute PDPS sampling algorithm and return final denoised results.
        
        Args:
            x: initial samples [B, C, H, W]
            y: measurements [B, C, H, W]
            likelihood: likelihood instance (required)
        
        Returns:
            Final denoised samples [B, C, H, W] ready to use
        """
        if likelihood is None:
            raise ValueError("PDPS requires likelihood")
        
        # Execute PDPS sampling
        x_samples = self._pdps_sampling(x, y, likelihood.likelihood_fn)
        
        # Denoise samples before returning
        x_final = self.denoiser_fn(x_samples)
        
        return x_final
    
    def _pdps_sampling(self, x, y, likelihood_fn):
        """
        PDPS core algorithm implementation.
        
        Args:
            x: initial samples [B, C, H, W]
            y: measurements [B, C, H, W]
            likelihood_fn: likelihood score function
        
        Returns:
            Final samples [B, C, H, W] before denoising
        """
        params = self.config.params
        
        # Extract parameters
        T = params['T']
        T0 = params['T0']
        warm_steps = params['warm_steps']
        in_steps_warm = params['in_steps_warm']
        diff_steps = params['diff_steps']
        in_steps_diff = params['in_steps_diff']
        mc_chains = params['mc_chains']
        burn_fraction = params['burn_fraction']
        snr = params['snr']
        in_snr = params['in_snr']
        info_freq = params['info_freq']
        
        # Initialize
        start_time = time.time()
        x = x.to(self.device)
        z0 = torch.randn(mc_chains, *x.shape, device=self.device)
        
        # Warm-up phase
        for i in range(warm_steps):
            score, z0 = self._score_estimation(
                T, x, y, self.prior_fn, likelihood_fn, in_steps_warm,
                mc_chains, burn_fraction, z0, in_snr
            )
            noise = torch.randn_like(x)
            eps = self._tamed_stepsize(score, noise, snr)
            dx = eps * score + torch.sqrt(2 * eps) * noise
            x = x + dx
            
            if (i + 1) % info_freq == 0:
                elapsed = time.time() - start_time
                print(f"{self.device}, Warm-up [{i + 1}/{warm_steps}] time: {elapsed:.2f}s")
                start_time = time.time()
        
        # Diffusion phase
        start_time = time.time()
        delta = torch.tensor((T - T0) / diff_steps, device=self.device)
        
        for i in range(diff_steps - 1):
            t = T - i * delta
            score, z0 = self._score_estimation(
                t, x, y, self.prior_fn, likelihood_fn, in_steps_diff,
                mc_chains, burn_fraction, z0, in_snr
            )
            noise = torch.randn_like(x)
            dx = (x + 2. * score) * delta + torch.sqrt(2 * delta) * noise
            x = x + dx
            
            if (i + 1) % info_freq == 0:
                elapsed = time.time() - start_time
                print(f"{self.device}, Diffusion [{i + 1}/{diff_steps}] time: {elapsed:.2f}s")
                start_time = time.time()
        
        # Final denoising step
        score, _ = self._score_estimation(
            T0, x, y, self.prior_fn, likelihood_fn, in_steps_diff,
            mc_chains, burn_fraction, z0, in_snr
        )
        dx = (x + 2. * score) * T0
        x_final = x + dx
        
        return x_final
    
    def _score_estimation(self, t, x, y, prior_score_fn, likelihood_fn,
                         steps, mc_chains, burn_frac, z0, snr):
        """
        Estimate posterior score using MCMC.
        
        Args:
            t: diffusion time
            x: current state [B, C, H, W]
            y: measurements [B, C, H, W]
            prior_score_fn: prior score function
            likelihood_fn: likelihood score function
            steps: MCMC steps
            mc_chains: number of MCMC chains
            burn_frac: burn-in fraction
            z0: initial MCMC states [mc_chains, B, C, H, W]
            snr: signal-to-noise ratio
        
        Returns:
            score_estimate: estimated score [B, C, H, W]
            z_final: final MCMC states [mc_chains, B, C, H, W]
        """
        device = x.device
        t = torch.as_tensor(t, device=device)
        
        # Diffusion parameters
        mu = torch.exp(-t)
        sigma2 = 1. - torch.exp(-2. * t)
        
        # Reshape z0: (mc_chains, B, C, H, W) → (mc_chains * B, C, H, W)
        if z0.shape != (mc_chains, *x.shape):
            raise ValueError(f"z0 shape mismatch: expected {(mc_chains, *x.shape)}, got {z0.shape}")
        
        z0 = z0.reshape(-1, *z0.shape[2:])
        
        # Expand x and y for mc_chains
        xt = x.unsqueeze(0).expand((mc_chains, *x.shape)).reshape(-1, *x.shape[1:])
        y_expanded = y.unsqueeze(0).expand((mc_chains, *y.shape)).reshape(-1, *y.shape[1:])
        
        # Posterior score function
        def posterior_score(x0):
            prior_score = prior_score_fn(x0)
            posterior = prior_score + mu * (xt - mu * x0) / sigma2
            posterior += likelihood_fn(x0, y_expanded)
            return posterior
        
        # Inner ULA MCMC
        z_temp = z0
        z_traj = torch.empty((steps, *z0.shape), device=device)
        
        for i in range(steps):
            score = posterior_score(z_temp)
            noise = torch.randn_like(z_temp)
            eps = self._tamed_stepsize(score, noise, snr)
            dz = eps * score + torch.sqrt(2 * eps) * noise
            z_temp = z_temp + dz
            z_traj[i] = z_temp.clone()
        
        # Burn-in and averaging
        num_retained = int(burn_frac * steps)
        z_mc = z_traj[-num_retained:]  # (num_retained, mc_chains * B, C, H, W)
        z_mc = z_mc.view((num_retained * mc_chains, -1, *x.shape[1:]))
        
        # Final state for next iteration
        z_final = z_traj[-1].view((mc_chains, -1, *x.shape[1:]))
        
        # Score estimate
        denoiser = z_mc.mean(dim=0)
        score_estimate = (mu * denoiser - x) / (sigma2 + 1e-7)
        
        return score_estimate, z_final
    
    def _tamed_stepsize(self, grad, noise, snr):
        """
        Compute tamed step size for Langevin dynamics.
        
        Args:
            grad: gradient tensor
            noise: noise tensor
            snr: signal-to-noise ratio
        
        Returns:
            Step size tensor
        """
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1)
        d = math.prod(noise.shape[1:])
        noise_norm = torch.sqrt(torch.tensor(d, dtype=grad.dtype, device=grad.device))
        step_size = (snr * noise_norm / grad_norm) ** 2 * 2
        step_size = step_size.view(-1, *([1] * (grad.dim() - 1)))
        return step_size
