"""
DPS sampler implementation.
Implements DPS algorithm with score transformation (VE → DDPM).
"""
from .base import BaseSampler
from functools import partial
import torch
import numpy as np


class DPSSampler(BaseSampler):
    """DPS (Diffusion Posterior Sampling) algorithm implementation."""
    
    def __init__(self, config, device):
        """
        Initialize DPS sampler with unified prior and DDPM components.
        
        Args:
            config: Config instance with DPS parameters
            device: torch device for computation
        """
        super().__init__(config, device)
        
        # Import DPS components
        from third_party.DPS.gaussian_diffusion import (
            get_named_beta_schedule, create_sampler
        )
        
        # Get DPS configuration
        dps_cfg = config.params
        
        # Create beta schedule
        betas = get_named_beta_schedule(
            dps_cfg['noise_schedule'],
            dps_cfg['steps']
        )
        
        # Compute DDPM parameters for score transformation
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        self.s = np.sqrt(alphas_cumprod)  # sqrt(alpha_bar_t)
        self.sigma_ddpm = np.sqrt(1.0 / alphas_cumprod - 1.0)  # sqrt(1/alpha_bar_t - 1)
        
        # Initialize unified prior (VE interface only)
        from ..prior import get_prior
        self.prior = get_prior(device=device, **config.prior)
        
        # Create DDPM sampler
        self.ddpm_sampler = create_sampler(
            sampler=dps_cfg['sampler'],
            steps=dps_cfg['steps'],
            noise_schedule=dps_cfg['noise_schedule'],
            model_mean_type=dps_cfg['model_mean_type'],
            model_var_type=dps_cfg['model_var_type'],
            dynamic_threshold=dps_cfg.get('dynamic_threshold', False),
            clip_denoised=dps_cfg['clip_denoised'],
            rescale_timesteps=dps_cfg.get('rescale_timesteps', False),
            timestep_respacing=str(dps_cfg['steps']),
        )
        
        # Save scale parameter
        self.scale = dps_cfg['scale']
    
    def _noise_predictor(self, x, t):
        """
        DDPM-style noise predictor using score transformation.
        Transforms VE score to DDPM noise prediction.
        
        This is DPS-specific: converts the VE prior's score function
        to DDPM's noise prediction format.
        
        Args:
            x: input at diffusion time t, [B, C, H, W]
            t: diffusion timestep (int or array of ints)
        
        Returns:
            Predicted noise [B, C, H, W]
        """
        # Get DDPM parameters at time t
        s_t = torch.as_tensor(self.s[t], device=x.device)
        sigma_t = torch.as_tensor(self.sigma_ddpm[t], device=x.device)
        
        # Scale x for VE score computation
        x_scaled = x / s_t
        
        # Score transformation: VE score → DDPM noise
        ve_score = self.prior.score_fn(x_scaled, sigma_t)
        ddpm_noise = -ve_score * sigma_t
        
        return ddpm_noise
    
    def sample(self, x, y, likelihood=None, **kwargs):
        """
        Execute DPS sampling algorithm with full parameter control.
        
        Args:
            x: initial samples [B, C, H, W]
            y: measurements [B, C, H, W]
            likelihood: likelihood instance (required)
        
        Returns:
            Final denoised samples [B, C, H, W]
        """
        if likelihood is None:
            raise ValueError("DPS requires likelihood")
        
        # Import conditioning method
        from third_party.DPS.condition_methods import get_conditioning_method
        
        # Create conditioning function with config scale
        noiser = type('gaussian', (), {})
        cond_method = get_conditioning_method('ps', likelihood.operator, noiser, scale=self.scale)
        measurement_cond_fn = cond_method.conditioning
        
        # Create sampling function
        sample_fn = partial(
            self.ddpm_sampler.p_sample_loop,
            model=self._noise_predictor,  # ← Use DPS-specific noise predictor
            measurement_cond_fn=measurement_cond_fn
        )
        progress_context = kwargs.get('progress_context')
        
        # Sample for each image
        samples = []
        for i in range(y.shape[0]):
            y_i = y[i].unsqueeze(0)
            x_start = torch.randn((1, *x.shape[1:]), device=self.device).requires_grad_()
            sample = sample_fn(
                x_start=x_start,
                measurement=y_i,
                record=False,
                save_root=None,
                progress_desc=self._progress_description(
                    i,
                    y.shape[0],
                    progress_context,
                ),
            )
            samples.append(sample)
        
        return torch.cat(samples, dim=0)

    @staticmethod
    def _progress_description(local_index, total, context):
        """Describe one inherited DPS reverse loop without changing its use."""
        if not isinstance(context, dict):
            return f"DPS item {local_index + 1}/{total}"

        start_idx = int(context.get('start_idx', 0))
        mode = context.get('mode')
        if mode == 'single':
            global_index = start_idx + local_index
            total_samples = int(context.get('total_samples', total))
            return (
                f"DPS sample {global_index + 1}/{total_samples}"
            )

        if mode == 'batch':
            batch_size = int(context.get('batch_size', total))
            num_samples = int(context.get('num_samples', 1))
            if batch_size > 0 and num_samples > 0:
                image_index = start_idx + local_index % batch_size
                sample_index = local_index // batch_size
                return (
                    f"DPS image {image_index:03d}, "
                    f"sample {sample_index + 1}/{num_samples}"
                )

        return f"DPS item {local_index + 1}/{total}"
