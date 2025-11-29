"""
EDM prior model with VE (Variance Exploding) interface.
Provides score function and denoiser for PDPS-style sampling.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import pickle


class EDMPrior:
    """
    EDM prior model with VE interface.
    Loads pre-trained EDM network and provides score-based functions.
    """
    
    def __init__(self, model_path, device, sigma_ve, sigma_final):
        """
        Initialize EDM prior model.
        
        Args:
            model_path: path to EDM checkpoint file
            device: torch device
            sigma_ve: noise level for VE score function
            sigma_final: noise level for final denoising
        """
        self.model_path = model_path
        self.device = device
        self.sigma_ve = sigma_ve
        self.sigma_final = sigma_final
        
        # Load EDM network
        with open(model_path, 'rb') as f:
            self.network = pickle.load(f)['ema'].to(device)
            self.network.eval()
    
    def score_fn(self, x, sigma=None):
        """
        VE-style score function: ∇_x log p(x).
        
        Args:
            x: input tensor [B, C, H, W]
            sigma: noise level (uses self.sigma_ve if None)
        
        Returns:
            Score tensor [B, C, H, W]
        """
        sigma = sigma or self.sigma_ve
        sigma_tensor = torch.tensor(sigma, device=self.device)
        return (self.network(x, sigma_tensor, None) - x) / (sigma_tensor ** 2)
    
    def denoiser(self, x, sigma=None):
        """
        Denoising function.
        
        Args:
            x: noisy input [B, C, H, W]
            sigma: noise level (uses self.sigma_final if None)
        
        Returns:
            Denoised output [B, C, H, W]
        """
        sigma = sigma or self.sigma_final
        sigma_tensor = torch.tensor(sigma, device=self.device)
        return self.network(x, sigma_tensor, None)
