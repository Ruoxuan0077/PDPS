"""
Universal sampling runner.
Supports PDPS and DPS methods through unified sampler interface.
"""
import torch
from ..likelihood import get_operator, get_likelihood
from ..utils.io import load_image, save_image


class Runner:
    def __init__(self, sampler, config, device):
        self.sampler = sampler
        self.config = config
        self.device = device
        self.output_dir = config.get_output_dir()
        
        # Initialize common components
        self._init_models()
    
    def _init_models(self):
        """Initialize operator and likelihood (common for all methods)."""
        operator = get_operator(device=self.device, **self.config.operator)
        likelihood = get_likelihood(operator=operator, **self.config.likelihood)
        
        self.likelihood = likelihood
        self.get_measurement = likelihood.get_label
        self.operator = operator
    
    def run_single(self, label: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
        """
        Run sampling for a single image.
        
        Args:
            label: clean image tensor [1, C, H, W]
            start_idx: starting index for saved samples
        
        Returns:
            Final samples tensor [N, C, H, W]
        """
        # Get measurement
        y = self.get_measurement(label).to(self.device)
        
        # Save label and measurement
        save_image(label, f'{self.output_dir}/label.png')
        save_image(y, f'{self.output_dir}/input.png')
        
        # Initialize samples
        num_samples = self.config.num_samples
        x = torch.randn(num_samples, *label.shape[1:]).to(self.device)
        y = y.repeat(num_samples, *[1] * (y.dim() - 1))
        
        # Execute sampling (returns final usable results)
        x_final = self.sampler.sample(x, y.detach(), 
                                      likelihood=self.likelihood)
        
        # Save results
        for i in range(num_samples):
            save_image(x_final[i].cpu(), f'{self.output_dir}/{start_idx + i}.png')
        
        return x_final
    
    def run_batch(self, labels: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
        """
        Run sampling for multiple images.
        
        Args:
            labels: clean images tensor [B, C, H, W]
            start_idx: starting index for saved images
        
        Returns:
            Final samples tensor [B*N, C, H, W]
        """
        batch_size = labels.shape[0]
        num_samples = self.config.num_samples
        
        # Get measurements
        y = self.get_measurement(labels).to(self.device)
        
        # Save labels and measurements
        for i in range(batch_size):
            save_image(labels[i], f'{self.output_dir}/labels/{start_idx + i:03d}.png')
            save_image(y[i], f'{self.output_dir}/inputs/{start_idx + i:03d}.png')
        
        # Initialize samples
        x = torch.randn(batch_size * num_samples, *labels.shape[1:]).to(self.device)
        y = y.repeat(num_samples, *[1] * (y.dim() - 1))
        
        # Execute sampling (returns final usable results)
        x_final = self.sampler.sample(x, y.detach(),
                                      likelihood=self.likelihood)
        
        # Save results
        for i in range(batch_size):
            for j in range(num_samples):
                idx = i + j * batch_size
                save_image(x_final[idx].cpu(), f'{self.output_dir}/{start_idx + i:03d}_{j}.png')
        
        return x_final
    
    def run(self):
        """
        Execute sampling based on configuration mode.
        Automatically determines single or batch mode.
        """
        label_path = self.config.get_label_path()
        label = load_image(label_path)
        
        if self.config.mode == 'single':
            label = label.unsqueeze(0) if label.dim() == 3 else label
            self.run_single(label)
        else:
            self.run_batch(label)
