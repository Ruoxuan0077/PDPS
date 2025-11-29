"""DPS configuration."""
from dataclasses import dataclass
from . import BaseConfig

# DPS defaults
DEFAULTS = {
    'steps': 1000, 'noise_schedule': 'linear', 'sampler': 'ddpm',
    'model_mean_type': 'epsilon', 'model_var_type': 'learned_range',
    'clip_denoised': True, 'scale': 1.0,
}

# Paper presets: complete configurations (no dependency on PDPS)
PAPER = {
    # ffhq single
    ('ffhq', 'single', 'motion_deblur',    '065'): {'scale': 1.2, 'num_samples': 24},
    ('ffhq', 'single', 'motion_deblur',    '124'): {'scale': 1.2, 'num_samples': 24},
    ('ffhq', 'single', 'gaussian_deblur',  '097'): {'scale': 0.9, 'num_samples': 24},
    ('ffhq', 'single', 'gaussian_deblur',  '114'): {'scale': 0.9, 'num_samples': 24},
    ('ffhq', 'single', 'nonlinear_deblur', '126'): {'scale': 0.9, 'num_samples': 24},
    ('ffhq', 'single', 'nonlinear_deblur', '127'): {'scale': 0.9, 'num_samples': 24},
    # ffhq batch
    ('ffhq', 'batch',  'motion_deblur'):           {'scale': 1.2, 'num_samples': 1},
    ('ffhq', 'batch',  'gaussian_deblur'):         {'scale': 0.9, 'num_samples': 1},
    ('ffhq', 'batch',  'nonlinear_deblur'):        {'scale': 0.9, 'num_samples': 1},
    # afhq single
    ('afhq', 'single', 'motion_deblur',    '003'): {'scale': 1.2, 'num_samples': 24},
    ('afhq', 'single', 'motion_deblur',    '050'): {'scale': 1.2, 'num_samples': 24},
    ('afhq', 'single', 'nonlinear_deblur', '022'): {'scale': 0.9, 'num_samples': 24},
    ('afhq', 'single', 'nonlinear_deblur', '086'): {'scale': 0.9, 'num_samples': 24},
    # afhq batch
    ('afhq', 'batch',  'motion_deblur'):           {'scale': 1.2, 'num_samples': 1},
    ('afhq', 'batch',  'nonlinear_deblur'):        {'scale': 0.9, 'num_samples': 1},
}


@dataclass
class DPSConfig(BaseConfig):
    """DPS configuration."""
    scale: float = 1.0
    steps: int = 1000
    
    @property
    def method_name(self):
        return 'dps'
    
    @property
    def params(self):
        """DPS algorithm parameters."""
        return {**DEFAULTS, 'scale': self.scale, 'steps': self.steps}
    
    @classmethod
    def from_paper(cls, dataset, mode, task, image_id=None):
        """Create from paper preset."""
        key = (dataset, mode, task, image_id) if image_id else (dataset, mode, task)
        
        if key not in PAPER:
            raise ValueError(f"No paper config for {key}")
        
        cfg = PAPER[key]
        return cls(task=task, dataset=dataset, image_id=image_id,
                  num_samples=cfg['num_samples'], scale=cfg['scale'], _is_paper=True)
    
    @classmethod
    def from_custom(cls, task, dataset, image_id=None, num_samples=1, **kw):
        """Create custom config."""
        return cls(task=task, dataset=dataset, image_id=image_id,
                  num_samples=num_samples, scale=kw.get('scale', 1.0),
                  steps=kw.get('steps', 1000), _is_paper=False)
