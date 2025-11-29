"""PDPS configuration."""
from dataclasses import dataclass
from . import BaseConfig

# PDPS defaults
DEFAULTS = {
    'T0': 0.05, 'warm_steps': 400, 'in_steps_warm': 50, 'in_steps_diff': 20,
    'mc_chains': 20, 'burn_fraction': 0.5, 'snr': 0.16, 'in_snr': 0.075, 'info_freq': 10,
}

# Paper presets: all entries in uniform format
PAPER = {
    # ffhq single - each image registered individually
    ('ffhq', 'single', 'motion_deblur',    '065'): {'T': 0.5,  'num_samples': 24},
    ('ffhq', 'single', 'motion_deblur',    '124'): {'T': 0.5,  'num_samples': 24},
    ('ffhq', 'single', 'gaussian_deblur',  '097'): {'T': 0.2,  'num_samples': 24},
    ('ffhq', 'single', 'gaussian_deblur',  '114'): {'T': 0.2,  'num_samples': 24},
    ('ffhq', 'single', 'nonlinear_deblur', '126'): {'T': 3.5,  'num_samples': 24},
    ('ffhq', 'single', 'nonlinear_deblur', '127'): {'T': 3.5,  'num_samples': 24},
    # ffhq batch
    ('ffhq', 'batch',  'motion_blur'):           {'T': 0.5,  'num_samples': 1},
    ('ffhq', 'batch',  'gaussian_blur'):         {'T': 0.2,  'num_samples': 1},
    ('ffhq', 'batch',  'nonlinear_blur'):        {'T': 20.0, 'num_samples': 1},
    # afhq single
    ('afhq', 'single', 'motion_blur',    '003'): {'T': 0.5,  'num_samples': 24},
    ('afhq', 'single', 'motion_blur',    '050'): {'T': 0.5,  'num_samples': 24},
    ('afhq', 'single', 'nonlinear_blur', '022'): {'T': 5.0,  'num_samples': 24},
    ('afhq', 'single', 'nonlinear_blur', '086'): {'T': 9.0,  'num_samples': 24},
    # afhq batch
    ('afhq', 'batch',  'motion_blur'):           {'T': 0.5,  'num_samples': 1},
    ('afhq', 'batch',  'nonlinear_blur'):        {'T': 20.0, 'num_samples': 1},
}


@dataclass
class PDPSConfig(BaseConfig):
    """PDPS configuration."""
    T: float = 0.5
    
    @property
    def method_name(self):
        return 'pdps'
    
    @property
    def params(self):
        """PDPS algorithm parameters."""
        return {**DEFAULTS, 'T': self.T, 'diff_steps': int(1200 * self.T)}
    
    @classmethod
    def from_paper(cls, dataset, mode, task, image_id=None):
        """Create from paper preset."""
        key = (dataset, mode, task, image_id) if image_id else (dataset, mode, task)
        
        if key not in PAPER:
            raise ValueError(f"No paper config for {key}")
        
        cfg = PAPER[key]
        return cls(task=task, dataset=dataset, image_id=image_id,
                   num_samples=cfg['num_samples'], T=cfg['T'], _is_paper=True)
    
    @classmethod
    def from_custom(cls, task, dataset, image_id=None, num_samples=1, **kw):
        """Create custom config."""
        return cls(task=task, dataset=dataset, image_id=image_id,
                  num_samples=num_samples, T=kw.get('T', 0.5), _is_paper=False)
