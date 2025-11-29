"""PnP-Flow configuration."""
from dataclasses import dataclass
from . import BaseConfig

# PnP-Flow defaults
DEFAULTS = {
    'steps_pnp': 1000,
    'sum_samples': 5,
    'gamma_style': 'alpha_1_minus_t',
    'alpha': 1.0,
}

# Paper presets (placeholder - please fill in with actual paper configurations)
PAPER = {
    # ffhq single
    ('ffhq', 'single', 'gaussian_deblur',  '097'): {'lr_pnp': 6.0, 'num_samples': 24},
    ('ffhq', 'single', 'gaussian_deblur',  '114'): {'lr_pnp': 9.0, 'num_samples': 24},
    ('ffhq', 'single', 'motion_deblur',    '065'): {'lr_pnp': 9.5, 'num_samples': 24},
    ('ffhq', 'single', 'motion_deblur',    '124'): {'lr_pnp': 9.5, 'num_samples': 24},
    ('ffhq', 'single', 'nonlinear_deblur', '126'): {'lr_pnp': 1.0, 'num_samples': 24},
    ('ffhq', 'single', 'nonlinear_deblur', '127'): {'lr_pnp': 4.0, 'num_samples': 24},
}


@dataclass
class PnPFlowConfig(BaseConfig):
    """PnP-Flow configuration."""
    lr_pnp: float = 1.0
    
    @property
    def method_name(self):
        return 'pnp_flow'
    
    @property
    def params(self):
        """PnP-Flow algorithm parameters."""
        return {**DEFAULTS, 'lr_pnp': self.lr_pnp}
    
    @classmethod
    def from_paper(cls, dataset, mode, task, image_id=None):
        """Create from paper preset."""
        key = (dataset, mode, task, image_id) if image_id else (dataset, mode, task)
        
        if key not in PAPER:
            raise ValueError(f"No paper config for {key}")
        
        cfg = PAPER[key]
        return cls(task=task, dataset=dataset, image_id=image_id,
                  num_samples=cfg['num_samples'],
                  lr_pnp=cfg['lr_pnp'],
                  _is_paper=True)
    
    @classmethod
    def from_custom(cls, task, dataset, image_id=None, num_samples=1, **kw):
        """Create custom config."""
        return cls(task=task, dataset=dataset, image_id=image_id,
                  num_samples=num_samples,
                  lr_pnp=kw.get('lr_pnp', 1.0),
                  _is_paper=False)
