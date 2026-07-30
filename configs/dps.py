"""DPS configuration."""
from dataclasses import dataclass
import math

from . import BaseConfig

# DPS defaults
DEFAULTS = {
    'steps': 1000, 'noise_schedule': 'linear', 'sampler': 'ddpm',
    'model_mean_type': 'epsilon', 'model_var_type': 'fixed_small',
    'clip_denoised': True, 'scale': 1.0,
}

# Paper presets: complete configurations (no dependency on PDPS)
PAPER = {
    # ffhq single
    ('ffhq', 'single', 'motion_deblur',    '065'): {'scale': 1.3, 'num_samples': 24},
    ('ffhq', 'single', 'motion_deblur',    '124'): {'scale': 1.3, 'num_samples': 24},
    ('ffhq', 'single', 'gaussian_deblur',  '097'): {'scale': 0.9, 'num_samples': 24},
    ('ffhq', 'single', 'gaussian_deblur',  '114'): {'scale': 0.8, 'num_samples': 24},
    ('ffhq', 'single', 'nonlinear_deblur', '126'): {'scale': 0.5, 'num_samples': 24},
    ('ffhq', 'single', 'nonlinear_deblur', '127'): {'scale': 0.9, 'num_samples': 24},
    # ffhq batch
    ('ffhq', 'batch',  'motion_deblur'):           {'scale': 1.2, 'num_samples': 1},
    ('ffhq', 'batch',  'gaussian_deblur'):         {'scale': 0.9, 'num_samples': 1},
    ('ffhq', 'batch',  'nonlinear_deblur'):        {'scale': 0.2, 'num_samples': 1},
    # afhq single
    ('afhq', 'single', 'motion_deblur',    '003'): {'scale': 1.3, 'num_samples': 24},
    ('afhq', 'single', 'motion_deblur',    '050'): {'scale': 1.3, 'num_samples': 24},
    ('afhq', 'single', 'nonlinear_deblur', '022'): {'scale': 0.4, 'num_samples': 24},
    ('afhq', 'single', 'nonlinear_deblur', '086'): {'scale': 0.3, 'num_samples': 24},
    # afhq batch
    ('afhq', 'batch',  'motion_deblur'):           {'scale': 1.3, 'num_samples': 1},
    ('afhq', 'batch',  'nonlinear_deblur'):        {'scale': 0.2, 'num_samples': 1},
}


@dataclass(init=False)
class DPSConfig(BaseConfig):
    """DPS configuration."""
    scale: float = 1.0
    steps: int = 1000

    def __init__(
        self,
        task,
        dataset,
        image_id=None,
        num_samples=1,
        _is_paper=False,
        scale=1.0,
        steps=1000,
        *,
        seed=None,
        measurement_seed=42,
        strict_deterministic=False,
        batch_chunk_size=None,
    ):
        self.scale = scale
        self.steps = steps
        BaseConfig.__init__(
            self,
            task=task,
            dataset=dataset,
            image_id=image_id,
            num_samples=num_samples,
            _is_paper=_is_paper,
            seed=seed,
            measurement_seed=measurement_seed,
            strict_deterministic=strict_deterministic,
            batch_chunk_size=batch_chunk_size,
        )

    def __post_init__(self):
        super().__post_init__()
        if (
            isinstance(self.scale, bool)
            or not isinstance(self.scale, (int, float))
            or not math.isfinite(self.scale)
            or self.scale <= 0
        ):
            raise ValueError("scale must be positive and finite")
        if type(self.steps) is not int or self.steps < 21:
            raise ValueError(
                "steps must be an integer >= 21 for the linear beta "
                "schedule used by DPS"
            )

    @property
    def method_name(self):
        return 'dps'

    @property
    def params(self):
        """DPS algorithm parameters."""
        return {**DEFAULTS, 'scale': self.scale, 'steps': self.steps}

    @classmethod
    def from_paper(
        cls,
        dataset,
        mode,
        task,
        image_id=None,
        seed=None,
        measurement_seed=42,
        strict_deterministic=False,
        batch_chunk_size=None,
    ):
        """Create from paper preset."""
        cls.validate_paper_mode(mode, image_id)
        if mode == 'single':
            key = (dataset, mode, task, image_id)
        else:
            key = (dataset, mode, task)

        if key not in PAPER:
            raise ValueError(f"No paper config for {key}")

        cfg = PAPER[key]
        return cls(
            task=task,
            dataset=dataset,
            image_id=image_id,
            num_samples=cfg['num_samples'],
            scale=cfg['scale'],
            steps=DEFAULTS['steps'],
            _is_paper=True,
            seed=seed,
            measurement_seed=measurement_seed,
            strict_deterministic=strict_deterministic,
            batch_chunk_size=batch_chunk_size,
        )

    @classmethod
    def from_custom(cls, task, dataset, image_id=None, num_samples=1, **kw):
        """Create custom config."""
        supported = {
            'scale',
            'steps',
            'seed',
            'measurement_seed',
            'strict_deterministic',
            'batch_chunk_size',
        }
        unknown = set(kw) - supported
        if unknown:
            raise ValueError(
                f"Unknown DPS overrides: {sorted(unknown)}"
            )
        return cls(
            task=task,
            dataset=dataset,
            image_id=image_id,
            num_samples=num_samples,
            scale=kw.get('scale', DEFAULTS['scale']),
            steps=kw.get('steps', DEFAULTS['steps']),
            _is_paper=False,
            seed=kw.get('seed'),
            measurement_seed=kw.get('measurement_seed', 42),
            strict_deterministic=kw.get('strict_deterministic', False),
            batch_chunk_size=kw.get('batch_chunk_size'),
        )
