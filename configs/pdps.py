"""PDPS configuration."""
from dataclasses import dataclass
import math

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
    ('ffhq', 'batch',  'motion_deblur'):           {'T': 0.5,  'num_samples': 1},
    ('ffhq', 'batch',  'gaussian_deblur'):         {'T': 0.2,  'num_samples': 1},
    ('ffhq', 'batch',  'nonlinear_deblur'):        {'T': 20.0, 'num_samples': 1},
    # afhq single
    ('afhq', 'single', 'motion_deblur',    '003'): {'T': 0.5,  'num_samples': 24},
    ('afhq', 'single', 'motion_deblur',    '050'): {'T': 0.5,  'num_samples': 24},
    ('afhq', 'single', 'nonlinear_deblur', '022'): {'T': 5.0,  'num_samples': 24},
    ('afhq', 'single', 'nonlinear_deblur', '086'): {'T': 9.0,  'num_samples': 24},
    # afhq batch
    ('afhq', 'batch',  'motion_deblur'):           {'T': 0.5,  'num_samples': 1},
    ('afhq', 'batch',  'nonlinear_deblur'):        {'T': 20.0, 'num_samples': 1},
}


@dataclass(init=False)
class PDPSConfig(BaseConfig):
    """PDPS configuration."""
    T: float = 0.5
    warm_steps: int = DEFAULTS['warm_steps']
    T0: float = DEFAULTS['T0']

    def __init__(
        self,
        task,
        dataset,
        image_id=None,
        num_samples=1,
        _is_paper=False,
        T=0.5,
        warm_steps=DEFAULTS['warm_steps'],
        T0=DEFAULTS['T0'],
        *,
        seed=None,
        measurement_seed=42,
        strict_deterministic=False,
        batch_chunk_size=None,
    ):
        self.T = T
        self.warm_steps = warm_steps
        self.T0 = T0
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

        for name in ('T', 'T0'):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"PDPS requires finite {name} > 0")
        if self.T <= self.T0:
            raise ValueError(
                f"PDPS requires T > T0; got T={self.T}, T0={self.T0}"
            )
        if int(1200 * self.T) < 1:
            raise ValueError(
                "PDPS requires int(1200 * T) >= 1 reverse step"
            )
        if type(self.warm_steps) is not int or self.warm_steps < 0:
            raise ValueError("warm_steps must be a non-negative integer")

        positive_integer_names = (
            'in_steps_warm',
            'in_steps_diff',
            'mc_chains',
            'info_freq',
        )
        for name in positive_integer_names:
            value = DEFAULTS[name]
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        burn_fraction = DEFAULTS['burn_fraction']
        if (
            not math.isfinite(burn_fraction)
            or not 0 < burn_fraction <= 1
        ):
            raise ValueError("burn_fraction must be finite and in (0, 1]")
        for name in ('in_steps_warm', 'in_steps_diff'):
            if int(burn_fraction * DEFAULTS[name]) < 1:
                raise ValueError(
                    f"burn_fraction retains no samples for {name}"
                )

        for name in ('snr', 'in_snr'):
            value = DEFAULTS[name]
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

    @property
    def method_name(self):
        return 'pdps'

    @property
    def params(self):
        """PDPS algorithm parameters."""
        return {
            **DEFAULTS,
            'T': self.T,
            'T0': self.T0,
            'warm_steps': self.warm_steps,
            'diff_steps': int(1200 * self.T),
        }

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
            T=cfg['T'],
            warm_steps=DEFAULTS['warm_steps'],
            T0=DEFAULTS['T0'],
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
            'T',
            'T0',
            'warm_steps',
            'seed',
            'measurement_seed',
            'strict_deterministic',
            'batch_chunk_size',
        }
        unknown = set(kw) - supported
        if unknown:
            raise ValueError(
                f"Unknown PDPS overrides: {sorted(unknown)}"
            )
        return cls(
            task=task,
            dataset=dataset,
            image_id=image_id,
            num_samples=num_samples,
            T=kw.get('T', 0.5),
            warm_steps=kw.get('warm_steps', DEFAULTS['warm_steps']),
            T0=kw.get('T0', DEFAULTS['T0']),
            _is_paper=False,
            seed=kw.get('seed'),
            measurement_seed=kw.get('measurement_seed', 42),
            strict_deterministic=kw.get('strict_deterministic', False),
            batch_chunk_size=kw.get('batch_chunk_size'),
        )
