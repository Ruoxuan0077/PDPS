"""Total-variation reconstruction configuration."""
from dataclasses import dataclass
import math

from . import BaseConfig


TASK_DEFAULTS = {
    'gaussian_deblur': {
        'stepsize': 1.0,
        'lambda_tv': 0.01,
        'max_iter': 50,
    },
    'motion_deblur': {
        'stepsize': 1.0,
        'lambda_tv': 0.01,
        'max_iter': 150,
    },
    'nonlinear_deblur': {
        'stepsize': 0.05,
        'lambda_tv': 0.05,
        'max_iter': 50,
    },
}


PAPER_CASES = {
    # FFHQ single
    ('ffhq', 'single', 'motion_deblur', '065'),
    ('ffhq', 'single', 'motion_deblur', '124'),
    ('ffhq', 'single', 'gaussian_deblur', '097'),
    ('ffhq', 'single', 'gaussian_deblur', '114'),
    ('ffhq', 'single', 'nonlinear_deblur', '126'),
    ('ffhq', 'single', 'nonlinear_deblur', '127'),
    # FFHQ batch
    ('ffhq', 'batch', 'motion_deblur'),
    ('ffhq', 'batch', 'gaussian_deblur'),
    ('ffhq', 'batch', 'nonlinear_deblur'),
    # AFHQ single
    ('afhq', 'single', 'motion_deblur', '003'),
    ('afhq', 'single', 'motion_deblur', '050'),
    ('afhq', 'single', 'nonlinear_deblur', '022'),
    ('afhq', 'single', 'nonlinear_deblur', '086'),
    # AFHQ batch
    ('afhq', 'batch', 'motion_deblur'),
    ('afhq', 'batch', 'nonlinear_deblur'),
}


@dataclass(init=False)
class TVConfig(BaseConfig):
    """Configuration for deterministic proximal-gradient TV reconstruction."""

    lambda_tv: float = 0.01
    stepsize: float = 1.0
    max_iter: int = 50
    tv_inner_max_iter: int = 200
    tv_inner_tol: float = 1e-5

    def __init__(
        self,
        task,
        dataset,
        image_id=None,
        num_samples=1,
        _is_paper=False,
        lambda_tv=0.01,
        stepsize=1.0,
        max_iter=50,
        tv_inner_max_iter=200,
        tv_inner_tol=1e-5,
        *,
        seed=None,
        measurement_seed=42,
        strict_deterministic=False,
        batch_chunk_size=None,
    ):
        self.lambda_tv = lambda_tv
        self.stepsize = stepsize
        self.max_iter = max_iter
        self.tv_inner_max_iter = tv_inner_max_iter
        self.tv_inner_tol = tv_inner_tol
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
        if self.num_samples != 1:
            raise ValueError(
                "TV is deterministic and requires num_samples=1; "
                f"got {self.num_samples}"
            )
        for name in ('lambda_tv', 'stepsize', 'tv_inner_tol'):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        for name in ('max_iter', 'tv_inner_max_iter'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def method_name(self):
        return 'tv'

    @property
    def params(self):
        """Proximal-gradient and inner TV-prox parameters."""
        return {
            'lambda_tv': self.lambda_tv,
            'stepsize': self.stepsize,
            'max_iter': self.max_iter,
            'tv_inner_max_iter': self.tv_inner_max_iter,
            'tv_inner_tol': self.tv_inner_tol,
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
        """Use the reported outer TV parameters with a strict TV prox."""
        cls.validate_paper_mode(mode, image_id)
        if mode == 'single':
            key = (dataset, mode, task, image_id)
        else:
            key = (dataset, mode, task)
        if key not in PAPER_CASES:
            raise ValueError(f"No paper config for {key}")

        params = TASK_DEFAULTS[task]
        return cls(
            task=task,
            dataset=dataset,
            image_id=image_id,
            num_samples=1,
            lambda_tv=params['lambda_tv'],
            stepsize=params['stepsize'],
            max_iter=params['max_iter'],
            tv_inner_max_iter=1000,
            tv_inner_tol=1e-8,
            _is_paper=True,
            seed=seed,
            measurement_seed=measurement_seed,
            strict_deterministic=strict_deterministic,
            batch_chunk_size=batch_chunk_size,
        )

    @classmethod
    def from_custom(
        cls,
        task,
        dataset,
        image_id=None,
        num_samples=1,
        **overrides,
    ):
        """Create a custom TV configuration with task-specific defaults."""
        if task not in TASK_DEFAULTS:
            raise ValueError(f"No TV defaults for task: {task}")
        supported = {
            'lambda_tv',
            'stepsize',
            'max_iter',
            'tv_inner_max_iter',
            'tv_inner_tol',
            'seed',
            'measurement_seed',
            'strict_deterministic',
            'batch_chunk_size',
        }
        unknown = set(overrides) - supported
        if unknown:
            raise ValueError(
                f"Unknown TV overrides: {sorted(unknown)}"
            )

        params = TASK_DEFAULTS[task]
        return cls(
            task=task,
            dataset=dataset,
            image_id=image_id,
            num_samples=num_samples,
            lambda_tv=overrides.get('lambda_tv', params['lambda_tv']),
            stepsize=overrides.get('stepsize', params['stepsize']),
            max_iter=overrides.get('max_iter', params['max_iter']),
            tv_inner_max_iter=overrides.get('tv_inner_max_iter', 200),
            tv_inner_tol=overrides.get('tv_inner_tol', 1e-5),
            _is_paper=False,
            seed=overrides.get('seed'),
            measurement_seed=overrides.get('measurement_seed', 42),
            strict_deterministic=overrides.get(
                'strict_deterministic',
                False,
            ),
            batch_chunk_size=overrides.get('batch_chunk_size'),
        )
