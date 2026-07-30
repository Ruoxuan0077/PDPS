"""
Configuration system for sampling methods.
Extreme minimal design with shared base class.
"""
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import secrets
from typing import Optional


# Shared configurations
TASKS = {
    'motion_deblur':    {'operator': {'name': 'motion_blur', 'kernel_size': 15, 'intensity': 0.5}},
    'gaussian_deblur':  {'operator': {'name': 'gaussian_blur', 'kernel_size': 15, 'intensity': 2.0}},
    'nonlinear_deblur': {'operator': {
        'name': 'nonlinear_blur',
        'opt_yml_path': 'src/likelihood/utils/bkse/options/generate_blur/default.yml',
        'kernel_scale': 0.3,
        'preserve_input_size': False,
    }},
}

DEFAULT_PRIOR = {'name': 'edm', 'model_path': 'data/nn/edm/edm-ffhq-64x64-uncond-ve.pkl', 'sigma_ve': 0.09, 'sigma_final': 0.03}
DEFAULT_LIKELIHOOD = {'name': 'gaussian', 'sigma': 0.05}
DATASETS = {'ffhq', 'afhq'}
MAX_SEED = 2**63


@dataclass
class BaseConfig(ABC):
    """Base configuration with shared functionality."""
    task: str
    dataset: str
    image_id: Optional[str] = None
    num_samples: int = 1
    _is_paper: bool = False
    seed: Optional[int] = None
    measurement_seed: int = 42
    strict_deterministic: bool = False
    batch_chunk_size: Optional[int] = None

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(
                f"Unknown task: {self.task!r}. "
                f"Expected one of {sorted(TASKS)}"
            )
        if self.dataset not in DATASETS:
            raise ValueError(
                f"Unknown dataset: {self.dataset!r}. "
                f"Expected one of {sorted(DATASETS)}"
            )
        if type(self.num_samples) is not int or self.num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        if self.image_id is not None:
            if not isinstance(self.image_id, str) or not self.image_id.strip():
                raise ValueError("image_id must be a non-empty string or None")
        if type(self._is_paper) is not bool:
            raise ValueError("_is_paper must be a boolean")
        if type(self.strict_deterministic) is not bool:
            raise ValueError("strict_deterministic must be a boolean")
        if (
            self.batch_chunk_size is not None
            and (
                type(self.batch_chunk_size) is not int
                or self.batch_chunk_size < 1
            )
        ):
            raise ValueError(
                "batch_chunk_size must be a positive integer or None"
            )
        if self.batch_chunk_size is not None and self.image_id is not None:
            raise ValueError(
                "batch_chunk_size is only valid in batch mode"
            )

        if self.seed is None:
            self.seed = secrets.randbits(63)
        self._validate_seed('seed', self.seed)
        self._validate_seed('measurement_seed', self.measurement_seed)

    @staticmethod
    def _validate_seed(name, value):
        if type(value) is not int or not 0 <= value < MAX_SEED:
            raise ValueError(
                f"{name} must be an integer in [0, {MAX_SEED})"
            )

    @staticmethod
    def validate_paper_mode(mode, image_id):
        if mode not in {'single', 'batch'}:
            raise ValueError("paper mode must be 'single' or 'batch'")
        if mode == 'single' and image_id is None:
            raise ValueError("single paper mode requires image_id")
        if mode == 'batch' and image_id is not None:
            raise ValueError("batch paper mode does not accept image_id")

    @property
    def mode(self):
        return 'single' if self.image_id is not None else 'batch'

    @property
    @abstractmethod
    def method_name(self) -> str:
        pass

    @property
    @abstractmethod
    def params(self):
        pass

    @property
    def operator(self):
        return deepcopy(TASKS[self.task]['operator'])

    @property
    def likelihood(self):
        return deepcopy(DEFAULT_LIKELIHOOD)

    @property
    def prior(self):
        return deepcopy(DEFAULT_PRIOR)

    def run_spec(self):
        """Return the complete, JSON-serializable reconstruction config."""
        return {
            'schema_version': 1,
            'method': self.method_name,
            'task': self.task,
            'dataset': self.dataset,
            'mode': self.mode,
            'image_id': self.image_id,
            'num_samples': self.num_samples,
            'is_paper': self._is_paper,
            'seed': self.seed,
            'measurement_seed': self.measurement_seed,
            'strict_deterministic': self.strict_deterministic,
            'batch_chunk_size': self.batch_chunk_size,
            'params': deepcopy(self.params),
            'operator': self.operator,
            'likelihood': self.likelihood,
            'prior': self.prior,
        }

    @property
    def config_fingerprint(self):
        payload = json.dumps(
            self.run_spec(),
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True,
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]

    def get_label_path(self):
        base = f"data/label/{self.dataset}"
        if self.image_id is not None:
            return f"{base}/{self.image_id}.png"
        return base

    def get_output_dir(self):
        """Return a stable paper path or a unique custom-run path."""
        task_name = self.operator['name']
        base = 'paper' if self._is_paper else 'custom'
        if self.image_id is not None:
            loc = f"{self.dataset}_{self.image_id}"
        else:
            loc = self.dataset
        output_dir = (
            f"fig/{self.method_name}/{base}/{self.mode}/{task_name}/{loc}"
        )
        if not self._is_paper:
            output_dir = f"{output_dir}/{self.config_fingerprint}"
        return output_dir


# Factory functions
from .pdps import PDPSConfig
from .dps import DPSConfig
from .tv import TVConfig

METHODS = {'pdps': PDPSConfig, 'dps': DPSConfig, 'tv': TVConfig}


def from_paper(
    method,
    dataset,
    mode,
    task,
    image_id=None,
    seed=None,
    measurement_seed=42,
    strict_deterministic=False,
    batch_chunk_size=None,
):
    """Create config from paper preset."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    return METHODS[method].from_paper(
        dataset,
        mode,
        task,
        image_id,
        seed=seed,
        measurement_seed=measurement_seed,
        strict_deterministic=strict_deterministic,
        batch_chunk_size=batch_chunk_size,
    )


def from_custom(method, task, dataset, image_id=None, num_samples=1, **overrides):
    """Create custom config."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    return METHODS[method].from_custom(task, dataset, image_id, num_samples, **overrides)


__all__ = [
    'BaseConfig',
    'DPSConfig',
    'PDPSConfig',
    'TVConfig',
    'DATASETS',
    'DEFAULT_LIKELIHOOD',
    'DEFAULT_PRIOR',
    'TASKS',
    'from_custom',
    'from_paper',
]
