"""
Universal sampling runner.
Supports sampling and deterministic reconstruction through one interface.
"""
import os

import torch

from ..likelihood import get_likelihood, get_operator
from ..utils.io import load_image, save_image
from .runtime import _validate_strict_runtime


class Runner:
    def __init__(self, sampler, config, device, output_dir=None):
        _validate_strict_runtime(config)
        self.sampler = sampler
        self.config = config
        self.device = device
        self.output_dir = str(
            output_dir
            if output_dir is not None
            else config.get_output_dir()
        )

        # Initialize common components
        self._init_models()

    def _init_models(self):
        """Initialize operator and likelihood (common for all methods)."""
        operator = get_operator(device=self.device, **self.config.operator)
        likelihood = get_likelihood(
            operator=operator,
            **self.config.likelihood,
        )

        self.likelihood = likelihood
        self.get_measurement = likelihood.get_label

    @staticmethod
    def _require_finite(tensor, name):
        if not torch.isfinite(tensor).all().item():
            raise FloatingPointError(f"{name} contains NaN or Inf")

    @classmethod
    def _save_finite_image(
        cls,
        tensor,
        path,
        skip_existing=False,
    ):
        """Reject invalid numeric output immediately before writing it."""
        cls._require_finite(tensor, f"Image for {path}")
        save_image(tensor, path, skip=skip_existing)

    def prepare_single(
        self,
        label,
        *,
        save_common=True,
        skip_existing=False,
    ):
        """Create a single-image observation and optionally save it."""
        if label.ndim != 4 or label.shape[0] != 1:
            raise ValueError(
                "Single mode expects label shape [1, C, H, W]"
            )
        self._require_finite(label, "Label")
        measurement = self.get_measurement(
            label,
            seed=self.config.measurement_seed,
            noise_offset=0,
            noise_total=1,
        ).to(self.device)
        self._require_finite(measurement, "Measurement")

        if save_common:
            self._save_finite_image(
                label,
                os.path.join(self.output_dir, 'label.png'),
                skip_existing=skip_existing,
            )
            self._save_finite_image(
                measurement,
                os.path.join(self.output_dir, 'input.png'),
                skip_existing=skip_existing,
            )
        return measurement

    def run_single(
        self,
        label: torch.Tensor,
        start_idx: int = 0,
        num_samples=None,
        measurement=None,
        save_common=True,
        skip_existing=False,
    ) -> torch.Tensor:
        """
        Run sampling for a single image.

        Args:
            label: clean image tensor [1, C, H, W]
            start_idx: starting index for saved samples
            num_samples: samples handled by this worker
            measurement: precomputed common observation, when available
            save_common: whether this worker owns label.png and input.png
            skip_existing: retain already written files during resume

        Returns:
            Final samples tensor [N, C, H, W]
        """
        if self.sampler is None:
            raise RuntimeError("Cannot reconstruct without a sampler")
        if num_samples is None:
            num_samples = self.config.num_samples
        if type(num_samples) is not int or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")

        if measurement is None:
            y = self.prepare_single(
                label,
                save_common=save_common,
                skip_existing=skip_existing,
            )
        else:
            self._require_finite(label, "Label")
            y = measurement.detach().to(self.device)
            self._require_finite(y, "Measurement")
            if save_common:
                self._save_finite_image(
                    label,
                    os.path.join(self.output_dir, 'label.png'),
                    skip_existing=skip_existing,
                )
                self._save_finite_image(
                    y,
                    os.path.join(self.output_dir, 'input.png'),
                    skip_existing=skip_existing,
                )

        # Initialize samples
        # Preserve the original implementation's CPU RNG stream before
        # transferring the initialization to the worker device.
        x = torch.randn(num_samples, *label.shape[1:]).to(self.device)
        y = y.repeat(num_samples, *[1] * (y.dim() - 1))

        # Execute sampling (returns final usable results)
        x_final = self.sampler.sample(
            x,
            y.detach(),
            likelihood=self.likelihood,
            progress_context={
                'mode': 'single',
                'start_idx': start_idx,
                'batch_size': 1,
                'num_samples': num_samples,
                'total_samples': self.config.num_samples,
            },
        )
        if x_final.shape[0] != num_samples:
            raise ValueError(
                f"Sampler returned {x_final.shape[0]} samples, "
                f"expected {num_samples}"
            )
        self._require_finite(x_final, "Reconstruction")

        # Save results
        for index in range(num_samples):
            self._save_finite_image(
                x_final[index].cpu(),
                os.path.join(
                    self.output_dir,
                    f'{start_idx + index}.png',
                ),
                skip_existing=skip_existing,
            )

        return x_final

    def run_batch(
        self,
        labels: torch.Tensor,
        start_idx: int = 0,
        measurement_total=None,
        skip_existing=False,
    ) -> torch.Tensor:
        """
        Run sampling for multiple images.

        Args:
            labels: clean images tensor [B, C, H, W]
            start_idx: starting index for saved images
            measurement_total: global image count for partition-invariant noise
            skip_existing: retain already written files during resume

        Returns:
            Final samples tensor [B*N, C, H, W]
        """
        if self.sampler is None:
            raise RuntimeError("Cannot reconstruct without a sampler")
        if labels.ndim != 4 or labels.shape[0] < 1:
            raise ValueError(
                "Batch mode expects nonempty labels in [B, C, H, W]"
            )
        self._require_finite(labels, "Labels")
        batch_size = labels.shape[0]
        num_samples = self.config.num_samples
        if measurement_total is None:
            measurement_total = batch_size

        # Get measurements
        y = self.get_measurement(
            labels,
            seed=self.config.measurement_seed,
            noise_offset=start_idx,
            noise_total=measurement_total,
        ).to(self.device)
        self._require_finite(y, "Measurements")

        # Save labels and measurements
        for index in range(batch_size):
            self._save_finite_image(
                labels[index],
                os.path.join(
                    self.output_dir,
                    'labels',
                    f'{start_idx + index:03d}.png',
                ),
                skip_existing=skip_existing,
            )
            self._save_finite_image(
                y[index],
                os.path.join(
                    self.output_dir,
                    'inputs',
                    f'{start_idx + index:03d}.png',
                ),
                skip_existing=skip_existing,
            )

        # Initialize samples
        expected_samples = batch_size * num_samples
        # Preserve the original implementation's CPU RNG stream before
        # transferring the initialization to the worker device.
        x = torch.randn(
            expected_samples,
            *labels.shape[1:],
        ).to(self.device)
        y = y.repeat(num_samples, *[1] * (y.dim() - 1))

        # Execute sampling
        x_final = self.sampler.sample(
            x,
            y.detach(),
            likelihood=self.likelihood,
            progress_context={
                'mode': 'batch',
                'start_idx': start_idx,
                'batch_size': batch_size,
                'num_samples': num_samples,
            },
        )
        if x_final.shape[0] != expected_samples:
            raise ValueError(
                f"Sampler returned {x_final.shape[0]} samples, "
                f"expected {expected_samples}"
            )
        self._require_finite(x_final, "Reconstructions")

        # Save results
        for index in range(batch_size):
            for sample in range(num_samples):
                flat_index = index + sample * batch_size
                self._save_finite_image(
                    x_final[flat_index].cpu(),
                    os.path.join(
                        self.output_dir,
                        f'{start_idx + index:03d}_{sample}.png',
                    ),
                    skip_existing=skip_existing,
                )

        return x_final

    def run(self, skip_existing=False):
        """
        Execute sampling based on configuration mode.
        Automatically determines single or batch mode.
        """
        label = load_image(self.config.get_label_path())

        if self.config.mode == 'single':
            label = label.unsqueeze(0) if label.dim() == 3 else label
            self.run_single(label, skip_existing=skip_existing)
        else:
            self.run_batch(
                label,
                measurement_total=label.shape[0],
                skip_existing=skip_existing,
            )
