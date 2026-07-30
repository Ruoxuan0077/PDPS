"""Proximal-gradient reconstruction with a total-variation prior."""
import torch
from torch.nn import functional as F

from .base import BaseSampler


def _load_tv_prior():
    """Import DeepInv's TV prior without patching the installed package."""
    schedulers = torch.optim.lr_scheduler
    if not hasattr(schedulers, 'LRScheduler'):
        legacy_scheduler = getattr(schedulers, '_LRScheduler', None)
        if legacy_scheduler is None:
            raise ImportError(
                "This PyTorch installation exposes neither LRScheduler "
                "nor _LRScheduler"
            )
        schedulers.LRScheduler = legacy_scheduler

    try:
        from deepinv.optim.prior import TVPrior
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "TV reconstruction requires deepinv==0.3.2"
        ) from error

    return TVPrior


class TVSampler(BaseSampler):
    """Deterministic PGD solver for data fidelity plus isotropic TV."""

    def __init__(self, config, device):
        super().__init__(config, device)

        TVPrior = _load_tv_prior()

        params = config.params
        self.lambda_tv = float(params['lambda_tv'])
        self.stepsize = float(params['stepsize'])
        self.max_iter = int(params['max_iter'])
        self.tv_prior = TVPrior(
            n_it_max=int(params['tv_inner_max_iter']),
            def_crit=float(params['tv_inner_tol']),
        )

    def _ensure_finite(self, tensor, name, iteration=None):
        if torch.isfinite(tensor).all().item():
            return

        location = "" if iteration is None else f" at iteration {iteration}"
        raise FloatingPointError(
            f"TV task {self.config.task}: "
            f"{name} contains NaN or Inf{location}"
        )

    @staticmethod
    def _freeze_operator_modules(operator):
        """Avoid tracking gradients for fixed operator networks and kernels."""
        for value in vars(operator).values():
            if isinstance(value, torch.nn.Module):
                value.eval()
                for parameter in value.parameters():
                    parameter.requires_grad_(False)

    def _reset_tv_prox(self):
        """Reset DeepInv's warm start at the beginning of each problem."""
        model = self.tv_prior.TVModel
        model.restart = True
        model.x2 = None
        model.u2 = None

    @staticmethod
    def _initialize_from_measurement(y, target_shape):
        """Use the measurement as x0, resizing only when dimensions differ."""
        if y.shape[-2:] == target_shape:
            return y.detach().clone()

        return F.interpolate(
            y,
            size=target_shape,
            mode='bilinear',
            align_corners=True,
        ).detach()

    def sample(self, x, y, likelihood=None, **kwargs):
        """Run a fixed number of proximal-gradient iterations."""
        if likelihood is None:
            raise ValueError("TV reconstruction requires a likelihood")
        if x.ndim != 4 or y.ndim != 4:
            raise ValueError("TV expects x and y in [B, C, H, W] format")
        if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
            raise ValueError(
                "TV requires x and y to have matching batch and channel axes"
            )

        operator = likelihood.operator
        self._freeze_operator_modules(operator)

        y = y.detach().to(self.device)
        self._ensure_finite(y, "measurement")
        current = self._initialize_from_measurement(y, x.shape[-2:])
        self._ensure_finite(current, "initialization")
        self._reset_tv_prox()

        report_every = max(1, self.max_iter // 10)
        for iteration in range(1, self.max_iter + 1):
            with torch.enable_grad():
                z = current.detach().requires_grad_(True)
                residual = operator.forward(z) - y
                self._ensure_finite(residual, "residual", iteration)

                data_losses = 0.5 * residual.flatten(1).square().sum(1)
                gradient = torch.autograd.grad(
                    data_losses.sum(),
                    z,
                    create_graph=False,
                )[0]

            self._ensure_finite(gradient, "data gradient", iteration)
            candidate = (z - self.stepsize * gradient).detach()
            self._ensure_finite(candidate, "gradient step", iteration)

            with torch.no_grad():
                current = self.tv_prior.prox(
                    candidate,
                    gamma=self.stepsize * self.lambda_tv,
                ).detach()

            self._ensure_finite(current, "proximal output", iteration)
            if iteration == 1 or iteration % report_every == 0:
                mean_loss = data_losses.detach().mean().item()
                print(
                    f"[TV] iteration {iteration}/{self.max_iter}, "
                    f"data fidelity={mean_loss:.6e}"
                )

        return current
