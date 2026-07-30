"""Process-global strict deterministic runtime configuration."""
import os

import torch


VALID_CUBLAS_WORKSPACE_CONFIGS = {':4096:8', ':16:8'}
_STRICT_MODE_ACTIVATED = False
_STRICT_WORKSPACE_CONFIG = None


def _configure_strict_determinism(config):
    """Enable fail-closed deterministic CUDA execution when requested."""
    global _STRICT_MODE_ACTIVATED
    global _STRICT_WORKSPACE_CONFIG

    if not config.strict_deterministic:
        if _STRICT_MODE_ACTIVATED:
            raise RuntimeError(
                "A non-strict run cannot follow a strict run in the same "
                "Python process because CUDA deterministic settings are "
                "process-global. Start a fresh process."
            )
        return

    if not _STRICT_MODE_ACTIVATED and torch.cuda.is_initialized():
        raise RuntimeError(
            "Strict deterministic mode must be activated before any CUDA "
            "use, even when CUBLAS_WORKSPACE_CONFIG is already set. Start "
            "a fresh process and try again."
        )

    workspace_config = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if workspace_config is None:
        if torch.cuda.is_initialized():
            raise RuntimeError(
                "Strict deterministic mode requires "
                "CUBLAS_WORKSPACE_CONFIG before CUDA is initialized. "
                "Start a fresh process and try again."
            )
        workspace_config = ':4096:8'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = workspace_config
    elif workspace_config not in VALID_CUBLAS_WORKSPACE_CONFIGS:
        raise ValueError(
            "CUBLAS_WORKSPACE_CONFIG must be ':4096:8' or ':16:8' "
            f"for strict deterministic mode; got {workspace_config!r}"
        )
    if (
        _STRICT_WORKSPACE_CONFIG is not None
        and workspace_config != _STRICT_WORKSPACE_CONFIG
    ):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG changed after strict mode was "
            f"activated: {_STRICT_WORKSPACE_CONFIG!r} -> "
            f"{workspace_config!r}. Start a fresh process."
        )

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _STRICT_MODE_ACTIVATED = True
    _STRICT_WORKSPACE_CONFIG = workspace_config


def _strict_runtime_is_configured():
    """Return whether this process activated strict deterministic mode."""
    return _STRICT_MODE_ACTIVATED


def _validate_strict_runtime(config):
    """Refuse a strict run unless the process was configured first."""
    if not getattr(config, 'strict_deterministic', False):
        return

    problems = []
    if not _strict_runtime_is_configured():
        problems.append("strict runtime activation was bypassed")
    if not torch.are_deterministic_algorithms_enabled():
        problems.append("deterministic algorithms are disabled")
    warn_only_enabled = getattr(
        torch,
        'is_deterministic_algorithms_warn_only_enabled',
        lambda: False,
    )()
    if warn_only_enabled:
        problems.append("deterministic algorithms use warn-only mode")
    if not torch.backends.cudnn.deterministic:
        problems.append("cuDNN deterministic mode is disabled")
    if torch.backends.cudnn.benchmark:
        problems.append("cuDNN benchmarking is enabled")
    if (
        os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        not in VALID_CUBLAS_WORKSPACE_CONFIGS
    ):
        problems.append("CUBLAS_WORKSPACE_CONFIG is not deterministic")

    if problems:
        raise RuntimeError(
            "strict_deterministic=True requires the deterministic "
            "runtime configured by src.core.execute; "
            + "; ".join(problems)
        )
