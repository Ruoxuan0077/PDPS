"""Deterministic interpolation helpers for likelihood gradients."""
import math

import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable


_TRANSPOSE_TABLE_CACHE = {}


def _transpose_table(input_size, output_size, device, dtype):
    """List output contributions to each input coordinate in fixed order."""
    key = (
        input_size,
        output_size,
        device.type,
        device.index,
        dtype,
    )
    cached = _TRANSPOSE_TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    if input_size < 1 or output_size < 1:
        raise ValueError("interpolation dimensions must be positive")

    contributions = [[] for _ in range(input_size)]
    if output_size == 1:
        contributions[0].append((0, 1.0))
    else:
        scale = (input_size - 1) / (output_size - 1)
        for output_index in range(output_size):
            source = (
                float(input_size - 1)
                if output_index == output_size - 1
                else output_index * scale
            )
            lower = min(math.floor(source), input_size - 1)
            upper = min(lower + 1, input_size - 1)
            upper_weight = source - lower
            lower_weight = 1.0 - upper_weight
            if lower_weight != 0.0:
                contributions[lower].append(
                    (output_index, lower_weight)
                )
            if upper != lower and upper_weight != 0.0:
                contributions[upper].append(
                    (output_index, upper_weight)
                )

    width = max(len(items) for items in contributions)
    indices = torch.zeros((input_size, width), dtype=torch.long)
    weights = torch.zeros((input_size, width), dtype=dtype)
    for input_index, items in enumerate(contributions):
        for slot, (output_index, weight) in enumerate(items):
            indices[input_index, slot] = output_index
            weights[input_index, slot] = weight

    table = (
        indices.to(device=device),
        weights.to(device=device),
    )
    _TRANSPOSE_TABLE_CACHE[key] = table
    return table


def _fixed_bilinear_transpose(
    grad_output,
    input_height,
    input_width,
):
    """Apply the bilinear transpose without atomic CUDA accumulation."""
    output_height, output_width = grad_output.shape[-2:]
    height_indices, height_weights = _transpose_table(
        input_height,
        output_height,
        grad_output.device,
        grad_output.dtype,
    )
    width_indices, width_weights = _transpose_table(
        input_width,
        output_width,
        grad_output.device,
        grad_output.dtype,
    )

    width_result = torch.zeros(
        (*grad_output.shape[:-1], input_width),
        device=grad_output.device,
        dtype=grad_output.dtype,
    )
    for slot in range(width_indices.shape[1]):
        selected = grad_output.index_select(
            -1,
            width_indices[:, slot],
        )
        weight = width_weights[:, slot].view(
            *((1,) * (selected.ndim - 1)),
            input_width,
        )
        width_result = width_result + selected * weight

    input_result = torch.zeros(
        (
            *grad_output.shape[:-2],
            input_height,
            input_width,
        ),
        device=grad_output.device,
        dtype=grad_output.dtype,
    )
    for slot in range(height_indices.shape[1]):
        selected = width_result.index_select(
            -2,
            height_indices[:, slot],
        )
        weight = height_weights[:, slot].view(
            *((1,) * (selected.ndim - 2)),
            input_height,
            1,
        )
        input_result = input_result + selected * weight

    return input_result


class _DeterministicBilinearResize(torch.autograd.Function):
    """Native forward with an explicit deterministic first derivative."""

    @staticmethod
    def forward(ctx, input_tensor, size, scale_factor):
        if input_tensor.ndim != 4:
            raise ValueError("bilinear resize expects [B, C, H, W]")
        output = F.interpolate(
            input_tensor,
            size=size,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=True,
        )
        ctx.input_height = input_tensor.shape[-2]
        ctx.input_width = input_tensor.shape[-1]
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return (
            _fixed_bilinear_transpose(
                grad_output,
                ctx.input_height,
                ctx.input_width,
            ),
            None,
            None,
        )


def deterministic_bilinear_resize(
    input_tensor,
    *,
    size=None,
    scale_factor=None,
):
    """Resize with native values and a deterministic CUDA backward."""
    if (size is None) == (scale_factor is None):
        raise ValueError("provide exactly one of size or scale_factor")
    return _DeterministicBilinearResize.apply(
        input_tensor,
        size,
        scale_factor,
    )
