"""Image IO and GPU discovery helpers."""
import os
from pathlib import Path
import uuid

import numpy as np
from PIL import Image
import torch


def get_free_gpus(mem_threshold=5000):
    """
    Return visible PyTorch device IDs below the used-memory threshold.

    IDs are deliberately taken from PyTorch's logical device namespace, so
    CUDA_VISIBLE_DEVICES remapping is respected. Instantaneous utilization is
    not used as an availability gate because a just-finished kernel can make
    an otherwise usable device fail nondeterministically.
    """
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("No CUDA devices are visible to PyTorch")

    free_gpus = []
    query_errors = []
    for device_id in range(device_count):
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
        except RuntimeError as error:
            query_errors.append(f"cuda:{device_id}: {error}")
            continue
        memory_used = (total_bytes - free_bytes) / (1024 ** 2)
        if memory_used < mem_threshold:
            free_gpus.append(device_id)

    if not free_gpus:
        details = (
            f"; query errors: {query_errors}"
            if query_errors
            else ""
        )
        raise RuntimeError(
            "No visible CUDA device satisfies the configured memory "
            f"threshold (memory used < {mem_threshold} MiB){details}"
        )
    return free_gpus


def _load_single_image(image_path):
    with Image.open(image_path) as image:
        pixels = np.array(image.convert('RGB'))
    tensor = torch.from_numpy(pixels).to(torch.float32)
    tensor = tensor.permute(2, 0, 1)
    return (tensor - 128) / 127.5


def load_image(image_path):
    """Load one RGB PNG or a sorted, nonempty directory of RGB PNGs."""
    image_path = Path(image_path)
    if image_path.is_file():
        return _load_single_image(image_path)
    if image_path.is_dir():
        images = sorted(
            path
            for path in image_path.iterdir()
            if path.is_file() and path.suffix == '.png'
        )
        if not images:
            raise ValueError(
                f"Image directory contains no PNG files: {image_path}"
            )
        return torch.stack(
            [_load_single_image(path) for path in images],
            dim=0,
        )
    raise FileNotFoundError(f"Image path not found: {image_path}")


def save_image(tensor, path, skip=False):
    """Atomically save one RGB tensor in the repository's image scaling."""
    path = Path(path)
    if skip and path.is_file():
        return
    if not torch.is_tensor(tensor):
        raise TypeError("save_image expects a torch.Tensor")

    image = tensor.detach().cpu()
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(
                "save_image accepts only one image, not a tensor batch"
            )
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(
            "save_image expects shape [3, H, W] or [1, 3, H, W]"
        )

    image = (image * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    image = image.permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.parent / (
        f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp'
    )
    try:
        with temporary.open('xb') as handle:
            Image.fromarray(image).save(handle, format='PNG')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
