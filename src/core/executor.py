"""
Parallel execution manager.
Handles single-GPU and multi-GPU execution for all sampling methods.
"""
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from ..utils.io import get_free_gpus, load_image
from .run_manager import RunManager
from .runtime import _configure_strict_determinism


MAX_SEED = 2**63


def _worker_seed(base_seed, start):
    return (base_seed + start) % MAX_SEED


def _run_batch_chunks(
    runner,
    labels,
    *,
    start_idx,
    measurement_total,
    chunk_size,
    skip_existing,
):
    """Run one batch partition without retaining all images on the GPU."""
    partition_size = labels.shape[0]
    if partition_size < 1:
        raise ValueError("Batch partition must contain at least one image")
    if (
        chunk_size is not None
        and (type(chunk_size) is not int or chunk_size < 1)
    ):
        raise ValueError("chunk_size must be a positive integer or None")
    effective_chunk = partition_size if chunk_size is None else chunk_size

    for local_start in range(0, partition_size, effective_chunk):
        local_end = min(local_start + effective_chunk, partition_size)
        runner.run_batch(
            labels[local_start:local_end],
            start_idx=start_idx + local_start,
            measurement_total=measurement_total,
            skip_existing=skip_existing,
        )


def _seed_everything(seed, gpu_id):
    """Seed every RNG used by the reconstruction process."""
    torch.cuda.set_device(gpu_id)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _worker(
    gpu_id,
    config,
    start,
    end,
    total,
    output_dir,
    skip_existing,
    shared_label,
    shared_measurement,
):
    """Worker process for multi-GPU execution."""
    _configure_strict_determinism(config)
    seed = _worker_seed(config.seed, start)
    _seed_everything(seed, gpu_id)
    device = torch.device(f'cuda:{gpu_id}')

    from ..samplers import create_sampler
    from .runner import Runner

    sampler = create_sampler(config, device)
    runner = Runner(
        sampler,
        config,
        device,
        output_dir=output_dir,
    )

    if config.mode == 'single':
        label = shared_label
        label = label.unsqueeze(0) if label.dim() == 3 else label
        runner.run_single(
            label,
            start_idx=start,
            num_samples=end - start,
            measurement=shared_measurement,
            save_common=False,
            skip_existing=skip_existing,
        )
    else:
        labels = load_image(config.get_label_path())
        _run_batch_chunks(
            runner,
            labels[start:end],
            start_idx=start,
            measurement_total=total,
            chunk_size=config.batch_chunk_size,
            skip_existing=skip_existing,
        )


def _input_path(config):
    path = Path(config.get_label_path())
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path


def _total_work(config):
    label_path = _input_path(config)
    if config.mode == 'single':
        if not label_path.is_file():
            raise FileNotFoundError(f"Label image not found: {label_path}")
        return config.num_samples, 'samples'

    if not label_path.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_path}")
    labels = sorted(
        path for path in label_path.iterdir()
        if path.is_file() and path.suffix == '.png'
    )
    if not labels:
        raise ValueError(f"Label directory contains no PNG files: {label_path}")
    return len(labels), 'images'


def _build_partitions(gpu_ids, total, base_seed):
    active_gpu_ids = gpu_ids[:min(len(gpu_ids), total)]
    if not active_gpu_ids:
        raise RuntimeError("No GPUs are available for reconstruction")

    chunk = (total + len(active_gpu_ids) - 1) // len(active_gpu_ids)
    partitions = []
    for index, gpu_id in enumerate(active_gpu_ids):
        start = index * chunk
        end = min(start + chunk, total)
        if start < end:
            partitions.append({
                'gpu_id': gpu_id,
                'start': start,
                'end': end,
                'worker_seed': _worker_seed(base_seed, start),
            })
    return partitions


def _prepare_shared_single(config, gpu_id, output_dir, skip_existing):
    """Create and save the common single-image observation exactly once."""
    _seed_everything(config.seed, gpu_id)
    device = torch.device(f'cuda:{gpu_id}')

    from .runner import Runner

    runner = Runner(
        sampler=None,
        config=config,
        device=device,
        output_dir=output_dir,
    )
    label = load_image(config.get_label_path())
    label = label.unsqueeze(0) if label.dim() == 3 else label
    measurement = runner.prepare_single(
        label,
        save_common=True,
        skip_existing=skip_existing,
    )
    label = label.detach().cpu()
    measurement = measurement.detach().cpu()
    del runner
    torch.cuda.empty_cache()
    return label, measurement


def _terminate_processes(processes, timeout=10.0):
    """Stop and reap every worker that may still be writing outputs."""
    cleanup_errors = []

    for gpu_id, process in processes:
        try:
            if process.pid is not None and process.is_alive():
                process.terminate()
        except Exception as error:
            cleanup_errors.append(
                f"GPU {gpu_id} worker termination failed: {error}"
            )

    deadline = time.monotonic() + timeout
    for gpu_id, process in processes:
        try:
            if process.pid is not None:
                process.join(
                    timeout=max(0.0, deadline - time.monotonic())
                )
        except Exception as error:
            cleanup_errors.append(
                f"GPU {gpu_id} worker join failed: {error}"
            )

    survivors = []
    for gpu_id, process in processes:
        try:
            if process.pid is not None and process.is_alive():
                process.kill()
                survivors.append((gpu_id, process))
        except Exception as error:
            cleanup_errors.append(
                f"GPU {gpu_id} worker kill failed: {error}"
            )

    deadline = time.monotonic() + timeout
    for gpu_id, process in survivors:
        try:
            process.join(
                timeout=max(0.0, deadline - time.monotonic())
            )
            if process.is_alive():
                cleanup_errors.append(
                    f"GPU {gpu_id} worker is still alive after kill"
                )
        except Exception as error:
            cleanup_errors.append(
                f"GPU {gpu_id} worker reap failed: {error}"
            )

    for message in cleanup_errors:
        print(f"Worker cleanup warning: {message}", file=sys.stderr)


def execute(config, *, overwrite=False, resume=False):
    """
    Execute sampling with automatic GPU detection.

    Args:
        config: Config instance (PDPSConfig, DPSConfig, etc.)
        overwrite: replace only this config's exact output directory
        resume: continue only a matching manifest-backed run
    """
    _configure_strict_determinism(config)
    total, unit = _total_work(config)
    gpu_ids = get_free_gpus()
    if not gpu_ids:
        raise RuntimeError("GPU detection returned no usable devices")

    partitions = _build_partitions(gpu_ids, total, config.seed)
    active_gpu_ids = [item['gpu_id'] for item in partitions]
    print(f"Selected CUDA device IDs: {active_gpu_ids}")

    manager = RunManager(
        config,
        total=total,
        partitions=partitions,
    )
    processes = []
    try:
        should_run = manager.prepare(
            overwrite=overwrite,
            resume=resume,
        )
        if not should_run:
            manager.complete()
            print(
                "All manifest-declared outputs already exist; "
                "nothing to resume."
            )
            return str(manager.manifest_path)

        if len(partitions) == 1:
            partition = partitions[0]
            gpu_id = partition['gpu_id']
            print(
                f"CUDA device {gpu_id}: assigned {unit} indices "
                f"[{partition['start']}, {partition['end']})"
            )
            _seed_everything(partition['worker_seed'], gpu_id)
            device = torch.device(f'cuda:{gpu_id}')

            from ..samplers import create_sampler
            from .runner import Runner

            sampler = create_sampler(config, device)
            runner = Runner(
                sampler,
                config,
                device,
                output_dir=manager.output_dir,
            )
            if config.mode == 'single':
                runner.run(skip_existing=resume)
            else:
                labels = load_image(config.get_label_path())
                _run_batch_chunks(
                    runner,
                    labels,
                    start_idx=partition['start'],
                    measurement_total=total,
                    chunk_size=config.batch_chunk_size,
                    skip_existing=resume,
                )
        else:
            print(
                f"Distributing {total} {unit} across "
                f"{len(partitions)} GPUs"
            )
            shared_label = None
            shared_measurement = None
            if config.mode == 'single':
                shared_label, shared_measurement = _prepare_shared_single(
                    config,
                    partitions[0]['gpu_id'],
                    manager.output_dir,
                    resume,
                )

            mp.set_start_method('spawn', force=True)
            for partition in partitions:
                gpu_id = partition['gpu_id']
                start = partition['start']
                end = partition['end']
                process = mp.Process(
                    target=_worker,
                    args=(
                        gpu_id,
                        config,
                        start,
                        end,
                        total,
                        str(manager.output_dir),
                        resume,
                        shared_label,
                        shared_measurement,
                    ),
                )
                processes.append((gpu_id, process))
                process.start()
                print(
                    f"CUDA device {gpu_id}: assigned {unit} indices "
                    f"[{start}, {end})"
                )

            for _, process in processes:
                process.join()

            failed = [
                (gpu_id, process.pid, process.exitcode)
                for gpu_id, process in processes
                if process.exitcode != 0
            ]
            if failed:
                raise RuntimeError(
                    "Sampling workers failed "
                    f"(gpu_id, pid, exitcode): {failed}"
                )

        manager.complete()
    except BaseException as error:
        _terminate_processes(processes)
        try:
            manager.fail(error)
        except Exception as manifest_error:
            print(
                f"Additionally failed to update run.json: {manifest_error}"
            )
        raise
    finally:
        manager.release()

    return str(manager.manifest_path)
