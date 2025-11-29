"""
Parallel execution manager.
Handles single-GPU and multi-GPU execution for all sampling methods.
"""
import torch
import torch.multiprocessing as mp
from ..utils.io import get_free_gpus, load_image


def _worker(gpu_id, config, start, end):
    """Worker process for multi-GPU execution."""
    device = torch.device(f'cuda:{gpu_id}')
    
    from ..samplers import create_sampler
    from .runner import Runner
    
    sampler = create_sampler(config, device)
    runner = Runner(sampler, config, device)
    label = load_image(config.get_label_path())
    
    if config.mode == 'single':
        label = label.unsqueeze(0) if label.dim() == 3 else label
        num = end - start
        original = config.num_samples
        config.num_samples = num
        runner.run_single(label, start_idx=start)
        config.num_samples = original
    else:
        runner.run_batch(label[start:end], start_idx=start)


def execute(config):
    """
    Execute sampling with automatic GPU detection.
    
    Args:
        config: Config instance (PDPSConfig, DPSConfig, etc.)
    """
    gpu_ids = get_free_gpus()
    print(f"GPUs: {gpu_ids}")
    
    if len(gpu_ids) == 1:
        # Single GPU
        device = torch.device(f'cuda:{gpu_ids[0]}')
        from ..samplers import create_sampler
        from .runner import Runner
        
        sampler = create_sampler(config, device)
        runner = Runner(sampler, config, device)
        runner.run()
    else:
        # Multi-GPU parallel
        if config.mode == 'single':
            total = config.num_samples
            unit = 'samples'
        else:
            total = load_image(config.get_label_path()).shape[0]
            unit = 'images'
        
        chunk = (total + len(gpu_ids) - 1) // len(gpu_ids)
        print(f"Distributing {total} {unit} across {len(gpu_ids)} GPUs")
        
        mp.set_start_method('spawn', force=True)
        processes = []
        
        for i, gpu_id in enumerate(gpu_ids):
            start = i * chunk
            end = min(start + chunk, total)
            if start < end:
                p = mp.Process(target=_worker, args=(gpu_id, config, start, end))
                p.start()
                processes.append(p)
                print(f"GPU {gpu_id}: [{start}, {end})")
        
        for p in processes:
            p.join()

