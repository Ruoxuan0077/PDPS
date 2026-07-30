"""
Post-processing: metrics calculation for PDPS results.
Supports both single-image and batch mode evaluation.
"""
import json
import os
from pathlib import Path
import re
import warnings

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


SINGLE_RESULT = re.compile(r'^\d+\.png$')
BATCH_RESULT = re.compile(r'^(\d+)_(\d+)\.png$')


def _manifest_result_files(output_dir, mode):
    """Return the exact declared result set, or None for legacy outputs."""
    output_path = Path(output_dir).resolve()
    manifest_path = output_path / 'run.json'
    if not manifest_path.exists():
        return None

    try:
        with manifest_path.open('r', encoding='utf-8') as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read valid manifest: {manifest_path}") from error

    if not isinstance(manifest, dict):
        raise ValueError("run.json must contain a JSON object")
    if manifest.get('schema_version') != 1:
        raise ValueError(
            f"Unsupported run.json schema: "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get('status') != 'completed':
        raise RuntimeError(
            f"Cannot evaluate run with status "
            f"{manifest.get('status')!r}; expected 'completed'"
        )
    if manifest.get('config', {}).get('mode') != mode:
        raise ValueError("run.json mode does not match the evaluator")

    result_files = manifest.get('expected', {}).get('results')
    if not isinstance(result_files, list) or not result_files:
        raise ValueError("run.json declares no result files")

    pattern = SINGLE_RESULT if mode == 'single' else BATCH_RESULT
    checked = []
    seen = set()
    missing = []
    for name in result_files:
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or len(Path(name).parts) != 1
            or pattern.fullmatch(name) is None
        ):
            raise ValueError(
                f"Unsafe or malformed result path in run.json: {name!r}"
            )
        if name in seen:
            raise ValueError(f"Duplicate result path in run.json: {name}")
        seen.add(name)

        candidate = (output_path / name).resolve()
        if candidate.parent != output_path:
            raise ValueError(
                f"Result path escapes output directory: {name!r}"
            )
        if not candidate.is_file():
            missing.append(name)
        checked.append(name)

    if missing:
        raise FileNotFoundError(
            f"run.json declares {len(missing)} missing results; "
            f"first missing: {missing[0]}"
        )
    return checked


def _legacy_result_files(output_dir, mode):
    warnings.warn(
        "No run.json found; evaluating legacy outputs by filename scan",
        RuntimeWarning,
        stacklevel=2,
    )
    pattern = SINGLE_RESULT if mode == 'single' else BATCH_RESULT
    files = [
        name
        for name in os.listdir(output_dir)
        if pattern.fullmatch(name)
    ]
    if mode == 'single':
        files.sort(key=lambda name: int(Path(name).stem))
    else:
        files.sort(
            key=lambda name: tuple(
                int(part)
                for part in Path(name).stem.split('_')
            )
        )
    if not files:
        raise FileNotFoundError(
            f"No reconstruction PNG files found in {output_dir}"
        )
    return files


def _result_files(output_dir, mode):
    files = _manifest_result_files(output_dir, mode)
    if files is not None:
        return files
    return _legacy_result_files(output_dir, mode)


def compute_metrics(generated, reference):
    """
    Calculate PSNR and SSIM between two images.
    
    Args:
        generated: numpy array [H, W, 3]
        reference: numpy array [H, W, 3]
    
    Returns:
        dict with 'psnr' and 'ssim' keys
    """
    psnr = peak_signal_noise_ratio(reference, generated, data_range=255)
    ssim = structural_similarity(reference, generated, data_range=255, channel_axis=-1)
    return {'psnr': psnr, 'ssim': ssim}


def evaluate_single(output_dir):
    """
    Evaluate single-image mode results.
    
    Expected structure:
        output_dir/label.png, input.png, 0.png, 1.png, ...
    
    Returns:
        dict: {'samples': [...], 'average': {'psnr': ..., 'ssim': ...}}
    """
    reference_path = os.path.join(output_dir, 'label.png')
    if not os.path.isfile(reference_path):
        raise FileNotFoundError(f"Reference image not found: {reference_path}")
    ref_img = np.array(
        Image.open(reference_path).convert('RGB'),
        dtype=np.float32,
    )
    sample_files = _result_files(output_dir, 'single')

    # Calculate metrics for each sample
    results = []
    for fname in sample_files:
        gen_img = np.array(Image.open(os.path.join(output_dir, fname)).convert('RGB'),
                          dtype=np.float32)
        metrics = compute_metrics(gen_img, ref_img)
        results.append({'filename': fname, **metrics})

    # Calculate average
    avg_psnr = np.mean([r['psnr'] for r in results])
    avg_ssim = np.mean([r['ssim'] for r in results])

    return {
        'samples': results,
        'average': {'psnr': avg_psnr, 'ssim': avg_ssim}
    }


def evaluate_batch(output_dir, label_dir):
    """
    Evaluate batch mode results.
    
    Expected structure:
        output_dir/000_0.png, 000_1.png, 001_0.png, ...
        label_dir/000.png, 001.png, ...
    
    Returns:
        dict: {'images': {...}, 'average': {'psnr': ..., 'ssim': ...}}
    """
    result_files = _result_files(output_dir, 'batch')

    images = {}
    for fname in result_files:
        # Parse: "000_0.png" -> image_id='000', sample_id=0
        match = BATCH_RESULT.fullmatch(fname)
        image_id = match.group(1)
        sample_id = int(match.group(2))

        if image_id not in images:
            images[image_id] = []

        # Load and compute
        gen_path = os.path.join(output_dir, fname)
        ref_path = os.path.join(label_dir, f"{image_id}.png")

        if not os.path.isfile(ref_path):
            raise FileNotFoundError(f"Reference image not found: {ref_path}")
        gen_img = np.array(
            Image.open(gen_path).convert('RGB'),
            dtype=np.float32,
        )
        ref_img = np.array(
            Image.open(ref_path).convert('RGB'),
            dtype=np.float32,
        )
        metrics = compute_metrics(gen_img, ref_img)
        images[image_id].append({'sample': sample_id, **metrics})

    # Calculate overall average
    all_psnrs = [m['psnr'] for samples in images.values() for m in samples]
    all_ssims = [m['ssim'] for samples in images.values() for m in samples]

    return {
        'images': images,
        'average': {'psnr': np.mean(all_psnrs), 'ssim': np.mean(all_ssims)}
    }


def save_report(results, output_path, mode='single'):
    """
    Save metrics report to text file.
    
    Args:
        results: dict from evaluate_single() or evaluate_batch()
        output_path: file path to save
        mode: 'single' or 'batch'
    """
    with open(output_path, 'w') as f:
        if mode == 'single':
            f.write("Filename\tPSNR\tSSIM\n")
            for s in results['samples']:
                f.write(f"{s['filename']}\t{s['psnr']:.4f}\t{s['ssim']:.4f}\n")
        else:
            f.write("Image_ID\tSample\tPSNR\tSSIM\n")
            for img_id, samples in sorted(results['images'].items()):
                for s in samples:
                    f.write(f"{img_id}\t{s['sample']}\t{s['psnr']:.4f}\t{s['ssim']:.4f}\n")
        
        f.write(f"\nAverage\t{results['average']['psnr']:.4f}\t{results['average']['ssim']:.4f}\n")
