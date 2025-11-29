"""
Post-processing: metrics calculation for PDPS results.
Supports both single-image and batch mode evaluation.
"""
import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


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
    ref_img = np.array(Image.open(os.path.join(output_dir, 'label.png')).convert('RGB'), 
                       dtype=np.float32)
    
    # Find sample files (exclude label.png and input.png)
    sample_files = sorted([
        f for f in os.listdir(output_dir)
        if f.endswith('.png') and f not in {'label.png', 'input.png'}
    ], key=lambda x: int(x.split('.')[0]))
    
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
    result_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    
    images = {}
    for fname in result_files:
        # Parse: "000_0.png" -> image_id='000', sample_id=0
        parts = fname.split('_')
        image_id = parts[0]
        sample_id = int(parts[1].split('.')[0])
        
        if image_id not in images:
            images[image_id] = []
        
        # Load and compute
        gen_path = os.path.join(output_dir, fname)
        ref_path = os.path.join(label_dir, f"{image_id}.png")
        
        if os.path.exists(ref_path):
            gen_img = np.array(Image.open(gen_path).convert('RGB'), dtype=np.float32)
            ref_img = np.array(Image.open(ref_path).convert('RGB'), dtype=np.float32)
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

