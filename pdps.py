"""
PDPS command-line interface.
Supports paper reproduction and custom experiments with automatic multi-GPU execution.
"""
import argparse
import os
from configs import from_paper, from_custom
from src.core import execute
from src.utils import evaluate_single, evaluate_batch, save_report


def parse_args():
    """Parse PDPS-specific arguments."""
    parser = argparse.ArgumentParser(
        description='PDPS sampling for image restoration',
        epilog='Examples:\n'
               '  python pdps.py --paper -d ffhq -m single -t gaussian_deblur -i 097\n'
               '  python pdps.py -t motion_deblur -d ffhq -i 065 -T 0.6 -n 10',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--paper', action='store_true', help='Use paper config')
    parser.add_argument('-t', '--task', required=True, 
                        choices=['motion_deblur', 'gaussian_deblur', 'nonlinear_deblur'])
    parser.add_argument('-d', '--dataset', required=True, choices=['ffhq', 'afhq'])
    parser.add_argument('-m', '--mode', choices=['single', 'batch'], help='Paper mode')
    parser.add_argument('-i', '--image', help='Image ID for single mode')
    
    # PDPS parameters
    parser.add_argument('-T', type=float, help='Diffusion time')
    parser.add_argument('-n', '--num-samples', type=int, help='Samples per image')
    parser.add_argument('-w', '--warm-steps', type=int, help='Warm-up steps')
    parser.add_argument('--eval', action='store_true', help='Evaluate results')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create PDPS configuration
    if args.paper:
        if not args.mode:
            raise ValueError("--mode required for paper reproduction")
        config = from_paper('pdps', args.dataset, args.mode, args.task, args.image)
        print(f"Paper: {config}")
    else:
        if not args.T:
            raise ValueError("-T required for custom PDPS")
        
        overrides = {k: v for k, v in {
            'T': args.T,
            'warm_steps': args.warm_steps,
        }.items() if v is not None}
        
        config = from_custom('pdps', args.task, args.dataset,
                           args.image, args.num_samples or 1, **overrides)
        print(f"Custom: {config}")
    
    print(f"Output: {config.get_output_dir()}")
    
    # Execute sampling
    execute(config)
    print("Sampling done!")
    
    # Evaluate results
    if args.eval:
        print("\nEvaluating results...")
        
        if config.mode == 'single':
            results = evaluate_single(config.get_output_dir())
        else:
            results = evaluate_batch(config.get_output_dir(), config.get_label_path())
        
        print(f"Average PSNR: {results['average']['psnr']:.4f}")
        print(f"Average SSIM: {results['average']['ssim']:.4f}")
        
        report_path = os.path.join(config.get_output_dir(), 'metrics.txt')
        save_report(results, report_path, config.mode)
        print(f"Report saved: {report_path}")


if __name__ == '__main__':
    main()
