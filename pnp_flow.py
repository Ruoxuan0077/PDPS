"""
PnP-Flow command-line interface.
Supports paper reproduction and custom experiments with automatic multi-GPU execution.
"""
import argparse
import os
from configs import from_paper, from_custom
from src.core import execute
from src.utils import evaluate_single, evaluate_batch, save_report


def parse_args():
    """Parse PnP-Flow-specific arguments."""
    parser = argparse.ArgumentParser(
        description='PnP-Flow sampling for image restoration',
        epilog='Examples:\n'
               '  python pnp_flow.py --paper -d ffhq -m single -t gaussian_deblur -i 097\n'
               '  python pnp_flow.py -t motion_deblur -d ffhq -i 065 --lr-pnp 1.0 -n 10',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--paper', action='store_true', help='Use paper config')
    parser.add_argument('-t', '--task', required=True,
                       choices=['motion_deblur', 'gaussian_deblur', 'nonlinear_deblur'])
    parser.add_argument('-d', '--dataset', required=True, choices=['ffhq', 'afhq'])
    parser.add_argument('-m', '--mode', choices=['single', 'batch'], help='Paper mode')
    parser.add_argument('-i', '--image', help='Image ID for single mode')
    
    # PnP-Flow parameters
    parser.add_argument('--lr-pnp', type=float, help='PnP learning rate')
    parser.add_argument('-n', '--num-samples', type=int, help='Samples per image')
    parser.add_argument('--eval', action='store_true', help='Evaluate results')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create PnP-Flow configuration
    if args.paper:
        if not args.mode:
            raise ValueError("--mode required for paper reproduction")
        config = from_paper('pnp_flow', args.dataset, args.mode, args.task, args.image)
        print(f"Paper: {config}")
    else:
        overrides = {}
        if args.lr_pnp is not None:
            overrides['lr_pnp'] = args.lr_pnp
        
        config = from_custom('pnp_flow', args.task, args.dataset,
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

