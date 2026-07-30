"""Shared command-line plumbing for method-specific entry scripts."""
import argparse
import os

from configs import DATASETS, TASKS

from .core import execute
from .utils import evaluate_batch, evaluate_single, save_report


TASK_CHOICES = tuple(sorted(TASKS))
DATASET_CHOICES = tuple(sorted(DATASETS))


def create_parser(*, description, epilog):
    """Create a parser with arguments shared by every reconstruction method."""
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--paper',
        action='store_true',
        help='Use paper config',
    )
    parser.add_argument(
        '-t',
        '--task',
        required=True,
        choices=TASK_CHOICES,
    )
    parser.add_argument(
        '-d',
        '--dataset',
        required=True,
        choices=DATASET_CHOICES,
    )
    parser.add_argument(
        '-m',
        '--mode',
        choices=['single', 'batch'],
        help='Paper mode',
    )
    parser.add_argument(
        '-i',
        '--image',
        help='Image ID for single mode',
    )
    return parser


def add_execution_arguments(parser):
    """Add execution, output, and evaluation arguments to ``parser``."""
    parser.add_argument(
        '--seed',
        type=int,
        help='Sampling seed; generated and recorded when omitted',
    )
    parser.add_argument(
        '--measurement-seed',
        type=int,
        help='Measurement-noise seed (default: 42)',
    )
    parser.add_argument(
        '--strict-deterministic',
        action='store_true',
        help=(
            'Enable fail-closed deterministic PyTorch/CUDA mode; '
            'may be slower and rejects unsupported operations'
        ),
    )
    parser.add_argument(
        '--batch-chunk-size',
        type=int,
        help=(
            'Process batch-mode images in chunks of this size; '
            'recorded as part of the run configuration'
        ),
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        '--overwrite',
        action='store_true',
        help='Replace this exact output directory',
    )
    output_group.add_argument(
        '--resume',
        action='store_true',
        help='Resume a matching manifest-backed run',
    )
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Evaluate results',
    )
    return parser


def prepare_common_arguments(args):
    """Validate common CLI relationships and collect config overrides."""
    if args.mode == 'single' and args.image is None:
        raise ValueError("single mode requires --image")
    if args.mode == 'batch' and args.image is not None:
        raise ValueError("batch mode does not accept --image")
    if args.paper and not args.mode:
        raise ValueError("--mode required for paper reproduction")

    return {
        key: value
        for key, value in {
            'seed': args.seed,
            'measurement_seed': args.measurement_seed,
            'strict_deterministic': args.strict_deterministic,
            'batch_chunk_size': args.batch_chunk_size,
        }.items()
        if value is not None
    }


def reject_paper_overrides(overrides):
    """Reject algorithm overrides that would obscure a paper preset."""
    incompatible = [
        name for name, value in overrides.items()
        if value is not None
    ]
    if incompatible:
        raise ValueError(
            "Paper presets do not accept algorithm overrides: "
            f"{', '.join(incompatible)}"
        )


def execute_and_evaluate(config, args, *, done_message):
    """Execute one configured run and optionally evaluate its outputs."""
    output_dir = config.get_output_dir()
    print(f"Output: {output_dir}")
    execute(
        config,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    print(done_message)

    if not args.eval:
        return

    print("\nEvaluating results...")
    if config.mode == 'single':
        results = evaluate_single(output_dir)
    else:
        results = evaluate_batch(
            output_dir,
            config.get_label_path(),
        )

    print(f"Average PSNR: {results['average']['psnr']:.4f}")
    print(f"Average SSIM: {results['average']['ssim']:.4f}")

    report_path = os.path.join(output_dir, 'metrics.txt')
    save_report(results, report_path, config.mode)
    print(f"Report saved: {report_path}")
