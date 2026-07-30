"""
PDPS command-line interface.
Supports paper reproduction and custom experiments with automatic multi-GPU execution.
"""
from configs import from_paper, from_custom
from src.cli import (
    add_execution_arguments,
    create_parser,
    execute_and_evaluate,
    prepare_common_arguments,
    reject_paper_overrides,
)


def parse_args():
    """Parse PDPS-specific arguments."""
    parser = create_parser(
        description='PDPS sampling for image restoration',
        epilog='Examples:\n'
               '  python pdps.py --paper -d ffhq -m single -t gaussian_deblur -i 097\n'
               '  python pdps.py -t motion_deblur -d ffhq -i 065 -T 0.6 -n 10',
    )

    # PDPS parameters
    parser.add_argument('-T', type=float, help='Diffusion time')
    parser.add_argument(
        '--t0',
        type=float,
        help='Reverse-process terminal time (default: 0.05)',
    )
    parser.add_argument('-n', '--num-samples', type=int, help='Samples per image')
    parser.add_argument('-w', '--warm-steps', type=int, help='Warm-up steps')
    add_execution_arguments(parser)

    return parser.parse_args()


def main():
    args = parse_args()
    execution_overrides = prepare_common_arguments(args)

    # Create PDPS configuration
    if args.paper:
        reject_paper_overrides({
            '-T': args.T,
            '--t0': args.t0,
            '--warm-steps': args.warm_steps,
            '--num-samples': args.num_samples,
        })
        config = from_paper(
            'pdps',
            args.dataset,
            args.mode,
            args.task,
            args.image,
            **execution_overrides,
        )
        print(f"Paper: {config}")
    else:
        if args.T is None:
            raise ValueError("-T required for custom PDPS")

        overrides = {k: v for k, v in {
            'T': args.T,
            'T0': args.t0,
            'warm_steps': args.warm_steps,
        }.items() if v is not None}
        overrides.update(execution_overrides)
        num_samples = (
            1 if args.num_samples is None else args.num_samples
        )
        config = from_custom(
            'pdps',
            args.task,
            args.dataset,
            args.image,
            num_samples,
            **overrides,
        )
        print(f"Custom: {config}")

    execute_and_evaluate(
        config,
        args,
        done_message="Sampling done!",
    )


if __name__ == '__main__':
    main()
