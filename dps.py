"""
DPS command-line interface.
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
    """Parse DPS-specific arguments."""
    parser = create_parser(
        description='DPS sampling for image restoration',
        epilog='Examples:\n'
               '  python dps.py --paper -d ffhq -m single -t gaussian_deblur -i 097\n'
               '  python dps.py -t motion_deblur -d ffhq -i 065 --scale 1.5 -n 5',
    )

    # DPS parameters
    parser.add_argument('--scale', type=float, help='Guidance scale')
    parser.add_argument('--steps', type=int, help='Diffusion steps')
    parser.add_argument('-n', '--num-samples', type=int, help='Samples per image')
    add_execution_arguments(parser)

    return parser.parse_args()


def main():
    args = parse_args()
    execution_overrides = prepare_common_arguments(args)

    # Create DPS configuration
    if args.paper:
        reject_paper_overrides({
            '--scale': args.scale,
            '--steps': args.steps,
            '--num-samples': args.num_samples,
        })
        config = from_paper(
            'dps',
            args.dataset,
            args.mode,
            args.task,
            args.image,
            **execution_overrides,
        )
        print(f"Paper: {config}")
    else:
        overrides = {k: v for k, v in {
            'scale': args.scale,
            'steps': args.steps,
        }.items() if v is not None}
        overrides.update(execution_overrides)
        num_samples = (
            1 if args.num_samples is None else args.num_samples
        )
        config = from_custom(
            'dps',
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
