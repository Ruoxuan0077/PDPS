"""
Total-variation reconstruction command-line interface.
Uses the shared operators, runner, and evaluation code.
"""
from configs import from_custom, from_paper
from src.cli import (
    add_execution_arguments,
    create_parser,
    execute_and_evaluate,
    prepare_common_arguments,
    reject_paper_overrides,
)


def parse_args():
    """Parse TV-specific arguments."""
    parser = create_parser(
        description='PGD-TV reconstruction for image restoration',
        epilog='Examples:\n'
               '  python tv.py --paper -d ffhq -m single '
               '-t gaussian_deblur -i 097\n'
               '  python tv.py -d ffhq -t motion_deblur -i 065 '
               '--lambda-tv 0.01 --stepsize 1.0 --max-iter 150',
    )

    parser.add_argument('--lambda-tv', type=float, help='TV weight')
    parser.add_argument('--stepsize', type=float, help='PGD step size')
    parser.add_argument('--max-iter', type=int, help='PGD iterations')
    parser.add_argument(
        '--tv-inner-iters',
        type=int,
        help='Maximum iterations for each TV prox',
    )
    parser.add_argument(
        '--tv-inner-tol',
        type=float,
        help='Convergence tolerance for each TV prox',
    )
    parser.add_argument(
        '-n',
        '--num-samples',
        type=int,
        help='Must be 1 because TV is deterministic',
    )
    add_execution_arguments(parser)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_samples not in (None, 1):
        raise ValueError(
            "TV is deterministic; --num-samples must be 1"
        )
    execution_overrides = prepare_common_arguments(args)

    if args.paper:
        reject_paper_overrides({
            '--lambda-tv': args.lambda_tv,
            '--stepsize': args.stepsize,
            '--max-iter': args.max_iter,
            '--tv-inner-iters': args.tv_inner_iters,
            '--tv-inner-tol': args.tv_inner_tol,
            '--num-samples': args.num_samples,
        })
        config = from_paper(
            'tv',
            args.dataset,
            args.mode,
            args.task,
            args.image,
            **execution_overrides,
        )
        print(f"Paper: {config}")
    else:
        overrides = {
            key: value
            for key, value in {
                'lambda_tv': args.lambda_tv,
                'stepsize': args.stepsize,
                'max_iter': args.max_iter,
                'tv_inner_max_iter': args.tv_inner_iters,
                'tv_inner_tol': args.tv_inner_tol,
            }.items()
            if value is not None
        }
        overrides.update(execution_overrides)
        num_samples = 1 if args.num_samples is None else args.num_samples
        config = from_custom(
            'tv',
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
        done_message="Reconstruction done!",
    )


if __name__ == '__main__':
    main()
