"""
Run the ablation variants under the round-robin multi-instance protocol.

Each variant trains one policy over all 25 public instances with the exact
protocol of the main method (see multi_train.py); the 'full' variant comes
from the main `python multi_train.py` run and is not repeated here.
Per-instance results are extracted from the training logs by
extract_results.py (last 10 visits).
"""

import multi_train

DEFAULT_VARIANTS = ['no_hgnn', 'shared_encoder', 'no_reward_norm']


def build_parser():
    parser = multi_train.build_parser()
    parser.description = __doc__
    parser.add_argument('--variants', default=','.join(DEFAULT_VARIANTS),
                        help=f'Comma-separated variants (default: {",".join(DEFAULT_VARIANTS)})')
    return parser


def main(args):
    variants = [item.strip() for item in args.variants.split(',') if item.strip()]
    results = {}
    for variant in variants:
        print("=" * 80)
        print(f"[Ablation] Variant: {variant}")
        args.variant = variant
        args.visdom_env = f'usv_ablation_{variant}'
        results[variant] = multi_train.main(args)
    return results


if __name__ == '__main__':
    main(build_parser().parse_args())
