"""Command line runner for single-instance DRL baselines."""

import argparse
import csv
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __package__ in (None, ""):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from drl_baselines.registry import get_algorithm, list_algorithms
else:
    from .registry import get_algorithm, list_algorithms

from config import get_config
from utils import load_instance_from_config


def resolve_project_path(path_text: str) -> str:
    """Resolve relative CLI paths against the project root, not the IDE cwd."""
    path = Path(path_text)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def build_parser():
    parser = argparse.ArgumentParser(description="Run one DRL baseline on one public instance.")
    parser.add_argument(
        "--algorithm",
        default="A2C",
        choices=list_algorithms(),
        help="DRL baseline to run. Defaults to A2C for direct IDE execution.",
    )
    parser.add_argument("--n-usvs", type=int, default=2)
    parser.add_argument("--n-tasks", type=int, default=20)
    parser.add_argument("--data-dir", default="data/public")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hgnn-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-trajectories", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument(
        "--no-visdom",
        action="store_true",
        help="Disable Visdom logging. Visdom is enabled by default for DRL baselines.",
    )
    parser.add_argument("--visdom-server", default="http://localhost")
    parser.add_argument("--visdom-port", type=int, default=8097)
    parser.add_argument("--visdom-env", default="drl_baselines")
    return parser


def write_result_csv(path: str, result):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = result.to_dict()
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main():
    args = build_parser().parse_args()
    data_dir = resolve_project_path(args.data_dir)
    model_dir = resolve_project_path(args.model_dir)
    result_dir = resolve_project_path(args.result_dir)

    cfg = get_config(
        n_usvs=args.n_usvs,
        n_tasks=args.n_tasks,
        data_dir=data_dir,
        max_epochs=args.max_epochs,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        hgnn_layers=args.hgnn_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        n_trajectories=args.n_trajectories,
        eval_interval=args.eval_interval,
        use_visdom=not args.no_visdom,
        visdom_server=args.visdom_server,
        visdom_port=args.visdom_port,
        visdom_env=args.visdom_env,
        model_dir=model_dir,
        result_dir=result_dir,
    )
    instance = load_instance_from_config(cfg)
    algorithm = get_algorithm(args.algorithm, seed=args.seed)
    result = algorithm.train(instance, cfg)

    instance_id = instance.get("instance_id", f"u{args.n_usvs}_t{args.n_tasks}")
    result_path = os.path.join(
        result_dir,
        f"drl_{args.algorithm}_{instance_id}_seed{args.seed}.csv",
    )
    write_result_csv(result_path, result)
    print(result.to_dict())
    print(f"[DRL] Result saved: {result_path}")


if __name__ == "__main__":
    main()
