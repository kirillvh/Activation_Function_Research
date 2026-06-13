from __future__ import annotations

import argparse

from .activations import activation_names
from .config import load_config
from .trainer import train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an image classifier")
    parser.add_argument(
        "--config",
        default="configs/mnist.yaml",
        help="Path to the training YAML config",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values with dotted key paths",
    )
    parser.add_argument(
        "--list-activations",
        action="store_true",
        help="Print available activation names and exit",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.list_activations:
        print("\n".join(activation_names()))
        return
    config = load_config(args.config, args.overrides)
    train_experiment(config)


if __name__ == "__main__":
    main()
