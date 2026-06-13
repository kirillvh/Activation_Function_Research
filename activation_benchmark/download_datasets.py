from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any


DATASET_NAMES = ("mnist", "cifar10", "cifar100")


def _dataset_classes() -> dict[str, type[Any]]:
    try:
        from torchvision.datasets import CIFAR10, CIFAR100, MNIST
    except ImportError as error:
        raise RuntimeError(
            "torchvision is required to download datasets. "
            "Install requirements.txt first."
        ) from error
    return {
        "mnist": MNIST,
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
    }


def normalize_dataset_names(names: Iterable[str]) -> list[str]:
    normalized = [name.lower().replace("-", "") for name in names]
    if "all" in normalized:
        return list(DATASET_NAMES)
    unknown = sorted(set(normalized) - set(DATASET_NAMES))
    if unknown:
        choices = ", ".join(("all", *DATASET_NAMES))
        raise ValueError(
            f"Unknown datasets: {', '.join(unknown)}. "
            f"Available choices: {choices}"
        )
    return list(dict.fromkeys(normalized))


def download_datasets(
    names: Iterable[str],
    *,
    root: str | Path = "data",
) -> list[str]:
    selected = normalize_dataset_names(names)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    classes = _dataset_classes()

    completed: list[str] = []
    for name in selected:
        dataset_class = classes[name]
        print(f"Downloading and extracting {name} into {root.resolve()}...")
        dataset_class(root=str(root), train=True, download=True)
        dataset_class(root=str(root), train=False, download=True)
        completed.append(name)
        print(f"{name} is ready.")
    return completed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download, verify, and extract TorchVision datasets used by "
            "this project"
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Datasets to prepare: all, mnist, cifar10, cifar100",
    )
    parser.add_argument(
        "--root",
        default="data",
        help="Dataset root used by the YAML configs (default: data)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    completed = download_datasets(args.datasets, root=args.root)
    print("Prepared datasets: " + ", ".join(completed))


if __name__ == "__main__":
    main()
