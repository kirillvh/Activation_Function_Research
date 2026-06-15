from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


DATASET_NAMES = ("mnist", "cifar10", "cifar100", "mini_speech_commands")
MINI_SPEECH_COMMANDS_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "mini_speech_commands.zip"
)
MINI_SPEECH_COMMANDS_CLASSES = (
    "down",
    "go",
    "left",
    "no",
    "right",
    "stop",
    "up",
    "yes",
)


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
    aliases = {
        "minispeechcommands": "mini_speech_commands",
        "speechcommands": "mini_speech_commands",
    }
    normalized = []
    for name in names:
        compact = name.lower().replace("-", "").replace("_", "")
        normalized.append(aliases.get(compact, name.lower().replace("-", "")))
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


def _mini_speech_commands_ready(dataset_dir: Path) -> bool:
    return all(
        (dataset_dir / class_name).is_dir()
        and any((dataset_dir / class_name).glob("*.wav"))
        for class_name in MINI_SPEECH_COMMANDS_CLASSES
    )


def download_mini_speech_commands(
    root: str | Path = "data",
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dataset_dir = root / "mini_speech_commands"
    if _mini_speech_commands_ready(dataset_dir):
        return dataset_dir

    archive_path = root / "mini_speech_commands.zip"
    print(
        "Downloading Mini Speech Commands from TensorFlow "
        f"to {archive_path.resolve()}..."
    )
    with urllib.request.urlopen(MINI_SPEECH_COMMANDS_URL) as response:
        with archive_path.open("wb") as archive:
            shutil.copyfileobj(response, archive)

    root_resolved = root.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (root / member.filename).resolve()
            if not destination.is_relative_to(root_resolved):
                raise ValueError(
                    f"Unsafe path in Mini Speech Commands archive: "
                    f"{member.filename}"
                )
        archive.extractall(root)
    archive_path.unlink()

    if not _mini_speech_commands_ready(dataset_dir):
        raise RuntimeError(
            "Mini Speech Commands download did not create the expected "
            f"dataset at {dataset_dir}"
        )
    return dataset_dir


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
        if name == "mini_speech_commands":
            print(
                "Downloading and extracting mini_speech_commands into "
                f"{root.resolve()}..."
            )
            download_mini_speech_commands(root)
            completed.append(name)
            print(f"{name} is ready.")
            continue
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
            "Download, verify, and extract image and audio datasets used "
            "by this project"
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help=(
            "Datasets to prepare: all, mnist, cifar10, cifar100, "
            "mini_speech_commands"
        ),
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
