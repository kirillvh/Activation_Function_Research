from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def set_by_path(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    target = config
    for part in parts[:-1]:
        child = target.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set '{dotted_path}': '{part}' is not a mapping")
        target = child
    target[parts[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"Override must use key=value syntax: {text!r}")
    key, raw_value = text.split("=", 1)
    if not key:
        raise ValueError(f"Override key cannot be empty: {text!r}")
    return key, yaml.safe_load(raw_value)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def load_config(
    path: str | Path,
    overrides: Iterable[str] = (),
) -> dict[str, Any]:
    config = load_yaml(path)
    for override in overrides:
        key, value = parse_override(override)
        set_by_path(config, key, value)
    validate_training_config(config)
    return config


def validate_training_config(config: dict[str, Any]) -> None:
    required_sections = (
        "experiment",
        "data",
        "model",
        "training",
        "checkpoint",
        "tensorboard",
    )
    missing = [name for name in required_sections if name not in config]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")

    positive_values = {
        "data.batch_size": config["data"].get("batch_size"),
        "training.epochs": config["training"].get("epochs"),
        "checkpoint.keep_latest": config["checkpoint"].get("keep_latest"),
    }
    for name, value in positive_values.items():
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    learning_rate = config["training"].get("learning_rate")
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError("training.learning_rate must be positive")

    scheduler = str(config["training"].get("scheduler", "none")).lower()
    if scheduler not in {"none", "cosine", "step", "plateau"}:
        raise ValueError(
            "training.scheduler must be one of: none, cosine, step, plateau"
        )
    minimum_learning_rate = config["training"].get(
        "scheduler_min_learning_rate",
        0.0,
    )
    if (
        not isinstance(minimum_learning_rate, (int, float))
        or minimum_learning_rate < 0
        or minimum_learning_rate > learning_rate
    ):
        raise ValueError(
            "training.scheduler_min_learning_rate must be between zero "
            "and training.learning_rate"
        )

    dataset_name = str(config["data"].get("dataset", "mnist")).lower()
    dataset_training_sizes = {
        "mnist": 60000,
        "cifar10": 50000,
        "cifar100": 50000,
        "synthetic_pqd": None,
        "mini_speech_commands": None,
    }
    if dataset_name not in dataset_training_sizes:
        choices = ", ".join(dataset_training_sizes)
        raise ValueError(
            f"data.dataset must be one of: {choices}"
        )
    training_size = dataset_training_sizes[dataset_name]
    if dataset_name == "mini_speech_commands":
        validation_percentage = config["data"].get(
            "validation_percentage",
            10,
        )
        test_percentage = config["data"].get("test_percentage", 10)
        if (
            not isinstance(validation_percentage, (int, float))
            or not isinstance(test_percentage, (int, float))
            or validation_percentage <= 0
            or test_percentage <= 0
            or validation_percentage + test_percentage >= 100
        ):
            raise ValueError(
                "Mini Speech Commands validation_percentage and "
                "test_percentage must be positive and sum to less than 100"
            )
    else:
        validation_size = config["data"].get("validation_size")
        if not isinstance(validation_size, int) or validation_size < 1:
            raise ValueError("data.validation_size must be a positive integer")
        if training_size is not None and validation_size >= training_size:
            raise ValueError(
                f"data.validation_size must be between 1 and "
                f"{training_size - 1} for {dataset_name}"
            )

    architecture = str(
        config["model"].get("architecture", "standard")
    ).lower()
    allowed_architectures = {
        "mnist": {"standard"},
        "cifar10": {"standard", "deep", "resnet18"},
        "cifar100": {"standard", "deep", "resnet18"},
        "synthetic_pqd": {"signal_cnn"},
        "mini_speech_commands": {"raw_audio_cnn"},
    }
    if architecture not in allowed_architectures[dataset_name]:
        choices = ", ".join(sorted(allowed_architectures[dataset_name]))
        raise ValueError(
            f"model.architecture must be one of: {choices} for {dataset_name}"
        )

    save_every = config["checkpoint"].get("save_every_steps")
    if config["checkpoint"].get("enabled") and (
        not isinstance(save_every, int) or save_every < 1
    ):
        raise ValueError(
            "checkpoint.save_every_steps must be positive when checkpoints are enabled"
        )
    if (
        config["checkpoint"].get("resume")
        and config["training"].get("initial_checkpoint")
    ):
        raise ValueError(
            "checkpoint.resume and training.initial_checkpoint cannot "
            "be used together"
        )


def save_yaml(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def flatten_dict(
    value: dict[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, dict):
            result.update(flatten_dict(child, path))
        else:
            result[path] = child
    return result
