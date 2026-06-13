from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import (
    load_config,
    load_yaml,
    parse_override,
    set_by_path,
)
from .trainer import train_experiment


def parameter_combinations(
    matrix: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    if not matrix:
        raise ValueError("benchmark.matrix cannot be empty")
    keys = list(matrix)
    for key, values in matrix.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"benchmark.matrix.{key} must be a non-empty list")
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(matrix[key] for key in keys))
    ]


def _slug(value: Any) -> str:
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    return text.strip("-") or "value"


def _combination_label(parameters: dict[str, Any]) -> str:
    return ", ".join(
        f"{key.split('.')[-1]}={value}" for key, value in parameters.items()
    )


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(
    rows: list[dict[str, Any]],
    parameter_keys: list[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in parameter_keys)].append(row)

    metrics = (
        "best_validation_accuracy",
        "final_validation_accuracy",
        "test_accuracy",
        "test_loss",
        "final_test_accuracy",
        "final_test_loss",
        "duration_seconds",
    )
    aggregate_rows: list[dict[str, Any]] = []
    for parameter_values, group in groups.items():
        aggregate: dict[str, Any] = dict(zip(parameter_keys, parameter_values))
        aggregate["label"] = _combination_label(
            dict(zip(parameter_keys, parameter_values))
        )
        aggregate["runs"] = len(group)
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            aggregate[f"{metric}_mean"] = mean(values)
            aggregate[f"{metric}_std"] = pstdev(values)
        aggregate_rows.append(aggregate)
    return aggregate_rows


def _plot_bars(
    aggregate: list[dict[str, Any]],
    metric: str,
    title: str,
    ylabel: str,
    path: Path,
    percentage: bool = False,
) -> None:
    labels = [row["label"] for row in aggregate]
    values = [row[f"{metric}_mean"] for row in aggregate]
    errors = [row[f"{metric}_std"] for row in aggregate]
    if percentage:
        values = [value * 100 for value in values]
        errors = [value * 100 for value in errors]

    width = max(7, len(labels) * 1.7)
    figure, axis = plt.subplots(figsize=(width, 5))
    axis.bar(labels, values, yerr=errors, capsize=4)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_learning_curves(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for row in rows:
        history = row.get("history", [])
        if not history:
            continue
        axis.plot(
            [record["epoch"] for record in history],
            [record["validation_accuracy"] * 100 for record in history],
            marker="o",
            label=f"{row['label']} (repeat {row['repeat']})",
        )
    axis.set_title("Validation Accuracy by Epoch")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy (%)")
    axis.grid(alpha=0.25)
    if rows:
        axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    if "benchmark" not in config:
        raise ValueError("Benchmark config must contain a benchmark section")
    benchmark = config["benchmark"]
    combinations = parameter_combinations(benchmark["matrix"])
    repeats = benchmark.get("repeats", 1)
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("benchmark.repeats must be a positive integer")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_dir = (
        Path(benchmark.get("output_dir", "runs/benchmarks"))
        / f"{benchmark.get('name', 'benchmark')}_{timestamp}"
    )
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = benchmark_dir / "runs"
    base_config = load_config(benchmark["base_config"])
    seed_start = benchmark.get(
        "seed_start",
        base_config["experiment"]["seed"],
    )
    if not isinstance(seed_start, int):
        raise ValueError("benchmark.seed_start must be an integer")
    dataset_name = str(base_config["data"].get("dataset", "mnist"))
    dataset_title = (
        dataset_name.upper()
        .replace("CIFAR10", "CIFAR-10")
        .replace("CIFAR100", "CIFAR-100")
    )
    architecture = str(
        base_config["model"].get("architecture", "standard")
    )
    architecture_title = (
        "" if architecture == "standard" else f" {architecture.title()}"
    )
    parameter_keys = list(benchmark["matrix"])
    rows: list[dict[str, Any]] = []

    total_runs = len(combinations) * repeats
    run_number = 0
    for combination_index, parameters in enumerate(combinations):
        for repeat in range(repeats):
            run_number += 1
            training_config = copy.deepcopy(base_config)
            for key, value in parameters.items():
                set_by_path(training_config, key, value)
            training_config["experiment"]["seed"] = (
                seed_start + repeat
            )
            name_parts = [
                f"{key.split('.')[-1]}-{_slug(value)}"
                for key, value in parameters.items()
            ]
            training_config["experiment"]["name"] = (
                f"{combination_index + 1:02d}_{'_'.join(name_parts)}"
                f"_repeat-{repeat + 1}"
            )
            training_config["experiment"]["output_dir"] = str(runs_dir)

            label = _combination_label(parameters)
            print(f"\nBenchmark run {run_number}/{total_runs}: {label}")
            result = train_experiment(training_config)
            row = {
                **parameters,
                "label": label,
                "repeat": repeat + 1,
                **{key: value for key, value in result.items() if key != "history"},
                "history": result["history"],
            }
            rows.append(row)
            csv_rows = [
                {key: value for key, value in item.items() if key != "history"}
                for item in rows
            ]
            _write_csv(csv_rows, benchmark_dir / "runs.csv")

    aggregate = _aggregate(rows, parameter_keys)
    _write_csv(aggregate, benchmark_dir / "aggregate.csv")
    _plot_bars(
        aggregate,
        "test_accuracy",
        f"{dataset_title}{architecture_title} Test Accuracy",
        "Accuracy (%)",
        benchmark_dir / "accuracy.png",
        percentage=True,
    )
    _plot_bars(
        aggregate,
        "duration_seconds",
        "Training Duration",
        "Seconds",
        benchmark_dir / "duration.png",
    )
    _plot_learning_curves(rows, benchmark_dir / "learning_curves.png")

    summary = {
        "benchmark_dir": str(benchmark_dir),
        "base_config": benchmark["base_config"],
        "dataset": dataset_name,
        "architecture": architecture,
        "parameters": parameter_keys,
        "runs": rows,
        "aggregate": aggregate,
    }
    with (benchmark_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nBenchmark reports written to {benchmark_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a parameter benchmark")
    parser.add_argument(
        "--config",
        default="configs/benchmark_activations.yaml",
        help="Path to the benchmark YAML config",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override benchmark config values with dotted key paths",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    for override in args.overrides:
        key, value = parse_override(override)
        set_by_path(config, key, value)
    run_benchmark(config)


if __name__ == "__main__":
    main()
