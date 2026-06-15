from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import load_config, load_yaml, parse_override, set_by_path
from .multistart import (
    _latest_checkpoint,
    checkpoint_peuaf_frequencies,
    run_evolutionary_multistart,
    set_initial_peuaf_frequency,
)
from .trainer import train_experiment

CONDITIONS = (
    "gelu",
    "periodic_gelu",
    "direct_peuaf",
    "evolved_peuaf",
)
LABELS = {
    "gelu": "GELU",
    "periodic_gelu": "Periodic GELU",
    "direct_peuaf": "Direct PEUAF",
    "evolved_peuaf": "Evolved PEUAF",
}
COLORS = {
    "gelu": "#0072b2",
    "periodic_gelu": "#009e73",
    "direct_peuaf": "#777777",
    "evolved_peuaf": "#d55e00",
}


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


def _without_history(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "history"}


def _training_config(
    base: dict[str, Any],
    *,
    seed: int,
    condition: str,
    output_dir: Path,
    epochs: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["experiment"]["seed"] = seed
    config["experiment"]["name"] = f"seed-{seed}_{condition}"
    config["experiment"]["output_dir"] = str(output_dir)
    config["training"]["epochs"] = epochs
    config["training"]["evaluate_test"] = True
    config["checkpoint"]["resume"] = None
    config["tensorboard"]["enabled"] = False
    return config


def _final_peuaf_frequencies(result: dict[str, Any]) -> dict[str, float]:
    checkpoint = _latest_checkpoint(result["run_dir"])
    frequencies = checkpoint_peuaf_frequencies(checkpoint, weights="best")
    return {
        "final_frequency_min": min(frequencies),
        "final_frequency_mean": float(np.mean(frequencies)),
        "final_frequency_max": max(frequencies),
    }


def _run_direct(
    base: dict[str, Any],
    *,
    condition: str,
    seed: int,
    epochs: int,
    output_dir: Path,
) -> dict[str, Any]:
    config = _training_config(
        base,
        seed=seed,
        condition=condition,
        output_dir=output_dir,
        epochs=epochs,
    )
    if condition == "gelu":
        config["model"]["activation"] = "gelu"
        config["model"]["activation_kwargs"] = {}
    elif condition == "periodic_gelu":
        config["model"]["activation"] = "gelu_sine_triangle"
        config["model"]["activation_kwargs"] = {}
    elif condition == "direct_peuaf":
        config["model"]["activation"] = "peuaf"
        config["model"]["activation_kwargs"] = {}
        set_initial_peuaf_frequency(config, 0.5)
    else:
        raise ValueError(f"Unsupported direct condition: {condition}")

    started = time.perf_counter()
    result = train_experiment(config)
    row = {
        "condition": condition,
        "label": LABELS[condition],
        "seed": seed,
        "initial_frequency": 0.5 if condition == "direct_peuaf" else None,
        "selected_frequency": 0.5 if condition == "direct_peuaf" else None,
        "search_epoch_equivalents": 0,
        "selected_path_epochs": epochs,
        "total_duration_seconds": time.perf_counter() - started,
        "history": result["history"],
        **_without_history(result),
    }
    if condition == "direct_peuaf":
        row.update(_final_peuaf_frequencies(result))
    return row


def _run_evolved(
    base_config_path: str,
    study: dict[str, Any],
    *,
    seed: int,
    epochs: int,
    warmup_epochs: int,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    search = copy.deepcopy(study["search"])
    search["seed"] = seed
    search["warmup_epochs"] = warmup_epochs
    config = {
        "name": f"seed-{seed}_evolved_peuaf",
        "base_config": base_config_path,
        "base_overrides": {
            "model.activation": "peuaf",
            "model.activation_kwargs": {},
        },
        "output_dir": str(output_dir),
        "search": search,
        "final_training": {
            "epochs": epochs,
            "warm_start_selected": False,
        },
    }
    started = time.perf_counter()
    summary = run_evolutionary_multistart(config)
    result = summary["final_result"]
    selected = summary["selected_candidate"]
    population_size = int(search.get("population_size", 4))
    generations = int(search.get("generations", 2))
    row = {
        "condition": "evolved_peuaf",
        "label": LABELS["evolved_peuaf"],
        "seed": seed,
        "initial_frequency": 0.5,
        "selected_frequency": float(selected["initial_frequency"]),
        "warmup_frequency_mean": float(selected["learned_frequency_mean"]),
        "selected_generation": selected["generation"],
        "search_epoch_equivalents": (
            population_size * generations * warmup_epochs
        ),
        "selected_path_epochs": epochs,
        "total_duration_seconds": time.perf_counter() - started,
        "history": result["history"],
        **_without_history(result),
    }
    row.update(_final_peuaf_frequencies(result))

    selected_checkpoint = selected["checkpoint"]
    candidates = [
        {
            "seed": seed,
            "generation": candidate["generation"],
            "candidate": candidate["candidate"],
            "initial_frequency": candidate["initial_frequency"],
            "learned_frequency_mean": candidate["learned_frequency_mean"],
            "best_validation_accuracy": candidate[
                "best_validation_accuracy"
            ],
            "selected": int(candidate["checkpoint"] == selected_checkpoint),
        }
        for candidate in summary["candidate_runs"]
    ]
    return row, candidates


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate = []
    for condition in CONDITIONS:
        group = [row for row in rows if row["condition"] == condition]
        if not group:
            continue
        item: dict[str, Any] = {
            "condition": condition,
            "label": LABELS[condition],
            "runs": len(group),
        }
        for metric in (
            "best_validation_accuracy",
            "test_accuracy",
            "final_test_accuracy",
            "total_duration_seconds",
        ):
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = pstdev(values)
        aggregate.append(item)
    return aggregate


def _paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (int(row["seed"]), row["condition"]): row
        for row in rows
    }
    paired = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        accuracies = {
            condition: 100
            * float(lookup[(seed, condition)]["test_accuracy"])
            for condition in CONDITIONS
        }
        paired.append(
            {
                "seed": seed,
                **{
                    f"{condition}_accuracy": accuracies[condition]
                    for condition in CONDITIONS
                },
                "periodic_gelu_minus_gelu": (
                    accuracies["periodic_gelu"] - accuracies["gelu"]
                ),
                "direct_peuaf_minus_gelu": (
                    accuracies["direct_peuaf"] - accuracies["gelu"]
                ),
                "evolved_minus_direct": (
                    accuracies["evolved_peuaf"]
                    - accuracies["direct_peuaf"]
                ),
                "evolved_minus_gelu": (
                    accuracies["evolved_peuaf"] - accuracies["gelu"]
                ),
            }
        )
    return paired


def _plot_accuracy(
    rows: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9.5, 5.5))
    positions = np.arange(len(aggregate))
    values = [100 * row["test_accuracy_mean"] for row in aggregate]
    errors = [100 * row["test_accuracy_std"] for row in aggregate]
    bars = axis.bar(
        positions,
        values,
        yerr=errors,
        capsize=5,
        color=[COLORS[row["condition"]] for row in aggregate],
        edgecolor="#333333",
        linewidth=[
            1.8 if row["condition"] in {"periodic_gelu", "evolved_peuaf"}
            else 1.0
            for row in aggregate
        ],
        zorder=2,
    )
    rng = np.random.default_rng(1234)
    for index, item in enumerate(aggregate):
        group = [
            row for row in rows if row["condition"] == item["condition"]
        ]
        axis.scatter(
            index + rng.uniform(-0.07, 0.07, len(group)),
            [100 * float(row["test_accuracy"]) for row in group],
            color="white",
            edgecolor="#222222",
            s=36,
            zorder=4,
        )
    for bar, value in zip(bars, values):
        axis.annotate(
            f"{value:.2f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
        )
    axis.set_xticks(positions, [row["label"] for row in aggregate])
    axis.set_ylabel("Validation-selected test accuracy (%)")
    axis.set_title("Mini Speech Commands Raw-Waveform Comparison")
    axis.grid(axis="y", alpha=0.25, zorder=0)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_paired(rows: list[dict[str, Any]], path: Path) -> None:
    seeds = sorted({int(row["seed"]) for row in rows})
    lookup = {
        (int(row["seed"]), row["condition"]): (
            100 * float(row["test_accuracy"])
        )
        for row in rows
    }
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for seed in seeds:
        axis.plot(
            range(len(CONDITIONS)),
            [lookup[(seed, condition)] for condition in CONDITIONS],
            color="#aaaaaa",
            alpha=0.65,
            marker="o",
            linewidth=1.3,
            label=f"Seed {seed}",
            zorder=2,
        )
    means = [
        mean(lookup[(seed, condition)] for seed in seeds)
        for condition in CONDITIONS
    ]
    axis.plot(
        range(len(CONDITIONS)),
        means,
        color=COLORS["evolved_peuaf"],
        marker="o",
        markersize=9,
        linewidth=3.5,
        label="Mean across paired seeds",
        zorder=5,
    )
    axis.set_xticks(
        range(len(CONDITIONS)),
        [LABELS[condition] for condition in CONDITIONS],
    )
    axis.set_ylabel("Validation-selected test accuracy (%)")
    axis.set_title("Paired Seeds on Mini Speech Commands")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_learning_curves(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for condition in CONDITIONS:
        group = groups[condition]
        epochs = sorted(
            {
                int(record["epoch"])
                for row in group
                for record in row["history"]
            }
        )
        values_by_epoch = [
            [
                100 * float(record["validation_accuracy"])
                for row in group
                for record in row["history"]
                if int(record["epoch"]) == epoch
            ]
            for epoch in epochs
        ]
        means = np.asarray([mean(values) for values in values_by_epoch])
        deviations = np.asarray(
            [pstdev(values) for values in values_by_epoch]
        )
        proposed = condition in {"periodic_gelu", "evolved_peuaf"}
        axis.plot(
            epochs,
            means,
            color=COLORS[condition],
            linewidth=3.0 if proposed else 2.0,
            label=f"{LABELS[condition]} (mean +/- SD)",
            zorder=5 if proposed else 3,
        )
        axis.fill_between(
            epochs,
            means - deviations,
            means + deviations,
            color=COLORS[condition],
            alpha=0.18 if proposed else 0.10,
            linewidth=0,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation accuracy (%)")
    axis.set_title("Raw-Waveform Audio Validation Accuracy")
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_frequency_parameters(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True)
    for condition in ("direct_peuaf", "evolved_peuaf"):
        group = [row for row in rows if row["condition"] == condition]
        epochs = [int(record["epoch"]) for record in group[0]["history"]]
        means = [
            mean(
                float(row["history"][index]["peuaf_frequency_mean"])
                for row in group
            )
            for index in range(len(epochs))
        ]
        axes[0].plot(
            epochs,
            means,
            color=COLORS[condition],
            linewidth=2.5,
            label=LABELS[condition],
        )
    periodic = [
        row for row in rows if row["condition"] == "periodic_gelu"
    ]
    epochs = [int(record["epoch"]) for record in periodic[0]["history"]]
    for metric, label, color in (
        ("sine_triangle_frequency_mean", "Frequency w", "#009e73"),
        ("sine_triangle_blend_mean", "Triangle blend b", "#cc79a7"),
        (
            "sine_triangle_periodic_scale_mean",
            "Periodic amplitude a",
            "#e69f00",
        ),
    ):
        values = [
            mean(float(row["history"][index][metric]) for row in periodic)
            for index in range(len(epochs))
        ]
        axes[1].plot(epochs, values, color=color, linewidth=2.3, label=label)
    axes[0].set_ylabel("PEUAF frequency")
    axes[0].set_title("Learned Periodic Activation Parameters")
    axes[0].legend()
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Periodic GELU parameter")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_candidates(
    candidates: list[dict[str, Any]],
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5.5))
    for generation, color in ((1, "#9467bd"), (2, "#e67e22")):
        group = [
            row for row in candidates if int(row["generation"]) == generation
        ]
        if group:
            axis.scatter(
                [row["initial_frequency"] for row in group],
                [100 * row["best_validation_accuracy"] for row in group],
                color=color,
                alpha=0.65,
                label=f"Generation {generation}",
            )
    selected = [row for row in candidates if row["selected"]]
    axis.scatter(
        [row["initial_frequency"] for row in selected],
        [100 * row["best_validation_accuracy"] for row in selected],
        color="#f0e442",
        edgecolor="#222222",
        marker="*",
        s=180,
        label="Selected starts",
        zorder=5,
    )
    axis.set_xlabel("Candidate starting frequency w")
    axis.set_ylabel("Warm-up best validation accuracy (%)")
    axis.set_title("Audio PEUAF Frequency Search")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_cost(
    aggregate: list[dict[str, Any]],
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    for row in aggregate:
        condition = row["condition"]
        axis.scatter(
            row["total_duration_seconds_mean"] / 60,
            100 * row["test_accuracy_mean"],
            color=COLORS[condition],
            edgecolor="#222222",
            s=115 if condition in {"periodic_gelu", "evolved_peuaf"} else 80,
            label=row["label"],
        )
    axis.set_xlabel("Mean end-to-end CPU time per seed (minutes)")
    axis.set_ylabel("Mean test accuracy (%)")
    axis.set_title("Audio Accuracy Versus Search and Training Cost")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_audio_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    base_config_path = str(config["base_config"])
    base = load_config(base_config_path)
    repeats = int(config.get("repeats", 3))
    seed_start = int(config.get("seed_start", 42))
    epochs = int(config.get("epochs", base["training"]["epochs"]))
    warmup_epochs = int(config.get("warmup_epochs", 4))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(config.get("output_dir", "runs/benchmarks/publication"))
        / f"{config.get('name', 'audio_activations')}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for repeat in range(repeats):
        seed = seed_start + repeat
        print(f"\nAudio benchmark seed {seed} ({repeat + 1}/{repeats})")
        for condition in ("gelu", "periodic_gelu", "direct_peuaf"):
            row = _run_direct(
                base,
                condition=condition,
                seed=seed,
                epochs=epochs,
                output_dir=runs_dir / condition,
            )
            rows.append(row)
            _write_csv(
                [_without_history(item) for item in rows],
                output_dir / "runs.csv",
            )
        evolved, seed_candidates = _run_evolved(
            base_config_path,
            config,
            seed=seed,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            output_dir=runs_dir / "evolved_peuaf",
        )
        rows.append(evolved)
        candidates.extend(seed_candidates)
        _write_csv(
            [_without_history(item) for item in rows],
            output_dir / "runs.csv",
        )
        _write_csv(candidates, output_dir / "candidates.csv")

    aggregate = _aggregate(rows)
    paired = _paired_rows(rows)
    _write_csv(aggregate, output_dir / "aggregate.csv")
    _write_csv(paired, output_dir / "paired_differences.csv")
    _plot_accuracy(rows, aggregate, output_dir / "accuracy.png")
    _plot_paired(rows, output_dir / "paired_accuracy.png")
    _plot_learning_curves(rows, output_dir / "learning_curves.png")
    _plot_frequency_parameters(rows, output_dir / "activation_parameters.png")
    _plot_candidates(candidates, output_dir / "search_candidates.png")
    _plot_cost(aggregate, output_dir / "cost_accuracy.png")

    summary = {
        "benchmark_dir": str(output_dir),
        "dataset": "mini_speech_commands",
        "base_config": base_config_path,
        "repeats": repeats,
        "seed_start": seed_start,
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "search": config["search"],
        "runs": rows,
        "candidates": candidates,
        "aggregate": aggregate,
        "paired_differences": paired,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nAudio benchmark reports written to {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare GELU, Periodic GELU, and PEUAF on raw speech waveforms"
        )
    )
    parser.add_argument(
        "--config",
        default="configs/benchmark_audio_activations.yaml",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    for override in args.overrides:
        key, value = parse_override(override)
        set_by_path(config, key, value)
    run_audio_benchmark(config)


if __name__ == "__main__":
    main()
