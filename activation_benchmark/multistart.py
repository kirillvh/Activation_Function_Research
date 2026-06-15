from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoints import CheckpointManager
from .config import (
    load_config,
    load_yaml,
    parse_override,
    set_by_path,
)
from .trainer import train_experiment


def set_initial_peuaf_frequency(
    config: dict[str, Any],
    frequency: float,
) -> None:
    model = config["model"]
    configured = False
    if str(model.get("activation", "")).lower() == "peuaf":
        model.setdefault("activation_kwargs", {})["initial_w"] = frequency
        configured = True
    policy = str(model.get("activation_policy", "baseline")).lower()
    if policy != "baseline" or "peuaf_kwargs" in model:
        model.setdefault("peuaf_kwargs", {})["initial_w"] = frequency
        configured = True
    if not configured:
        raise ValueError(
            "The base config does not appear to contain a PEUAF activation"
        )


def initial_population(
    search: dict[str, Any],
    *,
    seed: int,
) -> list[float]:
    explicit = search.get("initial_frequencies")
    if explicit is not None:
        values = [float(value) for value in explicit]
    else:
        size = int(search.get("population_size", 6))
        lower = float(search.get("minimum_frequency", 0.0))
        upper = float(search.get("maximum_frequency", 1.0))
        strategy = str(search.get("initial_strategy", "grid")).lower()
        if strategy == "grid":
            values = np.linspace(lower, upper, size + 2)[1:-1].tolist()
        elif strategy == "random":
            values = np.random.default_rng(seed).uniform(
                lower,
                upper,
                size,
            ).tolist()
        else:
            raise ValueError(
                "search.initial_strategy must be 'grid' or 'random'"
            )
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("All PEUAF candidate frequencies must be in [0, 1]")
    return values


def evolve_frequencies(
    elites: list[float],
    *,
    population_size: int,
    mutation_std: float,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    children = list(elites)
    while len(children) < population_size:
        parents = rng.choice(elites, size=2, replace=True)
        child = float(parents.mean() + rng.normal(0.0, mutation_std))
        children.append(float(np.clip(child, 0.0, 1.0)))
    return children


def _latest_checkpoint(run_dir: str | Path) -> Path:
    return CheckpointManager(
        Path(run_dir) / "checkpoints",
        keep_latest=1,
    ).latest_path()


def checkpoint_peuaf_frequencies(
    checkpoint_path: str | Path,
    *,
    weights: str = "best",
) -> list[float]:
    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    key = "best_model" if weights == "best" else "model"
    model_state = state.get(key)
    if model_state is None:
        raise ValueError(f"Checkpoint does not contain {key!r} weights")
    frequencies = [
        float(value.reshape(-1).mean())
        for name, value in model_state.items()
        if name.endswith(".w")
    ]
    if not frequencies:
        raise ValueError("Checkpoint does not contain PEUAF frequency weights")
    return frequencies


def run_evolutionary_multistart(
    config: dict[str, Any],
) -> dict[str, Any]:
    search = config["search"]
    base_config = load_config(config["base_config"])
    for key, value in config.get("base_overrides", {}).items():
        set_by_path(base_config, key, value)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(config.get("output_dir", "runs/multistart"))
        / f"{config.get('name', 'peuaf_multistart')}_{timestamp}"
    )
    stage1_dir = output_dir / "candidates"
    stage1_dir.mkdir(parents=True, exist_ok=True)

    seed = int(search.get("seed", base_config["experiment"]["seed"]))
    population = initial_population(search, seed=seed)
    population_size = len(population)
    generations = int(search.get("generations", 1))
    elite_count = int(search.get("elite_count", 2))
    if not 1 <= elite_count <= population_size:
        raise ValueError("search.elite_count must fit within the population")
    metric = str(search.get("selection_metric", "best_validation_accuracy"))
    rows: list[dict[str, Any]] = []
    latest_generation_rows: list[dict[str, Any]] = []

    for generation in range(generations):
        generation_rows: list[dict[str, Any]] = []
        for candidate_index, frequency in enumerate(population):
            candidate = copy.deepcopy(base_config)
            for key, value in search.get("candidate_overrides", {}).items():
                set_by_path(candidate, key, value)
            set_initial_peuaf_frequency(candidate, frequency)
            candidate["experiment"]["seed"] = seed
            candidate["experiment"]["name"] = (
                f"generation-{generation + 1:02d}_"
                f"candidate-{candidate_index + 1:02d}_w-{frequency:.6f}"
            )
            candidate["experiment"]["output_dir"] = str(stage1_dir)
            candidate["training"]["epochs"] = int(
                search.get("warmup_epochs", 3)
            )
            candidate["training"]["evaluate_test"] = False
            candidate["checkpoint"]["enabled"] = True
            candidate["checkpoint"]["keep_latest"] = 1
            candidate["checkpoint"]["save_every_steps"] = 10**12
            candidate["checkpoint"]["resume"] = None
            candidate["tensorboard"]["enabled"] = bool(
                search.get("candidate_tensorboard", False)
            )
            print(
                f"\nGeneration {generation + 1}/{generations}, "
                f"candidate {candidate_index + 1}/{population_size}: "
                f"w={frequency:.6f}"
            )
            result = train_experiment(candidate)
            checkpoint = _latest_checkpoint(result["run_dir"])
            learned_frequencies = checkpoint_peuaf_frequencies(checkpoint)
            row = {
                "generation": generation + 1,
                "candidate": candidate_index + 1,
                "initial_frequency": frequency,
                "learned_frequency_min": min(learned_frequencies),
                "learned_frequency_mean": float(
                    np.mean(learned_frequencies)
                ),
                "learned_frequency_max": max(learned_frequencies),
                "checkpoint": str(checkpoint),
                "history": result["history"],
                **{
                    key: value
                    for key, value in result.items()
                    if key != "history"
                },
            }
            rows.append(row)
            generation_rows.append(row)

        generation_rows.sort(key=lambda row: float(row[metric]), reverse=True)
        latest_generation_rows = generation_rows
        elites = [
            float(row["learned_frequency_mean"])
            for row in generation_rows[:elite_count]
        ]
        if generation + 1 < generations:
            population = evolve_frequencies(
                elites,
                population_size=population_size,
                mutation_std=float(search.get("mutation_std", 0.08)),
                seed=seed + generation + 1,
            )

    halving = search.get("successive_halving")
    survivors = latest_generation_rows if halving else rows
    if halving:
        previous_epochs = int(search.get("warmup_epochs", 3))
        reduction_factor = float(halving.get("reduction_factor", 2.0))
        if reduction_factor <= 1.0:
            raise ValueError(
                "search.successive_halving.reduction_factor must exceed one"
            )
        for rung_index, cumulative_epochs in enumerate(
            halving.get("rung_epochs", []),
            start=1,
        ):
            cumulative_epochs = int(cumulative_epochs)
            if cumulative_epochs <= previous_epochs:
                raise ValueError(
                    "Successive-halving rung epochs must increase"
                )
            survivors.sort(
                key=lambda row: float(row[metric]),
                reverse=True,
            )
            survivor_count = max(
                1,
                math.ceil(len(survivors) / reduction_factor),
            )
            promoted = survivors[:survivor_count]
            rung_rows: list[dict[str, Any]] = []
            additional_epochs = cumulative_epochs - previous_epochs
            for candidate_index, parent in enumerate(promoted):
                candidate = copy.deepcopy(base_config)
                for key, value in search.get(
                    "candidate_overrides",
                    {},
                ).items():
                    set_by_path(candidate, key, value)
                frequency = float(parent["learned_frequency_mean"])
                set_initial_peuaf_frequency(candidate, frequency)
                candidate["experiment"]["seed"] = seed
                candidate["experiment"]["name"] = (
                    f"rung-{rung_index:02d}_"
                    f"candidate-{candidate_index + 1:02d}_w-{frequency:.6f}"
                )
                candidate["experiment"]["output_dir"] = str(stage1_dir)
                candidate["training"]["epochs"] = additional_epochs
                candidate["training"]["evaluate_test"] = False
                candidate["training"]["initial_checkpoint"] = parent[
                    "checkpoint"
                ]
                candidate["training"]["initial_checkpoint_weights"] = "best"
                candidate["checkpoint"]["enabled"] = True
                candidate["checkpoint"]["keep_latest"] = 1
                candidate["checkpoint"]["save_every_steps"] = 10**12
                candidate["checkpoint"]["resume"] = None
                candidate["tensorboard"]["enabled"] = bool(
                    search.get("candidate_tensorboard", False)
                )
                print(
                    f"\nSuccessive-halving rung {rung_index}, "
                    f"candidate {candidate_index + 1}/{survivor_count}: "
                    f"{previous_epochs} -> {cumulative_epochs} epochs, "
                    f"w={frequency:.6f}"
                )
                result = train_experiment(candidate)
                checkpoint = _latest_checkpoint(result["run_dir"])
                learned_frequencies = checkpoint_peuaf_frequencies(checkpoint)
                row = {
                    "generation": f"rung-{rung_index}",
                    "candidate": candidate_index + 1,
                    "initial_frequency": frequency,
                    "learned_frequency_min": min(learned_frequencies),
                    "learned_frequency_mean": float(
                        np.mean(learned_frequencies)
                    ),
                    "learned_frequency_max": max(learned_frequencies),
                    "cumulative_epochs": cumulative_epochs,
                    "parent_checkpoint": parent["checkpoint"],
                    "checkpoint": str(checkpoint),
                    "history": result["history"],
                    **{
                        key: value
                        for key, value in result.items()
                        if key != "history"
                    },
                }
                rows.append(row)
                rung_rows.append(row)
            survivors = rung_rows
            previous_epochs = cumulative_epochs

    best = max(survivors, key=lambda row: float(row[metric]))
    final_config = copy.deepcopy(base_config)
    set_initial_peuaf_frequency(
        final_config,
        float(best["initial_frequency"]),
    )
    final_config["experiment"]["name"] = "selected_full_training"
    final_config["experiment"]["output_dir"] = str(output_dir)
    final_config["experiment"]["seed"] = seed
    final_config["training"]["epochs"] = int(
        config.get("final_training", {}).get(
            "epochs",
            base_config["training"]["epochs"],
        )
    )
    final_config["training"]["evaluate_test"] = True
    warm_start_selected = bool(
        config.get("final_training", {}).get(
            "warm_start_selected",
            True,
        )
    )
    if warm_start_selected:
        final_config["training"]["initial_checkpoint"] = best["checkpoint"]
        final_config["training"]["initial_checkpoint_weights"] = "best"
    else:
        final_config["training"].pop("initial_checkpoint", None)
        final_config["training"].pop("initial_checkpoint_weights", None)
    final_config["checkpoint"]["resume"] = None
    print(
        "\nSelected candidate "
        f"w={best['initial_frequency']:.6f} with "
        f"{metric}={float(best[metric]):.4f}. Starting full training."
    )
    final_result = train_experiment(final_config)

    summary = {
        "output_dir": str(output_dir),
        "base_config": config["base_config"],
        "base_overrides": config.get("base_overrides", {}),
        "selection_metric": metric,
        "warm_start_selected": warm_start_selected,
        "selected_candidate": best,
        "candidate_runs": rows,
        "final_result": final_result,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nMulti-start reports written to {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evolve PEUAF starts, select the best, then train fully"
    )
    parser.add_argument(
        "--config",
        default="configs/peuaf_evolutionary_multistart.yaml",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values with dotted key paths",
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    for override in args.overrides:
        key, value = parse_override(override)
        set_by_path(config, key, value)
    run_evolutionary_multistart(config)


if __name__ == "__main__":
    main()
