from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .activations import PEUAF
from .config import load_yaml
from .expressivity_benchmark import triangle_wave
from .trainer import seed_everything


def frequency_candidate_losses(
    frequencies: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Evaluate many scalar PEUAF frequencies without building many models."""
    frequencies = frequencies.reshape(-1, 1, 1)
    scaled = frequencies * inputs.unsqueeze(0)
    predictions = torch.abs(
        scaled - 2.0 * torch.floor((scaled + 1.0) / 2.0)
    )
    return (predictions - targets.unsqueeze(0)).square().mean(dim=(1, 2))


def refine_frequency(
    initial_frequency: float,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    minimum_learning_rate: float = 0.0,
) -> tuple[float, float]:
    seed_everything(seed, deterministic=True)
    model = PEUAF(initial_w=initial_frequency)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(steps, 1),
        eta_min=minimum_learning_rate,
    )
    sample_count = inputs.shape[0]

    for _ in range(steps):
        if batch_size >= sample_count:
            batch_inputs = inputs
            batch_targets = targets
        else:
            indices = torch.randint(0, sample_count, (batch_size,))
            batch_inputs = inputs[indices]
            batch_targets = targets[indices]
        loss = nn.functional.mse_loss(model(batch_inputs), batch_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.constrain_parameters()
        scheduler.step()

    frequency = float(model.w.detach())
    with torch.inference_mode():
        final_loss = float(nn.functional.mse_loss(model(inputs), targets))
    return frequency, final_loss


def population_local_search(
    initial_frequencies: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run independent Adam searches for a population in one vectorized graph."""
    seed_everything(seed, deterministic=True)
    frequencies = nn.Parameter(initial_frequencies.detach().clone())
    optimizer = torch.optim.Adam([frequencies], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(steps, 1),
    )
    sample_count = inputs.shape[0]

    for _ in range(steps):
        if batch_size >= sample_count:
            batch_inputs = inputs
            batch_targets = targets
        else:
            indices = torch.randint(0, sample_count, (batch_size,))
            batch_inputs = inputs[indices]
            batch_targets = targets[indices]
        losses = frequency_candidate_losses(
            frequencies,
            batch_inputs,
            batch_targets,
        )
        optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        optimizer.step()
        with torch.no_grad():
            frequencies.clamp_(0.0, 1.0)
        scheduler.step()

    with torch.inference_mode():
        final_losses = frequency_candidate_losses(
            frequencies,
            inputs,
            targets,
        )
    return frequencies.detach(), final_losses


def _pygad_search(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    initial_frequency: float,
    population_size: int,
    generations: int,
    seed: int,
    use_torchga: bool,
) -> tuple[float, int]:
    try:
        import pygad
        from pygad import torchga
    except ImportError as error:
        raise RuntimeError(
            "PyGAD is required for the pygad and torchga search methods. "
            "Install it with `python -m pip install pygad`."
        ) from error

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    if use_torchga:
        model = PEUAF(initial_w=initial_frequency)
        torch_ga = torchga.TorchGA(
            model=model,
            num_solutions=population_size,
        )
        initial_population = np.clip(
            np.asarray(torch_ga.population_weights),
            0.0,
            1.0,
        )

        def fitness_function(
            ga_instance: pygad.GA,
            solution: np.ndarray,
            solution_index: int,
        ) -> float:
            del ga_instance, solution_index
            predictions = torchga.predict(model, solution, inputs)
            loss = nn.functional.mse_loss(predictions, targets).item()
            return 1.0 / (loss + 1e-12)

    else:
        initial_population = rng.uniform(
            0.0,
            1.0,
            size=(population_size, 1),
        )
        initial_population[0, 0] = initial_frequency

        def fitness_function(
            ga_instance: pygad.GA,
            solution: np.ndarray,
            solution_index: int,
        ) -> float:
            del ga_instance, solution_index
            frequency = torch.tensor(solution, dtype=inputs.dtype)
            loss = float(
                frequency_candidate_losses(
                    frequency,
                    inputs,
                    targets,
                )[0]
            )
            return 1.0 / (loss + 1e-12)

    ga = pygad.GA(
        num_generations=generations,
        num_parents_mating=max(2, population_size // 2),
        fitness_func=fitness_function,
        initial_population=initial_population,
        gene_space={"low": 0.0, "high": 1.0},
        parent_selection_type="tournament",
        crossover_type="uniform",
        mutation_type="random",
        mutation_probability=0.25,
        random_mutation_min_val=-0.1,
        random_mutation_max_val=0.1,
        keep_elitism=2,
        random_seed=seed,
        suppress_warnings=True,
    )
    ga.run()
    solution, _, _ = ga.best_solution()
    evaluations = population_size * (generations + 1)
    return float(np.clip(solution[0], 0.0, 1.0)), evaluations


def search_frequency(
    method: str,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    config: dict[str, Any],
    *,
    seed: int,
) -> tuple[float, int]:
    search = config["search"]
    initial_frequency = float(search.get("initial_w", 0.5))
    population_size = int(search.get("population_size", 24))
    generator = torch.Generator().manual_seed(seed)

    if method == "gradient":
        return initial_frequency, 0
    if method == "random":
        candidates = torch.rand(population_size, generator=generator)
        candidates[0] = initial_frequency
        losses = frequency_candidate_losses(candidates, inputs, targets)
        return float(candidates[losses.argmin()]), population_size
    if method == "grid":
        grid_points = int(search.get("grid_points", 201))
        candidates = torch.linspace(0.0, 1.0, grid_points)
        losses = frequency_candidate_losses(candidates, inputs, targets)
        return float(candidates[losses.argmin()]), grid_points
    if method == "multistart":
        candidates = torch.rand(population_size, generator=generator)
        candidates[0] = initial_frequency
        frequencies, losses = population_local_search(
            candidates,
            inputs,
            targets,
            steps=int(search.get("warmup_steps", 300)),
            learning_rate=float(search.get("warmup_learning_rate", 0.01)),
            batch_size=int(search.get("warmup_batch_size", len(inputs))),
            seed=seed,
        )
        evaluations = population_size * int(search.get("warmup_steps", 300))
        return float(frequencies[losses.argmin()]), evaluations
    if method in {"pygad", "torchga"}:
        return _pygad_search(
            inputs,
            targets,
            initial_frequency=initial_frequency,
            population_size=population_size,
            generations=int(search.get("generations", 30)),
            seed=seed,
            use_torchga=method == "torchga",
        )
    raise ValueError(f"Unknown frequency search method: {method!r}")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(
    rows: list[dict[str, Any]],
    group_key: str,
) -> list[dict[str, Any]]:
    aggregate: list[dict[str, Any]] = []
    for value in dict.fromkeys(row[group_key] for row in rows):
        group = [row for row in rows if row[group_key] == value]
        losses = [float(row["test_mse"]) for row in group]
        frequencies = [float(row["final_frequency"]) for row in group]
        aggregate.append(
            {
                group_key: value,
                "runs": len(group),
                "success_rate": mean(float(row["success"]) for row in group),
                "test_mse_mean": mean(losses),
                "test_mse_median": median(losses),
                "test_mse_std": pstdev(losses),
                "final_frequency_mean": mean(frequencies),
                "final_frequency_std": pstdev(frequencies),
                "duration_seconds_mean": mean(
                    float(row["duration_seconds"]) for row in group
                ),
            }
        )
    return aggregate


def _plot_summary(
    aggregate: list[dict[str, Any]],
    key: str,
    title: str,
    output_path: Path,
) -> None:
    labels = [str(row[key]) for row in aggregate]
    success = [100.0 * float(row["success_rate"]) for row in aggregate]
    losses = [max(float(row["test_mse_median"]), 1e-16) for row in aggregate]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(labels, success, color="#4472c4")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_title("Reached target basin")
    axes[1].bar(labels, losses, color="#ed7d31")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Median test MSE")
    axes[1].set_title("Final error")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_frequency_optimization_benchmark(
    config: dict[str, Any],
) -> dict[str, Any]:
    benchmark = config["benchmark"]
    data_config = config["data"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(benchmark.get("output_dir", "runs/benchmarks/research"))
        / f"{benchmark.get('name', 'frequency_optimization')}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = torch.linspace(
        float(data_config.get("x_min", 0.0)),
        float(data_config.get("x_max", 20.0)),
        int(data_config.get("train_samples", 512)),
    ).unsqueeze(1)
    test_inputs = torch.linspace(
        float(data_config.get("x_min", 0.0)),
        float(data_config.get("x_max", 20.0)),
        int(data_config.get("test_samples", 4096)),
    ).unsqueeze(1)
    target_frequency = float(data_config.get("target_frequency", 0.37))
    targets = triangle_wave(inputs, target_frequency)
    test_targets = triangle_wave(test_inputs, target_frequency)
    success_mse = float(benchmark.get("success_mse", 1e-6))
    base_seed = int(benchmark.get("seed", 42))

    batch_rows: list[dict[str, Any]] = []
    batch_study = config.get("batch_study", {})
    if batch_study.get("enabled", True):
        for batch_size in batch_study.get(
            "batch_sizes",
            [1, 8, 32, 128, len(inputs)],
        ):
            for repeat in range(int(batch_study.get("repeats", 10))):
                seed = base_seed + repeat
                started = time.perf_counter()
                frequency, _ = refine_frequency(
                    float(batch_study.get("initial_w", 0.5)),
                    inputs,
                    targets,
                    steps=int(batch_study.get("steps", 3000)),
                    learning_rate=float(
                        batch_study.get("learning_rate", 0.01)
                    ),
                    batch_size=int(batch_size),
                    seed=seed,
                    minimum_learning_rate=float(
                        batch_study.get("minimum_learning_rate", 1e-6)
                    ),
                )
                test_loss = float(
                    frequency_candidate_losses(
                        torch.tensor([frequency]),
                        test_inputs,
                        test_targets,
                    )[0]
                )
                batch_rows.append(
                    {
                        "batch_size": int(batch_size),
                        "repeat": repeat + 1,
                        "seed": seed,
                        "final_frequency": frequency,
                        "test_mse": test_loss,
                        "success": int(test_loss <= success_mse),
                        "duration_seconds": time.perf_counter() - started,
                    }
                )

    search_rows: list[dict[str, Any]] = []
    search_study = config.get("search_study", {})
    refinement = config["refinement"]
    for method in search_study.get(
        "methods",
        ["gradient", "random", "multistart", "grid", "pygad", "torchga"],
    ):
        for repeat in range(int(search_study.get("repeats", 5))):
            seed = base_seed + repeat
            started = time.perf_counter()
            selected, evaluations = search_frequency(
                str(method),
                inputs,
                targets,
                config,
                seed=seed,
            )
            final_frequency, _ = refine_frequency(
                selected,
                inputs,
                targets,
                steps=int(refinement.get("steps", 3000)),
                learning_rate=float(refinement.get("learning_rate", 0.01)),
                batch_size=int(refinement.get("batch_size", len(inputs))),
                seed=seed,
                minimum_learning_rate=float(
                    refinement.get("minimum_learning_rate", 1e-6)
                ),
            )
            test_loss = float(
                frequency_candidate_losses(
                    torch.tensor([final_frequency]),
                    test_inputs,
                    test_targets,
                )[0]
            )
            search_rows.append(
                {
                    "method": method,
                    "repeat": repeat + 1,
                    "seed": seed,
                    "selected_frequency": selected,
                    "final_frequency": final_frequency,
                    "search_evaluations": evaluations,
                    "test_mse": test_loss,
                    "success": int(test_loss <= success_mse),
                    "duration_seconds": time.perf_counter() - started,
                }
            )

    batch_aggregate = _aggregate(batch_rows, "batch_size")
    search_aggregate = _aggregate(search_rows, "method")
    _write_csv(batch_rows, output_dir / "batch_runs.csv")
    _write_csv(batch_aggregate, output_dir / "batch_aggregate.csv")
    _write_csv(search_rows, output_dir / "search_runs.csv")
    _write_csv(search_aggregate, output_dir / "search_aggregate.csv")
    if batch_aggregate:
        _plot_summary(
            batch_aggregate,
            "batch_size",
            "Batch Size and Frequency-Basin Escape",
            output_dir / "batch_size.png",
        )
    if search_aggregate:
        _plot_summary(
            search_aggregate,
            "method",
            "Global Initialization Followed by Backpropagation",
            output_dir / "search_methods.png",
        )

    summary = {
        "benchmark_dir": str(output_dir),
        "target_frequency": target_frequency,
        "success_mse": success_mse,
        "batch_aggregate": batch_aggregate,
        "search_aggregate": search_aggregate,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Study PEUAF frequency initialization and basin escape"
    )
    parser.add_argument(
        "--config",
        default="configs/benchmark_peuaf_frequency_optimization.yaml",
    )
    args = parser.parse_args()
    run_frequency_optimization_benchmark(load_yaml(args.config))


if __name__ == "__main__":
    main()
