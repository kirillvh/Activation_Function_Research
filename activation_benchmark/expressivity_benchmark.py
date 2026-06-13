from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from .activations import make_activation
from .config import load_yaml
from .trainer import constrain_model_parameters, seed_everything


def triangle_wave(x: torch.Tensor, frequency: float) -> torch.Tensor:
    scaled = frequency * x
    return torch.abs(
        scaled - 2.0 * torch.floor((scaled + 1.0) / 2.0)
    )


class OneNeuronRegressor(nn.Module):
    """One affine neuron, one activation, and one affine output."""

    def __init__(self, activation: str, initial_w: float) -> None:
        super().__init__()
        self.input = nn.Linear(1, 1)
        kwargs = {"initial_w": initial_w} if activation == "peuaf" else {}
        self.activation = make_activation(activation, **kwargs)
        self.output = nn.Linear(1, 1)
        with torch.no_grad():
            self.input.weight.fill_(1.0)
            self.input.bias.zero_()
            self.output.weight.fill_(1.0)
            self.output.bias.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.input(inputs)))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_expressivity_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    benchmark = config["benchmark"]
    data_config = config["data"]
    training = config["training"]
    model_config = config["model"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(benchmark.get("output_dir", "runs/benchmarks"))
        / f"{benchmark.get('name', 'expressivity')}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train = torch.linspace(
        data_config["x_min"],
        data_config["x_max"],
        data_config["train_samples"],
    ).unsqueeze(1)
    x_test = torch.linspace(
        data_config["x_min"],
        data_config["x_max"],
        data_config["test_samples"],
    ).unsqueeze(1)
    target_frequency = data_config["target_frequency"]
    y_train = triangle_wave(x_train, target_frequency)
    y_test = triangle_wave(x_test, target_frequency)
    criterion = nn.MSELoss()
    rows: list[dict[str, Any]] = []
    predictions: dict[str, torch.Tensor] = {}
    losses: dict[str, list[float]] = {}

    for activation in benchmark["activations"]:
        for repeat in range(benchmark.get("repeats", 1)):
            seed = benchmark.get("seed", 42) + repeat
            seed_everything(seed, deterministic=True)
            model = OneNeuronRegressor(
                activation,
                initial_w=model_config.get("initial_w", 0.5),
            )
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=training["learning_rate"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=training["steps"],
                eta_min=training.get("minimum_learning_rate", 0.0),
            )
            history: list[float] = []
            for step in range(training["steps"]):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x_train), y_train)
                loss.backward()
                optimizer.step()
                constrain_model_parameters(model)
                scheduler.step()
                if step % training.get("record_every_steps", 10) == 0:
                    history.append(float(loss.detach()))

            model.eval()
            with torch.inference_mode():
                prediction = model(x_test)
                test_mse = float(criterion(prediction, y_test))
                max_error = float((prediction - y_test).abs().max())
            label = f"{activation} repeat {repeat + 1}"
            predictions[label] = prediction.squeeze(1)
            losses[label] = history
            rows.append(
                {
                    "activation": activation,
                    "repeat": repeat + 1,
                    "seed": seed,
                    "parameter_count": sum(
                        parameter.numel()
                        for parameter in model.parameters()
                    ),
                    "test_mse": test_mse,
                    "max_absolute_error": max_error,
                }
            )

    aggregate: list[dict[str, Any]] = []
    for activation in benchmark["activations"]:
        group = [row for row in rows if row["activation"] == activation]
        aggregate.append(
            {
                "activation": activation,
                "runs": len(group),
                "test_mse_mean": mean(row["test_mse"] for row in group),
                "test_mse_std": pstdev(row["test_mse"] for row in group),
                "max_absolute_error_mean": mean(
                    row["max_absolute_error"] for row in group
                ),
                "max_absolute_error_std": pstdev(
                    row["max_absolute_error"] for row in group
                ),
            }
        )
    _write_csv(rows, output_dir / "runs.csv")
    _write_csv(aggregate, output_dir / "aggregate.csv")

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        x_test.squeeze(1),
        y_test.squeeze(1),
        color="black",
        linewidth=2,
        label=f"target triangle (frequency={target_frequency:g})",
    )
    for label, prediction in predictions.items():
        axis.plot(x_test.squeeze(1), prediction, alpha=0.65, label=label)
    axis.set_title("Fixed-Size Activation Expressivity")
    axis.set_xlabel("x")
    axis.set_ylabel("output")
    axis.grid(alpha=0.2)
    axis.legend(fontsize="small", ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "fits.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    record_every = training.get("record_every_steps", 10)
    for label, history in losses.items():
        axis.plot(
            [index * record_every for index in range(len(history))],
            history,
            label=label,
        )
    axis.set_yscale("log")
    axis.set_title("Triangle-Wave Fitting Loss")
    axis.set_xlabel("Optimization step")
    axis.set_ylabel("MSE")
    axis.grid(alpha=0.2)
    axis.legend(fontsize="small", ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curves.png", dpi=180)
    plt.close(figure)

    summary = {
        "benchmark_dir": str(output_dir),
        "target_frequency": target_frequency,
        "runs": rows,
        "aggregate": aggregate,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fixed-size activation expressivity"
    )
    parser.add_argument(
        "--config",
        default="configs/benchmark_triangle_expressivity.yaml",
    )
    args = parser.parse_args()
    run_expressivity_benchmark(load_yaml(args.config))


if __name__ == "__main__":
    main()
