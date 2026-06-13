from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .activations import (
    EUAF,
    PEUAF,
    SineTriangle,
    activation_names,
    make_activation,
)


@dataclass(frozen=True)
class ActivationCurve:
    label: str
    values: torch.Tensor
    derivatives: torch.Tensor
    metadata: dict[str, Any]


def sample_activation(
    name: str,
    inputs: torch.Tensor,
    *,
    initial_w: float | None = None,
    initial_blend: float | None = None,
) -> ActivationCurve:
    normalized = name.lower().replace("-", "_")
    kwargs = {}
    if normalized == "peuaf" and initial_w is not None:
        kwargs["initial_w"] = initial_w
    if "sine_triangle" in normalized:
        if initial_w is not None:
            kwargs["initial_w"] = initial_w
        if initial_blend is not None:
            kwargs["initial_blend"] = initial_blend
    activation = make_activation(normalized, **kwargs).to(dtype=inputs.dtype)

    differentiable_inputs = inputs.detach().clone().requires_grad_(True)
    values = activation(differentiable_inputs)
    derivatives = torch.autograd.grad(
        values.sum(),
        differentiable_inputs,
        create_graph=False,
    )[0]

    metadata: dict[str, Any] = {"activation": normalized}
    label = normalized
    if isinstance(activation, PEUAF):
        w = float(activation.w.detach())
        metadata.update(
            {
                "w": w,
                "positive_period": 2.0 / w if w > 0 else None,
                "positive_peak_x": 1.0 / w if w > 0 else None,
                "negative_slope_at_zero": 1.0,
                "positive_slope_at_zero": w,
            }
        )
        label = f"PEUAF (w={w:g})"
    elif isinstance(activation, EUAF):
        metadata.update(
            {
                "w": 1.0,
                "positive_period": 2.0,
                "positive_peak_x": 1.0,
                "negative_slope_at_zero": 1.0,
                "positive_slope_at_zero": 1.0,
            }
        )
        label = "EUAF (w=1 fixed)"
    elif isinstance(activation, SineTriangle):
        w = float(activation.frequency.detach())
        blend = float(activation.blend.detach())
        period = 2.0 * torch.pi * activation.phase_divisor / w
        local_slope = (
            activation.residual_scale
            + activation.periodic_scale
            * w
            / activation.phase_divisor
            * (1.0 - blend + 2.0 * blend / torch.pi)
        )
        metadata.update(
            {
                "w": w,
                "blend": blend,
                "period": float(period),
                "slope_at_zero": float(local_slope),
                "phase_divisor": activation.phase_divisor,
                "residual_scale": activation.residual_scale,
                "base_activation": activation.base_activation,
                "periodic_scale": float(
                    torch.as_tensor(activation.periodic_scale).detach()
                ),
            }
        )
        label = (
            f"{normalized} (w={w:g}, triangle={blend:g})"
        )

    return ActivationCurve(
        label=label,
        values=values.detach(),
        derivatives=derivatives.detach(),
        metadata=metadata,
    )


def checkpoint_euaf_weights(path: str | Path) -> list[tuple[str, float]]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model_state = state.get("model", state)
    weights: list[tuple[str, float]] = []
    for name, value in model_state.items():
        if not name.endswith(".w"):
            continue
        flattened = value.detach().reshape(-1)
        if flattened.numel() == 1:
            weights.append((name, float(flattened[0])))
        else:
            weights.extend(
                (f"{name}[{index}]", float(weight))
                for index, weight in enumerate(flattened)
            )
    if not weights:
        raise ValueError(f"No PEUAF '.w' parameters found in {path}")
    return weights


def _write_csv(
    path: Path,
    inputs: torch.Tensor,
    curves: list[ActivationCurve],
) -> None:
    fieldnames = ["x"]
    for index, curve in enumerate(curves):
        fieldnames.extend([f"y_{index}", f"dy_dx_{index}"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_index, x_value in enumerate(inputs.tolist()):
            row: dict[str, float] = {"x": x_value}
            for curve_index, curve in enumerate(curves):
                row[f"y_{curve_index}"] = float(curve.values[sample_index])
                row[f"dy_dx_{curve_index}"] = float(
                    curve.derivatives[sample_index]
                )
            writer.writerow(row)


def plot_activation_curves(
    curves: list[ActivationCurve],
    inputs: torch.Tensor,
    output_dir: str | Path,
    *,
    title: str = "Activation Function Shapes",
) -> dict[str, str]:
    if not curves:
        raise ValueError("At least one activation curve is required")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, (shape_axis, derivative_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
    )
    x_values = inputs.numpy()
    for curve in curves:
        shape_axis.plot(x_values, curve.values.numpy(), label=curve.label)
        derivative_axis.plot(
            x_values,
            curve.derivatives.numpy(),
            label=curve.label,
        )
        if curve.metadata["activation"] in {"euaf", "peuaf"}:
            peak = curve.metadata["positive_peak_x"]
            period = curve.metadata["positive_period"]
            if peak is not None and period is not None:
                for axis in (shape_axis, derivative_axis):
                    for landmark in (0.0, peak, period):
                        if inputs[0] <= landmark <= inputs[-1]:
                            axis.axvline(
                                landmark,
                                color="black",
                                alpha=0.12,
                                linewidth=1,
                            )

    shape_axis.axhline(0, color="black", alpha=0.25, linewidth=1)
    shape_axis.set_title(title)
    shape_axis.set_ylabel("f(x)")
    shape_axis.grid(alpha=0.2)
    shape_axis.legend()

    derivative_axis.axhline(0, color="black", alpha=0.25, linewidth=1)
    derivative_axis.set_xlabel("x")
    derivative_axis.set_ylabel("df/dx")
    derivative_axis.grid(alpha=0.2)
    derivative_axis.legend()
    figure.tight_layout()

    plot_path = output_dir / "activation_shapes.png"
    csv_path = output_dir / "activation_samples.csv"
    metadata_path = output_dir / "activation_metadata.json"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    _write_csv(csv_path, inputs, curves)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "x_min": float(inputs[0]),
                "x_max": float(inputs[-1]),
                "samples": inputs.numel(),
                "curves": [
                    {"label": curve.label, **curve.metadata}
                    for curve in curves
                ],
            },
            handle,
            indent=2,
        )
    return {
        "plot": str(plot_path),
        "csv": str(csv_path),
        "metadata": str(metadata_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot sampled activation values and derivatives"
    )
    parser.add_argument(
        "--activations",
        nargs="+",
        default=["peuaf", "relu", "gelu", "silu", "tanh"],
        choices=activation_names(),
        help="Registered activations to plot",
    )
    parser.add_argument(
        "--w",
        nargs="+",
        type=float,
        default=[0.5],
        help="Initial frequency values for parameterized activations",
    )
    parser.add_argument(
        "--blend",
        nargs="+",
        type=float,
        default=[0.5],
        help="Initial triangle blend values for sine-triangle activations",
    )
    parser.add_argument(
        "--checkpoint",
        help="Optional checkpoint whose learned PEUAF weights are plotted",
    )
    parser.add_argument("--x-min", type=float, default=-4.0)
    parser.add_argument("--x-max", type=float, default=8.0)
    parser.add_argument("--samples", type=int, default=2401)
    parser.add_argument(
        "--output-dir",
        default="runs/activation_plots",
    )
    parser.add_argument(
        "--title",
        default="Periodic and Reference Activation Shapes",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.x_max <= args.x_min:
        raise ValueError("--x-max must be greater than --x-min")
    if args.samples < 2:
        raise ValueError("--samples must be at least two")

    inputs = torch.linspace(
        args.x_min,
        args.x_max,
        args.samples,
        dtype=torch.float64,
    )
    curves: list[ActivationCurve] = []
    for name in args.activations:
        if name == "peuaf":
            curves.extend(
                sample_activation(name, inputs, initial_w=w)
                for w in args.w
            )
        elif "sine_triangle" in name:
            curves.extend(
                sample_activation(
                    name,
                    inputs,
                    initial_w=w,
                    initial_blend=blend,
                )
                for w in args.w
                for blend in args.blend
            )
        else:
            curves.append(sample_activation(name, inputs))

    if args.checkpoint:
        for parameter_name, weight in checkpoint_euaf_weights(args.checkpoint):
            curve = sample_activation("peuaf", inputs, initial_w=weight)
            curves.append(
                ActivationCurve(
                    label=f"{parameter_name} (w={weight:.5g})",
                    values=curve.values,
                    derivatives=curve.derivatives,
                    metadata={
                        **curve.metadata,
                        "checkpoint_parameter": parameter_name,
                    },
                )
            )

    paths = plot_activation_curves(
        curves,
        inputs,
        args.output_dir,
        title=args.title,
    )
    print(f"Plot: {paths['plot']}")
    print(f"Samples: {paths['csv']}")
    print(f"Metadata: {paths['metadata']}")


if __name__ == "__main__":
    main()
