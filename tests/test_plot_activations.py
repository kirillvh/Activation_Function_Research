import csv
import json

import pytest
import torch

from activation_benchmark.plot_activations import (
    checkpoint_euaf_weights,
    plot_activation_curves,
    sample_activation,
)


def test_sample_peuaf_reports_shape_and_slopes():
    inputs = torch.tensor(
        [-1e-4, 0.0, 1e-4, 2.0, 4.0],
        dtype=torch.float64,
    )
    curve = sample_activation("peuaf", inputs, initial_w=0.5)

    assert curve.values[1].item() == pytest.approx(0.0)
    assert curve.metadata["positive_period"] == pytest.approx(4.0)
    assert curve.metadata["negative_slope_at_zero"] == pytest.approx(1.0)
    assert curve.metadata["positive_slope_at_zero"] == pytest.approx(0.5)

    period_inputs = torch.tensor(
        [curve.metadata["positive_period"]],
        dtype=torch.float64,
    )
    period_curve = sample_activation(
        "peuaf",
        period_inputs,
        initial_w=0.5,
    )
    assert period_curve.values[0].item() == pytest.approx(0.0, abs=1e-12)


def test_plot_exports_png_csv_and_metadata(tmp_path):
    inputs = torch.linspace(-2, 4, 101, dtype=torch.float64)
    curves = [
        sample_activation("peuaf", inputs, initial_w=0.25),
        sample_activation("relu", inputs),
    ]

    paths = plot_activation_curves(curves, inputs, tmp_path)

    assert (tmp_path / "activation_shapes.png").stat().st_size > 0
    with (tmp_path / "activation_samples.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == inputs.numel()
    metadata = json.loads(
        (tmp_path / "activation_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["curves"][0]["activation"] == "peuaf"
    assert paths["plot"].endswith("activation_shapes.png")


def test_checkpoint_euaf_weights_reads_scalar_parameters(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": {
                "features.1.w": torch.tensor(0.2),
                "classifier.2.w": torch.tensor(0.3),
                "features.3.w": torch.tensor([0.4, 0.5]),
                "features.0.weight": torch.ones(2, 2),
            }
        },
        checkpoint,
    )

    weights = checkpoint_euaf_weights(checkpoint)

    assert [name for name, _ in weights] == [
        "features.1.w",
        "classifier.2.w",
        "features.3.w[0]",
        "features.3.w[1]",
    ]
    assert [value for _, value in weights] == pytest.approx(
        [0.2, 0.3, 0.4, 0.5]
    )
