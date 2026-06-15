import csv

import pytest

from activation_benchmark.cifar_peuaf_benchmark import (
    _paired_rows,
    _reference_rows,
)


def test_reference_rows_load_gelu_and_periodic_gelu(tmp_path):
    path = tmp_path / "runs.csv"
    fieldnames = [
        "seed",
        "activation",
        "epochs_completed",
        "duration_seconds",
        "run_dir",
        "dataset",
        "architecture",
        "parameter_count",
        "best_validation_accuracy",
        "test_accuracy",
        "final_test_accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for activation in ("gelu", "gelu_sine_triangle"):
            writer.writerow(
                {
                    "seed": 42,
                    "activation": activation,
                    "epochs_completed": 120,
                    "duration_seconds": 10,
                    "run_dir": "run",
                    "dataset": "cifar10",
                    "architecture": "standard",
                    "parameter_count": 100,
                    "best_validation_accuracy": 0.9,
                    "test_accuracy": 0.89,
                    "final_test_accuracy": 0.88,
                }
            )

    rows = _reference_rows(path, seeds={42})

    assert [row["condition"] for row in rows] == [
        "gelu",
        "periodic_gelu",
    ]


def test_cifar_paired_rows_report_percentage_points():
    rows = [
        {"seed": 42, "condition": "gelu", "test_accuracy": 0.90},
        {
            "seed": 42,
            "condition": "periodic_gelu",
            "test_accuracy": 0.91,
        },
        {
            "seed": 42,
            "condition": "direct_peuaf",
            "test_accuracy": 0.80,
        },
        {
            "seed": 42,
            "condition": "evolved_peuaf",
            "test_accuracy": 0.84,
        },
    ]

    paired = _paired_rows(rows)

    assert paired[0]["evolved_minus_direct"] == pytest.approx(4.0)
    assert paired[0]["periodic_gelu_minus_gelu"] == pytest.approx(1.0)
