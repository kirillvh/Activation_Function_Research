import pytest

from activation_benchmark.peuaf_search_benchmark import (
    _combined_history,
    _paired_differences,
)


def test_combined_history_offsets_epochs_and_steps():
    first = [{"epoch": 1, "global_step": 8}]
    second = [
        {"epoch": 1, "global_step": 8},
        {"epoch": 2, "global_step": 16},
    ]

    combined = _combined_history(first, second)

    assert [row["epoch"] for row in combined] == [1, 2, 3]
    assert [row["global_step"] for row in combined] == [8, 16, 24]


def test_paired_differences_are_percentage_points():
    rows = [
        {"seed": 42, "condition": "gelu", "test_accuracy": 0.90},
        {"seed": 42, "condition": "direct_peuaf", "test_accuracy": 0.80},
        {"seed": 42, "condition": "staged_peuaf", "test_accuracy": 0.81},
        {"seed": 42, "condition": "evolved_peuaf", "test_accuracy": 0.84},
    ]

    paired = _paired_differences(rows)

    assert paired[0]["evolved_minus_direct"] == pytest.approx(4.0)
    assert paired[0]["evolved_minus_staged"] == pytest.approx(3.0)
    assert paired[0]["evolved_minus_gelu"] == pytest.approx(-6.0)
