import pytest

from activation_benchmark.benchmark import parameter_combinations


def test_parameter_combinations_are_cartesian():
    combinations = parameter_combinations(
        {
            "model.activation": ["relu", "gelu"],
            "training.learning_rate": [0.1, 0.01],
        }
    )
    assert len(combinations) == 4
    assert combinations[0] == {
        "model.activation": "relu",
        "training.learning_rate": 0.1,
    }


def test_parameter_combinations_reject_empty_values():
    with pytest.raises(ValueError, match="non-empty list"):
        parameter_combinations({"model.activation": []})
