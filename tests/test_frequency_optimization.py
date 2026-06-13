import pytest
import torch

from activation_benchmark.expressivity_benchmark import triangle_wave
from activation_benchmark.frequency_optimization import (
    frequency_candidate_losses,
    population_local_search,
    search_frequency,
)


def test_frequency_grid_finds_triangle_target():
    inputs = torch.linspace(0.0, 20.0, 256).unsqueeze(1)
    targets = triangle_wave(inputs, 0.37)
    config = {
        "search": {
            "initial_w": 0.5,
            "population_size": 8,
            "grid_points": 201,
        }
    }

    selected, evaluations = search_frequency(
        "grid",
        inputs,
        targets,
        config,
        seed=42,
    )

    assert selected == pytest.approx(0.37)
    assert evaluations == 201


def test_vectorized_multistart_refines_independent_frequencies():
    inputs = torch.linspace(0.0, 20.0, 256).unsqueeze(1)
    targets = triangle_wave(inputs, 0.37)
    frequencies, losses = population_local_search(
        torch.tensor([0.2, 0.4, 0.8]),
        inputs,
        targets,
        steps=500,
        learning_rate=0.01,
        batch_size=len(inputs),
        seed=42,
    )

    best = int(losses.argmin())
    assert frequencies[best].item() == pytest.approx(0.37, abs=1e-5)
    assert losses[best].item() < 1e-8


def test_candidate_loss_is_zero_at_target_frequency():
    inputs = torch.linspace(0.0, 20.0, 64).unsqueeze(1)
    targets = triangle_wave(inputs, 0.37)
    losses = frequency_candidate_losses(
        torch.tensor([0.2, 0.37, 0.8]),
        inputs,
        targets,
    )
    assert losses[1].item() == pytest.approx(0.0)
