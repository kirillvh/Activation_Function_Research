from activation_benchmark.multistart import (
    checkpoint_peuaf_frequencies,
    evolve_frequencies,
    initial_population,
    set_initial_peuaf_frequency,
)


def test_grid_population_excludes_frequency_boundaries():
    population = initial_population(
        {
            "population_size": 3,
            "initial_strategy": "grid",
            "minimum_frequency": 0.0,
            "maximum_frequency": 1.0,
        },
        seed=42,
    )
    assert population == [0.25, 0.5, 0.75]


def test_evolution_is_reproducible_and_bounded():
    first = evolve_frequencies(
        [0.3, 0.4],
        population_size=6,
        mutation_std=0.2,
        seed=42,
    )
    second = evolve_frequencies(
        [0.3, 0.4],
        population_size=6,
        mutation_std=0.2,
        seed=42,
    )
    assert first == second
    assert all(0.0 <= value <= 1.0 for value in first)


def test_configures_mixed_model_peuaf_frequency():
    config = {
        "model": {
            "activation": "relu",
            "activation_policy": "mixed_last_block",
            "peuaf_kwargs": {"initial_w": 0.5},
        }
    }
    set_initial_peuaf_frequency(config, 0.25)
    assert config["model"]["peuaf_kwargs"]["initial_w"] == 0.25


def test_reads_learned_peuaf_frequencies_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    import torch

    torch.save(
        {
            "best_model": {
                "features.0.w": torch.tensor(0.25),
                "features.1.w": torch.tensor([0.5, 0.75]),
                "classifier.weight": torch.zeros(1, 1),
            }
        },
        checkpoint,
    )
    assert checkpoint_peuaf_frequencies(checkpoint) == [0.25, 0.625]
