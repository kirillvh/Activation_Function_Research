import pytest
import torch

from activation_benchmark.expressivity_benchmark import (
    OneNeuronRegressor,
    triangle_wave,
)


def test_triangle_wave_matches_expected_landmarks():
    inputs = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    values = triangle_wave(inputs, frequency=1.0)
    assert values.squeeze(1).tolist() == pytest.approx([0.0, 1.0, 0.0, 1.0])


def test_one_neuron_regressor_output_shape():
    model = OneNeuronRegressor("peuaf", initial_w=0.5)
    assert model(torch.randn(7, 1)).shape == (7, 1)
