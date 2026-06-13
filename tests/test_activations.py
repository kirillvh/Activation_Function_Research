import pytest
import torch

from activation_benchmark.activations import (
    EUAF,
    PEUAF,
    SineTriangle,
    activation_names,
    make_activation,
    triangle_wave,
)
from activation_benchmark.model import (
    CIFAR10Classifier,
    CIFAR10DeepClassifier,
    CIFAR10ResNet,
    MNISTClassifier,
    PQDSignalClassifier,
    build_model,
)


@pytest.mark.parametrize("name", activation_names())
def test_activation_preserves_shape(name):
    activation = make_activation(name)
    inputs = torch.randn(4, 8)
    assert activation(inputs).shape == inputs.shape


def test_model_output_shape():
    model = MNISTClassifier(
        activation="gelu",
        channels=[8, 16],
        hidden_features=32,
    )
    assert model(torch.randn(5, 1, 28, 28)).shape == (5, 10)


def test_euaf_matches_paper_definition():
    activation = EUAF()
    inputs = torch.tensor([-2.0, 0.0, 0.5, 1.5, 2.0, 3.5])
    positive = torch.abs(
        inputs - 2.0 * torch.floor((inputs + 1.0) / 2.0)
    )
    negative = inputs / (1.0 + torch.abs(inputs))
    expected = torch.where(inputs >= 0, positive, negative)

    assert torch.allclose(activation(inputs), expected)


def test_peuaf_matches_paper_definition_and_w_is_trainable():
    activation = make_activation("peuaf", initial_w=0.5)
    inputs = torch.tensor([0.25, 0.75, 1.25, 2.0])
    w = activation.w.detach()
    wx = w * inputs
    expected = torch.abs(
        wx - 2.0 * torch.floor((wx + 1.0) / 2.0)
    )

    assert torch.allclose(activation(inputs), expected)
    activation(inputs).sum().backward()

    assert isinstance(activation.w, torch.nn.Parameter)
    assert activation.w.grad is not None
    assert torch.isfinite(activation.w.grad)
    assert activation.w.grad.abs() > 0


def test_peuaf_negative_branch_and_zero_are_paper_exact():
    activation = PEUAF(initial_w=0.5)
    inputs = torch.tensor([-2.0, -0.5, 0.0])
    expected = inputs / (1.0 + torch.abs(inputs))
    assert torch.allclose(activation(inputs), expected)


def test_peuaf_parameter_projection():
    activation = PEUAF(initial_w=0.5)
    with torch.no_grad():
        activation.w.fill_(2.0)
    activation.constrain_parameters()
    assert activation.w.item() == pytest.approx(1.0)


def test_channelwise_peuaf_has_independent_frequencies():
    activation = PEUAF(initial_w=0.5, num_parameters=3)
    inputs = torch.ones(2, 3, 4)
    with torch.no_grad():
        activation.w.copy_(torch.tensor([0.25, 0.5, 0.75]))

    output = activation(inputs)

    assert activation.w.shape == (3,)
    assert output[0, :, 0].tolist() == pytest.approx([0.25, 0.5, 0.75])


def test_triangle_wave_matches_arcsine_formula_with_finite_corner_gradients():
    inputs = torch.tensor(
        [-torch.pi / 2, -0.3, 0.0, 0.7, torch.pi / 2],
        requires_grad=True,
    )
    values = triangle_wave(inputs)
    expected = 2.0 * torch.asin(torch.sin(inputs.detach())) / torch.pi

    assert torch.allclose(values.detach(), expected, atol=1e-6)
    values.sum().backward()
    assert torch.isfinite(inputs.grad).all()


def test_sine_triangle_parameters_are_trainable_and_bounded():
    activation = SineTriangle(
        initial_w=1.2,
        initial_blend=0.4,
        minimum_w=0.1,
        maximum_w=3.0,
    )
    inputs = torch.randn(8, requires_grad=True)
    activation(inputs).sum().backward()

    assert activation.frequency.item() == pytest.approx(1.2)
    assert activation.blend.item() == pytest.approx(0.4)
    assert activation.frequency_logit.grad is not None
    assert activation.blend_logit.grad is not None
    assert torch.isfinite(activation.frequency_logit.grad)
    assert torch.isfinite(activation.blend_logit.grad)


def test_residual_sine_triangle_preserves_an_identity_gradient_path():
    activation = make_activation(
        "sine_triangle_residual",
        initial_w=1.0,
        initial_blend=0.5,
    )
    inputs = torch.linspace(-1.0, 1.0, 101, requires_grad=True)
    activation(inputs).sum().backward()

    assert inputs.grad.min().item() > 0.7


@pytest.mark.parametrize("name", ["silu_sine_triangle", "gelu_sine_triangle"])
def test_baseline_residual_can_learn_periodic_amplitude(name):
    activation = make_activation(name)
    inputs = torch.randn(16, requires_grad=True)
    activation(inputs).sum().backward()

    assert activation.periodic_scale.item() == pytest.approx(0.1)
    assert activation.periodic_scale_logit.grad is not None
    assert torch.isfinite(activation.periodic_scale_logit.grad)


def test_model_supports_euaf():
    model = MNISTClassifier(
        activation="peuaf",
        activation_kwargs={"initial_w": 0.25},
        channels=[8, 16],
        hidden_features=32,
    )
    assert model(torch.randn(5, 1, 28, 28)).shape == (5, 10)
    assert sum(parameter.numel() for parameter in model.parameters()) == 26701


def test_cifar10_model_output_shape_and_euaf_parameters():
    model = CIFAR10Classifier(
        activation="peuaf",
        activation_kwargs={"initial_w": 0.25},
        channels=[8, 16, 32],
        hidden_features=64,
    )
    assert model(torch.randn(5, 3, 32, 32)).shape == (5, 10)
    euaf_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith(".w")
    ]
    assert len(euaf_parameters) == 7


def test_model_factory_dispatches_by_dataset():
    common = {
        "model": {
            "activation": "relu",
            "channels": [8, 16, 32],
            "hidden_features": 32,
            "dropout": 0.1,
        },
        "data": {"dataset": "cifar10"},
    }
    assert isinstance(build_model(common), CIFAR10Classifier)


def test_cifar100_model_uses_one_hundred_classes():
    config = {
        "model": {
            "activation": "sine_triangle",
            "channels": [8, 16, 32],
            "hidden_features": 32,
            "dropout": 0.1,
        },
        "data": {"dataset": "cifar100"},
    }
    model = build_model(config)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 100)


def test_deep_cifar10_model_is_deeper_and_supports_euaf():
    standard = CIFAR10Classifier(
        activation="peuaf",
        channels=[8, 16, 32],
        hidden_features=64,
    )
    deep = CIFAR10DeepClassifier(
        activation="peuaf",
        channels=[8, 16, 32, 64],
        block_depths=[2, 2, 3, 3],
        hidden_features=64,
    )

    standard_convolutions = sum(
        isinstance(module, torch.nn.Conv2d) for module in standard.modules()
    )
    deep_convolutions = sum(
        isinstance(module, torch.nn.Conv2d) for module in deep.modules()
    )
    deep_euaf_parameters = [
        parameter
        for name, parameter in deep.named_parameters()
        if name.endswith(".w")
    ]

    assert deep(torch.randn(5, 3, 32, 32)).shape == (5, 10)
    assert standard_convolutions == 6
    assert deep_convolutions == 10
    assert len(deep_euaf_parameters) == 11


def test_model_factory_selects_deep_cifar10_model():
    config = {
        "model": {
            "architecture": "deep",
            "activation": "relu",
            "channels": [8, 16, 32, 64],
            "block_depths": [2, 2, 3, 3],
            "hidden_features": 64,
            "dropout": 0.1,
        },
        "data": {"dataset": "cifar10"},
    }
    assert isinstance(build_model(config), CIFAR10DeepClassifier)


def test_signal_model_can_be_built_and_run():
    config = {
        "data": {"dataset": "synthetic_pqd"},
        "model": {
            "architecture": "signal_cnn",
            "activation": "peuaf",
            "activation_kwargs": {},
            "peuaf_kwargs": {"initial_w": 0.5},
            "channels": [4, 8, 16],
            "num_classes": 8,
        },
    }
    model = build_model(config)

    assert isinstance(model, PQDSignalClassifier)
    assert model(torch.randn(2, 1, 128)).shape == (2, 8)


@pytest.mark.parametrize(
    ("policy", "expected_peuaf_modules"),
    [
        ("baseline", 0),
        ("mixed_last_activation", 1),
        ("mixed_last_block", 2),
        ("peuaf_all", 6),
    ],
)
def test_signal_activation_policies(policy, expected_peuaf_modules):
    model = PQDSignalClassifier(
        activation="relu",
        channels=[4, 8, 16],
        peuaf_kwargs={"initial_w": 0.5},
        activation_policy=policy,
    )

    assert model(torch.randn(2, 1, 128)).shape == (2, 8)
    assert (
        sum(isinstance(module, PEUAF) for module in model.modules())
        == expected_peuaf_modules
    )


def test_signal_channelwise_peuaf_matches_feature_widths():
    model = PQDSignalClassifier(
        activation="peuaf",
        channels=[4, 8, 16],
        peuaf_kwargs={"initial_w": 0.5},
        peuaf_per_channel=True,
    )
    frequency_shapes = [
        tuple(module.w.shape)
        for module in model.modules()
        if isinstance(module, PEUAF)
    ]

    assert frequency_shapes == [(4,), (4,), (8,), (8,), (16,), (16,)]


@pytest.mark.parametrize(
    ("policy", "expected_peuaf_modules"),
    [
        ("baseline", 0),
        ("mixed_last_block", 2),
        ("mixed_last_stage", 4),
        ("peuaf_all", 17),
    ],
)
def test_resnet_activation_policies(policy, expected_peuaf_modules):
    model = CIFAR10ResNet(
        activation="relu",
        channels=[8, 16, 32, 64],
        activation_policy=policy,
        peuaf_kwargs={"initial_w": 0.5},
    )
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert (
        sum(isinstance(module, PEUAF) for module in model.modules())
        == expected_peuaf_modules
    )


def test_unknown_activation_is_helpful():
    with pytest.raises(ValueError, match="Unknown activation"):
        make_activation("does_not_exist")
