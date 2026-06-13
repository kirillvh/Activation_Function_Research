import pytest
import torch

from activation_benchmark.activations import PEUAF, SineTriangle
from activation_benchmark.trainer import (
    _cpu_model_state,
    build_optimizer,
    build_scheduler,
    constrain_model_parameters,
    load_initial_model_weights,
)


def test_cosine_scheduler_reaches_zero_at_target_epoch():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.003)
    config = {
        "training": {
            "epochs": 4,
            "scheduler": "cosine",
            "scheduler_min_learning_rate": 0.0,
        }
    }
    scheduler = build_scheduler(config, optimizer)

    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(config["training"]["epochs"]):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    assert rates[0] == pytest.approx(0.003)
    assert rates[-1] == pytest.approx(0.0, abs=1e-12)
    assert all(current >= following for current, following in zip(rates, rates[1:]))


def test_peuaf_uses_separate_optimizer_group_without_weight_decay():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), PEUAF(initial_w=0.5))
    config = {
        "training": {
            "optimizer": "nadam",
            "learning_rate": 0.01,
            "weight_decay": 0.001,
            "peuaf_learning_rate_multiplier": 10.0,
            "peuaf_weight_decay": 0.0,
        }
    }
    optimizer = build_optimizer(config, model)

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.1)
    assert optimizer.param_groups[1]["weight_decay"] == pytest.approx(0.0)


def test_sine_triangle_uses_general_activation_optimizer_settings():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        SineTriangle(initial_w=1.0),
    )
    config = {
        "training": {
            "optimizer": "adamw",
            "learning_rate": 0.01,
            "weight_decay": 0.001,
            "activation_learning_rate_multiplier": 0.2,
            "activation_weight_decay": 0.0,
        }
    }
    optimizer = build_optimizer(config, model)

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.002)
    assert optimizer.param_groups[1]["weight_decay"] == pytest.approx(0.0)


def test_constrain_model_parameters_projects_peuaf_frequency():
    model = torch.nn.Sequential(PEUAF(initial_w=0.5))
    with torch.no_grad():
        model[0].w.fill_(-2.0)
    constrain_model_parameters(model)
    assert model[0].w.item() == pytest.approx(0.0)


def test_cpu_model_state_is_an_independent_snapshot():
    model = torch.nn.Linear(2, 1)
    snapshot = _cpu_model_state(model)
    original_weight = snapshot["weight"].clone()

    with torch.no_grad():
        model.weight.add_(1.0)

    assert snapshot["weight"].device.type == "cpu"
    assert torch.equal(snapshot["weight"], original_weight)
    assert not torch.equal(snapshot["weight"], model.weight)


def test_load_initial_model_weights_can_select_embedded_best(tmp_path):
    source = torch.nn.Linear(2, 1)
    best = {
        key: value.detach().clone() + 2.0
        for key, value in source.state_dict().items()
    }
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": source.state_dict(), "best_model": best}, checkpoint)
    target = torch.nn.Linear(2, 1)

    load_initial_model_weights(
        target,
        checkpoint,
        weights="best",
        map_location="cpu",
    )

    for key, value in target.state_dict().items():
        assert torch.equal(value, best[key])
