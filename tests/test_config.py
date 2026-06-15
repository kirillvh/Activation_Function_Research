import pytest

from activation_benchmark.config import (
    deep_merge,
    load_config,
    parse_override,
    set_by_path,
)


def test_deep_merge_preserves_nested_values():
    merged = deep_merge(
        {"model": {"activation": "relu", "dropout": 0.1}},
        {"model": {"activation": "gelu"}},
    )
    assert merged == {"model": {"activation": "gelu", "dropout": 0.1}}


def test_dotted_override(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment: {name: test, output_dir: runs, seed: 1}
data:
  validation_size: 100
  batch_size: 16
model: {activation: relu}
training: {epochs: 1, learning_rate: 0.001}
checkpoint: {enabled: true, save_every_steps: 2, keep_latest: 3}
tensorboard: {enabled: false}
""",
        encoding="utf-8",
    )
    config = load_config(
        config_path,
        ["model.activation=gelu", "training.epochs=2"],
    )
    assert config["model"]["activation"] == "gelu"
    assert config["training"]["epochs"] == 2


def test_parse_and_set_list_value():
    key, value = parse_override("model.channels=[8, 16]")
    config = {}
    set_by_path(config, key, value)
    assert config == {"model": {"channels": [8, 16]}}


def test_cifar10_config_is_valid():
    config = load_config("configs/cifar10_smoke.yaml")
    assert config["data"]["dataset"] == "cifar10"


def test_deep_cifar10_config_is_valid():
    config = load_config("configs/cifar10_deep_smoke.yaml")
    assert config["model"]["architecture"] == "deep"
    assert config["model"]["block_depths"] == [2, 2, 3, 3]


def test_cifar100_uses_cifar_architectures():
    config = load_config("configs/cifar10_smoke.yaml")
    config["data"]["dataset"] = "cifar100"
    from activation_benchmark.config import validate_training_config

    validate_training_config(config)


def test_synthetic_pqd_config_is_valid():
    config = load_config("configs/synthetic_pqd.yaml")
    assert config["model"]["architecture"] == "signal_cnn"


def test_mini_speech_commands_config_is_valid():
    config = load_config("configs/audio_mini_speech_commands_smoke.yaml")
    assert config["model"]["architecture"] == "raw_audio_cnn"


def test_unknown_dataset_is_rejected():
    config = load_config("configs/cifar10_smoke.yaml")
    config["data"]["dataset"] = "unknown"
    from activation_benchmark.config import validate_training_config

    with pytest.raises(ValueError, match="data.dataset"):
        validate_training_config(config)


def test_unknown_architecture_is_rejected():
    config = load_config("configs/cifar10_smoke.yaml")
    config["model"]["architecture"] = "unknown"
    from activation_benchmark.config import validate_training_config

    with pytest.raises(ValueError, match="model.architecture"):
        validate_training_config(config)
