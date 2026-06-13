import numpy as np
from PIL import Image

from activation_benchmark.data import (
    _cifar10_transforms,
    _mnist_transforms,
    synthetic_pqd_dataset,
)


def test_cifar10_transforms_preserve_image_shape():
    config = {
        "data": {
            "augmentation": {
                "enabled": True,
                "crop_padding": 4,
                "horizontal_flip_probability": 0.5,
                "random_erasing_probability": 0.0,
            }
        }
    }
    training, evaluation = _cifar10_transforms(config)
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))

    assert training(image).shape == (3, 32, 32)
    assert evaluation(image).shape == (3, 32, 32)


def test_mnist_transforms_preserve_image_shape():
    config = {
        "data": {
            "augmentation": {
                "enabled": False,
                "random_erasing_probability": 0.0,
            }
        }
    }
    training, evaluation = _mnist_transforms(config)
    image = Image.fromarray(np.zeros((28, 28), dtype=np.uint8))

    assert training(image).shape == (1, 28, 28)
    assert evaluation(image).shape == (1, 28, 28)


def test_synthetic_pqd_dataset_is_balanced_and_deterministic():
    first = synthetic_pqd_dataset(
        80,
        signal_length=128,
        noise_std=0.05,
        seed=12,
    )
    second = synthetic_pqd_dataset(
        80,
        signal_length=128,
        noise_std=0.05,
        seed=12,
    )

    signals, labels = first.tensors
    assert signals.shape == (80, 1, 128)
    assert labels.bincount().tolist() == [10] * 8
    assert np.array_equal(signals.numpy(), second.tensors[0].numpy())
