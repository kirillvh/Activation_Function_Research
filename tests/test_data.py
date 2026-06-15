import wave

import numpy as np
import torch
from PIL import Image

from activation_benchmark.data import (
    MiniSpeechCommandsDataset,
    _cifar10_transforms,
    _mnist_transforms,
    _read_pcm_wav,
    _speech_split,
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


def test_pcm_wav_loader_pads_to_requested_length(tmp_path):
    path = tmp_path / "clip.wav"
    samples = np.asarray([-32768, -1000, 1000, 32767], dtype="<i2")
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(samples.tobytes())

    waveform = _read_pcm_wav(
        path,
        sample_rate=16000,
        sample_length=8,
    )

    assert waveform.dtype == torch.int16
    assert waveform.tolist() == [-32768, -1000, 1000, 32767, 0, 0, 0, 0]


def test_speech_split_keeps_a_speaker_in_one_partition():
    first = _speech_split(
        "yes/person_nohash_0.wav",
        split_seed=123,
        validation_percentage=10,
        test_percentage=10,
    )
    second = _speech_split(
        "no/person_nohash_12.wav",
        split_seed=123,
        validation_percentage=10,
        test_percentage=10,
    )

    assert first == second


def test_mini_speech_dataset_returns_normalized_channel_first_audio():
    dataset = MiniSpeechCommandsDataset(
        torch.tensor([[0, 16384, -32768, 8192]], dtype=torch.int16),
        torch.tensor([3]),
        [0],
    )

    waveform, label = dataset[0]

    assert waveform.shape == (1, 4)
    assert waveform.abs().max().item() == 1.0
    assert label.item() == 3
