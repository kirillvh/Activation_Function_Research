import torch

from activation_benchmark.model import RawAudioClassifier, build_model


def test_raw_audio_classifier_output_shape():
    model = RawAudioClassifier(
        activation="gelu",
        channels=[4, 8],
        kernel_sizes=[16, 5],
        strides=[4, 2],
        num_classes=8,
    )

    output = model(torch.randn(3, 1, 1600))

    assert output.shape == (3, 8)


def test_build_model_supports_mini_speech_commands():
    model = build_model(
        {
            "data": {"dataset": "mini_speech_commands"},
            "model": {
                "architecture": "raw_audio_cnn",
                "activation": "peuaf",
                "activation_kwargs": {"initial_w": 0.4},
                "channels": [4, 8],
                "kernel_sizes": [16, 5],
                "strides": [4, 2],
                "num_classes": 8,
            },
        }
    )

    assert model(torch.randn(2, 1, 1600)).shape == (2, 8)
