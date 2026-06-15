from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .activations import make_activation


class MNISTClassifier(nn.Module):
    def __init__(
        self,
        activation: str,
        activation_kwargs: dict[str, Any] | None = None,
        channels: list[int] | tuple[int, int] = (16, 32),
        hidden_features: int = 64,
        dropout: float = 0.15,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if len(channels) != 2:
            raise ValueError("model.channels must contain exactly two values")
        first_channels, second_channels = channels
        activation_kwargs = activation_kwargs or {}

        self.features = nn.Sequential(
            nn.Conv2d(1, first_channels, kernel_size=3, padding=1),
            make_activation(activation, **activation_kwargs),
            nn.MaxPool2d(2),
            nn.Conv2d(first_channels, second_channels, kernel_size=3, padding=1),
            make_activation(activation, **activation_kwargs),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(second_channels * 7 * 7, hidden_features),
            make_activation(activation, **activation_kwargs),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class CIFAR10Classifier(nn.Module):
    def __init__(
        self,
        activation: str,
        activation_kwargs: dict[str, Any] | None = None,
        channels: list[int] | tuple[int, int, int] = (32, 64, 128),
        hidden_features: int = 128,
        dropout: float = 0.2,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError(
                "CIFAR-10 model.channels must contain exactly three values"
            )
        first_channels, second_channels, third_channels = channels
        activation_kwargs = activation_kwargs or {}

        self.features = nn.Sequential(
            nn.Conv2d(3, first_channels, kernel_size=3, padding=1),
            make_activation(activation, **activation_kwargs),
            nn.Conv2d(first_channels, first_channels, kernel_size=3, padding=1),
            make_activation(activation, **activation_kwargs),
            nn.MaxPool2d(2),
            nn.Conv2d(first_channels, second_channels, kernel_size=3, padding=1),
            make_activation(activation, **activation_kwargs),
            nn.Conv2d(second_channels, second_channels, kernel_size=3, padding=1),
            make_activation(activation, **activation_kwargs),
            nn.MaxPool2d(2),
            nn.Conv2d(second_channels, third_channels, kernel_size=3, padding=1),
            make_activation(activation, **activation_kwargs),
            nn.Conv2d(third_channels, third_channels, kernel_size=3, padding=1),
            make_activation(activation, **activation_kwargs),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(third_channels * 4 * 4, hidden_features),
            make_activation(activation, **activation_kwargs),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class CIFAR10DeepClassifier(nn.Module):
    def __init__(
        self,
        activation: str,
        activation_kwargs: dict[str, Any] | None = None,
        channels: list[int] | tuple[int, int, int, int] = (32, 64, 128, 256),
        block_depths: list[int] | tuple[int, int, int, int] = (2, 2, 3, 3),
        hidden_features: int = 256,
        dropout: float = 0.3,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError(
                "Deep CIFAR-10 model.channels must contain exactly four values"
            )
        if len(block_depths) != len(channels):
            raise ValueError(
                "Deep CIFAR-10 model.block_depths must match model.channels"
            )
        if any(depth < 1 for depth in block_depths):
            raise ValueError(
                "Deep CIFAR-10 model.block_depths values must be positive"
            )
        activation_kwargs = activation_kwargs or {}

        layers: list[nn.Module] = []
        input_channels = 3
        for output_channels, depth in zip(channels, block_depths):
            for _ in range(depth):
                layers.extend(
                    [
                        nn.Conv2d(
                            input_channels,
                            output_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(output_channels),
                        make_activation(activation, **activation_kwargs),
                    ]
                )
                input_channels = output_channels
            layers.append(nn.MaxPool2d(2))

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], hidden_features),
            make_activation(activation, **activation_kwargs),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(inputs)))


class CIFARBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int,
        activation: str,
        activation_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.activation1 = make_activation(activation, **activation_kwargs)
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(output_channels)
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation2 = make_activation(activation, **activation_kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        output = self.activation1(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        return self.activation2(output + residual)


class CIFAR10ResNet(nn.Module):
    def __init__(
        self,
        activation: str,
        activation_kwargs: dict[str, Any] | None = None,
        channels: list[int] | tuple[int, int, int, int] = (64, 128, 256, 512),
        block_depths: list[int] | tuple[int, int, int, int] = (2, 2, 2, 2),
        activation_policy: str = "baseline",
        peuaf_stages: list[int] | tuple[int, ...] = (),
        peuaf_last_n_blocks: int = 0,
        peuaf_kwargs: dict[str, Any] | None = None,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if len(channels) != 4 or len(block_depths) != 4:
            raise ValueError(
                "CIFAR ResNet channels and block_depths must contain four values"
            )
        if any(depth < 1 for depth in block_depths):
            raise ValueError("CIFAR ResNet block depths must be positive")
        invalid_stages = set(peuaf_stages) - {1, 2, 3, 4}
        if invalid_stages:
            raise ValueError("model.peuaf_stages values must be from 1 to 4")
        policies = {
            "baseline",
            "mixed_last_block",
            "mixed_last_stage",
            "peuaf_all",
            "custom",
        }
        if activation_policy not in policies:
            raise ValueError(
                f"Unknown ResNet activation policy: {activation_policy!r}"
            )
        total_blocks = sum(block_depths)
        if not 0 <= peuaf_last_n_blocks <= total_blocks:
            raise ValueError(
                "model.peuaf_last_n_blocks must be between zero "
                "and the total residual block count"
            )

        activation_kwargs = activation_kwargs or {}
        peuaf_kwargs = peuaf_kwargs or {}
        if activation_policy == "mixed_last_block":
            peuaf_last_n_blocks = 1
        elif activation_policy == "mixed_last_stage":
            peuaf_stages = [4]
        elif activation_policy == "peuaf_all":
            peuaf_stages = [1, 2, 3, 4]

        stem_activation = (
            "peuaf" if activation_policy == "peuaf_all" else activation
        )
        stem_kwargs = (
            peuaf_kwargs
            if activation_policy == "peuaf_all"
            else activation_kwargs
        )
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels[0]),
            make_activation(stem_activation, **stem_kwargs),
        )

        input_channels = channels[0]
        stages: list[nn.Module] = []
        global_block_index = 0
        for stage_index, (output_channels, depth) in enumerate(
            zip(channels, block_depths),
            start=1,
        ):
            blocks: list[nn.Module] = []
            for block_index in range(depth):
                use_peuaf = (
                    stage_index in peuaf_stages
                    or global_block_index >= total_blocks - peuaf_last_n_blocks
                )
                block_activation = "peuaf" if use_peuaf else activation
                block_kwargs = peuaf_kwargs if use_peuaf else activation_kwargs
                stride = 2 if stage_index > 1 and block_index == 0 else 1
                blocks.append(
                    CIFARBasicBlock(
                        input_channels=input_channels,
                        output_channels=output_channels,
                        stride=stride,
                        activation=block_activation,
                        activation_kwargs=block_kwargs,
                    )
                )
                input_channels = output_channels
                global_block_index += 1
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels[-1], num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.stem(inputs)
        for stage in self.stages:
            output = stage(output)
        output = self.pool(output).flatten(1)
        return self.classifier(output)


class PQDSignalClassifier(nn.Module):
    """Compact six-convolution network for 1D power-quality signals."""

    def __init__(
        self,
        activation: str,
        activation_kwargs: dict[str, Any] | None = None,
        peuaf_kwargs: dict[str, Any] | None = None,
        channels: list[int] | tuple[int, int, int] = (16, 32, 64),
        num_classes: int = 8,
        activation_policy: str = "baseline",
        peuaf_per_channel: bool = False,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError(
                "Signal model.channels must contain exactly three values"
            )
        policies = {
            "baseline",
            "mixed_last_activation",
            "mixed_last_block",
            "peuaf_all",
        }
        if activation_policy not in policies:
            raise ValueError(
                f"Unknown signal activation policy: {activation_policy!r}"
            )
        activation_kwargs = activation_kwargs or {}
        peuaf_kwargs = peuaf_kwargs or {}
        blocks: list[nn.Module] = []
        input_channels = 1
        for stage_index, output_channels in enumerate(channels):
            stage_uses_peuaf = (
                activation_policy == "peuaf_all"
                or (
                    activation_policy == "mixed_last_block"
                    and stage_index == len(channels) - 1
                )
            )
            first_activation = "peuaf" if stage_uses_peuaf else activation
            second_activation = first_activation
            if (
                activation_policy == "mixed_last_activation"
                and stage_index == len(channels) - 1
            ):
                second_activation = "peuaf"
            first_kwargs = (
                {
                    **peuaf_kwargs,
                    **(
                        {"num_parameters": output_channels}
                        if peuaf_per_channel
                        else {}
                    ),
                }
                if first_activation == "peuaf"
                else activation_kwargs
            )
            second_kwargs = (
                {
                    **peuaf_kwargs,
                    **(
                        {"num_parameters": output_channels}
                        if peuaf_per_channel
                        else {}
                    ),
                }
                if second_activation == "peuaf"
                else activation_kwargs
            )
            blocks.extend(
                [
                    nn.Conv1d(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    make_activation(first_activation, **first_kwargs),
                    nn.Conv1d(
                        output_channels,
                        output_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    make_activation(second_activation, **second_kwargs),
                    nn.BatchNorm1d(output_channels),
                    nn.MaxPool1d(kernel_size=3, stride=1),
                ]
            )
            input_channels = output_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(channels[-1], num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.features(inputs)).flatten(1)
        return self.classifier(features)


class RawAudioClassifier(nn.Module):
    """Compact strided 1D CNN for one-second raw speech waveforms."""

    def __init__(
        self,
        activation: str,
        activation_kwargs: dict[str, Any] | None = None,
        channels: list[int] | tuple[int, ...] = (16, 32, 64, 96, 128),
        kernel_sizes: list[int] | tuple[int, ...] = (80, 9, 9, 9, 9),
        strides: list[int] | tuple[int, ...] = (4, 4, 4, 2, 2),
        dropout: float = 0.2,
        num_classes: int = 8,
    ) -> None:
        super().__init__()
        if not (
            len(channels) == len(kernel_sizes) == len(strides)
            and len(channels) > 0
        ):
            raise ValueError(
                "Audio channels, kernel_sizes, and strides must have "
                "the same non-zero length"
            )
        activation_kwargs = activation_kwargs or {}
        layers: list[nn.Module] = []
        input_channels = 1
        for output_channels, kernel_size, stride in zip(
            channels,
            kernel_sizes,
            strides,
        ):
            layers.extend(
                [
                    nn.Conv1d(
                        input_channels,
                        output_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(output_channels),
                    make_activation(activation, **activation_kwargs),
                ]
            )
            input_channels = output_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(inputs)))


def build_model(config: dict[str, Any]) -> nn.Module:
    model_config = config["model"]
    dataset_name = str(config["data"].get("dataset", "mnist")).lower()
    architecture = str(model_config.get("architecture", "standard")).lower()
    default_num_classes = 100 if dataset_name == "cifar100" else 10
    common = {
        "activation": model_config["activation"],
        "activation_kwargs": model_config.get("activation_kwargs"),
        "hidden_features": model_config.get("hidden_features", 64),
        "dropout": model_config.get("dropout", 0.15),
        "num_classes": model_config.get(
            "num_classes",
            default_num_classes,
        ),
    }
    if dataset_name == "mnist":
        if architecture != "standard":
            raise ValueError(
                "MNIST model.architecture must be 'standard'"
            )
        return MNISTClassifier(
            channels=model_config.get("channels", [16, 32]),
            **common,
        )
    if dataset_name in {"cifar10", "cifar100"}:
        if architecture == "standard":
            return CIFAR10Classifier(
                channels=model_config.get("channels", [32, 64, 128]),
                **common,
            )
        if architecture == "deep":
            return CIFAR10DeepClassifier(
                channels=model_config.get("channels", [32, 64, 128, 256]),
                block_depths=model_config.get("block_depths", [2, 2, 3, 3]),
                **common,
            )
        if architecture == "resnet18":
            return CIFAR10ResNet(
                activation=model_config["activation"],
                activation_kwargs=model_config.get("activation_kwargs"),
                channels=model_config.get(
                    "channels",
                    [64, 128, 256, 512],
                ),
                block_depths=model_config.get(
                    "block_depths",
                    [2, 2, 2, 2],
                ),
                activation_policy=model_config.get(
                    "activation_policy",
                    "baseline",
                ),
                peuaf_stages=model_config.get("peuaf_stages", []),
                peuaf_last_n_blocks=model_config.get(
                    "peuaf_last_n_blocks",
                    0,
                ),
                peuaf_kwargs=model_config.get("peuaf_kwargs"),
                num_classes=model_config.get(
                    "num_classes",
                    default_num_classes,
                ),
            )
        raise ValueError(
            "CIFAR model.architecture must be one of: "
            "standard, deep, resnet18"
        )
    if dataset_name == "synthetic_pqd":
        if architecture != "signal_cnn":
            raise ValueError(
                "Synthetic PQD model.architecture must be 'signal_cnn'"
            )
        return PQDSignalClassifier(
            activation=model_config["activation"],
            activation_kwargs=model_config.get("activation_kwargs"),
            peuaf_kwargs=model_config.get("peuaf_kwargs"),
            channels=model_config.get("channels", [16, 32, 64]),
            num_classes=model_config.get("num_classes", 8),
            activation_policy=model_config.get(
                "activation_policy",
                "baseline",
            ),
            peuaf_per_channel=model_config.get(
                "peuaf_per_channel",
                False,
            ),
        )
    if dataset_name == "mini_speech_commands":
        if architecture != "raw_audio_cnn":
            raise ValueError(
                "Mini Speech Commands model.architecture must be "
                "'raw_audio_cnn'"
            )
        return RawAudioClassifier(
            activation=model_config["activation"],
            activation_kwargs=model_config.get("activation_kwargs"),
            channels=model_config.get(
                "channels",
                [16, 32, 64, 96, 128],
            ),
            kernel_sizes=model_config.get(
                "kernel_sizes",
                [80, 9, 9, 9, 9],
            ),
            strides=model_config.get("strides", [4, 4, 4, 2, 2]),
            dropout=model_config.get("dropout", 0.2),
            num_classes=model_config.get("num_classes", 8),
        )
    raise ValueError(f"Unsupported dataset: {dataset_name!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
