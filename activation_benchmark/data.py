from __future__ import annotations

import hashlib
import random
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


@dataclass(frozen=True)
class DatasetLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


_MINI_SPEECH_CACHE: dict[
    tuple[str, int, int],
    tuple[torch.Tensor, torch.Tensor, tuple[str, ...]],
] = {}


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _torchvision_transforms():
    try:
        from torchvision import transforms
    except ImportError as error:
        raise RuntimeError(
            "torchvision is required for image datasets. "
            "Install requirements.txt first."
        ) from error
    return transforms


def _mnist_transforms(config: dict[str, Any]):
    transforms = _torchvision_transforms()

    augmentation = config["data"]["augmentation"]
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    evaluation = transforms.Compose([transforms.ToTensor(), normalize])

    operations: list[Any] = []
    if augmentation.get("enabled", False):
        operations.append(
            transforms.RandomAffine(
                degrees=augmentation.get("rotation_degrees", 0),
                translate=(
                    augmentation.get("translate", 0),
                    augmentation.get("translate", 0),
                ),
                scale=(
                    augmentation.get("scale_min", 1.0),
                    augmentation.get("scale_max", 1.0),
                ),
            )
        )
    operations.extend([transforms.ToTensor(), normalize])
    erasing_probability = augmentation.get("random_erasing_probability", 0.0)
    if augmentation.get("enabled", False) and erasing_probability > 0:
        operations.append(transforms.RandomErasing(p=erasing_probability))
    training = transforms.Compose(operations)
    return training, evaluation


def _cifar_transforms(config: dict[str, Any], dataset_name: str):
    transforms = _torchvision_transforms()
    augmentation = config["data"]["augmentation"]
    if dataset_name == "cifar100":
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        )
    else:
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        )
    evaluation = transforms.Compose([transforms.ToTensor(), normalize])

    operations: list[Any] = []
    if augmentation.get("enabled", False):
        crop_padding = augmentation.get("crop_padding", 4)
        if crop_padding > 0:
            operations.append(
                transforms.RandomCrop(32, padding=crop_padding)
            )
        flip_probability = augmentation.get(
            "horizontal_flip_probability",
            0.5,
        )
        if flip_probability > 0:
            operations.append(
                transforms.RandomHorizontalFlip(p=flip_probability)
            )
        rotation = augmentation.get("rotation_degrees", 0)
        translate = augmentation.get("translate", 0)
        shear = augmentation.get("shear_degrees", 0)
        if rotation or translate or shear:
            operations.append(
                transforms.RandomAffine(
                    degrees=rotation,
                    translate=(translate, translate),
                    shear=shear,
                )
            )
    operations.extend([transforms.ToTensor(), normalize])
    erasing_probability = augmentation.get("random_erasing_probability", 0.0)
    if augmentation.get("enabled", False) and erasing_probability > 0:
        operations.append(transforms.RandomErasing(p=erasing_probability))
    return transforms.Compose(operations), evaluation


def _cifar10_transforms(config: dict[str, Any]):
    """Backward-compatible CIFAR-10 transform entry point."""
    return _cifar_transforms(config, "cifar10")


def _dataset_classes(dataset_name: str):
    try:
        from torchvision.datasets import CIFAR10, CIFAR100, MNIST
    except ImportError as error:
        raise RuntimeError(
            "torchvision is required for image datasets. "
            "Install requirements.txt first."
        ) from error
    if dataset_name == "mnist":
        return MNIST
    if dataset_name == "cifar10":
        return CIFAR10
    if dataset_name == "cifar100":
        return CIFAR100
    raise ValueError(f"Unsupported dataset: {dataset_name!r}")


def synthetic_pqd_dataset(
    sample_count: int,
    *,
    signal_length: int,
    noise_std: float,
    seed: int,
) -> TensorDataset:
    """Generate balanced, noisy power-quality disturbance waveforms."""
    if sample_count < 8:
        raise ValueError("synthetic PQD datasets need at least eight samples")
    generator = torch.Generator().manual_seed(seed)
    time_axis = torch.linspace(0.0, 1.0, signal_length).unsqueeze(0)
    labels = torch.arange(sample_count) % 8
    labels = labels[torch.randperm(sample_count, generator=generator)]

    phase = 2.0 * torch.pi * torch.rand(
        sample_count, 1, generator=generator
    )
    frequency = 4.5 + torch.rand(sample_count, 1, generator=generator)
    amplitude = 0.9 + 0.2 * torch.rand(
        sample_count, 1, generator=generator
    )
    angle = 2.0 * torch.pi * frequency * time_axis + phase
    signals = amplitude * torch.sin(angle)

    start = 0.2 + 0.25 * torch.rand(
        sample_count, 1, generator=generator
    )
    duration = 0.2 + 0.2 * torch.rand(
        sample_count, 1, generator=generator
    )
    window = (time_axis >= start) & (time_axis <= start + duration)

    sag = labels == 1
    signals[sag] *= torch.where(window[sag], 0.45, 1.0)
    swell = labels == 2
    signals[swell] *= torch.where(window[swell], 1.55, 1.0)
    interruption = labels == 3
    signals[interruption] *= torch.where(
        window[interruption], 0.08, 1.0
    )

    harmonics = labels == 4
    signals[harmonics] += (
        0.28 * torch.sin(3.0 * angle[harmonics])
        + 0.16 * torch.sin(5.0 * angle[harmonics])
    )

    transient = labels == 5
    elapsed = (time_axis - start[transient]).clamp_min(0.0)
    burst = (
        (time_axis >= start[transient])
        * torch.exp(-18.0 * elapsed)
        * torch.sin(
            2.0
            * torch.pi
            * (8.0 * frequency[transient])
            * elapsed
            + phase[transient]
        )
    )
    signals[transient] += 0.8 * burst

    flicker = labels == 6
    modulation_frequency = 1.5 + torch.rand(
        int(flicker.sum()), 1, generator=generator
    )
    signals[flicker] *= 1.0 + 0.25 * torch.sin(
        2.0 * torch.pi * modulation_frequency * time_axis
    )

    notches = labels == 7
    cycle_position = torch.remainder(
        frequency[notches] * time_axis
        + phase[notches] / (2.0 * torch.pi),
        1.0,
    )
    notch_mask = (cycle_position < 0.055) | (cycle_position > 0.945)
    signals[notches] *= torch.where(notch_mask, 0.15, 1.0)

    signals += noise_std * torch.randn(
        signals.shape,
        generator=generator,
    )
    return TensorDataset(signals.unsqueeze(1).float(), labels.long())


def _read_pcm_wav(
    path: str | Path,
    *,
    sample_rate: int,
    sample_length: int,
) -> torch.Tensor:
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        file_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        frames = audio.readframes(audio.getnframes())
    if file_rate != sample_rate:
        raise ValueError(
            f"{path} uses {file_rate} Hz; expected {sample_rate} Hz"
        )
    if sample_width == 1:
        values = np.frombuffer(frames, dtype=np.uint8).astype(np.int16)
        values = (values - 128) << 8
    elif sample_width == 2:
        values = np.frombuffer(frames, dtype="<i2")
    elif sample_width == 4:
        values = (
            np.frombuffer(frames, dtype="<i4") >> 16
        ).astype(np.int16)
    else:
        raise ValueError(
            f"{path} has unsupported {sample_width}-byte PCM samples"
        )
    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1).astype(np.int16)
    waveform = torch.zeros(sample_length, dtype=torch.int16)
    copy_count = min(sample_length, len(values))
    waveform[:copy_count] = torch.from_numpy(values[:copy_count].copy())
    return waveform


def _speech_split(
    path: str | Path,
    *,
    split_seed: int,
    validation_percentage: float,
    test_percentage: float,
) -> str:
    speaker = Path(path).stem.split("_nohash_", maxsplit=1)[0]
    digest = hashlib.sha1(
        f"{split_seed}:{speaker}".encode("utf-8")
    ).hexdigest()
    percentage = int(digest[:8], 16) / 0xFFFFFFFF * 100.0
    if percentage < test_percentage:
        return "test"
    if percentage < test_percentage + validation_percentage:
        return "validation"
    return "train"


def _load_mini_speech_commands(
    dataset_dir: Path,
    *,
    sample_rate: int,
    sample_length: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    key = (str(dataset_dir.resolve()), sample_rate, sample_length)
    if key in _MINI_SPEECH_CACHE:
        return _MINI_SPEECH_CACHE[key]

    cache_path = dataset_dir / (
        f".waveform_cache_{sample_rate}hz_{sample_length}samples.pt"
    )
    if cache_path.exists():
        payload = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )
        cached = (
            payload["waveforms"],
            payload["labels"],
            tuple(payload["relative_paths"]),
        )
        _MINI_SPEECH_CACHE[key] = cached
        return cached

    class_names = tuple(
        sorted(
            path.name
            for path in dataset_dir.iterdir()
            if path.is_dir() and any(path.glob("*.wav"))
        )
    )
    if not class_names:
        raise RuntimeError(
            f"No WAV class directories found in {dataset_dir}"
        )
    paths = [
        path
        for class_name in class_names
        for path in sorted((dataset_dir / class_name).glob("*.wav"))
    ]
    print(f"Loading {len(paths)} Mini Speech Commands WAV files...")
    waveforms = torch.stack(
        [
            _read_pcm_wav(
                path,
                sample_rate=sample_rate,
                sample_length=sample_length,
            )
            for path in paths
        ]
    )
    labels = torch.tensor(
        [class_names.index(path.parent.name) for path in paths],
        dtype=torch.long,
    )
    relative_paths = tuple(
        path.relative_to(dataset_dir).as_posix() for path in paths
    )
    cached = (waveforms, labels, relative_paths)
    torch.save(
        {
            "waveforms": waveforms,
            "labels": labels,
            "relative_paths": relative_paths,
        },
        cache_path,
    )
    _MINI_SPEECH_CACHE[key] = cached
    return cached


class MiniSpeechCommandsDataset(Dataset):
    def __init__(
        self,
        waveforms: torch.Tensor,
        labels: torch.Tensor,
        indices: list[int],
        *,
        augmentation: dict[str, Any] | None = None,
    ) -> None:
        self.waveforms = waveforms
        self.labels = labels
        self.indices = indices
        self.augmentation = augmentation or {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.indices[item]
        waveform = self.waveforms[index].float() / 32768.0
        if self.augmentation.get("enabled", False):
            maximum_shift = int(
                self.augmentation.get("time_shift_samples", 0)
            )
            if maximum_shift > 0:
                shift = int(
                    torch.randint(
                        -maximum_shift,
                        maximum_shift + 1,
                        (),
                    )
                )
                waveform = torch.roll(waveform, shift)
                if shift > 0:
                    waveform[:shift] = 0
                elif shift < 0:
                    waveform[shift:] = 0
            gain_min = float(self.augmentation.get("gain_min", 1.0))
            gain_max = float(self.augmentation.get("gain_max", 1.0))
            if gain_min != 1.0 or gain_max != 1.0:
                gain = torch.empty(()).uniform_(gain_min, gain_max)
                waveform = waveform * gain
            noise_std = float(self.augmentation.get("noise_std", 0.0))
            if noise_std > 0:
                waveform = waveform + noise_std * torch.randn_like(waveform)
        if self.augmentation.get("peak_normalize", True):
            waveform = waveform / waveform.abs().max().clamp_min(1e-4)
        return waveform.unsqueeze(0), self.labels[index]


def _mini_speech_commands_loaders(
    config: dict[str, Any],
    *,
    loader_options: dict[str, Any],
    loader_generator: torch.Generator,
) -> DatasetLoaders:
    data_config = config["data"]
    root = Path(data_config["root"])
    dataset_dir = root / "mini_speech_commands"
    if not dataset_dir.exists() and data_config.get("download", True):
        from .download_datasets import download_mini_speech_commands

        dataset_dir = download_mini_speech_commands(root)
    if not dataset_dir.exists():
        raise RuntimeError(
            f"Mini Speech Commands is missing at {dataset_dir}. "
            "Run download_datasets.bat or enable data.download."
        )

    sample_rate = int(data_config.get("sample_rate", 16000))
    sample_length = int(data_config.get("sample_length", sample_rate))
    waveforms, labels, relative_paths = _load_mini_speech_commands(
        dataset_dir,
        sample_rate=sample_rate,
        sample_length=sample_length,
    )
    split_seed = int(
        data_config.get("split_seed", config["experiment"]["seed"])
    )
    validation_percentage = float(
        data_config.get("validation_percentage", 10.0)
    )
    test_percentage = float(data_config.get("test_percentage", 10.0))
    split_indices = {"train": [], "validation": [], "test": []}
    for index, relative_path in enumerate(relative_paths):
        split_indices[
            _speech_split(
                relative_path,
                split_seed=split_seed,
                validation_percentage=validation_percentage,
                test_percentage=test_percentage,
            )
        ].append(index)

    limits = {
        "train": data_config.get("max_train_samples"),
        "validation": data_config.get("max_validation_samples"),
        "test": data_config.get("max_test_samples"),
    }
    for split, limit in limits.items():
        if limit is not None:
            candidates = split_indices[split]
            generator = torch.Generator().manual_seed(
                split_seed + {"train": 11, "validation": 12, "test": 13}[split]
            )
            shuffled = torch.randperm(
                len(candidates),
                generator=generator,
            ).tolist()
            split_indices[split] = [
                candidates[index] for index in shuffled[: int(limit)]
            ]
        if not split_indices[split]:
            raise RuntimeError(
                f"Mini Speech Commands {split} split is empty"
            )

    training_dataset = MiniSpeechCommandsDataset(
        waveforms,
        labels,
        split_indices["train"],
        augmentation=data_config.get("augmentation"),
    )
    validation_dataset = MiniSpeechCommandsDataset(
        waveforms,
        labels,
        split_indices["validation"],
    )
    test_dataset = MiniSpeechCommandsDataset(
        waveforms,
        labels,
        split_indices["test"],
    )
    return DatasetLoaders(
        train=DataLoader(
            training_dataset,
            shuffle=True,
            generator=loader_generator,
            **loader_options,
        ),
        validation=DataLoader(
            validation_dataset,
            shuffle=False,
            **loader_options,
        ),
        test=DataLoader(
            test_dataset,
            shuffle=False,
            **loader_options,
        ),
    )


def build_data_loaders(config: dict[str, Any]) -> DatasetLoaders:
    dataset_name = str(config["data"].get("dataset", "mnist")).lower()
    data_config = config["data"]
    training_seed = config["experiment"]["seed"]
    split_seed = data_config.get("split_seed", training_seed)
    loader_generator = torch.Generator().manual_seed(training_seed)
    loader_options = {
        "batch_size": data_config["batch_size"],
        "num_workers": data_config.get("num_workers", 0),
        "pin_memory": data_config.get("pin_memory", False),
        "worker_init_fn": _seed_worker,
    }

    if dataset_name == "mini_speech_commands":
        return _mini_speech_commands_loaders(
            config,
            loader_options=loader_options,
            loader_generator=loader_generator,
        )

    if dataset_name == "synthetic_pqd":
        dataset_options = {
            "signal_length": data_config.get("signal_length", 256),
            "noise_std": data_config.get("noise_std", 0.08),
        }
        train_dataset = synthetic_pqd_dataset(
            data_config.get("train_size", 4096),
            seed=split_seed,
            **dataset_options,
        )
        validation_dataset = synthetic_pqd_dataset(
            data_config["validation_size"],
            seed=split_seed + 1,
            **dataset_options,
        )
        test_dataset = synthetic_pqd_dataset(
            data_config.get("test_size", 2048),
            seed=split_seed + 2,
            **dataset_options,
        )
        return DatasetLoaders(
            train=DataLoader(
                train_dataset,
                shuffle=True,
                generator=loader_generator,
                **loader_options,
            ),
            validation=DataLoader(
                validation_dataset,
                shuffle=False,
                **loader_options,
            ),
            test=DataLoader(
                test_dataset,
                shuffle=False,
                **loader_options,
            ),
        )

    dataset_class = _dataset_classes(dataset_name)
    if dataset_name == "mnist":
        training_transform, evaluation_transform = _mnist_transforms(config)
    else:
        training_transform, evaluation_transform = _cifar_transforms(
            config,
            dataset_name,
        )

    common = {
        "root": data_config["root"],
        "download": data_config.get("download", True),
    }

    train_dataset = dataset_class(
        train=True,
        transform=training_transform,
        **common,
    )
    validation_dataset = dataset_class(
        train=True,
        transform=evaluation_transform,
        **common,
    )
    test_dataset = dataset_class(
        train=False,
        transform=evaluation_transform,
        **common,
    )

    split_generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(len(train_dataset), generator=split_generator).tolist()
    validation_size = data_config["validation_size"]
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]

    max_train = data_config.get("max_train_samples")
    max_validation = data_config.get("max_validation_samples")
    max_test = data_config.get("max_test_samples")
    if max_train is not None:
        training_indices = training_indices[:max_train]
    if max_validation is not None:
        validation_indices = validation_indices[:max_validation]
    test_indices = list(range(len(test_dataset)))
    if max_test is not None:
        test_indices = test_indices[:max_test]

    return DatasetLoaders(
        train=DataLoader(
            Subset(train_dataset, training_indices),
            shuffle=True,
            generator=loader_generator,
            **loader_options,
        ),
        validation=DataLoader(
            Subset(validation_dataset, validation_indices),
            shuffle=False,
            **loader_options,
        ),
        test=DataLoader(
            Subset(test_dataset, test_indices),
            shuffle=False,
            **loader_options,
        ),
    )


def build_mnist_loaders(config: dict[str, Any]) -> DatasetLoaders:
    """Backward-compatible MNIST loader entry point."""
    config = {
        **config,
        "data": {**config["data"], "dataset": "mnist"},
    }
    return build_data_loaders(config)
