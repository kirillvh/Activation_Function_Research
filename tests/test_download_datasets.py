import pytest

from activation_benchmark import download_datasets as downloader


def test_normalize_dataset_names_handles_all_aliases_and_duplicates():
    assert downloader.normalize_dataset_names(["all"]) == [
        "mnist",
        "cifar10",
        "cifar100",
    ]
    assert downloader.normalize_dataset_names(
        ["CIFAR-10", "mnist", "cifar10"]
    ) == ["cifar10", "mnist"]


def test_normalize_dataset_names_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown datasets"):
        downloader.normalize_dataset_names(["imagenet"])


def test_download_datasets_prepares_train_and_test_splits(
    tmp_path,
    monkeypatch,
):
    calls = []

    class FakeDataset:
        def __init__(self, *, root, train, download):
            calls.append(
                {
                    "root": root,
                    "train": train,
                    "download": download,
                }
            )

    monkeypatch.setattr(
        downloader,
        "_dataset_classes",
        lambda: {
            "mnist": FakeDataset,
            "cifar10": FakeDataset,
            "cifar100": FakeDataset,
        },
    )

    completed = downloader.download_datasets(
        ["mnist", "cifar10"],
        root=tmp_path / "datasets",
    )

    assert completed == ["mnist", "cifar10"]
    assert [call["train"] for call in calls] == [True, False, True, False]
    assert all(call["download"] for call in calls)
    assert all(call["root"] == str(tmp_path / "datasets") for call in calls)
