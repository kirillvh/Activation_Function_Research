import os

import pytest

from activation_benchmark import runtime
from activation_benchmark.runtime import _validated_cpu_indices


def test_cpu_affinity_indices_are_sorted_and_unique():
    assert _validated_cpu_indices([0, 0]) == [0]


def test_cpu_affinity_rejects_unavailable_index():
    with pytest.raises(ValueError):
        _validated_cpu_indices([os.cpu_count() or 1])


def test_environment_can_disable_machine_specific_affinity(monkeypatch):
    monkeypatch.setenv("ACTIVATION_BENCHMARK_DISABLE_AFFINITY", "true")
    monkeypatch.setattr(
        runtime,
        "set_process_cpu_affinity",
        lambda indices: pytest.fail("affinity should be disabled"),
    )

    settings = runtime.configure_runtime(
        {"runtime": {"cpu_affinity": [999999]}}
    )

    assert settings["cpu_affinity"] is None
    assert settings["cpu_affinity_disabled"] is True
