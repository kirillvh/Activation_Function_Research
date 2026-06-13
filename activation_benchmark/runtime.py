from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from typing import Any

import torch

_configured_interop_threads: int | None = None


def _environment_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _validated_cpu_indices(indices: list[int] | tuple[int, ...]) -> list[int]:
    cpu_count = os.cpu_count() or 1
    normalized = sorted(set(int(index) for index in indices))
    if not normalized:
        raise ValueError("runtime.cpu_affinity cannot be empty")
    invalid = [index for index in normalized if index < 0 or index >= cpu_count]
    if invalid:
        raise ValueError(
            "runtime.cpu_affinity contains unavailable logical CPUs: "
            + ", ".join(map(str, invalid))
        )
    return normalized


def set_process_cpu_affinity(indices: list[int] | tuple[int, ...]) -> list[int]:
    normalized = _validated_cpu_indices(indices)
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessAffinityMask.argtypes = [
            wintypes.HANDLE,
            ctypes.c_size_t,
        ]
        kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        mask = sum(1 << index for index in normalized)
        if not kernel32.SetProcessAffinityMask(
            process,
            ctypes.c_size_t(mask),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    elif hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, normalized)
    else:
        raise RuntimeError(
            "CPU affinity is not supported on this operating system"
        )
    return normalized


def configure_runtime(config: dict[str, Any]) -> dict[str, Any]:
    global _configured_interop_threads

    runtime = config.get("runtime", {})
    affinity = runtime.get("cpu_affinity")
    configured_affinity: list[int] | None = None
    affinity_disabled = _environment_flag(
        "ACTIVATION_BENCHMARK_DISABLE_AFFINITY"
    )
    if affinity is not None and not affinity_disabled:
        if not isinstance(affinity, (list, tuple)):
            raise ValueError("runtime.cpu_affinity must be a list of CPU indices")
        configured_affinity = set_process_cpu_affinity(affinity)

    intraop_threads = runtime.get("intraop_threads")
    if intraop_threads is not None:
        if not isinstance(intraop_threads, int) or intraop_threads < 1:
            raise ValueError("runtime.intraop_threads must be positive")
        torch.set_num_threads(intraop_threads)

    interop_threads = runtime.get("interop_threads")
    if interop_threads is not None:
        if not isinstance(interop_threads, int) or interop_threads < 1:
            raise ValueError("runtime.interop_threads must be positive")
        if _configured_interop_threads is None:
            try:
                torch.set_num_interop_threads(interop_threads)
            except RuntimeError:
                current = torch.get_num_interop_threads()
                if current != interop_threads:
                    raise
            _configured_interop_threads = torch.get_num_interop_threads()
        elif _configured_interop_threads != interop_threads:
            raise RuntimeError(
                "PyTorch inter-op threads cannot be changed after parallel "
                "work has started in this process"
            )

    return {
        "cpu_affinity": configured_affinity,
        "cpu_affinity_disabled": affinity_disabled,
        "intraop_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
    }
