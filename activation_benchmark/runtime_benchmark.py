from __future__ import annotations

import argparse
import json
import time

import torch
from torch import nn

from .model import CIFAR10ResNet
from .runtime import configure_runtime


def run_runtime_benchmark(
    *,
    cpu_affinity: list[int] | None,
    intraop_threads: int,
    interop_threads: int,
    iterations: int,
    batch_size: int,
) -> dict[str, float | int | list[int] | None]:
    settings = configure_runtime(
        {
            "runtime": {
                "cpu_affinity": cpu_affinity,
                "intraop_threads": intraop_threads,
                "interop_threads": interop_threads,
            }
        }
    )
    torch.manual_seed(42)
    model = CIFAR10ResNet(
        activation="relu",
        channels=[16, 32, 64, 128],
        block_depths=[1, 1, 1, 1],
    )
    inputs = torch.randn(batch_size, 3, 32, 32)
    targets = torch.randint(0, 10, (batch_size,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(inputs), targets).backward()
        optimizer.step()

    started = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(inputs), targets).backward()
        optimizer.step()
    duration = time.perf_counter() - started
    result: dict[str, float | int | list[int] | None] = {
        **settings,
        "iterations": iterations,
        "batch_size": batch_size,
        "duration_seconds": duration,
        "samples_per_second": iterations * batch_size / duration,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure CPU affinity and PyTorch thread settings"
    )
    parser.add_argument(
        "--cpu-affinity",
        help="Comma-separated logical CPU indices, or omit for current affinity",
    )
    parser.add_argument("--intraop-threads", type=int, default=8)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    affinity = (
        [int(value) for value in args.cpu_affinity.split(",")]
        if args.cpu_affinity
        else None
    )
    run_runtime_benchmark(
        cpu_affinity=affinity,
        intraop_threads=args.intraop_threads,
        interop_threads=args.interop_threads,
        iterations=args.iterations,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
