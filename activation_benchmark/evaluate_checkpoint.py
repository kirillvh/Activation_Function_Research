from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .data import build_data_loaders
from .model import build_model
from .trainer import evaluate, resolve_device, seed_everything


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    split: str = "test",
    weights: str = "model",
    requested_device: str | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = state["config"]
    experiment = config["experiment"]
    seed_everything(
        experiment["seed"],
        experiment.get("deterministic", True),
    )
    device = resolve_device(
        requested_device or config["training"].get("device", "auto")
    )
    loaders = build_data_loaders(config)
    model = build_model(config).to(device)

    state_key = "best_model" if weights == "best" else "model"
    model_state = state.get(state_key)
    if model_state is None:
        raise ValueError(
            f"Checkpoint does not contain {state_key!r} weights"
        )
    model.load_state_dict(model_state)
    loader = getattr(loaders, split)
    metrics = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    return {
        "checkpoint": str(checkpoint_path),
        "epoch": state.get("epoch", -1) + 1,
        "global_step": state.get("global_step"),
        "split": split,
        "weights": weights,
        **metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate model weights stored in a training checkpoint"
    )
    parser.add_argument("checkpoint", help="Path to a step_*.pt checkpoint")
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
    )
    parser.add_argument(
        "--weights",
        choices=("model", "best"),
        default="model",
        help="Evaluate current weights or embedded validation-best weights",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override such as cpu or cuda",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_checkpoint(
        args.checkpoint,
        split=args.split,
        weights=args.weights,
        requested_device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
