from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch


_STEP_PATTERN = re.compile(r"step_(\d+)\.pt$")


class CheckpointManager:
    def __init__(self, directory: str | Path, keep_latest: int = 3) -> None:
        if keep_latest < 1:
            raise ValueError("keep_latest must be at least one")
        self.directory = Path(directory)
        self.keep_latest = keep_latest
        self.directory.mkdir(parents=True, exist_ok=True)

    def _step_files(self) -> list[tuple[int, Path]]:
        checkpoints: list[tuple[int, Path]] = []
        for path in self.directory.glob("step_*.pt"):
            match = _STEP_PATTERN.match(path.name)
            if match:
                checkpoints.append((int(match.group(1)), path))
        return sorted(checkpoints)

    def save(self, state: dict[str, Any], step: int) -> Path:
        if step < 0:
            raise ValueError("step cannot be negative")
        destination = self.directory / f"step_{step:08d}.pt"
        temporary = destination.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(destination)
        self.prune()
        return destination

    def prune(self) -> None:
        checkpoints = self._step_files()
        for _, path in checkpoints[: -self.keep_latest]:
            path.unlink()

    def latest_path(self) -> Path:
        checkpoints = self._step_files()
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {self.directory}")
        return checkpoints[-1][1]

    def load(
        self,
        path: str | Path | None = None,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        checkpoint_path = self.latest_path() if path is None else Path(path)
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
