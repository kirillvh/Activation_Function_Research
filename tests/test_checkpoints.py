import torch

from activation_benchmark.checkpoints import CheckpointManager


def test_checkpoint_retention_and_latest(tmp_path):
    manager = CheckpointManager(tmp_path, keep_latest=3)
    for step in range(1, 6):
        manager.save({"step": step, "tensor": torch.tensor(step)}, step)

    assert [path.name for path in tmp_path.glob("*.pt")] == [
        "step_00000003.pt",
        "step_00000004.pt",
        "step_00000005.pt",
    ]
    state = manager.load()
    assert state["step"] == 5
    assert state["tensor"].item() == 5

