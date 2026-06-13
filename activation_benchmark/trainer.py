from __future__ import annotations

import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .activations import PEUAF, SineTriangle, trainable_activation_parameters
from .checkpoints import CheckpointManager
from .config import flatten_dict, save_yaml, validate_training_config
from .data import build_data_loaders
from .model import build_model, count_parameters
from .runtime import configure_runtime


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def build_optimizer(config: dict[str, Any], model: nn.Module) -> Optimizer:
    training = config["training"]
    name = training.get("optimizer", "adamw").lower()
    base_learning_rate = training["learning_rate"]
    base_weight_decay = training.get("weight_decay", 0.0)
    activation_parameters = trainable_activation_parameters(model)
    activation_parameter_ids = {
        id(parameter) for parameter in activation_parameters
    }
    base_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in activation_parameter_ids
    ]
    parameter_groups: list[dict[str, Any]] = [
        {
            "params": base_parameters,
            "lr": base_learning_rate,
            "weight_decay": base_weight_decay,
        }
    ]
    if activation_parameters:
        parameter_groups.append(
            {
                "params": activation_parameters,
                "lr": base_learning_rate
                * training.get(
                    "activation_learning_rate_multiplier",
                    training.get("peuaf_learning_rate_multiplier", 1.0),
                ),
                "weight_decay": training.get(
                    "activation_weight_decay",
                    training.get("peuaf_weight_decay", 0.0),
                ),
                "name": "activation_parameters",
            }
        )
    if name == "adamw":
        return torch.optim.AdamW(parameter_groups)
    if name == "adam":
        return torch.optim.Adam(parameter_groups)
    if name == "nadam":
        return torch.optim.NAdam(
            parameter_groups,
            momentum_decay=training.get("momentum_decay", 0.004),
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameter_groups,
            momentum=training.get("momentum", 0.9),
        )
    raise ValueError(
        "training.optimizer must be one of: adamw, adam, nadam, sgd"
    )


def build_scheduler(
    config: dict[str, Any],
    optimizer: Optimizer,
) -> LRScheduler | ReduceLROnPlateau | None:
    training = config["training"]
    name = training.get("scheduler", "none").lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training["epochs"],
            eta_min=training.get("scheduler_min_learning_rate", 0.0),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training.get("scheduler_step_size", 3),
            gamma=training.get("scheduler_gamma", 0.5),
        )
    if name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=training.get("scheduler_mode", "max"),
            factor=training.get("scheduler_factor", 0.2),
            patience=training.get("scheduler_patience", 5),
            threshold=training.get("scheduler_threshold", 1e-4),
            min_lr=training.get("scheduler_min_learning_rate", 0.0),
        )
    raise ValueError(
        "training.scheduler must be one of: none, cosine, step, plateau"
    )


def constrain_model_parameters(model: nn.Module) -> None:
    for module in model.modules():
        constrain = getattr(module, "constrain_parameters", None)
        if constrain is not None:
            constrain()


def peuaf_statistics(model: nn.Module) -> dict[str, float]:
    modules = [module for module in model.modules() if isinstance(module, PEUAF)]
    if not modules:
        return {}
    frequencies = torch.cat(
        [module.w.detach().reshape(-1).cpu() for module in modules]
    )
    gradient_tensors = [
        module.w.grad.detach().abs().reshape(-1).cpu()
        for module in modules
        if module.w.grad is not None
    ]
    gradients = (
        torch.cat(gradient_tensors)
        if gradient_tensors
        else torch.empty(0)
    )
    return {
        "peuaf_frequency_min": float(frequencies.min()),
        "peuaf_frequency_mean": float(frequencies.mean()),
        "peuaf_frequency_max": float(frequencies.max()),
        "peuaf_frequency_gradient_abs_mean": (
            float(gradients.mean()) if gradients.numel() else 0.0
        ),
    }


def sine_triangle_statistics(model: nn.Module) -> dict[str, float]:
    modules = [
        module for module in model.modules() if isinstance(module, SineTriangle)
    ]
    if not modules:
        return {}
    frequencies = torch.cat(
        [module.frequency.detach().reshape(-1).cpu() for module in modules]
    )
    blends = torch.cat(
        [module.blend.detach().reshape(-1).cpu() for module in modules]
    )
    scales = torch.cat(
        [
            torch.as_tensor(module.periodic_scale)
            .detach()
            .reshape(-1)
            .cpu()
            for module in modules
        ]
    )
    gradient_tensors = [
        parameter.grad.detach().abs().reshape(-1).cpu()
        for module in modules
        for parameter in (
            module.frequency_logit,
            module.blend_logit,
            module.periodic_scale_logit,
        )
        if parameter is not None and parameter.grad is not None
    ]
    gradients = (
        torch.cat(gradient_tensors)
        if gradient_tensors
        else torch.empty(0)
    )
    return {
        "sine_triangle_frequency_min": float(frequencies.min()),
        "sine_triangle_frequency_mean": float(frequencies.mean()),
        "sine_triangle_frequency_max": float(frequencies.max()),
        "sine_triangle_blend_min": float(blends.min()),
        "sine_triangle_blend_mean": float(blends.mean()),
        "sine_triangle_blend_max": float(blends.max()),
        "sine_triangle_periodic_scale_min": float(scales.min()),
        "sine_triangle_periodic_scale_mean": float(scales.mean()),
        "sine_triangle_periodic_scale_max": float(scales.max()),
        "sine_triangle_parameter_gradient_abs_mean": (
            float(gradients.mean()) if gradients.numel() else 0.0
        ),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += batch_size
    if total_samples == 0:
        raise RuntimeError("Cannot evaluate an empty dataset")
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _cpu_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def load_initial_model_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    weights: str,
    map_location: torch.device | str,
) -> None:
    if weights not in {"model", "best"}:
        raise ValueError(
            "training.initial_checkpoint_weights must be 'model' or 'best'"
        )
    state = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    key = "best_model" if weights == "best" else "model"
    model_state = state.get(key)
    if model_state is None:
        raise ValueError(f"Checkpoint does not contain {key!r} weights")
    model.load_state_dict(model_state)


def _checkpoint_state(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | ReduceLROnPlateau | None,
    config: dict[str, Any],
    epoch: int,
    batch_in_epoch: int,
    epoch_complete: bool,
    global_step: int,
    best_validation_accuracy: float,
    best_validation_epoch: int | None,
    best_model_state: dict[str, torch.Tensor] | None,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "epoch_complete": epoch_complete,
        "global_step": global_step,
        "best_validation_accuracy": best_validation_accuracy,
        "best_validation_epoch": best_validation_epoch,
        "best_model": best_model_state,
        "history": history,
        "rng_state": _rng_state(),
    }


def _write_history(history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    fieldnames: list[str] = []
    for record in history:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _run_name(config: dict[str, Any]) -> str:
    configured = config["experiment"].get("name")
    if configured:
        return str(configured)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = config["data"].get("dataset", "mnist")
    architecture = config["model"].get("architecture", "standard")
    activation = config["model"]["activation"]
    return f"{timestamp}_{dataset}_{architecture}_{activation}"


def _tensorboard_hparams(config: dict[str, Any]) -> str:
    lines = ["| Parameter | Value |", "| --- | --- |"]
    for key, value in flatten_dict(config).items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def train_experiment(config: dict[str, Any]) -> dict[str, Any]:
    validate_training_config(config)
    runtime_settings = configure_runtime(config)
    experiment = config["experiment"]
    training = config["training"]
    checkpoint_config = config["checkpoint"]
    seed = experiment["seed"]
    seed_everything(seed, experiment.get("deterministic", True))
    device = resolve_device(training.get("device", "auto"))

    run_dir = Path(experiment["output_dir"]) / _run_name(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, run_dir / "config.yaml")

    loaders = build_data_loaders(config)
    model = build_model(config).to(device)
    initial_checkpoint = training.get("initial_checkpoint")
    if initial_checkpoint:
        initial_weights = training.get(
            "initial_checkpoint_weights",
            "best",
        )
        load_initial_model_weights(
            model,
            initial_checkpoint,
            weights=initial_weights,
            map_location=device,
        )
        print(
            f"Initialized {initial_weights} weights from "
            f"{initial_checkpoint}"
        )
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    manager = CheckpointManager(
        run_dir / "checkpoints",
        keep_latest=checkpoint_config["keep_latest"],
    )

    writer: SummaryWriter | None = None
    if config["tensorboard"].get("enabled", True):
        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
        writer.add_text("config", _tensorboard_hparams(config), 0)

    start_epoch = 0
    resume_batch = -1
    global_step = 0
    best_validation_accuracy = 0.0
    best_validation_epoch: int | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    initial_validation_accuracy: float | None = None
    history: list[dict[str, Any]] = []
    resume_rng_state: dict[str, Any] | None = None
    resume = checkpoint_config.get("resume")
    if initial_checkpoint:
        initial_validation = evaluate(
            model,
            loaders.validation,
            criterion,
            device,
        )
        initial_validation_accuracy = initial_validation["accuracy"]
        best_validation_accuracy = initial_validation_accuracy
        best_validation_epoch = 0
        best_model_state = _cpu_model_state(model)
        print(
            "warm-start validation: "
            f"loss {initial_validation['loss']:.4f}, "
            f"acc {initial_validation_accuracy:.2%}"
        )
        if writer is not None:
            writer.add_scalar(
                "validation/loss",
                initial_validation["loss"],
                0,
            )
            writer.add_scalar(
                "validation/accuracy",
                initial_validation_accuracy,
                0,
            )
    if resume:
        if resume == "latest":
            checkpoint_path = manager.latest_path()
        else:
            checkpoint_path = Path(resume)
        state = manager.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        global_step = state["global_step"]
        best_validation_accuracy = state.get("best_validation_accuracy", 0.0)
        best_validation_epoch = state.get("best_validation_epoch")
        best_model_state = state.get("best_model")
        history = state.get("history", [])
        if state.get("epoch_complete", False):
            start_epoch = state["epoch"] + 1
        else:
            start_epoch = state["epoch"]
            resume_batch = state.get("batch_in_epoch", -1)
            resume_rng_state = state.get("rng_state")
        print(f"Resumed {checkpoint_path} at step {global_step}")

    parameter_count = count_parameters(model)
    print(
        f"Run: {run_dir} | device: {device} | "
        f"parameters: {parameter_count:,} | "
        f"threads: {runtime_settings['intraop_threads']} intra-op, "
        f"{runtime_settings['interop_threads']} inter-op | "
        f"affinity: {runtime_settings['cpu_affinity'] or 'unchanged'}"
    )
    start_time = time.perf_counter()
    last_epoch = max(start_epoch - 1, 0)
    last_batch = -1

    try:
        for epoch in range(start_epoch, training["epochs"]):
            last_epoch = epoch
            model.train()
            epoch_learning_rate = optimizer.param_groups[0]["lr"]
            if loaders.train.generator is not None:
                loaders.train.generator.manual_seed(seed + epoch)

            iterator = iter(loaders.train)
            batches_to_skip = resume_batch + 1 if epoch == start_epoch else 0
            for _ in range(batches_to_skip):
                try:
                    next(iterator)
                except StopIteration:
                    break
            if epoch == start_epoch and resume_rng_state is not None:
                _restore_rng_state(resume_rng_state)

            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            peuaf_gradient_sum = 0.0
            peuaf_gradient_batches = 0
            sine_triangle_gradient_sum = 0.0
            sine_triangle_gradient_batches = 0
            epoch_started = time.perf_counter()
            for batch_index, (inputs, targets) in enumerate(
                iterator,
                start=batches_to_skip,
            ):
                last_batch = batch_index
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                batch_peuaf_stats = peuaf_statistics(model)
                if batch_peuaf_stats:
                    peuaf_gradient_sum += batch_peuaf_stats[
                        "peuaf_frequency_gradient_abs_mean"
                    ]
                    peuaf_gradient_batches += 1
                batch_sine_triangle_stats = sine_triangle_statistics(model)
                if batch_sine_triangle_stats:
                    sine_triangle_gradient_sum += batch_sine_triangle_stats[
                        "sine_triangle_parameter_gradient_abs_mean"
                    ]
                    sine_triangle_gradient_batches += 1
                clip_norm = training.get("gradient_clip_norm")
                if clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
                constrain_model_parameters(model)

                batch_size = targets.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_correct += (logits.argmax(dim=1) == targets).sum().item()
                epoch_samples += batch_size
                global_step += 1

                if writer is not None:
                    writer.add_scalar("train/batch_loss", loss.item(), global_step)
                    writer.add_scalar(
                        "train/learning_rate",
                        optimizer.param_groups[0]["lr"],
                        global_step,
                    )
                if global_step % training.get("log_every_steps", 50) == 0:
                    print(
                        f"epoch {epoch + 1}/{training['epochs']} "
                        f"step {global_step} loss {loss.item():.4f}"
                    )
                if (
                    checkpoint_config.get("enabled", True)
                    and global_step % checkpoint_config["save_every_steps"] == 0
                ):
                    manager.save(
                        _checkpoint_state(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            config=config,
                            epoch=epoch,
                            batch_in_epoch=batch_index,
                            epoch_complete=False,
                            global_step=global_step,
                            best_validation_accuracy=best_validation_accuracy,
                            best_validation_epoch=best_validation_epoch,
                            best_model_state=best_model_state,
                            history=history,
                        ),
                        global_step,
                    )

            if epoch_samples == 0:
                if batches_to_skip >= len(loaders.train):
                    resume_batch = -1
                    continue
                raise RuntimeError("Training dataset is empty")

            train_metrics = {
                "loss": epoch_loss / epoch_samples,
                "accuracy": epoch_correct / epoch_samples,
            }
            validation_metrics = evaluate(
                model,
                loaders.validation,
                criterion,
                device,
            )
            if (
                best_model_state is None
                or validation_metrics["accuracy"] > best_validation_accuracy
            ):
                best_validation_accuracy = validation_metrics["accuracy"]
                best_validation_epoch = epoch + 1
                best_model_state = _cpu_model_state(model)
            if isinstance(scheduler, ReduceLROnPlateau):
                metric = (
                    validation_metrics["accuracy"]
                    if scheduler.mode == "max"
                    else validation_metrics["loss"]
                )
                scheduler.step(metric)
            elif scheduler is not None:
                scheduler.step()
            next_learning_rate = optimizer.param_groups[0]["lr"]

            epoch_duration = time.perf_counter() - epoch_started
            record = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "learning_rate": epoch_learning_rate,
                "next_learning_rate": next_learning_rate,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "epoch_duration_seconds": epoch_duration,
            }
            frequency_stats = peuaf_statistics(model)
            if frequency_stats:
                frequency_stats["peuaf_frequency_gradient_abs_mean"] = (
                    peuaf_gradient_sum / peuaf_gradient_batches
                    if peuaf_gradient_batches
                    else 0.0
                )
                record.update(frequency_stats)
            sine_triangle_stats = sine_triangle_statistics(model)
            if sine_triangle_stats:
                sine_triangle_stats[
                    "sine_triangle_parameter_gradient_abs_mean"
                ] = (
                    sine_triangle_gradient_sum / sine_triangle_gradient_batches
                    if sine_triangle_gradient_batches
                    else 0.0
                )
                record.update(sine_triangle_stats)
            history.append(record)
            _write_history(history, run_dir / "history.csv")

            if writer is not None:
                writer.add_scalar("train/epoch_loss", train_metrics["loss"], epoch + 1)
                writer.add_scalar(
                    "train/epoch_accuracy",
                    train_metrics["accuracy"],
                    epoch + 1,
                )
                writer.add_scalar(
                    "train/epoch_learning_rate",
                    next_learning_rate,
                    epoch + 1,
                )
                for key, value in frequency_stats.items():
                    writer.add_scalar(f"peuaf/{key}", value, epoch + 1)
                for key, value in sine_triangle_stats.items():
                    writer.add_scalar(
                        f"sine_triangle/{key}",
                        value,
                        epoch + 1,
                    )
                writer.add_scalar(
                    "validation/loss",
                    validation_metrics["loss"],
                    epoch + 1,
                )
                writer.add_scalar(
                    "validation/accuracy",
                    validation_metrics["accuracy"],
                    epoch + 1,
                )
            print(
                f"epoch {epoch + 1}: lr {epoch_learning_rate:.6g} -> "
                f"{next_learning_rate:.6g}, "
                f"train loss {train_metrics['loss']:.4f}, "
                f"train acc {train_metrics['accuracy']:.2%}, "
                f"val loss {validation_metrics['loss']:.4f}, "
                f"val acc {validation_metrics['accuracy']:.2%}"
            )

            if checkpoint_config.get("enabled", True):
                manager.save(
                    _checkpoint_state(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        epoch=epoch,
                        batch_in_epoch=last_batch,
                        epoch_complete=True,
                        global_step=global_step,
                        best_validation_accuracy=best_validation_accuracy,
                        best_validation_epoch=best_validation_epoch,
                        best_model_state=best_model_state,
                        history=history,
                    ),
                    global_step,
                )
            resume_batch = -1

        final_test_metrics = evaluate(model, loaders.test, criterion, device)
        test_model_selection = "final"
        if (
            training.get("evaluate_best_validation", True)
            and best_model_state is not None
        ):
            model.load_state_dict(best_model_state)
            test_model_selection = "best_validation"
        test_metrics = evaluate(model, loaders.test, criterion, device)
        duration = time.perf_counter() - start_time
        final_validation = (
            history[-1]["validation_accuracy"] if history else float("nan")
        )
        result = {
            "run_dir": str(run_dir),
            "dataset": config["data"].get("dataset", "mnist"),
            "architecture": config["model"].get("architecture", "standard"),
            "activation": config["model"]["activation"],
            "seed": seed,
            "device": str(device),
            "runtime": runtime_settings,
            "parameter_count": parameter_count,
            "epochs_completed": len(history),
            "global_step": global_step,
            "best_validation_accuracy": best_validation_accuracy,
            "best_validation_epoch": best_validation_epoch,
            "initial_validation_accuracy": initial_validation_accuracy,
            "final_validation_accuracy": final_validation,
            "test_model_selection": test_model_selection,
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "final_test_loss": final_test_metrics["loss"],
            "final_test_accuracy": final_test_metrics["accuracy"],
            "duration_seconds": duration,
            "history": history,
        }
        with (run_dir / "result.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        if writer is not None:
            writer.add_scalar("test/loss", test_metrics["loss"], global_step)
            writer.add_scalar("test/accuracy", test_metrics["accuracy"], global_step)
            writer.add_scalar(
                "test/final_accuracy",
                final_test_metrics["accuracy"],
                global_step,
            )
            writer.flush()
        print(
            f"test ({test_model_selection}) loss {test_metrics['loss']:.4f}, "
            f"acc {test_metrics['accuracy']:.2%}, "
            f"final-model acc {final_test_metrics['accuracy']:.2%}, "
            f"duration {duration:.1f}s"
        )
        return result
    finally:
        if writer is not None:
            writer.close()
