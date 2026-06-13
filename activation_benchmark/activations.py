from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class EUAF(nn.Module):
    """Paper-defined fixed-frequency Elementary Universal Activation Function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positive = torch.abs(
            x - 2.0 * torch.floor((x + 1.0) / 2.0)
        )
        negative = x / (1.0 + torch.abs(x))
        return torch.where(x >= 0, positive, negative)


class PEUAF(nn.Module):
    """Paper-defined PEUAF with a projected trainable frequency in [0, 1]."""

    def __init__(
        self,
        initial_w: float = 0.5,
        num_parameters: int = 1,
    ) -> None:
        super().__init__()
        if not 0.0 <= initial_w <= 1.0:
            raise ValueError("PEUAF initial_w must be between zero and one")
        if not isinstance(num_parameters, int) or num_parameters < 1:
            raise ValueError("PEUAF num_parameters must be a positive integer")
        if num_parameters == 1:
            initial = torch.tensor(float(initial_w))
        else:
            initial = torch.full((num_parameters,), float(initial_w))
        self.w = nn.Parameter(initial)

    def _broadcast_frequency(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.clamp(self.w, 0.0, 1.0)
        if w.numel() == 1:
            return w
        if x.ndim < 2 or x.shape[1] != w.numel():
            raise ValueError(
                "Channel-wise PEUAF expects inputs shaped [N, C, ...] "
                f"with C={w.numel()}"
            )
        return w.view(1, -1, *([1] * (x.ndim - 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._broadcast_frequency(x)
        wx = w * x
        positive = torch.abs(
            wx - 2.0 * torch.floor((wx + 1.0) / 2.0)
        )
        negative = x / (1.0 + torch.abs(x))
        return torch.where(x >= 0, positive, negative)

    @torch.no_grad()
    def constrain_parameters(self) -> None:
        self.w.clamp_(0.0, 1.0)


class Sine(nn.Module):
    """A fixed-frequency sine activation."""

    def __init__(self, frequency: float = 1.0) -> None:
        super().__init__()
        self.frequency = float(frequency)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.frequency * x)


def triangle_wave(x: torch.Tensor) -> torch.Tensor:
    """Stable [-1, 1] triangle wave equivalent to 2*asin(sin(x))/pi."""
    phase = (x - math.pi / 2.0) / (2.0 * math.pi)
    return 1.0 - 4.0 * torch.abs(phase - torch.round(phase))


def _inverse_sigmoid(value: float) -> float:
    return math.log(value / (1.0 - value))


class SineTriangle(nn.Module):
    """Learnable blend of sine and a numerically stable triangle wave."""

    def __init__(
        self,
        initial_w: float = 1.0,
        initial_blend: float = 0.5,
        minimum_w: float = 0.05,
        maximum_w: float = 4.0,
        phase_divisor: float = 1.0,
        residual_scale: float = 0.0,
        periodic_scale: float = 1.0,
        base_activation: str = "none",
        trainable_periodic_scale: bool = False,
        maximum_periodic_scale: float = 1.0,
        num_parameters: int = 1,
    ) -> None:
        super().__init__()
        if not 0.0 < minimum_w < maximum_w:
            raise ValueError("minimum_w and maximum_w must define a positive range")
        if not minimum_w < initial_w < maximum_w:
            raise ValueError("initial_w must lie strictly inside the frequency range")
        if not 0.0 < initial_blend < 1.0:
            raise ValueError("initial_blend must lie strictly between zero and one")
        if phase_divisor <= 0.0:
            raise ValueError("phase_divisor must be positive")
        if base_activation not in {"none", "silu", "gelu"}:
            raise ValueError("base_activation must be one of: none, silu, gelu")
        if maximum_periodic_scale <= 0.0:
            raise ValueError("maximum_periodic_scale must be positive")
        if trainable_periodic_scale and not (
            0.0 < periodic_scale < maximum_periodic_scale
        ):
            raise ValueError(
                "A trainable periodic_scale must lie strictly between zero "
                "and maximum_periodic_scale"
            )
        if not isinstance(num_parameters, int) or num_parameters < 1:
            raise ValueError("num_parameters must be a positive integer")

        normalized_w = (initial_w - minimum_w) / (maximum_w - minimum_w)
        frequency_logit = _inverse_sigmoid(normalized_w)
        blend_logit = _inverse_sigmoid(initial_blend)
        shape = () if num_parameters == 1 else (num_parameters,)
        self.frequency_logit = nn.Parameter(
            torch.full(shape, frequency_logit)
        )
        self.blend_logit = nn.Parameter(torch.full(shape, blend_logit))
        self.minimum_w = float(minimum_w)
        self.maximum_w = float(maximum_w)
        self.phase_divisor = float(phase_divisor)
        self.residual_scale = float(residual_scale)
        self.base_activation = base_activation
        self.maximum_periodic_scale = float(maximum_periodic_scale)
        self.fixed_periodic_scale = float(periodic_scale)
        if trainable_periodic_scale:
            normalized_scale = periodic_scale / maximum_periodic_scale
            self.periodic_scale_logit = nn.Parameter(
                torch.full(shape, _inverse_sigmoid(normalized_scale))
            )
        else:
            self.register_parameter("periodic_scale_logit", None)

    @property
    def frequency(self) -> torch.Tensor:
        scale = self.maximum_w - self.minimum_w
        return self.minimum_w + scale * torch.sigmoid(self.frequency_logit)

    @property
    def blend(self) -> torch.Tensor:
        return torch.sigmoid(self.blend_logit)

    @property
    def periodic_scale(self) -> torch.Tensor | float:
        if self.periodic_scale_logit is None:
            return self.fixed_periodic_scale
        return self.maximum_periodic_scale * torch.sigmoid(
            self.periodic_scale_logit
        )

    def _broadcast(self, parameter: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if parameter.numel() == 1:
            return parameter
        if x.ndim < 2 or x.shape[1] != parameter.numel():
            raise ValueError(
                "Channel-wise SineTriangle expects inputs shaped [N, C, ...] "
                f"with C={parameter.numel()}"
            )
        return parameter.view(1, -1, *([1] * (x.ndim - 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frequency = self._broadcast(self.frequency, x)
        blend = self._broadcast(self.blend, x)
        periodic_scale = self.periodic_scale
        if isinstance(periodic_scale, torch.Tensor):
            periodic_scale = self._broadcast(periodic_scale, x)
        phase = frequency * x / self.phase_divisor
        periodic = (
            blend * triangle_wave(phase)
            + (1.0 - blend) * torch.sin(phase)
        )
        if self.base_activation == "silu":
            base = F.silu(x)
        elif self.base_activation == "gelu":
            base = F.gelu(x)
        else:
            base = self.residual_scale * x
        return base + periodic_scale * periodic


def _literal_sine_triangle(**kwargs: Any) -> SineTriangle:
    kwargs.setdefault("phase_divisor", math.pi)
    return SineTriangle(**kwargs)


def _residual_sine_triangle(**kwargs: Any) -> SineTriangle:
    kwargs.setdefault("residual_scale", 1.0)
    kwargs.setdefault("periodic_scale", 0.25)
    return SineTriangle(**kwargs)


def _silu_sine_triangle(**kwargs: Any) -> SineTriangle:
    kwargs.setdefault("base_activation", "silu")
    kwargs.setdefault("periodic_scale", 0.1)
    kwargs.setdefault("trainable_periodic_scale", True)
    return SineTriangle(**kwargs)


def _gelu_sine_triangle(**kwargs: Any) -> SineTriangle:
    kwargs.setdefault("base_activation", "gelu")
    kwargs.setdefault("periodic_scale", 0.1)
    kwargs.setdefault("trainable_periodic_scale", True)
    return SineTriangle(**kwargs)


def _gelu_sine_triangle_deep(**kwargs: Any) -> SineTriangle:
    kwargs.setdefault("base_activation", "gelu")
    kwargs.setdefault("periodic_scale", 0.01)
    kwargs.setdefault("trainable_periodic_scale", True)
    return SineTriangle(**kwargs)


_ACTIVATIONS: dict[str, Callable[..., nn.Module]] = {
    "euaf": EUAF,
    "peuaf": PEUAF,
    "sine": Sine,
    "sine_triangle_literal": _literal_sine_triangle,
    "sine_triangle": SineTriangle,
    "sine_triangle_residual": _residual_sine_triangle,
    "silu_sine_triangle": _silu_sine_triangle,
    "gelu_sine_triangle": _gelu_sine_triangle,
    "gelu_sine_triangle_deep": _gelu_sine_triangle_deep,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "prelu": nn.PReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
    "hardswish": nn.Hardswish,
}


def activation_names() -> tuple[str, ...]:
    return tuple(_ACTIVATIONS)


def make_activation(name: str, **kwargs: Any) -> nn.Module:
    normalized = name.lower().replace("-", "_")
    try:
        constructor = _ACTIVATIONS[normalized]
    except KeyError as error:
        choices = ", ".join(activation_names())
        raise ValueError(
            f"Unknown activation {name!r}. Available activations: {choices}"
        ) from error
    return constructor(**kwargs)


def trainable_activation_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, (PEUAF, SineTriangle)):
            parameters.extend(module.parameters(recurse=False))
    return parameters
