from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ScalarStep:
    step: int
    weight: float
    prediction: float
    loss: float
    gradient: float


def forward(x: float, weight: float) -> float:
    # TODO: prediction = ?
    raise NotImplementedError


def half_squared_error(prediction: float, target: float) -> float:
    # TODO: return 0.5 * error squared.
    raise NotImplementedError


def loss_at_weight(weight: float, x: float, target: float) -> float:
    # TODO: compose forward and loss.
    raise NotImplementedError


def central_difference(
    function: Callable[[float], float], value: float, epsilon: float = 1e-5
) -> float:
    # TODO: evaluate function at value+epsilon and value-epsilon.
    raise NotImplementedError


def backward(x: float, weight: float, target: float) -> float:
    # TODO: derive d(loss)/d(weight) by the chain rule.
    raise NotImplementedError


def train_weight(
    x: float,
    target: float,
    *,
    initial_weight: float = 0.0,
    learning_rate: float = 0.05,
    steps: int = 20,
) -> tuple[float, list[ScalarStep]]:
    # TODO: forward -> loss -> backward -> update, repeated for `steps`.
    raise NotImplementedError
