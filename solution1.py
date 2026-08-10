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
    return x * weight


def half_squared_error(prediction: float, target: float) -> float:
    # TODO: return 0.5 * error squared.
    return 0.5 * (prediction - target)**2


def loss_at_weight(weight: float, x: float, target: float) -> float:
    # TODO: compose forward and loss.
    return half_squared_error(forward(x, weight), target)


def central_difference(
    function: Callable[[float], float], value: float, epsilon: float = 1e-5
) -> float:
    # TODO: evaluate function at value+epsilon and value-epsilon.
    raise NotImplementedError


def backward(x: float, weight: float, target: float) -> float:
    # TODO: derive d(loss)/d(weight) by the chain rule.
    return (forward(x, weight) - target) * x


def train_weight(
    x: float,
    target: float,
    *,
    initial_weight: float = 0.0,
    learning_rate: float = 0.05,
    steps: int = 20,
) -> tuple[float, list[ScalarStep]]:
    # TODO: forward -> loss -> backward -> update, repeated for `steps`.
    weight = initial_weight
    history: list[ScalarStep] = []
    for step in range(steps):
        prediction = forward(x, weight)
        loss = half_squared_error(prediction, target)
        gradient = backward(x, weight, target)
        history.append(ScalarStep(step, weight, prediction, loss, gradient))
        weight -= learning_rate * gradient
    return (weight, history)

learnedWeight, steps = train_weight(3, 12)
print(learnedWeight)
print("steps, weight, prediction, loss, gradient")
for item in steps:
    print(f"{item.step}, {item.weight}, {item.prediction}, {item.loss}, {item.gradient}")