from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionStep:
    step: int
    weight: float
    bias: float
    loss: float


def forward(x: np.ndarray, weight: float, bias: float) -> np.ndarray:
    # TODO: vectorized predictions for every example.
    return x * weight + bias


def mean_half_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    # TODO: average 0.5 * error^2 over the batch.
    return 0.5 * np.mean((predictions - targets)**2)


def loss_and_gradients(
    x: np.ndarray, targets: np.ndarray, weight: float, bias: float
) -> tuple[float, float, float]:
    # TODO: return loss, dL/dweight, and dL/dbias.
    predictions = forward(x, weight, bias)
    errors = predictions -  targets
    loss = mean_half_squared_error(predictions, targets)
    gradient_weight = np.mean(errors * x)
    gradient_bias = np.mean(errors)
    return loss, gradient_weight, gradient_bias

#at learning_rate = 0.9, why does weight oscillate while bias converge?

def train(
    x: np.ndarray,
    targets: np.ndarray,
    *,
    initial_weight: float = 0.0,
    initial_bias: float = 0.0,
    learning_rate: float = 0.6,
    steps: int = 20,
) -> tuple[float, float, list[RegressionStep]]:
    # TODO: repeat gradient descent updates.
    weight = initial_weight
    bias = initial_weight
    history: list[RegressionStep] = []
    for step in range(steps):
        loss, gradient_weight, gradient_bias = loss_and_gradients(x, targets, weight, bias)
        history.append(RegressionStep(step, weight, bias, loss))
        weight -= learning_rate * gradient_weight
        bias -= learning_rate * gradient_bias
    return weight, bias, history

x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
y = 3.0 * x + 2.0
learned_w, learned_b, history = train(x, y)
print(f'w = {learned_w}, b = {learned_b}')
print("step | weight     | bias | loss       ")
for item in history:
    print(f"{item.step:>4} | {item.weight:>10.6f} | {item.bias:>10.6f} | {item.loss:>10.6f}")
