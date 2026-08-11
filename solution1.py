from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from solution1_viz import show_training_report

# For record keeping only.
@dataclass(frozen=True)
class ScalarStep:
    step: int
    weight: float
    prediction: float
    loss: float
    gradient: float


def forward(x: float, weight: float) -> float:
    return x * weight


def half_squared_error(prediction: float, target: float) -> float:
    # TODO: return 0.5 * error squared.
    return 0.5 * (prediction - target)**2

def backward(x: float, weight: float, target: float) -> float:
    # TODO: derive d(loss)/d(weight) by the chain rule.
    #     loss = 0.5 * (weight*x - target)²
    #     let  e = weight*x - target       (the error)
    #     d(loss)/d(e)      = e           (0.5*e² differentiates to e — that's the 0.5 paying off)
    #     d(e)/d(weight)    = x           (target is a constant; weight*x differentiates to x)
    #     chain rule — multiply them:
    #     d(loss)/d(weight) = e * x = (prediction - target) * x
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
print("step | weight     | prediction | loss       | gradient")
for item in steps:
    print(f"{item.step:>4} | {item.weight:>10.6f} | {item.prediction:>10.6f} | {item.loss:>10.6f} | {item.gradient:>10.6f}")

# The same run as a picture.  All of the matplotlib code lives in
# solution1_viz.py; it only needs the four functions above.
figure_path = show_training_report(3, 12, forward, half_squared_error, backward, train_weight)
print(f"\nSaved the training graphs to {figure_path}")
