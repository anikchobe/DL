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
    raise NotImplementedError


def mean_half_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    # TODO: average 0.5 * error^2 over the batch.
    raise NotImplementedError


def loss_and_gradients(
    x: np.ndarray, targets: np.ndarray, weight: float, bias: float
) -> tuple[float, float, float]:
    # TODO: return loss, dL/dweight, and dL/dbias.
    raise NotImplementedError


def train(
    x: np.ndarray,
    targets: np.ndarray,
    *,
    initial_weight: float = 0.0,
    initial_bias: float = 0.0,
    learning_rate: float = 0.1,
    steps: int = 300,
) -> tuple[float, float, list[RegressionStep]]:
    # TODO: repeat gradient descent updates.
    raise NotImplementedError
