"""Stage 10 starter: manual backpropagation and gradient checking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solution4 import softmax_cross_entropy_with_gradient
from solution9 import MLPParameters, forward, initialize_mlp, predict


@dataclass
class MLPGradients:
    weights1: np.ndarray
    bias1: np.ndarray
    weights2: np.ndarray
    bias2: np.ndarray


@dataclass(frozen=True)
class MLPStep:
    step: int
    loss: float
    accuracy: float


def loss_and_gradients(
    x: np.ndarray, labels: np.ndarray, parameters: MLPParameters
) -> tuple[float, MLPGradients]:
    # TODO, in reverse order:
    # 1. dlogits from softmax cross-entropy
    # 2. dW2, db2, dH
    # 3. dZ1 = dH * (Z1 > 0)
    # 4. dW1, db1
    raise NotImplementedError


def accuracy(
    x: np.ndarray, labels: np.ndarray, parameters: MLPParameters
) -> float:
    raise NotImplementedError


def train_full_batch(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    hidden_units: int = 8,
    learning_rate: float = 0.1,
    steps: int = 2_000,
    seed: int = 0,
) -> tuple[MLPParameters, list[MLPStep]]:
    raise NotImplementedError


def make_noisy_xor(
    examples_per_corner: int = 50, *, noise: float = 0.12, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


def gradient_check(
    x: np.ndarray,
    labels: np.ndarray,
    parameters: MLPParameters,
    *,
    epsilon: float = 1e-5,
    checks_per_array: int = 8,
    seed: int = 0,
) -> dict[str, float]:
    # TODO: sample entries from all four parameter arrays.
    raise NotImplementedError
