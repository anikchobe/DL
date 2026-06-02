"""Stage 5 starter: multiclass linear classifier and gradient checking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solution4 import softmax_cross_entropy_with_gradient


@dataclass
class LinearParameters:
    weights: np.ndarray  # (features, classes)
    bias: np.ndarray  # (classes,)


@dataclass
class LinearGradients:
    weights: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class LinearStep:
    step: int
    loss: float
    accuracy: float


def initialize(
    feature_count: int, class_count: int, *, seed: int = 0
) -> LinearParameters:
    # TODO: small random weights and a zero bias.
    raise NotImplementedError


def logits(x: np.ndarray, parameters: LinearParameters) -> np.ndarray:
    # TODO: x @ weights + bias, with shape checks first.
    raise NotImplementedError


def loss_and_gradients(
    x: np.ndarray, labels: np.ndarray, parameters: LinearParameters
) -> tuple[float, LinearGradients]:
    # TODO: backprop through logits = X @ W + b.
    raise NotImplementedError


def predict(x: np.ndarray, parameters: LinearParameters) -> np.ndarray:
    raise NotImplementedError


def accuracy(
    x: np.ndarray, labels: np.ndarray, parameters: LinearParameters
) -> float:
    raise NotImplementedError


def make_three_class_blobs(
    examples_per_class: int = 80, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


def gradient_check(
    x: np.ndarray,
    labels: np.ndarray,
    parameters: LinearParameters,
    *,
    epsilon: float = 1e-5,
) -> dict[str, float]:
    # TODO: perturb selected entries in W and b and compare numerical gradients.
    raise NotImplementedError


def train(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    learning_rate: float = 0.2,
    steps: int = 300,
    seed: int = 0,
) -> tuple[LinearParameters, list[LinearStep]]:
    raise NotImplementedError
