from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BinaryParameters:
    weights: np.ndarray
    bias: float


def sigmoid(logits: np.ndarray) -> np.ndarray:
    # TODO: 1/(1+exp(-z)) equals exp(-log(1+exp(-z))).  Use np.logaddexp for the
    # inner term so no exponential is ever formed.  One line, no overflow.
    raise NotImplementedError


def binary_cross_entropy_from_logits(
    logits: np.ndarray, labels: np.ndarray
) -> float:
    # TODO: mean of np.logaddexp(0, logits) - labels * logits.
    # Never build probabilities first: log(0.0) is -inf.
    raise NotImplementedError


def forward(
    x: np.ndarray, parameters: BinaryParameters
) -> tuple[np.ndarray, np.ndarray]:
    # TODO: return logits and probabilities.
    raise NotImplementedError


def loss_and_gradients(
    x: np.ndarray, labels: np.ndarray, parameters: BinaryParameters
) -> tuple[float, np.ndarray, float]:
    # TODO: dL/dlogits simplifies to (probabilities-labels)/batch_size.
    raise NotImplementedError


def predict(x: np.ndarray, parameters: BinaryParameters) -> np.ndarray:
    raise NotImplementedError


def make_binary_blobs(
    examples_per_class: int = 100, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    # TODO: generate two deterministic 2-D Gaussian clusters and shuffle them.
    raise NotImplementedError


def train(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    learning_rate: float = 0.2,
    steps: int = 250,
) -> BinaryParameters:
    # TODO: initialize zeros and run full-batch gradient descent.
    raise NotImplementedError
