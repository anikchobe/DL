"""Stage 8 starter: mini-batch softmax regression on MNIST."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from solution4 import softmax_cross_entropy_with_gradient
from solution6 import prepare_mnist
from solution7 import PreparedMNIST, prepare_splits


@dataclass
class MNISTLinearModel:
    weights: np.ndarray  # (784, 10)
    bias: np.ndarray  # (10,)


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float


def initialize_model(
    feature_count: int = 784, class_count: int = 10
) -> MNISTLinearModel:
    # TODO: zero weights and bias are fine for a linear classifier.
    raise NotImplementedError


def iterate_minibatches(
    x: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    *,
    rng: np.random.Generator,
    shuffle: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    # TODO: generate a permutation once per epoch and yield the final short batch.
    raise NotImplementedError


def logits(x: np.ndarray, model: MNISTLinearModel) -> np.ndarray:
    raise NotImplementedError


def loss_and_gradients(
    x: np.ndarray, labels: np.ndarray, model: MNISTLinearModel
) -> tuple[float, np.ndarray, np.ndarray]:
    # TODO: logits -> softmax cross-entropy -> dW and db.
    raise NotImplementedError


def predict(x: np.ndarray, model: MNISTLinearModel) -> np.ndarray:
    raise NotImplementedError


def evaluate(
    x: np.ndarray,
    labels: np.ndarray,
    model: MNISTLinearModel,
    *,
    batch_size: int = 2048,
) -> tuple[float, float]:
    # TODO: accumulate loss * batch_size and correct counts, then divide once.
    raise NotImplementedError


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    epochs: int = 8,
    batch_size: int = 128,
    learning_rate: float = 0.25,
    seed: int = 0,
    class_count: int = 10,
) -> tuple[MNISTLinearModel, list[EpochMetrics]]:
    # TODO: shuffle, update each mini-batch, evaluate after each epoch.
    raise NotImplementedError


def limit_splits(
    splits: PreparedMNIST,
    *,
    training_examples: int | None = None,
    validation_examples: int | None = None,
    test_examples: int | None = None,
) -> PreparedMNIST:
    raise NotImplementedError


def show_weight_templates(
    model: MNISTLinearModel, *, save_path: str | Path | None = None
) -> None:
    # TODO: reshape each column of weights to 28x28 and show it.
    raise NotImplementedError
