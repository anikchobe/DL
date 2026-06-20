"""Stage 11 starter: train and diagnose the MLP on MNIST."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from solution4 import cross_entropy_from_logits
from solution6 import prepare_mnist
from solution7 import PreparedMNIST, prepare_splits, show_image_grid
from solution8 import iterate_minibatches, limit_splits
from solution9 import MLPParameters, forward, initialize_mlp, predict
from solution10 import loss_and_gradients


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float


def evaluate(
    x: np.ndarray,
    labels: np.ndarray,
    parameters: MLPParameters,
    *,
    batch_size: int = 1024,
) -> tuple[float, float]:
    raise NotImplementedError


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    hidden_units: int = 128,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.08,
    seed: int = 0,
    class_count: int = 10,
) -> tuple[MLPParameters, list[EpochMetrics]]:
    # TODO: combine Stage 8 mini-batches with Stage 10 gradients.
    raise NotImplementedError


def overfit_tiny_batch(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    example_count: int = 64,
    hidden_units: int = 64,
    steps: int = 1_500,
    learning_rate: float = 0.15,
    seed: int = 0,
) -> tuple[MLPParameters, float]:
    # TODO: repeatedly update on the same tiny set and report its accuracy.
    raise NotImplementedError


def confusion_matrix(
    labels: np.ndarray, predictions: np.ndarray, class_count: int = 10
) -> np.ndarray:
    # TODO: np.add.at(matrix, (labels, predictions), 1).
    raise NotImplementedError


def most_confused_pairs(
    matrix: np.ndarray, count: int = 10
) -> list[tuple[int, int, int]]:
    raise NotImplementedError


def save_model(parameters: MLPParameters, path: str | Path) -> Path:
    # TODO: save four arrays with np.savez_compressed.
    raise NotImplementedError


def load_model(path: str | Path) -> MLPParameters:
    # TODO: np.load(..., allow_pickle=False).
    raise NotImplementedError


def plot_training_history(
    history: list[EpochMetrics], *, save_path: str | Path | None = None
) -> None:
    # TODO: one loss figure and one accuracy figure.
    raise NotImplementedError
