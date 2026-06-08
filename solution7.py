"""Stage 7 starter: inspect and visualize MNIST."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from solution6 import RawMNIST, prepare_mnist


@dataclass(frozen=True)
class PreparedMNIST:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def image_to_ascii(image: np.ndarray, characters: str = " .:-=+*#%@") -> str:
    # TODO: map pixel intensities to character indices and join rows with newlines.
    raise NotImplementedError


def label_counts(labels: np.ndarray, class_count: int = 10) -> np.ndarray:
    # TODO: use np.bincount.
    raise NotImplementedError


def normalize_and_flatten(images: np.ndarray) -> np.ndarray:
    # TODO: (N,28,28) uint8 -> (N,784) float32 in [0,1].
    raise NotImplementedError


def prepare_splits(
    raw: RawMNIST,
    *,
    validation_size: int = 10_000,
    seed: int = 0,
) -> PreparedMNIST:
    # TODO: deterministic shuffled train/validation split; never mix in test data.
    raise NotImplementedError


def class_mean_images(images: np.ndarray, labels: np.ndarray) -> np.ndarray:
    # TODO: average every training image belonging to each digit.
    raise NotImplementedError


def show_image_grid(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    predictions: np.ndarray | None = None,
    rows: int = 3,
    columns: int = 5,
    save_path: str | Path | None = None,
) -> None:
    # TODO: import matplotlib.pyplot inside the function and draw a grid.
    raise NotImplementedError


def show_class_means(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    save_path: str | Path | None = None,
) -> None:
    # TODO: draw the ten prototypes from class_mean_images.
    raise NotImplementedError
