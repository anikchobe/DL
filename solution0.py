"""Stage 0 starter: NumPy arrays, shapes, and broadcasting."""

from __future__ import annotations

import numpy as np


def manual_dot(left: np.ndarray, right: np.ndarray) -> float:
    # TODO: validate two equal-length vectors and use an explicit Python loop.
    raise NotImplementedError


def affine_one(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    # TODO: compute one feature vector times a weight matrix, then add bias.
    raise NotImplementedError


def affine_batch(
    x: np.ndarray, weights: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    # TODO: compute the whole batch without a Python loop over examples.
    raise NotImplementedError


def one_hot(labels: np.ndarray, number_of_classes: int) -> np.ndarray:
    # TODO: make a zero matrix, then use advanced indexing to place the ones.
    raise NotImplementedError


def flatten_images(images: np.ndarray) -> np.ndarray:
    # TODO: convert (N, rows, columns) to (N, rows * columns).
    raise NotImplementedError
