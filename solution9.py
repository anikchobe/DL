"""Stage 9 starter: forward propagation through a one-hidden-layer MLP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solution4 import softmax


@dataclass
class MLPParameters:
    weights1: np.ndarray  # (input_features, hidden_units)
    bias1: np.ndarray  # (hidden_units,)
    weights2: np.ndarray  # (hidden_units, classes)
    bias2: np.ndarray  # (classes,)


@dataclass(frozen=True)
class ForwardCache:
    x: np.ndarray
    pre_activation1: np.ndarray
    activation1: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray


def initialize_mlp(
    input_features: int,
    hidden_units: int,
    class_count: int,
    *,
    seed: int = 0,
    dtype: np.dtype | type = np.float64,
) -> MLPParameters:
    # TODO: use random nonidentical weights.  Try He scale sqrt(2/fan_in).
    raise NotImplementedError


def relu(values: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def forward(x: np.ndarray, parameters: MLPParameters) -> ForwardCache:
    # TODO: Z1=XW1+b1; H=ReLU(Z1); logits=HW2+b2; probabilities=softmax.
    raise NotImplementedError


def predict(x: np.ndarray, parameters: MLPParameters) -> np.ndarray:
    raise NotImplementedError


def parameter_count(parameters: MLPParameters) -> int:
    raise NotImplementedError


def shape_trace(
    batch_size: int, input_features: int, hidden_units: int, class_count: int
) -> str:
    # TODO: print every intermediate shape one per line.
    raise NotImplementedError
