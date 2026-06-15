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
    rng = np.random.default_rng(seed)
    weights1 = rng.normal(
        0.0, np.sqrt(2.0 / input_features), size=(input_features, hidden_units)
    ).astype(dtype)
    weights2 = rng.normal(
        0.0, np.sqrt(2.0 / hidden_units), size=(hidden_units, class_count)
    ).astype(dtype)
    return MLPParameters(
        weights1,
        np.zeros(hidden_units, dtype=dtype),
        weights2,
        np.zeros(class_count, dtype=dtype),
    )


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def forward(x: np.ndarray, parameters: MLPParameters) -> ForwardCache:
    # TODO: Z1=XW1+b1; H=ReLU(Z1); logits=HW2+b2; probabilities=softmax.
    x = np.asarray(x, dtype=parameters.weights1.dtype)
    pre_activation1 = x @ parameters.weights1 + parameters.bias1
    activation1 = relu(pre_activation1)
    model_logits = activation1 @ parameters.weights2 + parameters.bias2
    probabilities = softmax(model_logits)
    return ForwardCache(
        x, pre_activation1, activation1, model_logits, probabilities
    )


def predict(x: np.ndarray, parameters: MLPParameters) -> np.ndarray:
    return np.argmax(forward(x, parameters).logits, axis=1)


def parameter_count(parameters: MLPParameters) -> int:
    return sum(
        array.size
        for array in (
            parameters.weights1,
            parameters.bias1,
            parameters.weights2,
            parameters.bias2,
        )
    )


def shape_trace(
    batch_size: int, input_features: int, hidden_units: int, class_count: int
) -> str:
    # TODO: print every intermediate shape one per line.
    return "\n".join(
        [
            f"X          ({batch_size}, {input_features})",
            f"W1         ({input_features}, {hidden_units})",
            f"b1         ({hidden_units},)",
            f"Z1 = XW1+b ({batch_size}, {hidden_units})",
            f"H = ReLU   ({batch_size}, {hidden_units})",
            f"W2         ({hidden_units}, {class_count})",
            f"b2         ({class_count},)",
            f"logits     ({batch_size}, {class_count})",
            f"probs      ({batch_size}, {class_count})",
        ]
    )


if __name__ == "__main__":
    print(shape_trace(128, 784, 64, 10))
    example = initialize_mlp(784, 64, 10)
    print("parameter count:", parameter_count(example))

    x = np.arange(24, dtype=np.float64).reshape(6, 4) / 24.0
    small = initialize_mlp(4, 5, 3, seed=2)
    cache = forward(x, small)
    print("Z1 shape:", cache.pre_activation1.shape)
    print("H shape: ", cache.activation1.shape)
    print("logits:  ", cache.logits.shape)
    print("row sums:", cache.probabilities.sum(axis=1))


# Notes:
#
# He init is sqrt(2 / fan_in) because ReLU zeroes roughly half its inputs.  A
# plain normal with a fixed small scale either shrinks the signal layer by
# layer or blows it up; the 2 compensates for the half that survives.
#
# Why zero init fails for hidden units, in one line: two hidden units with
# identical weights produce identical activations, receive identical gradients,
# and stay identical forever.  They are effectively one unit.  The output
# layer is different because each column is tied to a different class label.
#
# The cache is not optional.  Backward needs x for dW1, Z1 for the ReLU mask,
# and H1 for dW2.  Recomputing them would work but doubles the forward cost
# and risks using a different value if anything changed in between.
