from __future__ import annotations

from dataclasses import dataclass, replace

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
    cache = forward(x, parameters)
    loss, _, gradientLogits = softmax_cross_entropy_with_gradient(
        cache.logits, labels
    )
    gradientLogits = gradientLogits.astype(parameters.weights2.dtype, copy=False)

    gradientWeights2 = cache.activation1.T @ gradientLogits
    gradientBias2 = np.sum(gradientLogits, axis=0)
    gradientActivation1 = gradientLogits @ parameters.weights2.T

    gradientPreActivation1 = gradientActivation1 * (
        cache.pre_activation1 > 0.0
    )

    gradientWeights1 = cache.x.T @ gradientPreActivation1
    gradientBias1 = np.sum(gradientPreActivation1, axis=0)

    return loss, MLPGradients(
        gradientWeights1,
        gradientBias1,
        gradientWeights2,
        gradientBias2,
    )


def accuracy(
    x: np.ndarray, labels: np.ndarray, parameters: MLPParameters
) -> float:
    return float(np.mean(predict(x, parameters) == labels))


def train_full_batch(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    hidden_units: int = 8,
    learning_rate: float = 0.1,
    steps: int = 2_000,
    seed: int = 0,
) -> tuple[MLPParameters, list[MLPStep]]:
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    parameters = initialize_mlp(
        x.shape[1], hidden_units, int(labels.max()) + 1, seed=seed
    )
    history: list[MLPStep] = []

    for step in range(steps):
        loss, gradients = loss_and_gradients(x, labels, parameters)
        if step % 25 == 0 or step == steps - 1:
            history.append(MLPStep(step, loss, accuracy(x, labels, parameters)))
        parameters.weights1 -= learning_rate * gradients.weights1
        parameters.bias1 -= learning_rate * gradients.bias1
        parameters.weights2 -= learning_rate * gradients.weights2
        parameters.bias2 -= learning_rate * gradients.bias2
    return parameters, history


def make_noisy_xor(
    examples_per_corner: int = 50, *, noise: float = 0.12, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    corners = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    corner_labels = np.array([0, 1, 1, 0], dtype=np.int64)
    features = np.vstack(
        [rng.normal(corner, noise, size=(examples_per_corner, 2)) for corner in corners]
    )
    labels = np.concatenate(
        [
            np.full(examples_per_corner, label, dtype=np.int64)
            for label in corner_labels
        ]
    )
    order = rng.permutation(features.shape[0])
    return features[order], labels[order]


def _relative_error(analytic: float, numerical: float) -> float:
    return abs(analytic - numerical) / max(1.0, abs(analytic), abs(numerical))


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
    _, analyticGradients = loss_and_gradients(x, labels, parameters)
    rng = np.random.default_rng(seed)
    errors = {}

    for name in ("weights1", "bias1", "weights2", "bias2"):
        values = getattr(parameters, name)
        analytic = getattr(analyticGradients, name)
        selected = rng.choice(
            values.size, size=min(checks_per_array, values.size), replace=False
        )
        maximum = 0.0

        for flat_index in selected:
            index = np.unravel_index(int(flat_index), values.shape)
            plus_values = values.copy()
            minus_values = values.copy()
            plus_values[index] += epsilon
            minus_values[index] -= epsilon
            plus = replace(parameters, **{name: plus_values})
            minus = replace(parameters, **{name: minus_values})
            plus_loss = loss_and_gradients(x, labels, plus)[0]
            minus_loss = loss_and_gradients(x, labels, minus)[0]
            numerical = (plus_loss - minus_loss) / (2.0 * epsilon)
            maximum = max(
                maximum, _relative_error(float(analytic[index]), numerical)
            )
        errors[name] = maximum

    return errors


if __name__ == "__main__":
    check_x, check_y = make_noisy_xor(examples_per_corner=2, seed=10)
    check_model = initialize_mlp(2, 5, 2, seed=11)
    # Keep the biases off exact ReLU kinks so finite differences behave.
    check_model.bias1 += 0.2
    print("gradient-check errors:")
    for name, error in gradient_check(check_x, check_y, check_model).items():
        print(f"  {name:8s}: {error:.3e}")

    features, targets = make_noisy_xor(seed=5)
    trained, records = train_full_batch(features, targets, seed=6)
    print("initial loss:", records[0].loss)
    print("final loss:  ", records[-1].loss)
    print("XOR accuracy:", accuracy(features, targets, trained))


# Notes:
#
# Backprop is the chain rule run right to left, not the input data literally
# travelling backwards.  dH1 is the derivative with respect to the hidden
# activation; dZ1 is the derivative with respect to the value before ReLU.
# They differ by the ReLU mask.
#
# The mask uses the saved pre-activation Z1, not H1.  Checking H1 > 0 works by
# accident (ReLU output is positive exactly when the input was), but it hides
# the fact that the derivative belongs to the pre-activation.  At exactly zero
# I choose derivative zero, same as np.maximum does.
#
# If one gradient array fails the check, check the easy suspects first:
# missing /B, using W2 instead of W2.T, summing db on the wrong axis, or
# accidentally modifying probabilities in place before reusing them.
