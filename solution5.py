from __future__ import annotations

from dataclasses import dataclass, replace

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
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=(feature_count, class_count))
    bias = np.zeros(class_count, dtype=np.float64)
    return LinearParameters(weights, bias)


def logits(x: np.ndarray, parameters: LinearParameters) -> np.ndarray:
    # TODO: x @ weights + bias, with shape checks first.
    return x @ parameters.weights + parameters.bias


def loss_and_gradients(
    x: np.ndarray, labels: np.ndarray, parameters: LinearParameters
) -> tuple[float, LinearGradients]:
    # TODO: backprop through logits = X @ W + b.
    model_logits = logits(x, parameters)
    loss, _, gradientLogits = softmax_cross_entropy_with_gradient(
        model_logits, labels
    )
    gradientWeights = x.T @ gradientLogits
    gradientBias = np.sum(gradientLogits, axis=0)
    return loss, LinearGradients(gradientWeights, gradientBias)


def predict(x: np.ndarray, parameters: LinearParameters) -> np.ndarray:
    return np.argmax(logits(x, parameters), axis=1)


def accuracy(
    x: np.ndarray, labels: np.ndarray, parameters: LinearParameters
) -> float:
    return float(np.mean(predict(x, parameters) == labels))


def make_three_class_blobs(
    examples_per_class: int = 80, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.array([[-2.2, -1.2], [2.2, -1.2], [0.0, 2.5]])
    features = []
    labels = []
    for class_index, center in enumerate(centers):
        features.append(
            rng.normal(center, 0.55, size=(examples_per_class, center.shape[0]))
        )
        labels.append(np.full(examples_per_class, class_index, dtype=np.int64))
    x = np.vstack(features)
    y = np.concatenate(labels)
    order = rng.permutation(x.shape[0])
    return x[order], y[order]


def train(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    learning_rate: float = 0.2,
    steps: int = 300,
    seed: int = 0,
) -> tuple[LinearParameters, list[LinearStep]]:
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    parameters = initialize(x.shape[1], int(labels.max()) + 1, seed=seed)
    history: list[LinearStep] = []

    for step in range(steps):
        loss, gradients = loss_and_gradients(x, labels, parameters)
        history.append(LinearStep(step, loss, accuracy(x, labels, parameters)))
        parameters.weights -= learning_rate * gradients.weights
        parameters.bias -= learning_rate * gradients.bias
    return parameters, history


def _relative_error(analytic: float, numerical: float) -> float:
    return abs(analytic - numerical) / max(1.0, abs(analytic), abs(numerical))


def gradient_check(
    x: np.ndarray,
    labels: np.ndarray,
    parameters: LinearParameters,
    *,
    epsilon: float = 1e-5,
    checks_per_array: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    # TODO: perturb selected entries in W and b and compare numerical gradients.
    _, analytic_gradients = loss_and_gradients(x, labels, parameters)
    rng = np.random.default_rng(seed)
    maximum_errors = {}

    for name in ("weights", "bias"):
        values = getattr(parameters, name)
        analytic_values = getattr(analytic_gradients, name)
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
                maximum, _relative_error(float(analytic_values[index]), numerical)
            )

        maximum_errors[name] = maximum

    return maximum_errors


if __name__ == "__main__":
    features, targets = make_three_class_blobs(seed=7)
    initial = initialize(2, 3, seed=7)
    print(
        "gradient-check errors:",
        gradient_check(features[:12], targets[:12], initial),
    )

    model, records = train(features, targets)
    print("initial loss:", records[0].loss)
    print("final loss:  ", records[-1].loss)
    print("accuracy:    ", accuracy(features, targets, model))
    print("weights:")
    print(model.weights)
    print("bias:", model.bias)


# Notes:
#
# dW[f, c] = sum_i X[i, f] * dlogits[i, c].  One entry is every example's
# feature f voting on class c, added together.  The transpose in X.T @ dlogits
# is only there so the feature axis lines up; it is not doing any maths by
# itself.
#
# The bias gradient sums over examples because the bias is added to every
# example's logit for that class.  Summing over axis 0 gives one number per
# class, which is exactly the shape of b.
#
# dlogits already has the /batch_size inside it from solution4, so dW and db
# inherit it.  If I forgot that division, the gradients would be N times too
# big and the same learning rate would explode.
#
# A passing gradient check only says the derivative matches this loss.  It
# does not say the loss is a good one, the learning rate is sane, or the data
# is separable.  It just rules out one class of bug.
