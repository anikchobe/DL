from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BinaryParameters:
    weights: np.ndarray
    bias: float


@dataclass(frozen=True)
class binaryEpoch:
    step: int
    loss: float
    accuracy: float

def sigmoid(logits: np.ndarray) -> np.ndarray:
    # TODO: 1/(1+exp(-z)) equals exp(-log(1+exp(-z))).  Use np.logaddexp for the
    # inner term so no exponential is ever formed.  One line, no overflow.
    values = np.asarray(logits, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -values))


def binary_cross_entropy_from_logits(
    logits: np.ndarray, labels: np.ndarray
) -> float:
    # TODO: mean of np.logaddexp(0, logits) - labels * logits.
    # Never build probabilities first: log(0.0) is -inf.
    return np.mean(np.logaddexp(0.0, logits) - labels * logits)


def forward(
    x: np.ndarray, parameters: BinaryParameters
) -> tuple[np.ndarray, np.ndarray]:
    # TODO: return logits and probabilities.
    logits = x @ parameters.weights + parameters.bias
    return logits, sigmoid(logits)


def loss_and_gradients(
    x: np.ndarray, labels: np.ndarray, parameters: BinaryParameters
) -> tuple[float, np.ndarray, float]:
    # TODO: dL/dlogits simplifies to (probabilities-labels)/batch_size.
    logits, probabilities = forward(x, parameters)
    loss = binary_cross_entropy_from_logits(logits, labels)
    gradientLogits = (probabilities - labels)/x.shape[0]
    gradientWeights = x.T @ gradientLogits
    gradientBias = np.sum(gradientLogits)
    return loss, gradientWeights, gradientBias

def predict(x: np.ndarray, parameters: BinaryParameters) -> np.ndarray:
    _, probabilities = forward(x, parameters)
    return (probabilities >= 0.5).astype(np.int64)

    
def make_binary_blobs(
    examples_per_class: int = 100, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    # TODO: generate two deterministic 2-D Gaussian clusters and shuffle them.
    rng = np.random.default_rng(seed)
    class_0 = rng.normal(loc=(-2,-2), scale=0.75, size=(examples_per_class, 2))
    class_1 = rng.normal(loc=(2,2), scale=0.75, size=(examples_per_class, 2))
    x = np.vstack((class_0, class_1))
    labels = np.concatenate((np.zeros(examples_per_class), np.ones(examples_per_class))).astype(np.int64)
    order = rng.permutation(x.shape[0])
    return x[order], labels[order]

def accuracy(x: np.ndarray, labels: np.ndarray, parameters: BinaryParameters) -> float:
    return np.mean(predict(x, parameters) == labels)


def train(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    learning_rate: float = 0.2,
    steps: int = 250,
) -> tuple[BinaryParameters, list[binaryEpoch]]:
    # TODO: initialize zeros and run full-batch gradient descent.
    x = np.asarray(x, dtype=np.float64)
    parameters = BinaryParameters(np.zeros(x.shape[1], dtype=np.float64), 0.0)
    history: list[binaryEpoch] = []
    for step in range(steps):
        loss, gradientWeights, gradientBias = loss_and_gradients(x, labels, parameters)
        history.append(binaryEpoch(step, loss, accuracy(x, labels, parameters)))
        parameters.weights -= learning_rate * gradientWeights
        parameters.bias -= learning_rate * gradientBias
    return parameters, history


x, labels = make_binary_blobs()
parameters, history = train(x, labels, steps=200)
print(f'parameters = {parameters}\n')

s = np.array([2.01, 1.99])
t = np.array([-2.01, -1.99])
print(forward(s, parameters))
print(forward(t, parameters))
# print(f'parameters = {parameters}')
# for step in history:
#     print(step)
