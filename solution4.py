"""Stage 4 starter: stable softmax and cross-entropy.

Shapes used throughout:

    logits, probabilities: (examples, classes)
    labels:                (examples,)  integer class indices, not one-hot
"""

from __future__ import annotations

import numpy as np


def log_softmax(logits: np.ndarray) -> np.ndarray:
    # TODO: log softmax(z)[c] = z[c] - max(z) - log(sum(exp(z - max(z)))).
    # Both reductions are over classes, so axis=1 with keepdims=True.  Without
    # keepdims the (examples,) result broadcasts against the class axis: an
    # error when examples != classes, and a silent wrong answer when they match.
    raise NotImplementedError


def softmax(logits: np.ndarray) -> np.ndarray:
    # TODO: exp(log_softmax(logits)).  Do not write exp/sum again here, and
    # never recover log-probabilities as log(softmax(...)): that route rounds
    # anything ~746 below the row maximum to 0.0 and then to -inf.
    raise NotImplementedError


def cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    # TODO: negative mean of log_probabilities[np.arange(examples), labels].
    # The two index arrays are zipped, not crossed: entry i of the first pairs
    # with entry i of the second.  log_probabilities[:, labels] is a different
    # thing that also runs.
    raise NotImplementedError


def softmax_cross_entropy_with_gradient(
    logits: np.ndarray, labels: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    # TODO: return (loss, probabilities, gradient) so softmax is computed once.
    # gradient = (probabilities - one_hot(labels)) / batch_size, built as a
    # copy of the probabilities: `g = probabilities` aliases the array that is
    # about to be returned, and the in-place subtraction would corrupt it.
    raise NotImplementedError


def predict(logits: np.ndarray) -> np.ndarray:
    # TODO: argmax over classes.  softmax is not needed here -- it preserves
    # the order of every row -- but say why before you skip it.
    raise NotImplementedError
