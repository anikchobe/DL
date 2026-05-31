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
    logits = np.asarray(logits, dtype=np.float64)
    row_max = np.max(logits, axis=1, keepdims=True)
    shifted = logits - row_max
    log_denominator = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    return shifted - log_denominator


def softmax(logits: np.ndarray) -> np.ndarray:
    # TODO: exp(log_softmax(logits)).  Do not write exp/sum again here, and
    # never recover log-probabilities as log(softmax(...)): that route rounds
    # anything ~746 below the row maximum to 0.0 and then to -inf.
    return np.exp(log_softmax(logits))


def cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    # TODO: negative mean of log_probabilities[np.arange(examples), labels].
    # The two index arrays are zipped, not crossed: entry i of the first pairs
    # with entry i of the second.  log_probabilities[:, labels] is a different
    # thing that also runs.
    labels = np.asarray(labels, dtype=np.int64)
    log_probabilities = log_softmax(logits)
    picked = log_probabilities[np.arange(labels.shape[0]), labels]
    return float(-np.mean(picked))


def softmax_cross_entropy_with_gradient(
    logits: np.ndarray, labels: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    # TODO: return (loss, probabilities, gradient) so softmax is computed once.
    # gradient = (probabilities - one_hot(labels)) / batch_size, built as a
    # copy of the probabilities: `g = probabilities` aliases the array that is
    # about to be returned, and the in-place subtraction would corrupt it.
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = softmax(logits)
    loss = cross_entropy_from_logits(logits, labels)

    gradient = probabilities.copy()
    gradient[np.arange(labels.shape[0]), labels] -= 1.0
    gradient /= labels.shape[0]
    return loss, probabilities, gradient


def predict(logits: np.ndarray) -> np.ndarray:
    # TODO: argmax over classes.  softmax is not needed here -- it preserves
    # the order of every row -- but say why before you skip it.
    return np.argmax(logits, axis=1)


if __name__ == "__main__":
    logits = np.array([[3.0, 1.0, -2.0], [0.5, 0.2, -0.1]])
    labels = np.array([0, 2])

    loss, probabilities, gradient = softmax_cross_entropy_with_gradient(
        logits, labels
    )
    print("logits:")
    print(logits)
    print("probabilities:")
    print(probabilities)
    print("loss:", loss)
    print("dL/dlogits:")
    print(gradient)
    print("predictions:", predict(logits))

    # Squeezing a huge row down to ordinary values should not change anything.
    huge = np.array([[1001.0, 1000.0, 999.0]])
    print("huge row:", softmax(huge))
    print("shifted:  ", softmax(huge - 999.0))
    print("log(10):  ", np.log(10.0))


# Notes from working through the stage:
#
# The naive softmax exp(z)/sum(exp(z)) returns nan for [1001, 1000, 999]
# because exp overflows to inf and inf/inf is nan.  Subtracting the row max
# first makes the biggest exponent exactly 1, so the denominator is between 1
# and the number of classes.  That is an identity, not a trick: exp(k) cancels
# from the top and bottom.
#
# keepdims is the part I keep forgetting.  With a (10, 10) batch the shape
# (10,) happily broadcasts down the columns, every row still sums to one, and
# argmax comes back reversed.  A test that only checks the row sums passes for
# the broken version.  The useful test is "matches a Python loop over rows".
#
# The gradient rows sum to zero because adding a constant to a whole row of
# logits does not change the probabilities.  The true class gets p - 1, which
# is negative, so gradient descent pushes that logit up.  Every wrong class
# gets +p, so those logits go down.  That is the whole story.
