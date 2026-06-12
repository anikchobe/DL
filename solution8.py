from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from solution4 import softmax_cross_entropy_with_gradient
from solution6 import prepare_mnist
from solution7 import PreparedMNIST, prepare_splits


@dataclass
class MNISTLinearModel:
    weights: np.ndarray  # float32 (784, 10)
    bias: np.ndarray  # float32 (10,)


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float


def initialize_model(
    feature_count: int = 784, class_count: int = 10
) -> MNISTLinearModel:
    # TODO: zero weights and bias are fine for a linear classifier.
    return MNISTLinearModel(
        np.zeros((feature_count, class_count), dtype=np.float32),
        np.zeros(class_count, dtype=np.float32),
    )


def iterate_minibatches(
    x: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    *,
    rng: np.random.Generator,
    shuffle: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    # TODO: generate a permutation once per epoch and yield the final short batch.
    indices = rng.permutation(x.shape[0]) if shuffle else np.arange(x.shape[0])
    for start in range(0, x.shape[0], batch_size):
        batch_indices = indices[start : start + batch_size]
        yield x[batch_indices], labels[batch_indices]


def logits(x: np.ndarray, model: MNISTLinearModel) -> np.ndarray:
    return x @ model.weights + model.bias


def loss_and_gradients(
    x: np.ndarray, labels: np.ndarray, model: MNISTLinearModel
) -> tuple[float, np.ndarray, np.ndarray]:
    # TODO: logits -> softmax cross-entropy -> dW and db.
    batch_logits = logits(x, model)
    loss, _, gradientLogits = softmax_cross_entropy_with_gradient(
        batch_logits, labels
    )
    gradientWeights = x.T @ gradientLogits
    gradientBias = np.sum(gradientLogits, axis=0)
    return loss, gradientWeights, gradientBias


def predict(x: np.ndarray, model: MNISTLinearModel) -> np.ndarray:
    return np.argmax(logits(x, model), axis=1)


def evaluate(
    x: np.ndarray,
    labels: np.ndarray,
    model: MNISTLinearModel,
    *,
    batch_size: int = 2048,
) -> tuple[float, float]:
    # TODO: accumulate loss * batch_size and correct counts, then divide once.
    total_loss = 0.0
    correct = 0
    seen = 0
    rng = np.random.default_rng(0)

    for batch_x, batch_labels in iterate_minibatches(
        x, labels, batch_size, rng=rng, shuffle=False
    ):
        batch_logits = logits(batch_x, model)
        batch_loss = softmax_cross_entropy_with_gradient(
            batch_logits, batch_labels
        )[0]
        total_loss += batch_loss * batch_x.shape[0]
        correct += int(np.sum(np.argmax(batch_logits, axis=1) == batch_labels))
        seen += batch_x.shape[0]

    return total_loss / seen, correct / seen


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    epochs: int = 8,
    batch_size: int = 128,
    learning_rate: float = 0.25,
    seed: int = 0,
    class_count: int = 10,
) -> tuple[MNISTLinearModel, list[EpochMetrics]]:
    # TODO: shuffle, update each mini-batch, evaluate after each epoch.
    rng = np.random.default_rng(seed)
    model = initialize_model(x_train.shape[1], class_count)
    history: list[EpochMetrics] = []

    for epoch in range(1, epochs + 1):
        for batch_x, batch_labels in iterate_minibatches(
            x_train, y_train, batch_size, rng=rng, shuffle=True
        ):
            _, gradientWeights, gradientBias = loss_and_gradients(
                batch_x, batch_labels, model
            )
            model.weights -= np.float32(learning_rate) * gradientWeights.astype(
                np.float32
            )
            model.bias -= np.float32(learning_rate) * gradientBias.astype(np.float32)

        train_loss, train_accuracy = evaluate(x_train, y_train, model)
        validation_loss, validation_accuracy = evaluate(
            x_validation, y_validation, model
        )
        history.append(
            EpochMetrics(
                epoch,
                train_loss,
                train_accuracy,
                validation_loss,
                validation_accuracy,
            )
        )

    return model, history


def limit_splits(
    splits: PreparedMNIST,
    *,
    training_examples: int | None = None,
    validation_examples: int | None = None,
    test_examples: int | None = None,
) -> PreparedMNIST:
    def limit(array: np.ndarray, count: int | None) -> np.ndarray:
        return array if count is None else array[:count]

    return PreparedMNIST(
        limit(splits.x_train, training_examples),
        limit(splits.y_train, training_examples),
        limit(splits.x_validation, validation_examples),
        limit(splits.y_validation, validation_examples),
        limit(splits.x_test, test_examples),
        limit(splits.y_test, test_examples),
    )


def show_weight_templates(
    model: MNISTLinearModel, *, save_path: str | Path | None = None
) -> None:
    # TODO: reshape each column of weights to 28x28 and show it.
    import matplotlib.pyplot as plt

    scale = float(np.max(np.abs(model.weights)))
    figure, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for digit, axis in enumerate(np.asarray(axes).reshape(-1)):
        axis.imshow(
            model.weights[:, digit].reshape(28, 28),
            cmap="coolwarm",
            vmin=-scale,
            vmax=scale,
        )
        axis.set_title(str(digit))
        axis.axis("off")
    figure.suptitle("Linear classifier weight template for each class")
    figure.tight_layout()

    if save_path is None:
        plt.show()
    else:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150)
        plt.close(figure)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train softmax regression on MNIST")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use 12,000 train and 2,000 validation examples",
    )
    arguments = parser.parse_args()

    raw = prepare_mnist(arguments.data_dir)
    prepared = prepare_splits(raw)
    if arguments.quick:
        prepared = limit_splits(
            prepared,
            training_examples=12_000,
            validation_examples=2_000,
            test_examples=2_000,
        )

    trained_model, records = train(
        prepared.x_train,
        prepared.y_train,
        prepared.x_validation,
        prepared.y_validation,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
    )
    for record in records:
        print(
            f"epoch {record.epoch:2d}: "
            f"train loss {record.train_loss:.4f}, "
            f"train acc {record.train_accuracy:.2%}, "
            f"val loss {record.validation_loss:.4f}, "
            f"val acc {record.validation_accuracy:.2%}"
        )

    test_loss, test_accuracy = evaluate(
        prepared.x_test, prepared.y_test, trained_model
    )
    print(f"test loss {test_loss:.4f}, test accuracy {test_accuracy:.2%}")
    destination = Path(arguments.output_dir) / "linear_weight_templates.png"
    show_weight_templates(trained_model, save_path=destination)
    print("wrote:", destination)


# Notes:
#
# Zero init works here even though it will not work for hidden units later.
# The ten output columns start identical, but each class gets a different
# gradient because the labels point at different columns, so they separate on
# the first update.  Hidden units have no such label attached to them.
#
# The noise in a mini-batch gradient is not a bug.  A batch of 128 is a sample
# of the training set, so its gradient points downhill on that sample, not
# exactly downhill on the whole loss.  Smaller batches are noisier and usually
# need a smaller learning rate; larger batches are smoother but do more work
# per step.
#
# The final short batch is easy to drop by accident.  That silently changes
# the epoch definition: some examples get seen once fewer than everyone else.
# The `range(0, n, batch_size)` loop yields it naturally.
#
# train_loss and validation_loss are accumulated as loss * batch_size and
# divided by the total seen, not averaged over batches.  Otherwise the last
# short batch would count as much as a full one.
