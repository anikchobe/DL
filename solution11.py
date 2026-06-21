from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from solution4 import cross_entropy_from_logits
from solution6 import prepare_mnist
from solution7 import PreparedMNIST, prepare_splits, show_image_grid
from solution8 import iterate_minibatches, limit_splits
from solution9 import MLPParameters, forward, initialize_mlp, predict
from solution10 import loss_and_gradients


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float


def evaluate(
    x: np.ndarray,
    labels: np.ndarray,
    parameters: MLPParameters,
    *,
    batch_size: int = 1024,
) -> tuple[float, float]:
    total_loss = 0.0
    correct = 0
    seen = 0
    rng = np.random.default_rng(0)

    for batch_x, batch_labels in iterate_minibatches(
        x, labels, batch_size, rng=rng, shuffle=False
    ):
        cache = forward(batch_x, parameters)
        total_loss += cross_entropy_from_logits(
            cache.logits, batch_labels
        ) * batch_x.shape[0]
        correct += int(np.sum(np.argmax(cache.logits, axis=1) == batch_labels))
        seen += batch_x.shape[0]

    return total_loss / seen, correct / seen


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    hidden_units: int = 128,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.08,
    seed: int = 0,
    class_count: int = 10,
) -> tuple[MLPParameters, list[EpochMetrics]]:
    # TODO: combine Stage 8 mini-batches with Stage 10 gradients.
    rng = np.random.default_rng(seed)
    parameters = initialize_mlp(
        x_train.shape[1], hidden_units, class_count, seed=seed, dtype=np.float32
    )
    history: list[EpochMetrics] = []

    for epoch in range(1, epochs + 1):
        for batch_x, batch_labels in iterate_minibatches(
            x_train, y_train, batch_size, rng=rng, shuffle=True
        ):
            _, gradients = loss_and_gradients(batch_x, batch_labels, parameters)
            parameters.weights1 -= learning_rate * gradients.weights1
            parameters.bias1 -= learning_rate * gradients.bias1
            parameters.weights2 -= learning_rate * gradients.weights2
            parameters.bias2 -= learning_rate * gradients.bias2

        train_loss, train_accuracy = evaluate(x_train, y_train, parameters)
        validation_loss, validation_accuracy = evaluate(
            x_validation, y_validation, parameters
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

    return parameters, history


def overfit_tiny_batch(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    example_count: int = 64,
    hidden_units: int = 64,
    steps: int = 1_500,
    learning_rate: float = 0.15,
    seed: int = 0,
) -> tuple[MLPParameters, float]:
    # TODO: repeatedly update on the same tiny set and report its accuracy.
    tiny_x = x[:example_count]
    tiny_labels = labels[:example_count]
    parameters = initialize_mlp(
        tiny_x.shape[1],
        hidden_units,
        int(labels.max()) + 1,
        seed=seed,
        dtype=np.float32,
    )

    for _ in range(steps):
        _, gradients = loss_and_gradients(tiny_x, tiny_labels, parameters)
        parameters.weights1 -= learning_rate * gradients.weights1
        parameters.bias1 -= learning_rate * gradients.bias1
        parameters.weights2 -= learning_rate * gradients.weights2
        parameters.bias2 -= learning_rate * gradients.bias2

    accuracy = float(np.mean(predict(tiny_x, parameters) == tiny_labels))
    return parameters, accuracy


def confusion_matrix(
    labels: np.ndarray, predictions: np.ndarray, class_count: int = 10
) -> np.ndarray:
    # TODO: np.add.at(matrix, (labels, predictions), 1).
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def most_confused_pairs(
    matrix: np.ndarray, count: int = 10
) -> list[tuple[int, int, int]]:
    copy = np.asarray(matrix).copy()
    np.fill_diagonal(copy, 0)
    pairs = []
    for flat_index in np.argsort(copy, axis=None)[::-1]:
        true_digit, predicted_digit = np.unravel_index(int(flat_index), copy.shape)
        errors = int(copy[true_digit, predicted_digit])
        if errors == 0 or len(pairs) == count:
            break
        pairs.append((true_digit, predicted_digit, errors))
    return pairs


def save_model(parameters: MLPParameters, path: str | Path) -> Path:
    # TODO: save four arrays with np.savez_compressed.
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        weights1=parameters.weights1,
        bias1=parameters.bias1,
        weights2=parameters.weights2,
        bias2=parameters.bias2,
    )
    return destination


def load_model(path: str | Path) -> MLPParameters:
    with np.load(path, allow_pickle=False) as archive:
        return MLPParameters(
            archive["weights1"],
            archive["bias1"],
            archive["weights2"],
            archive["bias2"],
        )


def plot_training_history(
    history: list[EpochMetrics], *, save_path: str | Path | None = None
) -> None:
    # TODO: one loss figure and one accuracy figure.
    import matplotlib.pyplot as plt

    epochs = [record.epoch for record in history]
    train_loss = [record.train_loss for record in history]
    validation_loss = [record.validation_loss for record in history]
    train_accuracy = [record.train_accuracy for record in history]
    validation_accuracy = [record.validation_accuracy for record in history]

    loss_figure = plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_loss, marker="o", label="train")
    plt.plot(epochs, validation_loss, marker="o", label="validation")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.legend()
    plt.tight_layout()

    accuracy_figure = plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_accuracy, marker="o", label="train")
    plt.plot(epochs, validation_accuracy, marker="o", label="validation")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        loss_figure.savefig(
            destination.with_name(destination.stem + "_loss.png"), dpi=150
        )
        accuracy_figure.savefig(
            destination.with_name(destination.stem + "_accuracy.png"), dpi=150
        )
        plt.close(loss_figure)
        plt.close(accuracy_figure)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a one-hidden-layer MNIST MLP")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--hidden-units", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use 15,000 train, 2,000 validation, and 2,000 test examples",
    )
    parser.add_argument(
        "--skip-overfit-check",
        action="store_true",
        help="skip the small-batch memorization diagnostic",
    )
    arguments = parser.parse_args()

    raw = prepare_mnist(arguments.data_dir)
    prepared: PreparedMNIST = prepare_splits(raw)
    if arguments.quick:
        prepared = limit_splits(
            prepared,
            training_examples=15_000,
            validation_examples=2_000,
            test_examples=2_000,
        )

    if not arguments.skip_overfit_check:
        _, tiny_accuracy = overfit_tiny_batch(
            prepared.x_train,
            prepared.y_train,
            hidden_units=min(arguments.hidden_units, 64),
        )
        print(f"tiny-batch memorization accuracy: {tiny_accuracy:.2%}")

    model, records = train(
        prepared.x_train,
        prepared.y_train,
        prepared.x_validation,
        prepared.y_validation,
        hidden_units=arguments.hidden_units,
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

    test_loss, test_accuracy = evaluate(prepared.x_test, prepared.y_test, model)
    print(f"test loss {test_loss:.4f}, test accuracy {test_accuracy:.2%}")
    test_predictions = predict(prepared.x_test, model)
    matrix = confusion_matrix(prepared.y_test, test_predictions)
    print("most common confusions (true, predicted, count):")
    for item in most_confused_pairs(matrix):
        print(" ", item)

    output_directory = Path(arguments.output_dir)
    model_path = save_model(model, output_directory / "mnist_mlp.npz")
    plot_training_history(records, save_path=output_directory / "training.png")

    wrong = np.flatnonzero(test_predictions != prepared.y_test)[:15]
    if wrong.size:
        show_image_grid(
            (prepared.x_test[wrong].reshape(-1, 28, 28) * 255.0).astype(np.uint8),
            prepared.y_test[wrong],
            predictions=test_predictions[wrong],
            save_path=output_directory / "misclassified.png",
        )

    reloaded = load_model(model_path)
    assert np.array_equal(
        predict(prepared.x_test[:100], model),
        predict(prepared.x_test[:100], reloaded),
    )
    print("wrote:", model_path)
    print("wrote training curves and misclassified examples under:", output_directory)


# Notes:
#
# The tiny-batch test asks only "can this implementation fit anything at all?"
# It is deliberately not evidence about generalization: the same 64 examples
# are both the training set and the thing being measured.
#
# Rows of the confusion matrix are true digits, columns are predicted digits.
# Recall for digit c is M[c,c] / row sum; precision is M[c,c] / column sum.  A
# high overall accuracy can hide one digit doing badly, because the other nine
# can carry it.
#
# Saving raw arrays rather than pickle means the file can be loaded without
# executing arbitrary Python.  allow_pickle=False enforces that.  The arrays
# are the model; everything else can be rebuilt from code.
#
# The final test set is used once, after hyperparameters are chosen on
# validation.  Looking at test accuracy and then going back to tune is how a
# "final" test quietly becomes a validation set.
