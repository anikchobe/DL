"""Stage 7: look at the data before trusting a model with it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from solution6 import RawMNIST, prepare_mnist


@dataclass(frozen=True)
class PreparedMNIST:
    x_train: np.ndarray  # float32 (train_examples, 784)
    y_train: np.ndarray  # int64 (train_examples,)
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def image_to_ascii(image: np.ndarray, characters: str = " .:-=+*#%@") -> str:
    # TODO: map pixel intensities to character indices and join rows with newlines.
    image = np.asarray(image)
    scaled = image.astype(np.float64) / 255.0
    indices = np.rint(scaled * (len(characters) - 1)).astype(np.int64)
    return "\n".join(
        "".join(characters[index] for index in row) for row in indices
    )


def label_counts(labels: np.ndarray, class_count: int = 10) -> np.ndarray:
    # TODO: use np.bincount.
    return np.bincount(np.asarray(labels, dtype=np.int64), minlength=class_count)


def normalize_and_flatten(images: np.ndarray) -> np.ndarray:
    # TODO: (N,28,28) uint8 -> (N,784) float32 in [0,1].
    images = np.asarray(images)
    flattened = images.reshape(images.shape[0], -1)
    return flattened.astype(np.float32) / np.float32(255.0)


def prepare_splits(
    raw: RawMNIST,
    *,
    validation_size: int = 10_000,
    seed: int = 0,
) -> PreparedMNIST:
    # TODO: deterministic shuffled train/validation split; never mix in test data.
    example_count = raw.train_images.shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(example_count)
    validation_indices = order[:validation_size]
    training_indices = order[validation_size:]

    return PreparedMNIST(
        x_train=normalize_and_flatten(raw.train_images[training_indices]),
        y_train=raw.train_labels[training_indices].astype(np.int64),
        x_validation=normalize_and_flatten(raw.train_images[validation_indices]),
        y_validation=raw.train_labels[validation_indices].astype(np.int64),
        x_test=normalize_and_flatten(raw.test_images),
        y_test=raw.test_labels.astype(np.int64),
    )


def class_mean_images(images: np.ndarray, labels: np.ndarray) -> np.ndarray:
    images = np.asarray(images, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    means = np.zeros((10, images.shape[1], images.shape[2]), dtype=np.float64)
    for digit in range(10):
        means[digit] = images[labels == digit].mean(axis=0)
    return means


def show_image_grid(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    predictions: np.ndarray | None = None,
    rows: int = 3,
    columns: int = 5,
    save_path: str | Path | None = None,
) -> None:
    # TODO: import matplotlib.pyplot inside the function and draw a grid.
    import matplotlib.pyplot as plt

    images = np.asarray(images)
    labels = np.asarray(labels)
    count = min(rows * columns, images.shape[0])
    figure, axes = plt.subplots(rows, columns, figsize=(1.8 * columns, 1.9 * rows))
    for index, axis in enumerate(np.asarray(axes).reshape(-1)):
        axis.axis("off")
        if index >= count:
            continue
        axis.imshow(images[index], cmap="gray", vmin=0, vmax=255)
        if predictions is None:
            axis.set_title(f"label {int(labels[index])}")
        else:
            axis.set_title(
                f"true {int(labels[index])}, pred {int(predictions[index])}"
            )
    figure.tight_layout()

    if save_path is None:
        plt.show()
    else:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150)
        plt.close(figure)


def show_class_means(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    save_path: str | Path | None = None,
) -> None:
    import matplotlib.pyplot as plt

    means = class_mean_images(images, labels)
    figure, axes = plt.subplots(2, 5, figsize=(10, 4.2))
    for digit, axis in enumerate(np.asarray(axes).reshape(-1)):
        axis.imshow(means[digit], cmap="gray", vmin=0, vmax=255)
        axis.set_title(str(digit))
        axis.axis("off")
    figure.suptitle("Mean training image for each digit")
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

    parser = argparse.ArgumentParser(description="Inspect and visualize MNIST")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    arguments = parser.parse_args()

    raw = prepare_mnist(arguments.data_dir)
    print("first training label:", int(raw.train_labels[0]))
    print(image_to_ascii(raw.train_images[0]))
    print("training label counts:", label_counts(raw.train_labels))

    output_directory = Path(arguments.output_dir)
    show_image_grid(
        raw.train_images[:15],
        raw.train_labels[:15],
        save_path=output_directory / "mnist_grid.png",
    )
    show_class_means(
        raw.train_images,
        raw.train_labels,
        save_path=output_directory / "mnist_class_means.png",
    )

    splits = prepare_splits(raw)
    print("prepared shapes:")
    print("  train:     ", splits.x_train.shape, splits.y_train.shape)
    print("  validation:", splits.x_validation.shape, splits.y_validation.shape)
    print("  test:      ", splits.x_test.shape, splits.y_test.shape)
    print("wrote:", output_directory / "mnist_grid.png")
    print("wrote:", output_directory / "mnist_class_means.png")


# Notes:
#
# Flattening throws away the fact that neighbouring pixels are neighbours.  A
# fully connected layer cannot tell that pixel 10 and pixel 11 are adjacent any
# more than it can tell that pixel 10 and pixel 700 are.  That is exactly what
# convolution puts back in Extension 7.
#
# The train/validation split comes only from the 60,000 training images.  If I
# used the test set for validation I would be choosing hyperparameters on the
# data I later report as a clean final test, which makes that number optimistic.
#
# float32 is half of float64.  60,000 * 784 * 4 bytes is about 188 MB for the
# training images alone, versus 376 MB in float64.  The matrix multiplies are
# noticeably faster in float32 too.
