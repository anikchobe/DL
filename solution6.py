"""Stage 6 starter: download and parse raw MNIST IDX files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class MNISTDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class MNISTResource:
    key: str
    filename: str
    sha256: str
    compressed_size: int


@dataclass(frozen=True)
class RawMNIST:
    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray


MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)

# Copy the four filenames and checksums from problem6.md.
RESOURCES: tuple[MNISTResource, ...] = ()


def sha256_file(path: Path) -> str:
    # TODO: read in chunks and update hashlib.sha256().
    raise NotImplementedError


def verify_resource(path: Path, resource: MNISTResource) -> None:
    # TODO: hash the file and compare with resource.sha256.
    raise NotImplementedError


def download_mnist(data_directory: str | Path) -> dict[str, Path]:
    # TODO: create data/raw, use urllib.request, try each mirror, verify hashes,
    # and avoid downloading a file that is already present and valid.
    raise NotImplementedError


def parse_idx_images(path: str | Path) -> np.ndarray:
    # TODO: gzip.open -> read 16-byte big-endian header -> validate magic 2051 ->
    # np.frombuffer -> reshape to (image_count, rows, columns).
    raise NotImplementedError


def parse_idx_labels(path: str | Path) -> np.ndarray:
    # TODO: read 8-byte header, validate magic 2049, and parse uint8 labels.
    raise NotImplementedError


def load_raw_mnist(
    data_directory: str | Path, *, download: bool = True
) -> RawMNIST:
    # TODO: obtain paths, parse all four files, and validate matching shapes.
    raise NotImplementedError


def save_npz_cache(dataset: RawMNIST, path: str | Path) -> Path:
    # TODO: np.savez_compressed with four named arrays.
    raise NotImplementedError


def load_npz_cache(path: str | Path) -> RawMNIST:
    # TODO: np.load(..., allow_pickle=False) and reconstruct RawMNIST.
    raise NotImplementedError


def prepare_mnist(data_directory: str | Path) -> RawMNIST:
    # TODO: use the .npz cache when it exists, otherwise download and save it.
    raise NotImplementedError
