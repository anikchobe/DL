"""Stage 6: fetch and parse MNIST with nothing but the standard library."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
from pathlib import Path
import shutil
import struct
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
    train_images: np.ndarray  # uint8 (60000, 28, 28)
    train_labels: np.ndarray  # uint8 (60000,)
    test_images: np.ndarray  # uint8 (10000, 28, 28)
    test_labels: np.ndarray  # uint8 (10000,)


MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)

RESOURCES = (
    MNISTResource(
        "train_images",
        "train-images-idx3-ubyte.gz",
        "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
        9_912_422,
    ),
    MNISTResource(
        "train_labels",
        "train-labels-idx1-ubyte.gz",
        "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
        28_881,
    ),
    MNISTResource(
        "test_images",
        "t10k-images-idx3-ubyte.gz",
        "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
        1_648_877,
    ),
    MNISTResource(
        "test_labels",
        "t10k-labels-idx1-ubyte.gz",
        "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
        4_542,
    ),
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    # TODO: read in chunks and update hashlib.sha256().
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_resource(path: Path, resource: MNISTResource) -> None:
    if not Path(path).is_file():
        raise MNISTDataError(f"missing file: {path}")
    actual = sha256_file(path)
    if actual != resource.sha256:
        raise MNISTDataError(
            f"bad checksum for {resource.filename}: expected "
            f"{resource.sha256}, got {actual}"
        )


def _download_one(
    resource: MNISTResource,
    raw_directory: Path,
    *,
    mirrors: tuple[str, ...] = MIRRORS,
    timeout_seconds: float = 60.0,
) -> Path:
    destination = raw_directory / resource.filename
    if destination.exists():
        try:
            verify_resource(destination, resource)
            return destination
        except MNISTDataError:
            # Keep the bad file instead of pretending it was fine.
            corrupt = destination.with_suffix(destination.suffix + ".corrupt")
            if corrupt.exists():
                corrupt.unlink()
            destination.replace(corrupt)

    failures = []
    partial = destination.with_suffix(destination.suffix + ".part")
    for mirror in mirrors:
        url = mirror + resource.filename
        try:
            if partial.exists():
                partial.unlink()
            request = Request(url, headers={"User-Agent": "mnist-numpy-course/1.0"})
            print("Downloading", resource.filename, "from", mirror)
            with urlopen(request, timeout=timeout_seconds) as response, partial.open(
                "wb"
            ) as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            verify_resource(partial, resource)
            partial.replace(destination)
            return destination
        except (HTTPError, URLError, OSError, MNISTDataError) as error:
            failures.append(f"{url}: {error}")
            if partial.exists():
                partial.unlink()

    detail = "\n".join(f"  - {failure}" for failure in failures)
    raise MNISTDataError(
        f"could not download {resource.filename} from any mirror:\n{detail}"
    )


def download_mnist(
    data_directory: str | Path, *, mirrors: tuple[str, ...] = MIRRORS
) -> dict[str, Path]:
    # TODO: one helper, called once per resource; do not copy-paste four blocks.
    raw_directory = Path(data_directory) / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    return {
        resource.key: _download_one(resource, raw_directory, mirrors=mirrors)
        for resource in RESOURCES
    }


def _read_gzip_bytes(path: str | Path) -> bytes:
    try:
        with gzip.open(path, "rb") as source:
            return source.read()
    except (OSError, EOFError) as error:
        raise MNISTDataError(f"cannot read gzip file {path}: {error}") from error


def parse_idx_images(path: str | Path) -> np.ndarray:
    # TODO: 16-byte header, magic 2051, payload count * rows * columns.
    raw = _read_gzip_bytes(path)
    if len(raw) < 16:
        raise MNISTDataError(f"{path} is shorter than its 16-byte header")

    magic, image_count, rows, columns = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise MNISTDataError(f"bad image magic number {magic}; expected 2051")
    if image_count == 0 or rows == 0 or columns == 0:
        raise MNISTDataError("image dimensions must be positive")

    payload = raw[16:]
    expected = image_count * rows * columns
    if len(payload) != expected:
        raise MNISTDataError(
            f"image payload has {len(payload)} bytes; expected {expected}"
        )

    return np.frombuffer(payload, dtype=np.uint8).copy().reshape(
        image_count, rows, columns
    )


def parse_idx_labels(path: str | Path) -> np.ndarray:
    raw = _read_gzip_bytes(path)
    if len(raw) < 8:
        raise MNISTDataError(f"{path} is shorter than its 8-byte header")

    magic, label_count = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise MNISTDataError(f"bad label magic number {magic}; expected 2049")

    payload = raw[8:]
    if len(payload) != label_count:
        raise MNISTDataError(
            f"label payload has {len(payload)} bytes; expected {label_count}"
        )
    return np.frombuffer(payload, dtype=np.uint8).copy()


def load_raw_mnist(
    data_directory: str | Path, *, download: bool = True
) -> RawMNIST:
    root = Path(data_directory)
    raw_directory = root / "raw"
    paths = {resource.key: raw_directory / resource.filename for resource in RESOURCES}

    if download:
        paths = download_mnist(root)
    else:
        for resource in RESOURCES:
            verify_resource(paths[resource.key], resource)

    train_images = parse_idx_images(paths["train_images"])
    train_labels = parse_idx_labels(paths["train_labels"])
    test_images = parse_idx_images(paths["test_images"])
    test_labels = parse_idx_labels(paths["test_labels"])

    if train_images.shape != (60_000, 28, 28):
        raise MNISTDataError(f"unexpected training image shape: {train_images.shape}")
    if train_labels.shape != (60_000,):
        raise MNISTDataError(f"unexpected training label shape: {train_labels.shape}")
    if test_images.shape != (10_000, 28, 28):
        raise MNISTDataError(f"unexpected test image shape: {test_images.shape}")
    if test_labels.shape != (10_000,):
        raise MNISTDataError(f"unexpected test label shape: {test_labels.shape}")

    return RawMNIST(train_images, train_labels, test_images, test_labels)


def save_npz_cache(dataset: RawMNIST, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        train_images=dataset.train_images,
        train_labels=dataset.train_labels,
        test_images=dataset.test_images,
        test_labels=dataset.test_labels,
    )
    return destination


def load_npz_cache(path: str | Path) -> RawMNIST:
    with np.load(path, allow_pickle=False) as archive:
        return RawMNIST(
            archive["train_images"],
            archive["train_labels"],
            archive["test_images"],
            archive["test_labels"],
        )


def prepare_mnist(data_directory: str | Path) -> RawMNIST:
    root = Path(data_directory)
    cache = root / "mnist_uint8.npz"
    if cache.exists():
        return load_npz_cache(cache)
    dataset = load_raw_mnist(root, download=True)
    save_npz_cache(dataset, cache)
    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and parse raw MNIST")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="require the four verified files to already exist",
    )
    arguments = parser.parse_args()

    dataset = load_raw_mnist(arguments.data_dir, download=not arguments.no_download)
    cache_path = save_npz_cache(
        dataset, Path(arguments.data_dir) / "mnist_uint8.npz"
    )
    print("train images:", dataset.train_images.shape, dataset.train_images.dtype)
    print("train labels:", dataset.train_labels.shape, dataset.train_labels.dtype)
    print("test images: ", dataset.test_images.shape, dataset.test_images.dtype)
    print("test labels: ", dataset.test_labels.shape, dataset.test_labels.dtype)
    print("cache:", cache_path)


# Header notes, because I keep having to look them up:
#
# images:  magic 2051, count, rows, columns, then count*rows*columns bytes
# labels:  magic 2049, count, then count bytes
#
# Both headers are big-endian.  That is a property of the file format, not of
# this laptop, so ">IIII" is required even on a little-endian machine.  The
# magic number is just a sanity check that I opened the right kind of file --
# a label file parsed as images would produce nonsense shapes otherwise.
#
# np.frombuffer does not copy, so the array is read-only and still points at
# the bytes object.  .copy() gives a normal writable array and lets the bytes
# go away.  The reshape itself is free because the data is contiguous.
