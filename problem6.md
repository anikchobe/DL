## Stage 6 — Fetch MNIST and parse the bytes yourself

### Goal

Understand where the arrays come from.  Download four compressed binary files,
verify their contents, decode their headers, and turn their payloads into NumPy
arrays without using a machine-learning dataset loader.

MNIST provides:

```text
train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
```

The course downloader tries these HTTPS mirrors in order:

```text
https://storage.googleapis.com/cvdf-datasets/mnist/
https://ossci-datasets.s3.amazonaws.com/mnist/
```

### Part A: resources and checksums

Represent each file with a small record containing a logical key, filename, and
SHA-256 checksum.

Use these checksums for the compressed bytes:

```text
train-images-idx3-ubyte.gz
440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609

train-labels-idx1-ubyte.gz
3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c

t10k-images-idx3-ubyte.gz
8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6

t10k-labels-idx1-ubyte.gz
f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6
```

Write:

```python
def sha256_file(path: Path) -> str:
    ...
```

Read in chunks instead of loading an arbitrary file into one giant Python bytes
object.

### Part B: download safely

Write:

```python
def download_mnist(data_directory: str | Path) -> dict[str, Path]:
    ...
```

Requirements:

- create `data_directory/raw` when necessary;
- skip an existing file only after its checksum passes;
- download to a temporary `.part` file;
- verify the temporary file before renaming it;
- try the second mirror if the first fails;
- preserve or rename a corrupt existing file rather than silently accepting it;
- produce one useful final error if every mirror fails.

Use only the standard library, for example:

```python
from urllib.request import Request, urlopen
```

Do not write four nearly identical download blocks.  Write one helper and call it
for each resource.

### Part C: understand the IDX image header

After gzip decompression, an image file begins with four big-endian 32-bit unsigned
integers:

```text
offset 0:  magic number      2051 = 0x00000803
offset 4:  number of images
offset 8:  rows
offset 12: columns
offset 16: first pixel byte
```

Draw the 16-byte header before coding.  Use:

```python
struct.unpack(">IIII", header)
```

The `>` means big-endian.

Write:

```python
def parse_idx_images(path: str | Path) -> np.ndarray:
    ...
```

Steps:

1. open with `gzip.open(path, "rb")`;
2. read and unpack the header;
3. verify magic number 2051;
4. verify positive dimensions;
5. verify that payload length equals `count * rows * columns`;
6. decode unsigned bytes with `np.frombuffer`;
7. copy and reshape to `(count, rows, columns)`.

Why make a copy?  An array created directly from an immutable `bytes` payload may
be read-only and retains a relationship with that backing buffer.

### Part D: parse labels

A label file begins with:

```text
offset 0: magic number       2049 = 0x00000801
offset 4: number of labels
offset 8: first label byte
```

Write:

```python
def parse_idx_labels(path: str | Path) -> np.ndarray:
    ...
```

Return shape `(count,)` and dtype `uint8`.

### Part E: combine and cache

Build:

```python
@dataclass(frozen=True)
class RawMNIST:
    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
```

Write loaders that verify:

```text
train images: (60000, 28, 28)
train labels: (60000,)
test images:  (10000, 28, 28)
test labels:  (10000,)
```

Then cache the arrays with:

```python
np.savez_compressed(...)
```

Load with:

```python
np.load(path, allow_pickle=False)
```

### Network-independent tests

Do not make every parser test download MNIST.  Create tiny gzip files in a temporary
directory:

```python
images = np.arange(24, dtype=np.uint8).reshape(3, 2, 4)
header = struct.pack(">IIII", 2051, 3, 2, 4)
```

Write the header and payload, parse them, and compare arrays.  Also test:

- wrong magic number;
- truncated payload;
- invalid gzip data;
- cache save/load round trip.

### Questions to answer

1. Why is the file header big-endian even if your computer is little-endian?
2. What does the magic number protect against?
3. What is the difference between compressed file bytes and decompressed IDX
   bytes?
4. Why verify a checksum before parsing?
5. What does `np.frombuffer` avoid compared with constructing a Python integer for
   every pixel?
6. Why should the parser calculate expected payload size rather than trust the
   file?

### Stretch

Write a generic IDX parser that decodes the type and number of dimensions from the
four magic-number bytes rather than hardcoding image and label layouts.  Keep the
specialized version first; it is easier to inspect.

