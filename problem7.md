## Stage 7 — Inspect and visualize before training

### Goal

Treat dataset inspection as part of engineering, not as decoration.  Confirm what
the bytes mean, make the train/validation/test roles explicit, and prepare arrays
for the model.

### Part A: print one image in the terminal

Write:

```python
def image_to_ascii(
    image: np.ndarray,
    characters: str = " .:-=+*#%@",
) -> str:
    ...
```

Map low pixel values to spaces and high pixel values to dense characters.  Print
several digits and compare the output with their labels.

Questions to observe:

- Is pixel `(0, 0)` usually foreground or background?
- Are digits centered?
- Do some images use the full `0..255` range?
- Can you recognize ambiguous handwriting before a model sees it?

### Part B: Matplotlib grid

Write:

```python
def show_image_grid(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    predictions: np.ndarray | None = None,
    rows: int = 3,
    columns: int = 5,
    save_path: str | Path | None = None,
) -> None:
    ...
```

Import Matplotlib inside the function so the rest of the course can run without it.
Use a grayscale image display and turn axes off.  Later the same function will show
true and predicted labels for mistakes.

### Part C: inspect labels

Write:

```python
def label_counts(labels: np.ndarray, class_count: int = 10) -> np.ndarray:
    ...
```

Use `np.bincount`.  Print counts for train and test.  Do not assume that “ten
classes” means exactly equal counts.

As an optional visualization, calculate the mean training image for each digit.
This produces ten blurry class prototypes.

### Part D: normalize and flatten

Write:

```python
def normalize_and_flatten(images: np.ndarray) -> np.ndarray:
    ...
```

Transform:

```text
uint8 (N, 28, 28), values 0..255
    ->
float32 (N, 784), values 0..1
```

Be explicit about operation order.  Converting to `float32` before division avoids
integer arithmetic and makes the intended dtype clear.

Measure memory use:

```python
images.nbytes
flat.nbytes
```

Explain why `float64` uses twice the storage of `float32`.

### Part E: create a validation split

The original data has training and test sets but no validation set.  Write:

```python
def prepare_splits(
    raw: RawMNIST,
    *,
    validation_size: int = 10_000,
    seed: int = 0,
) -> PreparedMNIST:
    ...
```

Use a seeded random permutation of the 60,000 training indices:

```text
50,000 for parameter learning
10,000 for model/hyperparameter decisions
10,000 original test examples held aside
```

Do not create validation data by taking examples from the test set.

### Minimum tests

- ASCII output has the same number of rows as the image.
- Pixel zero maps to the least dense character and the maximum maps to the most
  dense character.
- normalization returns `float32`, values in `[0, 1]`, and shape `(N, 784)`.
- the same split seed produces the same split.
- different seeds change the selected examples.
- train and validation sizes add back to 60,000.
- test data is unchanged.

### Questions to answer

1. Why flatten an image for a fully connected network?
2. What spatial information is no longer explicit after flattening?
3. Why normalize inputs before gradient descent?
4. What is the different purpose of training, validation, and test data?
5. Why is repeatedly reporting test accuracy while tuning a form of information
   leakage?
6. Why should visualization happen before model training?

### Stretch

Find the nearest training image to a selected test image under Euclidean distance.
This is not the final classifier; it is a way to reason about pixels as vectors.

