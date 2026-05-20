## NumPy as a tensor calculator

### Goal

Become comfortable enough with arrays and shapes that the later mathematics is
visible rather than hidden behind NumPy syntax.

### Part A: one dot product

Write:

```python
def manual_dot(left: np.ndarray, right: np.ndarray) -> float:
    ...
```

Requirements:

- both inputs must be one-dimensional;
- both must have the same length;
- use an explicit Python loop;
- do not call `np.dot`, `np.sum(left * right)`, or `@` in this first function.

Example:

```python
manual_dot(
    np.array([1.0, 2.0, 3.0]),
    np.array([4.0, 5.0, 6.0]),
)  # 32.0
```

Then compare it with:

```python
left @ right
```

### Part B: an affine transformation

A neural-network layer begins with:

```text
output = input @ weights + bias
```

Implement both:

```python
def affine_one(
    x: np.ndarray,          # (features,)
    weights: np.ndarray,    # (features, outputs)
    bias: np.ndarray,       # (outputs,)
) -> np.ndarray:
    ...


def affine_batch(
    x: np.ndarray,          # (examples, features)
    weights: np.ndarray,    # (features, outputs)
    bias: np.ndarray,       # (outputs,)
) -> np.ndarray:
    ...
```

For this example:

```python
x = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
])

weights = np.array([
    [10.0, 20.0, 30.0],
    [ 1.0,  2.0,  3.0],
])

bias = np.array([100.0, 200.0, 300.0])
```

predict all intermediate and final shapes before running it:

```text
X:       (?, ?)
W:       (?, ?)
X @ W:   (?, ?)
b:       (?,)
result:  (?, ?)
```

Explain why NumPy is allowed to add `bias` to a matrix with two rows.

### Part C: one-hot labels

Write:

```python
def one_hot(labels: np.ndarray, number_of_classes: int) -> np.ndarray:
    ...
```

Example:

```python
one_hot(np.array([2, 0, 1]), 3)
```

returns:

```python
np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])
```

Try to implement the final placement of ones without a loop:

```python
encoded[np.arange(...), labels] = 1.0
```

### Part D: flatten toy images

Write:

```python
def flatten_images(images: np.ndarray) -> np.ndarray:
    ...
```

Convert:

```text
(examples, rows, columns) -> (examples, rows * columns)
```

Use a toy batch with shape `(2, 3, 4)` and verify that flattening preserves the
order of the pixel values.

### Minimum tests

- `manual_dot` matches `@` for positive, negative, and zero values.
- mismatched vector lengths raise an error.
- `affine_batch` produces the same rows as calling `affine_one` repeatedly.
- one-hot rows each sum to one.
- flattening and reshaping back reproduce the original toy images.

### Questions to answer

1. In `X @ W`, which dimension disappears and why must it match?
2. What does `axis=0` usually mean for a `(batch, features)` array?
3. What does `axis=1` usually mean?
4. Why can `bias` have shape `(outputs,)` instead of `(batch, outputs)`?
5. What information is lost when `(28, 28)` is flattened to `(784,)`?

### Stretch

Time a Python loop and `@` on larger arrays.  Do not interpret this as “NumPy has
no loops”; its loops run in optimized compiled code rather than in the Python
interpreter.
