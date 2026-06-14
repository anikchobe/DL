## Stage 9 — Add a hidden layer: forward propagation through an MLP

### Goal

Build the first genuinely nonlinear network and make every intermediate tensor
explicit before attempting its gradients.

Use:

```text
input -> affine -> ReLU -> affine -> softmax
```

For batch size `B`, input features `D`, hidden units `H`, and classes `C`:

```text
X:  (B, D)
W1: (D, H)
b1: (H,)
W2: (H, C)
b2: (C,)
```

### Part A: parameter records

Define:

```python
@dataclass
class MLPParameters:
    weights1: np.ndarray
    bias1: np.ndarray
    weights2: np.ndarray
    bias2: np.ndarray
```

Add a function that returns the total number of scalar parameters.  For MNIST with
64 hidden units, calculate this number by hand first.

### Part B: initialize

Do not initialize every hidden weight to zero.  If every hidden unit starts
identically, it receives the same forward values and gradient and remains
identical.

Use a fixed random generator and a scale such as:

```text
sqrt(2 / fan_in)
```

for ReLU layers.  This is commonly called He initialization.

### Part C: ReLU

Implement:

```python
def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)
```

Test negative, zero, and positive values.

### Part D: forward cache

Implement:

```text
Z1 = X @ W1 + b1
H1 = ReLU(Z1)
Z2 = H1 @ W2 + b2
P  = softmax(Z2)
```

Return:

```python
@dataclass(frozen=True)
class ForwardCache:
    x: np.ndarray
    pre_activation1: np.ndarray
    activation1: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
```

The cache is not merely for debugging.  The backward pass needs the inputs and
intermediate values to compute parameter gradients.

### Part E: shape trace

For:

```text
B = 128
D = 784
H = 64
C = 10
```

fill in every line before executing:

```text
X @ W1        (?, ?)
X @ W1 + b1   (?, ?)
ReLU(...)     (?, ?)
H1 @ W2       (?, ?)
logits + b2   (?, ?)
softmax       (?, ?)
```

Write the trace out by hand before executing.

### Minimum tests

- all parameter shapes match the chosen dimensions;
- two different hidden units do not start with identical weight columns;
- ReLU outputs no negative values;
- probability rows sum to one;
- forward output shapes match the shape trace;
- a dimension mismatch produces a useful error;
- parameter count matches the formula.

### Questions to answer

1. What makes this model nonlinear?
2. What would happen if ReLU were replaced by the identity function?
3. Why does zero initialization fail for hidden units but work for the Stage 8
   output layer?
4. Which cached arrays are needed to compute `dW2`, `dW1`, and the ReLU derivative?
5. What memory cost does caching impose?
6. At inference time, which training-only values are unnecessary?

### Stretch

For a tiny two-dimensional input and two hidden units, print the hidden activation
of every example.  Think of the hidden layer as learning a new feature space.

