## Stage 10 — Backpropagate through the MLP and solve XOR

### Goal

Apply the chain rule in reverse operation order, verify all four parameter-gradient
arrays, and demonstrate why a hidden nonlinear layer is useful.

### Part A: start at the loss

From softmax cross-entropy:

```text
dZ2 = (probabilities - one_hot(labels)) / B
```

Check:

```text
dZ2: (B, C)
```

### Part B: second affine layer

From:

```text
Z2 = H1 @ W2 + b2
```

implement:

```text
dW2 = H1.T @ dZ2
db2 = sum(dZ2, axis=0)
dH1 = dZ2 @ W2.T
```

Write all input and output shapes next to the code.

### Part C: ReLU

From:

```text
H1 = ReLU(Z1)
```

implement:

```text
dZ1 = dH1 * (Z1 > 0)
```

The comparison produces a Boolean mask.  NumPy multiplies it as zeros and ones.
At exactly zero, ReLU has no unique derivative; choose zero and avoid exact kinks
when gradient checking.

### Part D: first affine layer

From:

```text
Z1 = X @ W1 + b1
```

implement:

```text
dW1 = X.T @ dZ1
db1 = sum(dZ1, axis=0)
```

You could also compute `dX = dZ1 @ W1.T`, but it is not needed to update this
network because the input pixels are not parameters.

### Part E: return named gradients

Use:

```python
@dataclass
class MLPGradients:
    weights1: np.ndarray
    bias1: np.ndarray
    weights2: np.ndarray
    bias2: np.ndarray
```

Assert that every gradient has the same shape as its parameter.

### Part F: gradient check all arrays

Use a tiny network, for example:

```text
B = 5
D = 3
H = 4
C = 3
```

Check sampled entries from `W1`, `b1`, `W2`, and `b2`.  Use `float64`.  Nudge hidden
biases so pre-activations are not exactly zero.

If one array fails, isolate it.  Common mistakes include:

- forgetting division by batch size;
- using `W2` instead of `W2.T`;
- applying the ReLU mask to `H1` rather than the saved `Z1`;
- summing a bias gradient along the wrong axis;
- modifying probabilities in place and then reusing them.

### Part G: XOR

Create noisy examples around four corners:

```text
(-1, -1) -> class 0
(-1,  1) -> class 1
( 1, -1) -> class 1
( 1,  1) -> class 0
```

A single linear decision boundary cannot solve XOR.  Train the MLP with a small
hidden layer and show that it can.

This progression is important:

```text
linear model fails for a structural reason
hidden nonlinear model succeeds
```

Do not use MNIST to debug the first MLP backward pass.  XOR is faster and easier to
plot.

### Minimum tests

- all four gradient arrays match finite differences;
- gradient checking does not mutate the original model;
- one update lowers loss for a sufficiently small learning rate;
- training solves noisy XOR with high accuracy;
- replacing ReLU with identity prevents reliable XOR separation;
- fixed seeds reproduce the initial model and data.

### Questions to answer

1. Why does backpropagation traverse operations in reverse order?
2. Is backpropagation “sending the input backward” through the network?
3. What is the difference between `dH1` and `dZ1`?
4. Why is the ReLU mask based on the forward pre-activation?
5. What does one entry of `dW1` mean?
6. Why is one scalar loss able to produce a derivative for every parameter?
7. What does XOR demonstrate that MNIST alone would not make as obvious?

### Stretch

Plot the learned XOR decision regions.  Then inspect hidden activations to see how
the two hidden features transform the four clusters.

