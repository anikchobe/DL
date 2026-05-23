## Problem 2 — Learn a line from a batch

### Goal

Move from one example and one parameter to several examples, a weight, and a bias.
Learn how a shared parameter receives gradient contributions from the whole batch.

Use:

```python
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
targets = 3.0 * x + 2.0
```

The model is:

```text
prediction[i] = weight * x[i] + bias
```

### Part A: vectorized forward pass

Write:

```python
def forward(x: np.ndarray, weight: float, bias: float) -> np.ndarray:
    ...
```

Do not loop over examples.

### Part B: mean loss

Write:

```python
def mean_half_squared_error(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    ...
```

The result must be one scalar, even though there is one error per example.

### Part C: two gradients

Write:

```python
def loss_and_gradients(
    x: np.ndarray,
    targets: np.ndarray,
    weight: float,
    bias: float,
) -> tuple[float, float, float]:
    ...
```

Let:

```text
errors = predictions - targets
```

Derive:

```text
dloss/dweight = mean(errors * x)
dloss/dbias   = mean(errors)
```

First calculate these with an explicit loop.  Then replace the loop with NumPy
expressions and verify that both implementations agree.

### Part D: train both parameters

Run gradient descent until the model recovers approximately:

```text
weight = 3
bias   = 2
```

Keep the weight and bias updates separate and visible.

### Minimum tests

- Shapes are preserved: five inputs produce five predictions.
- Analytic gradients agree with central differences for both weight and bias.
- Training decreases the mean loss.
- The final parameters recover the known line.
- Empty batches and mismatched shapes produce errors.

### Questions to answer

1. Why is `dloss/dbias` a sum or mean across all examples?
2. What changes if the loss is a sum instead of a mean?
3. If the batch size doubles by duplicating every example, should the mean-loss
   gradient change?
4. Which values are data and which values are learned parameters?
5. What does it mean that the same weight is “shared” by every example?

### Stretch

Add Gaussian noise to the targets.  The learned line will no longer pass through
every point.  Explain what minimizing mean squared error means in that case.

### Some math
predictions = forward(x, weight, bias)   # 5 predictions
errors = predictions - targets           # 5 errors, still a vector
loss = float(0.5 * np.mean(errors**2))   # collapse to 1 scalar

gradient_weight = float(np.mean(errors * x))   # how should the slope change?
gradient_bias   = float(np.mean(errors))       # how should the intercept change?

Two returns instead of Stage 1's one, because there are now two knobs. Each answers the same question independently: "if I nudge this knob up a little, what happens to the loss?"

The derivation

loss = (1/N) · Σᵢ 0.5·(w·xᵢ + b − tᵢ)²        let eᵢ = w·xᵢ + b − tᵢ

∂loss/∂w = (1/N) · Σᵢ eᵢ · xᵢ  = mean(errors * x)     ← the xᵢ comes from d(w·xᵢ)/dw
∂loss/∂b = (1/N) · Σᵢ eᵢ · 1   = mean(errors)         ← d(b)/db = 1

The 1/N in both gradients isn't an extra design choice — line 48's comment says it: the mean in the loss is where it comes from. Mean loss ⇒ mean gradient, automatically.
