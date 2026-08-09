## Make one number learn

### Goal

Implement the complete training loop with one trainable scalar so every quantity
can be printed and understood.

Use this model:

```text
x = 3
prediction = weight * x
target = 12
```

Starting from `weight = 0`, the desired weight is 4.

### Part A: forward propagation

Write:

```python
def forward(x: float, weight: float) -> float:
    ...
```

“Forward propagation” currently means only evaluating the model to produce a
prediction.

### Part B: loss

Write:

```python
def half_squared_error(prediction: float, target: float) -> float:
    ...
```

Use:

```text
loss = 0.5 * (prediction - target)^2
```

The factor `0.5` is optional mathematically, but it cancels the `2` when the square
is differentiated.

Evaluate the loss at weights `0`, `2`, `4`, and `6`.  Sketch or print a table of
weight versus loss.

### Part C: a numerical derivative

Write:

```python
def central_difference(
    function: Callable[[float], float],
    value: float,
    epsilon: float = 1e-5,
) -> float:
    ...
```

Use:

```text
f(value + epsilon) - f(value - epsilon)
---------------------------------------
              2 * epsilon
```

This approximates the slope by looking at the loss on both sides of the current
weight.

### Part D: the analytic derivative

Derive and implement:

```python
def backward(x: float, weight: float, target: float) -> float:
    ...
```

Write the two local relationships:

```text
loss       = 0.5 * (prediction - target)^2
prediction = weight * x
```

Then apply the chain rule.  Compare `backward(...)` with `central_difference(...)`
at several weights, including negative weights.

### Part E: update the weight

Write:

```python
def train_weight(
    x: float,
    target: float,
    *,
    initial_weight: float = 0.0,
    learning_rate: float = 0.05,
    steps: int = 20,
) -> tuple[float, list[ScalarStep]]:
    ...
```

Each iteration must perform these operations in order:

```text
prediction = forward(...)
loss       = compute_loss(...)
gradient   = backward(...)
weight     = weight - learning_rate * gradient
```

Record a row such as:

```text
step | weight | prediction | loss | gradient
```

### Minimum tests

- At `weight = 4`, the prediction is 12 and the loss is zero.
- Numerical and analytic derivatives agree to a small tolerance.
- Training decreases the loss.
- The final weight is close to 4.
- Setting a negative learning rate raises an error.

### Experiments

Try learning rates:

```text
0.001
0.01
0.05
0.2
1.0
```

Record whether training is slow, stable, oscillating, or divergent.

### Questions to answer

1. What is the difference between a prediction, an error, a loss, and a gradient?
2. If the gradient is negative, why does subtracting it increase the weight?
3. Is the gradient the amount by which the weight changes?
4. Why is numerical differentiation too expensive for a model with millions of
   parameters?
5. Which four operations in this stage remain present in the final MNIST trainer?

### Stretch

Plot loss versus weight.  Mark each weight visited by gradient descent.
